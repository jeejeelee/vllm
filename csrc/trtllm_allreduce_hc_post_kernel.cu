/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Fused AllReduce + hc_post kernel for DeepseekV4-style models.
 * The per-token computation matches the tilelang reference in
 * `vllm/model_executor/layers/mhc.py::mhc_post_tilelang`:
 *
 *   out[n, hco, h] = post[n, hco] * x[n, h]
 *                  + sum_{hci} comb[n, hci, hco] * residual[n, hci, h]
 *
 * Communication uses the FlashInfer one-shot Lamport scheme; the workspace
 * layout is identical to flashinfer's `LamportComm` (see
 * `flashinfer/include/flashinfer/comm/trtllm_allreduce_fusion.cuh`).
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <cstdint>
#include <cstring>

#include "core/registration.h"
#include "cuda_utils.h"
#include "trtllm_allreduce_hc_post_kernel.h"

namespace vllm {
namespace trtllm_ar_hc_post {

// 16-byte vectorized access; bf16 ⇒ 8 elements per vector.
static constexpr int kBytesPerAccess = 16;
static constexpr int kVecSize = kBytesPerAccess / sizeof(__nv_bfloat16);

// Number of barrier flag slots per rank in the workspace (shared with
// FlashInfer).
static constexpr int kBarrierFlagCount = 256;

// -----------------------------------------------------------------------------
// neg_zero (Lamport sentinel): bit pattern 0x8000 for bf16.
// -----------------------------------------------------------------------------
__device__ __forceinline__ bool is_neg_zero(__nv_bfloat16 v) {
  return __bfloat16_as_ushort(v) == 0x8000;
}

__device__ __forceinline__ __nv_bfloat16 neg_zero_bf16() {
  return __ushort_as_bfloat16(static_cast<unsigned short>(0x8000));
}

// -----------------------------------------------------------------------------
// 16-byte vectorized container of 8 bf16. Aligned for ld.global.v4.b32 / st.
// -----------------------------------------------------------------------------
struct alignas(16) BF16Vec8 {
  __nv_bfloat16 data[kVecSize];

  __device__ __forceinline__ static BF16Vec8 fill(__nv_bfloat16 v) {
    BF16Vec8 r;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) r.data[i] = v;
    return r;
  }

  __device__ __forceinline__ bool has_neg_zero() const {
    bool any = false;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) any |= is_neg_zero(data[i]);
    return any;
  }

  __device__ __forceinline__ void remove_neg_zero() {
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      if (is_neg_zero(data[i])) {
        data[i] = __ushort_as_bfloat16(0);
      }
    }
  }

  __device__ __forceinline__ static BF16Vec8 load(__nv_bfloat16 const* addr) {
    BF16Vec8 r;
    *reinterpret_cast<float4*>(&r.data[0]) =
        *reinterpret_cast<float4 const*>(addr);
    return r;
  }

  __device__ __forceinline__ void store(__nv_bfloat16* addr) const {
    *reinterpret_cast<float4*>(addr) =
        *reinterpret_cast<float4 const*>(&data[0]);
  }

  __device__ __forceinline__ static BF16Vec8 load_volatile(
      __nv_bfloat16 const* addr) {
    BF16Vec8 r;
    uint4 v;
    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(addr));
    *reinterpret_cast<uint4*>(&r.data[0]) = v;
    return r;
  }

  __device__ __forceinline__ void store_volatile(__nv_bfloat16* addr) const {
    uint4 v = *reinterpret_cast<uint4 const*>(&data[0]);
    asm volatile(
        "st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(addr),
        "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
  }
};

// -----------------------------------------------------------------------------
// LamportComm — mirrors the FlashInfer int32-metadata layout. The kernel
// rotates among 3 sub-buffers using `flag_value % 3` so that the previous
// round's data isn't clobbered before all peers read it.
// -----------------------------------------------------------------------------
template <int NRanks>
struct LamportComm {
  __device__ __forceinline__ LamportComm(void** workspace, int rank) {
    int* meta = reinterpret_cast<int*>(workspace[NRanks * 3]);
    counter_ptr = &meta[0];
    flag_ptr = &meta[2];
    clear_ptr = &meta[4];
    flag_value = *flag_ptr;
    int comm_size = meta[3];
    clear_size = *clear_ptr;
    int data_offset = flag_value % 3;
    int clear_offset = (flag_value + 2) % 3;
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      data_bufs[r] = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + r]) +
                     static_cast<int64_t>(data_offset) * comm_size;
    }
    clear_buf = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + rank]) +
                static_cast<int64_t>(clear_offset) * comm_size;
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(counter_ptr, 1);
    }
  }

  __device__ __forceinline__ void update(int new_clear_size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      while (*reinterpret_cast<int volatile*>(counter_ptr) != gridDim.x) {
      }
      *flag_ptr = (flag_value + 1) % 3;
      *clear_ptr = new_clear_size;
      *counter_ptr = 0;
    }
  }

  int* counter_ptr;
  int* flag_ptr;
  int* clear_ptr;
  uint8_t* data_bufs[NRanks];
  uint8_t* clear_buf;
  int clear_size;
  int flag_value;
};

// -----------------------------------------------------------------------------
// SyncComm — used by the two-shot kernel. Different metadata slots from the
// Lamport one-shot path, and uses workspace[0..NRanks-1] as comm buffers and
// workspace[NRanks..2*NRanks-1] as per-rank barrier-flag arrays.
// -----------------------------------------------------------------------------
template <int NRanks>
struct SyncComm {
  __device__ __forceinline__ SyncComm(void** workspace) {
    int* meta = reinterpret_cast<int*>(workspace[NRanks * 3]);
    counter_ptr = &meta[0];
    flag_ptr = &meta[1];
    flag_value = *flag_ptr;
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      comm_bufs[r] = workspace[r];
      barrier_flags[r] = workspace[NRanks + r];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(counter_ptr, 1);
    }
  }

  __device__ __forceinline__ void update(int new_flag_value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      while (*reinterpret_cast<int volatile*>(counter_ptr) != gridDim.x) {
      }
      *flag_ptr = new_flag_value;
      *counter_ptr = 0;
    }
  }

  int* counter_ptr;
  int* flag_ptr;
  void* comm_bufs[NRanks];
  void* barrier_flags[NRanks];
  int flag_value;
};

// -----------------------------------------------------------------------------
// Per-block tri-state Barrier across all ranks (rotates flag value 0→1→2→0…).
// Ported from FlashInfer's `Barrier`. The first NRanks threads of each block
// are responsible for posting and observing the flag.
// -----------------------------------------------------------------------------
template <int NRanks>
class Barrier {
 public:
  __device__ __forceinline__ Barrier(int rank, SyncComm<NRanks> const& comm) {
    if (threadIdx.x < NRanks) {
      m_flag_value = comm.flag_value;
      int target_rank = threadIdx.x;
      m_target_flag =
          reinterpret_cast<int*>(comm.barrier_flags[target_rank]) + rank;
      m_current_flag = reinterpret_cast<int*>(comm.barrier_flags[rank]) +
                       blockIdx.x * NRanks + target_rank;
    }
  }

  __device__ __forceinline__ void sync() {
    __syncthreads();
    if (threadIdx.x < NRanks) {
      m_flag_value = next_flag(m_flag_value);
      // Post our flag value into every cluster slot of every peer to avoid the
      // ABA problem when gridDim < kBarrierFlagCount (peers reading slots that
      // haven't been launched yet).
      for (int flag_idx = blockIdx.x; flag_idx < kBarrierFlagCount;
           flag_idx += gridDim.x) {
        st_flag(m_target_flag + flag_idx * NRanks, m_flag_value);
      }
      while (ld_flag(m_current_flag) == prev_flag(m_flag_value)) {
      }
    }
    __syncthreads();
  }

  int m_flag_value;

 private:
  __device__ __forceinline__ void st_flag(int* addr, int flag) {
    asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(addr));
  }
  __device__ __forceinline__ int ld_flag(int* addr) {
    int flag;
    asm volatile("ld.global.acquire.sys.b32 %0, [%1];"
                 : "=r"(flag)
                 : "l"(addr));
    return flag;
  }
  __device__ __forceinline__ int next_flag(int flag) {
    return flag == 2 ? 0 : flag + 1;
  }
  __device__ __forceinline__ int prev_flag(int flag) {
    return flag == 0 ? 2 : flag - 1;
  }

  int* m_target_flag;
  int* m_current_flag;
};

// -----------------------------------------------------------------------------
// Cross-rank vector reduction with fp32 accumulation (bf16 inputs).
// -----------------------------------------------------------------------------
template <int NRanks>
__device__ __forceinline__ BF16Vec8 reduce_sum(BF16Vec8 const* peers) {
  BF16Vec8 acc;
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    float v = static_cast<float>(peers[0].data[i]);
#pragma unroll
    for (int r = 1; r < NRanks; ++r) {
      v += static_cast<float>(peers[r].data[i]);
    }
    acc.data[i] = static_cast<__nv_bfloat16>(v);
  }
  return acc;
}

// =============================================================================
// Main kernel: one-shot Lamport AR + per-token hc_post.
//
// Grid:   one CTA per token, num_tokens CTAs total (with strided iteration if
//         num_tokens > grid_size).
// Block:  threads_per_token = hidden_dim / kVecSize (capped at 1024). Each
//         thread owns one vec_t-sized slice of one token.
// =============================================================================
template <int NRanks, int HC_MULT, bool LaunchWithPdl>
__global__ void __launch_bounds__(1024)
    ar_hc_post_oneshot_lamport_kernel(Params params) {
  static_assert(HC_MULT > 0 && HC_MULT <= 16, "HC_MULT out of range");
  int const threads_per_token = blockDim.x;
  int const access_id_in_token = threadIdx.x;
  int const token_stride = gridDim.x;

  int const hidden_vecs = params.hidden_dim / kVecSize;  // checked on host
  int const tot_access = params.num_tokens * hidden_vecs;
  int const access_stride = token_stride * hidden_vecs;

  __shared__ float post_smem[HC_MULT];
  __shared__ float comb_smem[HC_MULT * HC_MULT];

  // Cast pointers once.
  __nv_bfloat16 const* allreduce_in_ptr =
      reinterpret_cast<__nv_bfloat16 const*>(params.allreduce_in);
  __nv_bfloat16 const* residual_ptr =
      reinterpret_cast<__nv_bfloat16 const*>(params.residual);
  float const* post_ptr = reinterpret_cast<float const*>(params.post);
  float const* comb_ptr = reinterpret_cast<float const*>(params.comb);
  __nv_bfloat16* out_ptr = reinterpret_cast<__nv_bfloat16*>(params.out);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (LaunchWithPdl) {
    asm volatile("griddepcontrol.wait;");
  }
#endif

  LamportComm<NRanks> comm(params.workspace, params.rank);
  int const clear_access = comm.clear_size / kVecSize;

  // -------- Phase 1: push local input to all peers (with neg-zero scrub). ----
  int access_id = blockIdx.x * threads_per_token + access_id_in_token;
  for (int idx = access_id; idx < tot_access; idx += access_stride) {
    BF16Vec8 v = BF16Vec8::load(allreduce_in_ptr + idx * kVecSize);
    v.remove_neg_zero();
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(comm.data_bufs[r]) +
                           (params.rank * tot_access + idx) * kVecSize;
      v.store(dst);
    }
  }

  // -------- Phase 2: clear last-round residue from our local sub-buffer. ----
  for (int idx = access_id; idx < clear_access; idx += access_stride) {
    BF16Vec8 const neg = BF16Vec8::fill(neg_zero_bf16());
    neg.store(reinterpret_cast<__nv_bfloat16*>(comm.clear_buf) +
              idx * kVecSize);
  }

  // -------- Phase 3 + 4 + 5: poll peers, summed AR, then hc_post. ----------
  bool first_iter = true;
  for (int token_id = blockIdx.x; token_id < params.num_tokens;
       token_id += token_stride) {
    // Ensure the previous iteration's compute has finished reading smem
    // before the current iteration overwrites it.
    if (!first_iter) {
      __syncthreads();
    }
    first_iter = false;

    // Pre-load post[token, :] and comb[token, :, :] into shared memory.
    // Strided loops keep this safe for any blockDim, including blockDim <
    // HC_MULT*HC_MULT.
    for (int i = threadIdx.x; i < HC_MULT; i += blockDim.x) {
      post_smem[i] = post_ptr[token_id * HC_MULT + i];
    }
    for (int i = threadIdx.x; i < HC_MULT * HC_MULT; i += blockDim.x) {
      comb_smem[i] = comb_ptr[token_id * HC_MULT * HC_MULT + i];
    }
    __syncthreads();  // smem populated before any thread reads it below.

    int const idx = token_id * hidden_vecs + access_id_in_token;

    // Lamport poll: spin until all peers have written non-neg-zero data.
    // Each thread polls its own (token, slice) independently.
    BF16Vec8 peer_vals[NRanks];
    bool done = false;
    while (!done) {
      done = true;
#pragma unroll
      for (int r = 0; r < NRanks; ++r) {
        __nv_bfloat16 const* src = reinterpret_cast<__nv_bfloat16 const*>(
                                       comm.data_bufs[params.rank]) +
                                   (r * tot_access + idx) * kVecSize;
        peer_vals[r] = BF16Vec8::load_volatile(src);
        done &= !peer_vals[r].has_neg_zero();
      }
    }
    BF16Vec8 ar_sum = reduce_sum<NRanks>(peer_vals);

    if (access_id_in_token >= hidden_vecs) {
      continue;  // padding thread (shouldn't happen with our launch shape)
    }

    // Load residual[token, 0..HC_MULT-1, h_slice] once into registers.
    BF16Vec8 residual_reg[HC_MULT];
#pragma unroll
    for (int hci = 0; hci < HC_MULT; ++hci) {
      __nv_bfloat16 const* base =
          residual_ptr +
          ((token_id * HC_MULT + hci) * hidden_vecs + access_id_in_token) *
              kVecSize;
      residual_reg[hci] = BF16Vec8::load(base);
    }

    // For each output sub-row hco, compute the linear combination + post*x.
#pragma unroll
    for (int hco = 0; hco < HC_MULT; ++hco) {
      float const post_val = post_smem[hco];
      float acc[kVecSize];
#pragma unroll
      for (int e = 0; e < kVecSize; ++e) {
        acc[e] = post_val * static_cast<float>(ar_sum.data[e]);
      }
#pragma unroll
      for (int hci = 0; hci < HC_MULT; ++hci) {
        // mhc_post indexes comb as a[i_n, i_hci, i_hco]; flat layout
        // [hci * HC_MULT + hco].
        float const c = comb_smem[hci * HC_MULT + hco];
#pragma unroll
        for (int e = 0; e < kVecSize; ++e) {
          acc[e] += c * static_cast<float>(residual_reg[hci].data[e]);
        }
      }
      BF16Vec8 out_vec;
#pragma unroll
      for (int e = 0; e < kVecSize; ++e) {
        out_vec.data[e] = static_cast<__nv_bfloat16>(acc[e]);
      }
      __nv_bfloat16* dst = out_ptr + ((token_id * HC_MULT + hco) * hidden_vecs +
                                      access_id_in_token) *
                                         kVecSize;
      out_vec.store(dst);
    }
  }

  comm.update(params.num_tokens * params.hidden_dim * NRanks);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (LaunchWithPdl) {
    asm volatile("griddepcontrol.launch_dependents;");
  }
#endif
}

// =============================================================================
// Two-shot kernel: reduce-scatter + all-gather with Barrier sync.
//
// Phase A : every rank copies its full local input into its own peer-visible
//           comm buffer at offset `idx`.
// Barrier.
// Phase B : every rank reduces ONE slice (begin_tokens[rank],
// +token_num_per_ranks[rank])
//           by reading all peers' comm buffers and writes the AR-summed result
//           into ALL peers' comm buffers at offset `tot_access + idx`.
// Barrier.
// Phase C : every rank reads back the AR-summed data for ALL tokens and
//           applies hc_post per token.
//
// Grid:   one block per "cluster" position; cluster_num = ceil(N / NRanks).
//         Each block processes up to NRanks tokens (one per rank's slice).
// Block:  threads_per_token = hidden_dim / kVecSize.
// =============================================================================
template <int NRanks, int HC_MULT, bool LaunchWithPdl>
__global__ void __launch_bounds__(1024)
    ar_hc_post_twoshot_sync_kernel(Params params) {
  // Token partition is in params.begin_tokens / params.token_num_per
  // (only the first NRanks slots are valid).
  int const* begin_tokens = params.begin_tokens;
  int const* token_num_per = params.token_num_per;

  int const access_id_in_token = threadIdx.x;
  int const hidden_vecs = params.hidden_dim / kVecSize;
  int const tot_access = params.num_tokens * hidden_vecs;
  int const cluster_num = gridDim.x;
  int const access_stride = cluster_num * hidden_vecs;
  int const token_id = blockIdx.x;  // cluster id
  int const access_id = token_id * hidden_vecs + access_id_in_token;

  __shared__ float post_smem[HC_MULT];
  __shared__ float comb_smem[HC_MULT * HC_MULT];

  __nv_bfloat16 const* allreduce_in_ptr =
      reinterpret_cast<__nv_bfloat16 const*>(params.allreduce_in);
  __nv_bfloat16 const* residual_ptr =
      reinterpret_cast<__nv_bfloat16 const*>(params.residual);
  float const* post_ptr = reinterpret_cast<float const*>(params.post);
  float const* comb_ptr = reinterpret_cast<float const*>(params.comb);
  __nv_bfloat16* out_ptr = reinterpret_cast<__nv_bfloat16*>(params.out);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (LaunchWithPdl) {
    asm volatile("griddepcontrol.wait;");
  }
#endif
  SyncComm<NRanks> comm(params.workspace);

  // -------- Phase A: copy our local input into our own comm_buf -------------
  __nv_bfloat16* my_buf =
      reinterpret_cast<__nv_bfloat16*>(comm.comm_bufs[params.rank]);
#pragma unroll
  for (int r = 0; r < NRanks; ++r) {
    int const start = begin_tokens[r] * hidden_vecs;
    int const end = (begin_tokens[r] + token_num_per[r]) * hidden_vecs;
    for (int idx = start + access_id; idx < end; idx += access_stride) {
      BF16Vec8 v = BF16Vec8::load(allreduce_in_ptr + idx * kVecSize);
      v.store(my_buf + idx * kVecSize);
    }
  }

  Barrier<NRanks> barrier(params.rank, comm);
  barrier.sync();

  // -------- Phase B: reduce our own slice; write result to all peers --------
  {
    int const start = begin_tokens[params.rank] * hidden_vecs;
    int const end =
        (begin_tokens[params.rank] + token_num_per[params.rank]) * hidden_vecs;
    for (int idx = start + access_id; idx < end; idx += access_stride) {
      BF16Vec8 vals[NRanks];
#pragma unroll
      for (int r = 0; r < NRanks; ++r) {
        vals[r] = BF16Vec8::load(
            reinterpret_cast<__nv_bfloat16 const*>(comm.comm_bufs[r]) +
            idx * kVecSize);
      }
      BF16Vec8 sum_val = reduce_sum<NRanks>(vals);
#pragma unroll
      for (int r = 0; r < NRanks; ++r) {
        sum_val.store(reinterpret_cast<__nv_bfloat16*>(comm.comm_bufs[r]) +
                      (tot_access + idx) * kVecSize);
      }
    }
  }
  barrier.sync();

  // -------- Phase C: hc_post over all tokens, reading our own comm_buf -----
  bool first_iter = true;
#pragma unroll
  for (int r = 0; r < NRanks; ++r) {
    int const start_idx = begin_tokens[r] * hidden_vecs;
    int const end_idx = (begin_tokens[r] + token_num_per[r]) * hidden_vecs;
    for (int idx = start_idx + access_id, tidx = begin_tokens[r] + token_id;
         idx < end_idx; idx += access_stride, tidx += cluster_num) {
      // Inter-iteration smem barrier (not needed for the first slot).
      if (!first_iter) {
        __syncthreads();
      }
      first_iter = false;

      // Load post[tidx] / comb[tidx] into smem (covers any blockDim).
      for (int i = threadIdx.x; i < HC_MULT; i += blockDim.x) {
        post_smem[i] = post_ptr[tidx * HC_MULT + i];
      }
      for (int i = threadIdx.x; i < HC_MULT * HC_MULT; i += blockDim.x) {
        comb_smem[i] = comb_ptr[tidx * HC_MULT * HC_MULT + i];
      }
      __syncthreads();

      BF16Vec8 ar_sum = BF16Vec8::load(
          reinterpret_cast<__nv_bfloat16 const*>(comm.comm_bufs[params.rank]) +
          (tot_access + idx) * kVecSize);

      if (access_id_in_token >= hidden_vecs) continue;

      // Same hc_post body as the one-shot kernel.
      BF16Vec8 residual_reg[HC_MULT];
#pragma unroll
      for (int hci = 0; hci < HC_MULT; ++hci) {
        __nv_bfloat16 const* base =
            residual_ptr +
            ((tidx * HC_MULT + hci) * hidden_vecs + access_id_in_token) *
                kVecSize;
        residual_reg[hci] = BF16Vec8::load(base);
      }
#pragma unroll
      for (int hco = 0; hco < HC_MULT; ++hco) {
        float const post_val = post_smem[hco];
        float acc[kVecSize];
#pragma unroll
        for (int e = 0; e < kVecSize; ++e) {
          acc[e] = post_val * static_cast<float>(ar_sum.data[e]);
        }
#pragma unroll
        for (int hci = 0; hci < HC_MULT; ++hci) {
          float const c = comb_smem[hci * HC_MULT + hco];
#pragma unroll
          for (int e = 0; e < kVecSize; ++e) {
            acc[e] += c * static_cast<float>(residual_reg[hci].data[e]);
          }
        }
        BF16Vec8 out_vec;
#pragma unroll
        for (int e = 0; e < kVecSize; ++e) {
          out_vec.data[e] = static_cast<__nv_bfloat16>(acc[e]);
        }
        __nv_bfloat16* dst = out_ptr + ((tidx * HC_MULT + hco) * hidden_vecs +
                                        access_id_in_token) *
                                           kVecSize;
        out_vec.store(dst);
      }
    }
  }

  comm.update(barrier.m_flag_value);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (LaunchWithPdl) {
    asm volatile("griddepcontrol.launch_dependents;");
  }
#endif
}

// -----------------------------------------------------------------------------
// Launcher.
// -----------------------------------------------------------------------------
static int get_sm_count_cached() {
  static int sm = 0;
  if (sm == 0) {
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    sm = prop.multiProcessorCount;
  }
  return sm;
}

static int get_sm_version_cached() {
  static int sm = 0;
  if (sm == 0) {
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    int major = 0, minor = 0;
    CUDA_CHECK(
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    CUDA_CHECK(
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    sm = major * 10 + minor;
  }
  return sm;
}

template <int NRanks, int HC_MULT>
static void launch_oneshot(Params const& params, bool launch_with_pdl) {
  int const threads_per_token = params.hidden_dim / kVecSize;
  int const sm_count = get_sm_count_cached();
  int const grid_size = std::min(sm_count, std::max(1, params.num_tokens));

  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid_size;
  cfg.blockDim = threads_per_token;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = params.stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = launch_with_pdl ? 1 : 0;
  cfg.attrs = (get_sm_version_cached() >= 90) ? attr : nullptr;
  cfg.numAttrs = (get_sm_version_cached() >= 90) ? 1 : 0;

  if (launch_with_pdl) {
    CUDA_CHECK(cudaLaunchKernelEx(
        &cfg, ar_hc_post_oneshot_lamport_kernel<NRanks, HC_MULT, true>,
        params));
  } else {
    CUDA_CHECK(cudaLaunchKernelEx(
        &cfg, ar_hc_post_oneshot_lamport_kernel<NRanks, HC_MULT, false>,
        params));
  }
}

template <int NRanks, int HC_MULT>
static void launch_twoshot(Params params, bool launch_with_pdl) {
  // Compute the per-rank token partition (mirrors FlashInfer's launcher).
  int const remaining = params.num_tokens % NRanks;
  int const per_rank = params.num_tokens / NRanks;
  int cluster_num = per_rank + (remaining ? 1 : 0);
  cluster_num = std::max(cluster_num, 1);

  for (int r = 0; r < NRanks; ++r) {
    params.begin_tokens[r] = r * per_rank + (remaining > r ? r : remaining);
    params.token_num_per[r] = per_rank + (remaining > r ? 1 : 0);
  }

  int const threads_per_token = params.hidden_dim / kVecSize;
  TORCH_CHECK(threads_per_token >= NRanks,
              "two-shot requires hidden_dim/kVecSize >= NRanks");

  int const sm_count = get_sm_count_cached();
  int const grid_size = std::min(sm_count, cluster_num);

  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid_size;
  cfg.blockDim = threads_per_token;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = params.stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = launch_with_pdl ? 1 : 0;
  cfg.attrs = (get_sm_version_cached() >= 90) ? attr : nullptr;
  cfg.numAttrs = (get_sm_version_cached() >= 90) ? 1 : 0;

  if (launch_with_pdl) {
    CUDA_CHECK(cudaLaunchKernelEx(
        &cfg, ar_hc_post_twoshot_sync_kernel<NRanks, HC_MULT, true>, params));
  } else {
    CUDA_CHECK(cudaLaunchKernelEx(
        &cfg, ar_hc_post_twoshot_sync_kernel<NRanks, HC_MULT, false>, params));
  }
}

template <int NRanks, int HC_MULT>
static void launch(Params const& params, bool launch_with_pdl,
                   bool use_oneshot) {
  TORCH_CHECK(params.hidden_dim % kVecSize == 0,
              "hidden_dim must be divisible by ", kVecSize);
  int const threads_per_token = params.hidden_dim / kVecSize;
  TORCH_CHECK(threads_per_token <= 1024,
              "hidden_dim/kVecSize must be <= 1024 (got ", threads_per_token,
              "); cluster_size > 1 not yet supported.");
  if (use_oneshot) {
    launch_oneshot<NRanks, HC_MULT>(params, launch_with_pdl);
  } else {
    launch_twoshot<NRanks, HC_MULT>(params, launch_with_pdl);
  }
}

void run(Params const& params, bool launch_with_pdl, bool use_oneshot) {
  TORCH_CHECK(params.hc_mult == 4,
              "Only hc_mult == 4 is currently supported, got ", params.hc_mult);
  switch (params.nranks) {
    case 2:
      launch<2, 4>(params, launch_with_pdl, use_oneshot);
      break;
    case 4:
      launch<4, 4>(params, launch_with_pdl, use_oneshot);
      break;
    case 8:
      launch<8, 4>(params, launch_with_pdl, use_oneshot);
      break;
    default:
      TORCH_CHECK(false,
                  "trtllm_ar_hc_post: unsupported nranks=", params.nranks,
                  " (supported: 2, 4, 8)");
  }
}

}  // namespace trtllm_ar_hc_post
}  // namespace vllm

// =============================================================================
// Torch op wrapper: writes into `out` in place.
// =============================================================================
void trtllm_ar_hc_post(torch::Tensor const& allreduce_in,
                       torch::Tensor const& residual, torch::Tensor const& post,
                       torch::Tensor const& comb, torch::Tensor& out,
                       torch::Tensor& workspace, int64_t rank, int64_t nranks,
                       bool launch_with_pdl, bool use_oneshot) {
  TORCH_CHECK(allreduce_in.is_cuda() && residual.is_cuda() && post.is_cuda() &&
                  comb.is_cuda() && out.is_cuda() && workspace.is_cuda(),
              "All tensors must be CUDA");
  TORCH_CHECK(allreduce_in.scalar_type() == at::ScalarType::BFloat16,
              "allreduce_in must be bf16");
  TORCH_CHECK(residual.scalar_type() == at::ScalarType::BFloat16,
              "residual must be bf16");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16,
              "out must be bf16");
  TORCH_CHECK(post.scalar_type() == at::ScalarType::Float, "post must be fp32");
  TORCH_CHECK(comb.scalar_type() == at::ScalarType::Float, "comb must be fp32");

  TORCH_CHECK(allreduce_in.dim() == 2,
              "allreduce_in must be [N, H], got dim=", allreduce_in.dim());
  TORCH_CHECK(residual.dim() == 3,
              "residual must be [N, hc, H], got dim=", residual.dim());
  TORCH_CHECK(post.dim() == 3 || post.dim() == 2,
              "post must be [N, hc] or [N, hc, 1], got dim=", post.dim());
  TORCH_CHECK(comb.dim() == 3,
              "comb must be [N, hc, hc], got dim=", comb.dim());
  TORCH_CHECK(out.dim() == 3, "out must be [N, hc, H], got dim=", out.dim());

  int const N = static_cast<int>(allreduce_in.size(0));
  int const H = static_cast<int>(allreduce_in.size(1));
  int const HC = static_cast<int>(residual.size(1));

  TORCH_CHECK(residual.size(0) == N && residual.size(2) == H,
              "residual shape mismatch");
  TORCH_CHECK(post.size(0) == N && post.size(1) == HC, "post shape mismatch");
  TORCH_CHECK(comb.size(0) == N && comb.size(1) == HC && comb.size(2) == HC,
              "comb shape mismatch");
  TORCH_CHECK(out.size(0) == N && out.size(1) == HC && out.size(2) == H,
              "out shape mismatch");

  TORCH_CHECK(allreduce_in.is_contiguous(), "allreduce_in must be contiguous");
  TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  TORCH_CHECK(post.is_contiguous(), "post must be contiguous");
  TORCH_CHECK(comb.is_contiguous(), "comb must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

  c10::cuda::CUDAGuard guard(allreduce_in.device());

  vllm::trtllm_ar_hc_post::Params params;
  params.nranks = static_cast<int>(nranks);
  params.rank = static_cast<int>(rank);
  params.num_tokens = N;
  params.hidden_dim = H;
  params.hc_mult = HC;
  params.workspace = reinterpret_cast<void**>(workspace.mutable_data_ptr());
  params.allreduce_in = allreduce_in.data_ptr();
  params.residual = residual.data_ptr();
  params.post = post.data_ptr();
  params.comb = comb.data_ptr();
  params.out = out.mutable_data_ptr();
  params.stream = at::cuda::getCurrentCUDAStream(allreduce_in.get_device());

  vllm::trtllm_ar_hc_post::run(params, launch_with_pdl, use_oneshot);
}
