
/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <torch/cuda.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "cuda_utils.h"
#include "core/registration.h"
#include "minimax_reduce_rms_kernel.h"

#include <algorithm>
#include <optional>
#define FINAL_MASK 0xffffffff
#define MINIMAX_REDUCE_RMS_WARP_SIZE 32

namespace vllm {
namespace tensorrt_llm {

template <int NRanks>
struct LamportComm {
  __device__ __forceinline__ LamportComm(void** workspace, int rank) {
    counter_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[0];
    flag_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[2];
    clear_ptr = &reinterpret_cast<int64_t*>(workspace[NRanks * 3 + 1])[0];
    flag_value = *flag_ptr;
    auto comm_size = reinterpret_cast<int64_t*>(workspace[NRanks * 3 + 1])[1];
    clear_size = *clear_ptr;
    int data_offset = flag_value % 3;
    int clear_offset = (flag_value + 2) % 3;
    for (int r = 0; r < NRanks; ++r) {
      data_bufs[r] = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + r]) +
                     data_offset * comm_size;
    }
    clear_buf = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + rank]) +
                clear_offset * comm_size;
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(counter_ptr, 1);
    }
  }

  __device__ __forceinline__ void update(int64_t new_clear_size) {
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
  int64_t* clear_ptr;
  uint8_t* data_bufs[NRanks];
  uint8_t* clear_buf;
  int64_t clear_size;
  int flag_value;
};

template <>
struct LamportComm<1> {
  __device__ __forceinline__ LamportComm(void**, int) {}
  __device__ __forceinline__ void update(int64_t) {}
  int64_t clear_size = 0;
};

__device__ __forceinline__ bool is_neg_zero(float v) {
  return *reinterpret_cast<uint32_t*>(&v) == 0x80000000;
}

__device__ __forceinline__ bool is_neg_zero(float4 v) {
  return is_neg_zero(v.x) || is_neg_zero(v.y) || is_neg_zero(v.z) ||
         is_neg_zero(v.w);
}

__device__ __forceinline__ float4 get_neg_zero() {
  float4 vec;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    reinterpret_cast<uint32_t*>(&vec)[i] = 0x80000000;
  }
  return vec;
}

__device__ __forceinline__ float4 ld_global_volatile(float4* addr) {
  float4 val;
  asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
               : "l"(addr));
  return val;
}

__device__ __forceinline__ float ld_global_volatile(float* addr) {
  float val;
  asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(val) : "l"(addr));
  return val;
}

template <typename T, int NUM>
__inline__ __device__ T warpReduceSumV2(T* val) {
#pragma unroll
  for (int i = 0; i < NUM; i++) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
      val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
  }
  return (T)(0.0f);
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceSumV2(T* val) {
  static __shared__ T shared[NUM][33];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceSumV2<T, NUM>(val);

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < NUM; i++) {
      shared[i][wid] = val[i];
    }
  }

  __syncthreads();

  bool is_mask = threadIdx.x < (blockDim.x >> 5);
#pragma unroll
  for (int i = 0; i < NUM; i++) {
    val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
  }
  warpReduceSumV2<T, NUM>(val);
  return (T)0.0f;
}

template <typename DType>
class IndexHelper {
 public:
  __device__ __forceinline__ IndexHelper(MiniMaxReduceRMSParams const& params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    cg::grid_group grid = cg::this_grid();
    token_id = grid.cluster_rank();
    access_id_in_token = cluster.thread_rank();
    token_stride = grid.num_clusters();
#else
    token_id = blockIdx.x;
    access_id_in_token = threadIdx.x;
    token_stride = gridDim.x;
#endif
    access_id = token_id * params.hidden_dim / kElemsPerAccess<DType> +
                access_id_in_token;
    access_stride = token_stride * params.hidden_dim / kElemsPerAccess<DType>;
    tot_access = params.size_q / kElemsPerAccess<DType>;
  }

  int token_id;
  int access_id_in_token;
  int token_stride;
  int access_id;
  int access_stride;
  int tot_access;
};

/**
 * Simple per-token kernel (legacy, non-float4 path).
 */
template <typename DType, int NRanks>
__global__ void __launch_bounds__(1024)
    minimax_reduce_rms_kernel_lamport(MiniMaxReduceRMSParams params) {
  IndexHelper<DType> index_helper(params);
  int token_id = index_helper.token_id;
  int access_id_in_token = index_helper.access_id_in_token;
  int token_stride = index_helper.token_stride;
  int access_id = index_helper.access_id;
  int access_stride = index_helper.access_stride;
  int tot_access = index_helper.tot_access;
  int tot_tokens = params.size_q / params.hidden_dim;
  float4 clear_vec = get_neg_zero();

  LamportComm<NRanks> comm(params.workspace, params.rank);
  int clear_access = comm.clear_size / kElemsPerAccess<DType>;
  for (int idx = access_id; idx < tot_access;
       idx += access_stride, token_id += token_stride) {
    alignas(16) DType vals[kElemsPerAccess<DType>];
    float sum_variance = 0.F;
    *reinterpret_cast<float4*>(vals) =
        reinterpret_cast<float4*>(params.allreduce_in)[idx];
#pragma unroll
    for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
      sum_variance += static_cast<float>(vals[i]) * static_cast<float>(vals[i]);
    }
    blockReduceSumV2<float, 1>(&sum_variance);
    if (is_neg_zero(sum_variance)) {
      sum_variance = 0.F;
    }
    if (threadIdx.x == 0) {
      for (int r = 0; r < NRanks; ++r) {
        reinterpret_cast<float*>(
            comm.data_bufs[r])[(params.rank * tot_tokens) + token_id] =
            (sum_variance);
      }
    }

    bool done = false;
    float vars_all_ranks[NRanks];
    while (!done) {
      done = true;
#pragma unroll
      for (int r = 0; r < NRanks; ++r) {
        vars_all_ranks[r] = ld_global_volatile(&reinterpret_cast<float*>(
            comm.data_bufs[params.rank])[(r * tot_tokens) + token_id]);
        done &= !is_neg_zero(vars_all_ranks[r]);
      }
    }
    sum_variance = 0.F;
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      sum_variance += vars_all_ranks[r];
    }

    alignas(16) DType norm_weight[kElemsPerAccess<DType>];
    *reinterpret_cast<typename ElemsPerAccess<DType>::vec_type*>(norm_weight) =
        reinterpret_cast<typename ElemsPerAccess<DType>::vec_type*>(
            params.rms_gamma)[access_id_in_token];

#pragma unroll
    for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
      vals[i] = static_cast<DType>(
          static_cast<float>(vals[i]) *
          rsqrtf(
              (sum_variance / static_cast<float>(params.hidden_dim) / NRanks) +
              params.rms_eps) *
          static_cast<float>(norm_weight[i]));
    }

    reinterpret_cast<float4*>(params.rms_norm_out)[idx] =
        *reinterpret_cast<float4*>(vals);
  }
  for (int idx = access_id; idx < clear_access; idx += access_stride) {
    reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
  }
  comm.update(params.size_q * NRanks);
}

/**
 * Float4 variant: process 4 rows at once, allreduce variance sums as float4
 * for better memory coalescing.
 * IsQK: when true, process Q+K jointly with doubled comm buffer.
 * HasRoPE: when true, apply RoPE (neox or interleave) after RMS norm.
 * IsNeox: when true, use NeoX-style RoPE (warp shuffle); else interleave.
 */
template <typename DType, int NRanks, bool IsQK, bool HasRoPE = true,
          bool IsNeox = true>
__global__ void __launch_bounds__(1024)
    minimax_reduce_rms_kernel_lamport_float4(MiniMaxReduceRMSParams params) {
  int tot_tokens = params.size_q / params.hidden_dim;
  int tot_groups =
      (tot_tokens + 3) / 4;  // ceiling: last group may have 1-3 valid rows
  int access_per_row_q = params.hidden_dim / kElemsPerAccess<DType>;
  int access_per_row_k =
      IsQK ? (params.hidden_dim_k / kElemsPerAccess<DType>) : 0;
  int input_access_per_row_q =
      params.input_row_stride_q / kElemsPerAccess<DType>;
  int input_access_per_row_k =
      IsQK ? (params.input_row_stride_k / kElemsPerAccess<DType>) : 0;
  int output_access_per_row_q =
      params.output_row_stride_q / kElemsPerAccess<DType>;
  int output_access_per_row_k =
      IsQK ? (params.output_row_stride_k / kElemsPerAccess<DType>) : 0;
  int q_warps = (access_per_row_q + MINIMAX_REDUCE_RMS_WARP_SIZE - 1) /
                MINIMAX_REDUCE_RMS_WARP_SIZE;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  cg::grid_group grid = cg::this_grid();
  int group_id = grid.cluster_rank();
  int access_id_in_token = cluster.thread_rank();
  int group_stride = grid.num_clusters();
#else
  int group_id = blockIdx.x;
  int access_id_in_token = threadIdx.x;
  int group_stride = gridDim.x;
#endif
  bool is_q = (access_id_in_token < q_warps * MINIMAX_REDUCE_RMS_WARP_SIZE);
  int k_thread_idx =
      IsQK ? (access_id_in_token - q_warps * MINIMAX_REDUCE_RMS_WARP_SIZE) : 0;
  float4 clear_vec = get_neg_zero();

  LamportComm<NRanks> comm(params.workspace, params.rank);

  for (int g = group_id; g < tot_groups; g += group_stride) {
    alignas(16) DType vals[4][kElemsPerAccess<DType>]{};
    float sum_variance[4] = {0.F, 0.F, 0.F, 0.F};
    float sum_variance_k[4] = {0.F, 0.F, 0.F, 0.F};

    if (is_q) {
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        int token_r = g * 4 + r;
        if (token_r >= tot_tokens || access_id_in_token >= access_per_row_q) {
          continue;
        }
        int idx_r = token_r * input_access_per_row_q + access_id_in_token;
        *reinterpret_cast<float4*>(&vals[r][0]) =
            reinterpret_cast<float4 const*>(params.allreduce_in)[idx_r];
#pragma unroll
        for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
          float x = static_cast<float>(vals[r][i]);
          sum_variance[r] += x * x;
        }
      }
    } else if constexpr (IsQK) {
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        int token_r = g * 4 + r;
        if (token_r >= tot_tokens || k_thread_idx >= access_per_row_k) {
          continue;
        }
        int idx_r = token_r * input_access_per_row_k + k_thread_idx;
        *reinterpret_cast<float4*>(&vals[r][0]) =
            reinterpret_cast<float4 const*>(params.allreduce_in_k)[idx_r];
#pragma unroll
        for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
          float x = static_cast<float>(vals[r][i]);
          sum_variance_k[r] += x * x;
        }
      }
    }

    // Fused reduction: pack Q+K variances in one pass when IsQK
    if constexpr (IsQK) {
      float sums[8] = {
          sum_variance[0],   sum_variance[1],   sum_variance[2],
          sum_variance[3],   sum_variance_k[0], sum_variance_k[1],
          sum_variance_k[2], sum_variance_k[3],
      };
      blockReduceSumV2<float, 8>(sums);
      sum_variance[0] = sums[0];
      sum_variance[1] = sums[1];
      sum_variance[2] = sums[2];
      sum_variance[3] = sums[3];
      sum_variance_k[0] = sums[4];
      sum_variance_k[1] = sums[5];
      sum_variance_k[2] = sums[6];
      sum_variance_k[3] = sums[7];
    } else {
      blockReduceSumV2<float, 4>(sum_variance);
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      if (is_neg_zero(sum_variance[r])) sum_variance[r] = 0.F;
      if constexpr (IsQK) {
        if (is_neg_zero(sum_variance_k[r])) sum_variance_k[r] = 0.F;
      }
    }

    __shared__ float4 local_var_q;
    __shared__ float4 local_var_k;

    if constexpr (NRanks > 1) {
      if (threadIdx.x == 0) {
        float4 sum4;
        sum4.x = sum_variance[0];
        sum4.y = sum_variance[1];
        sum4.z = sum_variance[2];
        sum4.w = sum_variance[3];
        local_var_q = sum4;
#pragma unroll
        for (int r = 0; r < NRanks; ++r) {
          if constexpr (IsQK) {
            reinterpret_cast<float4*>(
                comm.data_bufs[r])[(params.rank * 2 * tot_groups) + 2 * g] =
                sum4;
          } else {
            reinterpret_cast<float4*>(
                comm.data_bufs[r])[(params.rank * tot_groups) + g] = sum4;
          }
        }
        if constexpr (IsQK) {
          float4 sum4k;
          sum4k.x = sum_variance_k[0];
          sum4k.y = sum_variance_k[1];
          sum4k.z = sum_variance_k[2];
          sum4k.w = sum_variance_k[3];
          local_var_k = sum4k;
#pragma unroll
          for (int r = 0; r < NRanks; ++r) {
            reinterpret_cast<float4*>(
                comm.data_bufs[r])[(params.rank * 2 * tot_groups) + 2 * g + 1] =
                sum4k;
          }
        }
      }
      __syncthreads();

      float4 vars_all_ranks[NRanks];
      {
        float4 lv = local_var_q;
        sum_variance[0] = lv.x;
        sum_variance[1] = lv.y;
        sum_variance[2] = lv.z;
        sum_variance[3] = lv.w;
      }

      bool done = false;
      if (is_q) {
        while (!done) {
          done = true;
#pragma unroll
          for (int r = 0; r < NRanks; ++r) {
            if (r == params.rank) continue;
            if constexpr (IsQK) {
              vars_all_ranks[r] = ld_global_volatile(&reinterpret_cast<float4*>(
                  comm.data_bufs[params.rank])[(r * 2 * tot_groups) + 2 * g]);
            } else {
              vars_all_ranks[r] = ld_global_volatile(&reinterpret_cast<float4*>(
                  comm.data_bufs[params.rank])[(r * tot_groups) + g]);
            }
            done &= !is_neg_zero(vars_all_ranks[r]);
          }
        }
      } else if constexpr (IsQK) {
        {
          float4 lk = local_var_k;
          sum_variance[0] = lk.x;
          sum_variance[1] = lk.y;
          sum_variance[2] = lk.z;
          sum_variance[3] = lk.w;
        }
        while (!done) {
          done = true;
          for (int r = 0; r < NRanks; ++r) {
            if (r == params.rank) continue;
            vars_all_ranks[r] = ld_global_volatile(&reinterpret_cast<float4*>(
                comm.data_bufs[params.rank])[(r * 2 * tot_groups) + 2 * g + 1]);
            done &= !is_neg_zero(vars_all_ranks[r]);
          }
        }
      }

#pragma unroll
      for (int r = 0; r < NRanks; ++r) {
        if (r == params.rank) continue;
        sum_variance[0] += vars_all_ranks[r].x;
        sum_variance[1] += vars_all_ranks[r].y;
        sum_variance[2] += vars_all_ranks[r].z;
        sum_variance[3] += vars_all_ranks[r].w;
      }
    } else {
      // NRanks == 1: broadcast via shared memory, no cross-rank communication
      if (threadIdx.x == 0) {
        float4 sum4;
        sum4.x = sum_variance[0];
        sum4.y = sum_variance[1];
        sum4.z = sum_variance[2];
        sum4.w = sum_variance[3];
        local_var_q = sum4;
        if constexpr (IsQK) {
          float4 sum4k;
          sum4k.x = sum_variance_k[0];
          sum4k.y = sum_variance_k[1];
          sum4k.z = sum_variance_k[2];
          sum4k.w = sum_variance_k[3];
          local_var_k = sum4k;
        }
      }
      __syncthreads();
      if (is_q) {
        float4 lv = local_var_q;
        sum_variance[0] = lv.x;
        sum_variance[1] = lv.y;
        sum_variance[2] = lv.z;
        sum_variance[3] = lv.w;
      } else if constexpr (IsQK) {
        float4 lk = local_var_k;
        sum_variance[0] = lk.x;
        sum_variance[1] = lk.y;
        sum_variance[2] = lk.z;
        sum_variance[3] = lk.w;
      }
    }

    // RMS norm + optional RoPE, then store
    if (is_q) {
      bool can_store_q = (access_id_in_token < access_per_row_q);

      int threads_per_head_q = 0, thread_in_head_q = 0, embed_dim_q = 0;
      int pair_offset_q = 0, threads_in_rotary_q = 0;
      bool in_rotary_q = false, is_first_half_q = false;
      if constexpr (HasRoPE) {
        threads_per_head_q = params.head_size / kElemsPerAccess<DType>;
        thread_in_head_q = access_id_in_token % threads_per_head_q;
        embed_dim_q = params.rot_dim / 2;
        pair_offset_q = embed_dim_q / kElemsPerAccess<DType>;
        threads_in_rotary_q = params.rot_dim / kElemsPerAccess<DType>;
        in_rotary_q = (thread_in_head_q < threads_in_rotary_q);
        is_first_half_q = (thread_in_head_q < pair_offset_q);
      }

      alignas(16) DType norm_weight[kElemsPerAccess<DType>]{};
      if (can_store_q) {
        *reinterpret_cast<typename ElemsPerAccess<DType>::vec_type*>(
            norm_weight) =
            reinterpret_cast<typename ElemsPerAccess<DType>::vec_type const*>(
                params.rms_gamma)[access_id_in_token];
      }

#pragma unroll
      for (int r = 0; r < 4; ++r) {
        int token_r = g * 4 + r;
        if (token_r >= tot_tokens) continue;

        float fvals[kElemsPerAccess<DType>]{};
        if (can_store_q) {
          float scale =
              rsqrtf((sum_variance[r] / static_cast<float>(params.hidden_dim) /
                      NRanks) +
                     params.rms_eps);
#pragma unroll
          for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
            fvals[i] = static_cast<float>(vals[r][i]) * scale *
                       static_cast<float>(norm_weight[i]);
          }
        }

        if constexpr (HasRoPE && IsNeox) {
          int64_t pos =
              (can_store_q && in_rotary_q) ? params.positions[token_r] : 0;
          DType const* cos_base =
              reinterpret_cast<DType const*>(params.cos_sin_cache) +
              pos * params.rot_dim;
          __syncwarp();  // reconverge after divergent fvals computation
#pragma unroll
          for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
            float paired = __shfl_xor_sync(FINAL_MASK, fvals[i], pair_offset_q);
            if (in_rotary_q && can_store_q) {
              if (is_first_half_q) paired = -paired;
              int dim_in_head = thread_in_head_q * kElemsPerAccess<DType> + i;
              // Equivalent to ((dim_in_head*2) % rot_dim) / 2, but avoids
              // expensive runtime integer modulo.
              int half_dim =
                  is_first_half_q ? dim_in_head : (dim_in_head - embed_dim_q);
              float cos_val = static_cast<float>(__ldg(cos_base + half_dim));
              float sin_val =
                  static_cast<float>(__ldg(cos_base + embed_dim_q + half_dim));
              fvals[i] = fvals[i] * cos_val + paired * sin_val;
            }
          }
          // No __syncwarp() needed: __shfl_xor_sync(FINAL_MASK,...) is the
          // last warp-synchronizing op in this iteration; the next iteration's
          // __syncwarp() reconverges before the following shuffle.
        } else if constexpr (HasRoPE && !IsNeox) {
          if (in_rotary_q && can_store_q) {
            int64_t pos = params.positions[token_r];
            DType const* cos_base =
                reinterpret_cast<DType const*>(params.cos_sin_cache) +
                pos * params.rot_dim;
#pragma unroll
            for (int i = 0; i < kElemsPerAccess<DType>; i += 2) {
              int dim_in_head = thread_in_head_q * kElemsPerAccess<DType> + i;
              int half_dim = dim_in_head / 2;
              float cos_val = static_cast<float>(__ldg(cos_base + half_dim));
              float sin_val =
                  static_cast<float>(__ldg(cos_base + embed_dim_q + half_dim));
              float x = fvals[i], y = fvals[i + 1];
              fvals[i] = x * cos_val - y * sin_val;
              fvals[i + 1] = x * sin_val + y * cos_val;
            }
          }
        }

        if (can_store_q) {
#pragma unroll
          for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
            vals[r][i] = static_cast<DType>(fvals[i]);
          }
          int idx_out = token_r * output_access_per_row_q + access_id_in_token;
          reinterpret_cast<float4*>(params.rms_norm_out)[idx_out] =
              *reinterpret_cast<float4*>(&vals[r][0]);
        }
      }
    } else if constexpr (IsQK) {
      bool can_store_k = (k_thread_idx < access_per_row_k);

      int threads_per_head_k = 0, thread_in_head_k = 0, embed_dim_k = 0;
      int pair_offset_k = 0, threads_in_rotary_k = 0;
      bool in_rotary_k = false, is_first_half_k = false;
      if constexpr (HasRoPE) {
        threads_per_head_k = params.head_size / kElemsPerAccess<DType>;
        thread_in_head_k = k_thread_idx % threads_per_head_k;
        embed_dim_k = params.rot_dim / 2;
        pair_offset_k = embed_dim_k / kElemsPerAccess<DType>;
        threads_in_rotary_k = params.rot_dim / kElemsPerAccess<DType>;
        in_rotary_k = (thread_in_head_k < threads_in_rotary_k);
        is_first_half_k = (thread_in_head_k < pair_offset_k);
      }

      alignas(16) DType norm_weight_k[kElemsPerAccess<DType>]{};
      if (can_store_k) {
        *reinterpret_cast<typename ElemsPerAccess<DType>::vec_type*>(
            norm_weight_k) =
            reinterpret_cast<typename ElemsPerAccess<DType>::vec_type const*>(
                params.rms_gamma_k)[k_thread_idx];
      }

#pragma unroll
      for (int r = 0; r < 4; ++r) {
        int token_r = g * 4 + r;
        if (token_r >= tot_tokens) continue;

        float fvals_k[kElemsPerAccess<DType>]{};
        if (can_store_k) {
          float scale_k =
              rsqrtf((sum_variance[r] /
                      static_cast<float>(params.hidden_dim_k) / NRanks) +
                     params.rms_eps);
#pragma unroll
          for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
            fvals_k[i] = static_cast<float>(vals[r][i]) * scale_k *
                         static_cast<float>(norm_weight_k[i]);
          }
        }

        if constexpr (HasRoPE && IsNeox) {
          int64_t pos =
              (can_store_k && in_rotary_k) ? params.positions[token_r] : 0;
          DType const* cos_base =
              reinterpret_cast<DType const*>(params.cos_sin_cache) +
              pos * params.rot_dim;
          __syncwarp();  // reconverge after divergent fvals_k computation
#pragma unroll
          for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
            float paired =
                __shfl_xor_sync(FINAL_MASK, fvals_k[i], pair_offset_k);
            if (in_rotary_k && can_store_k) {
              if (is_first_half_k) paired = -paired;
              int dim_in_head = thread_in_head_k * kElemsPerAccess<DType> + i;
              int half_dim =
                  is_first_half_k ? dim_in_head : (dim_in_head - embed_dim_k);
              float cos_val = static_cast<float>(__ldg(cos_base + half_dim));
              float sin_val =
                  static_cast<float>(__ldg(cos_base + embed_dim_k + half_dim));
              fvals_k[i] = fvals_k[i] * cos_val + paired * sin_val;
            }
          }
          // No __syncwarp() needed here (see Q section comment above).
        } else if constexpr (HasRoPE && !IsNeox) {
          if (in_rotary_k && can_store_k) {
            int64_t pos = params.positions[token_r];
            DType const* cos_base =
                reinterpret_cast<DType const*>(params.cos_sin_cache) +
                pos * params.rot_dim;
#pragma unroll
            for (int i = 0; i < kElemsPerAccess<DType>; i += 2) {
              int dim_in_head = thread_in_head_k * kElemsPerAccess<DType> + i;
              int half_dim = dim_in_head / 2;
              float cos_val = static_cast<float>(__ldg(cos_base + half_dim));
              float sin_val =
                  static_cast<float>(__ldg(cos_base + embed_dim_k + half_dim));
              float x = fvals_k[i], y = fvals_k[i + 1];
              fvals_k[i] = x * cos_val - y * sin_val;
              fvals_k[i + 1] = x * sin_val + y * cos_val;
            }
          }
        }

        if (can_store_k) {
#pragma unroll
          for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
            vals[r][i] = static_cast<DType>(fvals_k[i]);
          }
          int idx_out = token_r * output_access_per_row_k + k_thread_idx;
          reinterpret_cast<float4*>(params.rms_norm_out_k)[idx_out] =
              *reinterpret_cast<float4*>(&vals[r][0]);
        }
      }
    }
  }

  if constexpr (NRanks > 1) {
    int clear_access =
        static_cast<int>(comm.clear_size / kElemsPerAccess<DType>);
    int clear_stride = group_stride * blockDim.x;
    for (int idx = group_id * blockDim.x + threadIdx.x; idx < clear_access;
         idx += clear_stride) {
      reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
    }
    comm.update(IsQK ? (2 * tot_groups * 8 * NRanks)
                     : (tot_groups * 8 * NRanks));
  }
}

int get_sm_count() {
  static int sm_count = 0;
  if (sm_count == 0) {
    int device_id;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    sm_count = device_prop.multiProcessorCount;
  }
  return sm_count;
}

template <typename KernelFunc>
int get_max_active_blocks(KernelFunc kernel, int block_size,
                          int dynamic_smem = 0) {
  int max_active = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active, kernel, block_size, dynamic_smem));
  return std::max(max_active, 1);
}

inline int getSMVersion(bool queryRealSmArch = false) {
  int device{-1};
  CUDA_CHECK(cudaGetDevice(&device));
  int sm_major = 0, sm_minor = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_major,
                                    cudaDevAttrComputeCapabilityMajor, device));
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_minor,
                                    cudaDevAttrComputeCapabilityMinor, device));
  int sm = sm_major * 10 + sm_minor;
  if (sm == 121 && !queryRealSmArch) return 120;
  return sm;
}

template <typename DType, int NRanks>
void minimax_reduce_rms_kernel_launcher(MiniMaxReduceRMSParams const& params) {
  static int SM = getSMVersion();
  int token_num = params.size_q / params.hidden_dim;
  int sm_count = get_sm_count();
  int cluster_size = 1;
  int threads_per_token = params.hidden_dim / kElemsPerAccess<DType>;

  int max_blocks_per_sm = get_max_active_blocks(
      minimax_reduce_rms_kernel_lamport<DType, NRanks>, threads_per_token);
  int max_grid = max_blocks_per_sm * sm_count;
  int grid_size =
      (std::min(max_grid, token_num * cluster_size) / cluster_size) *
      cluster_size;

  cudaLaunchConfig_t cfg;
  cfg.gridDim = grid_size;
  cfg.blockDim = threads_per_token;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = params.stream;

  cudaLaunchAttribute attribute[2];
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[1].id = cudaLaunchAttributeClusterDimension;
  attribute[1].val.clusterDim.x = cluster_size;
  attribute[1].val.clusterDim.y = 1;
  attribute[1].val.clusterDim.z = 1;
  cfg.attrs = attribute;
  cfg.numAttrs = SM >= 90 ? 2 : 0;

  CUDA_CHECK(cudaLaunchKernelEx(
      &cfg, minimax_reduce_rms_kernel_lamport<DType, NRanks>, params));
}

template <typename DType, int NRanks>
void minimax_reduce_rms_kernel_launcher_float4(
    MiniMaxReduceRMSParams const& params) {
  TORCH_CHECK(params.size_q % params.hidden_dim == 0);
  TORCH_CHECK(params.hidden_dim % kElemsPerAccess<DType> == 0);
  if (params.input_row_stride_q > 0) {
    TORCH_CHECK(params.input_row_stride_q % kElemsPerAccess<DType> == 0);
  }
  if (params.allreduce_in_k != nullptr) {
    TORCH_CHECK(params.hidden_dim >= params.hidden_dim_k);
    TORCH_CHECK(params.size_k % params.hidden_dim_k == 0);
    TORCH_CHECK(params.hidden_dim_k % kElemsPerAccess<DType> == 0);
    TORCH_CHECK(params.size_q / params.hidden_dim ==
                params.size_k / params.hidden_dim_k);
    if (params.input_row_stride_k > 0) {
      TORCH_CHECK(params.input_row_stride_k % kElemsPerAccess<DType> == 0);
    }
  }
  int token_num = params.size_q / params.hidden_dim;
  int tot_groups = (token_num + 3) / 4;
  if (tot_groups == 0) return;

  static int SM = getSMVersion();
  int sm_count = get_sm_count();
  int access_per_row_q = params.hidden_dim / kElemsPerAccess<DType>;
  int access_per_row_k = (params.allreduce_in_k != nullptr)
                             ? (params.hidden_dim_k / kElemsPerAccess<DType>)
                             : 0;
  auto divUp = [](int a, int b) { return (a + b - 1) / b * b; };
  int block_size = divUp(access_per_row_q, MINIMAX_REDUCE_RMS_WARP_SIZE) +
                   ((params.allreduce_in_k != nullptr)
                        ? divUp(access_per_row_k, MINIMAX_REDUCE_RMS_WARP_SIZE)
                        : 0);
  bool is_qk = (params.allreduce_in_k != nullptr);
  bool has_rope = (params.cos_sin_cache != nullptr && params.rot_dim > 0);

#define GET_OCCUPANCY(QK, ROPE, NEOX)                                          \
  get_max_active_blocks(                                                       \
      minimax_reduce_rms_kernel_lamport_float4<DType, NRanks, QK, ROPE, NEOX>, \
      block_size)

  int max_blocks_per_sm;
  if (is_qk && has_rope && params.is_neox) {
    max_blocks_per_sm = GET_OCCUPANCY(true, true, true);
  } else if (is_qk && has_rope && !params.is_neox) {
    max_blocks_per_sm = GET_OCCUPANCY(true, true, false);
  } else if (is_qk && !has_rope) {
    max_blocks_per_sm = GET_OCCUPANCY(true, false, false);
  } else if (!is_qk && has_rope && params.is_neox) {
    max_blocks_per_sm = GET_OCCUPANCY(false, true, true);
  } else if (!is_qk && has_rope && !params.is_neox) {
    max_blocks_per_sm = GET_OCCUPANCY(false, true, false);
  } else {
    max_blocks_per_sm = GET_OCCUPANCY(false, false, false);
  }
#undef GET_OCCUPANCY

  int max_grid = max_blocks_per_sm * sm_count;
  int grid_size = std::min(max_grid, tot_groups);

  cudaLaunchConfig_t cfg;
  cfg.gridDim = grid_size;
  cfg.blockDim = block_size;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = params.stream;

  cudaLaunchAttribute attribute[2];
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[1].id = cudaLaunchAttributeClusterDimension;
  attribute[1].val.clusterDim.x = 1;
  attribute[1].val.clusterDim.y = 1;
  attribute[1].val.clusterDim.z = 1;
  cfg.attrs = attribute;
  cfg.numAttrs = SM >= 90 ? 2 : 0;

#define LAUNCH_KERNEL(QK, ROPE, NEOX)                                          \
  CUDA_CHECK(cudaLaunchKernelEx(                                               \
      &cfg,                                                                    \
      minimax_reduce_rms_kernel_lamport_float4<DType, NRanks, QK, ROPE, NEOX>, \
      params))

  if (is_qk && has_rope && params.is_neox) {
    LAUNCH_KERNEL(true, true, true);
  } else if (is_qk && has_rope && !params.is_neox) {
    LAUNCH_KERNEL(true, true, false);
  } else if (is_qk && !has_rope) {
    LAUNCH_KERNEL(true, false, false);
  } else if (!is_qk && has_rope && params.is_neox) {
    LAUNCH_KERNEL(false, true, true);
  } else if (!is_qk && has_rope && !params.is_neox) {
    LAUNCH_KERNEL(false, true, false);
  } else {
    LAUNCH_KERNEL(false, false, false);
  }
#undef LAUNCH_KERNEL
}

template <int NRanks>
void dispatch_dtype(MiniMaxReduceRMSParams const& params) {
  constexpr bool use_float4 = true;

  if (params.dtype == at::ScalarType::Half) {
    if constexpr (use_float4) {
      minimax_reduce_rms_kernel_launcher_float4<half, NRanks>(params);
    } else {
      minimax_reduce_rms_kernel_launcher<half, NRanks>(params);
    }
  } else if (params.dtype == at::ScalarType::BFloat16) {
    if constexpr (use_float4) {
      minimax_reduce_rms_kernel_launcher_float4<__nv_bfloat16, NRanks>(params);
    } else {
      minimax_reduce_rms_kernel_launcher<__nv_bfloat16, NRanks>(params);
    }
  } else if (params.dtype == at::ScalarType::Float) {
    if constexpr (use_float4) {
      minimax_reduce_rms_kernel_launcher_float4<float, NRanks>(params);
    } else {
      minimax_reduce_rms_kernel_launcher<float, NRanks>(params);
    }
  } else {
    TORCH_CHECK(false, "Unsupported data type for minimax_reduce_rms_op");
  }
}

void minimax_reduce_rms_op(MiniMaxReduceRMSParams const& params) {
  if (params.nranks == 1) {
    dispatch_dtype<1>(params);
  } else if (params.nranks == 2) {
    dispatch_dtype<2>(params);
  } else if (params.nranks == 4) {
    dispatch_dtype<4>(params);
  } else if (params.nranks == 8) {
    dispatch_dtype<8>(params);
  } else if (params.nranks == 16) {
    dispatch_dtype<16>(params);
  } else {
    TORCH_CHECK(false, "minimax_reduce_rms_op: unsupported ranks number!");
  }
}
}  // namespace tensorrt_llm
}  // namespace vllm

// ============================================================================
// C++ / Torch wrapper functions
// ============================================================================

torch::Tensor minimax_allreduce_rms(
    torch::Tensor const& input, torch::Tensor const& norm_weight,
    std::optional<torch::Tensor> const& workspace, int64_t const rank,
    int64_t const nranks, double const eps) {
  auto params = vllm::tensorrt_llm::MiniMaxReduceRMSParams();
  params.nranks = static_cast<int>(nranks);
  params.rank = static_cast<int>(rank);
  params.dtype = input.scalar_type();
  params.size_q = static_cast<int>(input.numel());
  params.hidden_dim = static_cast<int>(input.size(-1));
  params.workspace =
      workspace.has_value()
          ? reinterpret_cast<void**>(workspace->mutable_data_ptr())
          : nullptr;
  params.allreduce_in = input.data_ptr();
  params.rms_gamma = norm_weight.data_ptr();
  params.rms_eps = static_cast<float>(eps);
  params.stream = at::cuda::getCurrentCUDAStream(input.get_device());
  params.input_row_stride_q = static_cast<int>(input.size(-1));
  params.output_row_stride_q = static_cast<int>(input.size(-1));

  torch::Tensor rms_norm_out = torch::empty_like(input);
  params.rms_norm_out = rms_norm_out.mutable_data_ptr();

  vllm::tensorrt_llm::minimax_reduce_rms_op(params);
  return rms_norm_out;
}

void minimax_allreduce_rms_qk(torch::Tensor qkv,
                              torch::Tensor const& norm_weight_q,
                              torch::Tensor const& norm_weight_k,
                              torch::Tensor workspace, int64_t const q_size,
                              int64_t const kv_size, int64_t const rank,
                              int64_t const nranks, double const eps) {
  TORCH_CHECK(qkv.dim() == 2, "minimax_allreduce_rms_qk: qkv must be 2D");
  TORCH_CHECK(qkv.is_contiguous(),
              "minimax_allreduce_rms_qk: qkv must be contiguous");
  int64_t qkv_dim = qkv.size(-1);
  TORCH_CHECK(qkv_dim == q_size + 2 * kv_size,
              "minimax_allreduce_rms_qk: qkv last dim must equal "
              "q_size + 2 * kv_size");
  TORCH_CHECK(rank < nranks,
              "minimax_allreduce_rms_qk: rank must be less than nranks");

  int64_t num_tokens = qkv.size(0);
  int elem_bytes = qkv.element_size();

  auto params = vllm::tensorrt_llm::MiniMaxReduceRMSParams();
  params.nranks = static_cast<int>(nranks);
  params.rank = static_cast<int>(rank);
  params.dtype = qkv.scalar_type();
  params.size_q = static_cast<int>(num_tokens * q_size);
  params.hidden_dim = static_cast<int>(q_size);
  params.size_k = static_cast<int>(num_tokens * kv_size);
  params.hidden_dim_k = static_cast<int>(kv_size);
  params.input_row_stride_q = static_cast<int>(qkv_dim);
  params.input_row_stride_k = static_cast<int>(qkv_dim);
  params.output_row_stride_q = static_cast<int>(qkv_dim);
  params.output_row_stride_k = static_cast<int>(qkv_dim);
  params.workspace = reinterpret_cast<void**>(workspace.mutable_data_ptr());

  uint8_t* base = static_cast<uint8_t*>(qkv.data_ptr());
  params.allreduce_in = base;
  params.allreduce_in_k = base + q_size * elem_bytes;
  params.rms_gamma = norm_weight_q.data_ptr();
  params.rms_gamma_k = norm_weight_k.data_ptr();
  params.rms_eps = static_cast<float>(eps);
  params.stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

  params.rms_norm_out = params.allreduce_in;
  params.rms_norm_out_k = params.allreduce_in_k;

  vllm::tensorrt_llm::minimax_reduce_rms_op(params);
}

void minimax_allreduce_rms_rope_fusion(
    torch::Tensor& qkv, int64_t const q_size, int64_t const kv_size,
    torch::Tensor const& norm_weight_q, torch::Tensor const& norm_weight_k,
    std::optional<torch::Tensor> const& workspace, int64_t const rank,
    int64_t const nranks, double const eps, torch::Tensor const& positions,
    torch::Tensor const& cos_sin_cache, int64_t const head_size,
    bool const is_neox) {
  TORCH_CHECK(qkv.dim() == 2,
              "minimax_allreduce_rms_rope_fusion: qkv must be 2D");
  int64_t num_tokens = qkv.size(0);
  int64_t total_hidden = qkv.size(-1);
  TORCH_CHECK(total_hidden == q_size + 2 * kv_size,
              "minimax_allreduce_rms_rope_fusion: total_hidden must equal "
              "q_size + 2*kv_size");
  TORCH_CHECK(rank < nranks,
              "minimax_allreduce_rms_rope_fusion: rank must be < nranks");
  TORCH_CHECK(positions.dim() == 1 && positions.scalar_type() == torch::kInt64,
              "minimax_allreduce_rms_rope_fusion: positions must be 1D int64");
  TORCH_CHECK(positions.size(0) == num_tokens,
              "minimax_allreduce_rms_rope_fusion: positions length must match "
              "num_tokens");
  TORCH_CHECK(cos_sin_cache.dim() == 2,
              "minimax_allreduce_rms_rope_fusion: cos_sin_cache must be 2D");
  int64_t rot_dim = cos_sin_cache.size(1);
  TORCH_CHECK(rot_dim % 2 == 0 && rot_dim <= head_size,
              "minimax_allreduce_rms_rope_fusion: invalid rot_dim");
  TORCH_CHECK(q_size % head_size == 0 && kv_size % head_size == 0,
              "minimax_allreduce_rms_rope_fusion: q/kv_size must be divisible "
              "by head_size");

  auto params = vllm::tensorrt_llm::MiniMaxReduceRMSParams();
  params.nranks = static_cast<int>(nranks);
  params.rank = static_cast<int>(rank);
  params.dtype = qkv.scalar_type();
  params.size_q = static_cast<int>(num_tokens * q_size);
  params.hidden_dim = static_cast<int>(q_size);
  params.size_k = static_cast<int>(num_tokens * kv_size);
  params.hidden_dim_k = static_cast<int>(kv_size);
  params.workspace =
      workspace.has_value()
          ? reinterpret_cast<void**>(workspace->mutable_data_ptr())
          : nullptr;

  void* qkv_ptr = qkv.mutable_data_ptr();
  params.allreduce_in = qkv_ptr;
  params.allreduce_in_k =
      static_cast<char*>(qkv_ptr) + q_size * qkv.element_size();
  params.rms_norm_out = qkv_ptr;
  params.rms_norm_out_k =
      static_cast<char*>(qkv_ptr) + q_size * qkv.element_size();

  params.input_row_stride_q = static_cast<int>(total_hidden);
  params.input_row_stride_k = static_cast<int>(total_hidden);
  params.output_row_stride_q = static_cast<int>(total_hidden);
  params.output_row_stride_k = static_cast<int>(total_hidden);

  params.rms_gamma = norm_weight_q.data_ptr();
  params.rms_gamma_k = norm_weight_k.data_ptr();
  params.rms_eps = static_cast<float>(eps);
  params.stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

  params.positions = positions.data_ptr<int64_t>();
  params.cos_sin_cache = cos_sin_cache.data_ptr();
  params.head_size = static_cast<int>(head_size);
  params.rot_dim = static_cast<int>(rot_dim);
  params.is_neox = is_neox;

  vllm::tensorrt_llm::minimax_reduce_rms_op(params);
}
