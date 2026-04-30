/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Fused AllReduce + hc_post kernel for DeepseekV4-style models.
 *
 * Reuses the FlashInfer trtllm AR-fusion workspace (one-shot Lamport layout):
 * `workspace[0..NRanks-1]`           : peer ipc data buffers
 * `workspace[NRanks..2*NRanks-1]`    : peer barrier buffers
 * `workspace[2*NRanks..3*NRanks-1]`  : peer Lamport-roundrobin data buffers
 * `workspace[3*NRanks]`              : counter/flag/clear int32 metadata
 *
 * The FlashInfer create_allreduce_fusion_workspace() routine in
 * `flashinfer.comm.allreduce` creates a workspace tensor with this layout;
 * vLLM acquires it via
 * `vllm/distributed/device_communicators/flashinfer_all_reduce.py:get_fi_ar_workspace()`.
 */

#pragma once

#include <cuda_bf16.h>
#include <torch/types.h>

namespace vllm {
namespace trtllm_ar_hc_post {

struct Params {
  int nranks{};
  int rank{};
  int num_tokens{};
  int hidden_dim{};
  int hc_mult{};               // expected to be 4 for DeepseekV4
  void** workspace{};          // device-side void* array (FlashInfer layout)
  void const* allreduce_in{};  // [num_tokens, hidden_dim]                bf16
  void const* residual{};      // [num_tokens, hc_mult, hidden_dim]       bf16
  void const* post{};          // [num_tokens, hc_mult]                   fp32
  void const* comb{};          // [num_tokens, hc_mult, hc_mult]          fp32
  void* out{};                 // [num_tokens, hc_mult, hidden_dim]       bf16
  // Two-shot only: token partitioning across ranks. Size capped at the
  // max supported NRanks (8). Unused slots are zero.
  int begin_tokens[8]{};
  int token_num_per[8]{};
  cudaStream_t stream{};
};

// `use_oneshot`:
//   - true  → one-shot Lamport (each rank pushes full data, polls peers); best
//             for small token counts where AR latency dominates.
//   - false → two-shot reduce-scatter + all-gather with Barrier sync; best for
//             large token counts where AR bandwidth dominates.
void run(Params const& params, bool launch_with_pdl, bool use_oneshot);

}  // namespace trtllm_ar_hc_post
}  // namespace vllm
