
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

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <torch/types.h>

namespace vllm {
namespace tensorrt_llm {

template <typename DType>
struct ElemsPerAccess;

template <>
struct ElemsPerAccess<half> {
  static constexpr int value = 8;
  using vec_type = float4;
};

template <>
struct ElemsPerAccess<nv_bfloat16> {
  static constexpr int value = 8;
  using vec_type = float4;
};

template <>
struct ElemsPerAccess<float> {
  static constexpr int value = 4;
  using vec_type = float4;
};

template <typename DType>
static constexpr int kElemsPerAccess = ElemsPerAccess<DType>::value;

struct MiniMaxReduceRMSParams {
  int nranks{};
  int rank{};
  at::ScalarType dtype{at::ScalarType::Undefined};
  int size_q{};
  int hidden_dim{};
  int size_k{};
  int hidden_dim_k{};
  void** workspace{};
  void* allreduce_in{};
  void* rms_norm_out{};
  void* rms_gamma{};
  void* allreduce_in_k{};
  void* rms_norm_out_k{};
  void* rms_gamma_k{};
  float rms_eps{};
  cudaStream_t stream{};

  // RoPE parameters (set cos_sin_cache=nullptr to disable RoPE)
  int64_t* positions{};
  void* cos_sin_cache{};
  int head_size{};
  int rot_dim{};
  bool is_neox{true};

  // Strided I/O: row stride (in elements) for Q and K buffers.
  // For contiguous Q/K, set to hidden_dim / hidden_dim_k respectively.
  // For fused qkv in-place, set all four to (q_size + 2*kv_size).
  int input_row_stride_q{};
  int input_row_stride_k{};
  int output_row_stride_q{};
  int output_row_stride_k{};
};

void minimax_reduce_rms_op(MiniMaxReduceRMSParams const& params);

}  // namespace tensorrt_llm
}  // namespace vllm
