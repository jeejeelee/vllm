"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023).
Punica: Multi-Tenant LoRA Serving.
https://arxiv.org/abs/2310.18547
"""

from typing import Dict, List, Tuple

import torch
import triton
import triton.language as tl

from vllm.utils import direct_register_custom_op


@triton.jit
def _sgmv_expand_slice_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    b_seq_start_loc,
    seq_lens,
    lora_indices,
    slice_start_loc,
    input_d0_stride,
    input_d1_stride,
    input_d2_stride,  # 1
    ls_d0_ptr,  # lora stride(0)
    ls_d1_ptr,
    ls_d2_ptr,
    cm_stride,
    cn_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    CAST_TYPE: tl.constexpr,
):
    """

    Similar to the 'sgmv_expand' operator, but with an added parameter
    'slice_offset'. The reason for not reusing the 'sgmv_expand' operator
    might be that in the future, we could implement a fusion operator to
    achieve the current functionality instead of having to call it multiple
    times.
    """
    pid = tl.program_id(axis=0)
    cur_batch = tl.program_id(axis=1)
    slice_id = tl.program_id(axis=2)
    cta_n_num = tl.cdiv(N, BLOCK_N)
    pid_m = pid // cta_n_num
    pid_n = pid % cta_n_num
    M = tl.load(seq_lens + cur_batch)
    if pid_m * BLOCK_M > M:
        return
    lora_index = tl.load(lora_indices + cur_batch)
    if lora_index == -1:
        return

    cur_seq_start = tl.load(b_seq_start_loc + cur_batch)
    offset_m = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    offset_k = tl.arange(0, BLOCK_K)
    ram = tl.max_contiguous(tl.multiple_of(offset_m % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)

    # input
    cur_input_ptr = input_ptr + slice_id * input_d0_stride
    a_ptr = (cur_input_ptr + cur_seq_start * input_d1_stride +
             ram[:, None] * input_d1_stride +
             offset_k[None, :] * input_d2_stride, )
    # lora
    cur_lora_ptr = tl.load(lora_ptr + slice_id).to(
        tl.pointer_type(out_ptr.dtype.element_ty))
    cur_lora_d0_stride = tl.load(ls_d0_ptr + slice_id)
    cur_lora_d1_stride = tl.load(ls_d1_ptr + slice_id)
    cur_lora_d2_stride = tl.load(ls_d2_ptr + slice_id)

    b_ptr = (cur_lora_ptr + cur_lora_d0_stride * lora_index +
             offset_k[:, None] * cur_lora_d2_stride +
             rbn[None, :] * cur_lora_d1_stride)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            tiled_a = tl.load(a_ptr)
            tiled_b = tl.load(b_ptr)
        else:
            tiled_a = tl.load(a_ptr,
                              mask=offset_k[None, :] < K - k * BLOCK_K,
                              other=0)
            tiled_b = tl.load(b_ptr,
                              mask=offset_k[:, None] < K - k * BLOCK_K,
                              other=0)
        if CAST_TYPE:
            tiled_a = tiled_a.to(cur_lora_ptr.dtype.element_ty)
        accumulator += tl.dot(
            tiled_a,
            tiled_b,
        )
        a_ptr += BLOCK_K * input_d2_stride
        b_ptr += BLOCK_K * cur_lora_d2_stride

    tiled_c = accumulator.to(cur_lora_ptr.dtype.element_ty)

    cur_slice_start = tl.load(slice_start_loc + slice_id)

    offset_cm = cur_seq_start + tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N + cur_slice_start
    c_ptr = (out_ptr + offset_cm[:, None] * cm_stride +
             offset_cn[None, :] * cn_stride)
    M = tl.load(seq_lens + cur_batch)
    c_mask = (offset_cm[:, None] <
              (cur_seq_start + M)) & (offset_cn[None, :] <
                                      (cur_slice_start + N))
    if ADD_INPUTS:
        tiled_out = tl.load(c_ptr, mask=c_mask)
        tiled_c += tiled_out
    tl.store(c_ptr, tiled_c, mask=c_mask)


_LORA_PTR_DICT: Dict[Tuple[int, ...], Tuple[torch.tensor, ...]] = {}


#TODO Optimize
def _get_lora_ptr(lora_weights, offset_start, device):

    key = tuple(lora_weight.data_ptr() for lora_weight in lora_weights)
    if _LORA_PTR_DICT.get(key) is None:
        slice_offset_lst = []
        tensor_ptrs = []
        lora_strides_d0 = []
        lora_strides_d1 = []
        lora_strides_d2 = []
        slice_offset = offset_start
        for lora_b_weight in lora_weights:
            if lora_b_weight.ndim == 4:  # shape:(lora_num,1,size,rank)
                assert lora_b_weight.size(1) == 1
                lora_b_weight = lora_b_weight.squeeze(dim=1)
            else:
                assert lora_b_weight.ndim == 3  # shape:(lora_num,size,rank)
            assert lora_b_weight.is_contiguous()
            tensor_ptrs.append(lora_b_weight.data_ptr())
            lora_strides_d0.append(lora_b_weight.stride(0))
            lora_strides_d1.append(lora_b_weight.stride(1))
            lora_strides_d2.append(lora_b_weight.stride(2))
            slice_offset_lst.append(slice_offset)
            slice_offset += lora_b_weight.size(1)

        slice_start_tensor = torch.tensor(slice_offset_lst, device=device)
        # note these are device tensors
        lora_ptr_tensor = torch.tensor(tensor_ptrs, device=device)
        lora_strides_d0_tensor = torch.tensor(lora_strides_d0, device=device)
        lora_strides_d1_tensor = torch.tensor(lora_strides_d1, device=device)
        lora_strides_d2_tensor = torch.tensor(lora_strides_d2, device=device)

        _LORA_PTR_DICT[key] = (
            slice_start_tensor,
            lora_ptr_tensor,
            lora_strides_d0_tensor,
            lora_strides_d1_tensor,
            lora_strides_d2_tensor,
        )
    return _LORA_PTR_DICT.get(key)


@torch.inference_mode()
def _sgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_stacked: List[torch.Tensor],
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    offset_start: int = 0,
    add_inputs: bool = False,
) -> None:
    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert lora_b_stacked[0].dtype in [
        torch.float16,
        torch.bfloat16,
    ]

    assert inputs.size(1) == token_nums
    assert inputs.size(0) == len(lora_b_stacked)

    assert b_seq_start_loc.size(0) == batches
    assert lora_indices_tensor.size(0) == batches
    assert output_tensor.is_contiguous()
    (
        slice_start_tensor,
        lora_ptr_tensor,
        lora_strides_d0_tensor,
        lora_strides_d1_tensor,
        lora_strides_d2_tensor,
    ) = _get_lora_ptr(lora_b_stacked, offset_start, b_seq_start_loc.device)

    # TODO tuning this config
    N, K = lora_b_stacked[0].shape[-2:]  # K= rank,N=hidden_size

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 16
    EVEN_K = K % BLOCK_K == 0
    ADD_INPUTS = add_inputs
    CAST_TYPE = False

    if inputs.dtype == torch.float32 and lora_b_stacked[0].dtype in [
            torch.float16,
            torch.bfloat16,
    ]:
        CAST_TYPE = True
    grid = (
        triton.cdiv(max_seq_length, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        batches,
        len(lora_ptr_tensor),
    )
    _sgmv_expand_slice_kernel[grid](
        inputs,
        lora_ptr_tensor,
        output_tensor,
        N,
        K,
        b_seq_start_loc,
        seq_len_tensor,
        lora_indices_tensor,
        slice_start_tensor,
        inputs.stride(0),
        inputs.stride(1),
        inputs.stride(2),
        lora_strides_d0_tensor,
        lora_strides_d1_tensor,
        lora_strides_d2_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        ADD_INPUTS,
        CAST_TYPE,
    )
    return


def _sgmv_expand_slice_fake(
    inputs: torch.Tensor,
    lora_b_stacked: Tuple[torch.Tensor, ...],
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = False,
) -> None:
    return


try:
    direct_register_custom_op(
        op_name="sgmv_expand_slice",
        op_func=_sgmv_expand_slice,
        mutates_args=["output_tensor"],
        fake_impl=_sgmv_expand_slice_fake,
    )
    sgmv_expand_slice = torch.ops.vllm.sgmv_expand_slice

except AttributeError:
    sgmv_expand_slice = _sgmv_expand_slice
