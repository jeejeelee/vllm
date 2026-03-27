# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MiniMaxText01RMSNormTP.forward_qk."""

import pytest
import torch
import torch.nn as nn
from torch.multiprocessing import spawn

from tests.utils import ensure_current_vllm_config, init_test_distributed_environment
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.model_executor.layers.mamba.linear_attn import (
    MiniMaxText01RMSNormTP,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import cuda_device_count_stateless, set_random_seed


@ensure_current_vllm_config()
def _worker_forward_qk(
    local_rank,
    world_size,
    port,
    num_tokens,
    hidden_q_full,
    hidden_k_full,
    dtype,
    seed,
    eps,
):
    """Per-rank worker that exercises both paths of forward_qk."""

    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(
        world_size, 1, local_rank, port, local_rank=local_rank
    )

    hq = hidden_q_full // world_size
    hk = hidden_k_full // world_size

    q_norm = MiniMaxText01RMSNormTP(hidden_q_full, eps=eps).cuda()
    k_norm = MiniMaxText01RMSNormTP(hidden_k_full, eps=eps).cuda()
    assert q_norm._ar_workspace is not None, (
        f"workspace not initialised (tp={world_size})"
    )

    set_random_seed(seed)
    qw = torch.randn(hidden_q_full, dtype=dtype, device="cuda")
    kw = torch.randn(hidden_k_full, dtype=dtype, device="cuda")
    q_norm.weight = nn.Parameter(qw[local_rank * hq : (local_rank + 1) * hq])
    k_norm.weight = nn.Parameter(kw[local_rank * hk : (local_rank + 1) * hk])

    torch.manual_seed(seed + 1000 + local_rank)
    qkv = torch.randn(num_tokens, hq + hk + hk, dtype=dtype, device="cuda")

    # Reference: NCCL fallback (temporarily clear workspace)
    saved_ws = q_norm._ar_workspace
    q_norm._ar_workspace = None
    ref_q, ref_k, ref_v = MiniMaxText01RMSNormTP.forward_qk(
        q_norm, k_norm, qkv.clone(), hq, hk
    )
    q_norm._ar_workspace = saved_ws

    fused_q, fused_k, fused_v = MiniMaxText01RMSNormTP.forward_qk(
        q_norm, k_norm, qkv.clone(), hq, hk
    )
    torch.accelerator.synchronize()

    # atol = 5e-2 if dtype == torch.float16 else 1e-2
    # rtol = 5e-2 if dtype == torch.float16 else 1e-2
    torch.testing.assert_close(
        fused_q,
        ref_q,
        atol=3e-2,
        rtol=3e-2,
        msg=f"Q mismatch rank={local_rank} tp={world_size} dtype={dtype}",
    )
    torch.testing.assert_close(
        fused_k,
        ref_k,
        atol=3e-2,
        rtol=3e-2,
        msg=f"K mismatch rank={local_rank} tp={world_size} dtype={dtype}",
    )
    torch.testing.assert_close(
        fused_v,
        ref_v,
        atol=0,
        rtol=0,
        msg=f"V should be unchanged rank={local_rank} tp={world_size} dtype={dtype}",
    )

    cleanup_dist_env_and_memory()


@ensure_current_vllm_config()
def _worker_forward_qk_rope(
    local_rank,
    world_size,
    port,
    num_tokens,
    head_size,
    num_heads_q,
    num_heads_k,
    dtype,
    is_neox,
    seed,
    eps,
):
    """Per-rank worker: tests forward_qk with fused RoPE against reference."""

    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(
        world_size, 1, local_rank, port, local_rank=local_rank
    )

    # Per-rank hidden dims
    hq = head_size * num_heads_q // world_size
    hk = head_size * num_heads_k // world_size
    hidden_q_full = head_size * num_heads_q
    hidden_k_full = head_size * num_heads_k

    rotary_emb = get_rope(
        head_size,
        max_position=8192,
        is_neox_style=is_neox,
    ).to(device=device, dtype=dtype)

    q_norm = MiniMaxText01RMSNormTP(hidden_q_full, eps=eps).cuda()
    k_norm = MiniMaxText01RMSNormTP(hidden_k_full, eps=eps).cuda()
    assert q_norm._ar_workspace is not None, (
        f"workspace not initialised (tp={world_size})"
    )

    set_random_seed(seed)
    qw = torch.randn(hidden_q_full, dtype=dtype, device="cuda")
    kw = torch.randn(hidden_k_full, dtype=dtype, device="cuda")
    q_norm.weight = nn.Parameter(qw[local_rank * hq : (local_rank + 1) * hq])
    k_norm.weight = nn.Parameter(kw[local_rank * hk : (local_rank + 1) * hk])

    torch.manual_seed(seed + 1000 + local_rank)
    qkv = torch.randn(num_tokens, hq + hk + hk, dtype=dtype, device="cuda")
    positions = torch.randint(0, 8192, (num_tokens,), device="cuda")

    # Reference: NCCL fallback + separate RoPE
    saved_ws = q_norm._ar_workspace
    q_norm._ar_workspace = None
    ref_q, ref_k, ref_v = MiniMaxText01RMSNormTP.forward_qk(
        q_norm,
        k_norm,
        qkv.clone(),
        hq,
        hk,
        positions=positions,
        rotary_emb=rotary_emb,
    )
    q_norm._ar_workspace = saved_ws

    # Fused path: allreduce+RMS+RoPE in one kernel
    fused_q, fused_k, fused_v = MiniMaxText01RMSNormTP.forward_qk(
        q_norm,
        k_norm,
        qkv.clone(),
        hq,
        hk,
        positions=positions,
        rotary_emb=rotary_emb,
    )
    torch.accelerator.synchronize()

    torch.testing.assert_close(
        fused_q,
        ref_q,
        atol=3e-2,
        rtol=3e-2,
        msg=f"Q mismatch rank={local_rank} tp={world_size} neox={is_neox}",
    )
    torch.testing.assert_close(
        fused_k,
        ref_k,
        atol=3e-2,
        rtol=3e-2,
        msg=f"K mismatch rank={local_rank} tp={world_size} neox={is_neox}",
    )
    torch.testing.assert_close(
        fused_v,
        ref_v,
        atol=0,
        rtol=0,
        msg=f"V should be unchanged rank={local_rank} tp={world_size}",
    )

    cleanup_dist_env_and_memory()


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="CUDA required",
)
@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("num_tokens", [1, 128, 333])
@pytest.mark.parametrize(
    "hidden_dims",
    [(6144, 1024), (1024, 1024)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize("seed", [42])
def test_minimax_reduce_rms_qk(
    world_size,
    num_tokens,
    hidden_dims,
    dtype,
    eps,
    seed,
):
    num_gpus = cuda_device_count_stateless()
    if num_gpus < world_size:
        pytest.skip(f"Need >= {world_size} GPUs, have {num_gpus}")
    hidden_q_full, hidden_k_full = hidden_dims
    port = str(get_open_port())
    spawn(
        _worker_forward_qk,
        args=(
            world_size,
            port,
            num_tokens,
            hidden_q_full,
            hidden_k_full,
            dtype,
            seed,
            eps,
        ),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="CUDA required",
)
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("num_tokens", [1, 64, 333])
@pytest.mark.parametrize(
    "head_config",
    [
        # (head_size, num_heads_q, num_heads_k)
        (128, 48, 8),  # MiniMax-M2 typical: 6144q / 1024k
        (128, 8, 8),  # equal q/k heads
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize("seed", [42])
def test_minimax_reduce_rms_qk_rope(
    world_size,
    num_tokens,
    head_config,
    dtype,
    is_neox,
    eps,
    seed,
):
    num_gpus = cuda_device_count_stateless()
    if num_gpus < world_size:
        pytest.skip(f"Need >= {world_size} GPUs, have {num_gpus}")
    if not hasattr(torch.ops._C, "minimax_allreduce_rms_rope_fusion"):
        pytest.skip("minimax_allreduce_rms_rope_fusion op not compiled")
    head_size, num_heads_q, num_heads_k = head_config
    if (head_size * num_heads_q) % world_size != 0 or (
        head_size * num_heads_k
    ) % world_size != 0:
        pytest.skip("hidden dim not divisible by world_size")
    port = str(get_open_port())
    spawn(
        _worker_forward_qk_rope,
        args=(
            world_size,
            port,
            num_tokens,
            head_size,
            num_heads_q,
            num_heads_k,
            dtype,
            is_neox,
            seed,
            eps,
        ),
        nprocs=world_size,
        join=True,
    )
