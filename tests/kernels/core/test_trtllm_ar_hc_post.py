# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for fused AllReduce + hc_post kernel.

Reference path: NCCL all-reduce on the local x slice, then the existing
torch.ops.vllm.mhc_post (tilelang) applied to the AR-summed result.

Fused path: torch.ops._C.trtllm_ar_hc_post, which performs both in one
launch using the FlashInfer one-shot Lamport scheme.
"""

import pytest
import torch
from torch.multiprocessing import spawn

from tests.utils import ensure_current_vllm_config, init_test_distributed_environment
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    tensor_model_parallel_all_reduce,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_random_seed


def _hc_post_torch_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation matching mhc.py::mhc_post_tilelang.

    Shapes:
      x        : [N, H]            bf16
      residual : [N, hc, H]        bf16
      post     : [N, hc, 1]        f32
      comb     : [N, hc, hc]       f32
    Returns:
      out      : [N, hc, H]        bf16
    """
    if post.dim() == 3 and post.shape[-1] == 1:
        post = post.squeeze(-1)
    return (
        post.unsqueeze(-1) * x.unsqueeze(1)
        + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
    ).type_as(x)


@ensure_current_vllm_config()
def _worker(
    local_rank,
    world_size,
    port,
    num_tokens,
    hidden_size,
    hc_mult,
    dtype,
    seed,
    use_oneshot,
):
    if not hasattr(torch.ops._C, "trtllm_ar_hc_post"):
        cleanup_dist_env_and_memory()
        return

    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(
        world_size, 1, local_rank, port, local_rank=local_rank
    )

    # Acquire the FlashInfer one-shot Lamport workspace (same allocator the
    # AR + RMSNorm fusion pass uses).
    from vllm.distributed.device_communicators.flashinfer_all_reduce import (
        get_fi_ar_workspace,
    )
    from vllm.distributed.parallel_state import get_tp_group

    workspace = get_fi_ar_workspace(
        world_size=world_size,
        rank=local_rank,
        max_token_num=max(num_tokens, 16),
        hidden_dim=hidden_size,
        dtype=dtype,
        group=get_tp_group().device_group,
    )
    if workspace is None:
        pytest.skip("FlashInfer workspace unavailable on this platform")

    set_random_seed(seed + local_rank)

    # x_local mimics the per-rank wo_b output that would feed an AR.
    x_local = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device) * 0.05
    residual = (
        torch.randn(num_tokens, hc_mult, hidden_size, dtype=dtype, device=device) * 0.05
    )
    post = torch.randn(num_tokens, hc_mult, 1, dtype=torch.float32, device=device)
    comb = torch.randn(num_tokens, hc_mult, hc_mult, dtype=torch.float32, device=device)

    # ---------- reference path: NCCL AR + torch hc_post ----------
    x_ar_ref = tensor_model_parallel_all_reduce(x_local.clone())
    out_ref = _hc_post_torch_ref(x_ar_ref, residual, post, comb)

    # ---------- fused path ----------
    out_fused = torch.empty(
        num_tokens, hc_mult, hidden_size, dtype=dtype, device=device
    )
    # Make sure all peers reach the fused kernel together.
    torch.distributed.barrier(group=get_tp_group().device_group)

    torch.ops._C.trtllm_ar_hc_post(
        x_local,
        residual,
        post,
        comb,
        out_fused,
        workspace.workspace_tensor,
        local_rank,
        world_size,
        True,  # launch_with_pdl
        use_oneshot,
    )
    torch.cuda.synchronize()

    # bf16 ULP-level tolerance. At the pytest matrix's max N=128 the empirical
    # max_abs is ~5e-3; 1e-2 leaves ~2x margin. Larger N (>1k) sees up to 1.2e-2
    # which would need 1.5e-2 — but those sizes aren't part of this matrix.
    torch.testing.assert_close(out_fused, out_ref, atol=1e-2, rtol=1e-2)

    cleanup_dist_env_and_memory()


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="CUDA required",
)
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("num_tokens", [1, 32, 128])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("hc_mult", [4])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("use_oneshot", [True, False])
@pytest.mark.parametrize("seed", [123])
def test_trtllm_ar_hc_post(
    world_size, num_tokens, hidden_size, hc_mult, dtype, use_oneshot, seed
):
    num_gpus = current_platform.device_count()
    if num_gpus < world_size:
        pytest.skip(f"Need >= {world_size} GPUs, have {num_gpus}")
    port = str(get_open_port())
    spawn(
        _worker,
        args=(
            world_size,
            port,
            num_tokens,
            hidden_size,
            hc_mult,
            dtype,
            seed,
            use_oneshot,
        ),
        nprocs=world_size,
        join=True,
    )
