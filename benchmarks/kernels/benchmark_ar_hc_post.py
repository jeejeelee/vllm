# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark fused AllReduce + hc_post (DeepseekV4) vs unfused baseline.

Baseline path:
    x_ar = tensor_model_parallel_all_reduce(x_local)
    out  = torch.ops.vllm.mhc_post(x_ar, residual, post, comb)

Fused path:
    torch.ops._C.trtllm_ar_hc_post(
        x_local, residual, post, comb, out, workspace,
        rank, nranks, launch_with_pdl=True)

Usage:
    torchrun --nproc_per_node=2 benchmark_ar_hc_post.py
    torchrun --nproc_per_node=4 benchmark_ar_hc_post.py \
        --hidden-size 7168 --hc-mult 4 \
        --tokens 1,16,64,256,1024
"""

import argparse
import os

import torch
import torch.distributed as dist

from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)


def _setup_distributed() -> tuple[int, int, torch.device]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method=f"tcp://{os.environ['MASTER_ADDR']}:"
        f"{os.environ['MASTER_PORT']}",
        local_rank=local_rank,
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    return rank, world_size, device


def _bench(
    fn,
    *,
    warmup: int = 10,
    iters: int = 50,
) -> float:
    """Return median per-call time in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_evts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_evts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_evts[i].record()
        fn()
        end_evts[i].record()
    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(start_evts, end_evts)]
    times_ms.sort()
    median = times_ms[iters // 2]
    return median * 1000.0  # ms -> us


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden-size", type=int, default=7168)
    p.add_argument("--hc-mult", type=int, default=4)
    p.add_argument(
        "--tokens",
        type=str,
        default="1,8,32,128,512,1024,2048",
    )
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)

    # Required so torch.ops.vllm.mhc_post is registered.
    import vllm.model_executor.layers.mhc  # noqa: F401

    cfg = VllmConfig(compilation_config=CompilationConfig())
    with set_current_vllm_config(cfg):
        rank, world_size, device = _setup_distributed()
        # Trigger registration of `torch.ops.vllm.fused_ar_hc_post`
        # (direct_register_custom_op runs at import time).
        import vllm.compilation.passes.fusion.allreduce_rms_fusion  # noqa: F401
        from vllm.distributed.device_communicators.flashinfer_all_reduce import (
            get_fi_ar_workspace,
        )
        from vllm.distributed.parallel_state import get_tp_group

        token_list = [int(t) for t in args.tokens.split(",")]
        max_tokens = max(token_list)

        workspace = get_fi_ar_workspace(
            world_size=world_size,
            rank=rank,
            max_token_num=max_tokens,
            hidden_dim=args.hidden_size,
            dtype=dtype,
            group=get_tp_group().device_group,
        )
        if workspace is None:
            if rank == 0:
                print("[skip] FlashInfer workspace unavailable")
            return

        # Inline threshold lookup (mirrors `_FI_ALLREDUCE_ONE_SHOT_MAX_SIZES_MB`
        # in vllm/compilation/passes/fusion/allreduce_rms_fusion.py).
        _ONESHOT_TABLE = {
            90: {2: 32, 4: 2, 8: 0.5},
            100: {2: 32, 4: 4, 8: 1},
            103: {2: 32, 4: 4, 8: 2},
        }
        cap_major, cap_minor = torch.cuda.get_device_capability()
        _DEVICE_CAP = cap_major * 10 + cap_minor

        def trtllm_ar_hc_post_use_oneshot(num_tokens, hidden_dim, dtype, world_size):
            max_mb = _ONESHOT_TABLE.get(_DEVICE_CAP, {}).get(world_size)
            if max_mb is None:
                return True
            elem = torch.tensor([], dtype=dtype).element_size()
            return num_tokens * hidden_dim * elem <= max_mb * 1024 * 1024

        if rank == 0:
            print(
                f"world_size={world_size} hidden_size={args.hidden_size} "
                f"hc_mult={args.hc_mult} dtype={args.dtype}"
            )
            print(
                f"{'tokens':>8} {'baseline_us':>12} {'1shot_us':>10} "
                f"{'2shot_us':>10} {'wrapper_us':>11} "
                f"{'auto':>5} {'speedup':>8}"
            )

        torch.manual_seed(0xC0FFEE + rank)

        for n in token_list:
            x_local = (
                torch.randn(n, args.hidden_size, dtype=dtype, device=device) * 0.05
            )
            residual = (
                torch.randn(
                    n,
                    args.hc_mult,
                    args.hidden_size,
                    dtype=dtype,
                    device=device,
                )
                * 0.05
            )
            post = torch.randn(n, args.hc_mult, 1, dtype=torch.float32, device=device)
            comb = torch.randn(
                n,
                args.hc_mult,
                args.hc_mult,
                dtype=torch.float32,
                device=device,
            )
            out_buf = torch.empty(
                n, args.hc_mult, args.hidden_size, dtype=dtype, device=device
            )

            auto_oneshot = trtllm_ar_hc_post_use_oneshot(
                num_tokens=n,
                hidden_dim=args.hidden_size,
                dtype=dtype,
                world_size=world_size,
            )

            # ---- baseline: NCCL AR + tilelang mhc_post -------------------
            def baseline():
                x_ar = tensor_model_parallel_all_reduce(x_local)
                _ = torch.ops.vllm.mhc_post(x_ar, residual, post, comb)

            def fused(use_oneshot: bool):
                def _f():
                    torch.ops._C.trtllm_ar_hc_post(
                        x_local,
                        residual,
                        post,
                        comb,
                        out_buf,
                        workspace.workspace_tensor,
                        rank,
                        world_size,
                        True,
                        use_oneshot,
                    )

                return _f

            # ---- wrapper: high-level torch.ops.vllm.fused_ar_hc_post ------
            # Includes assert + workspace lookup + auto oneshot/twoshot
            # decision overhead vs the direct kernel call above.
            def wrapper_call():
                torch.ops.vllm.fused_ar_hc_post(
                    x_local,
                    residual,
                    post,
                    comb,
                    world_size,
                    True,  # launch_with_pdl
                    max_tokens,  # max_token_num
                )

            base_us = _bench(baseline, warmup=args.warmup, iters=args.iters)
            one_us = _bench(fused(True), warmup=args.warmup, iters=args.iters)
            two_us = _bench(fused(False), warmup=args.warmup, iters=args.iters)
            wrap_us = _bench(wrapper_call, warmup=args.warmup, iters=args.iters)

            if rank == 0:
                pick_us = one_us if auto_oneshot else two_us
                speedup = base_us / pick_us if pick_us > 0 else float("nan")
                tag = "1shot" if auto_oneshot else "2shot"
                print(
                    f"{n:>8} {base_us:>12.2f} {one_us:>10.2f} "
                    f"{two_us:>10.2f} {wrap_us:>11.2f} "
                    f"{tag:>5} {speedup:>8.2f}x"
                )

    dist.barrier()


if __name__ == "__main__":
    main()
