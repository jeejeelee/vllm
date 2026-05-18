# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL FP8 block-wise grouped MoE experts (Blackwell SM100).

Wraps the standalone implementation in
`/home/jeejeelee/Code/vllm_dev/vllm/z_test/cute_moe_fp8/` that uses
the CUTLASS upstream `BlockwiseContiguousGroupedGemmKernel` (CuTe DSL,
tcgen05 MMA) for W13/W2 GEMM and a Triton SwiGLU+quant + scatter+reduce.

Selection: enabled when env `VLLM_USE_CUTE_DSL_MOE=1` on a Blackwell GPU.
"""
import os
import sys

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


_CUTE_PKG_DIR = "/home/jeejeelee/Code/vllm_dev/vllm/z_test"
if _CUTE_PKG_DIR not in sys.path:
    sys.path.insert(0, _CUTE_PKG_DIR)


def _import_cute_orchestrator():
    """Lazy import; only triggered when backend actually runs."""
    from cute_moe_fp8.fused_moe import fused_moe_fp8_block_cute
    return fused_moe_fp8_block_cute


class CuteDslFp8BlockExpertsBase:
    """Shared base for modular interface."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        self.moe_config = moe_config
        self.quant_config = quant_config
        self.topk = moe_config.experts_per_token
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.hidden_dim = moe_config.hidden_dim
        self.local_num_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        """Blackwell-family (SM100) only; requires env opt-in."""
        if os.environ.get("VLLM_USE_CUTE_DSL_MOE", "0") != "1":
            return False
        p = current_platform
        return p.is_cuda() and p.is_device_capability_family(100)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        # Conservative: same constraint as TrtLlm modular path.
        return (
            not moe_parallel_config.use_all2all_kernels
            or moe_parallel_config.use_ag_rs_all2all_kernels
        ) and not moe_parallel_config.enable_eplb

    def supports_chunking(self) -> bool:
        return False

    def supports_expert_map(self) -> bool:
        return False


class CuteDslFp8BlockExpertsModular(
    CuteDslFp8BlockExpertsBase, mk.FusedMoEExpertsModular
):
    """Modular interface: external topk routing; we do W13 + SwiGLU+quant + W2 + reduce."""

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (
            kFp8Static128BlockSym,
            kFp8Dynamic128Sym,
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # Internal allocations handled by our kernel; no shared workspaces.
        return ((0,), (0,), (M, K))

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Our kernel applies topk_weights and reduces internally.
        return TopKWeightAndReduceNoOP()

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert a1q_scale is not None, "block FP8 path needs activation scales"
        assert not apply_router_weight_on_input, (
            "CuteDsl backend handles router weight internally; "
            "apply_router_weight_on_input=True is not supported"
        )

        fused_moe_fp8_block_cute = _import_cute_orchestrator()

        topk_ids_i32 = (
            topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
        )

        # vLLM keeps weights in plain [E, N, K]; scales [E, N//128, K//128] fp32.
        # MiniMax-M2 uses gate-first concat (is_act_and_mul=True), so swap_gate_up=False.
        result = fused_moe_fp8_block_cute(
            a_fp8=hidden_states,
            a_scale=a1q_scale,
            w13=w1,
            w13_scale=self.quant_config.w1_scale,
            w2=w2,
            w2_scale=self.quant_config.w2_scale,
            topk_ids=topk_ids_i32,
            topk_weights=topk_weights,
            swap_gate_up=False,
        )
        output.copy_(result)
