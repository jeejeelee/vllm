# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CSA summaries attached to the attention KV pages."""

from typing import Any

import torch

from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend

from .csa_tail_ring import Step5CSATailRing


class Step5SparseSummaryCache:
    """FP8 summaries of each page's closed eight-token regions.

    A region is summarized once, by its eighth row, from the rows waiting in
    ``Step5CSATailRing``; this cache holds no per-request state.
    """

    def __init__(self, pages: int, block_size: int, device: torch.device):
        if block_size % 8:
            raise ValueError("Step5 KV blocks must align to eight-token regions")
        self.block_size = block_size
        self.mean = torch.zeros(
            (pages, block_size // 8, 256), dtype=torch.uint8, device=device
        )

    def update_decode(self, key, z, slots, positions, ring: Step5CSATailRing):
        from .ops.summary_update import update_decode

        n = slots.shape[0]
        update_decode(
            key,
            z,
            slots,
            positions,
            ring.metadata().base[:n],
            self.mean,
            ring.rows,
            ring.capacity,
        )

    def update_verify(
        self, key, z, slots, query_start_loc, seq_lens, ring: Step5CSATailRing
    ):
        from .ops.summary_update import update_verify

        update_verify(
            key,
            z,
            slots,
            query_start_loc,
            seq_lens,
            ring.metadata().base[: slots.shape[0]],
            self.mean,
            ring.rows,
            ring.capacity,
        )

    def update_prefill(self, key, z, metadata, positions, ring: Step5CSATailRing):
        from .ops.summary_update import update_prefill

        n = metadata.shape[0]
        ring_metadata = ring.metadata()
        update_prefill(
            key,
            z,
            metadata,
            positions,
            ring_metadata.base[:n],
            ring_metadata.slot_mapping[:n],
            self.mean,
            ring.rows,
            ring.capacity,
        )

    def copy(self, copies: list[Any]) -> None:
        copies = [c for c in copies if c.src_block_id != c.dst_block_id]
        if copies:
            sources = [c.src_block_id for c in copies]
            self.mean[[c.dst_block_id for c in copies]] = self.mean[sources].clone()


class Step5SparseSummaryCacheLayer(torch.nn.Module, AttentionLayerBase):
    """Integrate summary allocation and COW with the vLLM KV lifecycle."""

    def __init__(
        self,
        *,
        prefix,
        target_impl,
        main_layer_name,
        static_forward_context,
    ):
        super().__init__()
        if prefix in static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        static_forward_context[prefix] = self
        self.layer_name = prefix
        self._target_impl = target_impl
        self.main_layer_name = main_layer_name
        self.kv_cache = torch.tensor([])
        self._cache: Step5SparseSummaryCache | None = None

    def get_attn_backend(self) -> type[AttentionBackend]:
        from .sparse_attention import Step5DSAAttentionBackend

        return Step5DSAAttentionBackend

    def get_kv_cache_spec(self, vllm_config):
        return None

    @staticmethod
    def page_budget(block_size: int) -> int:
        if block_size % 8:
            raise ValueError("Step5 KV blocks must align to eight-token regions")
        return block_size // 8 * 256

    def _peers(self, context):
        return [v for v in context.values() if isinstance(v, type(self))]

    def kv_cache_side_storage_memory_profile_key(self, forward_context):
        return self._peers(forward_context)[0]

    def prepare_kv_cache_side_storage_memory_profile(self, forward_context, device):
        from .flashinfer_sparse import Step5FlashInferSparse

        peers = self._peers(forward_context)
        adapter = peers[0]._target_impl.adapter
        if adapter is None:
            adapter = Step5FlashInferSparse(device)
        for peer in peers:
            peer._target_impl.adapter = adapter

    def bind_kv_cache_side_storage(self, forward_context):
        self.prepare_kv_cache_side_storage_memory_profile(
            forward_context, forward_context[self.main_layer_name].kv_cache.device
        )
        kv = forward_context[self.main_layer_name].kv_cache
        self._cache = Step5SparseSummaryCache(kv.shape[0], kv.shape[2], kv.device)
        self.kv_cache = self._cache.mean.view(kv.shape[0], -1)
        self._target_impl.summary_cache = self._cache

    def kv_cache_side_storage_transfer_view(self) -> torch.Tensor:
        # Full blocks only hold closed regions and open-region rows live in the
        # per-request tail ring, so the FP8 summaries are the whole page state.
        assert self._cache is not None
        return self._cache.mean.flatten(1)

    def zero_kv_cache_side_storage(self, block_ids):
        if self._cache is not None:
            self._cache.mean[block_ids] = 0

    def reset_kv_cache_side_storage_runtime_state(self):
        if self._cache is not None:
            self._cache.mean.zero_()
        return 1, 0

    def copy_kv_cache_side_storage(self, block_copies, num_blocks):
        if self._cache is not None and block_copies is not None:
            self._cache.copy(list(block_copies))
