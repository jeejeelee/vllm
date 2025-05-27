# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum

import numpy as np
import torch

# This value is chosen to have a balance between ITL and TTFT. Note it is
# not optimized for throughput.
DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048
POOLING_MODEL_MAX_NUM_BATCHED_TOKENS = 32768
MULTIMODAL_MODEL_MAX_NUM_BATCHED_TOKENS = 5120

# Exception strings for non-implemented encoder/decoder scenarios

# Reminder: Please update docs/features/compatibility_matrix.md
# If the feature combo become valid

STR_NOT_IMPL_ENC_DEC_SWA = \
    "Sliding window attention for encoder/decoder models " + \
                    "is not currently supported."

STR_NOT_IMPL_ENC_DEC_PREFIX_CACHE = \
    "Prefix caching for encoder/decoder models " + \
                    "is not currently supported."

STR_NOT_IMPL_ENC_DEC_CHUNKED_PREFILL = \
    "Chunked prefill for encoder/decoder models " + \
                    "is not currently supported."

STR_NOT_IMPL_ENC_DEC_LOGIT_SOFTCAP = (
    "Models with logits_soft_cap "
    "require FlashInfer backend, which is "
    "currently not supported for encoder/decoder "
    "models.")

STR_NOT_IMPL_ENC_DEC_LORA = ("LoRA is currently not currently "
                             "supported with encoder/decoder "
                             "models.")

STR_NOT_IMPL_ENC_DEC_PP = ("Pipeline parallelism is not "
                           "currently supported with "
                           "encoder/decoder models.")

STR_NOT_IMPL_ENC_DEC_MM = ("Multimodal is not currently "
                           "supported with encoder/decoder "
                           "models.")

STR_NOT_IMPL_ENC_DEC_SPEC_DEC = ("Speculative decoding is not "
                                 "currently supported with encoder/"
                                 "decoder models.")

STR_NOT_IMPL_ENC_DEC_BACKEND = ("XFormers and Flash-Attention are the only "
                                "backends currently supported with encoder/"
                                "decoder models.")

STR_NOT_IMPL_ENC_DEC_PROMPT_ADAPTER = ("Prompt adapters are not "
                                       "currently supported with encoder/"
                                       "decoder models.")

# Efficiently import all enc/dec error strings
# rather than having to import all of the above
STR_NOT_IMPL_ENC_DEC_ERR_STRS = {
    "STR_NOT_IMPL_ENC_DEC_SWA": STR_NOT_IMPL_ENC_DEC_SWA,
    "STR_NOT_IMPL_ENC_DEC_PREFIX_CACHE": STR_NOT_IMPL_ENC_DEC_PREFIX_CACHE,
    "STR_NOT_IMPL_ENC_DEC_CHUNKED_PREFILL":
    STR_NOT_IMPL_ENC_DEC_CHUNKED_PREFILL,
    "STR_NOT_IMPL_ENC_DEC_LOGIT_SOFTCAP": STR_NOT_IMPL_ENC_DEC_LOGIT_SOFTCAP,
    "STR_NOT_IMPL_ENC_DEC_LORA": STR_NOT_IMPL_ENC_DEC_LORA,
    "STR_NOT_IMPL_ENC_DEC_PP": STR_NOT_IMPL_ENC_DEC_PP,
    "STR_NOT_IMPL_ENC_DEC_MM": STR_NOT_IMPL_ENC_DEC_MM,
    "STR_NOT_IMPL_ENC_DEC_SPEC_DEC": STR_NOT_IMPL_ENC_DEC_SPEC_DEC,
    "STR_NOT_IMPL_ENC_DEC_BACKEND": STR_NOT_IMPL_ENC_DEC_BACKEND,
    "STR_NOT_IMPL_ENC_DEC_PROMPT_ADAPTER": STR_NOT_IMPL_ENC_DEC_PROMPT_ADAPTER,
}

# Constants related to forcing the attention backend selection

# String name of register which may be set in order to
# force auto-selection of attention backend by Attention
# wrapper
STR_BACKEND_ENV_VAR: str = "VLLM_ATTENTION_BACKEND"

# Possible string values of STR_BACKEND_ENV_VAR
# register, corresponding to possible backends
STR_FLASHINFER_ATTN_VAL: str = "FLASHINFER"
STR_TORCH_SDPA_ATTN_VAL: str = "TORCH_SDPA"
STR_ROCM_FLASH_ATTN_VAL: str = "ROCM_FLASH"
STR_XFORMERS_ATTN_VAL: str = "XFORMERS"
STR_FLASH_ATTN_VAL: str = "FLASH_ATTN"
STR_DUAL_CHUNK_FLASH_ATTN_VAL: str = "DUAL_CHUNK_FLASH_ATTN"
STR_INVALID_VAL: str = "INVALID"

GB_bytes = 1_000_000_000
"""The number of bytes in one gigabyte (GB)."""

GiB_bytes = 1 << 30
"""The number of bytes in one gibibyte (GiB)."""

STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
    "int8": torch.int8,
}

TORCH_DTYPE_TO_NUMPY_DTYPE = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.uint8: np.uint8,
    torch.int32: np.int32,
    torch.int64: np.int64,
}


class _Sentinel:
    ...


ALL_PINNED_SENTINEL = _Sentinel()


class Device(enum.Enum):
    GPU = enum.auto()
    CPU = enum.auto()


class LayerBlockType(enum.Enum):
    attention = "attention"
    mamba = "mamba"
