
import torch
from typing import Optional,Union,Any,TypeVar
from vllm.utils.constants import STR_DTYPE_TO_TORCH_DTYPE

T = TypeVar("T")


def get_kv_cache_torch_dtype(
        cache_dtype: Optional[Union[str, torch.dtype]],
        model_dtype: Optional[Union[str, torch.dtype]] = None) -> torch.dtype:
    if isinstance(cache_dtype, str):
        if cache_dtype == "auto":
            if isinstance(model_dtype, str):
                torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            elif isinstance(model_dtype, torch.dtype):
                torch_dtype = model_dtype
            else:
                raise ValueError(f"Invalid model dtype: {model_dtype}")
        elif cache_dtype in ["half", "bfloat16", "float"]:
            torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        elif cache_dtype == "fp8":
            torch_dtype = torch.uint8
        else:
            raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    elif isinstance(cache_dtype, torch.dtype):
        torch_dtype = cache_dtype
    else:
        raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    return torch_dtype


def _generate_random_fp8(
    tensor: torch.Tensor,
    low: float,
    high: float,
) -> None:
    # NOTE(zhaoyang): Due to NaN and Inf representation for fp8 data type,
    # it may occur Inf or NaN if we directly use torch.randint
    # to generate random data for fp8 data.
    # For example, s.11111.00 in fp8e5m2 format represents Inf.
    #     | E4M3        | E5M2
    #-----|-------------|-------------------
    # Inf | N/A         | s.11111.00
    # NaN | s.1111.111  | s.11111.{01,10,11}
    from vllm import _custom_ops as ops
    tensor_tmp = torch.empty_like(tensor, dtype=torch.float16)
    tensor_tmp.uniform_(low, high)
    ops.convert_fp8(tensor, tensor_tmp)
    del tensor_tmp

def create_kv_caches_with_random_flash(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
    seed: Optional[int] = None,
    device: Optional[str] = "cuda",
    cache_layout: Optional[str] = "NHD",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    from vllm.platforms import current_platform
    current_platform.seed_everything(seed)

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)
    generic_kv_cache_shape = (num_blocks, 2, block_size, num_heads, head_size)
    assert cache_layout in ("NHD", "HND")
    stride_order = (0, 1, 2, 3, 4) if cache_layout == "NHD" else (0, 1, 3, 2,
                                                                  4)

    kv_cache_allocation_shape = tuple(generic_kv_cache_shape[i]
                                      for i in stride_order)
    scale = head_size**-0.5

    key_caches: list[torch.Tensor] = []
    value_caches: list[torch.Tensor] = []

    for _ in range(num_layers):
        key_value_cache = torch.empty(size=kv_cache_allocation_shape,
                                      dtype=torch_dtype,
                                      device=device).permute(*stride_order)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_value_cache.uniform_(-scale, scale)
        elif cache_dtype == 'fp8':
            _generate_random_fp8(key_value_cache, -scale, scale)
        else:
            raise ValueError(
                f"Does not support key cache of type {cache_dtype}")
        key_caches.append(key_value_cache[:, 0])
        value_caches.append(key_value_cache[:, 1])
    return key_caches, value_caches


def create_kv_caches_with_random(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
    seed: Optional[int] = None,
    device: Optional[str] = "cuda",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:

    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            f"Does not support key cache of type fp8 with head_size {head_size}"
        )
    from vllm.platforms import current_platform
    current_platform.seed_everything(seed)

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=torch_dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches: list[torch.Tensor] = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape,
                                dtype=torch_dtype,
                                device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_cache.uniform_(-scale, scale)
        elif cache_dtype == 'fp8':
            _generate_random_fp8(key_cache, -scale, scale)
        else:
            raise ValueError(
                f"Does not support key cache of type {cache_dtype}")
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches: list[torch.Tensor] = []
    for _ in range(num_layers):
        value_cache = torch.empty(size=value_cache_shape,
                                  dtype=torch_dtype,
                                  device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            value_cache.uniform_(-scale, scale)
        elif cache_dtype == 'fp8':
            _generate_random_fp8(value_cache, -scale, scale)
        else:
            raise ValueError(
                f"Does not support value cache of type {cache_dtype}")
        value_caches.append(value_cache)
    return key_caches, value_caches



def bind_kv_cache(
        ctx: dict[str, Any],
        kv_cache: list[list[torch.Tensor]],  # [virtual_engine][layer_index]
) -> None:
    # Bind the kv_cache tensor to Attention modules, similar to
    # ctx[layer_name].kv_cache[ve]=kv_cache[ve][extract_layer_index(layer_name)]
    # Special things handled here:
    # 1. Some models have non-attention layers, e.g., Jamba
    # 2. Pipeline parallelism, each rank only has a subset of layers
    # 3. Encoder attention has no kv cache
    # 4. Encoder-decoder models, encoder-decoder attention and decoder-only
    #    attention of the same layer (e.g., bart's decoder.layers.1.self_attn
    #    and decoder.layers.1.encoder_attn) is mapped to the same kv cache
    #    tensor
    from vllm.attention import AttentionType
    from vllm.model_executor.models.utils import extract_layer_index
    layer_need_kv_cache = [
        layer_name for layer_name in ctx
        if (hasattr(ctx[layer_name], 'attn_type') and ctx[layer_name].attn_type
            in (AttentionType.DECODER, AttentionType.ENCODER_DECODER))
    ]
    layer_index_sorted = sorted(
        set(
            extract_layer_index(layer_name)
            for layer_name in layer_need_kv_cache))
    for layer_name in layer_need_kv_cache:
        kv_cache_idx = layer_index_sorted.index(
            extract_layer_index(layer_name))
        forward_ctx = ctx[layer_name]
        assert len(forward_ctx.kv_cache) == len(kv_cache)
        for ve, ve_kv_cache in enumerate(kv_cache):
            forward_ctx.kv_cache[ve] = ve_kv_cache[kv_cache_idx]