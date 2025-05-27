import torch
from typing import Optional,Union,Callable
from torch.library import Library
from packaging import version
from packaging.version import Version


def supports_dynamo() -> bool:
    base_torch_version = Version(Version(torch.__version__).base_version)
    return base_torch_version >= Version("2.4.0")


# Some backends use pytorch version < 2.4.0 which doesn't
# support `torch.library.custom_op`.
def supports_custom_op() -> bool:
    return hasattr(torch.library, "custom_op")

# create a library to hold the custom op
vllm_lib = Library("vllm", "FRAGMENT")  # noqa

def direct_register_custom_op(
        op_name: str,
        op_func: Callable,
        mutates_args: list[str],
        fake_impl: Optional[Callable] = None,
        target_lib: Optional[Library] = None,
        dispatch_key: str = "CUDA",
        tags: tuple[torch.Tag, ...] = (),
):
    """
    `torch.library.custom_op` can have significant overhead because it
    needs to consider complicated dispatching logic. This function
    directly registers a custom op and dispatches it to the CUDA backend.
    See https://gist.github.com/youkaichao/ecbea9ec9fc79a45d2adce1784d7a9a5
    for more details.

    By default, the custom op is registered to the vLLM library. If you
    want to register it to a different library, you can pass the library
    object to the `target_lib` argument.

    IMPORTANT: the lifetime of the operator is tied to the lifetime of the
    library object. If you want to bind the operator to a different library,
    make sure the library object is alive when the operator is used.
    """
    if not supports_custom_op():
        from vllm.platforms import current_platform
        assert not current_platform.is_cuda_alike(), (
            "cuda platform needs torch>=2.4 to support custom op, "
            "chances are you are using an old version of pytorch "
            "or a custom build of pytorch. It is recommended to "
            "use vLLM in a fresh new environment and let it install "
            "the required dependencies.")
        return

    import torch.library
    if hasattr(torch.library, "infer_schema"):
        schema_str = torch.library.infer_schema(op_func,
                                                mutates_args=mutates_args)
    else:
        # for pytorch 2.4
        import torch._custom_op.impl
        schema_str = torch._custom_op.impl.infer_schema(op_func, mutates_args)
    my_lib = target_lib or vllm_lib
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)
        
def is_torch_equal_or_newer(target: str) -> bool:
    """Check if the installed torch version is >= the target version.

    Args:
        target: a version string, like "2.6.0".

    Returns:
        Whether the condition meets.
    """
    try:
        torch_version = version.parse(str(torch.__version__))
        return torch_version >= version.parse(target)
    except Exception:
        # Fallback to PKG-INFO to load the package info, needed by the doc gen.
        return Version(importlib.metadata.version('torch')) >= Version(target)