# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import concurrent
import contextlib
import datetime
import getpass
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import os

import sys

import threading
import traceback
import types
import uuid
import weakref
from asyncio import FIRST_COMPLETED, AbstractEventLoop, Task
from collections import UserDict, defaultdict
from collections.abc import (AsyncGenerator, Awaitable, Hashable, Iterable,
                             Mapping)
from functools import cache, partial, wraps
from typing import (TYPE_CHECKING, Any, Callable, Generic, Literal, Optional,
                    TypeVar, Union)

import cloudpickle
import numpy as np
import numpy.typing as npt
import psutil
import torch
import torch.types
import zmq
import zmq.asyncio


from typing_extensions import ParamSpec, TypeIs, assert_never


from vllm.logger import enable_trace_function_call, init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

from .constants import STR_DTYPE_TO_TORCH_DTYPE, TORCH_DTYPE_TO_NUMPY_DTYPE

logger = init_logger(__name__)
P = ParamSpec('P')
T = TypeVar("T")
U = TypeVar("U")

_K = TypeVar("_K", bound=Hashable)
_V = TypeVar("_V")
_T = TypeVar("_T")


class Counter:

    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0


@cache
def get_max_shared_memory_bytes(gpu: int = 0) -> int:
    """Returns the maximum shared memory per thread block in bytes."""
    from vllm import _custom_ops as ops
    max_shared_mem = (
        ops.get_max_shared_memory_per_block_device_attribute(gpu))
    # value 0 will cause MAX_SEQ_LEN become negative and test_attention.py
    # will fail
    assert max_shared_mem > 0, "max_shared_mem can not be zero"
    return int(max_shared_mem)


def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total



def chunk_list(lst: list[T], chunk_size: int):
    """Yield successive chunk_size chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def get_dtype_size(dtype: torch.dtype) -> int:
    """Get the size of the data type in bytes."""
    return torch.tensor([], dtype=dtype).element_size()


# `collections` helpers
def is_list_of(
    value: object,
    typ: Union[type[T], tuple[type[T], ...]],
    *,
    check: Literal["first", "all"] = "first",
) -> TypeIs[list[T]]:
    if not isinstance(value, list):
        return False

    if check == "first":
        return len(value) == 0 or isinstance(value[0], typ)
    elif check == "all":
        return all(isinstance(v, typ) for v in value)

    assert_never(check)


def flatten_2d_lists(lists: Iterable[Iterable[T]]) -> list[T]:
    """Flatten a list of lists to a single list."""
    return [item for sublist in lists for item in sublist]


def full_groupby(values: Iterable[_V], *, key: Callable[[_V], _K]):
    """
    Unlike {class}`itertools.groupby`, groups are not broken by
    non-contiguous data.
    """
    groups = defaultdict[_K, list[_V]](list)

    for value in values:
        groups[key(value)].append(value)

    return groups.items()


# TODO: This function can be removed if transformer_modules classes are
# serialized by value when communicating between processes
def init_cached_hf_modules() -> None:
    """
    Lazy initialization of the Hugging Face modules.
    """
    from transformers.dynamic_module_utils import init_hf_modules
    init_hf_modules()



# Using dynamo with vLLM doesn't really work well with PyTorch versions < 2.4.0.
# In particular, the FakeScalarType is not supported for earlier versions of
# PyTorch which breaks dynamo for any ops registered using ScalarType.


class AtomicCounter:
    """An atomic, thread-safe counter"""

    def __init__(self, initial=0):
        """Initialize a new atomic counter to given initial value"""
        self._value = initial
        self._lock = threading.Lock()

    def inc(self, num=1):
        """Atomically increment the counter by num and return the new value"""
        with self._lock:
            self._value += num
            return self._value

    def dec(self, num=1):
        """Atomically decrement the counter by num and return the new value"""
        with self._lock:
            self._value -= num
            return self._value

    @property
    def value(self):
        return self._value


# Adapted from: https://stackoverflow.com/a/47212782/5082708
class LazyDict(Mapping[str, T], Generic[T]):

    def __init__(self, factory: dict[str, Callable[[], T]]):
        self._factory = factory
        self._dict: dict[str, T] = {}

    def __getitem__(self, key: str) -> T:
        if key not in self._dict:
            if key not in self._factory:
                raise KeyError(key)
            self._dict[key] = self._factory[key]()
        return self._dict[key]

    def __setitem__(self, key: str, value: Callable[[], T]):
        self._factory[key] = value

    def __iter__(self):
        return iter(self._factory)

    def __len__(self):
        return len(self._factory)


class ClassRegistry(UserDict[type[T], _V]):

    def __getitem__(self, key: type[T]) -> _V:
        for cls in key.mro():
            if cls in self.data:
                return self.data[cls]

        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return self.contains(key)

    def contains(self, key: object, *, strict: bool = False) -> bool:
        if not isinstance(key, type):
            return False

        if strict:
            return key in self.data

        return any(cls in self.data for cls in key.mro())



def is_in_doc_build() -> bool:
    try:
        from sphinx.ext.autodoc.mock import _MockModule
        return isinstance(zmq, _MockModule)
    except ModuleNotFoundError:
        return False



# Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/srt/utils.py#L630 # noqa: E501
def set_ulimit(target_soft_limit=65535):
    if sys.platform.startswith('win'):
        logger.info("Windows detected, skipping ulimit adjustment.")
        return

    import resource
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type,
                               (target_soft_limit, current_hard))
        except ValueError as e:
            logger.warning(
                "Found ulimit of %s and failed to automatically increase "
                "with error %s. This can cause fd limit errors like "
                "`OSError: [Errno 24] Too many open files`. Consider "
                "increasing with ulimit -n", current_soft, e)

# Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/utils.py#L28 # noqa: E501
def get_exception_traceback():
    etype, value, tb = sys.exc_info()
    err_str = "".join(traceback.format_exception(etype, value, tb))
    return err_str


def resolve_obj_by_qualname(qualname: str) -> Any:
    """
    Resolve an object by its fully qualified name.
    """
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)

def swap_dict_values(obj: dict[_K, _V], key1: _K, key2: _K) -> None:
    """
    Helper function to swap values for two keys
    """
    v1 = obj.get(key1)
    v2 = obj.get(key2)
    if v1 is not None:
        obj[key2] = v1
    else:
        obj.pop(key2, None)
    if v2 is not None:
        obj[key1] = v2
    else:
        obj.pop(key1, None)

