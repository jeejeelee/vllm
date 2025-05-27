# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import gc
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from functools import wraps
from typing import Callable, Optional

import torch


@dataclass
class MemorySnapshot:
    """Memory snapshot."""
    torch_peak: int = 0
    cuda_memory: int = 0
    torch_memory: int = 0
    non_torch_memory: int = 0
    timestamp: float = 0.0
    auto_measure: bool = True

    def __post_init__(self):
        if self.auto_measure:
            self.measure()

    def measure(self):
        # we measure the torch peak memory usage via allocated_bytes,
        # rather than `torch.cuda.memory_reserved()` .
        # After `torch.cuda.reset_peak_memory_stats()`,
        # `torch.cuda.memory_reserved()` will keep growing, and only shrink
        # when we call `torch.cuda.empty_cache()` or OOM happens.
        self.torch_peak = torch.cuda.memory_stats().get(
            "allocated_bytes.all.peak", 0)

        self.cuda_memory = torch.cuda.mem_get_info(
        )[1] - torch.cuda.mem_get_info()[0]

        # torch.cuda.memory_reserved() is how many bytes
        # PyTorch gets from cuda (by calling cudaMalloc, etc.)
        # this is used to measure the non-torch memory usage
        self.torch_memory = torch.cuda.memory_reserved()

        self.non_torch_memory = self.cuda_memory - self.torch_memory
        self.timestamp = time.time()

    def __sub__(self, other: MemorySnapshot) -> MemorySnapshot:
        return MemorySnapshot(
            torch_peak=self.torch_peak - other.torch_peak,
            cuda_memory=self.cuda_memory - other.cuda_memory,
            torch_memory=self.torch_memory - other.torch_memory,
            non_torch_memory=self.non_torch_memory - other.non_torch_memory,
            timestamp=self.timestamp - other.timestamp,
            auto_measure=False,
        )


@dataclass
class MemoryProfilingResult:
    """Memory profiling result. All numbers are in bytes.
    """
    non_kv_cache_memory: int = 0
    torch_peak_increase: int = 0
    non_torch_increase: int = 0
    weights_memory: float = 0
    before_create: MemorySnapshot = field(default_factory=MemorySnapshot)
    before_profile: MemorySnapshot = field(default_factory=MemorySnapshot)
    after_profile: MemorySnapshot = field(default_factory=MemorySnapshot)
    profile_time: float = 0.0


@contextlib.contextmanager
def memory_profiling(
        baseline_snapshot: MemorySnapshot,
        weights_memory: int) -> Generator[MemoryProfilingResult, None, None]:
    """Memory profiling context manager.
    baseline_snapshot: the memory snapshot before the current vLLM instance.
    weights_memory: memory used by PyTorch when loading the model weights.
        Note that, before loading the model weights, we also initialize the device
        and distributed environment, which may consume some memory. This part is not
        included in the weights_memory because PyTorch does not control it.

    The memory in one GPU can be classified into 3 categories:
    1. memory used by anything other than the current vLLM instance.
    2. memory used by torch in the current vLLM instance.
    3. memory used in the current vLLM instance, but not by torch.

    A quantitive example:

    Before creating the current vLLM instance:
        category 1: 1 GiB
        category 2: 0 GiB
        category 3: 0 GiB

    After creating the current vLLM instance and loading the model,
    (i.e. before profiling):
        category 1: 1 GiB
        category 2: 2 GiB (model weights take 2 GiB)
        category 3: 0.5 GiB (memory used by NCCL)

    During profiling (peak):
        category 1: 1 GiB
        category 2: 4 GiB (peak activation tensors take 2 GiB)
        category 3: 1 GiB (memory used by NCCL + buffers for some attention backends)

    After profiling:
        category 1: 1 GiB
        category 2: 3 GiB (after garbage-collecting activation tensors)
        category 3: 1 GiB (memory used by NCCL + buffers for some attention backends)

    In this case, non-kv cache takes 5 GiB in total, including:
    a. 2 GiB used by the model weights (category 2)
    b. 2 GiB reserved for the peak activation tensors (category 2)
    c. 1 GiB used by non-torch components (category 3)

    The memory used for loading weights (a.) is directly given from the argument `weights_memory`.

    The increase of `torch.cuda.memory_stats()["allocated_bytes.all.peak"]` during profiling gives (b.).

    The increase of `non_torch_memory` from creating the current vLLM instance until after profiling to get (c.).
    """ # noqa
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    result = MemoryProfilingResult()

    result.before_create = baseline_snapshot
    # the part of memory used for holding the model weights
    result.weights_memory = weights_memory

    result.before_profile.measure()

    yield result

    gc.collect()
    torch.cuda.empty_cache()

    result.after_profile.measure()

    diff_profile = result.after_profile - result.before_profile
    diff_from_create = result.after_profile - result.before_create
    result.torch_peak_increase = diff_profile.torch_peak
    result.non_torch_increase = diff_from_create.non_torch_memory
    result.profile_time = diff_profile.timestamp
    result.non_kv_cache_memory = result.non_torch_increase + result.torch_peak_increase + result.weights_memory  # noqa


@contextlib.contextmanager
def cprofile_context(save_file: Optional[str] = None):
    """Run a cprofile

    Args:
        save_file: path to save the profile result. "1" or
          None will result in printing to stdout.
    """
    import cProfile

    prof = cProfile.Profile()
    prof.enable()

    try:
        yield
    finally:
        prof.disable()
        if save_file and save_file != "1":
            prof.dump_stats(save_file)
        else:
            prof.print_stats(sort="cumtime")


def cprofile(save_file: Optional[str] = None, enabled: bool = True):
    """Decorator to profile a Python method using cProfile.

    Args:
        save_file: Path to save the profile result.
            If "1", None, or "", results will be printed to stdout.
        enabled: Set to false to turn this into a no-op
    """

    def decorator(func: Callable):

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled:
                # If profiling is disabled, just call the function directly.
                return func(*args, **kwargs)

            with cprofile_context(save_file):
                return func(*args, **kwargs)

        return wrapper

    return decorator
