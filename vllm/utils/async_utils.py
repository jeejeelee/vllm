# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import concurrent
import contextlib

from asyncio import FIRST_COMPLETED, AbstractEventLoop, Task

from collections.abc import (AsyncGenerator, Awaitable)
from functools import partial
from typing import (Callable, Optional,TypeVar)

from typing_extensions import ParamSpec


P = ParamSpec('P')
T = TypeVar("T")



def make_async(
    func: Callable[P, T],
    executor: Optional[concurrent.futures.Executor] = None
) -> Callable[P, Awaitable[T]]:
    """Take a blocking function, and run it on in an executor thread.

    This function prevents the blocking function from blocking the
    asyncio event loop.
    The code in this function needs to be thread safe.
    """

    def _async_wrapper(*args: P.args, **kwargs: P.kwargs) -> asyncio.Future:
        loop = asyncio.get_event_loop()
        p_func = partial(func, *args, **kwargs)
        return loop.run_in_executor(executor=executor, func=p_func)

    return _async_wrapper


def _next_task(iterator: AsyncGenerator[T, None],
               loop: AbstractEventLoop) -> Task:
    # Can use anext() in python >= 3.10
    return loop.create_task(iterator.__anext__())  # type: ignore[arg-type]


async def merge_async_iterators(
    *iterators: AsyncGenerator[T,
                               None], ) -> AsyncGenerator[tuple[int, T], None]:
    """Merge multiple asynchronous iterators into a single iterator.

    This method handle the case where some iterators finish before others.
    When it yields, it yields a tuple (i, item) where i is the index of the
    iterator that yields the item.
    """
    if len(iterators) == 1:
        # Fast-path single iterator case.
        async for item in iterators[0]:
            yield 0, item
        return

    loop = asyncio.get_running_loop()

    awaits = {_next_task(pair[1], loop): pair for pair in enumerate(iterators)}
    try:
        while awaits:
            done, _ = await asyncio.wait(awaits.keys(),
                                         return_when=FIRST_COMPLETED)
            for d in done:
                pair = awaits.pop(d)
                try:
                    item = await d
                    i, it = pair
                    awaits[_next_task(it, loop)] = pair
                    yield i, item
                except StopAsyncIteration:
                    pass
    finally:
        # Cancel any remaining iterators
        for f, (_, it) in awaits.items():
            with contextlib.suppress(BaseException):
                f.cancel()
                await it.aclose()


async def collect_from_async_generator(
        iterator: AsyncGenerator[T, None]) -> list[T]:
    """Collect all items from an async generator into a list."""
    items = []
    async for item in iterator:
        items.append(item)
    return items


async def _run_task_with_lock(task: Callable, lock: asyncio.Lock, *args,
                              **kwargs):
    """Utility function to run async task in a lock"""
    async with lock:
        return await task(*args, **kwargs)