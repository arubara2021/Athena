from __future__ import annotations

import asyncio
import functools
import random
import time
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, TypeVar

from core.exceptions import ResearchAgentError

T = TypeVar("T")


async def run_with_timeout(awaitable: Awaitable[T], timeout: float | None = None) -> T:
    if timeout is None:
        return await awaitable

    try:
        return await asyncio.wait_for(awaitable, timeout)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"Operation timed out after {timeout} seconds") from exc


async def gather_with_results(
    awaitables: Iterable[Awaitable[Any]],
) -> tuple[list[tuple[int, Any]], list[tuple[int, BaseException]]]:
    tasks = [asyncio.ensure_future(item) for item in awaitables]

    if not tasks:
        return [], []

    results = await asyncio.gather(*tasks, return_exceptions=True)

    successes: list[tuple[int, Any]] = []
    failures: list[tuple[int, BaseException]] = []

    for index, result in enumerate(results):
        if isinstance(result, BaseException):
            failures.append((index, result))
        else:
            successes.append((index, result))

    return successes, failures


async def gather_preserving_order(
    awaitables: Iterable[Awaitable[Any]],
    default: Any = None,
) -> list[Any]:
    tasks = [asyncio.ensure_future(item) for item in awaitables]

    if not tasks:
        return []

    results = await asyncio.gather(*tasks, return_exceptions=True)

    return [
        default if isinstance(result, BaseException) else result
        for result in results
    ]


async def async_map(
    items: Iterable[Any],
    func: Callable[[Any], Awaitable[Any]],
    concurrency: int = 5,
    raise_on_error: bool = False,
) -> list[Any]:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    semaphore = asyncio.Semaphore(concurrency)
    materialized_items = list(items)

    async def worker(item: Any) -> Any:
        async with semaphore:
            return await func(item)

    results = await asyncio.gather(
        *(worker(item) for item in materialized_items),
        return_exceptions=True,
    )

    if raise_on_error:
        for result in results:
            if isinstance(result, BaseException):
                raise result

    return results


async def cancel_tasks(tasks: Iterable[asyncio.Task[Any]], timeout: float = 5.0) -> None:
    materialized_tasks = list(tasks)

    if not materialized_tasks:
        return

    for task in materialized_tasks:
        if not task.done():
            task.cancel()

    await asyncio.gather(*materialized_tasks, return_exceptions=True)


async def sleep_with_jitter(base_delay: float, jitter: float = 0.1) -> None:
    if base_delay < 0:
        raise ValueError("base_delay must be non-negative")

    if jitter < 0:
        raise ValueError("jitter must be non-negative")

    delay = base_delay

    if jitter:
        delay += random.uniform(0.0, jitter * max(base_delay, 1.0))

    await asyncio.sleep(max(delay, 0.0))


async def run_sync_in_executor(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    loop = asyncio.get_running_loop()
    call = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(None, call)


async def wait_first_successful(
    awaitables: Iterable[Awaitable[Any]],
    timeout: float | None = None,
) -> Any:
    tasks = [asyncio.ensure_future(item) for item in awaitables]

    if not tasks:
        raise ValueError("awaitables cannot be empty")

    pending = set(tasks)
    deadline = None if timeout is None else time.monotonic() + timeout
    errors: list[BaseException] = []

    try:
        while pending:
            remaining = None if deadline is None else max(deadline - time.monotonic(), 0.0)

            done, pending = await asyncio.wait(
                pending,
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not done:
                raise asyncio.TimeoutError

            for task in done:
                if task.cancelled():
                    continue

                exc = task.exception()

                if exc is None:
                    return task.result()

                errors.append(exc)

        raise ResearchAgentError(
            "All awaitables failed",
            details={"errors": [str(item) for item in errors]},
        )
    finally:
        await cancel_tasks(pending)