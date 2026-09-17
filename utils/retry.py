from __future__ import annotations

import asyncio
import functools
import inspect
import random
import time
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from core import constants
from core.exceptions import ResearchAgentError

P = ParamSpec("P")
R = TypeVar("R")


class RetryPolicy:
    def __init__(
        self,
        max_attempts: int = constants.DEFAULT_MAX_RETRIES,
        base_delay: float = constants.DEFAULT_RETRY_BACKOFF_SECONDS,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        jitter: float = 0.1,
        retryable_exceptions: tuple[type[BaseException], ...] = (Exception,),
        fatal_exceptions: tuple[type[BaseException], ...] = (),
        on_retry: Callable[[int, BaseException, float], None] | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if base_delay < 0:
            raise ValueError("base_delay must be non-negative")
        if max_delay < base_delay:
            raise ValueError("max_delay must be greater than or equal to base_delay")
        if exponential_base <= 1:
            raise ValueError("exponential_base must be greater than 1")
        if jitter < 0:
            raise ValueError("jitter must be non-negative")

        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions
        self.fatal_exceptions = fatal_exceptions
        self.on_retry = on_retry

    def calculate_delay(self, attempt: int) -> float:
        attempt = max(1, attempt)
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)

        if self.jitter:
            delay += random.uniform(0.0, self.jitter * delay)

        return delay

    def is_retryable(self, exc: BaseException) -> bool:
        if self.fatal_exceptions and isinstance(exc, self.fatal_exceptions):
            return False
        return isinstance(exc, self.retryable_exceptions)


def build_retry_policy(
    max_attempts: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
    retryable_exceptions: type[BaseException] | tuple[type[BaseException], ...] = Exception,
    fatal_exceptions: tuple[type[BaseException], ...] = (),
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> RetryPolicy:
    return RetryPolicy(
        max_attempts=max_attempts if max_attempts is not None else constants.DEFAULT_MAX_RETRIES,
        base_delay=base_delay if base_delay is not None else constants.DEFAULT_RETRY_BACKOFF_SECONDS,
        max_delay=max_delay if max_delay is not None else 30.0,
        retryable_exceptions=_normalize_exceptions(retryable_exceptions),
        fatal_exceptions=fatal_exceptions,
        on_retry=on_retry,
    )


def _normalize_exceptions(
    exceptions: type[BaseException] | tuple[type[BaseException], ...],
) -> tuple[type[BaseException], ...]:
    if isinstance(exceptions, type):
        return (exceptions,)
    return tuple(exceptions)


def retry(
    policy: RetryPolicy | None = None,
    max_attempts: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
    retryable_exceptions: type[BaseException] | tuple[type[BaseException], ...] = Exception,
    fatal_exceptions: tuple[type[BaseException], ...] = (),
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> Callable[..., Any]:
    resolved_policy = policy or build_retry_policy(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        retryable_exceptions=retryable_exceptions,
        fatal_exceptions=fatal_exceptions,
        on_retry=on_retry,
    )

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_error: BaseException | None = None

                for attempt in range(1, resolved_policy.max_attempts + 1):
                    try:
                        return await func(*args, **kwargs)
                    except BaseException as exc:
                        last_error = exc

                        if attempt >= resolved_policy.max_attempts:
                            raise

                        if not resolved_policy.is_retryable(exc):
                            raise

                        delay = resolved_policy.calculate_delay(attempt)

                        if resolved_policy.on_retry:
                            resolved_policy.on_retry(attempt, exc, delay)

                        await asyncio.sleep(delay)

                raise ResearchAgentError(
                    "Retry loop ended unexpectedly",
                    details={"function": getattr(func, "__name__", "unknown")},
                ) from last_error

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: BaseException | None = None

            for attempt in range(1, resolved_policy.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except BaseException as exc:
                    last_error = exc

                    if attempt >= resolved_policy.max_attempts:
                        raise

                    if not resolved_policy.is_retryable(exc):
                        raise

                    delay = resolved_policy.calculate_delay(attempt)

                    if resolved_policy.on_retry:
                        resolved_policy.on_retry(attempt, exc, delay)

                    time.sleep(delay)

            raise ResearchAgentError(
                "Retry loop ended unexpectedly",
                details={"function": getattr(func, "__name__", "unknown")},
            ) from last_error

        return sync_wrapper

    return decorator