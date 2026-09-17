from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


class AsyncRateLimiter:
    def __init__(
        self,
        rate_limit: int,
        period_seconds: float = 60.0,
        name: str = "default",
    ) -> None:
        if rate_limit < 1:
            raise ValueError("rate_limit must be at least 1")
        if period_seconds <= 0:
            raise ValueError("period_seconds must be positive")

        self.name = name
        self.rate_limit = rate_limit
        self.period_seconds = period_seconds
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()

                while self._timestamps and now - self._timestamps[0] >= self.period_seconds:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.rate_limit:
                    self._timestamps.append(now)
                    return

                wait_until = self._timestamps[0] + self.period_seconds
                wait_time = max(wait_until - now, 0.01)

            await asyncio.sleep(wait_time)

    async def __aenter__(self) -> AsyncRateLimiter:
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rate_limit": self.rate_limit,
            "period_seconds": self.period_seconds,
            "current_usage": len(self._timestamps),
        }


class RateLimiterRegistry:
    def __init__(self) -> None:
        self._limiters: dict[str, AsyncRateLimiter] = {}
        self._lock = threading.Lock()

    def get(
        self,
        name: str,
        rate_limit: int = 10,
        period_seconds: float = 60.0,
    ) -> AsyncRateLimiter:
        with self._lock:
            limiter = self._limiters.get(name)

            if limiter is None:
                limiter = AsyncRateLimiter(
                    rate_limit=rate_limit,
                    period_seconds=period_seconds,
                    name=name,
                )
                self._limiters[name] = limiter

            return limiter

    def all_status(self) -> list[dict[str, Any]]:
        with self._lock:
            return [limiter.status() for limiter in self._limiters.values()]


default_rate_limiter_registry = RateLimiterRegistry()


def rate_limited(
    limiter: AsyncRateLimiter,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    def decorator(
        func: Callable[P, Awaitable[R]],
    ) -> Callable[P, Awaitable[R]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            await limiter.acquire()
            return await func(*args, **kwargs)

        return wrapper

    return decorator