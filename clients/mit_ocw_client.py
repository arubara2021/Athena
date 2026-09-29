from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Any
from urllib.parse import urljoin

import httpx

from core.models import (
    Source,
    SourcePlatform,
    SourceType,
    resolve_source_platform,
)
from utils.async_helpers import loop_is_alive, safe_aclose

_BASE_URL = "https://ocw.mit.edu"
_SEARCH_URL = "https://ocw.mit.edu/search/"
_TIMEOUT = 30.0
_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "User-Agent": "research-agent/0.2 (educational research agent)",
}


class MitOcwClient:
    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None
        self._last_close_error: str | None = None

    @property
    def bound_loop(self) -> asyncio.AbstractEventLoop | None:
        return self._client_loop

    @property
    def last_close_error(self) -> str | None:
        return self._last_close_error

    @property
    def is_closed(self) -> bool:
        return self._client is None

    async def __aenter__(self) -> "MitOcwClient":
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        await self.close()
        return False

    async def close(self) -> None:
        client = self._client
        owning_loop = self._client_loop
        self._client = None
        self._client_loop = None

        if client is None:
            self._last_close_error = None
            return

        ok = await safe_aclose(client, loop=owning_loop)
        self._last_close_error = None if ok else "close_failed"

    async def search(self, query: str, max_results: int = 10) -> list[Source]:
        cleaned_query = _clean(query)

        if not cleaned_query:
            return []

        limit = max(1, min(int(max_results or 10), 50))
        await self._ensure_client()

        sources: list[Source] = []
        seen: set[str] = set()
        tokens = _tokens(cleaned_query)

        param_candidates = (
            {"t": cleaned_query},
            {"q": cleaned_query},
            {"search": cleaned_query},
        )

        for params in param_candidates:
            if len(sources) >= limit:
                break

            try:
                response = await self._client.get(_SEARCH_URL, params=params)

                if response.status_code != 200:
                    continue

                links = re.findall(
                    r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                    response.text,
                    re.S,
                )

                for href, raw_title in links:
                    if len(sources) >= limit:
                        break

                    url = urljoin(_BASE_URL, href)

                    if "/courses/" not in url:
                        continue

                    title = _clean(_strip_markup(raw_title))

                    if len(title) < 8:
                        continue

                    if not _matches(tokens, title):
                        continue

                    source_id = _source_id("mit_ocw", url, title)

                    if source_id in seen:
                        continue

                    seen.add(source_id)

                    sources.append(
                        Source(
                            source_id=source_id,
                            title=title,
                            url=url,
                            platform=resolve_source_platform("mit_ocw"),
                            source_type=SourceType.COURSE,
                            abstract=_clip(f"MIT OpenCourseWare course page: {title}"),
                            authors=[],
                            year=None,
                            citation_count=None,
                            metadata={
                                "platform_name": "mit_ocw",
                            },
                        )
                    )

            except Exception:
                continue

        return sources

    async def _ensure_client(self) -> None:
        running_loop = _running_loop()
        client = self._client

        if client is not None and not client.is_closed:
            if self._client_loop is running_loop:
                return

            old_client = client
            old_loop = self._client_loop
            self._client = None
            self._client_loop = None

            if old_loop is None or not loop_is_alive(old_loop):
                await safe_aclose(old_client, loop=old_loop)
            else:
                try:
                    asyncio.run_coroutine_threadsafe(old_client.aclose(), old_loop)
                except Exception:
                    pass
        elif client is not None and client.is_closed:
            self._client = None
            self._client_loop = None

        self._client = httpx.AsyncClient(
            timeout=_TIMEOUT,
            headers=_HEADERS,
            follow_redirects=True,
        )
        self._client_loop = running_loop


def _tokens(text: str) -> list[str]:
    parts = str(text or "").lower().split()
    return [part for part in parts if len(part) > 2]


def _matches(tokens: list[str], text: str) -> bool:
    if not tokens:
        return True

    target = str(text or "").lower()
    return any(token in target for token in tokens)


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _strip_markup(value: Any) -> str:
    return re.sub(r"<[^>]+>", " ", str(value or ""))


def _clip(value: Any, limit: int = 1500) -> str:
    text = _clean(_strip_markup(value))
    return text[:limit]


def _source_id(platform: str, url: str, title: str) -> str:
    raw = f"{platform}|{url}|{title}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None