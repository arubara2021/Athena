from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Any

import httpx

from core.models import (
    Source,
    SourcePlatform,
    SourceType,
    resolve_source_platform,
)
from utils.async_helpers import loop_is_alive, safe_aclose

_SEARCH_URL = "https://openlibrary.org/search.json"
_TIMEOUT = 30.0
_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "research-agent/0.2 (educational research agent)",
}


class OpenLibraryClient:
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

    async def __aenter__(self) -> "OpenLibraryClient":
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

        try:
            response = await self._client.get(
                _SEARCH_URL,
                params={
                    "q": cleaned_query,
                    "limit": limit,
                    "fields": "key,title,author_name,first_publish_year,subject,ebook_access",
                },
            )

            if response.status_code != 200:
                return []

            payload = response.json()
            docs = payload.get("docs", []) or []
            sources: list[Source] = []

            for doc in docs[:limit]:
                key = str(doc.get("key") or "").strip()
                title = _clean(doc.get("title"))

                if not key or not title:
                    continue

                url = f"https://openlibrary.org{key}"

                subjects = doc.get("subject", [])

                if isinstance(subjects, str):
                    subjects = [subjects]

                subject_text = ", ".join(
                    _clean(item) for item in subjects[:12] if _clean(item)
                )

                if subject_text:
                    abstract = _clip(subject_text)
                else:
                    abstract = _clip(f"Open Library book: {title}")

                authors = [
                    _clean(author)
                    for author in doc.get("author_name", [])
                    if _clean(author)
                ]

                sources.append(
                    Source(
                        source_id=_source_id("open_library", url, title),
                        title=title,
                        url=url,
                        platform=resolve_source_platform("open_library"),
                        source_type=SourceType.DOCUMENTATION,
                        abstract=abstract,
                        authors=authors[:10],
                        year=_int(doc.get("first_publish_year")),
                        citation_count=None,
                        metadata={
                            "platform_name": "open_library",
                            "openlibrary_key": key,
                            "ebook_access": doc.get("ebook_access"),
                        },
                    )
                )

            return sources

        except Exception:
            return []

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


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _strip_markup(value: Any) -> str:
    return re.sub(r"<[^>]+>", " ", str(value or ""))


def _clip(value: Any, limit: int = 1500) -> str:
    text = _clean(_strip_markup(value))
    return text[:limit]


def _int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _source_id(platform: str, url: str, title: str) -> str:
    raw = f"{platform}|{url}|{title}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None