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

_WORKS_URL = "https://api.crossref.org/works"
_TIMEOUT = 30.0
_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "research-agent/0.2 (mailto:research-agent@example.com)",
}


class CrossrefClient:
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

    async def __aenter__(self) -> "CrossrefClient":
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
                _WORKS_URL,
                params={
                    "query": cleaned_query,
                    "rows": limit,
                    "sort": "relevance",
                    "order": "desc",
                },
            )

            if response.status_code != 200:
                return []

            payload = response.json()
            items = payload.get("message", {}).get("items", []) or []
            sources: list[Source] = []

            for item in items[:limit]:
                titles = item.get("title", []) or []
                title = _clean(titles[0]) if titles else ""

                if not title:
                    continue

                doi = _clean(item.get("DOI"))
                url = _clean(item.get("URL"))

                if not url and doi:
                    url = f"https://doi.org/{doi}"

                if not url:
                    continue

                abstract = _clip(item.get("abstract"))
                authors = []

                for author in item.get("author", []) or []:
                    if not isinstance(author, dict):
                        continue

                    given = _clean(author.get("given"))
                    family = _clean(author.get("family"))
                    name = _clean(f"{given} {family}")

                    if name:
                        authors.append(name)

                year = _extract_year(item)

                container_title_raw = item.get("container-title")

                if container_title_raw:
                    container_title = _clean((container_title_raw or [""])[0])
                else:
                    container_title = ""

                sources.append(
                    Source(
                        source_id=_source_id("crossref", url, title),
                        title=title,
                        url=url,
                        platform=resolve_source_platform("crossref"),
                        source_type=SourceType.PAPER,
                        abstract=abstract,
                        authors=authors[:10],
                        year=year,
                        citation_count=_int(item.get("is-referenced-by-count")),
                        metadata={
                            "platform_name": "crossref",
                            "doi": doi,
                            "container_title": container_title,
                            "type": _clean(item.get("type")),
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


def _extract_year(item: dict[str, Any]) -> int | None:
    for key in ("published-print", "published-online", "created", "issued"):
        value = item.get(key)

        if not isinstance(value, dict):
            continue

        date_parts = value.get("date-parts")

        if not isinstance(date_parts, list):
            continue

        if not date_parts:
            continue

        first_part = date_parts[0]

        if not isinstance(first_part, list):
            continue

        if not first_part:
            continue

        year = _int(first_part[0])

        if year is not None:
            return year

    return None


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