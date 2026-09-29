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

_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_TIMEOUT = 30.0
_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "research-agent/0.2 (educational research agent)",
}


class EuropePmcClient:
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

    async def __aenter__(self) -> "EuropePmcClient":
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
                    "query": cleaned_query,
                    "format": "json",
                    "pageSize": limit,
                    "resultType": "core",
                },
            )

            if response.status_code != 200:
                return []

            payload = response.json()
            items = payload.get("resultList", {}).get("result", []) or []
            sources: list[Source] = []

            for item in items[:limit]:
                title = _clean(item.get("title"))

                if not title:
                    continue

                item_id = _clean(item.get("id"))
                source_name = _clean(item.get("source")).upper()
                doi = _clean(item.get("doi"))
                full_text_url = _clean(item.get("fullTextUrl"))

                url = full_text_url

                if not url and doi:
                    url = f"https://doi.org/{doi}"

                if not url and item_id:
                    if source_name == "MED":
                        url = f"https://europepmc.org/article/MED/{item_id}"
                    else:
                        url = f"https://europepmc.org/search?query=ID:{item_id}"

                if not url:
                    continue

                abstract = _clip(item.get("abstractText"))
                year = _int(item.get("pubYear"))
                authors = _clean_authors(item.get("authorString"))

                sources.append(
                    Source(
                        source_id=_source_id("europe_pmc", url, title),
                        title=title,
                        url=url,
                        platform=resolve_source_platform("europe_pmc"),
                        source_type=SourceType.PAPER,
                        abstract=abstract,
                        authors=authors[:10],
                        year=year,
                        citation_count=_int(item.get("citedByCount")),
                        metadata={
                            "platform_name": "europe_pmc",
                            "europe_pmc_id": item_id,
                            "doi": doi,
                            "source": source_name,
                            "is_open_access": item.get("isOpenAccess"),
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


def _clean_authors(value: Any) -> list[str]:
    if not value:
        return []

    if isinstance(value, list):
        return [_clean(item) for item in value if _clean(item)]

    return [_clean(part) for part in str(value).split(",") if _clean(part)]


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