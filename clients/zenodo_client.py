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

_SEARCH_URL = "https://zenodo.org/api/records"
_TIMEOUT = 30.0
_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "research-agent/0.2 (educational research agent)",
}


class ZenodoClient:
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

    async def __aenter__(self) -> "ZenodoClient":
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
                    "size": limit,
                },
            )

            if response.status_code != 200:
                return []

            payload = response.json()
            hits = payload.get("hits", {}).get("hits", []) or []
            sources: list[Source] = []

            for hit in hits[:limit]:
                metadata = hit.get("metadata", {})

                if not isinstance(metadata, dict):
                    metadata = {}

                title = _clean(metadata.get("title"))

                if not title:
                    continue

                links = hit.get("links", {})

                if not isinstance(links, dict):
                    links = {}

                doi = _clean(metadata.get("doi"))
                url = _clean(links.get("html")) or _clean(links.get("self"))

                if not url and doi:
                    url = f"https://doi.org/{doi}"

                if not url:
                    continue

                abstract = _clip(metadata.get("description"))
                authors = []

                creators = metadata.get("creators", []) or []

                for creator in creators:
                    if isinstance(creator, dict):
                        name = _clean(creator.get("name"))
                    else:
                        name = _clean(creator)

                    if name:
                        authors.append(name)

                year = _year_from_text(metadata.get("publication_date"))

                resource_type = ""

                if isinstance(metadata.get("resource_type"), dict):
                    resource_type = _clean(metadata.get("resource_type", {}).get("title"))
                else:
                    resource_type = _clean(metadata.get("resource_type"))

                source_type = SourceType.DATASET

                if "publication" in resource_type.lower():
                    source_type = SourceType.PAPER
                elif "software" in resource_type.lower():
                    source_type = SourceType.OTHER
                elif "lesson" in resource_type.lower() or "learning" in resource_type.lower():
                    source_type = SourceType.COURSE

                sources.append(
                    Source(
                        source_id=_source_id("zenodo", url, title),
                        title=title,
                        url=url,
                        platform=resolve_source_platform("zenodo"),
                        source_type=source_type,
                        abstract=abstract,
                        authors=authors[:10],
                        year=year,
                        citation_count=None,
                        metadata={
                            "platform_name": "zenodo",
                            "zenodo_id": _clean(hit.get("id")),
                            "doi": doi,
                            "resource_type": resource_type,
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


def _year_from_text(value: Any) -> int | None:
    text = str(value or "")
    match = re.search(r"\b(19|20)\d{2}\b", text)

    if not match:
        return None

    return _int(match.group(0))


def _source_id(platform: str, url: str, title: str) -> str:
    raw = f"{platform}|{url}|{title}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None