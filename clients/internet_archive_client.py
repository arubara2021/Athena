from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from typing import Any

import httpx

from core.models import (
    Source,
    SourcePlatform,
    SourceType,
    resolve_source_platform,
)
from utils.async_helpers import loop_is_alive, safe_aclose

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://archive.org/advancedsearch.php"
_DETAIL_URL = "https://archive.org/details/{identifier}"
_TIMEOUT = 30.0
_USER_AGENT = "research-agent/0.2 (educational research agent)"
_HEADERS = {
    "Accept": "application/json",
    "User-Agent": _USER_AGENT,
}
_MAX_LIMIT = 200
_PAGE_SIZE = 50
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF = 1.5
_MIN_INTERVAL = 0.25
_CACHE_TTL = 300.0
_CACHE_MAX = 256

_SOLR_SPECIAL = re.compile(r'[+\-&|!(){}\[\]^"~*?:\\/]')
_HTML_TAG = re.compile(r"<[^>]+>")
_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2})\b")


class InternetArchiveClient:
    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None
        self._last_close_error: str | None = None
        self._client_lock = asyncio.Lock()
        self._rate_lock = asyncio.Lock()
        self._last_request = 0.0
        self._cache: dict[str, tuple[float, list[Source]]] = {}

    @property
    def bound_loop(self) -> asyncio.AbstractEventLoop | None:
        return self._client_loop

    @property
    def last_close_error(self) -> str | None:
        return self._last_close_error

    @property
    def is_closed(self) -> bool:
        return self._client is None

    async def __aenter__(self) -> "InternetArchiveClient":
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
        cleaned_query = _solr_safe(_clean(query))

        if not cleaned_query:
            return []

        limit = max(1, min(int(max_results or 10), _MAX_LIMIT))

        cache_key = _cache_key(cleaned_query, limit)
        cached = self._cache_get(cache_key)

        if cached is not None:
            return cached

        await self._ensure_client()

        docs: list[dict[str, Any]] = []
        page = 1

        try:
            while len(docs) < limit:
                rows = min(_PAGE_SIZE, limit - len(docs))
                payload = await self._request_page(cleaned_query, rows, page)

                if payload is None:
                    break

                page_docs = payload.get("response", {}).get("docs", []) or []

                if not page_docs:
                    break

                docs.extend(page_docs)

                if len(page_docs) < rows:
                    break

                page += 1

        except Exception:
            logger.exception("internet_archive: search failed query=%r", query)
            return []

        sources = self._parse_docs(docs[:limit])
        self._cache_set(cache_key, sources)
        return sources

    async def _request_page(
        self,
        query: str,
        rows: int,
        page: int,
    ) -> dict[str, Any] | None:
        params = [
            ("q", query),
            ("rows", rows),
            ("page", page),
            ("output", "json"),
            ("fl[]", "identifier"),
            ("fl[]", "title"),
            ("fl[]", "description"),
            ("fl[]", "year"),
            ("fl[]", "date"),
            ("fl[]", "publicdate"),
            ("fl[]", "mediatype"),
            ("fl[]", "creator"),
        ]

        response = await self._get_with_retry(params)

        if response is None:
            return None

        if response.status_code != 200:
            logger.warning(
                "internet_archive: status=%s query=%r",
                response.status_code,
                query,
            )
            return None

        try:
            return response.json()
        except Exception:
            logger.exception("internet_archive: invalid json for query=%r", query)
            return None

    async def _get_with_retry(
        self,
        params: list[tuple[str, Any]],
    ) -> httpx.Response | None:
        delay = _RETRY_BACKOFF

        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            await self._throttle()

            try:
                return await self._client.get(_SEARCH_URL, params=params)
            except httpx.HTTPError as exc:
                logger.warning(
                    "internet_archive: http error attempt=%d err=%s",
                    attempt,
                    exc,
                )

                if attempt >= _RETRY_ATTEMPTS:
                    return None

                await asyncio.sleep(delay)
                delay *= 2

        return None

    async def _throttle(self) -> None:
        async with self._rate_lock:
            now = time.monotonic()
            wait = _MIN_INTERVAL - (now - self._last_request)

            if wait > 0:
                await asyncio.sleep(wait)

            self._last_request = time.monotonic()

    async def _ensure_client(self) -> None:
        running_loop = _running_loop()

        if self._client is not None and not self._client.is_closed:
            if self._client_loop is running_loop:
                return

            old_client = self._client
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

        async with self._client_lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(
                    timeout=_TIMEOUT,
                    headers=_HEADERS,
                    follow_redirects=True,
                )
                self._client_loop = running_loop

    def _parse_docs(self, docs: list[dict[str, Any]]) -> list[Source]:
        sources: list[Source] = []
        seen: set[str] = set()

        for doc in docs:
            identifier = _clean(doc.get("identifier"))
            title = _clean(doc.get("title"))

            if not identifier or not title:
                continue

            if identifier in seen:
                continue

            seen.add(identifier)

            url = _DETAIL_URL.format(identifier=identifier)
            abstract = _clip(_join_value(doc.get("description")))

            if not abstract:
                abstract = _clip(f"Internet Archive item: {title}")

            mediatype = _clean(doc.get("mediatype")).lower()
            source_type = _map_mediatype(mediatype)
            authors = _clean_authors(doc.get("creator"))
            year = _extract_year(doc)

            sources.append(
                Source(
                    source_id=_source_id("internet_archive", url, title),
                    title=title,
                    url=url,
                    platform=resolve_source_platform("internet_archive"),
                    source_type=source_type,
                    abstract=abstract,
                    authors=authors[:10],
                    year=year,
                    citation_count=None,
                    metadata={
                        "platform_name": "internet_archive",
                        "identifier": identifier,
                        "mediatype": mediatype,
                    },
                )
            )

        return sources

    def _cache_get(self, key: str) -> list[Source] | None:
        entry = self._cache.get(key)

        if entry is None:
            return None

        ts, value = entry

        if time.monotonic() - ts > _CACHE_TTL:
            self._cache.pop(key, None)
            return None

        return value

    def _cache_set(self, key: str, value: list[Source]) -> None:
        if len(self._cache) >= _CACHE_MAX:
            oldest = min(self._cache.items(), key=lambda kv: kv[1][0])[0]
            self._cache.pop(oldest, None)

        self._cache[key] = (time.monotonic(), value)


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _solr_safe(query: str) -> str:
    return " ".join(_SOLR_SPECIAL.sub(" ", query).split())


def _strip_markup(value: Any) -> str:
    return _HTML_TAG.sub(" ", str(value or ""))


def _clip(value: Any, limit: int = 1500) -> str:
    text = _clean(_strip_markup(value))
    return text[:limit]


def _join_value(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(_clean(item) for item in value if _clean(item))

    return _clean(value)


def _clean_authors(value: Any) -> list[str]:
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(";") if p.strip()]
        return [_clean(p) for p in parts if _clean(p)]

    if isinstance(value, list):
        result: list[str] = []

        for item in value:
            result.extend(_clean_authors(item))

        return result

    return []


def _extract_year(doc: dict[str, Any]) -> int | None:
    for key in ("year", "date", "publicdate"):
        raw = doc.get(key)

        if raw is None:
            continue

        text = str(raw)
        match = _YEAR_RE.search(text)

        if match:
            return int(match.group(1))

        parsed = _int(raw)

        if parsed is not None:
            return parsed

    return None


def _int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _map_mediatype(mediatype: str) -> SourceType:
    if mediatype == "texts":
        return SourceType.DOCUMENTATION

    if mediatype == "movies":
        return SourceType.VIDEO

    if mediatype == "audio":
        return SourceType.OTHER

    return SourceType.OTHER


def _source_id(platform: str, url: str, title: str) -> str:
    raw = f"{platform}|{url}|{title}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_key(query: str, limit: int) -> str:
    return hashlib.sha256(f"ia|{query}|{limit}".encode("utf-8")).hexdigest()


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None