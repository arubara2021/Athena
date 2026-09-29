from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Any
from urllib.parse import urljoin, unquote

import httpx

from core.models import Source, SourcePlatform, SourceType
from utils.async_helpers import loop_is_alive, safe_aclose
from utils.logger import get_logger, get_trace_logger

_HTML_ENDPOINTS = (
    "https://chem.libretexts.org/Special:Search",
    "https://bio.libretexts.org/Special:Search",
    "https://phys.libretexts.org/Special:Search",
    "https://math.libretexts.org/Special:Search",
    "https://eng.libretexts.org/Special:Search",
    "https://biz.libretexts.org/Special:Search",
    "https://human.libretexts.org/Special:Search",
    "https://socialsci.libretexts.org/Special:Search",
    "https://med.libretexts.org/Special:Search",
    "https://stats.libretexts.org/Special:Search",
)

_ANCHOR_PATTERN = re.compile(
    r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)

_SKIP_HREF_MARKERS = (
    "special:",
    "user:",
    "talk:",
    "action=",
    "login",
    "signup",
    "createaccount",
    "helppage",
)

_TIMEOUT = 30.0
_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "User-Agent": "research-agent/0.2 (educational research agent)",
}


class LibretextsClient:
    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None
        self._last_close_error: str | None = None
        self._trace = get_trace_logger()
        self._logger = get_logger("clients.libretexts")

    @property
    def bound_loop(self) -> asyncio.AbstractEventLoop | None:
        return self._client_loop

    @property
    def last_close_error(self) -> str | None:
        return self._last_close_error

    @property
    def is_closed(self) -> bool:
        return self._client is None

    async def __aenter__(self) -> "LibretextsClient":
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

        ok = await safe_aclose(client, loop=owning_loop, logger=self._logger)
        self._last_close_error = None if ok else "close_failed"

    async def search(self, query: str, max_results: int = 10) -> list[Source]:
        cleaned_query = _clean(query)

        if not cleaned_query:
            return []

        limit = max(1, min(int(max_results or 10), 50))
        await self._ensure_client()

        return await self._search_html(cleaned_query, limit)

    async def _search_html(self, query: str, limit: int) -> list[Source]:
        sources: list[Source] = []
        seen: set[str] = set()
        tokens = _tokens(query)

        endpoint_results: dict[str, int] = {}
        endpoint_errors: dict[str, str] = {}

        for endpoint in _HTML_ENDPOINTS:
            if len(sources) >= limit:
                break

            try:
                response = await self._client.get(
                    endpoint,
                    params={
                        "query": query,
                        "type": "wiki",
                    },
                    headers=_HEADERS,
                )

                if response.status_code != 200:
                    endpoint_errors[endpoint] = f"http_{response.status_code}"
                    endpoint_results[endpoint] = 0
                    continue

                endpoint_count = 0

                for href, raw_title in _ANCHOR_PATTERN.findall(response.text):
                    if len(sources) >= limit:
                        break

                    if _is_skippable(href):
                        continue

                    if href.startswith("http") and "libretexts.org" not in href:
                        continue

                    if href.startswith("/"):
                        url = urljoin(endpoint, href)
                    elif href.startswith("http"):
                        url = href
                    else:
                        continue

                    title = _clean(_strip_markup(raw_title))

                    if not title or len(title) < 5:
                        slug = href.split("/")[-1].replace("_", " ").replace("-", " ")
                        title = unquote(slug).title()

                    if not title:
                        continue

                    if not _matches(tokens, title) and not _matches(tokens, href):
                        continue

                    source_id = _source_id("libretexts", url, title)

                    if source_id in seen:
                        continue

                    seen.add(source_id)

                    sources.append(
                        Source(
                            source_id=source_id,
                            title=title,
                            url=url,
                            platform=_platform_enum("libretexts"),
                            source_type=SourceType.DOCUMENTATION,
                            abstract=_clip(f"LibreTexts page: {title}"),
                            authors=[],
                            year=None,
                            citation_count=None,
                            metadata={"platform_name": "libretexts"},
                        )
                    )

                    endpoint_count += 1

                endpoint_results[endpoint] = endpoint_count
            except Exception as exc:
                endpoint_errors[endpoint] = _clip(str(exc), limit=160)
                endpoint_results[endpoint] = 0
                continue

        total_found = len(sources)

        if total_found == 0:
            self._trace.emit(
                "libretexts_empty",
                query=query,
                endpoints_attempted=len(_HTML_ENDPOINTS),
                endpoint_results=endpoint_results,
                endpoint_errors=endpoint_errors,
            )
        else:
            self._trace.emit(
                "libretexts_result",
                query=query,
                sources_found=total_found,
                endpoint_results=endpoint_results,
            )

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
                await safe_aclose(old_client, loop=old_loop, logger=self._logger)
            else:
                self._logger.warning(
                    "Libretexts client was bound to a different running "
                    "loop; orphaning previous client and rebinding"
                )

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


def _is_skippable(href: str) -> bool:
    lowered = href.lower()

    return any(marker in lowered for marker in _SKIP_HREF_MARKERS)


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


def _platform_enum(name: str) -> SourcePlatform:
    try:
        return SourcePlatform(name)
    except Exception:
        return SourcePlatform.WEB


def _source_id(platform: str, url: str, title: str) -> str:
    raw = f"{platform}|{url}|{title}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None