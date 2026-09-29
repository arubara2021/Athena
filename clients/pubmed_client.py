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

_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_TIMEOUT = 30.0
_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "research-agent/0.2 (educational research agent)",
}


class PubmedClient:
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

    async def __aenter__(self) -> "PubmedClient":
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
            search_response = await self._client.get(
                _ESEARCH_URL,
                params={
                    "db": "pubmed",
                    "term": cleaned_query,
                    "retmax": limit,
                    "retmode": "json",
                },
            )

            if search_response.status_code != 200:
                return []

            search_payload = search_response.json()
            ids = search_payload.get("esearchresult", {}).get("idlist", []) or []

            if not ids:
                return []

            ids = ids[:limit]

            summary_response = await self._client.get(
                _ESUMMARY_URL,
                params={
                    "db": "pubmed",
                    "id": ",".join(ids),
                    "retmode": "json",
                },
            )

            summary_payload: dict[str, Any] = {}

            if summary_response.status_code == 200:
                summary_payload = summary_response.json().get("result", {})

            detailed_info = await self._fetch_detailed_info(ids)

            sources: list[Source] = []

            for pubmed_id in ids:
                summary = summary_payload.get(pubmed_id, {})
                details = detailed_info.get(pubmed_id, {})

                title = details.get("title") or _clean(summary.get("title"))

                if not title:
                    continue

                url = f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"
                abstract = details.get("abstract") or None
                year = details.get("year") or _year_from_text(summary.get("pubdate"))

                authors = details.get("authors")

                if not authors:
                    authors = [
                        _clean(author.get("name"))
                        for author in summary.get("authors", [])
                        if _clean(author.get("name"))
                    ]

                sources.append(
                    Source(
                        source_id=_source_id("pubmed", url, title),
                        title=title,
                        url=url,
                        platform=resolve_source_platform("pubmed"),
                        source_type=SourceType.PAPER,
                        abstract=abstract,
                        authors=authors[:10],
                        year=year,
                        citation_count=None,
                        metadata={
                            "platform_name": "pubmed",
                            "pubmed_id": pubmed_id,
                            "journal": _clean(
                                summary.get("fulljournalname") or summary.get("source")
                            ),
                        },
                    )
                )

            return sources

        except Exception:
            return []

    async def _fetch_detailed_info(
        self,
        ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        try:
            response = await self._client.get(
                _EFETCH_URL,
                params={
                    "db": "pubmed",
                    "id": ",".join(ids),
                    "retmode": "xml",
                },
            )

            if response.status_code != 200:
                return {}

            return _parse_pubmed_xml(response.text)

        except Exception:
            return {}

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


def _parse_pubmed_xml(text: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    articles = re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", text, re.S)

    for article in articles:
        pmid_match = re.search(r"<PMID[^>]*>(\d+)</PMID>", article)

        if not pmid_match:
            continue

        pmid = pmid_match.group(1)

        title_match = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", article, re.S)
        abstract_match = re.search(r"<Abstract>(.*?)</Abstract>", article, re.S)
        year_match = re.search(r"<Year>(\d{4})</Year>", article)
        journal_match = re.search(r"<Title>(.*?)</Title>", article, re.S)
        authors = re.findall(r"<LastName>(.*?)</LastName>", article, re.S)

        result[pmid] = {
            "title": _clean(_strip_markup(title_match.group(1))) if title_match else "",
            "abstract": _clip(abstract_match.group(1)) if abstract_match else "",
            "year": _int(year_match.group(1)) if year_match else None,
            "journal": _clean(_strip_markup(journal_match.group(1))) if journal_match else "",
            "authors": [_clean(author) for author in authors if _clean(author)],
        }

    return result


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