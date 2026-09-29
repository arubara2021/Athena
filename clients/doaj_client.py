from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

from clients.base_client import BaseHTTPClient
from core import constants
from core.models import Source, SourcePlatform, SourceType
from utils.cache import build_cache_key
from utils.text import clean_text, truncate_text

_HTML_TAG = re.compile(r"<[^>]+>")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


class DoajClient(BaseHTTPClient):
    platform_name = SourcePlatform.DOAJ.value
    retryable_status_codes = (429, 500, 502, 503, 504)

    def __init__(self) -> None:
        super().__init__(
            base_url=constants.DOAJ_BASE_URL,
            rate_limit=10,
            period_seconds=60.0,
            cache_enabled=True,
            timeout=30.0,
            max_retries=2,
            retry_backoff_seconds=2.0,
        )

    async def search(self, query: str, max_results: int = 10) -> list[Source]:
        cleaned_query = clean_text(query)
        if not cleaned_query:
            return []

        limit = max(1, min(int(max_results or 10), 50))

        cache_key = build_cache_key(
            self.platform_name,
            "search_v2",
            cleaned_query,
            limit,
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        payload = None

        try:
            payload = await self._get_json(
                f"search/articles/{quote(cleaned_query, safe='')}",
                params={"pageSize": limit},
            )
        except Exception:
            payload = None

        if payload is None:
            try:
                payload = await self._get_json(
                    "search/articles",
                    params={
                        "query": cleaned_query,
                        "pageSize": limit,
                    },
                )
            except Exception:
                payload = None

        if payload is None:
            self._logger.warning("DOAJ search failed or returned no usable payload")
            return []

        results = payload.get("results", []) if isinstance(payload, dict) else []

        sources: list[Source] = []

        for item in results[:limit]:
            source = self._map_result(item)
            if source is not None:
                sources.append(source)

        if sources:
            self._cache_set(cache_key, sources)

        return sources

    def _map_result(self, item: Any) -> Source | None:
        try:
            if not isinstance(item, dict):
                return None

            bibjson = item.get("bibjson", {})
            if not isinstance(bibjson, dict):
                bibjson = {}

            title = clean_text(bibjson.get("title"))
            if not title:
                return None

            url_value = ""

            links = bibjson.get("link", [])
            if isinstance(links, list):
                for link in links:
                    if not isinstance(link, dict):
                        continue

                    link_url = clean_text(link.get("url"))
                    if link_url:
                        url_value = link_url
                        break

            if not url_value:
                url_value = clean_text(item.get("url"))

            doi = self._extract_doi(bibjson)

            if not url_value and doi:
                url_value = f"https://doi.org/{doi}"

            if not url_value:
                return None

            abstract = self._clip(bibjson.get("abstract"))

            authors: list[str] = []
            for author in bibjson.get("author", []) or []:
                if isinstance(author, dict):
                    name = clean_text(author.get("name"))
                else:
                    name = clean_text(author)

                if name:
                    authors.append(name)

            year = self._int(bibjson.get("year"))
            if year is None:
                year = self._year_from_text(item.get("created_date"))

            journal = ""
            journal_payload = bibjson.get("journal")
            if isinstance(journal_payload, dict):
                journal = clean_text(journal_payload.get("title"))

            return Source(
                source_id=self._make_source_id(self.platform_name, url_value),
                title=title,
                url=url_value,
                platform=SourcePlatform.DOAJ,
                source_type=SourceType.PAPER,
                abstract=abstract,
                authors=authors[:10],
                published_at=None,
                year=year,
                citation_count=None,
                has_code=None,
                difficulty=None,
                score=None,
                metadata={
                    "platform_name": "doaj",
                    "doaj_id": clean_text(item.get("id")),
                    "doi": doi,
                    "journal": journal,
                },
            )
        except Exception as exc:
            self._logger.warning(f"Skipping DOAJ result: {exc}")
            return None

    def _extract_doi(self, bibjson: dict[str, Any]) -> str:
        identifiers = bibjson.get("identifier")

        if isinstance(identifiers, dict):
            return clean_text(identifiers.get("id"))

        if isinstance(identifiers, list):
            for identifier in identifiers:
                if not isinstance(identifier, dict):
                    continue

                identifier_type = clean_text(identifier.get("type")).lower()
                identifier_value = clean_text(identifier.get("id"))

                if identifier_type == "doi" and identifier_value:
                    return identifier_value

                if identifier_value:
                    return identifier_value

        return ""

    def _clip(self, value: Any, limit: int = 1500) -> str | None:
        text = clean_text(_HTML_TAG.sub(" ", str(value or "")))
        if not text:
            return None

        return truncate_text(text, max_length=limit, suffix="")

    def _int(self, value: Any) -> int | None:
        try:
            if value is None:
                return None
            return int(value)
        except Exception:
            return None

    def _year_from_text(self, value: Any) -> int | None:
        text = str(value or "")
        match = _YEAR_RE.search(text)

        if not match:
            return None

        return self._int(match.group(0))