from __future__ import annotations

import os
from typing import Any

from clients.base_client import BaseHTTPClient
from core import constants
from core.config import get_settings
from core.exceptions import SourceFetchError
from core.models import Source, SourcePlatform, SourceType
from utils.cache import build_cache_key
from utils.text import clean_text, truncate_text


class SemanticScholarClient(BaseHTTPClient):
    platform_name = SourcePlatform.SEMANTIC_SCHOLAR.value

    def __init__(self) -> None:
        self._api_key = self._read_api_key()
        self._allow_without_key = self._read_allow_without_key()
        rate_limit = self._read_rate_limit()

        super().__init__(
            base_url=constants.SEMANTIC_SCHOLAR_BASE_URL,
            rate_limit=rate_limit,
            period_seconds=60.0,
            cache_enabled=True,
            timeout=30.0,
            max_retries=2,
            retry_backoff_seconds=2.0,
        )

        self.last_error: str | None = None
        self._skip_logged = False

    @staticmethod
    def _read_api_key() -> str:
        try:
            settings = get_settings()
            secret = settings.semantic_scholar_api_key

            if secret is None:
                return ""

            return str(secret.get_secret_value()).strip()
        except Exception:
            return ""

    @staticmethod
    def _read_allow_without_key() -> bool:
        try:
            settings = get_settings()
            return bool(getattr(settings, "allow_semantic_scholar_without_key", False))
        except Exception:
            return False

    def _read_rate_limit(self) -> int:
        try:
            settings = get_settings()

            if self._api_key:
                return int(settings.semantic_scholar_rate_limit_with_key_per_minute)

            return int(settings.semantic_scholar_rate_limit_per_minute)
        except Exception:
            return 3

    def _build_default_headers(self) -> dict[str, str]:
        headers = super()._build_default_headers() or {}

        contact = os.getenv("RESEARCH_AGENT_CONTACT", "").strip()

        if contact:
            headers["User-Agent"] = (
                f"{constants.APP_NAME}/{constants.APP_VERSION} "
                f"(personal research agent; educational use; contact: {contact})"
            )
        else:
            headers["User-Agent"] = (
                f"{constants.APP_NAME}/{constants.APP_VERSION} "
                f"(personal research agent; educational use)"
            )

        headers["Accept"] = "application/json"

        if self._api_key:
            headers["x-api-key"] = self._api_key

        return headers

    async def search(self, query: str, max_results: int = 10) -> list[Source]:
        if not self.platform_enabled:
            return []

        if not self._api_key and not self._allow_without_key:
            if not self._skip_logged:
                self._logger.warning(
                    "Semantic Scholar skipped because no API key is configured"
                )
                self._skip_logged = True

            self.last_error = "semantic_scholar_no_api_key"
            return []

        cleaned_query = clean_text(query)

        if not cleaned_query:
            return []

        cache_key = build_cache_key(
            self.platform_name,
            "search_v3",
            cleaned_query,
            max_results,
        )

        cached = self._cache_get(cache_key)

        if cached is not None:
            return cached

        params = {
            "query": cleaned_query,
            "limit": max_results,
            "fields": "title,abstract,year,authors,citationCount,externalIds,url,openAccessPdf,publicationDate,venue",
        }

        try:
            payload = await self._get_json("/paper/search", params=params)
        except SourceFetchError as exc:
            self.last_error = str(exc)
            raise
        except Exception as exc:
            self.last_error = str(exc)
            raise SourceFetchError(
                "Semantic Scholar search failed",
                details={"query": cleaned_query, "error": str(exc)},
            ) from exc

        if isinstance(payload, dict) and payload.get("error"):
            error_message = str(payload.get("error"))
            self.last_error = error_message

            raise SourceFetchError(
                f"Semantic Scholar API error: {error_message}",
                details={"query": cleaned_query},
            )

        self.last_error = None

        items = payload.get("data", []) if isinstance(payload, dict) else []
        sources: list[Source] = []

        for item in items[:max_results]:
            source = self._map_item(item)

            if source is not None:
                sources.append(source)

        self._cache_set(cache_key, sources)
        return sources

    def _map_item(self, item: dict) -> Source | None:
        try:
            title = clean_text(item.get("title", ""))

            if not title:
                return None

            paper_url = item.get("url") or ""
            open_access = item.get("openAccessPdf") or {}
            pdf_url = open_access.get("url") if isinstance(open_access, dict) else None
            final_url = self._clean_url(pdf_url or paper_url)

            if not final_url:
                return None

            authors = [
                clean_text(author.get("name", ""))
                for author in item.get("authors", []) or []
                if author.get("name")
            ]

            external_ids = item.get("externalIds") or {}
            citation_count = item.get("citationCount")
            publication_date = self._parse_datetime(item.get("publicationDate"))
            year = item.get("year") or (
                publication_date.year if publication_date else None
            )

            return Source(
                source_id=self._make_source_id(
                    self.platform_name,
                    str(item.get("paperId") or final_url),
                ),
                title=title,
                url=final_url,
                platform=SourcePlatform.SEMANTIC_SCHOLAR,
                source_type=SourceType.PAPER,
                abstract=truncate_text(
                    clean_text(item.get("abstract", "")),
                    max_length=2000,
                    suffix="",
                )
                or None,
                authors=authors,
                published_at=publication_date,
                year=self._parse_year(year),
                citation_count=int(citation_count)
                if isinstance(citation_count, int)
                else None,
                has_code=None,
                difficulty=None,
                score=None,
                metadata={
                    "paper_id": item.get("paperId"),
                    "external_ids": external_ids,
                    "has_pdf": bool(pdf_url),
                    "venue": clean_text(item.get("venue", "")),
                },
            )
        except Exception as exc:
            self._logger.warning(f"Skipping Semantic Scholar item: {exc}")
            return None