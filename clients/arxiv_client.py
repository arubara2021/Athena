from __future__ import annotations

import re

import feedparser

from core import constants
from clients.base_client import BaseHTTPClient
from core.exceptions import SourceFetchError
from core.models import Source, SourcePlatform, SourceType
from search.query_tokens import get_generic_terms, strip_filler
from utils.cache import build_cache_key
from utils.logger import get_logger, get_trace_logger
from utils.text import clean_text, truncate_text


class ArxivClient(BaseHTTPClient):
    platform_name = SourcePlatform.ARXIV.value

    def __init__(self) -> None:
        super().__init__(
            base_url=constants.ARXIV_BASE_URL,
            rate_limit=5,
            period_seconds=60.0,
            cache_enabled=True,
            timeout=30.0,
            max_retries=2,
            retry_backoff_seconds=2.0,
        )
        self._trace = get_trace_logger()
        self._logger = get_logger(f"clients.{self.platform_name}")

    async def search(self, query: str, max_results: int = 10) -> list[Source]:
        cleaned_query = clean_text(query)

        if not cleaned_query:
            return []

        cache_key = build_cache_key(
            self.platform_name,
            "search_v2",
            cleaned_query,
            max_results,
        )

        cached = self._cache_get(cache_key)

        if cached is not None:
            return cached

        search_query = self._build_search_query(cleaned_query)

        if not search_query:
            self._trace.emit(
                "arxiv_empty_query",
                original_query=cleaned_query,
            )
            return []

        self._trace.emit(
            "arxiv_query_built",
            original_query=cleaned_query,
            search_query=search_query,
        )

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        raw_text = await self._get_text("/query", params=params)

        try:
            feed = feedparser.parse(raw_text)
        except Exception as exc:
            raise SourceFetchError(
                "Failed to parse arXiv feed",
                details={"error": str(exc)},
            ) from exc

        entries = list(getattr(feed, "entries", []) or [])

        if not entries:
            self._trace.emit(
                "arxiv_empty_result",
                original_query=cleaned_query,
                search_query=search_query,
                max_results=max_results,
            )
            return []

        sources: list[Source] = []

        for entry in entries[:max_results]:
            source = self._map_entry(entry)

            if source is not None:
                sources.append(source)

        if sources:
            self._cache_set(cache_key, sources)

        self._trace.emit(
            "arxiv_result",
            original_query=cleaned_query,
            search_query=search_query,
            sources_found=len(sources),
        )

        return sources

    def _build_search_query(self, query: str) -> str:
        cleaned = strip_filler(clean_text(query))
        cleaned = re.sub(r"[^\w\s+#.-]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()

        raw_tokens = re.findall(r"[a-z0-9+#.]+", cleaned)
        generic_terms = get_generic_terms()

        original_words: list[str] = []

        for token in raw_tokens:
            if len(token) < 2:
                continue

            if token not in original_words:
                original_words.append(token)

        meaningful_tokens: list[str] = []

        for token in original_words:
            if token in generic_terms:
                continue

            meaningful_tokens.append(token)

        if len(meaningful_tokens) >= 2:
            selected = meaningful_tokens[:5]
            return " AND ".join(f"all:{token}" for token in selected)

        if len(meaningful_tokens) == 1 and len(original_words) >= 2:
            phrase = " ".join(original_words[:4])
            phrase = re.sub(r"\s+", " ", phrase).strip()

            if phrase:
                return f'all:"{phrase}"'

        if len(meaningful_tokens) == 1:
            return f"all:{meaningful_tokens[0]}"

        phrase_source = clean_text(query)
        phrase_source = re.sub(r"[\"']", " ", phrase_source)
        phrase_source = re.sub(r"[\\(){}\[\]^~*?:/]", " ", phrase_source)
        phrase_source = re.sub(r"\s+", " ", phrase_source).strip()

        if not phrase_source:
            return ""

        return f'all:"{phrase_source[:60]}"'

    def _map_entry(self, entry: dict) -> Source | None:
        try:
            title = clean_text(entry.get("title", ""))
            link = self._clean_url(entry.get("link", ""))

            if not title or not link:
                return None

            authors: list[str] = []

            for author in entry.get("authors", []) or []:
                name = clean_text(author.get("name", ""))

                if name:
                    authors.append(name)

            abstract = clean_text(entry.get("summary", ""))
            published_at = self._parse_datetime(entry.get("published"))
            arxiv_id = clean_text(entry.get("id", link))

            categories = [
                tag.get("term")
                for tag in entry.get("tags", []) or []
                if tag.get("term")
            ]

            return Source(
                source_id=self._make_source_id(self.platform_name, arxiv_id or link),
                title=title,
                url=link,
                platform=SourcePlatform.ARXIV,
                source_type=SourceType.PAPER,
                abstract=truncate_text(abstract, max_length=2000, suffix="") or None,
                authors=authors,
                published_at=published_at,
                year=published_at.year if published_at else None,
                citation_count=None,
                has_code=None,
                difficulty=None,
                score=None,
                metadata={
                    "arxiv_id": arxiv_id,
                    "categories": categories,
                },
            )
        except Exception as exc:
            self._logger.warning(f"Skipping arXiv entry: {exc}")
            return None