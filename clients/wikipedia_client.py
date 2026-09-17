from __future__ import annotations

import os
import re
from typing import Any
from urllib.parse import quote

from clients.base_client import BaseHTTPClient
from core import constants
from core.exceptions import SourceFetchError
from core.models import Source, SourcePlatform, SourceType
from search.query_tokens import (
    clean_topic,
    concept_tokens,
    contains_any_token,
    get_generic_terms,
    strip_filler,
)
from utils.cache import build_cache_key
from utils.text import clean_text, truncate_text


class WikipediaClient(BaseHTTPClient):
    platform_name = SourcePlatform.WIKIPEDIA.value

    def __init__(self) -> None:
        super().__init__(
            base_url="https://en.wikipedia.org/w/rest.php/v1",
            rate_limit=8,
            period_seconds=60.0,
            cache_enabled=True,
            timeout=30.0,
            max_retries=2,
            retry_backoff_seconds=2.0,
        )

    def _build_default_headers(self) -> dict[str, str]:
        headers = super()._build_default_headers()

        contact = os.getenv("RESEARCH_AGENT_CONTACT", "").strip()

        if contact:
            user_agent = (
                f"{constants.APP_NAME}/{constants.APP_VERSION} "
                f"(personal research agent; educational use; contact: {contact})"
            )
        else:
            user_agent = (
                f"{constants.APP_NAME}/{constants.APP_VERSION} "
                f"(personal research agent; educational use)"
            )

        headers["User-Agent"] = user_agent
        headers["Accept-Language"] = "en-US,en;q=0.9"

        return headers

    async def search(self, query: str, max_results: int = 10) -> list[Source]:
        cleaned_query = clean_text(query)

        if not cleaned_query:
            return []

        effective_query = self._build_search_query(cleaned_query)

        cache_key = build_cache_key(
            self.platform_name,
            "search_v3",
            cleaned_query,
            max_results,
        )

        cached = self._cache_get(cache_key)

        if cached is not None:
            return cached

        try:
            sources = await self._search_rest(effective_query, max_results)
        except Exception as exc:
            self._logger.warning(
                f"Wikipedia REST search failed, trying action API: {exc}"
            )
            sources = await self._search_action(effective_query, max_results)

        sources = self._filter_sources(sources, cleaned_query)

        if not sources and effective_query != cleaned_query:
            fallback_query = self._build_search_query(clean_topic(cleaned_query))

            try:
                fallback_sources = await self._search_rest(fallback_query, max_results)
            except Exception:
                fallback_sources = await self._search_action(fallback_query, max_results)

            sources = self._filter_sources(fallback_sources, cleaned_query)

        self._cache_set(cache_key, sources)
        return sources

    def _build_search_query(self, query: str) -> str:
        cleaned = strip_filler(clean_text(query))
        cleaned = re.sub(r"[^\w\s+#.-]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()

        raw_tokens = re.findall(r"[a-z0-9+#.]+", cleaned)
        generic_terms = get_generic_terms()

        tokens: list[str] = []

        for token in raw_tokens:
            if len(token) < 2:
                continue

            if token in generic_terms:
                continue

            if token not in tokens:
                tokens.append(token)

        if not tokens:
            return clean_text(query)[:80]

        return " ".join(tokens[:5])

    async def _search_rest(
        self,
        query: str,
        max_results: int,
    ) -> list[Source]:
        params = {
            "q": query,
            "limit": max_results,
        }

        payload = await self._get_json("search/page", params=params)
        self._raise_if_api_error(payload)

        pages = payload.get("pages", []) if isinstance(payload, dict) else []
        sources: list[Source] = []

        for page in pages[:max_results]:
            source = self._map_rest_page(page)

            if source is not None:
                sources.append(source)

        return sources

    async def _search_action(
        self,
        query: str,
        max_results: int,
    ) -> list[Source]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
            "srprop": "snippet|timestamp|wordcount|size",
            "format": "json",
            "formatversion": "2",
        }

        payload = await self._get_json(
            "https://en.wikipedia.org/w/api.php",
            params=params,
        )

        self._raise_if_api_error(payload)

        items = (
            (payload.get("query") or {}).get("search") or []
            if isinstance(payload, dict)
            else []
        )

        sources: list[Source] = []

        for item in items[:max_results]:
            source = self._map_action_item(item)

            if source is not None:
                sources.append(source)

        return sources

    def _raise_if_api_error(self, payload: Any) -> None:
        if isinstance(payload, dict) and payload.get("error"):
            error_payload = payload.get("error")

            error_details = (
                error_payload
                if isinstance(error_payload, dict)
                else {"error": error_payload}
            )

            raise SourceFetchError(
                "Wikipedia API returned error",
                details=error_details,
            )

    def _filter_sources(self, sources: list[Source], query: str) -> list[Source]:
        core_tokens = concept_tokens(query)
        phrases = self._filter_phrases(query)

        if not core_tokens and not phrases:
            return sources

        filtered: list[Source] = []

        for source in sources:
            metadata = source.metadata if isinstance(source.metadata, dict) else {}
            description = clean_text(metadata.get("description", ""))

            title_text = source.title.lower()
            full_text = f"{source.title} {description} {source.abstract or ''}".lower()

            phrase_title = False
            phrase_text = False

            for phrase in phrases:
                if phrase in title_text:
                    phrase_title = True

                if phrase in full_text:
                    phrase_text = True

            matched_total = 0
            matched_title = 0

            for token in core_tokens:
                if self._contains_token(full_text, token):
                    matched_total += 1

                if self._contains_token(title_text, token):
                    matched_title += 1

            if phrases and phrase_title:
                filtered.append(source)
                continue

            if phrases and phrase_text and matched_total >= max(2, len(core_tokens) // 2):
                filtered.append(source)
                continue

            if len(core_tokens) <= 1:
                if matched_title >= 1 or matched_total >= 1:
                    filtered.append(source)
                continue

            required = max(2, len(core_tokens) // 2)

            if matched_total >= required and matched_title >= 1:
                filtered.append(source)

        return filtered

    def _filter_phrases(self, query: str) -> list[str]:
        text = strip_filler(clean_text(query)).lower()
        text = re.sub(r"[^\w\s+#.-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        words = text.split()

        if len(words) < 2:
            return []

        generic_terms = get_generic_terms()
        meaningful = [word for word in words if word not in generic_terms and len(word) >= 2]

        if meaningful:
            words = meaningful

        phrases: set[str] = set()

        phrases.add(" ".join(words))

        for index in range(len(words) - 1):
            phrases.add(" ".join(words[index:index + 2]))

        for index in range(len(words) - 2):
            phrases.add(" ".join(words[index:index + 3]))

        cleaned: list[str] = []

        for phrase in phrases:
            normalized = re.sub(r"\s+", " ", phrase).strip()

            if len(normalized) < 4:
                continue

            if normalized not in cleaned:
                cleaned.append(normalized)

        return cleaned[:12]

    def _contains_token(self, text: str, token: str) -> bool:
        if not token:
            return False

        words = text.split()

        if token in words:
            return True

        pattern = r"(?:^|[^a-z0-9+#])" + re.escape(token) + r"(?:[^a-z0-9+#]|$)"

        return re.search(pattern, text) is not None

    def _map_rest_page(self, page: Any) -> Source | None:
        try:
            if not isinstance(page, dict):
                return None

            title = clean_text(page.get("title", ""))

            if not title:
                return None

            page_key = clean_text(page.get("key", "")) or title.replace(" ", "_")
            page_id = page.get("id")

            article_url = self._clean_url(
                f"https://en.wikipedia.org/wiki/{quote(page_key, safe='')}"
            )

            excerpt = clean_text(page.get("excerpt", ""))
            description = clean_text(page.get("description", ""))

            abstract_parts = [part for part in (excerpt, description) if part]
            abstract = truncate_text(
                " ".join(abstract_parts),
                max_length=1000,
                suffix="",
            ) or None

            return Source(
                source_id=self._make_source_id(
                    self.platform_name,
                    str(page_id or title),
                ),
                title=title,
                url=article_url,
                platform=SourcePlatform.WIKIPEDIA,
                source_type=SourceType.DOCUMENTATION,
                abstract=abstract,
                authors=[],
                published_at=None,
                year=None,
                citation_count=None,
                has_code=None,
                difficulty=None,
                score=None,
                metadata={
                    "page_id": page_id,
                    "page_key": page_key,
                    "description": description or None,
                },
            )
        except Exception as exc:
            self._logger.warning(f"Skipping Wikipedia REST page: {exc}")
            return None

    def _map_action_item(self, item: Any) -> Source | None:
        try:
            if not isinstance(item, dict):
                return None

            title = clean_text(item.get("title", ""))

            if not title:
                return None

            page_title = title.replace(" ", "_")

            article_url = self._clean_url(
                f"https://en.wikipedia.org/wiki/{quote(page_title, safe='')}"
            )

            snippet = clean_text(item.get("snippet", ""))
            timestamp = self._parse_datetime(item.get("timestamp"))
            page_id = item.get("pageid")

            return Source(
                source_id=self._make_source_id(
                    self.platform_name,
                    str(page_id or title),
                ),
                title=title,
                url=article_url,
                platform=SourcePlatform.WIKIPEDIA,
                source_type=SourceType.DOCUMENTATION,
                abstract=truncate_text(snippet, max_length=1000, suffix="") or None,
                authors=[],
                published_at=timestamp,
                year=timestamp.year if timestamp else None,
                citation_count=None,
                has_code=None,
                difficulty=None,
                score=None,
                metadata={
                    "page_id": page_id,
                    "word_count": item.get("wordcount"),
                    "size": item.get("size"),
                },
            )
        except Exception as exc:
            self._logger.warning(f"Skipping Wikipedia action item: {exc}")
            return None