from __future__ import annotations

from typing import Any

from clients.base_client import BaseHTTPClient
from core.config import get_settings
from core.models import Source, SourcePlatform, SourceType
from utils.cache import build_cache_key
from utils.text import clean_text, truncate_text


class SerpApiClient(BaseHTTPClient):
    platform_name = SourcePlatform.SERPAPI.value

    def __init__(self) -> None:
        settings = get_settings()
        api_key = settings.serpapi_api_key.get_secret_value().strip()
        super().__init__(
            base_url="https://serpapi.com",
            rate_limit=10,
            period_seconds=60.0,
            cache_enabled=True,
            timeout=30.0,
            max_retries=2,
            retry_backoff_seconds=2.0,
        )
        self._api_key = api_key

    async def search(self, query: str, max_results: int = 10) -> list[Source]:
        cleaned_query = clean_text(query)
        if not cleaned_query:
            return []

        if not self._api_key:
            self._logger.warning("SerpAPI key is not configured")
            return []

        cache_key = build_cache_key(
            self.platform_name,
            "search",
            cleaned_query,
            max_results,
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        params = {
            "q": cleaned_query,
            "api_key": self._api_key,
            "engine": "google",
            "num": min(max_results, 20),
        }

        try:
            data = await self._get_json(
                "search.json",
                params=params,
            )
        except Exception as exc:
            self._logger.warning(f"SerpAPI search failed: {exc}")
            return []

        results = data.get("organic_results", []) if isinstance(data, dict) else []
        sources: list[Source] = []
        for item in results[:max_results]:
            source = self._map_result(item, cleaned_query)
            if source is not None:
                sources.append(source)

        self._cache_set(cache_key, sources)
        return sources

    def _map_result(self, item: Any, query: str) -> Source | None:
        try:
            if not isinstance(item, dict):
                return None

            title = clean_text(item.get("title", ""))
            url = clean_text(item.get("link", ""))
            if not title or not url:
                return None

            snippet = clean_text(item.get("snippet", ""))
            abstract = truncate_text(snippet, max_length=2000, suffix="") or None

            return Source(
                source_id=self._make_source_id(self.platform_name, url),
                title=title,
                url=url,
                platform=SourcePlatform.SERPAPI,
                source_type=SourceType.BLOG,
                abstract=abstract,
                authors=[],
                published_at=None,
                year=None,
                citation_count=None,
                has_code=None,
                difficulty=None,
                score=None,
                metadata={
                    "search_provider": "serpapi",
                    "query": query,
                    "position": item.get("position"),
                },
            )
        except Exception as exc:
            self._logger.warning(f"Skipping SerpAPI result: {exc}")
            return None