from __future__ import annotations

from typing import Any

from clients.base_client import BaseHTTPClient
from core.config import get_settings
from core.models import Source, SourcePlatform, SourceType
from utils.cache import build_cache_key
from utils.text import clean_text, truncate_text


class SerperClient(BaseHTTPClient):
    platform_name = SourcePlatform.SERPER.value

    def __init__(self) -> None:
        settings = get_settings()
        api_key = settings.serper_api_key.get_secret_value().strip()
        super().__init__(
            base_url="https://google.serper.dev",
            rate_limit=10,
            period_seconds=60.0,
            cache_enabled=True,
            timeout=30.0,
            max_retries=2,
            retry_backoff_seconds=2.0,
        )
        self._api_key = api_key

    def _build_default_headers(self) -> dict[str, str]:
        headers = super()._build_default_headers()
        headers["Content-Type"] = "application/json"
        if self._api_key:
            headers["X-API-KEY"] = self._api_key
        return headers

    async def search(self, query: str, max_results: int = 10) -> list[Source]:
        cleaned_query = clean_text(query)
        if not cleaned_query:
            return []

        if not self._api_key:
            self._logger.warning("Serper API key is not configured")
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

        payload = {
            "q": cleaned_query,
            "num": min(max_results, 20),
        }

        try:
            client = await self._get_client()
            response = await client.post(
                f"{self._base_url}/search",
                json=payload,
                headers=self._build_default_headers(),
                timeout=self._timeout,
            )
            if response.status_code != 200:
                self._logger.warning(
                    f"Serper search failed with status {response.status_code}"
                )
                return []
            data = response.json()
        except Exception as exc:
            self._logger.warning(f"Serper search failed: {exc}")
            return []

        results = data.get("organic", []) if isinstance(data, dict) else []
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
                platform=SourcePlatform.SERPER,
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
                    "search_provider": "serper",
                    "query": query,
                    "position": item.get("position"),
                },
            )
        except Exception as exc:
            self._logger.warning(f"Skipping Serper result: {exc}")
            return None