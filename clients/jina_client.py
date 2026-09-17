from __future__ import annotations

from typing import Any

from clients.base_client import BaseHTTPClient
from core.config import get_settings
from core.models import Source, SourcePlatform, SourceType
from utils.cache import build_cache_key
from utils.text import clean_text, truncate_text


class JinaClient(BaseHTTPClient):
    platform_name = SourcePlatform.JINA.value

    def __init__(self) -> None:
        settings = get_settings()
        api_key = settings.jina_api_key.get_secret_value().strip()
        super().__init__(
            base_url="https://s.jina.ai",
            rate_limit=20,
            period_seconds=60.0,
            cache_enabled=True,
            timeout=45.0,
            max_retries=2,
            retry_backoff_seconds=2.0,
        )
        self._api_key = api_key

    def _build_default_headers(self) -> dict[str, str]:
        headers = super()._build_default_headers()
        headers["Accept"] = "application/json"
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def search(self, query: str, max_results: int = 10) -> list[Source]:
        cleaned_query = clean_text(query)
        if not cleaned_query:
            return []

        if not self._api_key:
            self._logger.warning("Jina API key is not configured")
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

        try:
            data = await self._get_json(
                cleaned_query,
                headers=self._build_default_headers(),
            )
        except Exception as exc:
            self._logger.warning(f"Jina search failed: {exc}")
            return []

        results = data.get("data", []) if isinstance(data, dict) else []
        sources: list[Source] = []
        for item in results[:max_results]:
            source = self._map_result(item, cleaned_query)
            if source is not None:
                sources.append(source)

        self._cache_set(cache_key, sources)
        return sources

    async def read_url(self, url: str) -> Source | None:
        cleaned_url = clean_text(url)
        if not cleaned_url:
            return None

        if not self._api_key:
            self._logger.warning("Jina API key is not configured")
            return None

        cache_key = build_cache_key(
            self.platform_name,
            "read_url",
            cleaned_url,
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached[0] if cached else None

        try:
            headers = self._build_default_headers()
            response = await self._request(
                f"https://r.jina.ai/{cleaned_url}",
                headers=headers,
            )
            data = response.json()
        except Exception as exc:
            self._logger.warning(f"Jina read_url failed: {exc}")
            return None

        if not isinstance(data, dict):
            return None

        content_data = data.get("data", data)
        if not isinstance(content_data, dict):
            return None

        title = clean_text(content_data.get("title", ""))
        if not title:
            title = clean_text(cleaned_url)

        content = clean_text(content_data.get("content", ""))
        description = clean_text(content_data.get("description", ""))
        abstract_text = description or content
        abstract = truncate_text(abstract_text, max_length=2000, suffix="") or None

        final_url = clean_text(content_data.get("url", cleaned_url))

        source = Source(
            source_id=self._make_source_id(self.platform_name, final_url),
            title=title,
            url=final_url,
            platform=SourcePlatform.JINA,
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
                "search_provider": "jina",
                "content_length": len(content),
                "full_content": truncate_text(content, max_length=10000, suffix=""),
            },
        )

        self._cache_set(cache_key, [source])
        return source

    def _map_result(self, item: Any, query: str) -> Source | None:
        try:
            if not isinstance(item, dict):
                return None

            title = clean_text(item.get("title", ""))
            url = clean_text(item.get("url", ""))
            if not title or not url:
                return None

            description = clean_text(item.get("description", ""))
            content = clean_text(item.get("content", ""))
            abstract_text = description or content
            abstract = truncate_text(abstract_text, max_length=2000, suffix="") or None

            return Source(
                source_id=self._make_source_id(self.platform_name, url),
                title=title,
                url=url,
                platform=SourcePlatform.JINA,
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
                    "search_provider": "jina",
                    "query": query,
                    "content_length": len(content),
                },
            )
        except Exception as exc:
            self._logger.warning(f"Skipping Jina result: {exc}")
            return None