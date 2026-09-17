from __future__ import annotations

from typing import Any

from clients.base_client import BaseHTTPClient
from core.config import get_settings
from core.models import Source, SourcePlatform, SourceType
from utils.cache import build_cache_key
from utils.text import clean_text, truncate_text


class CoreClient(BaseHTTPClient):
    platform_name = SourcePlatform.CORE.value

    def __init__(self) -> None:
        settings = get_settings()
        api_key = settings.core_api_key.get_secret_value().strip()
        super().__init__(
            base_url="https://api.core.ac.uk/v3",
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
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def search(self, query: str, max_results: int = 10) -> list[Source]:
        cleaned_query = clean_text(query)
        if not cleaned_query:
            return []

        if not self._api_key:
            self._logger.warning("CORE API key is not configured")
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
            "limit": min(max_results, 50),
        }

        try:
            data = await self._get_json(
                "search/works",
                params=params,
                headers=self._build_default_headers(),
            )
        except Exception as exc:
            self._logger.warning(f"CORE search failed: {exc}")
            return []

        results = data.get("results", []) if isinstance(data, dict) else []
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
            if not title:
                return None

            identifiers = item.get("identifiers", []) or []
            url = ""
            for identifier in identifiers:
                if isinstance(identifier, str) and identifier.startswith("http"):
                    url = identifier
                    break

            if not url:
                links = item.get("links", []) or []
                for link in links:
                    if isinstance(link, dict):
                        link_url = clean_text(link.get("url", ""))
                        if link_url:
                            url = link_url
                            break

            if not url:
                url = f"https://core.ac.uk/works/{item.get('id', '')}"

            abstract = clean_text(item.get("abstract", "")) or None
            if abstract:
                abstract = truncate_text(abstract, max_length=2000, suffix="")

            authors = []
            raw_authors = item.get("authors", []) or []
            for author in raw_authors:
                if isinstance(author, str) and author.strip():
                    authors.append(author.strip())
                elif isinstance(author, dict):
                    name = clean_text(author.get("name", ""))
                    if name:
                        authors.append(name)

            year = self._parse_year(item.get("yearPublished"))
            published_at = self._parse_datetime(item.get("publishedDate"))

            return Source(
                source_id=self._make_source_id(self.platform_name, str(item.get("id", url))),
                title=title,
                url=url,
                platform=SourcePlatform.CORE,
                source_type=SourceType.PAPER,
                abstract=abstract,
                authors=authors,
                published_at=published_at,
                year=year,
                citation_count=None,
                has_code=None,
                difficulty=None,
                score=None,
                metadata={
                    "search_provider": "core",
                    "query": query,
                    "core_id": item.get("id"),
                },
            )
        except Exception as exc:
            self._logger.warning(f"Skipping CORE result: {exc}")
            return None