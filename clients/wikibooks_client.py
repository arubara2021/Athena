from __future__ import annotations
from typing import Any
from urllib.parse import quote
from clients.base_client import BaseHTTPClient
from core.models import Source, SourcePlatform, SourceType
from utils.cache import build_cache_key
from utils.text import clean_text, truncate_text

class WikibooksClient(BaseHTTPClient):
    platform_name = SourcePlatform.WIKIBOOKS.value

    def __init__(self) -> None:
        super().__init__(
            base_url="https://en.wikibooks.org",
            rate_limit=8,
            period_seconds=60.0,
            cache_enabled=True,
            timeout=30.0,
            max_retries=2,
            retry_backoff_seconds=2.0,
        )

    def _build_default_headers(self) -> dict[str, str]:
        headers = super()._build_default_headers()
        headers["Api-User-Agent"] = "AthenaResearchAgent (https://github.com/arubara2021/Athena)"
        return headers

    async def search(self, query: str, max_results: int = 10) -> list[Source]:
        cleaned_query = clean_text(query)
        if not cleaned_query:
            return []
        cache_key = build_cache_key(self.platform_name, "search", cleaned_query, max_results)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        sources = await self._search_rest(cleaned_query, max_results)
        if not sources:
            sources = await self._search_action(cleaned_query, max_results)
        self._cache_set(cache_key, sources)
        return sources

    async def _search_rest(self, query: str, max_results: int) -> list[Source]:
        params = {"q": query, "limit": max_results}
        try:
            payload = await self._get_json("/w/rest.php/v1/search/page", params=params)
        except Exception:
            return []
        pages = payload.get("pages", []) if isinstance(payload, dict) else []
        sources: list[Source] = []
        for page in pages[:max_results]:
            source = self._map_rest_page(page)
            if source:
                sources.append(source)
        return sources

    async def _search_action(self, query: str, max_results: int) -> list[Source]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
            "srprop": "snippet",
            "format": "json",
            "formatversion": "2",
        }
        try:
            payload = await self._get_json("/w/api.php", params=params)
        except Exception:
            return []
        items = (payload.get("query") or {}).get("search") or [] if isinstance(payload, dict) else []
        sources: list[Source] = []
        for item in items[:max_results]:
            source = self._map_action_item(item)
            if source:
                sources.append(source)
        return sources

    def _map_rest_page(self, page: Any) -> Source | None:
        try:
            if not isinstance(page, dict):
                return None
            title = clean_text(page.get("title", ""))
            if not title:
                return None
            page_key = clean_text(page.get("key", "")) or title.replace(" ", "_")
            article_url = self._clean_url(f"https://en.wikibooks.org/wiki/{quote(page_key, safe='')}")
            excerpt = clean_text(page.get("excerpt", ""))
            description = clean_text(page.get("description", ""))
            abstract_parts = [part for part in (excerpt, description) if part]
            abstract = truncate_text(" ".join(abstract_parts), max_length=1000, suffix="") or None
            return Source(
                source_id=self._make_source_id(self.platform_name, str(page.get("id") or title)),
                title=title,
                url=article_url,
                platform=SourcePlatform.WIKIBOOKS,
                source_type=SourceType.DOCUMENTATION,
                abstract=abstract,
                authors=[],
                published_at=None,
                year=None,
                citation_count=None,
                has_code=None,
                difficulty=None,
                score=None,
                metadata={"page_id": page.get("id"), "description": description or None},
            )
        except Exception:
            return None

    def _map_action_item(self, item: Any) -> Source | None:
        try:
            if not isinstance(item, dict):
                return None
            title = clean_text(item.get("title", ""))
            if not title:
                return None
            page_title = title.replace(" ", "_")
            article_url = self._clean_url(f"https://en.wikibooks.org/wiki/{quote(page_title, safe='')}")
            snippet = clean_text(item.get("snippet", ""))
            abstract = truncate_text(snippet, max_length=1000, suffix="") or None
            return Source(
                source_id=self._make_source_id(self.platform_name, str(item.get("pageid") or title)),
                title=title,
                url=article_url,
                platform=SourcePlatform.WIKIBOOKS,
                source_type=SourceType.DOCUMENTATION,
                abstract=abstract,
                authors=[],
                published_at=None,
                year=None,
                citation_count=None,
                has_code=None,
                difficulty=None,
                score=None,
                metadata={"page_id": item.get("pageid")},
            )
        except Exception:
            return None