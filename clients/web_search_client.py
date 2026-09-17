from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from clients.base_client import BaseHTTPClient
from core.models import Source, SourcePlatform, SourceType
from utils.cache import build_cache_key
from utils.text import clean_text, truncate_text

_RESULT_LINK_PATTERN = re.compile(
    r'<a[^>]*?class="[^"]*result__a[^"]*"[^>]*?href="([^"]+)"[^>]*?>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_RESULT_SNIPPET_PATTERN = re.compile(
    r'<a[^>]*?class="[^"]*result__snippet[^"]*"[^>]*?>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)


class WebSearchClient(BaseHTTPClient):
    platform_name = SourcePlatform.WEB.value

    def __init__(self) -> None:
        super().__init__(
            base_url="https://api.duckduckgo.com",
            rate_limit=5,
            period_seconds=60.0,
            cache_enabled=True,
            timeout=30.0,
            max_retries=2,
            retry_backoff_seconds=2.0,
        )

    async def search(self, query: str, max_results: int = 10) -> list[Source]:
        cleaned_query = " ".join(clean_text(query).split()[:8])
        max_results = min(max_results, 8)
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
        sources: list[Source] = []
        try:
            sources = await self._search_html(cleaned_query, max_results)
        except Exception as exc:
            self._logger.warning(f"Web HTML search failed, using fallback: {exc}")
        if not sources:
            try:
                sources = await self._search_instant_answer(cleaned_query, max_results)
            except Exception as exc:
                self._logger.warning(f"Web instant answer fallback failed: {exc}")
                return []
        sources = sources[:max_results]
        self._cache_set(cache_key, sources)
        return sources

    async def _search_html(self, query: str, max_results: int) -> list[Source]:
        html = await self._get_text(
            "https://html.duckduckgo.com/html/",
            params={"q": query, "kl": "us-en"},
            headers={
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        if not html:
            return []
        link_matches = _RESULT_LINK_PATTERN.findall(html)
        snippet_matches = _RESULT_SNIPPET_PATTERN.findall(html)
        if not link_matches:
            return []
        sources: list[Source] = []
        seen_urls: set[str] = set()
        for index, (href, raw_title) in enumerate(link_matches):
            if len(sources) >= max_results:
                break
            url = self._resolve_result_url(href)
            if not url or url in seen_urls:
                continue
            title = clean_text(raw_title)
            if not title:
                continue
            snippet = ""
            if index < len(snippet_matches):
                snippet = clean_text(snippet_matches[index])
            seen_urls.add(url)
            sources.append(
                Source(
                    source_id=self._make_source_id(self.platform_name, url),
                    title=truncate_text(title, max_length=200, suffix=""),
                    url=url,
                    platform=SourcePlatform.WEB,
                    source_type=SourceType.OTHER,
                    abstract=truncate_text(snippet, max_length=1000, suffix="")
                    or None,
                    authors=[],
                    published_at=None,
                    year=None,
                    citation_count=None,
                    has_code=None,
                    difficulty=None,
                    score=None,
                    metadata={"result_type": "web_search", "engine": "duckduckgo_html"},
                )
            )
        return sources

    def _resolve_result_url(self, href: Any) -> str:
        raw = str(href or "").strip()
        if not raw:
            return ""
        candidate = self._clean_url(raw)
        try:
            parsed = urlparse(candidate)
        except Exception:
            return ""
        host = parsed.netloc.lower()
        if "duckduckgo.com" in host:
            if "/y.js" in parsed.path or "ad_provider" in parsed.query:
                return ""
            try:
                query_params = parse_qs(parsed.query)
            except Exception:
                return ""
            redirect = query_params.get("uddg", [""])[0]
            if not redirect:
                return ""
            return self._clean_url(unquote(redirect))
        return candidate

    async def _search_instant_answer(
        self,
        query: str,
        max_results: int,
    ) -> list[Source]:
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }
        payload = await self._get_json("/", params=params)
        if not isinstance(payload, dict):
            return []
        sources: list[Source] = []
        abstract_source = self._map_abstract(payload)
        if abstract_source is not None:
            sources.append(abstract_source)
        for topic in payload.get("RelatedTopics", []) or []:
            if len(sources) >= max_results:
                break
            mapped = self._map_related_topic(topic)
            if mapped is not None:
                sources.append(mapped)
        for result in payload.get("Results", []) or []:
            if len(sources) >= max_results:
                break
            mapped = self._map_result(result)
            if mapped is not None:
                sources.append(mapped)
        return sources

    def _map_abstract(self, payload: dict) -> Source | None:
        try:
            abstract_text = clean_text(payload.get("AbstractText", ""))
            abstract_url = self._clean_url(payload.get("AbstractURL", ""))
            heading = clean_text(payload.get("Heading", ""))
            if not abstract_url or not (abstract_text or heading):
                return None
            return Source(
                source_id=self._make_source_id(self.platform_name, abstract_url),
                title=heading
                or truncate_text(abstract_text, max_length=100, suffix=""),
                url=abstract_url,
                platform=SourcePlatform.WEB,
                source_type=SourceType.DOCUMENTATION,
                abstract=truncate_text(abstract_text, max_length=1000, suffix="")
                or None,
                authors=[clean_text(payload.get("AbstractSource", ""))]
                if payload.get("AbstractSource")
                else [],
                published_at=None,
                year=None,
                citation_count=None,
                has_code=None,
                difficulty=None,
                score=None,
                metadata={"result_type": "instant_answer"},
            )
        except Exception as exc:
            self._logger.warning(f"Skipping web abstract: {exc}")
            return None

    def _map_related_topic(self, topic: dict) -> Source | None:
        try:
            if not isinstance(topic, dict):
                return None
            nested_topics = topic.get("Topics")
            if isinstance(nested_topics, list) and nested_topics:
                return self._map_related_topic(nested_topics[0])
            text = clean_text(topic.get("Text", ""))
            first_url = self._clean_url(topic.get("FirstURL", ""))
            if not first_url or not text:
                return None
            return Source(
                source_id=self._make_source_id(self.platform_name, first_url),
                title=truncate_text(text, max_length=120, suffix=""),
                url=first_url,
                platform=SourcePlatform.WEB,
                source_type=SourceType.OTHER,
                abstract=truncate_text(text, max_length=1000, suffix="") or None,
                authors=[],
                published_at=None,
                year=None,
                citation_count=None,
                has_code=None,
                difficulty=None,
                score=None,
                metadata={"result_type": "related_topic"},
            )
        except Exception as exc:
            self._logger.warning(f"Skipping web related topic: {exc}")
            return None

    def _map_result(self, result: dict) -> Source | None:
        try:
            if not isinstance(result, dict):
                return None
            text = clean_text(result.get("Text", ""))
            first_url = self._clean_url(result.get("FirstURL", ""))
            if not first_url or not text:
                return None
            return Source(
                source_id=self._make_source_id(self.platform_name, first_url),
                title=truncate_text(text, max_length=120, suffix=""),
                url=first_url,
                platform=SourcePlatform.WEB,
                source_type=SourceType.OTHER,
                abstract=truncate_text(text, max_length=1000, suffix="") or None,
                authors=[],
                published_at=None,
                year=None,
                citation_count=None,
                has_code=None,
                difficulty=None,
                score=None,
                metadata={"result_type": "instant_result"},
            )
        except Exception as exc:
            self._logger.warning(f"Skipping web result: {exc}")
            return None