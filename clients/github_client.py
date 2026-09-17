from __future__ import annotations

from core.config import get_settings
from core import constants
from clients.base_client import BaseHTTPClient
from core.models import Source, SourcePlatform, SourceType
from utils.cache import build_cache_key
from utils.text import clean_text, truncate_text


class GithubClient(BaseHTTPClient):
    platform_name = SourcePlatform.GITHUB.value

    def __init__(self) -> None:
        try:
            settings = get_settings()
            self._token = settings.github_token.get_secret_value().strip()
        except Exception:
            self._token = ""

        rate_limit = 1 if not self._token else 10

        super().__init__(
            base_url=constants.GITHUB_BASE_URL,
            rate_limit=rate_limit,
            period_seconds=60.0,
            cache_enabled=True,
        )

    def _build_default_headers(self) -> dict[str, str]:
        headers = {
            **constants.DEFAULT_HEADERS,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        try:
            settings = get_settings()
            token = settings.github_token.get_secret_value().strip()
            if self._token:
                headers["Authorization"] = f"Bearer {self._token}"
        except Exception:
            pass
        return headers

    async def search(self, query: str, max_results: int = 10) -> list[Source]:
        cleaned_query = " ".join(clean_text(query).split()[:8])
        if not cleaned_query:
            return []

        cache_key = build_cache_key(self.platform_name, "search", cleaned_query, max_results)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        params = {
            "q": cleaned_query,
            "sort": "stars",
            "order": "desc",
            "per_page": max_results,
        }

        payload = await self._get_json("/search/repositories", params=params)
        items = payload.get("items", []) if isinstance(payload, dict) else []

        sources: list[Source] = []
        for item in items[:max_results]:
            source = self._map_item(item)
            if source is not None:
                sources.append(source)

        self._cache_set(cache_key, sources)
        return sources

    def _map_item(self, item: dict) -> Source | None:
        try:
            full_name = clean_text(item.get("full_name", ""))
            html_url = self._clean_url(item.get("html_url", ""))

            if not full_name or not html_url:
                return None

            description = clean_text(item.get("description", "") or "")
            owner = (item.get("owner") or {}).get("login", "")
            stars = item.get("stargazers_count")
            language = item.get("language")
            topics = item.get("topics", []) or []
            updated_at = self._parse_datetime(item.get("updated_at"))

            return Source(
                source_id=self._make_source_id(self.platform_name, full_name),
                title=full_name,
                url=html_url,
                platform=SourcePlatform.GITHUB,
                source_type=SourceType.REPOSITORY,
                abstract=truncate_text(description, max_length=1000, suffix="") or None,
                authors=[clean_text(owner)] if owner else [],
                published_at=updated_at,
                year=updated_at.year if updated_at else None,
                citation_count=None,
                has_code=True,
                difficulty=None,
                score=None,
                metadata={
                    "stars": int(stars) if isinstance(stars, int) else None,
                    "language": language,
                    "topics": topics,
                    "fork": bool(item.get("fork")),
                    "archived": bool(item.get("archived")),
                },
            )
        except Exception as exc:
            self._logger.warning(f"Skipping GitHub item: {exc}")
            return None