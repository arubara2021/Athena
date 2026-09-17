from __future__ import annotations

from core.config import get_settings
from core import constants
from clients.base_client import BaseHTTPClient
from core.models import Source, SourcePlatform, SourceType
from utils.cache import build_cache_key
from utils.text import clean_text, truncate_text


class HuggingFaceClient(BaseHTTPClient):
    platform_name = SourcePlatform.HUGGINGFACE.value

    def __init__(self) -> None:
        super().__init__(
            base_url=constants.HUGGINGFACE_BASE_URL,
            rate_limit=60,
            period_seconds=60.0,
            cache_enabled=True,
        )

    def _build_default_headers(self) -> dict[str, str]:
        headers = dict(constants.DEFAULT_HEADERS)
        try:
            settings = get_settings()
            token = settings.huggingface_token.get_secret_value().strip()
            if token:
                headers["Authorization"] = f"Bearer {token}"
        except Exception:
            pass
        return headers

    async def search(self, query: str, max_results: int = 10) -> list[Source]:
        cleaned_query = clean_text(query)
        if not cleaned_query:
            return []

        cache_key = build_cache_key(self.platform_name, "search", cleaned_query, max_results)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        params = {
            "search": cleaned_query,
            "limit": max_results,
            "sort": "downloads",
            "direction": "-1",
            "full": True,
        }

        payload = await self._get_json("/models", params=params)
        items = payload if isinstance(payload, list) else []

        sources: list[Source] = []
        for item in items[:max_results]:
            source = self._map_item(item)
            if source is not None:
                sources.append(source)

        self._cache_set(cache_key, sources)
        return sources

    def _map_item(self, item: dict) -> Source | None:
        try:
            model_id = clean_text(item.get("modelId") or item.get("id") or "")
            if not model_id:
                return None

            model_url = self._clean_url(f"https://huggingface.co/{model_id}")
            pipeline_tag = item.get("pipeline_tag")
            downloads = item.get("downloads")
            likes = item.get("likes")
            tags = item.get("tags", []) or []
            last_modified = self._parse_datetime(item.get("lastModified"))

            return Source(
                source_id=self._make_source_id(self.platform_name, model_id),
                title=model_id,
                url=model_url,
                platform=SourcePlatform.HUGGINGFACE,
                source_type=SourceType.MODEL,
                abstract=truncate_text(
                    clean_text(item.get("description", "") or ""),
                    max_length=1000,
                    suffix="",
                ) or None,
                authors=[],
                published_at=last_modified,
                year=last_modified.year if last_modified else None,
                citation_count=None,
                has_code=None,
                difficulty=None,
                score=None,
                metadata={
                "model_id": model_id,
                "author": clean_text(item.get("author", "") or ""),
                "pipeline_tag": pipeline_tag,
                "downloads": int(downloads) if isinstance(downloads, int) else None,
                "likes": int(likes) if isinstance(likes, int) else None,
                "tags": tags,
            },
            )
        except Exception as exc:
            self._logger.warning(f"Skipping HuggingFace item: {exc}")
            return None