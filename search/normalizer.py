from __future__ import annotations

from datetime import datetime
from typing import Any

from core.models import Source, SourcePlatform, SourceType
from utils.hashing import stable_hash
from utils.logger import get_logger
from utils.text import clean_text, truncate_text
from utils.url import get_url_fingerprint, is_valid_url, normalize_url


class SourceNormalizer:
    def __init__(
        self,
        max_abstract_length: int = 2000,
        max_metadata_keys: int = 20,
        max_authors: int = 50,
    ) -> None:
        self._max_abstract_length = max(100, max_abstract_length)
        self._max_metadata_keys = max(1, max_metadata_keys)
        self._max_authors = max(1, max_authors)
        self._logger = get_logger("search.normalizer")

    def normalize_sources(self, sources: list[Any]) -> list[Source]:
        normalized: list[Source] = []
        for source in sources:
            normalized_source = self.normalize_source(source)
            if normalized_source is not None:
                normalized.append(normalized_source)
        return normalized

    def normalize_source(self, source: Any) -> Source | None:
        try:
            if isinstance(source, dict):
                source = Source.model_validate(source)
            if not isinstance(source, Source):
                return None

            title = clean_text(source.title)
            if len(title) < 3:
                return None

            url = self._normalize_url(source.url)
            if not url or not is_valid_url(url):
                return None

            platform = SourcePlatform(self._enum_value(source.platform))
            source_type = SourceType(self._enum_value(source.source_type))

            abstract = None
            if source.abstract:
                abstract = truncate_text(
                    clean_text(source.abstract),
                    max_length=self._max_abstract_length,
                    suffix="",
                ) or None

            authors = self._normalize_authors(source.authors)
            year = self._normalize_year(source.year)
            citation_count = self._normalize_citation_count(source.citation_count)
            metadata = self._sanitize_metadata(source.metadata)
            published_at = source.published_at if isinstance(source.published_at, datetime) else None

            source_id = self._build_source_id(
                platform=platform.value,
                url=url,
                title=title,
            )

            return Source(
                source_id=source_id,
                title=title,
                url=url,
                platform=platform,
                source_type=source_type,
                abstract=abstract,
                summary=source.summary,
                authors=authors,
                published_at=published_at,
                year=year,
                citation_count=citation_count,
                has_code=source.has_code,
                difficulty=source.difficulty,
                score=source.score,
                metadata=metadata,
            )
        except Exception as exc:
            self._logger.warning(f"Skipping source during normalization: {exc}")
            return None

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value))

    def _normalize_url(self, value: Any) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        try:
            return normalize_url(raw)
        except Exception:
            return raw

    def _normalize_authors(self, authors: Any) -> list[str]:
        if not isinstance(authors, list):
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for author in authors:
            name = clean_text(author)
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(name)
            if len(normalized) >= self._max_authors:
                break
        return normalized

    def _normalize_year(self, value: Any) -> int | None:
        if value is None:
            return None
        try:
            year = int(value)
        except Exception:
            return None
        if 0 <= year <= 2100:
            return year
        return None

    def _normalize_citation_count(self, value: Any) -> int | None:
        if value is None:
            return None
        try:
            citation_count = int(value)
        except Exception:
            return None
        if citation_count >= 0:
            return citation_count
        return None

    def _build_source_id(self, platform: str, url: str, title: str) -> str:
        url_key = get_url_fingerprint(url) if is_valid_url(url) else url.lower()
        title_key = " ".join(title.lower().split())
        return stable_hash(
            {
                "platform": platform,
                "url": url_key,
                "title": title_key,
            }
        )

    def _sanitize_metadata(self, metadata: Any) -> dict[str, Any]:
        if not isinstance(metadata, dict):
            return {}
        sanitized: dict[str, Any] = {}
        for key, value in list(metadata.items())[: self._max_metadata_keys]:
            cleaned_key = clean_text(key)
            if not cleaned_key:
                continue
            sanitized[cleaned_key] = self._sanitize_value(value)
        return sanitized

    def _sanitize_value(self, value: Any, depth: int = 0) -> Any:
        if depth > 3:
            return str(value)
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return truncate_text(clean_text(value), max_length=500, suffix="")
        if isinstance(value, (list, tuple, set)):
            return [self._sanitize_value(item, depth + 1) for item in list(value)[:20]]
        if isinstance(value, dict):
            return {
                clean_text(key): self._sanitize_value(item, depth + 1)
                for key, item in list(value.items())[:20]
                if clean_text(key)
            }
        if hasattr(value, "value"):
            return str(value.value)
        return str(value)