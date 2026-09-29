from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from clients.base_client import BaseHTTPClient
from core import constants
from core.models import Source, SourcePlatform, SourceType
from search.query_tokens import get_generic_terms, tokenize
from utils.cache import build_cache_key
from utils.logger import get_logger, get_trace_logger
from utils.text import clean_text, truncate_text

_SOURCES_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "sources.yaml"
)

_DEFAULT_TIMEOUT_OVERRIDE = 45.0
_MIN_QUERY_TOKENS = 2
_MAX_QUERY_TOKENS = 6
_TARGET_QUERY_TOKENS = 4


class OpenAlexClient(BaseHTTPClient):
    platform_name = SourcePlatform.OPENALEX.value

    def __init__(self) -> None:
        per_platform_timeout = self._read_per_platform_timeout(
            default=_DEFAULT_TIMEOUT_OVERRIDE
        )

        super().__init__(
            base_url=constants.OPENALEX_BASE_URL,
            rate_limit=60,
            period_seconds=60.0,
            cache_enabled=True,
            timeout=per_platform_timeout,
            max_retries=2,
            retry_backoff_seconds=1.5,
        )

        self._trace = get_trace_logger()
        self._logger = get_logger(f"clients.{self.platform_name}")

    @staticmethod
    def _read_per_platform_timeout(default: float) -> float:
        try:
            import yaml

            if not _SOURCES_CONFIG_PATH.exists():
                return float(default)

            with open(_SOURCES_CONFIG_PATH, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}

            if not isinstance(loaded, dict):
                return float(default)

            platforms = loaded.get("platforms", {})

            if not isinstance(platforms, dict):
                return float(default)

            platform_config = platforms.get(SourcePlatform.OPENALEX.value, {})

            if not isinstance(platform_config, dict):
                return float(default)

            value = platform_config.get("per_platform_timeout_seconds")

            if value is None:
                return float(default)

            resolved = float(value)

            if resolved <= 0.0:
                return float(default)

            return resolved
        except Exception:
            return float(default)

    async def search(self, query: str, max_results: int = 10) -> list[Source]:
        cleaned_query = clean_text(query)

        if not cleaned_query:
            return []

        simplified_query = self._simplify_query(cleaned_query)

        if not simplified_query:
            return []

        if simplified_query != cleaned_query:
            self._trace.emit(
                "openalex_query_simplified",
                original_query=cleaned_query,
                simplified_query=simplified_query,
                original_tokens=len(tokenize(cleaned_query)),
                simplified_tokens=len(tokenize(simplified_query)),
            )
        else:
            self._trace.emit(
                "openalex_query_built",
                query=cleaned_query,
                tokens=len(tokenize(cleaned_query)),
            )

        cache_key = build_cache_key(
            self.platform_name,
            "search_v2",
            simplified_query,
            max_results,
        )

        cached = self._cache_get(cache_key)

        if cached is not None:
            return cached

        params = {
            "search": simplified_query,
            "per-page": max_results,
            "select": (
                "id,doi,title,abstract_inverted_index,publication_year,"
                "publication_date,cited_by_count,authorships,primary_location"
            ),
        }

        contact = os.getenv("RESEARCH_AGENT_CONTACT", "").strip()

        if contact:
            params["mailto"] = contact

        payload = await self._get_json("/works", params=params)
        results = payload.get("results", []) if isinstance(payload, dict) else []

        sources: list[Source] = []

        for item in results[:max_results]:
            source = self._map_item(item)

            if source is not None:
                sources.append(source)

        if sources:
            self._cache_set(cache_key, sources)

        self._trace.emit(
            "openalex_result",
            query=simplified_query,
            sources_found=len(sources),
        )

        return sources

    def _simplify_query(self, query: str) -> str:
        tokens = tokenize(query)

        if not tokens:
            return ""

        generics = get_generic_terms()

        meaningful: list[str] = []

        for token in tokens:
            if len(token) < 3:
                continue

            if token in generics:
                continue

            if token not in meaningful:
                meaningful.append(token)

        if len(meaningful) < _MIN_QUERY_TOKENS:
            fallback = " ".join(tokens[: _MAX_QUERY_TOKENS])
            return fallback.strip()

        if len(meaningful) <= _MAX_QUERY_TOKENS:
            return " ".join(meaningful)

        ranked = sorted(
            meaningful,
            key=lambda item: (-len(item), item),
        )

        selected = ranked[:_TARGET_QUERY_TOKENS]

        return " ".join(selected)

    def _reconstruct_abstract(self, inverted_index: dict | None) -> str | None:
        if not inverted_index or not isinstance(inverted_index, dict):
            return None

        word_positions: list[tuple[int, str]] = []

        for word, positions in inverted_index.items():
            if not isinstance(positions, list):
                continue

            for position in positions:
                try:
                    word_positions.append((int(position), str(word)))
                except Exception:
                    continue

        if not word_positions:
            return None

        word_positions.sort(key=lambda item: item[0])

        return " ".join(word for _, word in word_positions)

    def _map_item(self, item: dict) -> Source | None:
        try:
            title = clean_text(item.get("title", ""))

            if not title:
                return None

            location = item.get("primary_location") or {}
            landing_url = (
                location.get("landing_page_url")
                if isinstance(location, dict)
                else None
            )

            doi = item.get("doi")
            final_url = self._clean_url(
                landing_url or doi or item.get("id") or ""
            )

            if not final_url:
                return None

            authors: list[str] = []

            for authorship in item.get("authorships", []) or []:
                author = authorship.get("author") or {}
                name = clean_text(author.get("display_name", ""))

                if name:
                    authors.append(name)

            publication_date = self._parse_datetime(item.get("publication_date"))
            year = item.get("publication_year") or (
                publication_date.year if publication_date else None
            )

            abstract = self._reconstruct_abstract(
                item.get("abstract_inverted_index")
            )

            cited_by_count = item.get("cited_by_count")

            return Source(
                source_id=self._make_source_id(
                    self.platform_name,
                    str(item.get("id") or final_url),
                ),
                title=title,
                url=final_url,
                platform=SourcePlatform.OPENALEX,
                source_type=SourceType.PAPER,
                abstract=truncate_text(
                    clean_text(abstract or ""),
                    max_length=2000,
                    suffix="",
                )
                or None,
                authors=authors,
                published_at=publication_date,
                year=self._parse_year(year),
                citation_count=int(cited_by_count)
                if isinstance(cited_by_count, int)
                else None,
                has_code=None,
                difficulty=None,
                score=None,
                metadata={
                    "openalex_id": item.get("id"),
                    "doi": doi,
                },
            )
        except Exception as exc:
            self._logger.warning(f"Skipping OpenAlex item: {exc}")
            return None