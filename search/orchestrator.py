from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from core.exceptions import SearchError
from core.models import Source
from core.schemas import SearchQuerySchema
from search.dedupe import SourceDeduplicator
from search.normalizer import SourceNormalizer
from search.query_expander import QueryExpander
from search.source_fetcher import SourceFetcher
from search.validator import SourceValidator
from utils.logger import get_logger, get_trace_logger
from utils.text import clean_text


class SearchOrchestrationResult(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        extra="ignore",
    )

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    query: SearchQuerySchema
    expanded_queries: list[str] = Field(default_factory=list)
    corrected_topic: str = ""
    primary_concept: str = ""
    keywords: list[str] = Field(default_factory=list)
    intent: str = ""
    detected_level: str = ""
    short_search_queries: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    total_found: int = 0
    total_normalized: int = 0
    total_deduplicated: int = 0
    total_valid: int = 0
    errors: list[str] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime | None = None
    latency_ms: float = 0.0


class SearchOrchestrator:
    def __init__(
        self,
        use_llm_expansion: bool = False,
        query_expander: QueryExpander | None = None,
        source_fetcher: SourceFetcher | None = None,
        normalizer: SourceNormalizer | None = None,
        deduplicator: SourceDeduplicator | None = None,
        validator: SourceValidator | None = None,
    ) -> None:
        self._expander = query_expander or QueryExpander(use_llm=use_llm_expansion)
        self._fetcher = source_fetcher or SourceFetcher()
        self._owns_fetcher = source_fetcher is None
        self._normalizer = normalizer or SourceNormalizer()
        self._deduplicator = deduplicator or SourceDeduplicator()
        self._validator = validator or SourceValidator()
        self._entered = False
        self._logger = get_logger("search.orchestrator")
        self._trace = get_trace_logger()

    async def __aenter__(self) -> SearchOrchestrator:
        self._entered = True
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        await self.close()
        self._entered = False
        return False

    async def close(self) -> None:
        if self._owns_fetcher:
            await self._fetcher.close()

    async def run(
        self,
        query: SearchQuerySchema | str | dict[str, Any],
    ) -> SearchOrchestrationResult:
        started_at = datetime.now(timezone.utc)
        start = time.perf_counter()

        errors: list[str] = []

        query_schema = self._coerce_query(query)

        result = SearchOrchestrationResult(
            query=query_schema,
            started_at=started_at,
        )

        corrected_topic = clean_text(query_schema.topic)
        primary_concept = corrected_topic
        keywords: list[str] = []
        intent = ""
        detected_level = clean_text(query_schema.level)
        short_search_queries: list[str] = []
        expanded_queries = [corrected_topic]

        self._trace.emit(
            "search_started",
            topic=query_schema.topic,
            platforms=[
                p.value if hasattr(p, "value") else str(p)
                for p in query_schema.platforms
            ] if query_schema.platforms else "all",
            max_results=query_schema.max_results,
        )

        try:
            try:
                expand_start = time.perf_counter()

                expansion_result = await self._expander.expand(query_schema)

                expanded_queries = expansion_result.queries or [corrected_topic]

                corrected_topic = clean_text(
                    getattr(expansion_result, "corrected_topic", "")
                    or corrected_topic
                )

                primary_concept = clean_text(
                    getattr(expansion_result, "primary_concept", "")
                    or corrected_topic
                )

                keywords = self._normalize_string_list(
                    getattr(expansion_result, "keywords", [])
                )

                intent = clean_text(getattr(expansion_result, "intent", ""))

                detected_level = clean_text(
                    getattr(expansion_result, "detected_level", "")
                    or detected_level
                )

                short_search_queries = self._normalize_string_list(
                    getattr(expansion_result, "short_search_queries", [])
                )

                expand_ms = (time.perf_counter() - expand_start) * 1000

                if corrected_topic and corrected_topic != query_schema.topic:
                    self._trace.emit(
                        "search_query_corrected",
                        original_topic=query_schema.topic,
                        corrected_topic=corrected_topic,
                        primary_concept=primary_concept,
                        keywords=keywords,
                        intent=intent,
                        detected_level=detected_level,
                    )

                self._trace.emit(
                    "search_query_expansion",
                    topic=corrected_topic,
                    expanded_count=len(expanded_queries),
                    queries=expanded_queries,
                    latency_ms=round(expand_ms, 1),
                )
            except Exception as exc:
                expanded_queries = [query_schema.topic]
                corrected_topic = clean_text(query_schema.topic)
                primary_concept = corrected_topic
                keywords = []
                intent = ""
                detected_level = clean_text(query_schema.level)
                short_search_queries = []

                errors.append(f"query_expansion_failed: {exc}")

                self._trace.emit(
                    "search_query_expansion_failed",
                    topic=query_schema.topic,
                    error=str(exc),
                )

            try:
                query_schema = query_schema.model_copy(
                    update={
                        "topic": corrected_topic,
                        "corrected_topic": corrected_topic,
                        "primary_concept": primary_concept,
                        "keywords": keywords,
                        "intent": intent,
                        "detected_level": detected_level,
                        "short_search_queries": short_search_queries,
                    }
                )
            except Exception:
                pass

            result.expanded_queries = expanded_queries
            result.corrected_topic = corrected_topic
            result.primary_concept = primary_concept
            result.keywords = keywords
            result.intent = intent
            result.detected_level = detected_level
            result.short_search_queries = short_search_queries

            try:
                fetch_start = time.perf_counter()

                raw_sources = await self._fetcher.fetch(
                    queries=expanded_queries,
                    platforms=query_schema.platforms,
                    max_results=query_schema.max_results,
                )

                fetch_ms = (time.perf_counter() - fetch_start) * 1000

                self._trace.emit(
                    "search_fetch_completed",
                    total_found=len(raw_sources),
                    query_count=len(expanded_queries),
                    latency_ms=round(fetch_ms, 1),
                )
            except Exception as exc:
                raw_sources = []
                errors.append(f"source_fetch_failed: {exc}")

                self._trace.emit(
                    "search_fetch_failed",
                    error=str(exc),
                )

            result.total_found = len(raw_sources)

            normalized_sources = self._normalizer.normalize_sources(raw_sources)
            result.total_normalized = len(normalized_sources)

            deduplicated_sources = self._deduplicator.deduplicate(normalized_sources)
            result.total_deduplicated = len(deduplicated_sources)

            self._trace.emit(
                "search_processing",
                found=result.total_found,
                normalized=result.total_normalized,
                deduplicated=result.total_deduplicated,
                duplicates_removed=result.total_normalized - result.total_deduplicated,
            )

            valid_sources = self._validator.validate_sources(
                deduplicated_sources,
                query_schema,
                strict=True,
            )

            if not valid_sources:
                self._trace.emit(
                    "search_validation_strict_failed",
                    total_deduplicated=result.total_deduplicated,
                    fallback="lenient",
                )

                valid_sources = self._validator.validate_sources(
                    deduplicated_sources,
                    query_schema,
                    strict=False,
                )

            if not valid_sources:
                self._trace.emit(
                    "search_validation_lenient_failed",
                    fallback="no_query",
                )

                valid_sources = self._validator.validate_sources(
                    deduplicated_sources,
                    None,
                    strict=False,
                )

            result.total_valid = len(valid_sources)
            result.sources = self._select_with_diversity(
            valid_sources,
            query_schema.max_results,
        )
            result.errors = errors
            result.query = query_schema
            result.finished_at = datetime.now(timezone.utc)
            result.latency_ms = (time.perf_counter() - start) * 1000

            self._trace.emit(
                "search_completed",
                total_found=result.total_found,
                total_valid=result.total_valid,
                final_sources=len(result.sources),
                errors=errors,
                corrected_topic=corrected_topic,
                primary_concept=primary_concept,
                keywords=keywords,
                latency_ms=round(result.latency_ms, 1),
            )

            return result
        except Exception as exc:
            self._logger.error(f"Search orchestration failed: {exc}")

            result.errors = [*errors, f"search_orchestration_failed: {exc}"]
            result.finished_at = datetime.now(timezone.utc)
            result.latency_ms = (time.perf_counter() - start) * 1000

            self._trace.emit(
                "search_orchestration_failed",
                error=str(exc),
                latency_ms=round(result.latency_ms, 1),
            )

            return result
        finally:
            if self._owns_fetcher and not self._entered:
                await self._fetcher.close()

    def _select_with_diversity(
        self,
        sources: list[Any],
        max_results: int,
    ) -> list[Any]:
        if len(sources) <= max_results:
            return sources
        platform_counts: dict[str, int] = {}
        selected: list[Any] = []
        remaining: list[Any] = []
        for source in sources:
            platform = self._enum_value(source.platform)
            count = platform_counts.get(platform, 0)
            if count < 3:
                selected.append(source)
                platform_counts[platform] = count + 1
            else:
                remaining.append(source)
            if len(selected) >= max_results:
                break
        if len(selected) < max_results:
            for source in remaining:
                selected.append(source)
                if len(selected) >= max_results:
                    break
        return selected[:max_results]

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value))

    def _coerce_query(
        self,
        query: SearchQuerySchema | str | dict[str, Any],
    ) -> SearchQuerySchema:
        try:
            if isinstance(query, SearchQuerySchema):
                return query

            if isinstance(query, str):
                return SearchQuerySchema(topic=query)

            if isinstance(query, dict):
                return SearchQuerySchema.model_validate(query)

            raise SearchError(
                "Unsupported search query type",
                details={"type": type(query).__name__},
            )
        except SearchError:
            raise
        except Exception as exc:
            raise SearchError(
                "Invalid search query",
                details={"error": str(exc)},
            ) from exc

    @staticmethod
    def _normalize_string_list(values: Any) -> list[str]:
        if values is None:
            return []

        if isinstance(values, str):
            values = [values]

        if not isinstance(values, (list, tuple, set)):
            return []

        result: list[str] = []

        for value in values:
            text = clean_text(value)

            if text and text not in result:
                result.append(text)

        return result