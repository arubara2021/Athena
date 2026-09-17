from __future__ import annotations

from typing import Any

from core import constants
from core.exceptions import QueryExpansionError
from core.schemas import QueryExpansionResult, SearchQuerySchema
from search.query_intelligence import QueryIntelligence
from utils.logger import get_logger, get_trace_logger
from utils.text import clean_text, truncate_text


class QueryExpander:
    def __init__(
        self,
        use_llm: bool = False,
        max_expansions: int = 12,
        query_intelligence: QueryIntelligence | None = None,
    ) -> None:
        self._use_llm = use_llm
        self._max_expansions = max(1, int(max_expansions))
        self._query_intelligence = query_intelligence or QueryIntelligence(
            use_llm=use_llm
        )
        self._logger = get_logger("search.query_expander")
        self._trace = get_trace_logger()

    async def expand(
        self,
        query: SearchQuerySchema | str | dict[str, Any],
    ) -> QueryExpansionResult:
        query_schema = self._coerce_query(query)
        original_topic = query_schema.topic
        analysis = None

        try:
            analysis = await self._query_intelligence.analyze(
                topic=query_schema.topic,
                goal=query_schema.goal,
                level=query_schema.level,
            )
        except Exception as exc:
            self._logger.warning(
                f"Query intelligence failed, using minimal expansion: {exc}"
            )

        corrected_topic = self._clean_query(
            getattr(analysis, "corrected_topic", "") or original_topic
        )

        primary_concept = self._clean_query(
            getattr(analysis, "primary_concept", "") or corrected_topic
        )

        keywords = self._clean_keywords(
            getattr(analysis, "keywords", []) or [],
            primary_concept,
        )

        intent = clean_text(getattr(analysis, "intent", "") or "")

        detected_level = clean_text(
            getattr(analysis, "level", "") or query_schema.level or ""
        )

        short_search_queries = self._clean_queries(
            getattr(analysis, "short_search_queries", [])
            or getattr(analysis, "suggested_queries", [])
            or [],
            corrected_topic,
        )

        queries = self._compose_queries(
            corrected_topic,
            primary_concept,
            keywords,
            short_search_queries,
        )

        if not queries:
            raise QueryExpansionError(
                "Search topic is empty after AI query understanding",
                details={"raw_query": str(query)},
            )

        final_queries = self._dedupe_queries(queries)

        self._trace.emit(
            "query_expansion_completed",
            corrected_topic=corrected_topic,
            primary_concept=primary_concept,
            keywords=keywords,
            intent=intent,
            detected_level=detected_level,
            total_generated=len(queries),
            final_count=len(final_queries),
            final_queries=final_queries,
        )

        return QueryExpansionResult(
            queries=final_queries,
            corrected_topic=corrected_topic,
            original_topic=original_topic,
            primary_concept=primary_concept,
            keywords=keywords,
            intent=intent,
            detected_level=detected_level,
            short_search_queries=short_search_queries,
        )

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

            raise QueryExpansionError(
                "Unsupported query type",
                details={"type": type(query).__name__},
            )
        except QueryExpansionError:
            raise
        except Exception as exc:
            raise QueryExpansionError(
                "Invalid search query",
                details={"error": str(exc)},
            ) from exc

    def _compose_queries(
        self,
        corrected_topic: str,
        primary_concept: str,
        keywords: list[str],
        short_search_queries: list[str],
    ) -> list[str]:
        queries: list[str] = []

        base_candidates = [corrected_topic, primary_concept]
        base_candidates.extend(short_search_queries)

        for candidate in base_candidates:
            cleaned = self._clean_query(candidate)

            if cleaned and cleaned not in queries:
                queries.append(cleaned)

        if keywords:
            keyword_candidates = [
                " ".join(keywords[:4]),
                " ".join(keywords[:2]),
            ]

            if len(keywords) >= 3:
                keyword_candidates.append(" ".join(keywords[1:4]))

            if primary_concept:
                keyword_candidates.append(f"{primary_concept} {keywords[0]}")

            for candidate in keyword_candidates:
                cleaned = self._clean_query(candidate)

                if cleaned and cleaned not in queries:
                    queries.append(cleaned)

        return queries

    def _clean_query(self, value: Any) -> str:
        text = clean_text(value).lower()

        if not text:
            return ""

        words = text.split()

        if len(words) > 8:
            text = " ".join(words[:8])

        return truncate_text(
            text,
            max_length=constants.MAX_SEARCH_QUERY_LENGTH,
            suffix="",
        )

    def _clean_keywords(self, values: Any, fallback: str) -> list[str]:
        cleaned_keywords: list[str] = []

        if isinstance(values, str):
            values = [values]

        if isinstance(values, (list, tuple, set)):
            for value in values:
                text = clean_text(value).lower()
                words = text.split()[:3]
                phrase = " ".join(words).strip()

                if phrase and phrase not in cleaned_keywords:
                    cleaned_keywords.append(phrase)

                if len(cleaned_keywords) >= 8:
                    break

        if not cleaned_keywords and fallback:
            fallback_words = clean_text(fallback).lower().split()

            cleaned_keywords = [
                word
                for word in fallback_words[:6]
                if len(word) >= 2
            ]

        return cleaned_keywords

    def _clean_queries(self, values: Any, fallback: str) -> list[str]:
        cleaned_queries: list[str] = []

        if isinstance(values, str):
            values = [values]

        if isinstance(values, (list, tuple, set)):
            for value in values:
                cleaned = self._clean_query(value)

                if cleaned and cleaned not in cleaned_queries:
                    cleaned_queries.append(cleaned)

                if len(cleaned_queries) >= 6:
                    break

        if fallback:
            cleaned_fallback = self._clean_query(fallback)

            if cleaned_fallback and cleaned_fallback not in cleaned_queries:
                cleaned_queries.insert(0, cleaned_fallback)

        return cleaned_queries[:6]

    def _dedupe_queries(self, queries: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []

        for raw_query in queries:
            cleaned = clean_text(raw_query)

            if not cleaned:
                continue

            cleaned = truncate_text(
                cleaned,
                max_length=constants.MAX_SEARCH_QUERY_LENGTH,
                suffix="",
            )

            key = " ".join(cleaned.lower().split())

            if key in seen:
                continue

            seen.add(key)
            result.append(cleaned)

            if len(result) >= self._max_expansions:
                break

        return result