from __future__ import annotations

from typing import Any

from core import constants
from core.exceptions import QueryExpansionError
from core.schemas import QueryExpansionResult, SearchQuerySchema
from search.query_intelligence import QueryIntelligence
from utils.logger import get_logger, get_trace_logger
from utils.text import clean_text, truncate_text

_PROFILE_BEGINNER = "beginner"
_PROFILE_MIXED = "mixed"
_PROFILE_ADVANCED = "advanced"
_PROFILE_BALANCED = "balanced"

_BEGINNER_QUERY_MARKERS = (
    "beginner",
    "beginners",
    "basics",
    "basic",
    "fundamental",
    "fundamentals",
    "intro",
    "introduction",
    "introductory",
    "from scratch",
    "step by step",
    "for beginners",
    "explained",
    "tutorial",
    "course",
    "textbook",
    "guide",
    "primer",
    "101",
)

_ADVANCED_QUERY_MARKERS = (
    "advanced",
    "research paper",
    "state of the art",
    "state-of-the-art",
    "in depth",
    "in-depth",
    "deep dive",
    "architecture",
    "implementation",
    "novel",
    "survey",
    "frontier",
    "optimization",
)


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
        mode: str = "",
    ) -> QueryExpansionResult:
        query_schema = self._coerce_query(query)
        original_topic = query_schema.topic

        resolved_mode = self._resolve_mode(mode)
        use_llm_now = (
            self._use_llm
            and resolved_mode != constants.MODE_FAST
        )

        analysis = None

        try:
            analysis = await self._query_intelligence.analyze(
                topic=query_schema.topic,
                goal=query_schema.goal,
                level=query_schema.level,
                use_llm=use_llm_now,
            )
        except Exception as exc:
            self._logger.warning(
                f"Query intelligence failed, using minimal expansion: {exc}"
            )

        if analysis is not None:
            corrected_topic = self._clean_query(
                getattr(analysis, "corrected_topic", "") or original_topic
            )

            primary_concept = self._clean_query(
                getattr(analysis, "primary_concept", "") or corrected_topic
            )

            keywords = self._clean_string_list(
                getattr(analysis, "keywords", []) or []
            )

            intent = clean_text(getattr(analysis, "intent", "") or "")

            detected_level = clean_text(
                getattr(analysis, "level", "") or query_schema.level or ""
            )

            short_search_queries = self._clean_string_list(
                getattr(analysis, "short_search_queries", []) or []
            )

            target_domains = self._clean_string_list(
                getattr(analysis, "target_domains", []) or []
            )

            target_formats = self._clean_string_list(
                getattr(analysis, "target_formats", []) or []
            )

            domain_confidence = float(
                getattr(analysis, "domain_confidence", 0.0) or 0.0
            )

            level_profile = str(
                getattr(analysis, "level_profile", "") or ""
            ).strip().lower()

            beginner_ratio = self._safe_float(
                getattr(analysis, "beginner_ratio", 0.0),
                0.0,
            )

            min_beginner_sources = self._safe_int(
                getattr(analysis, "min_beginner_sources", 0),
                0,
            )

            exclude_research_platforms = bool(
                getattr(analysis, "exclude_research_platforms", False)
            )
        else:
            corrected_topic = self._clean_query(original_topic)
            primary_concept = corrected_topic
            keywords = []
            intent = ""
            detected_level = clean_text(query_schema.level)
            short_search_queries = []
            target_domains = ["general"]
            target_formats = []
            domain_confidence = 0.0
            level_profile = ""
            beginner_ratio = 0.0
            min_beginner_sources = 0
            exclude_research_platforms = False

        if not level_profile:
            level_profile = self._infer_profile(
                topic=original_topic,
                goal=query_schema.goal,
                level=detected_level,
            )

            if level_profile == _PROFILE_BEGINNER:
                beginner_ratio = max(beginner_ratio, 0.9)
                min_beginner_sources = max(min_beginner_sources, 5)
                exclude_research_platforms = True
            elif level_profile == _PROFILE_ADVANCED:
                beginner_ratio = min(beginner_ratio or 0.1, 0.2)
            elif level_profile == _PROFILE_MIXED:
                beginner_ratio = max(0.4, min(0.6, beginner_ratio or 0.5))
                min_beginner_sources = max(min_beginner_sources, 3)
            else:
                beginner_ratio = max(0.3, min(0.5, beginner_ratio or 0.4))
                min_beginner_sources = max(min_beginner_sources, 2)

        beginner_ratio = max(0.0, min(1.0, beginner_ratio))
        min_beginner_sources = max(0, min(20, min_beginner_sources))

        if not target_formats:
            target_formats = ["documentation", "course", "book"]

        if domain_confidence <= 0.0:
            if any(domain != "general" for domain in target_domains):
                domain_confidence = 0.55
            else:
                domain_confidence = 0.20

        queries = self._compose_queries(
            corrected_topic=corrected_topic,
            primary_concept=primary_concept,
            keywords=keywords,
            short_search_queries=short_search_queries,
            target_domains=target_domains,
            target_formats=target_formats,
            level=detected_level,
            level_profile=level_profile,
        )

        if not queries:
            raise QueryExpansionError(
                "Search topic is empty after AI query understanding",
                details={"raw_query": str(query)},
            )

        final_queries = self._dedupe_queries(queries)

        query_level_intents = self._tag_queries(final_queries)

        self._trace.emit(
            "query_expansion_completed",
            corrected_topic=corrected_topic,
            primary_concept=primary_concept,
            keywords=keywords,
            intent=intent,
            detected_level=detected_level,
            level_profile=level_profile,
            beginner_ratio=beginner_ratio,
            min_beginner_sources=min_beginner_sources,
            exclude_research_platforms=exclude_research_platforms,
            target_domains=target_domains,
            target_formats=target_formats,
            domain_confidence=domain_confidence,
            total_generated=len(queries),
            final_count=len(final_queries),
            final_queries=final_queries,
            query_level_intents=query_level_intents,
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
            target_domains=target_domains,
            target_formats=target_formats,
            domain_confidence=max(0.0, min(1.0, domain_confidence)),
            level_profile=level_profile,
            beginner_ratio=beginner_ratio,
            min_beginner_sources=min_beginner_sources,
            exclude_research_platforms=exclude_research_platforms,
            source_type_metadata={
                "query_level_intents": query_level_intents,
            },
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

    def _infer_profile(self, topic: str, goal: str, level: str) -> str:
        text = " ".join(
            [
                clean_text(topic).lower(),
                clean_text(goal).lower(),
                clean_text(level).lower(),
            ]
        ).strip()

        if not text:
            return _PROFILE_BALANCED

        mixed_markers = (
            "basics to advanced",
            "beginner to advanced",
            "fundamentals to advanced",
            "start to finish",
        )

        if any(marker in text for marker in mixed_markers):
            return _PROFILE_MIXED

        has_beginner = any(marker in text for marker in _BEGINNER_QUERY_MARKERS)
        has_advanced = any(marker in text for marker in _ADVANCED_QUERY_MARKERS)

        if has_beginner and has_advanced:
            return _PROFILE_MIXED

        if has_beginner:
            return _PROFILE_BEGINNER

        if has_advanced:
            return _PROFILE_ADVANCED

        return _PROFILE_BALANCED

    def _compose_queries(
        self,
        corrected_topic: str,
        primary_concept: str,
        keywords: list[str],
        short_search_queries: list[str],
        target_domains: list[str],
        target_formats: list[str],
        level: str,
        level_profile: str,
    ) -> list[str]:
        queries: list[str] = []

        concept = primary_concept or corrected_topic
        profile = str(level_profile or "").strip().lower()

        if profile == _PROFILE_BEGINNER:
            self._compose_beginner(
                queries,
                concept=concept,
                short_queries=short_search_queries,
                raw_topic=corrected_topic,
                raw_concept=concept,
            )
        elif profile == _PROFILE_ADVANCED:
            self._compose_advanced(
                queries,
                concept=concept,
                corrected_topic=corrected_topic,
                primary_concept=primary_concept,
                short_queries=short_search_queries,
            )
        elif profile == _PROFILE_MIXED:
            self._compose_mixed(
                queries,
                concept=concept,
                corrected_topic=corrected_topic,
                primary_concept=primary_concept,
                short_queries=short_search_queries,
            )
        else:
            self._compose_balanced(
                queries,
                concept=concept,
                corrected_topic=corrected_topic,
                primary_concept=primary_concept,
                short_queries=short_search_queries,
            )

        if keywords:
            keyword_candidates = [
                " ".join(keywords[:4]),
                " ".join(keywords[:2]),
            ]

            if len(keywords) >= 3:
                keyword_candidates.append(" ".join(keywords[1:4]))

            if concept:
                keyword_candidates.append(f"{concept} {keywords[0]}")

            for candidate in keyword_candidates:
                if profile == _PROFILE_BEGINNER:
                    self._add_beginner_query(
                        queries,
                        candidate,
                        raw_topic=corrected_topic,
                        raw_concept=concept,
                    )
                else:
                    self._add_query(queries, candidate)

        domain_labels = self._extract_domain_labels(
            target_domains,
            corrected_topic=corrected_topic,
            concept=concept,
        )

        if concept and domain_labels:
            candidate = f"{concept} {domain_labels[0]}"

            if profile == _PROFILE_BEGINNER:
                self._add_beginner_query(
                    queries,
                    candidate,
                    raw_topic=corrected_topic,
                    raw_concept=concept,
                )
            else:
                self._add_query(queries, candidate)

        if concept and target_formats:
            format_terms = self._extract_format_terms(target_formats, level)

            for term in format_terms[:2]:
                candidate = f"{concept} {term}"

                if profile == _PROFILE_BEGINNER:
                    self._add_beginner_query(
                        queries,
                        candidate,
                        raw_topic=corrected_topic,
                        raw_concept=concept,
                    )
                else:
                    self._add_query(queries, candidate)

        return queries

    def _compose_beginner(
        self,
        queries: list[str],
        concept: str,
        short_queries: list[str],
        raw_topic: str,
        raw_concept: str,
    ) -> None:
        if concept:
            templates = (
                "{concept} tutorial",
                "{concept} basics",
                "introduction to {concept}",
                "{concept} for beginners",
                "{concept} explained",
                "{concept} course",
                "{concept} fundamentals",
            )

            for template in templates:
                self._add_query(queries, template.format(concept=concept))

        for query in short_queries:
            self._add_beginner_query(
                queries,
                query,
                raw_topic=raw_topic,
                raw_concept=raw_concept,
            )

    def _compose_advanced(
        self,
        queries: list[str],
        concept: str,
        corrected_topic: str,
        primary_concept: str,
        short_queries: list[str],
    ) -> None:
        self._add_focused_variant(queries, corrected_topic)
        self._add_focused_variant(queries, primary_concept)

        if concept:
            templates = (
                "{concept} research paper",
                "{concept} advanced",
                "{concept} architecture",
                "{concept} implementation",
            )

            for template in templates:
                self._add_query(queries, template.format(concept=concept))

        for query in short_queries:
            self._add_query(queries, query)

    def _compose_mixed(
        self,
        queries: list[str],
        concept: str,
        corrected_topic: str,
        primary_concept: str,
        short_queries: list[str],
    ) -> None:
        self._add_focused_variant(queries, corrected_topic)
        self._add_focused_variant(queries, primary_concept)

        if concept:
            beginner_templates = (
                "{concept} tutorial",
                "{concept} basics",
                "introduction to {concept}",
            )

            advanced_templates = (
                "{concept} advanced",
                "{concept} research paper",
                "{concept} implementation",
            )

            for template in beginner_templates:
                self._add_query(queries, template.format(concept=concept))

            for template in advanced_templates:
                self._add_query(queries, template.format(concept=concept))

        for query in short_queries:
            self._add_focused_variant(queries, query)

    def _compose_balanced(
        self,
        queries: list[str],
        concept: str,
        corrected_topic: str,
        primary_concept: str,
        short_queries: list[str],
    ) -> None:
        self._add_focused_variant(queries, corrected_topic)
        self._add_focused_variant(queries, primary_concept)

        if concept:
            beginner_templates = (
                "{concept} tutorial",
                "{concept} basics",
            )

            advanced_templates = (
                "{concept} advanced",
                "{concept} research paper",
            )

            for template in beginner_templates:
                self._add_query(queries, template.format(concept=concept))

            for template in advanced_templates:
                self._add_query(queries, template.format(concept=concept))

        for query in short_queries:
            self._add_focused_variant(queries, query)

    def _add_beginner_query(
        self,
        queries: list[str],
        candidate: Any,
        raw_topic: str,
        raw_concept: str,
    ) -> None:
        cleaned = self._clean_query(candidate)

        if not cleaned:
            return

        if cleaned == raw_topic or cleaned == raw_concept:
            return

        if not self._has_beginner_marker(cleaned):
            cleaned = f"introduction to {cleaned}"

        self._add_query(queries, cleaned)

    def _extract_domain_labels(
        self,
        target_domains: list[str],
        corrected_topic: str,
        concept: str,
    ) -> list[str]:
        domain_labels: list[str] = []

        for domain in target_domains[:2]:
            domain_text = str(domain or "").strip().lower()

            if not domain_text or domain_text == "general":
                continue

            label = domain_text.replace("_", " ")

            if not label:
                continue

            if label in corrected_topic or label in concept:
                continue

            if label not in domain_labels:
                domain_labels.append(label)

        return domain_labels

    def _extract_format_terms(
        self,
        target_formats: list[str],
        level: str,
    ) -> list[str]:
        format_terms: list[str] = []

        for source_format in target_formats[:4]:
            normalized = str(source_format or "").strip().lower()

            if normalized == "course":
                term = "course"
            elif normalized == "book":
                term = "book"
            elif normalized == "documentation":
                term = "documentation"
            elif normalized == "video":
                term = "video"
            elif normalized == "repository":
                term = "implementation"
            elif normalized == "model":
                term = "model"
            elif normalized == "dataset":
                term = "dataset"
            elif normalized == "research_paper":
                term = "paper"
            else:
                continue

            if term not in format_terms:
                format_terms.append(term)

        if level and "beginner" in str(level).lower():
            preferred_order = ["course", "book", "documentation", "video"]

            format_terms.sort(
                key=lambda item: preferred_order.index(item)
                if item in preferred_order
                else 99
            )

        return format_terms

    def _has_beginner_marker(self, query: str) -> bool:
        lowered = str(query or "").lower()
        return any(marker in lowered for marker in _BEGINNER_QUERY_MARKERS)

    def _has_advanced_marker(self, query: str) -> bool:
        lowered = str(query or "").lower()
        return any(marker in lowered for marker in _ADVANCED_QUERY_MARKERS)

    def _tag_queries(self, queries: list[str]) -> dict[str, str]:
        tags: dict[str, str] = {}

        for query in queries:
            if self._has_beginner_marker(query):
                tags[query] = _PROFILE_BEGINNER
            elif self._has_advanced_marker(query):
                tags[query] = _PROFILE_ADVANCED
            else:
                tags[query] = _PROFILE_BALANCED

        return tags

    def _add_query(self, queries: list[str], value: Any) -> None:
        cleaned = self._clean_query(value)

        if not cleaned:
            return

        if cleaned in queries:
            return

        queries.append(cleaned)
    def _add_focused_variant(
        self,
        queries: list[str],
        value: Any,
    ) -> None:
        cleaned = self._clean_query(value)

        if not cleaned:
            return

        words = cleaned.split()

        if len(words) > 3:
            focused = " ".join(words[:3])
            self._add_query(queries, focused)

            if len(words) > 4:
                short2 = " ".join(words[1:4])
                self._add_query(queries, short2)

            if len(words) > 5:
                short3 = " ".join(words[2:5])
                self._add_query(queries, short3)

        self._add_query(queries, cleaned)
        self._add_short_broad_variants(queries, cleaned)

    def _add_short_broad_variants(
        self,
        queries: list[str],
        cleaned: str,
    ) -> None:
        words = cleaned.split()

        if not words:
            return

        if len(words) == 1:
            self._add_query(queries, f"{words[0]} tutorial")
            self._add_query(queries, f"{words[0]} introduction")
            self._add_query(queries, f"introduction to {words[0]}")
            return

        if len(words) == 2:
            joined = " ".join(words)
            self._add_query(queries, f"{joined} tutorial")
            self._add_query(queries, f"{joined} introduction")
            self._add_query(queries, f"{joined} basics")
            self._add_query(queries, f"{joined} principles")
            return

        if len(words) == 3:
            joined = " ".join(words)
            self._add_query(queries, f"{joined} overview")
            self._add_query(queries, f"{joined} introduction")
            return
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

    def _clean_string_list(self, values: Any) -> list[str]:
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

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)
        
    @staticmethod
    def _resolve_mode(mode: str) -> str:
        text = str(mode or "").strip().lower()

        if text in constants.VALID_MODES:
            return text

        return constants.DEFAULT_MODE

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)