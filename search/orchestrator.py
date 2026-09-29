from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field

from core import constants
from core.config import get_settings
from core.exceptions import SearchError
from core.models import SourcePlatform
from core.schemas import SearchQuerySchema
from ranking.difficulty_classifier import DifficultyClassifier
from search.dedupe import SourceDeduplicator
from search.llm_relevance import LLMRelevanceFilter
from search.normalizer import SourceNormalizer
from search.query_expander import QueryExpander
from search.query_profile import build_query_profile
from search.source_fetcher import SourceFetcher, load_domain_routing_config
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
    mode: str = constants.DEFAULT_MODE
    expanded_queries: list[str] = Field(default_factory=list)
    corrected_topic: str = ""
    primary_concept: str = ""
    keywords: list[str] = Field(default_factory=list)
    intent: str = ""
    detected_level: str = ""
    short_search_queries: list[str] = Field(default_factory=list)
    sources: list[Any] = Field(default_factory=list)
    total_found: int = 0
    total_normalized: int = 0
    total_deduplicated: int = 0
    total_valid: int = 0
    errors: list[str] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime | None = None
    latency_ms: float = 0.0

    target_domains: list[str] = Field(default_factory=list)
    target_formats: list[str] = Field(default_factory=list)
    domain_confidence: float = 0.0
    domain_routed: bool = False
    domain_expanded: bool = False
    domain_platforms: list[str] = Field(default_factory=list)

    level_profile: str = ""
    beginner_ratio: float = 0.0
    min_beginner_sources: int = 0
    exclude_research_platforms: bool = False

    preserve_tokens: list[str] = Field(default_factory=list)
    concept_origin: str = ""

    fetch_report: dict[str, Any] = Field(default_factory=dict)


_INTENT_ONLY_DOMAINS = frozenset({"education", "general"})


def _search_setting(name: str, default: Any = None) -> Any:
    try:
        settings = get_settings()
    except Exception:
        settings = None

    if settings is not None:
        section = getattr(settings, "search", None)

        if isinstance(section, dict):
            value = section.get(name)

            if value is not None:
                return value
        elif section is not None:
            value = getattr(section, name, None)

            if value is not None:
                return value

        value = getattr(settings, f"search_{name}", None)

        if value is not None:
            return value

    try:
        config_path = (
            Path(__file__).resolve().parent.parent
            / "configs"
            / "settings.yaml"
        )

        if config_path.exists():
            raw = yaml.safe_load(
                config_path.read_text(encoding="utf-8")
            ) or {}

            section = raw.get("search", {}) if isinstance(raw, dict) else {}

            if isinstance(section, dict) and name in section:
                return section[name]
    except Exception:
        pass

    return default


class SearchOrchestrator:
    def __init__(
        self,
        use_llm_expansion: bool | None = None,
        query_expander: QueryExpander | None = None,
        source_fetcher: SourceFetcher | None = None,
        normalizer: SourceNormalizer | None = None,
        deduplicator: SourceDeduplicator | None = None,
        validator: SourceValidator | None = None,
        difficulty_classifier: DifficultyClassifier | None = None,
        llm_relevance: LLMRelevanceFilter | None = None,
        platforms: list[Any] | None = None,
        concurrency: int | None = None,
        max_query_variants: int | None = None,
        request_timeout: float | None = None,
    ) -> None:
        resolved_use_llm_expansion = (
            bool(use_llm_expansion)
            if use_llm_expansion is not None
            else bool(_search_setting("use_llm_expansion", False))
        )

        self._expander = query_expander or QueryExpander(
            use_llm=resolved_use_llm_expansion
        )

        if source_fetcher is None:
            self._fetcher = SourceFetcher(
                platforms=platforms,
                concurrency=concurrency,
                max_query_variants=max_query_variants,
                request_timeout=request_timeout,
            )
        else:
            self._fetcher = source_fetcher

            if platforms is not None:
                resolve_platforms = getattr(
                    source_fetcher, "_resolve_platforms", None
                )

                if callable(resolve_platforms):
                    try:
                        source_fetcher._default_platforms = (
                            resolve_platforms(platforms)
                        )
                    except Exception:
                        pass

            for attribute_name, attribute_value in (
                ("concurrency", concurrency),
                ("max_query_variants", max_query_variants),
                ("request_timeout", request_timeout),
            ):
                if attribute_value is None:
                    continue

                try:
                    setattr(
                        source_fetcher,
                        f"_{attribute_name}",
                        attribute_value,
                    )
                except Exception:
                    pass

                try:
                    setattr(
                        source_fetcher,
                        attribute_name,
                        attribute_value,
                    )
                except Exception:
                    pass

        self._owns_fetcher = source_fetcher is None
        self._normalizer = normalizer or SourceNormalizer()
        self._deduplicator = deduplicator or SourceDeduplicator()
        self._validator = validator or SourceValidator()
        self._difficulty_classifier = (
            difficulty_classifier or DifficultyClassifier()
        )
        self._llm_relevance = (
            llm_relevance or self._build_llm_relevance_filter()
        )
        self._entered = False

        self._logger = get_logger("search.orchestrator")
        self._trace = get_trace_logger()
    @staticmethod
    def _beginner_platform_names() -> list[str]:
        return [
            "wikipedia",
            "wikibooks",
            "wikiversity",
            "openstax",
            "libretexts",
            "mit_ocw",
            "open_library",
            "internet_archive",
        ]
    def _build_llm_relevance_filter(self) -> LLMRelevanceFilter:
        enabled = bool(
            _search_setting("llm_relevance_enabled", False)
        )
        batch_size = self._as_int(
            _search_setting("llm_relevance_batch_size", 10),
            10,
        )
        max_sources = self._as_int(
            _search_setting("llm_relevance_max_sources", 40),
            40,
        )
        min_sources = self._as_int(
            _search_setting("llm_relevance_min_sources", 3),
            3,
        )
        model_limit = self._as_int(
            _search_setting("llm_relevance_model_limit", 3),
            3,
        )

        return LLMRelevanceFilter(
            enabled=enabled,
            batch_size=batch_size,
            max_sources=max_sources,
            min_sources=min_sources,
            model_limit=model_limit,
        )

    async def __aenter__(self) -> SearchOrchestrator:
        self._entered = True
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc: Any,
        tb: Any,
    ) -> bool:
        await self.close()
        self._entered = False
        return False

    async def close(self) -> None:
        if self._owns_fetcher:
            await self._fetcher.close()

    async def search(
        self,
        query: SearchQuerySchema | str | dict[str, Any],
        mode: str = "",
    ) -> SearchOrchestrationResult:
        return await self.run(query, mode=mode)

    def _domain_registry(self) -> Any:
        try:
            from core.domain_registry import get_domain_registry

            return get_domain_registry()
        except Exception:
            return None

    @staticmethod
    def _clean_domain_list(values: Any) -> list[str]:
        if values is None:
            return []

        if isinstance(values, str):
            values = [item.strip() for item in values.split(",")]

        if not isinstance(values, (list, tuple, set)):
            values = [values]

        cleaned: list[str] = []

        for value in values:
            text = str(value or "").strip().lower()
            text = text.replace("-", "_").replace(" ", "_")

            if text and text not in cleaned:
                cleaned.append(text)

        return cleaned

    def _extract_domain_fields(
        self,
        query_schema: SearchQuerySchema,
        expansion_result: Any,
    ) -> tuple[list[str], list[str], float]:
        config = load_domain_routing_config()

        domains: list[str] = []
        formats: list[str] = []
        confidence = 0.0

        for obj in (expansion_result, query_schema):
            if obj is None:
                continue

            raw_domains = getattr(obj, "target_domains", None)

            if raw_domains:
                domains.extend(self._clean_domain_list(raw_domains))

            raw_formats = getattr(obj, "target_formats", None)

            if raw_formats and not formats:
                formats = self._clean_domain_list(raw_formats)

            raw_confidence = getattr(obj, "domain_confidence", None)

            if raw_confidence is not None:
                try:
                    confidence = float(raw_confidence)
                except Exception:
                    pass

        registry = self._domain_registry()

        if registry is not None:
            try:
                resolved = registry.resolve_domains(domains)
            except Exception:
                resolved = domains
        else:
            resolved = domains

        unique: list[str] = []

        for domain in resolved:
            if domain and domain not in unique:
                unique.append(domain)

        max_domains = self._as_int(
            config.get("max_domains_per_query"), 3
        )
        unique = unique[:max_domains]

        confidence = max(0.0, min(1.0, confidence))

        return unique, formats, confidence

    def _resolve_platform_values(
        self, platforms: Any
    ) -> list[str]:
        resolved: list[str] = []

        if platforms is None:
            return resolved

        if isinstance(platforms, str):
            platforms = [platforms]

        if not isinstance(platforms, (list, tuple, set)):
            platforms = []

        for platform in platforms:
            value = self._enum_value(platform)

            if value and value not in resolved:
                resolved.append(value)

        return resolved

    def _filter_platforms_by_source_types(
        self,
        platforms: list[Any],
        source_types: list[Any],
    ) -> list[Any]:
        if not platforms or not source_types:
            return platforms

        type_map = getattr(
            self._fetcher, "_SOURCE_TYPE_CATEGORY_MAP", {}
        )

        categories: list[str] = []

        for source_type in source_types:
            value = self._enum_value(source_type).lower()
            category = type_map.get(value)

            if category and category not in categories:
                categories.append(category)

        if not categories:
            return platforms

        allowed: set[str] = set()

        for category in categories:
            try:
                allowed.update(
                    self._fetcher.category_platforms(category)
                )
            except Exception:
                pass

        if not allowed:
            return platforms

        filtered = [
            platform
            for platform in platforms
            if self._enum_value(platform) in allowed
        ]

        return filtered or platforms

    def _select_domain_platforms(
        self,
        query_schema: SearchQuerySchema,
        expansion_result: Any,
    ) -> tuple[list[Any] | None, bool, list[str], list[str], float]:
        config = load_domain_routing_config()

        (
            target_domains,
            target_formats,
            confidence,
        ) = self._extract_domain_fields(
            query_schema,
            expansion_result,
        )

        if not self._as_bool(config.get("enabled"), True):
            return (
                None,
                False,
                target_domains,
                target_formats,
                confidence,
            )

        registry = self._domain_registry()

        if registry is None:
            return (
                None,
                False,
                target_domains,
                target_formats,
                confidence,
            )

        explicit_platforms = self._resolve_platform_values(
            getattr(query_schema, "platforms", [])
        )

        if explicit_platforms and self._as_bool(
            config.get("respect_explicit_platforms"),
            True,
        ):
            return (
                explicit_platforms,
                False,
                target_domains,
                target_formats,
                confidence,
            )

        if not target_domains:
            return (
                None,
                False,
                target_domains,
                target_formats,
                confidence,
            )

        subject_domains = [
            domain
            for domain in target_domains
            if domain not in _INTENT_ONLY_DOMAINS
        ]

        if not subject_domains:
            return (
                None,
                False,
                target_domains,
                target_formats,
                confidence,
            )

        low_confidence = (
            confidence > 0.0
            and confidence
            < self._as_float(config.get("min_confidence"), 0.35)
        )

        include_universal = low_confidence and self._as_bool(
            config.get("use_universal_when_low_confidence"),
            True,
        )

        max_platforms = self._as_int(
            config.get("max_domain_platforms"), 16
        )
        min_platforms = self._as_int(
            config.get("min_sources_before_domain_expansion"),
            8,
        )

        try:
            platforms = registry.get_platforms_for_domains(
                subject_domains,
                include_fallback=False,
                include_universal=include_universal,
                min_platforms=min_platforms,
                max_platforms=max_platforms,
            )
        except Exception:
            platforms = []

        if not platforms:
            try:
                platforms = registry.get_universal_platforms()
            except Exception:
                platforms = []

        platforms = self._filter_platforms_by_source_types(
            platforms,
            getattr(query_schema, "source_types", []),
        )

        return (
            platforms,
            True,
            target_domains,
            target_formats,
            confidence,
        )

    async def run(
        self,
        query: SearchQuerySchema | str | dict[str, Any],
        mode: str = "",
    ) -> SearchOrchestrationResult:
        started_at = datetime.now(timezone.utc)
        start = time.perf_counter()
        errors: list[str] = []

        resolved_mode = self._resolve_mode(mode)
        mode_min_sources = self._mode_min_sources(resolved_mode)

        query_schema = self._coerce_query(query)

        result = SearchOrchestrationResult(
            query=query_schema,
            mode=resolved_mode,
            started_at=started_at,
        )

        corrected_topic = clean_text(query_schema.topic)
        primary_concept = corrected_topic
        keywords: list[str] = []
        intent = ""
        detected_level = clean_text(query_schema.level)
        short_search_queries: list[str] = []
        expanded_queries = [corrected_topic]
        expansion_result = None
        preserve_tokens: list[str] = []

        level_profile = str(
            getattr(query_schema, "level_profile", "") or ""
        ).strip().lower()
        beginner_ratio = self._as_float(
            getattr(query_schema, "beginner_ratio", 0.0), 0.0
        )
        min_beginner_sources = self._as_int(
            getattr(query_schema, "min_beginner_sources", 0), 0
        )
        exclude_research_platforms = self._as_bool(
            getattr(
                query_schema, "exclude_research_platforms", False
            ),
            False,
        )

        self._trace.emit(
            "search_started",
            topic=query_schema.topic,
            platforms=(
                [
                    p.value if hasattr(p, "value") else str(p)
                    for p in query_schema.platforms
                ]
                if query_schema.platforms
                else "all"
            ),
            max_results=query_schema.max_results,
            mode=resolved_mode,
            mode_min_sources=mode_min_sources,
        )

        try:
            try:
                expand_start = time.perf_counter()
                expansion_result = await self._expander.expand(
                    query_schema,
                    mode=resolved_mode,
                )

                expanded_queries = (
                    expansion_result.queries or [corrected_topic]
                )

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

                intent = clean_text(
                    getattr(expansion_result, "intent", "")
                )

                detected_level = clean_text(
                    getattr(expansion_result, "detected_level", "")
                    or detected_level
                )

                short_search_queries = self._normalize_string_list(
                    getattr(
                        expansion_result,
                        "short_search_queries",
                        [],
                    )
                )

                raw_profile = getattr(
                    expansion_result, "level_profile", ""
                )

                if raw_profile:
                    level_profile = str(raw_profile).strip().lower()

                raw_ratio = getattr(
                    expansion_result, "beginner_ratio", None
                )

                if raw_ratio is not None:
                    beginner_ratio = self._as_float(
                        raw_ratio, beginner_ratio
                    )

                raw_min_beginner = getattr(
                    expansion_result,
                    "min_beginner_sources",
                    None,
                )

                if raw_min_beginner is not None:
                    min_beginner_sources = self._as_int(
                        raw_min_beginner, min_beginner_sources
                    )

                raw_exclude = getattr(
                    expansion_result,
                    "exclude_research_platforms",
                    None,
                )

                if raw_exclude is not None:
                    exclude_research_platforms = self._as_bool(
                        raw_exclude, exclude_research_platforms
                    )

                raw_preserve = getattr(
                    expansion_result, "preserve_tokens", None
                )

                if raw_preserve:
                    preserve_tokens = self._normalize_string_list(
                        raw_preserve
                    )

                beginner_ratio = max(
                    0.0, min(1.0, beginner_ratio)
                )
                min_beginner_sources = max(
                    0, min(20, min_beginner_sources)
                )

                expand_ms = (
                    time.perf_counter() - expand_start
                ) * 1000

                if (
                    corrected_topic
                    and corrected_topic != query_schema.topic
                ):
                    self._trace.emit(
                        "search_query_corrected",
                        original_topic=query_schema.topic,
                        corrected_topic=corrected_topic,
                        primary_concept=primary_concept,
                        keywords=keywords,
                        intent=intent,
                        detected_level=detected_level,
                        level_profile=level_profile,
                        preserve_tokens=preserve_tokens,
                    )

                self._trace.emit(
                    "search_query_expansion",
                    topic=corrected_topic,
                    expanded_count=len(expanded_queries),
                    queries=expanded_queries,
                    level_profile=level_profile,
                    beginner_ratio=beginner_ratio,
                    min_beginner_sources=min_beginner_sources,
                    exclude_research_platforms=exclude_research_platforms,
                    preserve_tokens=preserve_tokens,
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
                        "level_profile": level_profile,
                        "beginner_ratio": beginner_ratio,
                        "min_beginner_sources": min_beginner_sources,
                        "exclude_research_platforms": (
                            exclude_research_platforms
                        ),
                        "preserve_tokens": preserve_tokens,
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
            result.level_profile = level_profile
            result.beginner_ratio = beginner_ratio
            result.min_beginner_sources = min_beginner_sources
            result.exclude_research_platforms = (
                exclude_research_platforms
            )
            result.preserve_tokens = preserve_tokens

            (
                domain_platforms,
                domain_routed,
                target_domains,
                target_formats,
                domain_confidence,
            ) = self._select_domain_platforms(
                query_schema, expansion_result
            )

            result.target_domains = target_domains
            result.target_formats = target_formats
            result.domain_confidence = domain_confidence
            result.domain_routed = domain_routed
            result.domain_platforms = [
                self._enum_value(platform)
                for platform in domain_platforms or []
            ]

            if domain_platforms is not None:
                fetch_platforms: Any = domain_platforms
            else:
                fetch_platforms = query_schema.platforms

            level_profile = str(
                getattr(query_schema, "level_profile", "") or ""
            ).strip().lower()

            if level_profile in {"beginner", "mixed", "balanced"}:
                beginner_platform_names = (
                    self._beginner_platform_names()
                )

                existing: list[str] = []

                if isinstance(fetch_platforms, list):
                    for item in fetch_platforms:
                        existing.append(
                            self._enum_value(item).lower()
                        )
                elif fetch_platforms is None:
                    existing = list(beginner_platform_names)

                for name in beginner_platform_names:
                    if name not in existing:
                        existing.append(name)

                fetch_platforms = existing

            fetch_domains = (
                target_domains if domain_routed else None
            )

            raw_sources = await self._fetcher.fetch(
                queries=expanded_queries,
                platforms=fetch_platforms,
                max_results=query_schema.max_results,
                domains=fetch_domains,
                domain_mode=domain_routed,
                level_profile=query_schema.level_profile,
                beginner_ratio=query_schema.beginner_ratio,
                min_beginner_sources=query_schema.min_beginner_sources,
                exclude_research_platforms=(
                    query_schema.exclude_research_platforms
                ),
                mode=resolved_mode,
            )

            self._emit_fetch_report(result)

            domain_expanded = False

            if (
                domain_routed
                and target_domains
                and len(raw_sources) < mode_min_sources
            ):
                config = load_domain_routing_config()

                min_sources = self._as_int(
                    config.get(
                        "min_sources_before_domain_expansion"
                    ),
                    8,
                )

                allow_domain_fallback = self._as_bool(
                    config.get("allow_domain_fallback"),
                    True,
                )

                use_universal = self._as_bool(
                    config.get("use_universal_when_low_sources"),
                    True,
                )

                registry = self._domain_registry()

                if (
                    allow_domain_fallback
                    and registry is not None
                    and len(raw_sources) < min_sources
                ):
                    current_platforms = {
                        self._enum_value(platform)
                        for platform in fetch_platforms or []
                    }

                    max_expanded_platforms = (
                        self._as_int(
                            config.get("max_domain_platforms"), 16
                        )
                        * 2
                    )

                    try:
                        expanded_platforms = (
                            registry.get_platforms_for_domains(
                                target_domains,
                                include_fallback=True,
                                include_universal=use_universal,
                                min_platforms=min_sources,
                                max_platforms=max_expanded_platforms,
                            )
                        )
                    except Exception:
                        expanded_platforms = []

                    new_platforms = [
                        platform
                        for platform in expanded_platforms
                        if self._enum_value(platform)
                        not in current_platforms
                    ]

                    if new_platforms:
                        additional_sources = (
                            await self._fetcher.fetch(
                                queries=expanded_queries,
                                platforms=new_platforms,
                                max_results=(
                                    query_schema.max_results
                                ),
                                domains=target_domains,
                                domain_mode=True,
                                level_profile=(
                                    query_schema.level_profile
                                ),
                                beginner_ratio=(
                                    query_schema.beginner_ratio
                                ),
                                min_beginner_sources=(
                                    query_schema.min_beginner_sources
                                ),
                                exclude_research_platforms=(
                                    query_schema.exclude_research_platforms
                                ),
                                mode=resolved_mode,
                            )
                        )

                        raw_sources = self._merge_sources(
                            raw_sources, additional_sources
                        )
                        domain_expanded = True

                        current_platforms.update(
                            self._enum_value(platform)
                            for platform in new_platforms
                        )

                        self._emit_fetch_report(result)

                    if (
                        use_universal
                        and len(raw_sources) < min_sources
                    ):
                        try:
                            universal_platforms = (
                                registry.get_universal_platforms()
                            )
                        except Exception:
                            universal_platforms = []

                        new_platforms = [
                            platform
                            for platform in universal_platforms
                            if self._enum_value(platform)
                            not in current_platforms
                        ]

                        if new_platforms:
                            additional_sources = (
                                await self._fetcher.fetch(
                                    queries=expanded_queries,
                                    platforms=new_platforms,
                                    max_results=(
                                        query_schema.max_results
                                    ),
                                    domains=target_domains,
                                    domain_mode=True,
                                    level_profile=(
                                        query_schema.level_profile
                                    ),
                                    beginner_ratio=(
                                        query_schema.beginner_ratio
                                    ),
                                    min_beginner_sources=(
                                        query_schema.min_beginner_sources
                                    ),
                                    exclude_research_platforms=(
                                        query_schema.exclude_research_platforms
                                    ),
                                    mode=resolved_mode,
                                )
                            )

                            raw_sources = self._merge_sources(
                                raw_sources, additional_sources
                            )

                            domain_expanded = True

                            self._emit_fetch_report(result)

            result.domain_expanded = domain_expanded

            fallback_enabled = bool(
                _search_setting(
                    "keyless_fallback_enabled",
                    constants.KEYLESS_FALLBACK_ENABLED,
                )
            )

            try:
                min_sources_before_fallback = max(
                    1,
                    int(
                        _search_setting(
                            "min_sources_before_fallback",
                            constants.MIN_SOURCES_BEFORE_FALLBACK,
                        )
                    ),
                )
            except Exception:
                min_sources_before_fallback = (
                    constants.MIN_SOURCES_BEFORE_FALLBACK
                )

            if (
                fallback_enabled
                and len(raw_sources) < mode_min_sources
                and len(raw_sources) < min_sources_before_fallback
            ):
                attempted_platforms: list[Any] = list(
                    query_schema.platforms or []
                )

                if domain_platforms is not None:
                    attempted_platforms = [
                        *attempted_platforms,
                        *[
                            self._enum_value(platform)
                            for platform in domain_platforms
                        ],
                    ]

                fallback_platforms = self._fallback_platforms(
                    attempted_platforms
                )

                if fallback_platforms:
                    try:
                        fallback_sources = (
                            await self._fetcher.fetch(
                                queries=expanded_queries,
                                platforms=fallback_platforms,
                                max_results=(
                                    query_schema.max_results
                                ),
                                level_profile=(
                                    query_schema.level_profile
                                ),
                                beginner_ratio=(
                                    query_schema.beginner_ratio
                                ),
                                min_beginner_sources=(
                                    query_schema.min_beginner_sources
                                ),
                                exclude_research_platforms=(
                                    query_schema.exclude_research_platforms
                                ),
                                mode=resolved_mode,
                            )
                        )

                        raw_sources = self._merge_sources(
                            raw_sources, fallback_sources
                        )

                        self._emit_fetch_report(result)

                        self._trace.emit(
                            "search_keyless_fallback_completed",
                            fallback_platforms=[
                                self._enum_value(platform)
                                for platform in fallback_platforms
                            ],
                            total_found=len(raw_sources),
                            domain_routed=domain_routed,
                        )
                    except Exception as exc:
                        errors.append(
                            f"keyless_fallback_failed: {exc}"
                        )

            result.total_found = len(raw_sources)

            normalized_sources = self._normalizer.normalize_sources(
                raw_sources
            )
            result.total_normalized = len(normalized_sources)

            deduplicated_sources = (
                self._deduplicator.deduplicate(
                    normalized_sources
                )
            )
            result.total_deduplicated = len(deduplicated_sources)

            self._trace.emit(
                "search_processing",
                found=result.total_found,
                normalized=result.total_normalized,
                deduplicated=result.total_deduplicated,
                duplicates_removed=(
                    result.total_normalized
                    - result.total_deduplicated
                ),
            )

            original_topic_for_profile = ""

            if expansion_result is not None:
                original_topic_for_profile = (
                    getattr(
                        expansion_result, "original_topic", ""
                    )
                    or ""
                )

            if not original_topic_for_profile:
                original_topic_for_profile = (
                    query_schema.topic or ""
                )

            profile = build_query_profile(
                query=query_schema.topic or "",
                corrected_topic=query_schema.corrected_topic or "",
                primary_concept=query_schema.primary_concept or "",
                level_profile=query_schema.level_profile or "",
                intent=query_schema.intent or "",
                target_domains=list(
                    query_schema.target_domains or []
                ),
                target_formats=list(
                    query_schema.target_formats or []
                ),
                preserve_tokens=list(
                    query_schema.preserve_tokens or []
                ),
                original_query=original_topic_for_profile,
            )

            result.concept_origin = profile.concept_origin

            self._trace.emit(
                "search_query_profile_built",
                concept_phrase=profile.concept_phrase,
                concept_tokens=profile.concept_tokens,
                concept_fingerprint=profile.concept_fingerprint,
                concept_origin=profile.concept_origin,
                qualifier_tokens=profile.qualifier_tokens,
                user_typed_tokens=profile.user_typed_tokens,
                user_typed_specificity=round(
                    profile.user_typed_specificity, 4
                ),
                expanded_tokens=profile.expanded_tokens,
                level_profile=profile.level_profile,
                level_words=profile.level_words,
                expected_source_types=profile.expected_source_types,
                specificity_score=round(
                    profile.specificity_score, 4
                ),
                has_specific_concept=profile.has_specific_concept,
            )

            level_for_classifier = (
                query_schema.level_profile
                or query_schema.detected_level
                or query_schema.level
                or ""
            )

            try:
                classified_sources = (
                    self._difficulty_classifier.classify_sources(
                        deduplicated_sources,
                        level=level_for_classifier,
                    )
                )
            except Exception as exc:
                self._logger.warning(
                    f"Difficulty classification failed: {exc}"
                )
                classified_sources = deduplicated_sources

            self._trace.emit(
                "search_difficulty_classified",
                input_count=len(deduplicated_sources),
                output_count=len(classified_sources),
                level=level_for_classifier,
                level_profile=query_schema.level_profile or "",
            )

            valid_sources = self._validator.validate_sources(
                classified_sources,
                query_schema,
                strict=True,
                profile=profile,
            )

            if not valid_sources:
                self._trace.emit(
                    "search_validation_strict_failed",
                    total_deduplicated=result.total_deduplicated,
                    fallback="lenient",
                )

                valid_sources = self._validator.validate_sources(
                    classified_sources,
                    query_schema,
                    strict=False,
                    profile=profile,
                )

            if not valid_sources:
                self._trace.emit(
                    "search_validation_lenient_failed",
                    fallback="no_query",
                )

                valid_sources = self._validator.validate_sources(
                    classified_sources,
                    None,
                    strict=False,
                    profile=None,
                )

            llm_relevance_allowed = (
                self._llm_relevance.enabled
                and resolved_mode == constants.MODE_DEEP
            )

            if llm_relevance_allowed and valid_sources:
                try:
                    before_count = len(valid_sources)
                    valid_sources = (
                        await self._llm_relevance.filter_sources(
                            valid_sources,
                            profile,
                            mode=resolved_mode,
                        )
                    )

                    self._trace.emit(
                        "search_llm_relevance_applied",
                        input_count=before_count,
                        output_count=len(valid_sources),
                        dropped_count=(
                            before_count - len(valid_sources)
                        ),
                        mode=resolved_mode,
                    )
                except Exception as exc:
                    self._logger.warning(
                        f"LLM relevance filter failed: {exc}"
                    )
                    self._trace.emit(
                        "search_llm_relevance_failed",
                        error=str(exc),
                        mode=resolved_mode,
                    )
            elif valid_sources:
                self._trace.emit(
                    "search_llm_relevance_skipped",
                    reason=(
                        "mode_not_deep"
                        if not llm_relevance_allowed
                        else "disabled_by_config"
                    ),
                    mode=resolved_mode,
                    valid_count=len(valid_sources),
                )

            result.total_valid = len(valid_sources)

            result.sources = self._select_with_diversity(
                valid_sources,
                query_schema.max_results,
                level_profile=query_schema.level_profile or "",
                beginner_ratio=query_schema.beginner_ratio,
                min_beginner_sources=(
                    query_schema.min_beginner_sources
                ),
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
                level_profile=level_profile,
                beginner_ratio=beginner_ratio,
                min_beginner_sources=min_beginner_sources,
                exclude_research_platforms=(
                    exclude_research_platforms
                ),
                preserve_tokens=preserve_tokens,
                concept_origin=profile.concept_origin,
                target_domains=target_domains,
                domain_routed=domain_routed,
                domain_expanded=domain_expanded,
                latency_ms=round(result.latency_ms, 1),
                mode=resolved_mode,
                mode_min_sources=mode_min_sources,
            )

            return result
        except Exception as exc:
            self._logger.error(
                f"Search orchestration failed: {exc}"
            )

            result.errors = [
                *errors,
                f"search_orchestration_failed: {exc}",
            ]
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

    def _emit_fetch_report(
        self, result: SearchOrchestrationResult
    ) -> None:
        try:
            report_getter = getattr(
                self._fetcher, "get_last_fetch_report", None
            )

            if not callable(report_getter):
                return

            report = report_getter()

            if not isinstance(report, dict) or not report:
                return

            result.fetch_report = report

            totals = report.get("totals", {}) or {}
            per_platform = report.get("per_platform", []) or []

            self._trace.emit(
                "search_fetch_report",
                totals=totals,
                per_platform=per_platform,
            )
        except Exception as exc:
            self._logger.warning(
                f"Failed to emit fetch report: {exc}"
            )

    def _merge_sources(
        self,
        primary: list[Any],
        secondary: list[Any],
    ) -> list[Any]:
        seen: set[str] = set()
        merged: list[Any] = []

        for source in [*primary, *secondary]:
            key = str(
                getattr(source, "source_id", "")
                or getattr(source, "url", "")
                or getattr(source, "title", "")
                or ""
            ).lower()

            if not key or key in seen:
                continue

            seen.add(key)
            merged.append(source)

        return merged

    def _fallback_platforms(
        self, current_platforms: list[Any]
    ) -> list[SourcePlatform]:
        values = _search_setting(
            "fallback_platforms",
            constants.DEFAULT_KEYLESS_FALLBACK_PLATFORMS,
        )

        if isinstance(values, str):
            values = [values]

        if not isinstance(values, (list, tuple, set)):
            values = []

        resolved: list[SourcePlatform] = []

        seen = {
            self._enum_value(platform)
            for platform in current_platforms or []
        }

        for value in values:
            try:
                platform = SourcePlatform(
                    str(value).strip().lower()
                )
            except Exception:
                continue

            if platform in resolved:
                continue

            if self._enum_value(platform) in seen:
                continue

            resolved.append(platform)

        return resolved

    def _select_with_diversity(
        self,
        sources: list[Any],
        max_results: int,
        level_profile: str = "",
        beginner_ratio: float = 0.0,
        min_beginner_sources: int = 0,
    ) -> list[Any]:
        limit = max(0, int(max_results))

        if len(sources) <= limit:
            return sources

        normalized_profile = (
            str(level_profile or "").strip().lower()
        )

        use_level_split = (
            normalized_profile
            in {"mixed", "beginner", "balanced"}
            and float(beginner_ratio or 0.0) > 0.0
        )

        if not use_level_split:
            return self._diversify_select(sources, limit)

        beginner_pool: list[Any] = []
        advanced_pool: list[Any] = []

        for source in sources:
            level = self._source_level(source)

            if level in {"beginner", "intermediate"}:
                beginner_pool.append(source)
            else:
                advanced_pool.append(source)

        target_beginner = max(
            int(min_beginner_sources or 0),
            int(round(limit * float(beginner_ratio))),
        )
        target_beginner = max(0, min(limit, target_beginner))
        target_advanced = limit - target_beginner

        selected_beginner = self._diversify_select(
            beginner_pool, target_beginner
        )

        shortfall = target_beginner - len(selected_beginner)

        extra_advanced = max(
            0, target_advanced + max(0, shortfall)
        )
        selected_advanced = self._diversify_select(
            advanced_pool, extra_advanced
        )

        final: list[Any] = []
        final.extend(selected_beginner)
        final.extend(selected_advanced)

        if len(final) < limit:
            chosen_ids = {id(source) for source in final}

            for source in sources:
                if id(source) in chosen_ids:
                    continue

                final.append(source)

                if len(final) >= limit:
                    break

        return final[:limit]

    def _diversify_select(
        self,
        sources: list[Any],
        limit: int,
    ) -> list[Any]:
        limit = max(0, int(limit))

        if limit <= 0:
            return []

        if len(sources) <= limit:
            return list(sources)

        platform_counts: dict[str, int] = {}
        selected: list[Any] = []
        remaining: list[Any] = []

        distinct_platforms = {
            self._enum_value(source.platform)
            for source in sources
        }

        max_per_platform = max(
            1,
            int(limit / max(1, len(distinct_platforms))),
        )

        for source in sources:
            platform = self._enum_value(source.platform)
            count = platform_counts.get(platform, 0)

            if count < max_per_platform:
                selected.append(source)
                platform_counts[platform] = count + 1
            else:
                remaining.append(source)

            if len(selected) >= limit:
                break

        if len(selected) < limit:
            for source in remaining:
                selected.append(source)

                if len(selected) >= limit:
                    break

        return selected[:limit]

    @staticmethod
    def _source_level(source: Any) -> str:
        difficulty = getattr(source, "difficulty", None)

        if difficulty is None:
            return ""

        return str(
            getattr(difficulty, "value", difficulty) or ""
        ).strip().lower()

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

    @staticmethod
    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _as_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return bool(value)

        if isinstance(value, str):
            text = value.strip().lower()

            if text in {"true", "yes", "on", "1"}:
                return True

            if text in {"false", "no", "off", "0"}:
                return False

        return bool(default)

    @staticmethod
    def _resolve_mode(mode: str) -> str:
        text = str(mode or "").strip().lower()

        if text in constants.VALID_MODES:
            return text

        return constants.DEFAULT_MODE

    @staticmethod
    def _mode_min_sources(mode: str) -> int:
        mode_defaults = constants.MODE_DEFAULTS.get(
            mode, constants.MODE_DEFAULTS[constants.DEFAULT_MODE]
        )

        try:
            return max(
                1,
                int(
                    mode_defaults.get(
                        "min_sources",
                        constants.MIN_SOURCES_BEFORE_FALLBACK,
                    )
                ),
            )
        except Exception:
            return int(constants.MIN_SOURCES_BEFORE_FALLBACK)