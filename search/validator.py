from __future__ import annotations

from typing import Any

from core.exceptions import ValidationError as SearchValidationError
from core.models import Difficulty, Source, SourcePlatform, SourceType
from core.schemas import SearchQuerySchema
from search.query_profile import QueryProfile, build_query_profile
from search.relevance import (
    passes_concept_presence_gate,
    passes_title_fingerprint_gate,
    source_relevance_vector,
)
from utils.logger import get_logger, get_trace_logger
from utils.text import clean_text
from utils.url import is_valid_url


class SourceValidator:
    def __init__(
        self,
        min_title_length: int = 3,
        max_title_length: int = 500,
        max_abstract_length: int = 5000,
        strict_threshold: float = 0.30,
        relaxed_threshold: float = 0.12,
    ) -> None:
        self._min_title_length = max(1, min_title_length)
        self._max_title_length = max(self._min_title_length, max_title_length)
        self._max_abstract_length = max(100, max_abstract_length)
        self._strict_threshold = max(0.0, min(1.0, strict_threshold))
        self._relaxed_threshold = max(0.0, min(1.0, relaxed_threshold))
        self._logger = get_logger("search.validator")
        self._trace = get_trace_logger()
        self.last_warnings: list[str] = []

    def validate_query(self, query: Any) -> SearchQuerySchema:
        try:
            if isinstance(query, SearchQuerySchema):
                return query

            if isinstance(query, str):
                return SearchQuerySchema(topic=query)

            if isinstance(query, dict):
                return SearchQuerySchema.model_validate(query)

            raise SearchValidationError(
                "Unsupported search query type",
                details={"type": type(query).__name__},
            )
        except SearchValidationError:
            raise
        except Exception as exc:
            raise SearchValidationError(
                "Invalid search query",
                details={"error": str(exc)},
            ) from exc

    def validate_sources(
        self,
        sources: list[Any],
        query: Any = None,
        strict: bool = True,
        profile: QueryProfile | None = None,
    ) -> list[Source]:
        self.last_warnings = []

        query_schema = self._coerce_query(query)

        if profile is None:
            profile = self._build_profile(query_schema)

        threshold = self._strict_threshold if strict else self._relaxed_threshold
        beginner_only = profile.level_profile == "beginner"

        valid_sources: list[Source] = []
        dropped_counts: dict[str, int] = {}
        dropped_samples: list[dict[str, Any]] = []
        input_count = 0

        for source in sources:
            input_count += 1

            if not self.validate_source(source):
                self._record_drop(
                    dropped_counts,
                    dropped_samples,
                    source,
                    "structural",
                )
                continue

            if strict:
                if not passes_concept_presence_gate(source, profile):
                    self._record_drop(
                        dropped_counts,
                        dropped_samples,
                        source,
                        "concept_presence",
                    )
                    continue

                if not passes_title_fingerprint_gate(source, profile):
                    self._record_drop(
                        dropped_counts,
                        dropped_samples,
                        source,
                        "title_fingerprint",
                    )
                    continue

                if beginner_only and self._is_advanced_source(source):
                    self._record_drop(
                        dropped_counts,
                        dropped_samples,
                        source,
                        "level_advanced_for_beginner",
                    )
                    continue
            else:
                if not passes_concept_presence_gate(source, profile):
                    self._record_drop(
                        dropped_counts,
                        dropped_samples,
                        source,
                        "concept_presence",
                    )
                    continue

            try:
                vector, score = source_relevance_vector(source, profile)
            except Exception as exc:
                self._logger.warning(f"Relevance vector failed: {exc}")
                self._record_drop(
                    dropped_counts,
                    dropped_samples,
                    source,
                    "relevance_error",
                )
                continue

            if score < threshold:
                self._record_drop(
                    dropped_counts,
                    dropped_samples,
                    source,
                    "below_threshold",
                    score=round(score, 4),
                )
                continue

            self._annotate_source(source, vector, score, profile, beginner_only)

            valid_sources.append(source)

        self._emit_completion(
            valid_sources=valid_sources,
            dropped_counts=dropped_counts,
            dropped_samples=dropped_samples,
            strict=strict,
            profile=profile,
            threshold=threshold,
            input_count=input_count,
        )

        if strict and input_count > 0 and not valid_sources:
            self._add_warning("strict_dropped_all_sources")

        if beginner_only and not valid_sources:
            self._add_warning("no_beginner_sources_found")

        return valid_sources

    def validate_source(self, source: Any) -> bool:
        try:
            return self._validate_source(source)
        except Exception as exc:
            self._logger.warning(f"Source validation failed: {exc}")
            return False

    def get_warnings(self) -> list[str]:
        return list(self.last_warnings)

    def _validate_source(self, source: Any) -> bool:
        if not isinstance(source, Source):
            return False

        title = clean_text(source.title)

        if len(title) < self._min_title_length:
            return False

        if len(title) > self._max_title_length:
            return False

        lower_title = title.lower()

        if "this paper has been withdrawn" in lower_title:
            return False

        if lower_title.strip() in {"withdrawn", "retracted"}:
            return False

        if not source.url or not is_valid_url(source.url):
            return False

        try:
            SourcePlatform(self._enum_value(source.platform))
            SourceType(self._enum_value(source.source_type))
        except Exception:
            return False

        if source.abstract is not None and len(source.abstract) > self._max_abstract_length:
            return False

        if source.year is not None and not (0 <= source.year <= 2100):
            return False

        if source.citation_count is not None and source.citation_count < 0:
            return False

        if source.score is not None and source.score < 0:
            return False

        if not isinstance(source.metadata, dict):
            return False

        return True

    def _coerce_query(self, query: Any) -> Any:
        if query is None:
            return None

        if isinstance(query, SearchQuerySchema):
            return query

        if isinstance(query, str):
            try:
                return SearchQuerySchema(topic=query)
            except Exception:
                return {"topic": query}

        if isinstance(query, dict):
            try:
                return SearchQuerySchema.model_validate(query)
            except Exception:
                return query

        return None

    def _build_profile(self, query_schema: Any) -> QueryProfile:
        if query_schema is None:
            return build_query_profile("")

        if isinstance(query_schema, SearchQuerySchema):
            return build_query_profile(
                query=query_schema.topic or "",
                corrected_topic=getattr(query_schema, "corrected_topic", "") or "",
                primary_concept=getattr(query_schema, "primary_concept", "") or "",
                level_profile=getattr(query_schema, "level_profile", "") or "",
                intent=getattr(query_schema, "intent", "") or "",
                target_domains=list(getattr(query_schema, "target_domains", []) or []),
                target_formats=list(getattr(query_schema, "target_formats", []) or []),
                preserve_tokens=list(getattr(query_schema, "preserve_tokens", []) or []),
                original_query=getattr(query_schema, "topic", "") or "",
            )

        if isinstance(query_schema, dict):
            return build_query_profile(
                query=str(query_schema.get("topic", "") or ""),
                corrected_topic=str(query_schema.get("corrected_topic", "") or ""),
                primary_concept=str(query_schema.get("primary_concept", "") or ""),
                level_profile=str(query_schema.get("level_profile", "") or ""),
                intent=str(query_schema.get("intent", "") or ""),
                target_domains=list(query_schema.get("target_domains", []) or []),
                target_formats=list(query_schema.get("target_formats", []) or []),
                preserve_tokens=list(query_schema.get("preserve_tokens", []) or []),
                original_query=str(query_schema.get("topic", "") or ""),
            )

        if isinstance(query_schema, str):
            return build_query_profile(
                query=query_schema,
                original_query=query_schema,
            )

        return build_query_profile("")

    def _is_advanced_source(self, source: Source) -> bool:
        difficulty = getattr(source, "difficulty", None)

        if difficulty is None:
            return False

        if isinstance(difficulty, Difficulty):
            return difficulty == Difficulty.ADVANCED

        text = str(difficulty).strip().lower()

        return text == "advanced"

    def _annotate_source(
        self,
        source: Source,
        vector: dict[str, float],
        score: float,
        profile: QueryProfile,
        beginner_only: bool,
    ) -> None:
        try:
            source.metadata["validator_relevance"] = round(score, 4)
            source.metadata["validator_vector"] = {
                key: round(value, 4) for key, value in vector.items()
            }
            source.metadata["validator_level_profile"] = profile.level_profile or ""
            source.metadata["validator_has_abbreviation"] = bool(
                profile.has_abbreviation
            )

            difficulty = getattr(source, "difficulty", None)
            difficulty_text = str(
                getattr(difficulty, "value", difficulty) or ""
            ).strip().lower()

            is_advanced = difficulty_text == "advanced"
            is_beginner_safe = difficulty_text in {"beginner", "intermediate"}

            source.metadata["validator_beginner_safe"] = is_beginner_safe
            source.metadata["validator_advanced_signal"] = is_advanced

            if beginner_only:
                if is_advanced:
                    source.metadata["learner_alignment_penalty"] = 0.20
                elif is_beginner_safe:
                    source.metadata["learner_alignment_boost"] = 1.25
        except Exception:
            pass

    def _record_drop(
        self,
        counter: dict[str, int],
        samples: list[dict[str, Any]],
        source: Any,
        reason: str,
        score: float | None = None,
    ) -> None:
        counter[reason] = counter.get(reason, 0) + 1

        if len(samples) >= 50:
            return

        try:
            title = clean_text(getattr(source, "title", ""))[:80]
            source_id = str(getattr(source, "source_id", ""))
            platform = self._enum_value(getattr(source, "platform", ""))
        except Exception:
            title = ""
            source_id = ""
            platform = ""

        entry = {
            "source_id": source_id,
            "title": title,
            "platform": platform,
            "reason": reason,
        }

        if score is not None:
            entry["score"] = score

        samples.append(entry)

        try:
            self._trace.emit(
                "validator_drop",
                source_id=source_id,
                title=title,
                platform=platform,
                reason=reason,
                score=score,
            )
        except Exception:
            pass

    def _emit_completion(
        self,
        valid_sources: list[Source],
        dropped_counts: dict[str, int],
        dropped_samples: list[dict[str, Any]],
        strict: bool,
        profile: QueryProfile,
        threshold: float,
        input_count: int,
    ) -> None:
        try:
            self._trace.emit(
                "validator_completed",
                strict=strict,
                threshold=round(threshold, 4),
                level_profile=profile.level_profile or "",
                concept_phrase=profile.concept_phrase or "",
                concept_fingerprint=profile.concept_fingerprint,
                fingerprint_size=len(profile.concept_fingerprint),
                user_typed_tokens=profile.user_typed_tokens,
                user_typed_specificity=round(profile.user_typed_specificity, 4),
                has_abbreviation=profile.has_abbreviation,
                has_specific_concept=profile.has_specific_concept,
                input_count=input_count,
                valid_count=len(valid_sources),
                dropped_total=sum(dropped_counts.values()),
                dropped_counts=dict(dropped_counts),
                dropped_samples=dropped_samples,
            )
        except Exception:
            pass

    def _add_warning(self, warning: str) -> None:
        if warning not in self.last_warnings:
            self.last_warnings.append(warning)

        self._logger.warning(warning)

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value) or "")