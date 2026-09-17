from __future__ import annotations

from typing import Any

from pydantic import Field

from core.models import (
    CoreModel,
    Difficulty,
    LearningPath,
    RankedSource,
    Source,
    SourcePlatform,
    SourceType,
)
from search.query_tokens import concept_tokens, tokenize
from utils.logger import get_logger
from utils.text import clean_text


class ReflectionScore(CoreModel):
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    level_match: float = Field(default=0.0, ge=0.0, le=1.0)
    source_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    path_coherence: float = Field(default=0.0, ge=0.0, le=1.0)
    composite: float = Field(default=0.0, ge=0.0, le=1.0)
    details: dict[str, Any] = Field(default_factory=dict)


class OutputEvaluator:
    _DEFAULT_WEIGHTS = {
        "relevance": 0.30,
        "completeness": 0.20,
        "level_match": 0.20,
        "source_quality": 0.15,
        "path_coherence": 0.15,
    }

    _DEFAULT_MIN_SOURCES = 3
    _DEFAULT_MIN_STEPS = 2
    _DEFAULT_TOP_N_FOR_RELEVANCE = 5

    _PLATFORM_QUALITY = {
        SourcePlatform.SEMANTIC_SCHOLAR.value: 0.95,
        SourcePlatform.ARXIV.value: 0.90,
        SourcePlatform.OPENALEX.value: 0.85,
        SourcePlatform.WIKIPEDIA.value: 0.80,
        SourcePlatform.GITHUB.value: 0.75,
        SourcePlatform.HUGGINGFACE.value: 0.70,
        SourcePlatform.WEB.value: 0.55,
    }

    _TYPE_QUALITY = {
        SourceType.PAPER.value: 0.90,
        SourceType.DOCUMENTATION.value: 0.85,
        SourceType.REPOSITORY.value: 0.80,
        SourceType.COURSE.value: 0.80,
        SourceType.MODEL.value: 0.75,
        SourceType.DATASET.value: 0.70,
        SourceType.VIDEO.value: 0.65,
        SourceType.BLOG.value: 0.55,
        SourceType.OTHER.value: 0.40,
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        min_sources: int | None = None,
        min_steps: int | None = None,
        top_n_for_relevance: int | None = None,
    ) -> None:
        self._weights = dict(weights or self._DEFAULT_WEIGHTS)
        self._min_sources = max(1, min_sources or self._DEFAULT_MIN_SOURCES)
        self._min_steps = max(1, min_steps or self._DEFAULT_MIN_STEPS)
        self._top_n = max(1, top_n_for_relevance or self._DEFAULT_TOP_N_FOR_RELEVANCE)
        self._logger = get_logger("agent.reflection.evaluator")

    def evaluate(
        self,
        ranked_sources: list[RankedSource],
        learning_path: LearningPath | None,
        topic: str,
        level: str = "",
        goal: str = "",
    ) -> ReflectionScore:
        relevance = self._score_relevance(ranked_sources, topic)
        completeness = self._score_completeness(ranked_sources, learning_path)
        level_match = self._score_level_match(ranked_sources, level)
        source_quality = self._score_source_quality(ranked_sources)
        path_coherence = self._score_path_coherence(learning_path, ranked_sources)

        composite = (
            relevance * self._weights.get("relevance", 0.30)
            + completeness * self._weights.get("completeness", 0.20)
            + level_match * self._weights.get("level_match", 0.20)
            + source_quality * self._weights.get("source_quality", 0.15)
            + path_coherence * self._weights.get("path_coherence", 0.15)
        )
        composite = max(0.0, min(1.0, composite))

        return ReflectionScore(
            relevance=round(relevance, 4),
            completeness=round(completeness, 4),
            level_match=round(level_match, 4),
            source_quality=round(source_quality, 4),
            path_coherence=round(path_coherence, 4),
            composite=round(composite, 4),
            details={
                "total_sources": len(ranked_sources),
                "total_steps": len(learning_path.steps) if learning_path else 0,
                "requested_level": level,
                "topic": topic,
            },
        )

    def _score_relevance(
        self,
        ranked_sources: list[RankedSource],
        topic: str,
    ) -> float:
        if not ranked_sources or not topic:
            return 0.0

        query_tokens = concept_tokens(clean_text(topic))
        if not query_tokens:
            query_tokens = set(tokenize(clean_text(topic).lower()))

        if not query_tokens:
            return 0.5

        top_sources = ranked_sources[: self._top_n]
        match_scores: list[float] = []

        for ranked in top_sources:
            source_text = self._source_text(ranked.source)
            source_tokens = set(tokenize(source_text.lower()))

            if not source_tokens:
                match_scores.append(0.0)
                continue

            matched = sum(1 for token in query_tokens if token in source_tokens)
            match_ratio = matched / len(query_tokens)
            match_scores.append(match_ratio)

        if not match_scores:
            return 0.0

        avg_match = sum(match_scores) / len(match_scores)
        top_bonus = max(match_scores) * 0.2 if match_scores else 0.0

        return max(0.0, min(1.0, avg_match * 0.8 + top_bonus))

    def _score_completeness(
        self,
        ranked_sources: list[RankedSource],
        learning_path: LearningPath | None,
    ) -> float:
        source_score = min(1.0, len(ranked_sources) / self._min_sources)

        if learning_path is None:
            return source_score * 0.6

        step_count = len(learning_path.steps)
        step_score = min(1.0, step_count / self._min_steps)

        resources_per_step: list[int] = []
        for step in learning_path.steps:
            resources_per_step.append(len(step.resources))

        if resources_per_step:
            avg_resources = sum(resources_per_step) / len(resources_per_step)
            resource_score = min(1.0, avg_resources / 2.0)
        else:
            resource_score = 0.0

        return (source_score * 0.4) + (step_score * 0.35) + (resource_score * 0.25)

    def _score_level_match(
        self,
        ranked_sources: list[RankedSource],
        level: str,
    ) -> float:
        if not level or not ranked_sources:
            return 0.5

        requested = clean_text(level).lower()

        if requested not in ("beginner", "intermediate", "advanced", "mixed"):
            return 0.5

        if requested == "mixed":
            return 0.7

        top_sources = ranked_sources[: self._top_n]
        match_count = 0
        total_with_difficulty = 0

        for ranked in top_sources:
            difficulty = ranked.source.difficulty
            if difficulty is None:
                continue

            total_with_difficulty += 1
            difficulty_value = (
                difficulty.value
                if hasattr(difficulty, "value")
                else str(difficulty)
            )

            if requested == "beginner":
                if difficulty_value in ("beginner", "intermediate"):
                    match_count += 1
            elif requested == "intermediate":
                if difficulty_value in ("beginner", "intermediate", "advanced"):
                    match_count += 1
            elif requested == "advanced":
                if difficulty_value in ("intermediate", "advanced"):
                    match_count += 1

        if total_with_difficulty == 0:
            return 0.5

        return match_count / total_with_difficulty

    def _score_source_quality(
        self,
        ranked_sources: list[RankedSource],
    ) -> float:
        if not ranked_sources:
            return 0.0

        quality_scores: list[float] = []

        for ranked in ranked_sources[: self._top_n]:
            source = ranked.source
            platform_value = self._enum_value(source.platform)
            type_value = self._enum_value(source.source_type)

            platform_score = self._PLATFORM_QUALITY.get(platform_value, 0.40)
            type_score = self._TYPE_QUALITY.get(type_value, 0.40)

            confidence_bonus = (ranked.confidence or 0.0) * 0.1

            combined = (platform_score * 0.5) + (type_score * 0.4) + confidence_bonus
            quality_scores.append(min(1.0, combined))

        if not quality_scores:
            return 0.0

        return sum(quality_scores) / len(quality_scores)

    def _score_path_coherence(
        self,
        learning_path: LearningPath | None,
        ranked_sources: list[RankedSource],
    ) -> float:
        if learning_path is None:
            return 0.0

        steps = learning_path.steps
        if not steps:
            return 0.0

        if len(steps) == 1:
            return 0.4

        score = 0.0

        step_numbers = [step.step for step in steps]
        is_sequential = all(
            step_numbers[i] < step_numbers[i + 1]
            for i in range(len(step_numbers) - 1)
        )
        if is_sequential:
            score += 0.3

        all_have_objectives = all(
            bool(clean_text(step.objective)) for step in steps
        )
        if all_have_objectives:
            score += 0.2

        all_have_resources = all(len(step.resources) > 0 for step in steps)
        if all_have_resources:
            score += 0.2

        source_ids_in_path: set[str] = set()
        for step in steps:
            for resource in step.resources:
                source_ids_in_path.add(resource.source.source_id)

        ranked_ids = {r.source.source_id for r in ranked_sources}
        overlap = len(source_ids_in_path & ranked_ids)
        if source_ids_in_path:
            overlap_ratio = overlap / len(source_ids_in_path)
            score += overlap_ratio * 0.3

        return max(0.0, min(1.0, score))

    def _source_text(self, source: Source) -> str:
        parts = [source.title]
        if source.abstract:
            parts.append(source.abstract)
        metadata = source.metadata if isinstance(source.metadata, dict) else {}
        description = metadata.get("description", "")
        if isinstance(description, str) and description:
            parts.append(description)
        return " ".join(parts)

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value))