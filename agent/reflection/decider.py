from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field

from agent.reflection.evaluator import ReflectionScore
from core.models import CoreModel
from utils.logger import get_logger


class ReflectionDecision(str, Enum):
    ACCEPT = "accept"
    MINOR_FIX = "minor_fix"
    RETRY = "retry"
    FULL_REDO = "full_redo"
    BUDGET_BLOCKED = "budget_blocked"


class DecisionResult(CoreModel):
    decision: ReflectionDecision
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = ""
    should_proceed: bool = True
    retry_allowed: bool = False
    weakest_criterion: str = ""
    suggestions: list[str] = Field(default_factory=list)


class ReflectionDecider:
    _DEFAULT_ACCEPT_THRESHOLD = 0.80
    _DEFAULT_MINOR_FIX_THRESHOLD = 0.60
    _DEFAULT_RETRY_THRESHOLD = 0.40
    _DEFAULT_MIN_BUDGET_FOR_RETRY = 5000

    def __init__(
        self,
        accept_threshold: float | None = None,
        minor_fix_threshold: float | None = None,
        retry_threshold: float | None = None,
        min_budget_for_retry: int | None = None,
    ) -> None:
        self._accept_threshold = self._clamp_threshold(
            accept_threshold or self._DEFAULT_ACCEPT_THRESHOLD
        )
        self._minor_fix_threshold = self._clamp_threshold(
            minor_fix_threshold or self._DEFAULT_MINOR_FIX_THRESHOLD
        )
        self._retry_threshold = self._clamp_threshold(
            retry_threshold or self._DEFAULT_RETRY_THRESHOLD
        )
        self._min_budget_for_retry = max(
            0, min_budget_for_retry or self._DEFAULT_MIN_BUDGET_FOR_RETRY
        )
        self._logger = get_logger("agent.reflection.decider")

    def decide(
        self,
        score: ReflectionScore,
        remaining_budget: int | None = None,
        current_retry_count: int = 0,
        max_retries: int = 2,
    ) -> DecisionResult:
        composite = score.composite
        weakest = self._find_weakest_criterion(score)
        suggestions = self._generate_suggestions(score, weakest)

        if composite >= self._accept_threshold:
            return DecisionResult(
                decision=ReflectionDecision.ACCEPT,
                score=composite,
                reasoning=(
                    f"Score {composite:.3f} meets accept threshold "
                    f"{self._accept_threshold:.2f}"
                ),
                should_proceed=True,
                retry_allowed=False,
                weakest_criterion=weakest,
                suggestions=suggestions,
            )

        if composite >= self._minor_fix_threshold:
            return DecisionResult(
                decision=ReflectionDecision.MINOR_FIX,
                score=composite,
                reasoning=(
                    f"Score {composite:.3f} is between minor_fix "
                    f"({self._minor_fix_threshold:.2f}) and accept "
                    f"({self._accept_threshold:.2f})"
                ),
                should_proceed=True,
                retry_allowed=False,
                weakest_criterion=weakest,
                suggestions=suggestions,
            )

        budget_sufficient = self._check_budget(remaining_budget)
        retries_remaining = current_retry_count < max_retries

        if composite >= self._retry_threshold:
            if budget_sufficient and retries_remaining:
                return DecisionResult(
                    decision=ReflectionDecision.RETRY,
                    score=composite,
                    reasoning=(
                        f"Score {composite:.3f} is between retry "
                        f"({self._retry_threshold:.2f}) and minor_fix "
                        f"({self._minor_fix_threshold:.2f}). "
                        f"Retrying with improved approach."
                    ),
                    should_proceed=True,
                    retry_allowed=True,
                    weakest_criterion=weakest,
                    suggestions=suggestions,
                )
            else:
                return DecisionResult(
                    decision=ReflectionDecision.BUDGET_BLOCKED,
                    score=composite,
                    reasoning=(
                        f"Score {composite:.3f} warrants retry but "
                        f"budget_sufficient={budget_sufficient}, "
                        f"retries_remaining={retries_remaining}. "
                        f"Accepting current output."
                    ),
                    should_proceed=True,
                    retry_allowed=False,
                    weakest_criterion=weakest,
                    suggestions=suggestions,
                )

        if budget_sufficient and retries_remaining:
            return DecisionResult(
                decision=ReflectionDecision.FULL_REDO,
                score=composite,
                reasoning=(
                    f"Score {composite:.3f} is below retry threshold "
                    f"{self._retry_threshold:.2f}. Full redo required."
                ),
                should_proceed=True,
                retry_allowed=True,
                weakest_criterion=weakest,
                suggestions=suggestions,
            )

        return DecisionResult(
            decision=ReflectionDecision.BUDGET_BLOCKED,
            score=composite,
            reasoning=(
                f"Score {composite:.3f} is below retry threshold but "
                f"budget or retry limit prevents redo. "
                f"Accepting current output as final."
            ),
            should_proceed=True,
            retry_allowed=False,
            weakest_criterion=weakest,
            suggestions=suggestions,
        )

    def _find_weakest_criterion(self, score: ReflectionScore) -> str:
        criteria = {
            "relevance": score.relevance,
            "completeness": score.completeness,
            "level_match": score.level_match,
            "source_quality": score.source_quality,
            "path_coherence": score.path_coherence,
        }
        return min(criteria, key=criteria.get)

    def _generate_suggestions(
        self,
        score: ReflectionScore,
        weakest: str,
    ) -> list[str]:
        suggestions: list[str] = []

        if score.relevance < 0.5:
            suggestions.append(
                "Search with more specific keywords targeting the primary concept"
            )

        if score.completeness < 0.5:
            suggestions.append(
                "Increase source count or add more learning path steps"
            )

        if score.level_match < 0.5:
            suggestions.append(
                "Filter or re-rank sources to better match the requested difficulty level"
            )

        if score.source_quality < 0.5:
            suggestions.append(
                "Prioritize sources from higher-authority platforms"
            )

        if score.path_coherence < 0.5:
            suggestions.append(
                "Rebuild learning path with logical step progression and resource alignment"
            )

        if not suggestions:
            suggestions.append("Output quality is acceptable")

        return suggestions

    def _check_budget(self, remaining_budget: int | None) -> bool:
        if remaining_budget is None:
            return True
        return remaining_budget >= self._min_budget_for_retry

    @staticmethod
    def _clamp_threshold(value: float) -> float:
        return max(0.0, min(1.0, value))