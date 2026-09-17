from __future__ import annotations

from typing import Any

from pydantic import Field

from agent.reflection.decider import (
    DecisionResult,
    ReflectionDecider,
    ReflectionDecision,
)
from agent.reflection.evaluator import OutputEvaluator, ReflectionScore
from core.models import CoreModel, LearningPath, RankedSource
from utils.logger import get_logger, get_trace_logger


class ReflectionOutcome(CoreModel):
    final_decision: ReflectionDecision
    final_score: float = Field(default=0.0, ge=0.0, le=1.0)
    total_iterations: int = Field(default=0, ge=0)
    scores_history: list[ReflectionScore] = Field(default_factory=list)
    decisions_history: list[DecisionResult] = Field(default_factory=list)
    accepted: bool = False
    retries_used: int = Field(default=0, ge=0)


class ReflectionEngine:
    _DEFAULT_MAX_RETRIES = 2

    def __init__(
        self,
        evaluator: OutputEvaluator | None = None,
        decider: ReflectionDecider | None = None,
        max_retries: int | None = None,
    ) -> None:
        self._evaluator = evaluator or OutputEvaluator()
        self._decider = decider or ReflectionDecider()
        self._max_retries = max(0, max_retries if max_retries is not None else self._DEFAULT_MAX_RETRIES)
        self._logger = get_logger("agent.reflection.engine")
        self._trace = get_trace_logger()

    @property
    def max_retries(self) -> int:
        return self._max_retries

    async def reflect(
        self,
        ranked_sources: list[RankedSource],
        learning_path: LearningPath | None,
        topic: str,
        level: str = "",
        goal: str = "",
        remaining_budget: int | None = None,
    ) -> ReflectionOutcome:
        scores_history: list[ReflectionScore] = []
        decisions_history: list[DecisionResult] = []
        retries_used = 0

        self._trace.emit(
            "reflection_started",
            topic=topic,
            level=level,
            max_retries=self._max_retries,
            source_count=len(ranked_sources),
        )

        for iteration in range(self._max_retries + 1):
            score = self._evaluator.evaluate(
                ranked_sources=ranked_sources,
                learning_path=learning_path,
                topic=topic,
                level=level,
                goal=goal,
            )
            scores_history.append(score)

            self._trace.emit(
                "reflection_evaluation",
                iteration=iteration,
                composite_score=score.composite,
                relevance=score.relevance,
                completeness=score.completeness,
                level_match=score.level_match,
                source_quality=score.source_quality,
                path_coherence=score.path_coherence,
            )

            decision = self._decider.decide(
                score=score,
                remaining_budget=remaining_budget,
                current_retry_count=retries_used,
                max_retries=self._max_retries,
            )
            decisions_history.append(decision)

            self._trace.emit(
                "reflection_decision",
                iteration=iteration,
                decision=decision.decision.value,
                score=decision.score,
                reasoning=decision.reasoning,
                retry_allowed=decision.retry_allowed,
                weakest_criterion=decision.weakest_criterion,
            )

            if decision.decision == ReflectionDecision.ACCEPT:
                self._trace.emit(
                    "reflection_completed",
                    outcome="accepted",
                    iterations=iteration + 1,
                    final_score=score.composite,
                )
                return ReflectionOutcome(
                    final_decision=ReflectionDecision.ACCEPT,
                    final_score=score.composite,
                    total_iterations=iteration + 1,
                    scores_history=scores_history,
                    decisions_history=decisions_history,
                    accepted=True,
                    retries_used=retries_used,
                )

            if decision.decision == ReflectionDecision.MINOR_FIX:
                self._trace.emit(
                    "reflection_completed",
                    outcome="minor_fix_accepted",
                    iterations=iteration + 1,
                    final_score=score.composite,
                )
                return ReflectionOutcome(
                    final_decision=ReflectionDecision.MINOR_FIX,
                    final_score=score.composite,
                    total_iterations=iteration + 1,
                    scores_history=scores_history,
                    decisions_history=decisions_history,
                    accepted=True,
                    retries_used=retries_used,
                )

            if decision.decision == ReflectionDecision.BUDGET_BLOCKED:
                self._trace.emit(
                    "reflection_completed",
                    outcome="budget_blocked",
                    iterations=iteration + 1,
                    final_score=score.composite,
                )
                return ReflectionOutcome(
                    final_decision=ReflectionDecision.BUDGET_BLOCKED,
                    final_score=score.composite,
                    total_iterations=iteration + 1,
                    scores_history=scores_history,
                    decisions_history=decisions_history,
                    accepted=True,
                    retries_used=retries_used,
                )

            if decision.retry_allowed and iteration < self._max_retries:
                retries_used += 1
                self._trace.emit(
                    "reflection_retry",
                    iteration=iteration,
                    retries_used=retries_used,
                    decision=decision.decision.value,
                    suggestions=decision.suggestions,
                )
                continue

            break

        final_score = scores_history[-1].composite if scores_history else 0.0
        final_decision = (
            decisions_history[-1].decision
            if decisions_history
            else ReflectionDecision.BUDGET_BLOCKED
        )

        self._trace.emit(
            "reflection_completed",
            outcome="max_iterations_reached",
            iterations=len(scores_history),
            final_score=final_score,
        )

        return ReflectionOutcome(
            final_decision=final_decision,
            final_score=final_score,
            total_iterations=len(scores_history),
            scores_history=scores_history,
            decisions_history=decisions_history,
            accepted=final_score >= self._decider._retry_threshold,
            retries_used=retries_used,
        )