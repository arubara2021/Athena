from __future__ import annotations

from typing import Any

from pydantic import Field

from core.models import CoreModel
from utils.logger import get_logger
from utils.text import clean_text


DEFAULT_QUALITY_THRESHOLD = 0.6
DEFAULT_MIN_FINDINGS_FOR_STEP_COMPLETE = 3
DEFAULT_MIN_RANKED_FOR_STEP_COMPLETE = 3


class ReflectResult(CoreModel):
    should_advance: bool = False
    should_complete: bool = False
    should_retry: bool = False
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    step_complete: bool = False
    reasoning: str = ""
    suggestions: list[str] = Field(default_factory=list)


class ReflectPhase:
    def __init__(
        self,
        quality_threshold: float | None = None,
        min_findings_for_step: int | None = None,
        min_ranked_for_step: int | None = None,
        enable_deep_reflection: bool = True,
    ) -> None:
        self._quality_threshold = (
            quality_threshold
            if quality_threshold is not None
            else DEFAULT_QUALITY_THRESHOLD
        )
        self._min_findings = (
            min_findings_for_step or DEFAULT_MIN_FINDINGS_FOR_STEP_COMPLETE
        )
        self._min_ranked = (
            min_ranked_for_step or DEFAULT_MIN_RANKED_FOR_STEP_COMPLETE
        )
        self._enable_deep = enable_deep_reflection
        self._logger = get_logger("agent.loop.reflect")

    def reflect(
        self,
        state: Any,
        observe_result: Any,
        think_result: Any,
        act_result: Any,
    ) -> ReflectResult:
        if think_result.is_complete:
            return ReflectResult(
                should_complete=True,
                step_complete=True,
                quality_score=1.0,
                reasoning="Goal completion requested",
            )

        if think_result.is_advance:
            return ReflectResult(
                should_advance=True,
                step_complete=True,
                quality_score=0.8,
                reasoning="Step advancement requested",
            )

        if not act_result.success and not act_result.skipped:
            if state.actions_in_current_step < state.max_actions_per_step:
                return ReflectResult(
                    should_retry=True,
                    step_complete=False,
                    quality_score=0.2,
                    reasoning=f"Action failed: {act_result.error}. Retrying with different approach.",
                    suggestions=["Try a different tool or adjust parameters"],
                )
            else:
                return ReflectResult(
                    should_advance=True,
                    step_complete=True,
                    quality_score=0.3,
                    reasoning="Action failed and max retries reached. Advancing to next step.",
                )

        current_step = state.current_step
        if current_step is None:
            return ReflectResult(
                should_complete=True,
                step_complete=True,
                quality_score=0.9,
                reasoning="No more steps in plan",
            )

        step_action = self._enum_value(
            getattr(current_step, "action_type", None)
        )

        if step_action == "search":
            return self._reflect_search_step(state, observe_result)
        elif step_action == "analyze":
            return self._reflect_analyze_step(state, observe_result)
        elif step_action == "generate":
            return self._reflect_generate_step(state, observe_result)
        elif step_action == "read":
            return self._reflect_read_step(state, observe_result)
        else:
            return self._reflect_generic_step(state, observe_result)

    def _reflect_search_step(self, state: Any, observe_result: Any) -> ReflectResult:
        findings_count = len(state.findings)

        if findings_count >= self._min_findings:
            return ReflectResult(
                should_advance=True,
                step_complete=True,
                quality_score=min(1.0, findings_count / (self._min_findings * 2)),
                reasoning=f"Search step complete: {findings_count} sources found",
            )

        if state.actions_in_current_step >= state.max_actions_per_step:
            return ReflectResult(
                should_advance=True,
                step_complete=True,
                quality_score=0.5,
                reasoning=f"Max actions reached with {findings_count} sources. Advancing.",
            )

        return ReflectResult(
            should_retry=True,
            step_complete=False,
            quality_score=findings_count / max(self._min_findings, 1),
            reasoning=f"Only {findings_count} sources found, need at least {self._min_findings}",
            suggestions=["Try different search queries or platforms"],
        )

    def _reflect_analyze_step(self, state: Any, observe_result: Any) -> ReflectResult:
        ranked_count = len(state.ranked_sources)

        if observe_result.ranked_updated and ranked_count >= self._min_ranked:
            return ReflectResult(
                should_advance=True,
                step_complete=True,
                quality_score=min(1.0, ranked_count / 10),
                reasoning=f"Analysis complete: {ranked_count} sources ranked",
            )

        if state.actions_in_current_step >= state.max_actions_per_step:
            return ReflectResult(
                should_advance=True,
                step_complete=True,
                quality_score=0.5,
                reasoning="Max actions reached during analysis. Advancing.",
            )

        return ReflectResult(
            should_retry=True,
            step_complete=False,
            quality_score=0.3,
            reasoning=f"Ranking incomplete: {ranked_count} ranked so far",
            suggestions=["Ensure sources are available for ranking"],
        )

    def _reflect_generate_step(self, state: Any, observe_result: Any) -> ReflectResult:
        if observe_result.learning_path_generated:
            return ReflectResult(
                should_advance=True,
                step_complete=True,
                quality_score=0.9,
                reasoning="Learning path generated successfully",
            )

        if state.actions_in_current_step >= state.max_actions_per_step:
            return ReflectResult(
                should_advance=True,
                step_complete=True,
                quality_score=0.5,
                reasoning="Max actions reached during generation. Advancing.",
            )

        return ReflectResult(
            should_retry=True,
            step_complete=False,
            quality_score=0.3,
            reasoning="Generation not yet complete",
            suggestions=["Ensure ranked sources are available for path generation"],
        )

    def _reflect_read_step(self, state: Any, observe_result: Any) -> ReflectResult:
        if observe_result.is_sufficient:
            return ReflectResult(
                should_advance=True,
                step_complete=True,
                quality_score=0.8,
                reasoning="Read step complete",
            )

        if state.actions_in_current_step >= state.max_actions_per_step:
            return ReflectResult(
                should_advance=True,
                step_complete=True,
                quality_score=0.5,
                reasoning="Max actions reached during read. Advancing.",
            )

        return ReflectResult(
            should_retry=True,
            step_complete=False,
            quality_score=0.3,
            reasoning="Read result insufficient",
            suggestions=["Try reading a different source"],
        )

    def _reflect_generic_step(self, state: Any, observe_result: Any) -> ReflectResult:
        if observe_result.is_sufficient:
            return ReflectResult(
                should_advance=True,
                step_complete=True,
                quality_score=0.7,
                reasoning="Step appears complete",
            )

        if state.actions_in_current_step >= state.max_actions_per_step:
            return ReflectResult(
                should_advance=True,
                step_complete=True,
                quality_score=0.5,
                reasoning="Max actions reached. Advancing.",
            )

        return ReflectResult(
            should_retry=True,
            step_complete=False,
            quality_score=0.3,
            reasoning="Step not yet complete",
        )

    @staticmethod
    def _enum_value(value: Any) -> str:
        if value is None:
            return ""
        return str(getattr(value, "value", value))