from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from core import constants
from core.config import get_project_root
from core.events import EventBus
from core.models import AgentActionType, AgentPlan, AgentStatus, PlanStep
from core.schemas import AgentInputSchema
from agent.loop.act import ActPhase
from agent.loop.observe import ObservePhase
from agent.loop.reflect import ReflectPhase
from agent.loop.state import (
    DEFAULT_MAX_ACTIONS_PER_STEP,
    DEFAULT_MAX_FINDINGS,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TOKEN_BUDGET,
    LoopConfig,
    LoopOutput,
    LoopState,
)
from agent.loop.think import ThinkPhase
from agent.memory.short_term import ShortTermMemory
from agent.planner.planner import AgentPlanner
from agent.tools.executor import ToolExecutor
from agent.tools.registry import ToolRegistry
from agent.token.cost_tracker import CostTracker
from utils.logger import get_logger, get_trace_logger

EVENT_AGENT_ITERATION = "agent.iteration"
EVENT_AGENT_STEP_COMPLETED = "agent.step_completed"
EVENT_AGENT_BUDGET_WARNING = "agent.budget_warning"
EVENT_AGENT_LOOP_COMPLETED = "agent.loop_completed"

_MODE_CONFIG_CACHE: dict[str, dict[str, Any]] | None = None


def _load_mode_configs() -> dict[str, dict[str, Any]]:
    global _MODE_CONFIG_CACHE

    if _MODE_CONFIG_CACHE is not None:
        return _MODE_CONFIG_CACHE

    configs: dict[str, dict[str, Any]] = {
        mode: dict(values)
        for mode, values in constants.MODE_DEFAULTS.items()
    }

    try:
        path = get_project_root() / "configs" / "settings.yaml"

        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}

            modes = raw.get("modes", {}) if isinstance(raw, dict) else {}

            if isinstance(modes, dict):
                for mode, values in modes.items():
                    if mode == "default_mode":
                        continue

                    if isinstance(values, dict):
                        merged = configs.setdefault(str(mode), {})
                        merged.update(values)
    except Exception:
        pass

    _MODE_CONFIG_CACHE = configs
    return configs


def get_mode_config(mode: str) -> dict[str, Any]:
    configs = _load_mode_configs()
    normalized = str(mode or constants.DEFAULT_MODE).strip().lower()

    if normalized not in configs:
        normalized = constants.DEFAULT_MODE

    return dict(configs.get(normalized, constants.MODE_DEFAULTS[constants.DEFAULT_MODE]))


class AgenticLoop:
    def __init__(
        self,
        registry: ToolRegistry | None = None,
        executor: ToolExecutor | None = None,
        planner: AgentPlanner | None = None,
        cost_tracker: CostTracker | None = None,
        short_term_memory: ShortTermMemory | None = None,
        event_bus: EventBus | None = None,
        config: LoopConfig | None = None,
    ) -> None:
        self._config = config or LoopConfig()
        self._registry = registry or ToolRegistry()
        self._executor = executor or ToolExecutor(
            registry=self._registry,
            cost_tracker=cost_tracker,
        )
        self._planner = planner or AgentPlanner()
        self._cost_tracker = cost_tracker
        self._memory = short_term_memory or ShortTermMemory()
        self._event_bus = event_bus or EventBus()

        self._think_phase = ThinkPhase()
        self._act_phase = ActPhase()
        self._observe_phase = ObservePhase()
        self._reflect_phase = ReflectPhase(
            quality_threshold=self._config.quality_threshold,
        )

        self._logger = get_logger("agent.loop.engine")
        self._trace = get_trace_logger()

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    async def run(
        self,
        goal: str,
        level: str = "",
        topic: str = "",
        token_budget: int | None = None,
        max_iterations: int | None = None,
        mode: str = "",
    ) -> LoopOutput:
        resolved_mode = self._resolve_mode(mode)
        mode_config = get_mode_config(resolved_mode)

        budget = self._resolve_budget(
            token_budget,
            mode_config.get("budget"),
        )
        iterations = self._resolve_iterations(
            max_iterations,
            mode_config.get("iterations"),
        )

        state = self._create_state(
            goal=goal,
            level=level,
            topic=topic,
            token_budget=budget,
            max_iterations=iterations,
        )

        self._set_status(state, AgentStatus.RUNNING)

        tools_summary = self._registry.to_summary()

        await self._ensure_plan(
            state,
            goal,
            level,
            topic,
            budget,
            iterations,
        )

        empty_search_streak = 0
        max_empty_streak = int(
            getattr(self._config, "max_empty_search_streak", 3)
        )

        early_exit_threshold = self._early_exit_threshold(
            resolved_mode,
            mode_config,
        )

        for iteration in range(1, max(1, iterations) + 1):
            state.iteration_count = iteration

            if self._is_terminal(state, budget):
                break

            previous_step_index = self._current_step_index(state)

            await self._emit(
                EVENT_AGENT_ITERATION,
                self._iteration_payload(state, budget),
            )

            try:
                think_result = await self._think_phase.think(
                    state, tools_summary
                )
            except Exception as exc:
                self._add_issue(
                    state, "error", f"think_failed: {exc}"
                )
                self._advance_state(state)
                await self._maybe_emit_step_completed(
                    state, previous_step_index
                )
                continue

            if getattr(think_result, "is_complete", False):
                self._set_status(state, AgentStatus.COMPLETED)
                break

            if getattr(think_result, "is_advance", False):
                self._advance_state(state)
                await self._maybe_emit_step_completed(
                    state, previous_step_index
                )
                continue

            try:
                act_result = await self._act_phase.act(
                    state, think_result, self._executor
                )
                observe_result = self._observe_phase.observe(
                    state, act_result
                )
                reflect_result = self._reflect_phase.reflect(
                    state,
                    observe_result,
                    think_result,
                    act_result,
                )
            except Exception as exc:
                self._add_issue(
                    state,
                    "error",
                    f"act_observe_reflect_failed: {exc}",
                )

                if (
                    self._actions_in_current_step(state)
                    >= self._config.max_actions_per_step
                ):
                    self._advance_state(state)
                    await self._maybe_emit_step_completed(
                        state, previous_step_index
                    )

                continue

            await self._emit(
                EVENT_AGENT_ITERATION,
                self._iteration_payload(state, budget),
            )

            if self._is_budget_low(state, budget):
                await self._emit(
                    EVENT_AGENT_BUDGET_WARNING,
                    self._iteration_payload(state, budget),
                )

            findings_count = len(
                getattr(state, "findings", []) or []
            )

            if findings_count == 0:
                empty_search_streak += 1
            else:
                empty_search_streak = 0

            if empty_search_streak >= max_empty_streak:
                self._add_issue(
                    state,
                    "error",
                    (
                        f"no_sources_after_"
                        f"{empty_search_streak}_iterations"
                    ),
                )
                self._set_status(state, AgentStatus.FAILED)
                break

            if (
                early_exit_threshold > 0
                and findings_count >= early_exit_threshold
            ):
                self._set_status(state, AgentStatus.COMPLETED)
                break

            if getattr(reflect_result, "should_complete", False):
                self._set_status(state, AgentStatus.COMPLETED)
                break

            if (
                getattr(reflect_result, "should_advance", False)
                or getattr(reflect_result, "should_skip", False)
            ):
                self._advance_state(state)
                await self._maybe_emit_step_completed(
                    state, previous_step_index
                )
                continue

            if getattr(reflect_result, "should_retry", False):
                if (
                    self._actions_in_current_step(state)
                    >= self._config.max_actions_per_step
                ):
                    self._advance_state(state)
                    await self._maybe_emit_step_completed(
                        state, previous_step_index
                    )
                continue

            if (
                self._actions_in_current_step(state)
                >= self._config.max_actions_per_step
            ):
                self._advance_state(state)
                await self._maybe_emit_step_completed(
                    state, previous_step_index
                )

        if (
            self._enum_value(getattr(state, "status", ""))
            == AgentStatus.RUNNING.value
        ):
            if self._is_plan_complete(state):
                self._set_status(state, AgentStatus.COMPLETED)
            elif self._tokens_remaining(state, budget) <= 0:
                self._set_status(state, AgentStatus.BUDGET_EXHAUSTED)
            elif self._iteration_count(state) >= self._max_iterations(
                state
            ):
                self._add_issue(
                    state, "warning", "Max iterations reached"
                )
                self._set_status(state, AgentStatus.COMPLETED)
            else:
                self._set_status(state, AgentStatus.COMPLETED)

        output = self._build_output(state, budget)

        await self._emit(
            EVENT_AGENT_LOOP_COMPLETED,
            {
                "loop_id": output.loop_id,
                "status": self._enum_value(output.status),
                "findings": len(output.findings),
                "ranked": len(output.ranked_sources),
                "findings_count": len(output.findings),
                "ranked_count": len(output.ranked_sources),
                "iterations": output.total_iterations,
                "iteration_count": output.total_iterations,
                "max_iterations": output.max_iterations,
                "tokens_used": output.total_tokens_used,
                "tokens_remaining": output.tokens_remaining,
                "errors_count": len(output.errors),
                "warnings_count": len(output.warnings),
                "current_step": self._current_step_index(state),
                "total_steps": self._total_steps(state),
                "mode": resolved_mode,
            },
        )

        return output

    def _resolve_mode(self, mode: str) -> str:
        text = str(mode or "").strip().lower()

        if text in constants.VALID_MODES:
            return text

        try:
            path = get_project_root() / "configs" / "settings.yaml"

            if path.exists():
                with open(path, "r", encoding="utf-8") as handle:
                    raw = yaml.safe_load(handle) or {}

                modes = (
                    raw.get("modes", {})
                    if isinstance(raw, dict)
                    else {}
                )

                default = str(
                    modes.get("default_mode") or ""
                ).strip().lower()

                if default in constants.VALID_MODES:
                    return default
        except Exception:
            pass

        return constants.DEFAULT_MODE

    def _early_exit_threshold(
        self,
        mode: str,
        mode_config: dict[str, Any],
    ) -> int:
        if mode == constants.MODE_FAST:
            try:
                return max(1, int(mode_config.get("top_n", 10)))
            except Exception:
                return 10

        return 0

    def _resolve_budget(
        self,
        token_budget: int | None,
        mode_budget: Any,
    ) -> int:
        value = token_budget

        if value is None and mode_budget is not None:
            value = mode_budget

        if value is None:
            value = getattr(self._config, "token_budget", None)

        try:
            parsed = int(value)
        except Exception:
            parsed = 0

        if parsed <= 0:
            parsed = int(DEFAULT_TOKEN_BUDGET)

        return max(0, parsed)

    def _resolve_iterations(
        self,
        max_iterations: int | None,
        mode_iterations: Any,
    ) -> int:
        value = max_iterations

        if value is None and mode_iterations is not None:
            value = mode_iterations

        if value is None:
            value = getattr(self._config, "max_iterations", None)

        try:
            parsed = int(value)
        except Exception:
            parsed = 0

        if parsed <= 0:
            parsed = int(DEFAULT_MAX_ITERATIONS)

        return max(1, parsed)

    def _resolve_max_findings(self) -> int:
        value = getattr(self._config, "max_findings", None)

        try:
            parsed = int(value)
        except Exception:
            parsed = int(DEFAULT_MAX_FINDINGS)

        return max(1, parsed)

    def _resolve_max_actions_per_step(self) -> int:
        value = getattr(self._config, "max_actions_per_step", None)

        try:
            parsed = int(value)
        except Exception:
            parsed = int(DEFAULT_MAX_ACTIONS_PER_STEP)

        return max(1, parsed)

    def _safe_positive_int(self, value: Any, fallback: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = fallback

        if parsed <= 0:
            parsed = fallback

        return parsed

    def _create_state(
        self,
        goal: str,
        level: str,
        topic: str,
        token_budget: int,
        max_iterations: int,
    ) -> LoopState:
        budget = max(0, int(token_budget or 0))
        iterations = max(
            1, int(max_iterations or DEFAULT_MAX_ITERATIONS)
        )
        findings_limit = self._resolve_max_findings()
        actions_limit = self._resolve_max_actions_per_step()

        try:
            state = LoopState(
                goal=goal,
                level=level,
                topic=topic,
                total_budget=budget,
                tokens_remaining=budget,
                max_iterations=iterations,
                max_findings=findings_limit,
                max_actions_per_step=actions_limit,
            )
        except Exception:
            state = LoopState(
                goal=goal,
                level=level,
                topic=topic,
            )

            for attribute_name, attribute_value in (
                ("total_budget", budget),
                ("tokens_remaining", budget),
                ("max_iterations", iterations),
                ("max_findings", findings_limit),
                ("max_actions_per_step", actions_limit),
            ):
                try:
                    setattr(state, attribute_name, attribute_value)
                except Exception:
                    pass

        return state

    async def _ensure_plan(
        self,
        state: LoopState,
        goal: str,
        level: str,
        topic: str,
        token_budget: int,
        max_iterations: int,
    ) -> None:
        if getattr(state, "plan", None) is not None:
            return

        try:
            agent_input = AgentInputSchema(
                goal=goal,
                level=level,
                topic=topic,
                max_iterations=max_iterations,
                token_budget=token_budget,
            )

            plan_result = self._planner.plan(agent_input)

            if inspect.isawaitable(plan_result):
                plan_result = await plan_result

            plan = getattr(plan_result, "plan", None)

            if plan is not None:
                state.plan = plan
                return
        except Exception as exc:
            self._add_issue(
                state, "warning", f"planner_failed: {exc}"
            )

        state.plan = self._fallback_plan(goal, topic)

    def _fallback_plan(self, goal: str, topic: str) -> AgentPlan:
        subject = topic or goal

        search_tokens = self._safe_positive_int(
            getattr(
                constants,
                "AGENT_SEARCH_BUDGET",
                DEFAULT_TOKEN_BUDGET // 10,
            ),
            DEFAULT_TOKEN_BUDGET // 10,
        )

        analyze_tokens = self._safe_positive_int(
            getattr(
                constants,
                "AGENT_RANKING_BUDGET",
                DEFAULT_TOKEN_BUDGET // 8,
            ),
            DEFAULT_TOKEN_BUDGET // 8,
        )

        read_tokens = self._safe_positive_int(
            getattr(
                constants,
                "AGENT_READING_BUDGET",
                DEFAULT_TOKEN_BUDGET // 5,
            ),
            DEFAULT_TOKEN_BUDGET // 5,
        )

        generate_tokens = self._safe_positive_int(
            getattr(
                constants,
                "AGENT_SYNTHESIS_BUDGET",
                DEFAULT_TOKEN_BUDGET // 10,
            ),
            DEFAULT_TOKEN_BUDGET // 10,
        )

        steps = [
            PlanStep(
                step_index=0,
                description=f"Search sources for {subject}",
                action_type=AgentActionType.SEARCH,
                estimated_tokens=search_tokens,
                priority=1,
                depends_on=[],
            ),
            PlanStep(
                step_index=1,
                description=f"Rank and analyze sources for {subject}",
                action_type=AgentActionType.ANALYZE,
                estimated_tokens=analyze_tokens,
                priority=2,
                depends_on=[0],
            ),
            PlanStep(
                step_index=2,
                description=f"Read and summarize best sources for {subject}",
                action_type=AgentActionType.READ,
                estimated_tokens=read_tokens,
                priority=3,
                depends_on=[1],
            ),
            PlanStep(
                step_index=3,
                description=f"Generate learning path for {subject}",
                action_type=AgentActionType.GENERATE,
                estimated_tokens=generate_tokens,
                priority=4,
                depends_on=[2],
            ),
        ]

        return AgentPlan(
            plan_id=str(uuid4()),
            goal=goal,
            steps=steps,
            total_estimated_tokens=sum(
                step.estimated_tokens for step in steps
            ),
            budget_allocated={},
        )

    def _is_terminal(self, state: LoopState, budget: int) -> bool:
        status = self._enum_value(getattr(state, "status", ""))

        if status in {
            AgentStatus.COMPLETED.value,
            AgentStatus.FAILED.value,
            AgentStatus.BUDGET_EXHAUSTED.value,
        }:
            return True

        if self._iteration_count(state) >= self._max_iterations(state):
            return True

        if self._tokens_remaining(state, budget) <= 0:
            return True

        return False

    def _is_plan_complete(self, state: LoopState) -> bool:
        value = getattr(state, "is_plan_complete", None)

        if isinstance(value, bool):
            return value

        return self._current_step_index(state) >= self._total_steps(
            state
        )

    def _advance_state(self, state: LoopState) -> None:
        current = self._current_step_index(state)
        total = self._total_steps(state)

        if total <= 0:
            self._set_status(state, AgentStatus.COMPLETED)
            return

        if current + 1 < total:
            try:
                state.current_step_index = current + 1
            except Exception:
                pass
        else:
            self._set_status(state, AgentStatus.COMPLETED)

    async def _maybe_emit_step_completed(
        self,
        state: LoopState,
        previous_step_index: int,
    ) -> None:
        current = self._current_step_index(state)

        if current != previous_step_index:
            await self._emit(
                EVENT_AGENT_STEP_COMPLETED,
                {
                    "loop_id": getattr(state, "loop_id", ""),
                    "step_index": previous_step_index,
                    "current_step": current,
                    "total_steps": self._total_steps(state),
                },
            )

    def _actions_in_current_step(self, state: LoopState) -> int:
        value = getattr(state, "actions_in_current_step", None)

        if isinstance(value, int):
            return value

        current = self._current_step_index(state)
        actions = getattr(state, "actions_taken", []) or []

        return len(
            [
                action
                for action in actions
                if int(getattr(action, "step_index", -1)) == current
            ]
        )

    def _iteration_payload(
        self,
        state: LoopState,
        budget: int,
    ) -> dict[str, Any]:
        return {
            "loop_id": getattr(state, "loop_id", ""),
            "iteration_count": self._iteration_count(state),
            "max_iterations": self._max_iterations(state),
            "findings_count": len(
                getattr(state, "findings", []) or []
            ),
            "ranked_count": len(
                getattr(state, "ranked_sources", []) or []
            ),
            "tokens_used": self._tokens_used(state),
            "tokens_remaining": self._tokens_remaining(state, budget),
            "errors_count": len(
                getattr(state, "errors", []) or []
            ),
            "warnings_count": len(
                getattr(state, "warnings", []) or []
            ),
            "current_step": self._current_step_index(state),
            "total_steps": self._total_steps(state),
            "status": self._enum_value(getattr(state, "status", "")),
        }

    def _is_budget_low(self, state: LoopState, budget: int) -> bool:
        remaining = self._tokens_remaining(state, budget)
        effective_budget = max(
            int(budget or 0), self._token_budget(state), 1
        )

        threshold_percent = float(
            getattr(
                constants,
                "AGENT_WRAP_UP_THRESHOLD_PERCENT",
                85.0,
            )
        )

        threshold_percent = max(0.0, min(100.0, threshold_percent))
        low_fraction = max(
            0.0, (100.0 - threshold_percent) / 100.0
        )

        return remaining > 0 and remaining <= (
            effective_budget * low_fraction
        )

    def _build_output(
        self,
        state: LoopState,
        budget: int,
    ) -> LoopOutput:
        return LoopOutput(
            loop_id=str(getattr(state, "loop_id", "")),
            status=getattr(state, "status", AgentStatus.FAILED),
            goal=str(getattr(state, "goal", "")),
            level=str(getattr(state, "level", "")),
            findings=list(getattr(state, "findings", []) or []),
            ranked_sources=list(
                getattr(state, "ranked_sources", []) or []
            ),
            learning_path=getattr(state, "learning_path", None),
            enhanced_learning_path=getattr(
                state, "enhanced_learning_path", None
            ),
            actions_taken=list(
                getattr(state, "actions_taken", []) or []
            ),
            total_iterations=self._iteration_count(state),
            total_tokens_used=self._tokens_used(state),
            total_budget=max(
                int(budget or 0), self._token_budget(state), 0
            ),
            tokens_remaining=self._tokens_remaining(state, budget),
            errors=list(getattr(state, "errors", []) or []),
            warnings=list(getattr(state, "warnings", []) or []),
            max_iterations=self._max_iterations(state),
        )

    def _current_step_index(self, state: LoopState) -> int:
        try:
            return int(getattr(state, "current_step_index", 0) or 0)
        except Exception:
            return 0

    def _total_steps(self, state: LoopState) -> int:
        plan = getattr(state, "plan", None)

        if plan is None:
            return 0

        steps = getattr(plan, "steps", None) or []
        return len(list(steps))

    def _iteration_count(self, state: LoopState) -> int:
        try:
            return int(getattr(state, "iteration_count", 0) or 0)
        except Exception:
            return 0

    def _max_iterations(self, state: LoopState) -> int:
        try:
            state_value = int(
                getattr(state, "max_iterations", 0) or 0
            )
        except Exception:
            state_value = 0

        if state_value > 0:
            return state_value

        try:
            config_value = int(
                getattr(self._config, "max_iterations", 0) or 0
            )
        except Exception:
            config_value = 0

        if config_value > 0:
            return config_value

        return int(DEFAULT_MAX_ITERATIONS)

    def _token_budget(self, state: LoopState) -> int:
        for attribute_name in ("total_budget", "token_budget"):
            value = getattr(state, attribute_name, None)

            if value is None:
                continue

            try:
                parsed = int(value)
            except Exception:
                continue

            if parsed >= 0:
                return parsed

        try:
            config_value = int(
                getattr(self._config, "token_budget", 0) or 0
            )
        except Exception:
            config_value = 0

        if config_value >= 0:
            return config_value

        return int(DEFAULT_TOKEN_BUDGET)

    def _tokens_used(self, state: LoopState) -> int:
        for attr in ("total_tokens_used", "tokens_used"):
            value = getattr(state, attr, None)

            if value is not None:
                try:
                    return max(0, int(value))
                except Exception:
                    continue

        if self._cost_tracker is not None:
            for attr in (
                "total_tokens_used",
                "tokens_used",
                "total_spent",
                "spent",
            ):
                value = getattr(self._cost_tracker, attr, None)

                if value is not None:
                    try:
                        return max(0, int(value))
                    except Exception:
                        continue

        return 0

    def _tokens_remaining(
        self,
        state: LoopState,
        budget: int,
    ) -> int:
        value = getattr(state, "tokens_remaining", None)

        if value is not None:
            try:
                remaining = int(value)

                if remaining >= 0:
                    return remaining
            except Exception:
                pass

        effective_budget = max(
            int(budget or 0), self._token_budget(state), 0
        )
        return max(0, effective_budget - self._tokens_used(state))

    def _set_status(
        self,
        state: LoopState,
        status: AgentStatus,
    ) -> None:
        if status == AgentStatus.COMPLETED:
            mark_completed = getattr(state, "mark_completed", None)

            if callable(mark_completed):
                mark_completed()
                return

        if status == AgentStatus.FAILED:
            mark_failed = getattr(state, "mark_failed", None)

            if callable(mark_failed):
                mark_failed("Agent loop failed")
                return

        if status == AgentStatus.BUDGET_EXHAUSTED:
            mark_budget_exhausted = getattr(
                state, "mark_budget_exhausted", None
            )

            if callable(mark_budget_exhausted):
                mark_budget_exhausted()
                return

        try:
            state.status = status
        except Exception:
            pass

    def _add_issue(
        self,
        state: LoopState,
        kind: str,
        message: str,
    ) -> None:
        method = getattr(state, f"add_{kind}", None)

        if callable(method):
            try:
                method(message)
                return
            except Exception:
                pass

        items = getattr(state, f"{kind}s", None)

        if isinstance(items, list):
            items.append(message)

    async def _emit(
        self,
        name: str,
        payload: dict[str, Any],
    ) -> None:
        try:
            await self._event_bus.emit(name, payload)
        except Exception as exc:
            self._logger.warning(
                f"Failed to emit loop event {name}: {exc}"
            )

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value) or "")