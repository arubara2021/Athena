from __future__ import annotations

import time
from typing import Any

from core.events import EventBus
from core.models import AgentStatus
from agent.loop.state import LoopConfig, LoopOutput, LoopState
from agent.loop.think import ThinkPhase
from agent.loop.act import ActPhase
from agent.loop.observe import ObservePhase
from agent.loop.reflect import ReflectPhase
from agent.planner.planner import AgentPlanner
from agent.tools.executor import ToolExecutor
from agent.tools.registry import ToolRegistry
from agent.token.cost_tracker import CostTracker
from agent.memory.short_term import ShortTermMemory
from utils.logger import get_logger, get_trace_logger


EVENT_AGENT_STARTED = "agent.started"
EVENT_AGENT_ITERATION = "agent.iteration"
EVENT_AGENT_STEP_COMPLETED = "agent.step_completed"
EVENT_AGENT_COMPLETED = "agent.completed"
EVENT_AGENT_FAILED = "agent.failed"
EVENT_AGENT_BUDGET_WARNING = "agent.budget_warning"


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

    def subscribe(self, event_name: str, handler: Any) -> None:
        self._event_bus.subscribe(event_name, handler)

    async def run(
        self,
        goal: str,
        level: str = "",
        topic: str = "",
        token_budget: int | None = None,
        max_iterations: int | None = None,
    ) -> LoopOutput:
        resolved_budget = token_budget or self._config.token_budget
        resolved_max_iter = max_iterations or self._config.max_iterations

        state = LoopState(
            goal=goal,
            level=level,
            topic=topic or goal,
            max_iterations=resolved_max_iter,
            max_actions_per_step=self._config.max_actions_per_step,
            max_findings=self._config.max_findings,
            total_budget=resolved_budget,
            tokens_remaining=resolved_budget,
        )
        state.mark_running()

        self._trace.emit(
            "agent_loop_started",
            loop_id=state.loop_id,
            goal=goal,
            level=level,
            token_budget=resolved_budget,
            max_iterations=resolved_max_iter,
        )
        await self._event_bus.emit(EVENT_AGENT_STARTED, state.to_summary())

        try:
            plan_result = await self._planner.plan(
                goal=goal,
                level=level,
                token_budget=resolved_budget,
                max_iterations=resolved_max_iter,
            )

            if plan_result.plan is None or not plan_result.plan.steps:
                state.add_error("Planning produced no steps")
                state.mark_failed("Planning failed")
                return self._build_output(state)

            state.plan = plan_result.plan

            self._memory.initialize(goal=goal, level=level, topic=state.topic)

            self._trace.emit(
                "agent_plan_created",
                loop_id=state.loop_id,
                total_steps=len(plan_result.plan.steps),
                goal_type=self._enum_value(plan_result.goal_type),
            )

            tools_summary = self._registry.to_summary()

            while not self._should_exit(state):
                state.increment_iteration()

                self._trace.emit(
                    "agent_iteration_start",
                    loop_id=state.loop_id,
                    iteration=state.iteration_count,
                    step_index=state.current_step_index,
                    tokens_remaining=state.tokens_remaining,
                )
                await self._event_bus.emit(EVENT_AGENT_ITERATION, state.to_summary())

                think_result = await self._think_phase.think(state, tools_summary)

                self._trace.emit(
                    "agent_think_result",
                    loop_id=state.loop_id,
                    iteration=state.iteration_count,
                    action=think_result.action,
                    tool_name=think_result.tool_name,
                    is_advance=think_result.is_advance,
                    is_complete=think_result.is_complete,
                    reasoning=think_result.reasoning[:200],
                )

                if think_result.is_complete:
                    state.mark_completed()
                    break

                if think_result.is_advance:
                    advanced = state.advance_step()
                    if not advanced:
                        state.mark_completed()
                        break
                    self._trace.emit(
                        "agent_step_advanced",
                        loop_id=state.loop_id,
                        new_step_index=state.current_step_index,
                    )
                    await self._event_bus.emit(
                        EVENT_AGENT_STEP_COMPLETED,
                        {"step_index": state.current_step_index - 1},
                    )
                    continue

                act_result = await self._act_phase.act(
                    state, think_result, self._executor
                )

                observe_result = self._observe_phase.observe(state, act_result)

                reflect_result = self._reflect_phase.reflect(
                    state, observe_result, think_result, act_result
                )

                self._trace.emit(
                    "agent_reflect_result",
                    loop_id=state.loop_id,
                    iteration=state.iteration_count,
                    should_advance=reflect_result.should_advance,
                    should_complete=reflect_result.should_complete,
                    should_retry=reflect_result.should_retry,
                    quality_score=reflect_result.quality_score,
                    reasoning=reflect_result.reasoning[:200],
                )

                if reflect_result.should_complete:
                    state.mark_completed()
                    break

                if reflect_result.should_advance:
                    advanced = state.advance_step()
                    if not advanced:
                        state.mark_completed()
                        break
                    await self._event_bus.emit(
                        EVENT_AGENT_STEP_COMPLETED,
                        {"step_index": state.current_step_index - 1},
                    )

                if state.is_budget_warning:
                    self._trace.emit(
                        "agent_budget_warning",
                        loop_id=state.loop_id,
                        percent_used=round(state.budget_percent_used, 1),
                        tokens_remaining=state.tokens_remaining,
                    )
                    await self._event_bus.emit(
                        EVENT_AGENT_BUDGET_WARNING,
                        state.to_summary(),
                    )

            if state.status == AgentStatus.RUNNING:
                if state.is_plan_complete:
                    state.mark_completed()
                elif state.tokens_remaining <= 0:
                    state.mark_budget_exhausted()
                elif state.iteration_count >= state.max_iterations:
                    state.add_warning("Max iterations reached")
                    state.mark_completed()
                else:
                    state.mark_completed()

        except Exception as exc:
            self._logger.error(f"Agent loop failed: {exc}")
            state.mark_failed(str(exc))
            self._trace.emit(
                "agent_loop_exception",
                loop_id=state.loop_id,
                error=str(exc),
            )

        output = self._build_output(state)

        self._trace.emit(
            "agent_loop_completed",
            loop_id=state.loop_id,
            status=self._enum_value(state.status),
            iterations=state.iteration_count,
            findings=len(state.findings),
            ranked=len(state.ranked_sources),
            tokens_used=state.total_tokens_used,
            latency_ms=round(state.latency_ms or 0, 1),
        )
        await self._event_bus.emit(
            EVENT_AGENT_COMPLETED if state.status == AgentStatus.COMPLETED else EVENT_AGENT_FAILED,
            output.to_summary(),
        )

        return output

    def _should_exit(self, state: LoopState) -> bool:
        if state.status in (
            AgentStatus.COMPLETED,
            AgentStatus.FAILED,
            AgentStatus.BUDGET_EXHAUSTED,
        ):
            return True

        if state.iteration_count >= state.max_iterations:
            return True

        if state.tokens_remaining <= 0:
            return True

        if state.is_plan_complete:
            return True

        return False

    def _build_output(self, state: LoopState) -> LoopOutput:
        return LoopOutput(
            loop_id=state.loop_id,
            status=state.status,
            goal=state.goal,
            level=state.level,
            findings=list(state.findings),
            ranked_sources=list(state.ranked_sources),
            actions_taken=list(state.actions_taken),
            total_iterations=state.iteration_count,
            total_tokens_used=state.total_tokens_used,
            total_budget=state.total_budget,
            tokens_remaining=state.tokens_remaining,
            errors=list(state.errors),
            warnings=list(state.warnings),
            started_at=state.started_at,
            finished_at=state.finished_at,
            latency_ms=state.latency_ms,
        )

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value))