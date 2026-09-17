from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import Field

from agent.planner.budget_alloc import BudgetAllocator, BudgetAllocation
from agent.planner.decomposer import DecompositionResult, GoalDecomposer, SubTask
from agent.planner.templates import GoalType, PlanTemplates
from core.models import AgentPlan, CoreModel, PlanStep, AgentActionType, AgentStatus
from core.schemas import AgentInputSchema
from utils.logger import get_logger, get_trace_logger
from utils.text import clean_text


class PlannerConfig(CoreModel):
    use_llm_decomposition: bool = True
    total_token_budget: int = Field(default=40000, ge=1000, le=500000)
    reserve_percent: float = Field(default=0.15, ge=0.0, le=0.5)
    max_sub_tasks: int = Field(default=8, ge=2, le=15)
    min_sub_tasks: int = Field(default=2, ge=1, le=10)
    max_iterations: int = Field(default=15, ge=1, le=50)


class PlannerResult(CoreModel):
    plan: AgentPlan | None = None
    decomposition: DecompositionResult | None = None
    budget_allocation: BudgetAllocation | None = None
    goal_type: GoalType = GoalType.EXPLORE
    success: bool = False
    error: str | None = None


class AgentPlanner:
    def __init__(
        self,
        config: PlannerConfig | None = None,
        decomposer: GoalDecomposer | None = None,
        budget_allocator: BudgetAllocator | None = None,
        templates: PlanTemplates | None = None,
    ) -> None:
        self._config = config or PlannerConfig()
        self._templates = templates or PlanTemplates()
        self._decomposer = decomposer or GoalDecomposer(
            use_llm=self._config.use_llm_decomposition,
            max_sub_tasks=self._config.max_sub_tasks,
            min_sub_tasks=self._config.min_sub_tasks,
            templates=self._templates,
        )
        self._budget_allocator = budget_allocator or BudgetAllocator(
            reserve_percent=self._config.reserve_percent,
        )
        self._logger = get_logger("agent.planner.planner")
        self._trace = get_trace_logger()

    @property
    def config(self) -> PlannerConfig:
        return self._config

    async def plan(
        self,
        goal: str,
        level: str = "",
        intent: str = "",
        token_budget: int | None = None,
        max_iterations: int | None = None,
    ) -> PlannerResult:
        cleaned_goal = clean_text(goal)
        if not cleaned_goal:
            return PlannerResult(
                plan=None,
                decomposition=None,
                budget_allocation=None,
                goal_type=GoalType.EXPLORE,
                success=False,
                error="Empty goal provided",
            )

        resolved_budget = token_budget or self._config.total_token_budget
        resolved_iterations = max_iterations or self._config.max_iterations

        self._trace.emit(
            "planner_started",
            goal=cleaned_goal,
            level=level,
            intent=intent,
            token_budget=resolved_budget,
            max_iterations=resolved_iterations,
        )

        try:
            decomposition = await self._decomposer.decompose(
                goal=cleaned_goal,
                level=level,
                intent=intent,
            )

            if not decomposition.sub_tasks:
                return PlannerResult(
                    plan=None,
                    decomposition=decomposition,
                    budget_allocation=None,
                    goal_type=decomposition.goal_type,
                    success=False,
                    error="Decomposition produced no sub-tasks",
                )

            budget_allocation = self._budget_allocator.allocate(
                total_budget=resolved_budget,
                sub_tasks=decomposition.sub_tasks,
            )

            plan = self._build_plan(
                goal=cleaned_goal,
                level=level,
                decomposition=decomposition,
                budget_allocation=budget_allocation,
                max_iterations=resolved_iterations,
            )

            self._trace.emit(
                "planner_completed",
                goal=cleaned_goal,
                goal_type=decomposition.goal_type.value,
                sub_task_count=len(decomposition.sub_tasks),
                total_budget=resolved_budget,
                reserved_tokens=budget_allocation.reserved_tokens,
                llm_used=decomposition.llm_used,
                confidence=decomposition.confidence,
            )

            return PlannerResult(
                plan=plan,
                decomposition=decomposition,
                budget_allocation=budget_allocation,
                goal_type=decomposition.goal_type,
                success=True,
                error=None,
            )

        except Exception as exc:
            self._logger.warning(f"Planning failed: {exc}")
            self._trace.emit(
                "planner_failed",
                goal=cleaned_goal,
                error=str(exc),
            )
            return PlannerResult(
                plan=None,
                decomposition=None,
                budget_allocation=None,
                goal_type=GoalType.EXPLORE,
                success=False,
                error=str(exc),
            )

    async def plan_from_input(
        self,
        agent_input: AgentInputSchema,
    ) -> PlannerResult:
        return await self.plan(
            goal=agent_input.goal,
            level=agent_input.level,
            intent="",
            token_budget=agent_input.token_budget,
            max_iterations=agent_input.max_iterations,
        )

    def _build_plan(
        self,
        goal: str,
        level: str,
        decomposition: DecompositionResult,
        budget_allocation: BudgetAllocation,
        max_iterations: int,
    ) -> AgentPlan:
        plan_id = str(uuid4())
        steps: list[PlanStep] = []

        for index, sub_task in enumerate(decomposition.sub_tasks):
            task_budget = self._budget_allocator.get_task_budget(
                budget_allocation, index
            )

            steps.append(
                PlanStep(
                    step_index=index,
                    description=sub_task.description,
                    action_type=sub_task.action_type,
                    estimated_tokens=task_budget,
                    priority=sub_task.priority,
                    depends_on=sub_task.depends_on,
                )
            )

        total_estimated = sum(step.estimated_tokens for step in steps)

        return AgentPlan(
            plan_id=plan_id,
            goal=goal,
            steps=steps,
            total_estimated_tokens=total_estimated,
            budget_allocated=budget_allocation.total_budget,
        )

    def replan(
        self,
        original_plan: AgentPlan,
        completed_steps: list[int],
        failed_steps: list[int],
        remaining_budget: int,
    ) -> AgentPlan | None:
        if not original_plan.steps:
            return None

        remaining_steps = [
            step for step in original_plan.steps
            if step.step_index not in completed_steps
            and step.step_index not in failed_steps
        ]

        if not remaining_steps:
            return None

        failed_descriptions = [
            step.description for step in original_plan.steps
            if step.step_index in failed_steps
        ]

        adjusted_steps: list[PlanStep] = []
        per_step_budget = max(
            500,
            remaining_budget // max(1, len(remaining_steps)),
        )

        for new_index, step in enumerate(remaining_steps):
            adjusted_steps.append(
                PlanStep(
                    step_index=new_index,
                    description=step.description,
                    action_type=step.action_type,
                    estimated_tokens=per_step_budget,
                    priority=step.priority,
                    depends_on=list(range(new_index)) if new_index > 0 else [],
                )
            )

        return AgentPlan(
            plan_id=str(uuid4()),
            goal=original_plan.goal,
            steps=adjusted_steps,
            total_estimated_tokens=sum(s.estimated_tokens for s in adjusted_steps),
            budget_allocated=remaining_budget,
        )

    def get_plan_summary(self, plan: AgentPlan) -> dict[str, Any]:
        return {
            "plan_id": plan.plan_id,
            "goal": plan.goal,
            "total_steps": len(plan.steps),
            "total_estimated_tokens": plan.total_estimated_tokens,
            "budget_allocated": plan.budget_allocated,
            "steps": [
                {
                    "index": step.step_index,
                    "description": step.description,
                    "action_type": step.action_type.value,
                    "estimated_tokens": step.estimated_tokens,
                    "priority": step.priority,
                }
                for step in plan.steps
            ],
        }