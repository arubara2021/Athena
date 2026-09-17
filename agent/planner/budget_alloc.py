from __future__ import annotations

from typing import Any

from pydantic import Field

from agent.planner.decomposer import SubTask
from core.models import CoreModel
from utils.logger import get_logger


class TaskBudget(CoreModel):
    task_index: int = Field(ge=0)
    description: str = ""
    allocated_tokens: int = Field(default=0, ge=0)
    priority: int = Field(default=0, ge=0)
    is_critical: bool = False


class BudgetAllocation(CoreModel):
    total_budget: int = Field(default=0, ge=0)
    reserved_tokens: int = Field(default=0, ge=0)
    distributable_tokens: int = Field(default=0, ge=0)
    task_budgets: list[TaskBudget] = Field(default_factory=list)
    reserve_percent: float = Field(default=0.15, ge=0.0, le=0.5)
    remaining_after_allocation: int = Field(default=0, ge=0)


class BudgetAllocator:
    _DEFAULT_RESERVE_PERCENT = 0.15
    _MIN_TASK_BUDGET = 500
    _CRITICAL_PRIORITY_THRESHOLD = 2
    _PRIORITY_WEIGHT_MULTIPLIER = 1.5

    def __init__(
        self,
        reserve_percent: float | None = None,
        min_task_budget: int | None = None,
        critical_priority_threshold: int | None = None,
    ) -> None:
        self._reserve_percent = max(
            0.0,
            min(0.5, reserve_percent or self._DEFAULT_RESERVE_PERCENT),
        )
        self._min_task_budget = max(
            100,
            min_task_budget or self._MIN_TASK_BUDGET,
        )
        self._critical_priority_threshold = max(
            1,
            critical_priority_threshold or self._CRITICAL_PRIORITY_THRESHOLD,
        )
        self._logger = get_logger("agent.planner.budget_alloc")

    def allocate(
        self,
        total_budget: int,
        sub_tasks: list[SubTask],
    ) -> BudgetAllocation:
        if total_budget <= 0 or not sub_tasks:
            return BudgetAllocation(
                total_budget=max(0, total_budget),
                reserved_tokens=0,
                distributable_tokens=0,
                task_budgets=[],
                reserve_percent=self._reserve_percent,
                remaining_after_allocation=0,
            )

        reserved_tokens = int(total_budget * self._reserve_percent)
        distributable_tokens = total_budget - reserved_tokens

        if distributable_tokens <= 0:
            return BudgetAllocation(
                total_budget=total_budget,
                reserved_tokens=reserved_tokens,
                distributable_tokens=0,
                task_budgets=[],
                reserve_percent=self._reserve_percent,
                remaining_after_allocation=0,
            )

        weighted_tasks = self._compute_weights(sub_tasks)
        total_weight = sum(w for _, w in weighted_tasks)

        if total_weight <= 0:
            equal_share = distributable_tokens // len(sub_tasks)
            task_budgets = [
                TaskBudget(
                    task_index=i,
                    description=task.description,
                    allocated_tokens=max(self._min_task_budget, equal_share),
                    priority=task.priority,
                    is_critical=task.priority <= self._critical_priority_threshold,
                )
                for i, task in enumerate(sub_tasks)
            ]
        else:
            task_budgets = []
            allocated_so_far = 0

            for i, (task, weight) in enumerate(weighted_tasks):
                share = int((weight / total_weight) * distributable_tokens)
                allocated = max(self._min_task_budget, share)
                allocated_so_far += allocated

                task_budgets.append(
                    TaskBudget(
                        task_index=i,
                        description=task.description,
                        allocated_tokens=allocated,
                        priority=task.priority,
                        is_critical=task.priority <= self._critical_priority_threshold,
                    )
                )

            if allocated_so_far > distributable_tokens:
                overflow = allocated_so_far - distributable_tokens
                per_task_reduction = overflow // len(task_budgets)
                for tb in task_budgets:
                    tb.allocated_tokens = max(
                        self._min_task_budget,
                        tb.allocated_tokens - per_task_reduction,
                    )

        total_allocated = sum(tb.allocated_tokens for tb in task_budgets)
        remaining = max(0, distributable_tokens - total_allocated)

        return BudgetAllocation(
            total_budget=total_budget,
            reserved_tokens=reserved_tokens,
            distributable_tokens=distributable_tokens,
            task_budgets=task_budgets,
            reserve_percent=self._reserve_percent,
            remaining_after_allocation=remaining,
        )

    def _compute_weights(
        self,
        sub_tasks: list[SubTask],
    ) -> list[tuple[SubTask, float]]:
        weighted: list[tuple[SubTask, float]] = []

        for task in sub_tasks:
            base_weight = 1.0

            if task.priority <= self._critical_priority_threshold:
                base_weight *= self._PRIORITY_WEIGHT_MULTIPLIER

            token_factor = task.estimated_tokens / 3000.0
            token_factor = max(0.5, min(2.0, token_factor))
            base_weight *= token_factor

            weighted.append((task, base_weight))

        return weighted

    def get_task_budget(
        self,
        allocation: BudgetAllocation,
        task_index: int,
    ) -> int:
        for tb in allocation.task_budgets:
            if tb.task_index == task_index:
                return tb.allocated_tokens
        return self._min_task_budget

    def get_reserved_budget(self, allocation: BudgetAllocation) -> int:
        return allocation.reserved_tokens

    def rebalance(
        self,
        allocation: BudgetAllocation,
        completed_indices: list[int],
        unused_tokens: dict[int, int],
    ) -> BudgetAllocation:
        freed_tokens = sum(
            unused_tokens.get(idx, 0)
            for idx in completed_indices
        )

        if freed_tokens <= 0:
            return allocation

        remaining_tasks = [
            tb for tb in allocation.task_budgets
            if tb.task_index not in completed_indices
        ]

        if not remaining_tasks:
            return allocation

        per_task_bonus = freed_tokens // len(remaining_tasks)

        updated_budgets = []
        for tb in allocation.task_budgets:
            if tb.task_index in completed_indices:
                updated_budgets.append(tb)
            else:
                updated_budgets.append(
                    TaskBudget(
                        task_index=tb.task_index,
                        description=tb.description,
                        allocated_tokens=tb.allocated_tokens + per_task_bonus,
                        priority=tb.priority,
                        is_critical=tb.is_critical,
                    )
                )

        return BudgetAllocation(
            total_budget=allocation.total_budget,
            reserved_tokens=allocation.reserved_tokens,
            distributable_tokens=allocation.distributable_tokens + freed_tokens,
            task_budgets=updated_budgets,
            reserve_percent=allocation.reserve_percent,
            remaining_after_allocation=allocation.remaining_after_allocation,
        )