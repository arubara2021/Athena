from __future__ import annotations

from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import Field

from core.models import CoreModel
from utils.logger import get_logger


PIPELINE_STARTED = "pipeline.started"
SEARCH_COMPLETED = "pipeline.search_completed"
RANKING_COMPLETED = "pipeline.ranking_completed"
SOURCE_SUMMARIES_COMPLETED = "pipeline.source_summaries_completed"
LEARNING_PATH_COMPLETED = "pipeline.learning_path_completed"
PATH_ENHANCED = "pipeline.path_enhanced"
RAG_INDEXING_STARTED = "pipeline.rag_indexing_started"
RAG_INDEXED = "pipeline.rag_indexed"
PIPELINE_COMPLETED = "pipeline.completed"
PIPELINE_FAILED = "pipeline.failed"
STAGE_FAILED = "pipeline.stage_failed"

AGENT_STARTED = "agent.started"
AGENT_THINKING = "agent.thinking"
AGENT_ACTING = "agent.acting"
AGENT_OBSERVING = "agent.observing"
AGENT_REFLECTING = "agent.reflecting"
AGENT_COMPLETED = "agent.completed"
AGENT_FAILED = "agent.failed"
AGENT_PLAN_CREATED = "agent.plan_created"
AGENT_STEP_COMPLETED = "agent.step_completed"
AGENT_BUDGET_WARNING = "agent.budget_warning"
AGENT_BUDGET_EXHAUSTED = "agent.budget_exhausted"
AGENT_MEMORY_RECALLED = "agent.memory_recalled"
AGENT_SYNTHESIZING = "agent.synthesizing"
TOOL_CALLED = "agent.tool_called"
TOOL_COMPLETED = "agent.tool_completed"
TOOL_FAILED = "agent.tool_failed"
BUDGET_WARNING = "agent.budget_warning_event"
BUDGET_CRITICAL = "agent.budget_critical"

WILDCARD = "*"

EventHandler = Callable[["Event"], Awaitable[None]]


class Event(CoreModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._logger = get_logger("core.events")

    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        self._handlers[event_name].append(handler)

    def unsubscribe(self, event_name: str, handler: EventHandler) -> None:
        if event_name in self._handlers:
            try:
                self._handlers[event_name].remove(handler)
            except ValueError:
                pass

    async def publish(self, event: Event) -> None:
        handlers = list(self._handlers.get(event.name, []))
        wildcard_handlers = list(self._handlers.get(WILDCARD, []))

        for handler in handlers + wildcard_handlers:
            try:
                await handler(event)
            except Exception as exc:
                self._logger.warning(
                    f"Event handler failed for {event.name}: {exc}"
                )

    async def emit(self, name: str, payload: dict[str, Any] | None = None) -> Event:
        event = Event(name=name, payload=payload or {})
        await self.publish(event)
        return event