from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import Field

from core.models import CoreModel, LearningPath, RankedSource, Source, TaskStatus
from core.schemas import SearchQuerySchema


class PipelineState(CoreModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    status: TaskStatus = TaskStatus.PENDING

    query: SearchQuerySchema | None = None

    corrected_topic: str = ""
    primary_concept: str = ""
    keywords: list[str] = Field(default_factory=list)
    query_warnings: list[str] = Field(default_factory=list)

    expanded_queries: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    ranked_sources: list[RankedSource] = Field(default_factory=list)
    source_summaries: list[Any] = Field(default_factory=list)
    learning_path: LearningPath | None = None
    enhanced_learning_path: Any = None
    rag_documents_indexed: int = 0

    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    latency_ms: float | None = None

    def mark_running(self) -> None:
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)
        self.touch()

    def mark_success(self) -> None:
        if self.errors:
            self.status = TaskStatus.PARTIAL
        elif self.warnings:
            self.status = TaskStatus.PARTIAL
        else:
            self.status = TaskStatus.SUCCESS

        self.finished_at = datetime.now(timezone.utc)
        self._finalize_latency()
        self.touch()

    def mark_failed(self, error: str | None = None) -> None:
        if error:
            self.add_error(error)

        self.status = TaskStatus.FAILED
        self.finished_at = datetime.now(timezone.utc)
        self._finalize_latency()
        self.touch()

    def add_error(self, error: Any) -> None:
        text = str(error or "").strip()

        if text:
            self.errors.append(text)
            self.touch()

    def add_warning(self, warning: Any) -> None:
        text = str(warning or "").strip()

        if text:
            self.warnings.append(text)
            self.touch()

    def add_query_warning(self, warning: Any) -> None:
        text = str(warning or "").strip()

        if text:
            self.query_warnings.append(text)
            self.touch()

    def set_query_understanding(
        self,
        corrected_topic: str = "",
        primary_concept: str = "",
        keywords: list[Any] | None = None,
        query_warnings: list[Any] | None = None,
    ) -> None:
        self.corrected_topic = str(corrected_topic or "").strip()
        self.primary_concept = str(primary_concept or "").strip()

        self.keywords = [
            str(item).strip()
            for item in (keywords or [])
            if str(item or "").strip()
        ]

        if query_warnings is not None:
            self.query_warnings = [
                str(item).strip()
                for item in query_warnings
                if str(item or "").strip()
            ]

        self.touch()

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def _finalize_latency(self) -> None:
        if self.started_at is None or self.finished_at is None:
            return

        delta = (self.finished_at - self.started_at).total_seconds() * 1000
        self.latency_ms = max(0.0, delta)

    def to_summary(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self._enum_value(self.status),
            "topic": self.query.topic if self.query else None,
            "corrected_topic": self.corrected_topic or None,
            "primary_concept": self.primary_concept or None,
            "keywords": self.keywords,
            "query_warnings": self.query_warnings,
            "total_sources": len(self.sources),
            "total_ranked": len(self.ranked_sources),
            "total_summaries": len(self.source_summaries),
            "has_learning_path": self.learning_path is not None,
            "has_enhanced_learning_path": self.enhanced_learning_path is not None,
            "rag_documents_indexed": self.rag_documents_indexed,
            "total_errors": len(self.errors),
            "total_warnings": len(self.warnings),
            "total_query_warnings": len(self.query_warnings),
            "latency_ms": self.latency_ms,
        }

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value))