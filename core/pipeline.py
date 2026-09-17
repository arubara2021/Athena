from __future__ import annotations

import time
from typing import Any

from pydantic import Field

from core.config import get_data_directory, get_output_directory
from core.events import (
    EventBus,
    LEARNING_PATH_COMPLETED,
    PATH_ENHANCED,
    PIPELINE_COMPLETED,
    PIPELINE_FAILED,
    PIPELINE_STARTED,
    RAG_INDEXED,
    RAG_INDEXING_STARTED,
    RANKING_COMPLETED,
    SEARCH_COMPLETED,
    SOURCE_SUMMARIES_COMPLETED,
    STAGE_FAILED,
)
from core.state import PipelineState
from core.exceptions import ResearchAgentError
from core.models import CoreModel, LearningPath, RankedSource, Source, TaskStatus
from ranking.consensus_ranker import ConsensusRanker
from ranking.learning_path_builder import LearningPathBuilder
from ranking.path_enhancer import PathEnhancer
from ranking.source_summarizer import SourceSummarizer
from core.schemas import SearchQuerySchema
from search.orchestrator import SearchOrchestrator
from storage.file_manager import FileManager
from storage.html_writer import HTMLWriter
from storage.json_writer import JSONWriter
from storage.markdown_writer import MarkdownWriter
from storage.sqlite_store import SQLiteStore
from utils.logger import get_logger, get_trace_logger


class PipelineResult(CoreModel):
    request_id: str
    status: TaskStatus
    query: SearchQuerySchema | None = None
    expanded_queries: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    ranked_sources: list[RankedSource] = Field(default_factory=list)
    source_summaries: list[Any] = Field(default_factory=list)
    learning_path: LearningPath | None = None
    enhanced_learning_path: Any = None
    rag_documents_indexed: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    saved_files: dict[str, str] = Field(default_factory=dict)
    started_at: Any = None
    finished_at: Any = None
    latency_ms: float | None = None


class ResearchPipeline:
    def __init__(
        self,
        search_orchestrator: SearchOrchestrator | None = None,
        consensus_ranker: ConsensusRanker | None = None,
        learning_path_builder: LearningPathBuilder | None = None,
        source_summarizer: SourceSummarizer | None = None,
        path_enhancer: PathEnhancer | None = None,
        html_writer: HTMLWriter | None = None,
        event_bus: EventBus | None = None,
        database_path: str | None = None,
        enable_rag: bool = True,
    ) -> None:
        self._search_orchestrator = search_orchestrator or SearchOrchestrator()
        self._consensus_ranker = consensus_ranker or ConsensusRanker()
        self._learning_path_builder = learning_path_builder or LearningPathBuilder()
        self._source_summarizer = source_summarizer or SourceSummarizer(use_llm=True)
        self._path_enhancer = path_enhancer or PathEnhancer(use_llm=True)
        self._enable_rag = enable_rag
        self._event_bus = event_bus or EventBus()
        self._logger = get_logger("core.pipeline")
        self._trace = get_trace_logger()

        output_directory = get_output_directory()

        self._file_manager = FileManager(output_directory)
        self._json_writer = JSONWriter(self._file_manager)
        self._markdown_writer = MarkdownWriter(self._file_manager)
        self._html_writer = html_writer or HTMLWriter(self._file_manager)

        resolved_database_path = database_path or str(
            get_data_directory() / "research_agent.db"
        )

        self._sqlite_store = SQLiteStore(resolved_database_path)

    async def __aenter__(self) -> ResearchPipeline:
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        await self.close()
        return False

    async def initialize(self) -> None:
        await self._sqlite_store.initialize()

    async def close(self) -> None:
        await self._search_orchestrator.close()

    def subscribe(self, event_name: str, handler: Any) -> None:
        self._event_bus.subscribe(event_name, handler)

    async def _save_memory(self, state: PipelineState) -> None:
        try:
            if state.status == TaskStatus.FAILED:
                return

            from agent.memory.episodic import EpisodicMemory
            from agent.memory.long_term import LongTermMemory

            topic = state.query.topic if state.query else ""
            goal = state.query.goal if state.query else ""
            level = state.query.level if state.query else ""

            outcome = self._enum_value(state.status)
            quality_score = 1.0 if state.status == TaskStatus.SUCCESS else 0.6

            long_term_memory = LongTermMemory()

            long_term_memory.save_topic(
                topic=topic,
                goal=goal,
                level=level,
                outcome=outcome,
                quality_score=quality_score,
                sources_used=len(state.sources),
                tokens_used=0,
                metadata={
                    "request_id": state.request_id,
                    "ranked_sources": len(state.ranked_sources),
                    "learning_steps": len(state.learning_path.steps) if state.learning_path else 0,
                },
            )

            episodic_memory = EpisodicMemory()

            episodic_memory.record_episode(
                task=topic,
                task_type="research_pipeline",
                goal=goal,
                level=level,
                strategy_used="linear_pipeline",
                actions_taken=[
                    "search",
                    "rank",
                    "summarize",
                    "learning_path",
                    "save_outputs",
                ],
                tools_used=[
                    "search_orchestrator",
                    "consensus_ranker",
                    "source_summarizer",
                    "learning_path_builder",
                    "path_enhancer",
                ],
                tokens_used=0,
                tokens_by_category={},
                total_iterations=1,
                outcome=outcome,
                quality_score=quality_score,
                success=state.status == TaskStatus.SUCCESS,
                errors=list(state.errors),
                findings_count=len(state.ranked_sources),
                started_at=state.created_at.isoformat() if state.created_at else "",
                latency_ms=state.latency_ms or 0.0,
            )

        except Exception as exc:
            state.add_warning(f"memory_save_failed: {exc}")

    async def run(
        self,
        query: SearchQuerySchema | str | dict[str, Any],
    ) -> PipelineResult:
        state = PipelineState()
        pipeline_start = time.perf_counter()

        try:
            query_schema = self._coerce_query(query)
        except Exception as exc:
            state.mark_failed(f"invalid_query: {exc}")

            self._trace.emit(
                "pipeline_failed",
                request_id=state.request_id,
                reason="invalid_query",
                error=str(exc),
            )

            await self._event_bus.emit(PIPELINE_FAILED, state.to_summary())
            return self._build_result(state, {})

        state.query = query_schema
        state.mark_running()

        self._trace.emit(
            "pipeline_started",
            request_id=state.request_id,
            topic=query_schema.topic,
            goal=query_schema.goal,
            level=query_schema.level,
            max_results=query_schema.max_results,
            enable_rag=self._enable_rag,
        )

        await self._event_bus.emit(PIPELINE_STARTED, state.to_summary())

        await self._run_search(state)

        if state.sources:
            await self._run_ranking(state)
        else:
            has_no_source_reason = False

            for error in state.errors:
                if str(error).startswith("no_sources_reason"):
                    has_no_source_reason = True
                    break

            if not has_no_source_reason:
                state.add_error("no_sources_found")

            self._trace.emit(
                "pipeline_no_sources",
                request_id=state.request_id,
            )

        if state.ranked_sources:
            await self._run_source_summaries(state)
            await self._run_learning_path(state)
            await self._run_path_enhancement(state)

        if self._enable_rag:
            await self._run_rag_indexing(state)

        total_ms = (time.perf_counter() - pipeline_start) * 1000

        if state.status == TaskStatus.FAILED:
            self._trace.emit(
                "pipeline_failed",
                request_id=state.request_id,
                errors=state.errors,
                total_latency_ms=round(total_ms, 1),
            )

            await self._event_bus.emit(PIPELINE_FAILED, state.to_summary())
        else:
            state.mark_success()

            self._trace.emit(
                "pipeline_completed",
                request_id=state.request_id,
                status=self._enum_value(state.status),
                total_sources=len(state.sources),
                total_ranked=len(state.ranked_sources),
                total_summaries=len(state.source_summaries),
                rag_indexed=state.rag_documents_indexed,
                errors=state.errors,
                warnings=state.warnings,
                total_latency_ms=round(total_ms, 1),
            )

            await self._event_bus.emit(PIPELINE_COMPLETED, state.to_summary())

        saved_files = await self._save_outputs(state)
        await self._save_memory(state)
        return self._build_result(state, saved_files)

    async def _run_search(self, state: PipelineState) -> None:
        if state.query is None:
            state.add_error("missing_query")
            return

        stage_start = time.perf_counter()

        try:
            search_result = await self._search_orchestrator.run(state.query)

            state.sources = search_result.sources
            state.expanded_queries = search_result.expanded_queries

            if getattr(search_result, "query", None) is not None:
                state.query = search_result.query

            seen_warnings = set(state.warnings)
            seen_errors = set(state.errors)

            def add_warning(value: Any) -> None:
                text = str(value)

                if text and text not in seen_warnings:
                    state.add_warning(text)
                    seen_warnings.add(text)

            def add_error(value: Any) -> None:
                text = str(value)

                if text and text not in seen_errors:
                    state.add_error(text)
                    seen_errors.add(text)

            for warning in getattr(search_result, "warnings", []) or []:
                add_warning(warning)

            platform_failures = [
                str(item)
                for item in getattr(search_result, "platform_failures", []) or []
            ]

            failed_platforms = [
                str(item)
                for item in getattr(search_result, "failed_platforms", []) or []
            ]

            platform_failure_detected = bool(platform_failures or failed_platforms)

            for error in search_result.errors:
                text = str(error)

                if text.startswith(
                    (
                        "platform_unavailable:",
                        "platform_skipped",
                        "platform_degraded:",
                        "platform_error:",
                        "search_platforms_failed",
                    )
                ):
                    platform_failure_detected = True
                    add_warning(text)
                elif text == "no_sources_matched_query":
                    add_warning(text)
                else:
                    add_error(text)

            for failure in platform_failures:
                platform_failure_detected = True
                add_warning(failure)

            if failed_platforms:
                platform_failure_detected = True
                add_warning(
                    "search_platforms_failed: " + ", ".join(sorted(set(failed_platforms)))
                )

            total_found = int(getattr(search_result, "total_found", 0) or 0)
            total_valid = int(getattr(search_result, "total_valid", 0) or 0)

            if not state.sources:
                reasons: list[str] = []

                if platform_failure_detected:
                    reasons.append("no_sources_reason_platform_failure")

                if total_found > 0 and total_valid == 0:
                    reasons.append("no_sources_reason_validation_rejection")

                if not reasons:
                    reasons.append("no_sources_reason_weak_query_or_no_results")

                for reason in reasons:
                    add_error(reason)

            stage_ms = (time.perf_counter() - stage_start) * 1000

            self._trace.emit(
                "stage_completed",
                request_id=state.request_id,
                stage="search",
                sources_found=len(state.sources),
                total_found=total_found,
                total_valid=total_valid,
                platform_failure_detected=platform_failure_detected,
                latency_ms=round(stage_ms, 1),
            )

            await self._event_bus.emit(
                SEARCH_COMPLETED,
                {
                    "request_id": state.request_id,
                    "total_sources": len(state.sources),
                    "total_found": total_found,
                    "total_valid": total_valid,
                    "platform_failure_detected": platform_failure_detected,
                },
            )
        except Exception as exc:
            stage_ms = (time.perf_counter() - stage_start) * 1000

            self._logger.error(f"Search stage failed: {exc}")
            state.add_error(f"search_failed: {exc}")

            self._trace.emit(
                "stage_failed",
                request_id=state.request_id,
                stage="search",
                error=str(exc),
                latency_ms=round(stage_ms, 1),
            )

            await self._event_bus.emit(
                STAGE_FAILED,
                {
                    "request_id": state.request_id,
                    "stage": "search",
                    "error": str(exc),
                },
            )

    async def _run_ranking(self, state: PipelineState) -> None:
        if state.query is None:
            state.add_error("missing_query")
            return

        stage_start = time.perf_counter()

        try:
            ranked_sources = await self._consensus_ranker.rank(
                state.sources,
                state.query,
            )

            state.ranked_sources = ranked_sources

            stage_ms = (time.perf_counter() - stage_start) * 1000

            self._trace.emit(
                "stage_completed",
                request_id=state.request_id,
                stage="ranking",
                total_ranked=len(state.ranked_sources),
                latency_ms=round(stage_ms, 1),
            )

            await self._event_bus.emit(
                RANKING_COMPLETED,
                {
                    "request_id": state.request_id,
                    "total_ranked": len(state.ranked_sources),
                },
            )
        except Exception as exc:
            stage_ms = (time.perf_counter() - stage_start) * 1000

            self._logger.error(f"Ranking stage failed: {exc}")
            state.add_error(f"ranking_failed: {exc}")

            self._trace.emit(
                "stage_failed",
                request_id=state.request_id,
                stage="ranking",
                error=str(exc),
                latency_ms=round(stage_ms, 1),
            )

            await self._event_bus.emit(
                STAGE_FAILED,
                {
                    "request_id": state.request_id,
                    "stage": "ranking",
                    "error": str(exc),
                },
            )

    async def _run_source_summaries(self, state: PipelineState) -> None:
        stage_start = time.perf_counter()

        try:
            ranked_source_objects = [
                ranked.source
                for ranked in state.ranked_sources
                if hasattr(ranked, "source")
            ]

            source_summaries = await self._source_summarizer.summarize_sources(
                ranked_source_objects
            )

            state.source_summaries = source_summaries

            stage_ms = (time.perf_counter() - stage_start) * 1000

            self._trace.emit(
                "stage_completed",
                request_id=state.request_id,
                stage="source_summaries",
                total_summaries=len(state.source_summaries),
                latency_ms=round(stage_ms, 1),
            )

            await self._event_bus.emit(
                SOURCE_SUMMARIES_COMPLETED,
                {
                    "request_id": state.request_id,
                    "total_summaries": len(state.source_summaries),
                },
            )
        except Exception as exc:
            stage_ms = (time.perf_counter() - stage_start) * 1000

            self._logger.warning(f"Source summary stage failed: {exc}")
            state.add_warning(f"source_summaries_failed: {exc}")

            self._trace.emit(
                "stage_failed",
                request_id=state.request_id,
                stage="source_summaries",
                error=str(exc),
                latency_ms=round(stage_ms, 1),
            )

            await self._event_bus.emit(
                STAGE_FAILED,
                {
                    "request_id": state.request_id,
                    "stage": "source_summaries",
                    "error": str(exc),
                },
            )

    async def _run_learning_path(self, state: PipelineState) -> None:
        if state.query is None:
            state.add_error("missing_query")
            return

        stage_start = time.perf_counter()

        try:
            learning_path = await self._learning_path_builder.build(
                state.ranked_sources,
                state.query,
            )

            state.learning_path = learning_path

            if not learning_path.steps:
                state.add_warning("learning_path_empty")

            stage_ms = (time.perf_counter() - stage_start) * 1000

            self._trace.emit(
                "stage_completed",
                request_id=state.request_id,
                stage="learning_path",
                total_steps=len(learning_path.steps),
                latency_ms=round(stage_ms, 1),
            )

            await self._event_bus.emit(
                LEARNING_PATH_COMPLETED,
                {
                    "request_id": state.request_id,
                    "total_steps": len(learning_path.steps),
                },
            )
        except Exception as exc:
            stage_ms = (time.perf_counter() - stage_start) * 1000

            self._logger.error(f"Learning path stage failed: {exc}")
            state.add_error(f"learning_path_failed: {exc}")

            self._trace.emit(
                "stage_failed",
                request_id=state.request_id,
                stage="learning_path",
                error=str(exc),
                latency_ms=round(stage_ms, 1),
            )

            await self._event_bus.emit(
                STAGE_FAILED,
                {
                    "request_id": state.request_id,
                    "stage": "learning_path",
                    "error": str(exc),
                },
            )

    async def _run_path_enhancement(self, state: PipelineState) -> None:
        if state.learning_path is None or not state.learning_path.steps:
            return

        stage_start = time.perf_counter()

        try:
            enhanced_learning_path = await self._path_enhancer.enhance(
                state.learning_path,
                state.query,
            )

            state.enhanced_learning_path = enhanced_learning_path

            stage_ms = (time.perf_counter() - stage_start) * 1000

            self._trace.emit(
                "stage_completed",
                request_id=state.request_id,
                stage="path_enhancement",
                latency_ms=round(stage_ms, 1),
            )

            await self._event_bus.emit(
                PATH_ENHANCED,
                {
                    "request_id": state.request_id,
                    "total_enhancements": len(
                        getattr(enhanced_learning_path, "enhancements", []) or []
                    ),
                },
            )
        except Exception as exc:
            stage_ms = (time.perf_counter() - stage_start) * 1000

            self._logger.warning(f"Path enhancement failed: {exc}")
            state.add_warning(f"path_enhancement_failed: {exc}")

            self._trace.emit(
                "stage_failed",
                request_id=state.request_id,
                stage="path_enhancement",
                error=str(exc),
                latency_ms=round(stage_ms, 1),
            )

            await self._event_bus.emit(
                STAGE_FAILED,
                {
                    "request_id": state.request_id,
                    "stage": "path_enhancement",
                    "error": str(exc),
                },
            )

    async def _run_rag_indexing(self, state: PipelineState) -> None:
        stage_start = time.perf_counter()

        await self._event_bus.emit(
            RAG_INDEXING_STARTED,
            {
                "request_id": state.request_id,
                "total_sources": len(state.ranked_sources),
            },
        )

        try:
            from rag.embedder import Embedder
            from rag.vector_store import VectorStore

            async with Embedder() as embedder:
                embeddings = await embedder.embed_ranked_sources(
                    state.ranked_sources
                )

                vector_store = VectorStore()

                indexed_documents = vector_store.add_ranked_sources(
                    ranked_sources=state.ranked_sources,
                    embeddings=embeddings,
                    run_id=state.request_id,
                )

                state.rag_documents_indexed = int(indexed_documents)

            stage_ms = (time.perf_counter() - stage_start) * 1000

            self._trace.emit(
                "stage_completed",
                request_id=state.request_id,
                stage="rag_indexing",
                total_documents=state.rag_documents_indexed,
                latency_ms=round(stage_ms, 1),
            )

            await self._event_bus.emit(
                RAG_INDEXED,
                {
                    "request_id": state.request_id,
                    "total_documents": state.rag_documents_indexed,
                },
            )
        except Exception as exc:
            stage_ms = (time.perf_counter() - stage_start) * 1000

            self._logger.warning(f"RAG indexing failed: {exc}")
            state.add_warning(f"rag_indexing_failed: {exc}")

            self._trace.emit(
                "stage_failed",
                request_id=state.request_id,
                stage="rag_indexing",
                error=str(exc),
                latency_ms=round(stage_ms, 1),
            )

            await self._event_bus.emit(
                STAGE_FAILED,
                {
                    "request_id": state.request_id,
                    "stage": "rag_indexing",
                    "error": str(exc),
                },
            )

    async def _save_outputs(self, state: PipelineState) -> dict[str, str]:
        saved_files: dict[str, str] = {}
        prefix = self._file_prefix(state)

        try:
            json_filename = self._file_manager.timestamped_filename(prefix, "json")
            json_path = self._json_writer.write(
                self._serialize_state(state),
                json_filename,
                subdirectory="json",
            )
            saved_files["json"] = str(json_path)
        except Exception as exc:
            state.add_error(f"json_save_failed: {exc}")

        try:
            markdown_filename = self._file_manager.timestamped_filename(prefix, "md")
            markdown_path = self._markdown_writer.write_state(
                state,
                markdown_filename,
                subdirectory="markdown",
            )
            saved_files["markdown"] = str(markdown_path)
        except Exception as exc:
            state.add_error(f"markdown_save_failed: {exc}")

        try:
            html_filename = self._file_manager.timestamped_filename(prefix, "html")
            html_path = self._html_writer.write_state(
                state,
                html_filename,
                subdirectory="reports",
            )
            saved_files["html"] = str(html_path)
        except Exception as exc:
            state.add_error(f"html_save_failed: {exc}")

        try:
            await self._sqlite_store.save_state(state)
            saved_files["sqlite"] = str(self._sqlite_store.database_path)
        except Exception as exc:
            state.add_error(f"sqlite_save_failed: {exc}")

        self._trace.emit(
            "outputs_saved",
            request_id=state.request_id,
            files=list(saved_files.keys()),
        )

        return saved_files

    def _build_result(
        self,
        state: PipelineState,
        saved_files: dict[str, str],
    ) -> PipelineResult:
        return PipelineResult(
            request_id=state.request_id,
            status=state.status,
            query=state.query,
            expanded_queries=state.expanded_queries,
            sources=state.sources,
            ranked_sources=state.ranked_sources,
            source_summaries=state.source_summaries,
            learning_path=state.learning_path,
            enhanced_learning_path=state.enhanced_learning_path,
            rag_documents_indexed=state.rag_documents_indexed,
            errors=state.errors,
            warnings=state.warnings,
            saved_files=saved_files,
            started_at=state.started_at,
            finished_at=state.finished_at,
            latency_ms=state.latency_ms,
        )

    def _serialize_state(self, state: PipelineState) -> dict[str, Any]:
        return {
            "request_id": state.request_id,
            "status": self._enum_value(state.status),
            "query": state.query.model_dump(mode="json") if state.query else None,
            "expanded_queries": state.expanded_queries,
            "sources": [source.model_dump(mode="json") for source in state.sources],
            "ranked_sources": [
                ranked.model_dump(mode="json") for ranked in state.ranked_sources
            ],
            "source_summaries": [
                self._serialize_model(summary) for summary in state.source_summaries
            ],
            "learning_path": state.learning_path.model_dump(mode="json")
            if state.learning_path
            else None,
            "enhanced_learning_path": self._serialize_model(
                state.enhanced_learning_path
            ),
            "rag_documents_indexed": state.rag_documents_indexed,
            "errors": state.errors,
            "warnings": state.warnings,
            "created_at": state.created_at.isoformat() if state.created_at else None,
            "finished_at": state.finished_at.isoformat() if state.finished_at else None,
            "latency_ms": state.latency_ms,
        }

    def _file_prefix(self, state: PipelineState) -> str:
        topic = state.query.topic if state.query else "research"
        cleaned = self._file_manager.sanitize_filename(topic, max_length=60)
        return cleaned or "research"

    def _serialize_model(self, value: Any) -> Any:
        if value is None:
            return None

        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")

        return value

    def _coerce_query(
        self,
        query: SearchQuerySchema | str | dict[str, Any],
    ) -> SearchQuerySchema:
        if isinstance(query, SearchQuerySchema):
            return query

        if isinstance(query, str):
            return SearchQuerySchema(topic=query)

        if isinstance(query, dict):
            return SearchQuerySchema.model_validate(query)

        raise ResearchAgentError(
            "Unsupported query type",
            details={"type": type(query).__name__},
        )

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value))