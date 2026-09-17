from __future__ import annotations

import time
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from ranking.scorer import HeuristicScorer
from uuid import uuid4

from pydantic import Field

from core.config import get_data_directory, get_output_directory, get_settings
from core.events import EventBus
from core.models import (
    AgentAction,
    AgentStatus,
    CoreModel,
    LearningPath,
    RankedSource,
    Source,
    TaskStatus,
)
from core.schemas import SearchQuerySchema
from core.state import PipelineState

from agent.loop.engine import AgenticLoop
from agent.loop.state import LoopConfig, LoopOutput
from agent.memory.long_term import LongTermMemory
from agent.memory.retrieval import MemoryRetrieval
from agent.memory.short_term import ShortTermMemory
from agent.planner.planner import AgentPlanner
from agent.reflection.engine import ReflectionEngine
from agent.token.budget import TokenBudget, load_token_budget_config
from agent.token.cost_tracker import CostTracker
from agent.tools.executor import ToolExecutor
from agent.tools.registry import ToolRegistry
from agent.tools.search_tools import register_search_tools
from agent.tools.read_tools import register_read_tools
from agent.tools.analysis_tools import register_analysis_tools
from agent.tools.gen_tools import register_gen_tools
from agent.tools.memory_tools import register_memory_tools

from ranking.consensus_ranker import ConsensusRanker
from ranking.learning_path_builder import LearningPathBuilder
from ranking.path_enhancer import PathEnhancer
from ranking.source_summarizer import SourceSummarizer

from storage.file_manager import FileManager
from storage.html_writer import HTMLWriter
from storage.json_writer import JSONWriter
from storage.markdown_writer import MarkdownWriter
from storage.sqlite_store import SQLiteStore

from utils.logger import get_logger, get_trace_logger

AGENT_STARTED = "agent.started"
AGENT_MEMORY_RECALLED = "agent.memory_recalled"
AGENT_PLAN_CREATED = "agent.plan_created"
AGENT_LOOP_STARTED = "agent.loop_started"
AGENT_LOOP_COMPLETED = "agent.loop_completed"
AGENT_SUMMARIES_COMPLETED = "agent.summaries_completed"
AGENT_LEARNING_PATH_COMPLETED = "agent.learning_path_completed"
AGENT_PATH_ENHANCED = "agent.path_enhanced"
AGENT_COMPLETED = "agent.completed"
AGENT_FAILED = "agent.failed"
AGENT_STAGE_FAILED = "agent.stage_failed"

_AGENT_CONFIG_CACHE: dict[str, Any] | None = None


class AgentConfig(CoreModel):
    max_iterations: int = Field(default=15, ge=1, le=50)
    max_retries_per_step: int = Field(default=2, ge=0, le=5)
    max_actions_per_step: int = Field(default=4, ge=1, le=10)
    fast_model: str = ""
    strong_model: str = ""
    judge_model: str = ""
    memory_enabled: bool = True
    reflection_enabled: bool = True
    token_budget: int = Field(default=40000, ge=1000, le=500000)
    quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    enable_rag: bool = False


class AgentResult(CoreModel):
    agent_id: str = ""
    status: str = "failed"
    goal: str = ""
    level: str = ""
    topic: str = ""
    findings: list[Source] = Field(default_factory=list)
    ranked_sources: list[RankedSource] = Field(default_factory=list)
    source_summaries: list[Any] = Field(default_factory=list)
    learning_path: LearningPath | None = None
    enhanced_learning_path: Any = None
    total_iterations: int = Field(default=0, ge=0)
    total_tokens_used: int = Field(default=0, ge=0)
    total_budget: int = Field(default=0, ge=0)
    tokens_remaining: int = Field(default=0, ge=0)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    saved_files: dict[str, str] = Field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    latency_ms: float | None = None

    @property
    def success(self) -> bool:
        return self.status in ("success", "partial")


def load_agent_config(config_path: str | Path | None = None) -> AgentConfig:
    global _AGENT_CONFIG_CACHE
    resolved_path = _resolve_agent_config_path(config_path)
    cache_key = str(resolved_path)
    if _AGENT_CONFIG_CACHE is not None and cache_key in _AGENT_CONFIG_CACHE:
        return _AGENT_CONFIG_CACHE[cache_key]
    config = _load_agent_config_from_yaml(resolved_path)
    if _AGENT_CONFIG_CACHE is None:
        _AGENT_CONFIG_CACHE = {}
    _AGENT_CONFIG_CACHE[cache_key] = config
    return config


def _resolve_agent_config_path(config_path: str | Path | None) -> Path:
    if config_path is not None:
        return Path(config_path)
    return Path(__file__).resolve().parents[1] / "configs" / "agent_config.yaml"


def _load_agent_config_from_yaml(path: Path) -> AgentConfig:
    logger = get_logger("agent.controller")
    if not path.exists():
        logger.warning(f"Agent config not found at {path}, using defaults")
        return AgentConfig()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except Exception:
        return AgentConfig()
    if not isinstance(raw, dict):
        return AgentConfig()
    agent_section = raw.get("agent", {})
    if not isinstance(agent_section, dict):
        agent_section = raw
    try:
        return AgentConfig.model_validate(agent_section)
    except Exception as exc:
        logger.warning(f"Failed to validate agent config: {exc}, using defaults")
        return AgentConfig()


class AutonomousAgent:
    def __init__(
        self,
        agent_config: AgentConfig | None = None,
        event_bus: EventBus | None = None,
        database_path: str | None = None,
    ) -> None:
        self._config = agent_config or load_agent_config()
        self._event_bus = event_bus or EventBus()
        self._registry = ToolRegistry()
        self._register_all_tools()
        self._budget_config = load_token_budget_config()
        self._short_term_memory = ShortTermMemory()
        self._long_term_memory = LongTermMemory()
        self._memory_retrieval = MemoryRetrieval(
            long_term=self._long_term_memory,
            short_term=self._short_term_memory,
        )
        self._planner = AgentPlanner()
        self._reflection_engine = ReflectionEngine()
        self._logger = get_logger("agent.controller")
        self._trace = get_trace_logger()

        output_directory = get_output_directory()
        self._file_manager = FileManager(output_directory)
        self._json_writer = JSONWriter(self._file_manager)
        self._markdown_writer = MarkdownWriter(self._file_manager)
        self._html_writer = HTMLWriter(self._file_manager)

        resolved_database_path = database_path or str(
            get_data_directory() / "research_agent.db"
        )
        self._sqlite_store = SQLiteStore(resolved_database_path)

    async def __aenter__(self) -> "AutonomousAgent":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        await self.close()
        return False

    async def initialize(self) -> None:
        await self._sqlite_store.initialize()

    async def close(self) -> None:
        pass

    def subscribe(self, event_name: str, handler: Any) -> None:
        self._event_bus.subscribe(event_name, handler)

    def _register_all_tools(self) -> None:
        register_search_tools(self._registry)
        register_read_tools(self._registry)
        register_analysis_tools(self._registry)
        register_gen_tools(self._registry)
        register_memory_tools(self._registry)

    async def run(
        self,
        goal: str,
        level: str = "",
        topic: str = "",
        max_results: int = 20,
    ) -> AgentResult:
        agent_id = str(uuid4())
        agent_start = time.perf_counter()
        resolved_topic = topic or goal

        errors: list[str] = []
        warnings: list[str] = []

        self._trace.emit(
            "agent_run_started",
            agent_id=agent_id,
            goal=goal,
            level=level,
            topic=resolved_topic,
            token_budget=self._config.token_budget,
            max_iterations=self._config.max_iterations,
        )

        await self._event_bus.emit(
            AGENT_STARTED,
            {
                "agent_id": agent_id,
                "goal": goal,
                "level": level,
                "topic": resolved_topic,
            },
        )

        budget = TokenBudget(
            task_id=agent_id,
            config=self._budget_config,
            total_budget=self._config.token_budget,
        )
        cost_tracker = CostTracker(
            task_id=agent_id,
            budget=budget,
        )
        executor = ToolExecutor(
            registry=self._registry,
            cost_tracker=cost_tracker,
        )

        loop_config = LoopConfig(
            max_iterations=self._config.max_iterations,
            max_actions_per_step=self._config.max_actions_per_step,
            token_budget=self._config.token_budget,
            quality_threshold=self._config.quality_threshold,
            enable_reflection=self._config.reflection_enabled,
            enable_memory=self._config.memory_enabled,
        )

        loop = AgenticLoop(
            registry=self._registry,
            executor=executor,
            planner=self._planner,
            cost_tracker=cost_tracker,
            short_term_memory=self._short_term_memory,
            event_bus=self._event_bus,
            config=loop_config,
        )

        self._short_term_memory.initialize(
            goal=goal,
            level=level,
            topic=resolved_topic,
        )

        memory_context = ""
        if self._config.memory_enabled:
            try:
                memory_context = self._memory_retrieval.retrieve_context(
                    goal=goal,
                    level=level,
                )
                self._trace.emit(
                    "agent_memory_recalled",
                    agent_id=agent_id,
                    context_length=len(memory_context),
                )
                await self._event_bus.emit(
                    AGENT_MEMORY_RECALLED,
                    {
                        "agent_id": agent_id,
                        "context_length": len(memory_context),
                    },
                )
            except Exception as exc:
                warnings.append(f"memory_recall_failed: {exc}")

        query_schema = SearchQuerySchema(
            topic=resolved_topic,
            goal=goal,
            level=level,
            max_results=max_results,
        )

        loop_output: LoopOutput | None = None

        try:
            self._trace.emit(
                "agent_loop_started",
                agent_id=agent_id,
                goal=goal,
            )
            await self._event_bus.emit(
                AGENT_LOOP_STARTED,
                {"agent_id": agent_id, "goal": goal},
            )

            loop_output = await loop.run(
                goal=goal,
                level=level,
                topic=resolved_topic,
                token_budget=self._config.token_budget,
                max_iterations=self._config.max_iterations,
            )

            self._trace.emit(
                "agent_loop_completed",
                agent_id=agent_id,
                status=self._enum_value(loop_output.status),
                findings=len(loop_output.findings),
                ranked=len(loop_output.ranked_sources),
                iterations=loop_output.total_iterations,
                tokens_used=loop_output.total_tokens_used,
            )
            await self._event_bus.emit(
                AGENT_LOOP_COMPLETED,
                {
                    "agent_id": agent_id,
                    "status": self._enum_value(loop_output.status),
                    "findings": len(loop_output.findings),
                    "ranked": len(loop_output.ranked_sources),
                    "iterations": loop_output.total_iterations,
                },
            )
        except Exception as exc:
            errors.append(f"agent_loop_failed: {exc}")
            self._trace.emit(
                "agent_loop_failed",
                agent_id=agent_id,
                error=str(exc),
            )
            await self._event_bus.emit(
                AGENT_STAGE_FAILED,
                {
                    "agent_id": agent_id,
                    "stage": "agent_loop",
                    "error": str(exc),
                },
            )

        findings = loop_output.findings if loop_output else []
        ranked_sources = loop_output.ranked_sources if loop_output else []
        total_iterations = loop_output.total_iterations if loop_output else 0
        total_tokens_used = loop_output.total_tokens_used if loop_output else 0

        if loop_output:
            errors.extend(loop_output.errors)
            warnings.extend(loop_output.warnings)

        if not ranked_sources and findings:
            try:
                fallback_ranker = ConsensusRanker(use_llm=True)
                ranked_sources = await fallback_ranker.rank(findings, query_schema)
            except Exception as exc:
                warnings.append(f"fallback_ranking_failed: {exc}")

            if not ranked_sources:
                try:
                    fallback_scorer = HeuristicScorer()
                    ranked_sources = fallback_scorer.score_sources(
                        findings,
                        query=query_schema,
                        goal=query_schema.goal,
                        level=query_schema.level,
                    )
                except Exception as exc:
                    warnings.append(f"heuristic_fallback_failed: {exc}")

        source_summaries: list[Any] = []
        learning_path: LearningPath | None = None
        enhanced_learning_path: Any = None

        if ranked_sources:
            try:
                source_summaries = await self._run_source_summaries(
                    ranked_sources,
                )
                self._trace.emit(
                    "agent_summaries_completed",
                    agent_id=agent_id,
                    total_summaries=len(source_summaries),
                )
                await self._event_bus.emit(
                    AGENT_SUMMARIES_COMPLETED,
                    {
                        "agent_id": agent_id,
                        "total_summaries": len(source_summaries),
                    },
                )
            except Exception as exc:
                warnings.append(f"source_summaries_failed: {exc}")
                await self._event_bus.emit(
                    AGENT_STAGE_FAILED,
                    {
                        "agent_id": agent_id,
                        "stage": "source_summaries",
                        "error": str(exc),
                    },
                )

            try:
                learning_path = await self._run_learning_path(
                    ranked_sources,
                    query_schema,
                )
                self._trace.emit(
                    "agent_learning_path_completed",
                    agent_id=agent_id,
                    total_steps=len(learning_path.steps) if learning_path else 0,
                )
                await self._event_bus.emit(
                    AGENT_LEARNING_PATH_COMPLETED,
                    {
                        "agent_id": agent_id,
                        "total_steps": len(learning_path.steps) if learning_path else 0,
                    },
                )
            except Exception as exc:
                warnings.append(f"learning_path_failed: {exc}")
                await self._event_bus.emit(
                    AGENT_STAGE_FAILED,
                    {
                        "agent_id": agent_id,
                        "stage": "learning_path",
                        "error": str(exc),
                    },
                )

            if learning_path and learning_path.steps:
                try:
                    enhanced_learning_path = await self._run_path_enhancement(
                        learning_path,
                        query_schema,
                    )
                    self._trace.emit(
                        "agent_path_enhanced",
                        agent_id=agent_id,
                    )
                    await self._event_bus.emit(
                        AGENT_PATH_ENHANCED,
                        {"agent_id": agent_id},
                    )
                except Exception as exc:
                    warnings.append(f"path_enhancement_failed: {exc}")
                    await self._event_bus.emit(
                        AGENT_STAGE_FAILED,
                        {
                            "agent_id": agent_id,
                            "stage": "path_enhancement",
                            "error": str(exc),
                        },
                    )

        pipeline_state = self._to_pipeline_state(
            agent_id=agent_id,
            query_schema=query_schema,
            findings=findings,
            ranked_sources=ranked_sources,
            source_summaries=source_summaries,
            learning_path=learning_path,
            enhanced_learning_path=enhanced_learning_path,
            errors=errors,
            warnings=warnings,
        )

        saved_files = await self._save_outputs(pipeline_state)
        total_ms = (time.perf_counter() - agent_start) * 1000

        status = "success"
        if errors and not ranked_sources:
            status = "failed"
        elif errors or warnings:
            status = "partial"

        result = AgentResult(
            agent_id=agent_id,
            status=status,
            goal=goal,
            level=level,
            topic=resolved_topic,
            findings=findings,
            ranked_sources=ranked_sources,
            source_summaries=source_summaries,
            learning_path=learning_path,
            enhanced_learning_path=enhanced_learning_path,
            total_iterations=total_iterations,
            total_tokens_used=total_tokens_used,
            total_budget=self._config.token_budget,
            tokens_remaining=budget.remaining,
            errors=errors,
            warnings=warnings,
            saved_files=saved_files,
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            latency_ms=total_ms,
        )

        self._trace.emit(
            "agent_run_completed",
            agent_id=agent_id,
            status=status,
            total_findings=len(findings),
            total_ranked=len(ranked_sources),
            total_summaries=len(source_summaries),
            total_iterations=total_iterations,
            total_tokens_used=total_tokens_used,
            tokens_remaining=budget.remaining,
            errors=errors,
            warnings=warnings,
            total_latency_ms=round(total_ms, 1),
        )

        if status == "failed":
            await self._event_bus.emit(AGENT_FAILED, result.model_dump(mode="json"))
        else:
            await self._event_bus.emit(AGENT_COMPLETED, result.model_dump(mode="json"))

        cost_tracker.close()
        return result

    async def _run_source_summaries(
        self,
        ranked_sources: list[RankedSource],
    ) -> list[Any]:
        summarizer = SourceSummarizer(use_llm=True)
        source_objects = [
            ranked.source
            for ranked in ranked_sources
            if hasattr(ranked, "source")
        ]
        return await summarizer.summarize_sources(source_objects)

    async def _run_learning_path(
        self,
        ranked_sources: list[RankedSource],
        query_schema: SearchQuerySchema,
    ) -> LearningPath | None:
        builder = LearningPathBuilder(use_llm=True)
        return await builder.build(ranked_sources, query_schema)

    async def _run_path_enhancement(
        self,
        learning_path: LearningPath,
        query_schema: SearchQuerySchema,
    ) -> Any:
        enhancer = PathEnhancer(use_llm=True)
        return await enhancer.enhance(learning_path, query_schema)

    def _to_pipeline_state(
        self,
        agent_id: str,
        query_schema: SearchQuerySchema,
        findings: list[Source],
        ranked_sources: list[RankedSource],
        source_summaries: list[Any],
        learning_path: LearningPath | None,
        enhanced_learning_path: Any,
        errors: list[str],
        warnings: list[str],
    ) -> PipelineState:
        state = PipelineState()
        state.request_id = agent_id
        state.query = query_schema
        state.sources = findings
        state.ranked_sources = ranked_sources
        state.source_summaries = source_summaries
        state.learning_path = learning_path
        state.enhanced_learning_path = enhanced_learning_path
        state.errors = list(errors)
        state.warnings = list(warnings)

        if errors and not ranked_sources:
            state.mark_failed(errors[0] if errors else None)
        elif errors or warnings:
            state.status = TaskStatus.PARTIAL
            state.finished_at = datetime.now(timezone.utc)
        else:
            state.mark_success()

        return state

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
            "agent_outputs_saved",
            request_id=state.request_id,
            files=list(saved_files.keys()),
        )

        return saved_files

    def _file_prefix(self, state: PipelineState) -> str:
        topic = state.query.topic if state.query else "agent_research"
        cleaned = self._file_manager.sanitize_filename(topic, max_length=60)
        return cleaned or "agent_research"

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

    def _serialize_model(self, value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        return value

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value))