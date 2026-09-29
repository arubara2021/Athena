from __future__ import annotations

import asyncio
import html
import importlib
import inspect
import json
import re
import time
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import Field

from core import constants
from core.config import (
    get_data_directory,
    get_output_directory,
    get_settings,
)
from core.events import EventBus
from core.models import (
    AgentAction,
    AgentActionType,
    AgentPlan,
    AgentStatus,
    CoreModel,
    PlanStep,
    RankedSource,
    Source,
)
from core.schemas import AgentInputSchema, SearchQuerySchema
from agent.loop.engine import AgenticLoop, get_mode_config
from agent.loop.state import LoopConfig, LoopOutput
from agent.memory.long_term import LongTermMemory
from agent.memory.retrieval import MemoryRetrieval
from agent.memory.short_term import ShortTermMemory
from agent.planner.planner import AgentPlanner
from agent.reflection.engine import ReflectionEngine
from agent.token.budget import load_token_budget_config
from agent.token.cost_tracker import CostTracker
from agent.tools.executor import ToolExecutor
from agent.tools.registry import ToolRegistry
from llm.provider import LLMProviderManager
from ranking.learning_path_builder import LearningPathBuilder
from ranking.path_enhancer import PathEnhancer
from ranking.scorer import HeuristicScorer
from ranking.source_summarizer import SourceSummarizer
from storage.sqlite_store import SQLiteStore
from utils.logger import get_logger, get_trace_logger
from utils.text import clean_text


class AgentResult(CoreModel):
    agent_id: str = ""
    status: str = "failed"
    goal: str = ""
    level: str = ""
    topic: str = ""
    mode: str = ""
    findings: list[Source] = Field(default_factory=list)
    ranked_sources: list[RankedSource] = Field(default_factory=list)
    source_summaries: list[Any] = Field(default_factory=list)
    learning_path: Any = None
    enhanced_learning_path: Any = None
    actions_taken: list[AgentAction] = Field(default_factory=list)
    total_iterations: int = Field(default=0, ge=0)
    total_tokens_used: int = Field(default=0, ge=0)
    total_budget: int = Field(default=0, ge=0)
    tokens_remaining: int = Field(default=0, ge=0)
    iteration_count: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=15, ge=1)
    final_output: Any = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    saved_files: dict[str, str] = Field(default_factory=dict)
    rag_documents_indexed: int = Field(default=0, ge=0)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    latency_ms: float | None = None

    @property
    def success(self) -> bool:
        return self.status in ("success", "partial")


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
    quality_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    enable_rag: bool = False
    mode: str = constants.DEFAULT_MODE


_AGENT_CONFIG_CACHE: dict[str, Any] | None = None


def _resolve_agent_config_path(
    config_path: str | Path | None,
) -> Path:
    if config_path is not None:
        return Path(config_path)

    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "agent_config.yaml"
    )


def _load_agent_config_from_yaml(path: Path) -> AgentConfig:
    logger = get_logger("agent.controller")

    if not path.exists():
        logger.warning(
            f"Agent config not found at {path}, using defaults"
        )
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
        logger.warning(
            f"Failed to validate agent config: {exc}, using defaults"
        )
        return AgentConfig()


def load_agent_config(
    config_path: str | Path | None = None,
) -> AgentConfig:
    resolved_path = _resolve_agent_config_path(config_path)
    cache_key = str(resolved_path)

    global _AGENT_CONFIG_CACHE

    if (
        _AGENT_CONFIG_CACHE is not None
        and cache_key in _AGENT_CONFIG_CACHE
    ):
        return _AGENT_CONFIG_CACHE[cache_key]

    config = _load_agent_config_from_yaml(resolved_path)

    if _AGENT_CONFIG_CACHE is None:
        _AGENT_CONFIG_CACHE = {}

    _AGENT_CONFIG_CACHE[cache_key] = config
    return config


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
        self._skipped_tool_modules: list[str] = []
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

        self._output_directory = get_output_directory()
        self._database_path = Path(
            database_path
            or str(get_data_directory() / "research_agent.db")
        )
        self._store = SQLiteStore(self._database_path)

        self._source_summarizer: SourceSummarizer | None = None
        self._path_enhancer: PathEnhancer | None = None

    def _get_source_summarizer(self) -> SourceSummarizer:
        if self._source_summarizer is None:
            self._source_summarizer = SourceSummarizer()

        return self._source_summarizer

    def _get_path_enhancer(self) -> PathEnhancer:
        if self._path_enhancer is None:
            self._path_enhancer = PathEnhancer(use_llm=True)

        return self._path_enhancer

    async def __aenter__(self) -> "AutonomousAgent":
        try:
            initialize = getattr(self._store, "initialize", None)

            if callable(initialize):
                await self._maybe_await(initialize())
        except Exception as exc:
            self._logger.warning(
                f"Failed to initialize agent storage: {exc}"
            )

        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc: Any,
        tb: Any,
    ) -> bool:
        return False

    def subscribe(self, event_name: str, handler: Any) -> None:
        subscribe = getattr(self._event_bus, "subscribe", None)

        if callable(subscribe):
            subscribe(event_name, handler)

    def unsubscribe(self, event_name: str, handler: Any) -> None:
        unsubscribe = getattr(self._event_bus, "unsubscribe", None)

        if callable(unsubscribe):
            unsubscribe(event_name, handler)

    async def _check_input_coherence(
        self,
        raw_query: str,
    ) -> tuple[bool, str]:
        query = clean_text(raw_query)

        if not query:
            return False, "empty_query"

        try:
            from search.query_tokens import (
                get_generic_terms,
                strip_filler,
            )
        except Exception:
            return True, ""

        stripped = strip_filler(query)

        if not stripped:
            return True, ""

        tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", stripped.lower())
            if len(token) > 1
        ]

        if not tokens:
            return True, ""

        try:
            generics = get_generic_terms()
        except Exception:
            generics = frozenset()

        content_tokens = [
            token for token in tokens if token not in generics
        ]

        if len(content_tokens) < 3:
            return True, ""

        try:
            from search.query_intelligence import QueryIntelligence

            intel = QueryIntelligence(use_llm=False)
            analysis = await intel.analyze(
                topic=query,
                goal="",
                level="",
            )

            target_domains = list(
                getattr(analysis, "target_domains", []) or []
            )
            subject_domains = [
                domain
                for domain in target_domains
                if domain not in {"education", "general"}
            ]

            if subject_domains:
                return True, ""
        except Exception:
            return True, ""

        longest = max(len(token) for token in content_tokens)

        if longest > 5:
            return True, ""

        self._trace.emit(
            "agent_input_coherence_failed",
            query=query,
            stripped=stripped,
            content_tokens=content_tokens,
            longest_token_length=longest,
        )

        return False, "no_coherent_subject_or_technical_terms"

    async def _refuse_incoherent_query(
        self,
        agent_id: str,
        goal: str,
        level: str,
        topic: str,
        mode: str,
        reason: str,
        run_started_at: datetime,
        agent_start: float,
    ) -> AgentResult:
        refusal_message = (
            "The query does not describe a coherent research topic. "
            "Please rephrase with a specific subject, for example "
            "'learn dense retrieval from basics' or "
            "'astrophysics fundamentals'."
        )

        self._logger.warning(
            f"Refusing incoherent query: topic={topic!r} "
            f"reason={reason}"
        )

        self._trace.emit(
            "agent_input_refused",
            agent_id=agent_id,
            topic=topic,
            reason=reason,
        )

        finished_at = datetime.now(timezone.utc)
        latency_ms = (time.perf_counter() - agent_start) * 1000

        result = AgentResult(
            agent_id=agent_id,
            status="failed",
            goal=goal,
            level=level,
            topic=topic,
            mode=mode,
            findings=[],
            ranked_sources=[],
            source_summaries=[],
            learning_path=None,
            enhanced_learning_path=None,
            actions_taken=[],
            total_iterations=0,
            total_tokens_used=0,
            total_budget=self._config.token_budget,
            tokens_remaining=self._config.token_budget,
            iteration_count=0,
            max_iterations=self._config.max_iterations,
            final_output=None,
            errors=[refusal_message],
            warnings=[],
            saved_files={},
            rag_documents_indexed=0,
            started_at=run_started_at,
            finished_at=finished_at,
            latency_ms=latency_ms,
        )

        try:
            saved_files = self._save_outputs(result, topic or goal)
            self._assign_saved_files(result, saved_files)
        except Exception as exc:
            self._logger.warning(
                f"Failed to save refusal outputs: {exc}"
            )

        try:
            await self._save_agent_run(result, [])
        except Exception as exc:
            self._logger.warning(
                f"Failed to persist refusal run: {exc}"
            )

        await self._emit(
            "agent.refused",
            {
                "agent_id": agent_id,
                "topic": topic,
                "reason": reason,
                "message": refusal_message,
            },
        )

        return result

    async def run(
        self,
        goal: str,
        level: str = "",
        topic: str = "",
        max_results: int = 0,
        mode: str = "",
    ) -> AgentResult:
        agent_id = str(uuid4())
        run_started_at = datetime.now(timezone.utc)
        agent_start = time.perf_counter()
        resolved_topic = clean_text(topic or goal)

        resolved_mode = self._resolve_mode(mode)
        mode_config = get_mode_config(resolved_mode)

        effective_max_results = self._resolve_max_results(
            max_results, mode_config
        )

        errors: list[str] = []
        warnings: list[str] = []

        is_coherent, refusal_reason = (
            await self._check_input_coherence(resolved_topic)
        )

        if not is_coherent:
            return await self._refuse_incoherent_query(
                agent_id=agent_id,
                goal=goal,
                level=level,
                topic=resolved_topic,
                mode=resolved_mode,
                reason=refusal_reason,
                run_started_at=run_started_at,
                agent_start=agent_start,
            )

        await self._emit(
            "agent.started",
            {
                "agent_id": agent_id,
                "goal": goal,
                "level": level,
                "topic": resolved_topic,
                "mode": resolved_mode,
                "token_budget": self._config.token_budget,
                "max_iterations": mode_config.get(
                    "iterations", self._config.max_iterations
                ),
            },
        )

        memory_context = await self._recall_memory(
            goal, resolved_topic
        )

        await self._emit(
            "agent.memory_recalled",
            {
                "agent_id": agent_id,
                "context_length": len(memory_context),
            },
        )

        query_schema = SearchQuerySchema(
            topic=resolved_topic,
            goal=goal,
            level=level,
            max_results=effective_max_results,
        )

        await self._create_plan(
            goal,
            level,
            resolved_topic,
            errors,
            warnings,
        )

        cost_tracker = CostTracker(
            task_id=agent_id,
            budget=self._config.token_budget,
        )

        manager = LLMProviderManager.get_shared()
        manager.register_token_hook(cost_tracker.on_token_usage)

        self._trace.emit(
            "agent_token_hook_registered",
            agent_id=agent_id,
            budget=self._config.token_budget,
        )

        try:
            executor = ToolExecutor(
                registry=self._registry,
                cost_tracker=cost_tracker,
            )

            loop_config = LoopConfig(
                max_iterations=int(
                    mode_config.get(
                        "iterations", self._config.max_iterations
                    )
                ),
                max_actions_per_step=self._config.max_actions_per_step,
                token_budget=int(
                    mode_config.get(
                        "budget", self._config.token_budget
                    )
                ),
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

            await self._emit(
                "agent.loop_started",
                {"agent_id": agent_id, "goal": goal},
            )

            loop_output: LoopOutput | None = None

            try:
                loop_output = await loop.run(
                    goal=goal,
                    level=level,
                    topic=resolved_topic,
                    token_budget=int(
                        mode_config.get(
                            "budget", self._config.token_budget
                        )
                    ),
                    max_iterations=int(
                        mode_config.get(
                            "iterations",
                            self._config.max_iterations,
                        )
                    ),
                    mode=resolved_mode,
                )

                await self._emit(
                    "agent.loop_completed",
                    {
                        "agent_id": agent_id,
                        "status": self._enum_value(
                            loop_output.status
                        ),
                        "findings": len(loop_output.findings),
                        "ranked": len(
                            loop_output.ranked_sources
                        ),
                        "iterations": loop_output.total_iterations,
                        "tokens_used": loop_output.total_tokens_used,
                        "tokens_remaining": (
                            loop_output.tokens_remaining
                        ),
                        "errors_count": len(loop_output.errors),
                        "warnings_count": len(
                            loop_output.warnings
                        ),
                    },
                )
            except Exception as exc:
                errors.append(f"agent_loop_failed: {exc}")
                self._trace.emit(
                    "agent_loop_failed",
                    agent_id=agent_id,
                    error=str(exc),
                )

                await self._emit(
                    "agent.stage_failed",
                    {
                        "agent_id": agent_id,
                        "stage": "agent_loop",
                        "error": str(exc),
                    },
                )

            findings = (
                loop_output.findings if loop_output else []
            )
            ranked_sources = (
                loop_output.ranked_sources if loop_output else []
            )
            total_iterations = (
                loop_output.total_iterations if loop_output else 0
            )

            tracker_tokens = self._cost_tracker_used(cost_tracker)
            loop_tokens = (
                loop_output.total_tokens_used
                if loop_output
                else 0
            )
            total_tokens_used = max(tracker_tokens, loop_tokens)
            tokens_remaining = max(
                0,
                int(
                    mode_config.get(
                        "budget", self._config.token_budget
                    )
                )
                - total_tokens_used,
            )

            if loop_output:
                errors.extend(loop_output.errors)
                warnings.extend(loop_output.warnings)

            if not ranked_sources and findings:
                ranked_sources = await self._fallback_rank(
                    findings,
                    query_schema,
                    effective_max_results,
                )

            source_summaries, learning_path = (
                await self._summarize_and_build_path(
                    ranked_sources,
                    query_schema,
                    resolved_mode,
                    warnings,
                )
            )

            await self._emit(
                "agent.summaries_completed",
                {
                    "agent_id": agent_id,
                    "total_summaries": len(source_summaries),
                },
            )

            await self._emit(
                "agent.learning_path_completed",
                {
                    "agent_id": agent_id,
                    "total_steps": self._count_learning_steps(
                        learning_path
                    ),
                },
            )

            enhanced_learning_path = None

            if resolved_mode == constants.MODE_DEEP:
                enhanced_learning_path = await self._enhance_path(
                    learning_path,
                    ranked_sources,
                    query_schema,
                    resolved_mode,
                    warnings,
                )

                if enhanced_learning_path:
                    await self._emit(
                        "agent.path_enhanced",
                        {"agent_id": agent_id},
                    )

            rag_documents_indexed = 0
            settings = get_settings()
            base_enable_rag = bool(
                getattr(self._config, "enable_rag", False)
                or getattr(settings, "agent_enable_rag", False)
            )
            mode_enable_rag = bool(
                mode_config.get("enable_rag", False)
            )

            if base_enable_rag and mode_enable_rag:
                rag_documents_indexed = await self._index_rag(
                    ranked_sources,
                    agent_id,
                    warnings,
                )
            elif base_enable_rag and not mode_enable_rag:
                self._trace.emit(
                    "agent_rag_skipped_by_mode",
                    agent_id=agent_id,
                    mode=resolved_mode,
                )

            tracker_tokens_final = self._cost_tracker_used(
                cost_tracker
            )
            total_tokens_used = max(
                total_tokens_used, tracker_tokens_final
            )
            tokens_remaining = max(
                0,
                int(
                    mode_config.get(
                        "budget", self._config.token_budget
                    )
                )
                - total_tokens_used,
            )

            status = self._determine_status(
                ranked_sources=ranked_sources,
                findings=findings,
                errors=errors,
                learning_path=learning_path,
            )

            finished_at = datetime.now(timezone.utc)
            latency_ms = (
                time.perf_counter() - agent_start
            ) * 1000

            result = AgentResult(
                agent_id=agent_id,
                status=status,
                goal=goal,
                level=level,
                topic=resolved_topic,
                mode=resolved_mode,
                findings=findings,
                ranked_sources=ranked_sources,
                source_summaries=source_summaries,
                learning_path=learning_path,
                enhanced_learning_path=enhanced_learning_path,
                actions_taken=(
                    loop_output.actions_taken
                    if loop_output
                    else []
                ),
                total_iterations=total_iterations,
                total_tokens_used=total_tokens_used,
                total_budget=int(
                    mode_config.get(
                        "budget", self._config.token_budget
                    )
                ),
                tokens_remaining=tokens_remaining,
                iteration_count=total_iterations,
                max_iterations=int(
                    mode_config.get(
                        "iterations", self._config.max_iterations
                    )
                ),
                final_output=loop_output,
                errors=errors,
                warnings=warnings,
                saved_files={},
                rag_documents_indexed=rag_documents_indexed,
                started_at=run_started_at,
                finished_at=finished_at,
                latency_ms=latency_ms,
            )

            saved_files = self._save_outputs(
                result, resolved_topic or goal
            )
            self._assign_saved_files(result, saved_files)

            await self._save_agent_run(result, warnings)

            if status == "failed":
                await self._emit(
                    "agent.failed",
                    {
                        "agent_id": agent_id,
                        "error": (
                            "; ".join(errors[:5])
                            or "Agent failed"
                        ),
                        "status": status,
                    },
                )
            else:
                await self._emit(
                    "agent.completed",
                    {
                        "agent_id": agent_id,
                        "status": status,
                        "mode": resolved_mode,
                        "findings_count": len(findings),
                        "ranked_count": len(ranked_sources),
                        "iterations": total_iterations,
                        "tokens_used": total_tokens_used,
                        "tokens_remaining": tokens_remaining,
                        "errors_count": len(errors),
                        "warnings_count": len(warnings),
                        "rag_documents_indexed": (
                            rag_documents_indexed
                        ),
                    },
                )

            return result
        finally:
            try:
                manager.unregister_token_hook(
                    cost_tracker.on_token_usage
                )
            except Exception:
                pass

            try:
                await LLMProviderManager.close_shared()
            except Exception as exc:
                self._logger.warning(
                    f"Failed to close shared LLM manager: {exc}"
                )

    def _resolve_mode(self, mode: str) -> str:
        text = str(mode or "").strip().lower()

        if text in constants.VALID_MODES:
            return text

        config_mode = str(
            getattr(self._config, "mode", "") or ""
        ).strip().lower()

        if config_mode in constants.VALID_MODES:
            return config_mode

        return constants.DEFAULT_MODE

    def _resolve_max_results(
        self,
        requested: int,
        mode_config: dict[str, Any],
    ) -> int:
        if requested and int(requested) > 0:
            return int(requested)

        try:
            return max(
                1, int(mode_config.get("top_n", 10))
            )
        except Exception:
            return 10

    async def _summarize_and_build_path(
        self,
        ranked_sources: list[RankedSource],
        query_schema: SearchQuerySchema,
        mode: str,
        warnings: list[str],
    ) -> tuple[list[Any], Any]:
        if not ranked_sources:
            return [], None

        summarize_task = asyncio.create_task(
            self._generate_summaries(
                ranked_sources, query_schema, mode, warnings
            )
        )
        path_task = asyncio.create_task(
            self._generate_learning_path(
                ranked_sources, query_schema, mode, warnings
            )
        )

        summaries, learning_path = await asyncio.gather(
            summarize_task,
            path_task,
            return_exceptions=True,
        )

        if isinstance(summaries, BaseException):
            warnings.append(
                f"source_summaries_failed: {summaries}"
            )
            summaries = self._fallback_summaries(
                [
                    ranked.source
                    for ranked in ranked_sources
                    if getattr(ranked, "source", None) is not None
                ]
            )

        if isinstance(learning_path, BaseException):
            warnings.append(
                f"learning_path_failed: {learning_path}"
            )
            learning_path = None

        return list(summaries or []), learning_path

    async def _register_all_tools(self) -> None:
        return None

    def _register_all_tools(self) -> None:
        registrations = [
            ("agent.tools.search_tools", "register_search_tools"),
            ("agent.tools.read_tools", "register_read_tools"),
            ("agent.tools.analysis_tools", "register_analysis_tools"),
            ("agent.tools.gen_tools", "register_gen_tools"),
            ("agent.tools.memory_tools", "register_memory_tools"),
        ]

        self._skipped_tool_modules = []

        for module_name, function_name in registrations:
            try:
                module = importlib.import_module(module_name)
                function = getattr(module, function_name, None)

                if callable(function):
                    function(self._registry)
                else:
                    self._skipped_tool_modules.append(
                        f"{module_name}: {function_name} not found"
                    )
            except Exception as exc:
                self._skipped_tool_modules.append(
                    f"{module_name}: {exc}"
                )

        if self._registry.tool_count <= 0:
            self._logger.warning("No agent tools were registered")

    async def _recall_memory(
        self, goal: str, topic: str
    ) -> str:
        if not self._config.memory_enabled:
            return ""

        query = clean_text(topic or goal)

        if not query:
            return ""

        for method_name in (
            "build_task_context",
            "retrieve_context",
            "retrieve_text",
            "retrieve",
            "search",
        ):
            method = getattr(
                self._memory_retrieval, method_name, None
            )

            if not callable(method):
                continue

            for parameters in self._memory_call_candidates(
                method_name, query
            ):
                supported_parameters = {
                    key: value
                    for key, value in parameters.items()
                    if self._parameter_supported(method, key)
                }

                try:
                    result = await self._maybe_await(
                        method(**supported_parameters)
                    )
                    text = self._memory_result_to_text(result)

                    if text:
                        return text
                except TypeError:
                    continue
                except Exception as exc:
                    self._logger.warning(
                        f"Memory recall failed: {exc}"
                    )
                    continue

        return ""

    def _memory_call_candidates(
        self,
        method_name: str,
        query: str,
    ) -> list[dict[str, Any]]:
        if method_name == "build_task_context":
            return [
                {"goal": query},
                {"goal": query, "level": ""},
                {"goal": query, "level": "", "task_type": ""},
            ]

        if method_name in ("retrieve_context", "retrieve_text"):
            return [
                {"goal": query},
                {"goal": query, "level": ""},
                {"goal": query, "level": "", "task_type": ""},
                {"query": query},
            ]

        if method_name == "retrieve":
            return [
                {"query": query},
                {"goal": query},
                {"query": query, "limit": None},
                {"query": query, "top_k": None},
                {"query": query, "limit": None, "top_k": None},
            ]

        return [
            {"query": query},
            {"query": query, "limit": None},
            {"query": query, "top_k": None},
            {"query": query, "limit": None, "top_k": None},
        ]

    def _parameter_supported(
        self, method: Any, name: str
    ) -> bool:
        try:
            signature = inspect.signature(method)
        except Exception:
            return True

        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            return True

        return name in signature.parameters

    def _memory_result_to_text(self, result: Any) -> str:
        if result is None:
            return ""

        if isinstance(result, str):
            return clean_text(result)

        if isinstance(result, dict):
            for key in (
                "context_text",
                "content",
                "text",
                "summary",
                "answer",
            ):
                text = clean_text(result.get(key, ""))

                if text:
                    return text

            nested = (
                result.get("results")
                or result.get("entries")
                or result.get("episodes")
                or result.get("memories")
            )

            if nested is not None:
                return self._memory_result_to_text(nested)

            try:
                return clean_text(
                    json.dumps(
                        result, ensure_ascii=False, default=str
                    )
                )
            except Exception:
                return ""

        if isinstance(result, (list, tuple, set)):
            parts: list[str] = []

            for item in result:
                text = self._memory_result_to_text(item)

                if text:
                    parts.append(text)

            return "\n".join(parts)

        for attribute_name in (
            "context_text",
            "content",
            "text",
            "summary",
            "answer",
        ):
            value = getattr(result, attribute_name, None)
            text = clean_text(value)

            if text:
                return text

        results = getattr(result, "results", None)

        if results is not None:
            return self._memory_result_to_text(results)

        return clean_text(result)

    async def _create_plan(
        self,
        goal: str,
        level: str,
        topic: str,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        try:
            agent_input = AgentInputSchema(
                goal=goal,
                level=level,
                topic=topic,
                max_iterations=self._config.max_iterations,
                token_budget=self._config.token_budget,
                use_memory=self._config.memory_enabled,
                use_reflection=self._config.reflection_enabled,
                max_retries_per_step=(
                    self._config.max_retries_per_step
                ),
                quality_threshold=self._config.quality_threshold,
            )

            plan_result = await self._maybe_await(
                self._planner.plan(agent_input)
            )
            plan = getattr(plan_result, "plan", None)

            if plan is not None:
                await self._emit(
                    "agent.plan_created",
                    {
                        "goal": goal,
                        "topic": topic,
                        "total_steps": len(
                            getattr(plan, "steps", []) or []
                        ),
                    },
                )
                return

            warnings.append("plan_failed: planner returned no plan")
        except Exception as exc:
            warnings.append(f"plan_failed: {exc}")

    async def _fallback_rank(
        self,
        findings: list[Source],
        query_schema: SearchQuerySchema,
        max_results: int,
    ) -> list[RankedSource]:
        if not findings:
            return []

        try:
            scorer = HeuristicScorer()

            scored: list[tuple[float, Source, float]] = []

            for source in findings:
                try:
                    score = float(
                        scorer.score_source(
                            source, query=query_schema
                        )
                    )
                except Exception:
                    score = 0.5

                try:
                    relevance = float(
                        scorer.relevance_factor(
                            source, query_schema
                        )
                    )
                except Exception:
                    relevance = 0.0

                scored.append((score, source, relevance))

            relevant_scores = [item[2] for item in scored]
            best_relevance = max(relevant_scores, default=0.0)
            effective_floor = max(best_relevance * 0.35, 0.05)

            gated = [
                item for item in scored
                if item[2] >= effective_floor
            ]

            if len(gated) < 3:
                ranked_by_relevance = sorted(
                    scored, key=lambda item: item[2], reverse=True
                )
                gated = ranked_by_relevance[:3]

            gated.sort(key=lambda item: item[0], reverse=True)

            limit = max(1, int(max_results))

            ranked: list[RankedSource] = []

            for rank, (score, source, relevance) in enumerate(
                gated[:limit], start=1
            ):
                confidence = max(0.30, min(0.95, relevance * 1.15))

                ranked.append(
                    RankedSource(
                        source=source,
                        rank=rank,
                        score=score,
                        confidence=confidence,
                        reason="heuristic_fallback",
                    )
                )

            return ranked
        except Exception:
            fallback: list[RankedSource] = []

            for rank, source in enumerate(
                findings[: max(1, int(max_results))], start=1
            ):
                fallback.append(
                    RankedSource(
                        source=source,
                        rank=rank,
                        score=0.5,
                        confidence=0.3,
                        reason="unranked_fallback",
                    )
                )

            return fallback

    def _call_with_supported_args(
        self,
        method: Any,
        base_args: list[Any],
        candidates: list[dict[str, Any]] | None = None,
    ) -> Any:
        try:
            signature = inspect.signature(method)
        except Exception:
            return method(*base_args)

        params = list(signature.parameters.values())

        accepts_var_keyword = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in params
        )

        positional_params = [
            parameter
            for parameter in params
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]

        keyword_names = {
            parameter.name
            for parameter in params
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }

        max_positional = len(positional_params)
        attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        for candidate in candidates or []:
            if accepts_var_keyword:
                attempts.append(
                    (tuple(base_args), dict(candidate))
                )

            call_args = list(base_args)

            for parameter in positional_params[len(base_args):]:
                if parameter.name in candidate:
                    call_args.append(candidate[parameter.name])

            filled_positional_names = {
                positional_params[index].name
                for index in range(
                    min(len(call_args), len(positional_params))
                )
            }

            call_kwargs = {
                key: value
                for key, value in candidate.items()
                if key in keyword_names
                and key not in filled_positional_names
            }

            attempts.append(
                (tuple(call_args[:max_positional]), call_kwargs)
            )

        attempts.append(
            (tuple(base_args[:max_positional]), {})
        )

        seen = set()
        last_error = None

        for call_args, call_kwargs in attempts:
            attempt_key = (
                len(call_args),
                tuple(sorted(call_kwargs)),
            )

            if attempt_key in seen:
                continue

            seen.add(attempt_key)

            try:
                return method(*call_args, **call_kwargs)
            except TypeError as exc:
                last_error = exc
                continue

        if last_error is not None:
            raise last_error

        return method(*base_args[:max_positional])

    async def _generate_summaries(
        self,
        ranked_sources: list[RankedSource],
        query_schema: SearchQuerySchema,
        mode: str,
        warnings: list[str],
    ) -> list[Any]:
        if not ranked_sources:
            return []

        sources = [
            ranked.source
            for ranked in ranked_sources
            if getattr(ranked, "source", None) is not None
        ]

        if not sources:
            return []

        if mode == constants.MODE_FAST:
            return self._fallback_summaries(sources)

        try:
            summarizer = self._get_source_summarizer()

            summarize_sources = getattr(
                summarizer, "summarize_sources", None
            )

            if callable(summarize_sources):
                result = await self._maybe_await(
                    self._call_with_supported_args(
                        summarize_sources,
                        [sources],
                        [
                            {
                                "query": query_schema,
                                "mode": mode,
                            },
                            {
                                "query_schema": query_schema,
                                "mode": mode,
                            },
                            {
                                "topic": getattr(
                                    query_schema, "topic", ""
                                ),
                                "mode": mode,
                            },
                            {
                                "goal": getattr(
                                    query_schema, "goal", ""
                                ),
                                "mode": mode,
                            },
                            {
                                "level": getattr(
                                    query_schema, "level", ""
                                ),
                                "mode": mode,
                            },
                            {"mode": mode},
                            {},
                        ],
                    )
                )

                if isinstance(result, list):
                    return result

                if (
                    isinstance(result, dict)
                    and isinstance(
                        result.get("summaries"), list
                    )
                ):
                    return result["summaries"]

                extracted = getattr(result, "summaries", None)

                if isinstance(extracted, list):
                    return extracted

            summarize_source = getattr(
                summarizer, "summarize_source", None
            )

            if callable(summarize_source):
                summaries: list[Any] = []

                for source in sources[:20]:
                    summary = await self._maybe_await(
                        self._call_with_supported_args(
                            summarize_source,
                            [source],
                            [
                                {"query": query_schema},
                                {"query_schema": query_schema},
                                {},
                            ],
                        )
                    )

                    if summary is not None:
                        summaries.append(summary)

                return summaries
        except Exception as exc:
            warnings.append(
                f"source_summaries_failed: {exc}; "
                f"using fallback summaries"
            )

        return self._fallback_summaries(sources)

    async def _generate_learning_path(
        self,
        ranked_sources: list[RankedSource],
        query_schema: SearchQuerySchema,
        mode: str,
        warnings: list[str],
    ) -> Any:
        if not ranked_sources:
            return None

        try:
            builder = LearningPathBuilder(
                use_llm=(mode != constants.MODE_FAST)
            )
            build = getattr(builder, "build", None)

            if callable(build):
                return await self._maybe_await(
                    self._call_with_supported_args(
                        build,
                        [ranked_sources, query_schema],
                        [{"mode": mode}, {}],
                    )
                )

            generate = getattr(builder, "generate", None)

            if callable(generate):
                return await self._maybe_await(
                    self._call_with_supported_args(
                        generate,
                        [ranked_sources, query_schema],
                        [{"mode": mode}, {}],
                    )
                )
        except Exception as exc:
            warnings.append(f"learning_path_failed: {exc}")

        return None

    async def _enhance_path(
        self,
        learning_path: Any,
        ranked_sources: list[RankedSource],
        query_schema: SearchQuerySchema,
        mode: str,
        warnings: list[str],
    ) -> Any:
        if learning_path is None:
            return None

        if mode == constants.MODE_FAST:
            return None

        try:
            enhancer = self._get_path_enhancer()

            enhance = getattr(enhancer, "enhance", None)

            if callable(enhance):
                return await self._maybe_await(
                    self._call_with_supported_args(
                        enhance,
                        [learning_path],
                        [
                            {
                                "ranked_sources": ranked_sources,
                                "query": query_schema,
                                "mode": mode,
                            },
                            {
                                "sources": ranked_sources,
                                "query": query_schema,
                                "mode": mode,
                            },
                            {
                                "ranked_sources": ranked_sources,
                                "query_schema": query_schema,
                                "mode": mode,
                            },
                            {
                                "sources": ranked_sources,
                                "query_schema": query_schema,
                                "mode": mode,
                            },
                            {"query": query_schema, "mode": mode},
                            {"query_schema": query_schema, "mode": mode},
                            {
                                "topic": getattr(
                                    query_schema, "topic", ""
                                ),
                                "mode": mode,
                            },
                            {"mode": mode},
                            {},
                        ],
                    )
                )

            enhance_path = getattr(enhancer, "enhance_path", None)

            if callable(enhance_path):
                return await self._maybe_await(
                    self._call_with_supported_args(
                        enhance_path,
                        [learning_path],
                        [
                            {
                                "ranked_sources": ranked_sources,
                                "query": query_schema,
                                "mode": mode,
                            },
                            {
                                "sources": ranked_sources,
                                "query": query_schema,
                                "mode": mode,
                            },
                            {"query": query_schema, "mode": mode},
                            {"mode": mode},
                            {},
                        ],
                    )
                )
        except Exception as exc:
            warnings.append(f"path_enhancement_failed: {exc}")

        return None

    async def _index_rag(
        self,
        ranked_sources: list[RankedSource],
        run_id: str,
        warnings: list[str],
    ) -> int:
        if not ranked_sources:
            return 0

        try:
            from rag.embedder import Embedder
            from rag.vector_store import VectorStore

            store = VectorStore()

            try:
                count = await self._index_ranked_sources(
                    ranked_sources,
                    run_id,
                    warnings,
                    store,
                )
            finally:
                try:
                    store.close()
                except Exception as close_exc:
                    self._logger.warning(
                        f"VectorStore close raised: {close_exc}"
                    )

            return count
        except Exception as exc:
            warnings.append(f"rag_indexing_failed: {exc}")
            return 0

    async def _index_ranked_sources(
        self,
        ranked_sources: list[RankedSource],
        run_id: str,
        warnings: list[str],
        store: Any,
    ) -> int:
        from rag.embedder import Embedder

        if not ranked_sources:
            return 0

        async with Embedder() as embedder:
            chunked_enabled = bool(
                embedder.chunked_index_enabled
            )
            chunk_config = dict(embedder.chunked_config or {})
            fallback_to_parent = bool(
                chunk_config.get("fallback_to_parent", True)
            )
            fail_open = bool(
                chunk_config.get("fail_open", True)
            )

            if not chunked_enabled:
                count = await self._index_parent_only(
                    embedder,
                    store,
                    ranked_sources,
                    run_id,
                )
                self._trace.emit(
                    "agent_rag_index_mode",
                    agent_id=run_id,
                    mode="parent",
                    documents_indexed=count,
                    chunks_indexed=0,
                )
                return count

            self._trace.emit(
                "agent_rag_chunk_attempt",
                agent_id=run_id,
                ranked_count=len(ranked_sources),
                chunk_tokens=chunk_config.get("chunk_tokens"),
                max_chunks_per_source=chunk_config.get(
                    "max_chunks_per_source"
                ),
            )

            try:
                count, chunk_count = await self._index_with_chunks(
                    embedder,
                    store,
                    ranked_sources,
                    run_id,
                    chunk_config,
                )

                if count > 0:
                    mode = (
                        "chunked"
                        if chunk_count > 0
                        else "parent"
                    )

                    self._trace.emit(
                        "agent_rag_index_mode",
                        agent_id=run_id,
                        mode=mode,
                        documents_indexed=count,
                        chunks_indexed=chunk_count,
                    )

                    return count

                self._trace.emit(
                    "agent_rag_chunk_empty",
                    agent_id=run_id,
                    documents_indexed=count,
                    chunks_indexed=chunk_count,
                )
            except Exception as exc:
                if not fail_open:
                    raise

                self._logger.warning(
                    f"Agent chunk indexing failed: {exc}"
                )
                warnings.append(f"chunk_indexing_failed: {exc}")

                self._trace.emit(
                    "agent_rag_chunk_failed",
                    agent_id=run_id,
                    error=str(exc),
                )

            if not fallback_to_parent:
                return 0

            self._trace.emit(
                "agent_rag_fallback_to_parent",
                agent_id=run_id,
            )

            count = await self._index_parent_only(
                embedder,
                store,
                ranked_sources,
                run_id,
            )

            self._trace.emit(
                "agent_rag_index_mode",
                agent_id=run_id,
                mode="fallback",
                documents_indexed=count,
                chunks_indexed=0,
            )

            return count

    async def _index_parent_only(
        self,
        embedder: Any,
        store: Any,
        ranked_sources: list[RankedSource],
        run_id: str,
    ) -> int:
        embeddings = await embedder.embed_ranked_sources(
            ranked_sources
        )

        add_ranked_sources = getattr(
            store, "add_ranked_sources", None
        )

        if not callable(add_ranked_sources):
            raise AttributeError(
                "VectorStore.add_ranked_sources is not available"
            )

        return int(
            add_ranked_sources(
                ranked_sources,
                embeddings,
                run_id=run_id,
            )
        )

    async def _index_with_chunks(
        self,
        embedder: Any,
        store: Any,
        ranked_sources: list[RankedSource],
        run_id: str,
        chunk_config: dict[str, Any],
    ) -> tuple[int, int]:
        build_document_text = getattr(
            store, "build_document_text", None
        )

        if not callable(build_document_text):
            raise AttributeError(
                "VectorStore.build_document_text is not available"
            )

        chunk_tokens = chunk_config.get("chunk_tokens")

        (
            parent_embeddings,
            chunk_embeddings,
            resolved_tokens,
        ) = await embedder.embed_ranked_sources_with_chunks(
            ranked_sources,
            chunk_tokens=chunk_tokens,
            text_builder=build_document_text,
        )

        if not parent_embeddings:
            return 0, 0

        has_chunk_embeddings = any(
            embedding
            for per_source in chunk_embeddings
            for embedding in per_source
        )

        if not has_chunk_embeddings:
            add_ranked_sources = getattr(
                store, "add_ranked_sources", None
            )

            if not callable(add_ranked_sources):
                raise AttributeError(
                    "VectorStore.add_ranked_sources is not available"
                )

            count = int(
                add_ranked_sources(
                    ranked_sources,
                    parent_embeddings,
                    run_id=run_id,
                )
            )
            return count, 0

        add_with_chunks = getattr(
            store, "add_ranked_sources_with_chunks", None
        )

        if not callable(add_with_chunks):
            raise AttributeError(
                "VectorStore.add_ranked_sources_with_chunks is not "
                "available"
            )

        count = int(
            add_with_chunks(
                ranked_sources,
                parent_embeddings,
                chunk_embeddings,
                run_id=run_id,
                chunk_tokens=resolved_tokens,
            )
        )

        chunk_count = sum(
            1
            for per_source in chunk_embeddings
            for embedding in per_source
            if embedding
        )

        return count, chunk_count

    def _save_outputs(
        self,
        result: AgentResult,
        topic: str,
    ) -> dict[str, str]:
        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%d_%H%M%S"
        )
        slug = self._slug(topic or result.goal or "agent_run")
        base_name = f"{slug}_{timestamp}"

        json_dir = self._output_directory / "json"
        markdown_dir = self._output_directory / "markdown"
        reports_dir = self._output_directory / "reports"

        json_dir.mkdir(parents=True, exist_ok=True)
        markdown_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        payload = self._result_payload(result)

        json_path = json_dir / f"{base_name}.json"
        markdown_path = markdown_dir / f"{base_name}.md"
        html_path = reports_dir / f"{base_name}.html"

        json_path.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

        markdown_path.write_text(
            self._render_markdown(payload),
            encoding="utf-8",
        )

        html_path.write_text(
            self._render_html(payload),
            encoding="utf-8",
        )

        return {
            "json": str(json_path),
            "markdown": str(markdown_path),
            "html": str(html_path),
            "sqlite": str(self._database_path),
        }

    async def _save_agent_run(
        self,
        result: AgentResult,
        warnings: list[str],
    ) -> None:
        try:
            save_agent_run = getattr(
                self._store, "save_agent_run", None
            )

            if callable(save_agent_run):
                await self._maybe_await(save_agent_run(result))
        except Exception as exc:
            warnings.append(f"agent_run_storage_failed: {exc}")

    def _result_payload(
        self, result: AgentResult
    ) -> dict[str, Any]:
        try:
            return result.model_dump(
                mode="json",
                exclude_none=False,
                by_alias=False,
            )
        except Exception:
            return {
                "agent_id": result.agent_id,
                "status": result.status,
                "goal": result.goal,
                "level": result.level,
                "topic": result.topic,
                "mode": result.mode,
                "total_iterations": result.total_iterations,
                "total_tokens_used": result.total_tokens_used,
                "total_budget": result.total_budget,
                "tokens_remaining": result.tokens_remaining,
                "errors": result.errors,
                "warnings": result.warnings,
                "saved_files": result.saved_files,
                "rag_documents_indexed": (
                    result.rag_documents_indexed
                ),
            }

    def _render_markdown(
        self, payload: dict[str, Any]
    ) -> str:
        try:
            from storage.markdown_writer import MarkdownWriter

            payload = dict(payload or {})
            payload.setdefault("kind", "agent")
            writer = MarkdownWriter.__new__(MarkdownWriter)
            writer._file_manager = None
            return writer.render_payload(payload)
        except Exception:
            lines = [
                f"# {payload.get('topic') or payload.get('goal') or 'Research Agent Run'}",
                "",
                f"- Agent ID: {payload.get('agent_id', '-')}",
                f"- Status: {str(payload.get('status', '-')).upper()}",
                f"- Goal: {payload.get('goal', '-')}",
                f"- Level: {payload.get('level', '-')}",
                f"- Mode: {payload.get('mode', '-')}",
                f"- Iterations: {payload.get('total_iterations', 0)}",
                f"- Tokens Used: {payload.get('total_tokens_used', 0)}",
                f"- Tokens Remaining: {payload.get('tokens_remaining', 0)}",
                f"- Budget: {payload.get('total_budget', 0)}",
                f"- RAG Documents: {payload.get('rag_documents_indexed', 0)}",
                "",
            ]
            return "\n".join(lines)

    def _render_html(
        self, payload: dict[str, Any]
    ) -> str:
        try:
            from storage.html_writer import HTMLWriter

            payload = dict(payload or {})
            payload.setdefault("kind", "agent")
            return HTMLWriter().render_payload(payload)
        except Exception:
            title = html.escape(
                str(
                    payload.get("topic")
                    or payload.get("goal")
                    or "Research Agent Run"
                )
            )
            body = html.escape(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str)
            )
            return (
                "<!doctype html>\n<html>\n<head>\n"
                '<meta charset="utf-8">\n'
                f"<title>{title}</title>\n"
                "<style>body{font-family:system-ui;margin:24px;"
                "background:#0a0e1a;color:#e8eefb}"
                "pre{white-space:pre-wrap;background:#121826;"
                "padding:16px;border-radius:8px}</style>\n"
                "</head>\n<body>\n"
                f"<h1>{title}</h1>\n<pre>{body}</pre>\n"
                "</body>\n</html>\n"
            )
    def _assign_saved_files(
        self,
        result: AgentResult,
        saved_files: dict[str, str],
    ) -> None:
        try:
            result.saved_files = saved_files
        except Exception:
            try:
                object.__setattr__(
                    result, "saved_files", saved_files
                )
            except Exception:
                pass

    def _determine_status(
        self,
        ranked_sources: list[RankedSource],
        findings: list[Source],
        errors: list[str],
        learning_path: Any,
    ) -> str:
        if not ranked_sources and not findings:
            return "failed"

        if errors:
            return "partial"

        if not learning_path:
            return "partial"

        return "success"

    def _cost_tracker_used(self, cost_tracker: Any) -> int:
        for attr in (
            "total_tokens_used",
            "tokens_used",
            "total_spent",
            "spent",
        ):
            value = getattr(cost_tracker, attr, None)

            if value is not None:
                try:
                    return int(value)
                except Exception:
                    continue

        return 0

    def _fallback_summaries(
        self,
        sources: list[Source],
    ) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []

        for source in sources[: constants.MAX_SOURCES_TO_SUMMARIZE]:
            title = str(
                getattr(source, "title", "") or "Untitled source"
            )
            abstract = str(
                getattr(source, "abstract", "") or ""
            ).strip()
            difficulty = (
                self._enum_value(
                    getattr(source, "difficulty", "")
                )
                or "unknown"
            )

            summaries.append(
                {
                    "source_id": str(
                        getattr(source, "source_id", "") or ""
                    ),
                    "title": title,
                    "url": str(
                        getattr(source, "url", "") or ""
                    ),
                    "difficulty": difficulty,
                    "summary": (
                        abstract[:700]
                        if abstract
                        else f"No abstract available for {title}."
                    ),
                    "why_useful": (
                        "Generated by agent fallback summarizer."
                    ),
                    "key_topics": [],
                }
            )

        return summaries

    def _count_learning_steps(
        self, learning_path: Any
    ) -> int:
        if learning_path is None:
            return 0

        steps = getattr(learning_path, "steps", None)

        if steps is None and isinstance(learning_path, dict):
            steps = learning_path.get("steps", [])

        return len(list(steps or []))

    async def _emit(
        self, name: str, payload: dict[str, Any]
    ) -> None:
        try:
            await self._event_bus.emit(name, payload)
        except Exception as exc:
            self._logger.warning(
                f"Failed to emit event {name}: {exc}"
            )

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value

        return value

    @staticmethod
    def _slug(value: str) -> str:
        cleaned = clean_text(value).lower()
        cleaned = "".join(
            ch
            if ch.isalnum() or ch in {" ", "-", "_"}
            else ""
            for ch in cleaned
        )
        cleaned = cleaned.replace(" ", "_")
        cleaned = "_".join(
            part for part in cleaned.split("_") if part
        )
        return (cleaned or "agent_run")[:80]

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value) or "")