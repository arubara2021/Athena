from __future__ import annotations

import asyncio
import html
import inspect
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from pydantic import Field

from core import constants
from core.config import (
    get_data_directory,
    get_output_directory,
)
from core.events import (
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
from core.models import LearningPath, RankedSource, Source, TaskStatus
from core.schemas import SearchQuerySchema
from core.state import PipelineState
from agent.loop.engine import get_mode_config
from ranking.consensus_ranker import ConsensusRanker
from ranking.learning_path_builder import LearningPathBuilder
from ranking.path_enhancer import PathEnhancer
from ranking.source_summarizer import SourceSummarizer
from search.orchestrator import SearchOrchestrator
from search.query_tokens import concept_tokens, get_generic_terms
from storage.file_manager import FileManager
from storage.html_writer import HTMLWriter
from storage.json_writer import JSONWriter
from storage.markdown_writer import MarkdownWriter
from storage.sqlite_store import SQLiteStore
from utils.logger import get_logger, get_trace_logger
from utils.text import clean_text


try:
    from core.models import CoreModel
except Exception:
    class CoreModel:
        pass


class PipelineResult(CoreModel):
    request_id: str
    status: TaskStatus
    query: Optional[SearchQuerySchema] = None
    mode: str = constants.DEFAULT_MODE
    corrected_topic: str = ""
    primary_concept: str = ""
    keywords: list[str] = Field(default_factory=list)
    expanded_queries: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    ranked_sources: list[RankedSource] = Field(default_factory=list)
    source_summaries: list[Any] = Field(default_factory=list)
    learning_path: Optional[LearningPath] = None
    enhanced_learning_path: Any = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    saved_files: dict[str, str] = Field(default_factory=dict)
    rag_documents_indexed: int = Field(default=0, ge=0)
    latency_ms: Optional[float] = None

    target_domains: list[str] = Field(default_factory=list)
    target_formats: list[str] = Field(default_factory=list)
    domain_confidence: float = 0.0
    detected_level: str = ""
    domain_routed: bool = False
    domain_expanded: bool = False
    domain_platforms: list[str] = Field(default_factory=list)
    source_selection_reasons: list[Any] = Field(default_factory=list)
    quality_guards: dict[str, Any] = Field(default_factory=dict)
    quality_gate_passed: bool = True


class ResearchPipeline:
    def __init__(
        self,
        search_orchestrator: SearchOrchestrator,
        consensus_ranker: ConsensusRanker,
        learning_path_builder: LearningPathBuilder,
        enable_rag: bool = False,
        event_bus: Optional[Any] = None,
        html_writer: Optional[HTMLWriter] = None,
        database_path: Optional[str] = None,
    ) -> None:
        self._search_orchestrator = search_orchestrator
        self._consensus_ranker = consensus_ranker
        self._learning_path_builder = learning_path_builder
        self._enable_rag = bool(enable_rag)

        try:
            from core.events import EventBus

            self._event_bus = event_bus or EventBus()
        except Exception:
            self._event_bus = event_bus

        self._logger = get_logger("core.pipeline")
        self._trace = get_trace_logger()

        output_directory = get_output_directory()
        self._file_manager = FileManager(output_directory)
        self._json_writer = JSONWriter(self._file_manager)
        self._markdown_writer = MarkdownWriter(self._file_manager)
        self._html_writer = html_writer or HTMLWriter(
            self._file_manager
        )

        resolved_database_path = (
            database_path
            or str(get_data_directory() / "research_agent.db")
        )
        self._sqlite_store = SQLiteStore(resolved_database_path)

        self._source_summarizer: Optional[SourceSummarizer] = None
        self._path_enhancer: Optional[PathEnhancer] = None

        self._education_platforms = {
            "openstax",
            "libretexts",
            "mit_ocw",
            "wikiversity",
            "wikibooks",
            "wikipedia",
            "open_library",
            "internet_archive",
        }

        self._educational_source_types = {
            "course",
            "book",
            "documentation",
            "video",
        }

    async def __aenter__(self) -> "ResearchPipeline":
        await self.initialize()
        return self

    async def __aexit__(
        self, exc_type: Any, exc: Any, tb: Any
    ) -> bool:
        await self.close()
        return False

    async def initialize(self) -> None:
        await self._sqlite_store.initialize()

    async def close(self) -> None:
        close = getattr(
            self._search_orchestrator, "close", None
        )

        if callable(close):
            await self._maybe_await(close())

    def subscribe(self, event_name: str, handler: Any) -> None:
        if self._event_bus is None:
            return

        subscribe = getattr(self._event_bus, "subscribe", None)

        if callable(subscribe):
            subscribe(event_name, handler)

    def _empty_meta(self) -> dict[str, Any]:
        return {
            "target_domains": [],
            "target_formats": [],
            "domain_confidence": 0.0,
            "detected_level": "",
            "domain_routed": False,
            "domain_expanded": False,
            "domain_platforms": [],
            "source_selection_reasons": [],
            "quality_guards": {},
            "quality_gate_passed": True,
        }

    async def run(
        self,
        query: SearchQuerySchema | str | dict[str, Any],
        mode: str = "",
    ) -> PipelineResult:
        state = PipelineState()
        pipeline_start = time.perf_counter()
        meta = self._empty_meta()

        resolved_mode = self._resolve_mode(mode)
        mode_config = get_mode_config(resolved_mode)

        try:
            query_schema = self._coerce_query(query)
        except Exception as exc:
            state.mark_failed(f"invalid_query: {exc}")
            state.latency_ms = (
                time.perf_counter() - pipeline_start
            ) * 1000
            await self._emit(
                PIPELINE_FAILED,
                {
                    "request_id": state.request_id,
                    "error": str(exc),
                    "errors": state.errors,
                },
            )
            return self._build_result(state, {}, meta, resolved_mode)

        query_schema = self._apply_mode_to_query(
            query_schema, mode_config
        )

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
            mode=resolved_mode,
        )

        await self._emit(
            PIPELINE_STARTED,
            {
                "request_id": state.request_id,
                "topic": query_schema.topic,
                "goal": query_schema.goal,
                "level": query_schema.level,
                "max_results": query_schema.max_results,
                "enable_rag": self._enable_rag,
                "mode": resolved_mode,
            },
        )

        try:
            search_meta = await self._run_search(state)
            meta.update(search_meta)

            if state.sources:
                await self._run_ranking(
                    state, resolved_mode, mode_config
                )
            else:
                has_no_source_reason = any(
                    str(error).startswith("no_sources_reason")
                    for error in state.errors
                )

                if not has_no_source_reason:
                    state.add_error("no_sources_found")

                self._trace.emit(
                    "pipeline_no_sources",
                    request_id=state.request_id,
                )

            if state.ranked_sources:
                meta["source_selection_reasons"] = (
                    self._source_selection_reasons(state, meta)
                )
                guard_report = self._evaluate_quality_guards(
                    state, meta
                )
                meta["quality_guards"] = guard_report
                meta["quality_gate_passed"] = bool(
                    guard_report.get("passed", False)
                )

                if guard_report.get("critical_failures"):
                    for failure in guard_report.get(
                        "critical_failures", []
                    ):
                        state.add_error(
                            f"quality_guard_failed: {failure}"
                        )

                    state.sources = []
                    state.ranked_sources = []
                    state.source_summaries = []
                    state.learning_path = None
                    state.enhanced_learning_path = None

                    state.mark_failed(
                        "; ".join(
                            guard_report.get(
                                "critical_failures", []
                            )
                        )
                    )
                    state.latency_ms = (
                        time.perf_counter() - pipeline_start
                    ) * 1000

                    await self._emit(
                        PIPELINE_FAILED,
                        {
                            "request_id": state.request_id,
                            "status": self._enum_value(
                                state.status
                            ),
                            "errors": state.errors,
                            "warnings": state.warnings,
                            "quality_guards": guard_report,
                            "total_sources": 0,
                            "total_ranked": 0,
                            "latency_ms": round(
                                state.latency_ms, 1
                            ),
                        },
                    )

                    saved_files = await self._save_outputs(
                        state, meta, resolved_mode
                    )
                    return self._build_result(
                        state,
                        saved_files,
                        meta,
                        resolved_mode,
                    )

                await self._summarize_and_build_path(
                    state, resolved_mode
                )

                if resolved_mode == constants.MODE_DEEP:
                    await self._run_path_enhancement(
                        state, resolved_mode
                    )

                meta["source_selection_reasons"] = (
                    self._source_selection_reasons(state, meta)
                )

            if self._enable_rag:
                await self._run_rag_indexing(state)
            else:
                self._trace.emit(
                    "pipeline_rag_not_requested",
                    request_id=state.request_id,
                    mode=resolved_mode,
                )

            total_ms = (
                time.perf_counter() - pipeline_start
            ) * 1000
            self._set_attr(state, "latency_ms", total_ms)

            guard_report = self._evaluate_quality_guards(
                state, meta
            )
            meta["quality_guards"] = guard_report
            meta["quality_gate_passed"] = bool(
                guard_report.get("passed", False)
            )

            if guard_report.get("critical_failures"):
                for failure in guard_report.get(
                    "critical_failures", []
                ):
                    state.add_error(
                        f"quality_guard_failed: {failure}"
                    )

                state.sources = []
                state.ranked_sources = []
                state.source_summaries = []
                state.learning_path = None
                state.enhanced_learning_path = None

                state.mark_failed(
                    "; ".join(
                        guard_report.get("critical_failures", [])
                    )
                )

                await self._emit(
                    PIPELINE_FAILED,
                    {
                        "request_id": state.request_id,
                        "status": self._enum_value(state.status),
                        "errors": state.errors,
                        "warnings": state.warnings,
                        "quality_guards": guard_report,
                        "total_sources": 0,
                        "total_ranked": 0,
                        "latency_ms": round(total_ms, 1),
                    },
                )

                saved_files = await self._save_outputs(
                    state, meta, resolved_mode
                )
                return self._build_result(
                    state, saved_files, meta, resolved_mode
                )

            self._finalize_status(state)

            if self._enum_value(state.status) == "failed":
                self._trace.emit(
                    "pipeline_failed",
                    request_id=state.request_id,
                    errors=state.errors,
                    total_latency_ms=round(total_ms, 1),
                )
                await self._emit(
                    PIPELINE_FAILED,
                    {
                        "request_id": state.request_id,
                        "status": self._enum_value(state.status),
                        "errors": state.errors,
                        "warnings": state.warnings,
                        "total_sources": len(state.sources),
                        "total_ranked": len(
                            state.ranked_sources
                        ),
                        "latency_ms": round(total_ms, 1),
                    },
                )
                saved_files = await self._save_outputs(
                    state, meta, resolved_mode
                )
                await self._save_memory(state)
                return self._build_result(
                    state, saved_files, meta, resolved_mode
                )

            self._trace.emit(
                "pipeline_completed",
                request_id=state.request_id,
                status=self._enum_value(state.status),
                total_sources=len(state.sources),
                total_ranked=len(state.ranked_sources),
                total_summaries=len(state.source_summaries),
                rag_indexed=getattr(
                    state, "rag_documents_indexed", 0
                ),
                errors=state.errors,
                warnings=state.warnings,
                total_latency_ms=round(total_ms, 1),
                mode=resolved_mode,
            )

            await self._emit(
                PIPELINE_COMPLETED,
                {
                    "request_id": state.request_id,
                    "status": self._enum_value(state.status),
                    "total_sources": len(state.sources),
                    "total_ranked": len(state.ranked_sources),
                    "total_summaries": len(
                        state.source_summaries
                    ),
                    "total_steps": self._count_learning_steps(
                        state.learning_path
                    ),
                    "rag_documents_indexed": getattr(
                        state, "rag_documents_indexed", 0
                    ),
                    "errors_count": len(state.errors),
                    "warnings_count": len(state.warnings),
                    "latency_ms": round(total_ms, 1),
                    "target_domains": meta.get(
                        "target_domains", []
                    ),
                    "detected_level": meta.get(
                        "detected_level", ""
                    ),
                    "domain_confidence": meta.get(
                        "domain_confidence", 0.0
                    ),
                    "quality_gate_passed": meta.get(
                        "quality_gate_passed", True
                    ),
                    "mode": resolved_mode,
                },
            )

            saved_files = await self._save_outputs(
                state, meta, resolved_mode
            )
            await self._save_memory(state)
            return self._build_result(
                state, saved_files, meta, resolved_mode
            )

        except Exception as exc:
            self._logger.error(f"Pipeline failed: {exc}")
            state.mark_failed(str(exc))
            state.latency_ms = (
                time.perf_counter() - pipeline_start
            ) * 1000

            await self._emit(
                PIPELINE_FAILED,
                {
                    "request_id": state.request_id,
                    "error": str(exc),
                    "errors": state.errors,
                    "warnings": state.warnings,
                },
            )

            return self._build_result(
                state, {}, meta, resolved_mode
            )

    def _resolve_mode(self, mode: str) -> str:
        text = str(mode or "").strip().lower()

        if text in constants.VALID_MODES:
            return text

        return constants.DEFAULT_MODE

    def _apply_mode_to_query(
        self,
        query_schema: SearchQuerySchema,
        mode_config: dict[str, Any],
    ) -> SearchQuerySchema:
        try:
            current = int(query_schema.max_results or 0)
        except Exception:
            current = 0

        try:
            minimum = int(constants.MIN_MAX_RESULTS)
        except Exception:
            minimum = 1

        try:
            maximum = int(constants.MAX_MAX_RESULTS)
        except Exception:
            maximum = 100

        clamped = max(minimum, min(maximum, current or minimum))

        if clamped == current:
            return query_schema

        try:
            return query_schema.model_copy(
                update={"max_results": clamped}
            )
        except Exception:
            return query_schema

    async def _run_search(
        self, state: PipelineState
    ) -> dict[str, Any]:
        meta = self._empty_meta()

        try:
            raw = await self._maybe_await(
                self._search_orchestrator.search(state.query)
            )

            enriched_query = self._get_field(raw, "query", None)

            if isinstance(enriched_query, SearchQuerySchema):
                state.query = enriched_query

            sources = self._as_list(
                self._get_field(raw, "sources", [])
            )
            state.sources = [
                source
                for source in sources
                if source is not None
            ]

            state.expanded_queries = self._clean_list_strings(
                self._get_field(raw, "expanded_queries", [])
            )
            state.corrected_topic = str(
                self._get_field(raw, "corrected_topic", "") or ""
            )
            state.primary_concept = str(
                self._get_field(raw, "primary_concept", "") or ""
            )
            state.keywords = self._clean_list_strings(
                self._get_field(raw, "keywords", [])
            )

            meta["target_domains"] = self._clean_list_strings(
                self._get_field(raw, "target_domains", [])
            )
            meta["target_formats"] = self._clean_list_strings(
                self._get_field(raw, "target_formats", [])
            )
            meta["domain_platforms"] = self._clean_list_strings(
                self._get_field(raw, "domain_platforms", [])
            )
            meta["domain_routed"] = bool(
                self._get_field(raw, "domain_routed", False)
            )
            meta["domain_expanded"] = bool(
                self._get_field(raw, "domain_expanded", False)
            )

            try:
                meta["domain_confidence"] = float(
                    self._get_field(raw, "domain_confidence", 0.0)
                    or 0.0
                )
            except Exception:
                meta["domain_confidence"] = 0.0

            meta["detected_level"] = str(
                self._get_field(raw, "detected_level", "")
                or self._get_field(raw, "level", "")
                or ""
            )

            if not meta["detected_level"]:
                meta["detected_level"] = self._infer_level(
                    state.query
                )

            if not meta["target_domains"]:
                meta["target_domains"] = (
                    self._detect_domains_fallback(state.query)
                )

            if not meta["target_formats"]:
                meta["target_formats"] = (
                    self._detect_formats_fallback(
                        state.query, meta["detected_level"]
                    )
                )

            if (
                meta["domain_confidence"] <= 0.0
                and meta["target_domains"]
            ):
                meta["domain_confidence"] = 0.35

            if state.query is not None and (
                meta["target_domains"]
                or meta["target_formats"]
                or meta["detected_level"]
            ):
                update: dict[str, Any] = {}

                if meta["target_domains"]:
                    update["target_domains"] = list(
                        meta["target_domains"]
                    )

                if meta["target_formats"]:
                    update["target_formats"] = list(
                        meta["target_formats"]
                    )

                if meta["detected_level"]:
                    update["detected_level"] = str(
                        meta["detected_level"]
                    )

                try:
                    state.query = state.query.model_copy(
                        update=update
                    )
                except Exception:
                    pass

            await self._emit(
                SEARCH_COMPLETED,
                {
                    "request_id": state.request_id,
                    "total_sources": len(state.sources),
                    "sources": len(state.sources),
                    "expanded_queries": state.expanded_queries,
                    "corrected_topic": state.corrected_topic,
                    "primary_concept": state.primary_concept,
                    "target_domains": meta["target_domains"],
                    "target_formats": meta["target_formats"],
                    "domain_confidence": meta["domain_confidence"],
                    "detected_level": meta["detected_level"],
                },
            )

        except Exception as exc:
            state.add_error(f"search_failed: {exc}")
            await self._emit(
                STAGE_FAILED,
                {
                    "request_id": state.request_id,
                    "stage": "search",
                    "error": str(exc),
                },
            )

        return meta

    async def _run_ranking(
        self,
        state: PipelineState,
        mode: str,
        mode_config: dict[str, Any],
    ) -> None:
        skip_ranking_llm = bool(
            mode_config.get("skip_ranking_llm", False)
        )

        if skip_ranking_llm:
            state.ranked_sources = self._fallback_rank(
                state.sources, state.query
            )
            await self._emit(
                RANKING_COMPLETED,
                {
                    "request_id": state.request_id,
                    "total_ranked": len(state.ranked_sources),
                    "ranked": len(state.ranked_sources),
                    "heuristic_only": True,
                    "mode": mode,
                },
            )
            return

        try:
            rank = self._consensus_ranker.rank

            try:
                signature = inspect.signature(rank)
                accepts_mode = "mode" in signature.parameters
            except Exception:
                accepts_mode = False

            if accepts_mode:
                raw = await self._maybe_await(
                    rank(
                        state.sources,
                        state.query,
                        mode=mode,
                    )
                )
            else:
                raw = await self._maybe_await(
                    rank(state.sources, state.query)
                )

            ranked = self._normalize_ranked(raw)

            if not ranked:
                raise ValueError("ranking returned no sources")

            state.ranked_sources = ranked

            await self._emit(
                RANKING_COMPLETED,
                {
                    "request_id": state.request_id,
                    "total_ranked": len(state.ranked_sources),
                    "ranked": len(state.ranked_sources),
                    "mode": mode,
                },
            )
        except Exception as exc:
            state.add_warning(
                f"ranking_failed: {exc}; using fallback ranking"
            )
            state.ranked_sources = self._fallback_rank(
                state.sources, state.query
            )
            await self._emit(
                RANKING_COMPLETED,
                {
                    "request_id": state.request_id,
                    "total_ranked": len(state.ranked_sources),
                    "ranked": len(state.ranked_sources),
                    "fallback": True,
                },
            )

    async def _summarize_and_build_path(
        self,
        state: PipelineState,
        mode: str,
    ) -> None:
        summarize_task = asyncio.create_task(
            self._run_source_summaries(state, mode)
        )
        path_task = asyncio.create_task(
            self._run_learning_path(state, mode)
        )

        summaries_result, path_result = await asyncio.gather(
            summarize_task,
            path_task,
            return_exceptions=True,
        )

        if isinstance(summaries_result, BaseException):
            state.add_warning(
                f"source_summaries_failed: {summaries_result}"
            )

        if isinstance(path_result, BaseException):
            state.add_warning(
                f"learning_path_failed: {path_result}"
            )

    def _source_selection_reasons(
        self,
        state: PipelineState,
        meta: dict[str, Any],
    ) -> list[dict[str, Any]]:
        reasons: list[dict[str, Any]] = []

        ranked_sources = list(
            getattr(state, "ranked_sources", []) or []
        )

        if not ranked_sources:
            return reasons

        target_domains = self._clean_list_strings(
            meta.get("target_domains", [])
        )
        detected_level = str(meta.get("detected_level", "") or "")
        domain_routed = bool(meta.get("domain_routed", False))

        for ranked in ranked_sources[:20]:
            source = getattr(ranked, "source", None)

            if source is None:
                continue

            source_id = str(
                getattr(source, "source_id", "") or ""
            )
            title = str(getattr(source, "title", "") or "")
            url = str(getattr(source, "url", "") or "")
            platform = self._enum_value(
                getattr(source, "platform", "")
            )
            source_type = self._enum_value(
                getattr(source, "source_type", "")
            )
            difficulty = (
                self._enum_value(getattr(source, "difficulty", ""))
                or "unknown"
            )

            score = getattr(ranked, "score", None)
            confidence = getattr(ranked, "confidence", None)
            base_reason = str(getattr(ranked, "reason", "") or "")

            selection_parts: list[str] = []

            if domain_routed and target_domains:
                selection_parts.append(
                    f"Domain routing selected sources for "
                    f"{', '.join(target_domains)}"
                )

            if self._beginner_friendly_source(source):
                selection_parts.append("Beginner-friendly source")

            if difficulty == "advanced":
                selection_parts.append("Advanced source")

            if source_type in self._educational_source_types:
                selection_parts.append(
                    f"Educational format: {source_type}"
                )

            if platform in self._education_platforms:
                selection_parts.append(
                    f"Educational platform: {platform}"
                )

            if base_reason:
                selection_parts.append(base_reason)

            if score is not None:
                try:
                    selection_parts.append(
                        f"score={float(score):.3f}"
                    )
                except Exception:
                    pass

            if confidence is not None:
                try:
                    selection_parts.append(
                        f"confidence={float(confidence):.3f}"
                    )
                except Exception:
                    pass

            if not selection_parts:
                selection_parts.append(
                    "Selected by heuristic/consensus ranking"
                )

            reasons.append(
                {
                    "source_id": source_id,
                    "title": title,
                    "url": url,
                    "platform": platform,
                    "source_type": source_type,
                    "difficulty": difficulty,
                    "score": score,
                    "confidence": confidence,
                    "detected_level": detected_level,
                    "target_domains": target_domains,
                    "selection_reason": "; ".join(selection_parts),
                }
            )

        return reasons

    def _evaluate_quality_guards(
        self,
        state: PipelineState,
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        guards: dict[str, Any] = {
            "passed": True,
            "critical_failures": [],
            "warnings": [],
            "checks": {},
        }

        query = getattr(state, "query", None)
        beginner_request = self._is_beginner_request(query)

        sources = list(getattr(state, "sources", []) or [])
        ranked_sources = list(
            getattr(state, "ranked_sources", []) or []
        )

        guards["checks"]["beginner_request"] = beginner_request
        guards["checks"]["sources_found"] = len(sources)
        guards["checks"]["ranked_sources"] = len(ranked_sources)
        guards["checks"]["domain_confidence"] = meta.get(
            "domain_confidence", 0.0
        )
        guards["checks"]["detected_level"] = meta.get(
            "detected_level", ""
        )
        guards["checks"]["target_domains"] = meta.get(
            "target_domains", []
        )

        if not sources and not ranked_sources:
            guards["critical_failures"].append("no_sources_found")

        if ranked_sources:
            beginner_friendly_count = 0
            relevant_count = 0

            for ranked in ranked_sources:
                source = getattr(ranked, "source", None)

                if source is None:
                    continue

                if self._beginner_friendly_source(source):
                    beginner_friendly_count += 1

                if self._relevant_source(source, state):
                    relevant_count += 1

            guards["checks"]["beginner_friendly_sources"] = (
                beginner_friendly_count
            )
            guards["checks"]["relevant_sources"] = relevant_count

            if beginner_request and beginner_friendly_count == 0:
                level_text = clean_text(
                    getattr(query, "level", "") or ""
                ).lower()

                is_range_request = (
                    "to" in level_text
                    and "beginner" in level_text
                    and (
                        "advanced" in level_text
                        or "advance" in level_text
                    )
                )

                if is_range_request:
                    guards["warnings"].append(
                        "no_beginner_sources_for_range_query"
                    )
                else:
                    guards["critical_failures"].append(
                        "no_beginner_sources_for_beginner_query"
                    )

            if relevant_count == 0:
                guards["critical_failures"].append(
                    "all_sources_off_topic"
                )
            elif relevant_count < max(
                1, int(len(ranked_sources) * 0.25)
            ):
                guards["warnings"].append("low_topic_relevance")

        try:
            domain_confidence = float(
                meta.get("domain_confidence", 0.0) or 0.0
            )
        except Exception:
            domain_confidence = 0.0

        if domain_confidence < 0.20:
            guards["warnings"].append("low_domain_confidence")

        if ranked_sources and not getattr(
            state, "learning_path", None
        ):
            guards["warnings"].append("learning_path_missing")

        guards["passed"] = not guards["critical_failures"]
        return guards

    def _is_beginner_request(self, query: Any) -> bool:
        if query is None:
            return False

        level_profile = str(
            getattr(query, "level_profile", "") or ""
        ).strip().lower()

        if level_profile == "beginner":
            return True

        if level_profile in {"advanced", "mixed", "balanced"}:
            return False

        level = clean_text(
            getattr(query, "level", "") or ""
        ).lower()
        goal = clean_text(
            getattr(query, "goal", "") or ""
        ).lower()
        topic = clean_text(
            getattr(query, "topic", "") or ""
        ).lower()

        has_beginner_level = (
            "beginner" in level or "basics" in level
        )
        has_advanced_level = any(
            marker in level
            for marker in ("advanced", "advance", "expert")
        )

        if has_beginner_level and has_advanced_level:
            return False

        if level == "beginner":
            return True

        text = f"{level} {goal} {topic}".lower()

        markers = (
            "beginner",
            "basics",
            "basic",
            "fundamentals",
            "fundamental",
            "introduction",
            "intro",
            "tutorial",
            "course",
            "textbook",
            "from scratch",
            "step by step",
        )

        return any(marker in text for marker in markers)

    def _beginner_friendly_source(self, source: Any) -> bool:
        if source is None:
            return False

        difficulty = self._enum_value(
            getattr(source, "difficulty", "")
        ).lower()
        source_type = self._enum_value(
            getattr(source, "source_type", "")
        ).lower()
        platform = self._enum_value(
            getattr(source, "platform", "")
        ).lower()

        if difficulty == "beginner":
            return True

        if source_type in self._educational_source_types:
            return True

        if platform in self._education_platforms:
            return True

        metadata = getattr(source, "metadata", {}) or {}

        if isinstance(metadata, dict):
            metadata_difficulty = str(
                metadata.get("difficulty", "")
            ).lower()

            if metadata_difficulty == "beginner":
                return True

        return False

    def _relevant_source(
        self, source: Any, state: PipelineState
    ) -> bool:
        if source is None:
            return False

        title = clean_text(
            getattr(source, "title", "") or ""
        ).lower()
        abstract = clean_text(
            getattr(source, "abstract", "") or ""
        ).lower()

        metadata = getattr(source, "metadata", {}) or {}
        metadata_text = ""

        if isinstance(metadata, dict):
            metadata_text = self._flatten_metadata_text(
                metadata
            ).lower()

        full_text = f"{title} {abstract} {metadata_text}"

        primary = (
            getattr(state, "primary_concept", "")
            or getattr(state, "corrected_topic", "")
            or (
                getattr(state.query, "topic", "")
                if getattr(state, "query", None)
                else ""
            )
        )
        primary = clean_text(primary).lower()

        if primary and primary in full_text:
            return True

        tokens: set[str] = set()

        if primary:
            try:
                tokens.update(concept_tokens(primary))
            except Exception:
                tokens.update(
                    re.findall(r"[a-z0-9+#]+", primary)
                )

        keywords = list(getattr(state, "keywords", []) or [])

        for keyword in keywords:
            keyword_text = clean_text(keyword).lower()

            if keyword_text and keyword_text in full_text:
                return True

            try:
                tokens.update(concept_tokens(keyword_text))
            except Exception:
                tokens.update(
                    re.findall(r"[a-z0-9+#]+", keyword_text)
                )

        try:
            generic_terms = get_generic_terms()
        except Exception:
            generic_terms = set()

        meaningful_tokens = {
            token
            for token in tokens
            if token
            and token not in generic_terms
            and len(token) >= 3
        }

        if not meaningful_tokens:
            return True

        matched = sum(
            1
            for token in meaningful_tokens
            if token in full_text
        )

        return matched >= max(
            1, len(meaningful_tokens) // 3
        )

    def _flatten_metadata_text(
        self, metadata: dict[str, Any]
    ) -> str:
        parts: list[str] = []

        for value in metadata.values():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, (int, float, bool)):
                parts.append(str(value))
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    if isinstance(
                        item, (str, int, float, bool)
                    ):
                        parts.append(str(item))

        return " ".join(parts)

    def _infer_level(self, query: Any) -> str:
        if query is None:
            return "mixed"

        level = clean_text(
            getattr(query, "level", "") or ""
        ).lower()
        goal = clean_text(
            getattr(query, "goal", "") or ""
        ).lower()
        topic = clean_text(
            getattr(query, "topic", "") or ""
        ).lower()

        if level in {
            "beginner",
            "intermediate",
            "advanced",
            "mixed",
        }:
            return level

        text = f"{level} {goal} {topic}".lower()

        has_beginner = any(
            marker in text
            for marker in (
                "beginner",
                "basics",
                "basic",
                "fundamentals",
                "introduction",
                "intro",
                "from scratch",
            )
        )
        has_advanced = any(
            marker in text
            for marker in (
                "advanced",
                "expert",
                "state of the art",
                "state-of-the-art",
                "deep dive",
                "in depth",
                "in-depth",
            )
        )

        if has_beginner and has_advanced:
            return "mixed"

        if has_beginner:
            return "beginner"

        if has_advanced:
            return "advanced"

        return "mixed"

    def _detect_domains_fallback(self, query: Any) -> list[str]:
        if query is None:
            return ["general"]

        topic = clean_text(
            getattr(query, "topic", "") or ""
        ).lower()
        goal = clean_text(
            getattr(query, "goal", "") or ""
        ).lower()
        text = f"{topic} {goal}"

        domain_keywords = {
            "computer_science": (
                "ai",
                "artificial intelligence",
                "machine learning",
                "ml",
                "deep learning",
                "neural",
                "algorithm",
                "programming",
                "software",
                "mixture of experts",
                "moe",
                "llm",
                "transformer",
            ),
            "medicine": (
                "diabetes",
                "medical",
                "clinical",
                "treatment",
                "disease",
                "health",
                "patient",
                "therapy",
            ),
            "biology": (
                "photosynthesis",
                "biology",
                "cell",
                "genetics",
                "organism",
                "life science",
            ),
            "chemistry": (
                "chemistry",
                "organic",
                "molecule",
                "reaction",
                "chemical",
            ),
            "law": (
                "law",
                "legal",
                "constitution",
                "court",
                "justice",
            ),
            "mathematics": (
                "math",
                "calculus",
                "algebra",
                "geometry",
                "probability",
            ),
            "physics": (
                "physics",
                "quantum",
                "mechanics",
                "thermodynamics",
            ),
        }

        domains: list[str] = []

        for domain, keywords in domain_keywords.items():
            if any(keyword in text for keyword in keywords):
                domains.append(domain)

        education_markers = (
            "learn",
            "course",
            "tutorial",
            "basics",
            "beginner",
            "fundamentals",
            "introduction",
        )

        if any(marker in text for marker in education_markers):
            if "education" not in domains:
                domains.append("education")

        if not domains:
            domains.append("general")

        return domains

    def _detect_formats_fallback(
        self,
        query: Any,
        detected_level: str,
    ) -> list[str]:
        if query is None:
            return []

        goal = clean_text(
            getattr(query, "goal", "") or ""
        ).lower()
        topic = clean_text(
            getattr(query, "topic", "") or ""
        ).lower()
        text = f"{goal} {topic}".lower()

        formats: list[str] = []

        if "video" in text:
            formats.append("video")

        if "book" in text or "textbook" in text:
            formats.append("book")

        if "course" in text or "tutorial" in text:
            formats.append("course")

        if "paper" in text or "research" in text:
            formats.append("research_paper")

        level = str(detected_level or "").lower()

        if level == "beginner":
            for item in (
                "course",
                "book",
                "documentation",
                "video",
            ):
                if item not in formats:
                    formats.append(item)
        elif level == "advanced":
            for item in ("research_paper", "documentation"):
                if item not in formats:
                    formats.append(item)
        else:
            for item in (
                "documentation",
                "course",
                "research_paper",
            ):
                if item not in formats:
                    formats.append(item)

        return formats[:6]

    async def _run_source_summaries(
        self,
        state: PipelineState,
        mode: str,
    ) -> None:
        sources = [
            ranked.source
            for ranked in state.ranked_sources
            if getattr(ranked, "source", None) is not None
        ]

        try:
            summarizer = self._get_source_summarizer()
            summarize_sources = getattr(
                summarizer, "summarize_sources", None
            )

            if not callable(summarize_sources):
                raise AttributeError(
                    "SourceSummarizer.summarize_sources is not "
                    "available"
                )

            raw = await self._maybe_await(
                self._call_with_supported_args(
                    summarize_sources,
                    [sources],
                    [
                        {"query": state.query, "mode": mode},
                        {
                            "query_schema": state.query,
                            "mode": mode,
                        },
                        {
                            "topic": getattr(
                                state.query, "topic", ""
                            ),
                            "mode": mode,
                        },
                        {
                            "goal": getattr(
                                state.query, "goal", ""
                            ),
                            "mode": mode,
                        },
                        {
                            "level": getattr(
                                state.query, "level", ""
                            ),
                            "mode": mode,
                        },
                        {"mode": mode},
                        {},
                    ],
                )
            )

            summaries = self._as_list(
                self._get_field(raw, "summaries", raw)
            )
            state.source_summaries = [
                item for item in summaries if item is not None
            ]

            await self._emit(
                SOURCE_SUMMARIES_COMPLETED,
                {
                    "request_id": state.request_id,
                    "total_summaries": len(
                        state.source_summaries
                    ),
                    "summaries": len(state.source_summaries),
                },
            )

        except Exception as exc:
            state.add_warning(
                f"source_summaries_failed: {exc}; using fallback "
                f"summaries"
            )
            state.source_summaries = self._fallback_summaries(
                sources
            )

            await self._emit(
                SOURCE_SUMMARIES_COMPLETED,
                {
                    "request_id": state.request_id,
                    "total_summaries": len(
                        state.source_summaries
                    ),
                    "summaries": len(state.source_summaries),
                    "fallback": True,
                },
            )

    async def _run_learning_path(
        self,
        state: PipelineState,
        mode: str,
    ) -> None:
        try:
            build = getattr(
                self._learning_path_builder, "build", None
            )

            if not callable(build):
                raise AttributeError(
                    "LearningPathBuilder.build is not available"
                )

            raw = await self._maybe_await(
                self._call_with_supported_args(
                    build,
                    [state.ranked_sources, state.query],
                    [{"mode": mode}, {}],
                )
            )

            if raw is None:
                raise ValueError(
                    "learning path builder returned None"
                )

            state.learning_path = raw
            steps = self._as_list(
                self._get_field(raw, "steps", [])
            )

            await self._emit(
                LEARNING_PATH_COMPLETED,
                {
                    "request_id": state.request_id,
                    "total_steps": len(steps),
                    "steps": len(steps),
                },
            )

        except Exception as exc:
            state.add_warning(f"learning_path_failed: {exc}")
            await self._emit(
                STAGE_FAILED,
                {
                    "request_id": state.request_id,
                    "stage": "learning_path",
                    "error": str(exc),
                },
            )

    async def _run_path_enhancement(
        self,
        state: PipelineState,
        mode: str,
    ) -> None:
        if state.learning_path is None:
            return

        if mode == constants.MODE_FAST:
            return

        try:
            enhancer = self._get_path_enhancer()
            enhance = getattr(enhancer, "enhance", None)

            if not callable(enhance):
                raise AttributeError(
                    "PathEnhancer.enhance is not available"
                )

            raw = await self._maybe_await(
                self._call_with_supported_args(
                    enhance,
                    [state.learning_path],
                    [
                        {
                            "ranked_sources": state.ranked_sources,
                            "query": state.query,
                            "mode": mode,
                        },
                        {
                            "sources": state.ranked_sources,
                            "query": state.query,
                            "mode": mode,
                        },
                        {
                            "ranked_sources": state.ranked_sources,
                            "query_schema": state.query,
                            "mode": mode,
                        },
                        {
                            "sources": state.ranked_sources,
                            "query_schema": state.query,
                            "mode": mode,
                        },
                        {"query": state.query, "mode": mode},
                        {
                            "query_schema": state.query,
                            "mode": mode,
                        },
                        {
                            "topic": getattr(
                                state.query, "topic", ""
                            ),
                            "mode": mode,
                        },
                        {"mode": mode},
                        {},
                    ],
                )
            )

            if raw is not None:
                state.enhanced_learning_path = raw

                await self._emit(
                    PATH_ENHANCED,
                    {"request_id": state.request_id},
                )

        except Exception as exc:
            state.add_warning(f"path_enhancement_failed: {exc}")
            await self._emit(
                STAGE_FAILED,
                {
                    "request_id": state.request_id,
                    "stage": "path_enhancement",
                    "error": str(exc),
                },
            )

    async def _run_rag_indexing(
        self, state: PipelineState
    ) -> None:
        if not state.ranked_sources:
            self._trace.emit(
                "pipeline_rag_skipped_no_sources",
                request_id=state.request_id,
            )
            return

        await self._emit(
            RAG_INDEXING_STARTED,
            {
                "request_id": state.request_id,
                "total_sources": len(state.ranked_sources),
            },
        )

        try:
            from rag.vector_store import VectorStore

            store = VectorStore()

            try:
                count, chunk_count, mode = (
                    await self._index_ranked_sources(state, store)
                )
            finally:
                try:
                    store.close()
                except Exception as close_exc:
                    self._logger.warning(
                        f"VectorStore close raised: {close_exc}"
                    )

            self._set_attr(state, "rag_documents_indexed", count)

            await self._emit(
                RAG_INDEXED,
                {
                    "request_id": state.request_id,
                    "documents_indexed": count,
                    "total_documents": count,
                    "chunks_indexed": chunk_count,
                    "mode": mode,
                },
            )

        except Exception as exc:
            self._set_attr(state, "rag_documents_indexed", 0)
            state.add_warning(f"rag_indexing_failed: {exc}")

            await self._emit(
                STAGE_FAILED,
                {
                    "request_id": state.request_id,
                    "stage": "rag_indexing",
                    "error": str(exc),
                },
            )

    async def _index_ranked_sources(
        self,
        state: PipelineState,
        store: Any,
    ) -> tuple[int, int, str]:
        from rag.embedder import Embedder

        ranked_sources = list(state.ranked_sources or [])
        run_id = state.request_id

        if not ranked_sources:
            return 0, 0, "empty"

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
                    "pipeline_rag_index_mode",
                    request_id=run_id,
                    mode="parent",
                    documents_indexed=count,
                    chunks_indexed=0,
                )
                return count, 0, "parent"

            self._trace.emit(
                "pipeline_rag_chunk_attempt",
                request_id=run_id,
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
                        "pipeline_rag_index_mode",
                        request_id=run_id,
                        mode=mode,
                        documents_indexed=count,
                        chunks_indexed=chunk_count,
                    )

                    return count, chunk_count, mode

                self._trace.emit(
                    "pipeline_rag_chunk_empty",
                    request_id=run_id,
                    documents_indexed=count,
                    chunks_indexed=chunk_count,
                )
            except Exception as exc:
                if not fail_open:
                    raise

                self._logger.warning(
                    f"Chunk indexing failed: {exc}"
                )
                state.add_warning(
                    f"chunk_indexing_failed: {exc}"
                )

                self._trace.emit(
                    "pipeline_rag_chunk_failed",
                    request_id=run_id,
                    error=str(exc),
                )

            if not fallback_to_parent:
                return 0, 0, "chunk_failed"

            self._trace.emit(
                "pipeline_rag_fallback_to_parent",
                request_id=run_id,
            )

            count = await self._index_parent_only(
                embedder,
                store,
                ranked_sources,
                run_id,
            )

            self._trace.emit(
                "pipeline_rag_index_mode",
                request_id=run_id,
                mode="fallback",
                documents_indexed=count,
                chunks_indexed=0,
            )

            return count, 0, "fallback"

    async def _index_parent_only(
        self,
        embedder: Any,
        store: Any,
        ranked_sources: list[Any],
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
        ranked_sources: list[Any],
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

    async def _save_outputs(
        self,
        state: PipelineState,
        meta: dict[str, Any] | None = None,
        mode: str = "",
    ) -> dict[str, str]:
        meta = meta or self._empty_meta()
        payload = self._state_payload(state, meta)
        saved_files: dict[str, str] = {}

        output_dir = get_output_directory()
        json_dir = output_dir / "json"
        markdown_dir = output_dir / "markdown"
        reports_dir = output_dir / "reports"

        json_dir.mkdir(parents=True, exist_ok=True)
        markdown_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        prefix = self._file_prefix(state)
        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%d_%H%M%S"
        )
        base_name = f"{prefix}_{timestamp}"

        try:
            json_path = json_dir / f"{base_name}.json"
            json_path.write_text(
                json.dumps(
                    payload,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                encoding="utf-8",
            )
            saved_files["json"] = str(json_path)
        except Exception as exc:
            state.add_error(f"json_save_failed: {exc}")

        try:
            markdown_text = ""

            render_state = getattr(
                self._markdown_writer, "render_state", None
            )

            if callable(render_state):
                try:
                    markdown_text = str(render_state(state) or "")
                except Exception:
                    markdown_text = ""

            if not markdown_text:
                markdown_text = self._render_markdown(payload)

            markdown_text += self._render_meta_sections(payload)

            markdown_path = markdown_dir / f"{base_name}.md"
            markdown_path.write_text(
                markdown_text, encoding="utf-8"
            )
            saved_files["markdown"] = str(markdown_path)
        except Exception as exc:
            state.add_error(f"markdown_save_failed: {exc}")

        try:
            html_text = ""

            render_state = getattr(
                self._html_writer, "render_state", None
            )

            if callable(render_state):
                try:
                    html_text = str(render_state(state) or "")
                except Exception:
                    html_text = ""

            if not html_text:
                html_text = self._render_html(payload)

            html_path = reports_dir / f"{base_name}.html"
            html_path.write_text(html_text, encoding="utf-8")
            saved_files["html"] = str(html_path)
        except Exception as exc:
            state.add_error(f"html_save_failed: {exc}")

        try:
            await self._sqlite_store.save_state(state)
            saved_files["sqlite"] = str(
                self._sqlite_store.database_path
            )
        except Exception as exc:
            state.add_error(f"sqlite_save_failed: {exc}")

        self._trace.emit(
            "outputs_saved",
            request_id=state.request_id,
            files=list(saved_files.keys()),
        )

        return saved_files

    async def _save_memory(self, state: PipelineState) -> None:
        if self._enum_value(state.status) == "failed":
            return

        try:
            episode = {
                "episode_id": str(uuid4()),
                "task": (
                    state.query.topic
                    if state.query
                    else state.request_id
                ),
                "task_type": "pipeline",
                "goal": state.query.goal if state.query else "",
                "level": state.query.level if state.query else "",
                "status": self._enum_value(state.status),
                "quality_score": self._quality_score(state),
                "total_iterations": 0,
                "total_tokens_used": 0,
                "actions_taken": [],
                "errors": state.errors,
                "warnings": state.warnings,
                "payload": {
                    "request_id": state.request_id,
                    "total_sources": len(state.sources),
                    "total_ranked": len(state.ranked_sources),
                    "total_steps": self._count_learning_steps(
                        state.learning_path
                    ),
                },
                "created_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "finished_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "latency_ms": state.latency_ms,
            }

            await self._sqlite_store.save_episode(episode)

        except Exception as exc:
            state.add_warning(f"memory_save_failed: {exc}")

    def _build_result(
        self,
        state: PipelineState,
        saved_files: dict[str, str],
        meta: dict[str, Any] | None = None,
        mode: str = "",
    ) -> PipelineResult:
        meta = meta or self._empty_meta()

        return PipelineResult(
            request_id=str(
                getattr(state, "request_id", "")
            ),
            status=getattr(
                state, "status", TaskStatus.FAILED
            ),
            query=getattr(state, "query", None),
            mode=mode or constants.DEFAULT_MODE,
            corrected_topic=str(
                getattr(state, "corrected_topic", "") or ""
            ),
            primary_concept=str(
                getattr(state, "primary_concept", "") or ""
            ),
            keywords=list(
                getattr(state, "keywords", []) or []
            ),
            expanded_queries=list(
                getattr(state, "expanded_queries", []) or []
            ),
            sources=list(getattr(state, "sources", []) or []),
            ranked_sources=list(
                getattr(state, "ranked_sources", []) or []
            ),
            source_summaries=list(
                getattr(state, "source_summaries", []) or []
            ),
            learning_path=getattr(
                state, "learning_path", None
            ),
            enhanced_learning_path=getattr(
                state, "enhanced_learning_path", None
            ),
            errors=list(getattr(state, "errors", []) or []),
            warnings=list(
                getattr(state, "warnings", []) or []
            ),
            saved_files=dict(saved_files or {}),
            rag_documents_indexed=int(
                getattr(state, "rag_documents_indexed", 0) or 0
            ),
            latency_ms=float(
                getattr(state, "latency_ms", 0.0) or 0.0
            ),
            target_domains=list(
                meta.get("target_domains", []) or []
            ),
            target_formats=list(
                meta.get("target_formats", []) or []
            ),
            domain_confidence=float(
                meta.get("domain_confidence", 0.0) or 0.0
            ),
            detected_level=str(
                meta.get("detected_level", "") or ""
            ),
            domain_routed=bool(
                meta.get("domain_routed", False)
            ),
            domain_expanded=bool(
                meta.get("domain_expanded", False)
            ),
            domain_platforms=list(
                meta.get("domain_platforms", []) or []
            ),
            source_selection_reasons=list(
                meta.get("source_selection_reasons", []) or []
            ),
            quality_guards=dict(
                meta.get("quality_guards", {}) or {}
            ),
            quality_gate_passed=bool(
                meta.get("quality_gate_passed", True)
            ),
        )

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

        raise TypeError(
            "query must be SearchQuerySchema, str, or dict"
        )

    def _normalize_ranked(
        self, raw: Any
    ) -> list[RankedSource]:
        if raw is None:
            return []

        if isinstance(raw, tuple):
            if raw and isinstance(raw[0], list):
                raw = raw[0]
            else:
                raw = list(raw)

        if isinstance(raw, dict):
            for key in (
                "rankings",
                "ranked_sources",
                "sources",
                "results",
            ):
                value = raw.get(key)

                if isinstance(value, list):
                    raw = value
                    break

        if not isinstance(raw, list):
            return []

        ranked: list[RankedSource] = []

        for index, item in enumerate(raw, start=1):
            normalized = self._normalize_ranked_item(
                item, index
            )

            if normalized is not None:
                ranked.append(normalized)

        return ranked

    def _normalize_ranked_item(
        self,
        item: Any,
        index: int,
    ) -> Optional[RankedSource]:
        if isinstance(item, RankedSource):
            return item

        if hasattr(item, "source") and hasattr(item, "rank"):
            return item

        if not isinstance(item, dict):
            return None

        source_data = item.get("source")
        source = None

        if isinstance(source_data, dict):
            source = self._dict_to_source(source_data)
        elif hasattr(source_data, "title"):
            source = source_data

        if source is None:
            source = self._dict_to_source(item)

        if source is None:
            return None

        try:
            return RankedSource(
                source=source,
                rank=int(item.get("rank", index)),
                score=float(item.get("score", 0.5)),
                confidence=float(
                    item.get("confidence", 0.5)
                ),
                reason=str(item.get("reason", "") or "")
                or None,
            )
        except Exception:
            return None

    def _dict_to_source(
        self,
        item: dict[str, Any],
    ) -> Optional[Source]:
        try:
            title = str(
                item.get("title", "") or ""
            ).strip()
            url = str(item.get("url", "") or "").strip()

            if not title and not url:
                return None

            year = item.get("year")
            citation_count = item.get("citation_count")

            return Source(
                source_id=str(
                    item.get("source_id", "") or url or title
                ),
                title=title or "Untitled source",
                url=url,
                platform=str(
                    item.get("platform", "web") or "web"
                ),
                source_type=str(
                    item.get("source_type", "other") or "other"
                ),
                abstract=str(
                    item.get("abstract", "") or ""
                )
                or None,
                year=int(year) if year is not None else None,
                citation_count=(
                    int(citation_count)
                    if citation_count is not None
                    else None
                ),
                metadata=item.get("metadata", {}) or {},
            )
        except Exception:
            return None

    def _fallback_rank(
        self,
        sources: list[Source],
        query: SearchQuerySchema,
    ) -> list[RankedSource]:
        ranked: list[RankedSource] = []
        limit = max(
            1,
            int(
                getattr(query, "max_results", len(sources))
                or len(sources)
            ),
        )

        for index, source in enumerate(
            sources[:limit], start=1
        ):
            ranked.append(
                RankedSource(
                    source=source,
                    rank=index,
                    score=0.5,
                    confidence=0.3,
                    reason="fallback_ranking",
                )
            )

        return ranked

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
                        "Generated by pipeline fallback summarizer."
                    ),
                    "key_topics": [],
                }
            )

        return summaries

    def _get_source_summarizer(self) -> SourceSummarizer:
        if self._source_summarizer is None:
            self._source_summarizer = SourceSummarizer()

        return self._source_summarizer

    def _get_path_enhancer(self) -> PathEnhancer:
        if self._path_enhancer is None:
            self._path_enhancer = PathEnhancer(use_llm=True)

        return self._path_enhancer

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
        attempts: list[
            tuple[tuple[Any, ...], dict[str, Any]]
        ] = []

        for candidate in candidates or []:
            if accepts_var_keyword:
                attempts.append(
                    (tuple(base_args), dict(candidate))
                )

            call_args = list(base_args)

            for parameter in positional_params[
                len(base_args) :
            ]:
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

    def _finalize_status(self, state: PipelineState) -> None:
        if not state.ranked_sources and not state.sources:
            self._mark_status(
                state,
                "failed",
                "; ".join(state.errors[:3]) or "no_sources_found",
            )
            return

        if state.errors and not state.ranked_sources:
            self._mark_status(
                state,
                "failed",
                "; ".join(state.errors[:3]) or "pipeline_failed",
            )
            return

        if state.errors or state.warnings:
            partial = getattr(TaskStatus, "PARTIAL", None)

            if partial is not None:
                try:
                    state.status = partial
                    return
                except Exception:
                    pass

            state.mark_success()
            return

        state.mark_success()

    def _mark_status(
        self,
        state: PipelineState,
        status: str,
        message: str,
    ) -> None:
        if status == "failed":
            try:
                state.mark_failed(message)
                return
            except Exception:
                pass

        if status == "success":
            try:
                state.mark_success()
                return
            except Exception:
                pass

        try:
            state.status = getattr(
                TaskStatus, status.upper(), TaskStatus.FAILED
            )
        except Exception:
            pass

    def _quality_score(self, state: PipelineState) -> float:
        score = 0.0

        if state.sources:
            score += 0.25

        if state.ranked_sources:
            score += 0.25

        if state.source_summaries:
            score += 0.20

        if state.learning_path:
            score += 0.20

        if getattr(state, "enhanced_learning_path", None):
            score += 0.10

        return min(1.0, score)

    def _state_payload(
        self,
        state: PipelineState,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self._manual_state_payload(state)
        meta = meta or self._empty_meta()

        payload.update(
            {
                "target_domains": meta.get(
                    "target_domains", []
                ),
                "target_formats": meta.get(
                    "target_formats", []
                ),
                "domain_confidence": meta.get(
                    "domain_confidence", 0.0
                ),
                "detected_level": meta.get(
                    "detected_level", ""
                ),
                "domain_routed": meta.get(
                    "domain_routed", False
                ),
                "domain_expanded": meta.get(
                    "domain_expanded", False
                ),
                "domain_platforms": meta.get(
                    "domain_platforms", []
                ),
                "source_selection_reasons": meta.get(
                    "source_selection_reasons", []
                ),
                "quality_guards": meta.get(
                    "quality_guards", {}
                ),
                "quality_gate_passed": meta.get(
                    "quality_gate_passed", True
                ),
            }
        )

        return payload

    def _manual_state_payload(
        self,
        state: PipelineState,
    ) -> dict[str, Any]:
        return {
            "request_id": getattr(state, "request_id", ""),
            "status": self._enum_value(
                getattr(state, "status", "")
            ),
            "query": self._model_to_dict(
                getattr(state, "query", None)
            ),
            "corrected_topic": getattr(
                state, "corrected_topic", ""
            ),
            "primary_concept": getattr(
                state, "primary_concept", ""
            ),
            "keywords": list(
                getattr(state, "keywords", []) or []
            ),
            "expanded_queries": list(
                getattr(state, "expanded_queries", []) or []
            ),
            "sources": [
                self._model_to_dict(source)
                for source in getattr(state, "sources", []) or []
            ],
            "ranked_sources": [
                self._serialize_ranked(ranked)
                for ranked in getattr(
                    state, "ranked_sources", []
                )
                or []
            ],
            "source_summaries": [
                self._model_to_dict(summary)
                for summary in getattr(
                    state, "source_summaries", []
                )
                or []
            ],
            "learning_path": self._model_to_dict(
                getattr(state, "learning_path", None)
            ),
            "enhanced_learning_path": self._model_to_dict(
                getattr(state, "enhanced_learning_path", None)
            ),
            "errors": list(
                getattr(state, "errors", []) or []
            ),
            "warnings": list(
                getattr(state, "warnings", []) or []
            ),
            "saved_files": dict(
                getattr(state, "saved_files", {}) or {}
            ),
            "rag_documents_indexed": int(
                getattr(state, "rag_documents_indexed", 0) or 0
            ),
            "latency_ms": float(
                getattr(state, "latency_ms", 0.0) or 0.0
            ),
        }

    def _serialize_ranked(
        self, ranked: Any
    ) -> dict[str, Any]:
        try:
            return ranked.model_dump(
                mode="json",
                exclude_none=False,
                by_alias=False,
            )
        except Exception:
            source = getattr(ranked, "source", ranked)

            return {
                "source": self._model_to_dict(source),
                "rank": getattr(ranked, "rank", None),
                "score": getattr(ranked, "score", None),
                "confidence": getattr(
                    ranked, "confidence", None
                ),
                "reason": getattr(ranked, "reason", None),
            }

    def _render_markdown(
        self, payload: dict[str, Any]
    ) -> str:
        try:
            from storage.markdown_writer import MarkdownWriter

            payload = dict(payload or {})
            payload.setdefault("kind", "pipeline")
            writer = MarkdownWriter.__new__(MarkdownWriter)
            writer._file_manager = None
            return writer.render_payload(payload)
        except Exception:
            topic = (
                payload.get("corrected_topic")
                or payload.get("query", {}).get("topic")
                or "Research Pipeline Run"
            )
            return (
                f"# {topic}\n\n"
                f"- Request ID: {payload.get('request_id', '-')}\n"
                f"- Status: {str(payload.get('status', '-')).upper()}\n\n"
            )

    def _render_meta_sections(
        self, payload: dict[str, Any]
    ) -> str:
        lines: list[str] = []

        lines.extend(
            [
                "",
                "## Domain Understanding",
                "",
                f"- Detected Domains: {', '.join(payload.get('target_domains', []) or []) or '-'}",
                f"- Detected Level: {payload.get('detected_level') or payload.get('query', {}).get('level') or '-'}",
                f"- Domain Confidence: {payload.get('domain_confidence', 0.0)}",
                f"- Domain Routed: {payload.get('domain_routed', False)}",
                f"- Domain Expanded: {payload.get('domain_expanded', False)}",
                f"- Domain Platforms: {', '.join(payload.get('domain_platforms', []) or []) or '-'}",
                "",
            ]
        )

        quality_guards = payload.get("quality_guards", {}) or {}

        lines.extend(
            [
                "## Quality Guards",
                "",
                f"- Passed: {quality_guards.get('passed', payload.get('quality_gate_passed', True))}",
            ]
        )

        critical_failures = (
            quality_guards.get("critical_failures", []) or []
        )

        if critical_failures:
            lines.append("- Critical Failures:")
            lines.extend(
                f"  - {item}" for item in critical_failures
            )
        else:
            lines.append("- Critical Failures: None")

        guard_warnings = (
            quality_guards.get("warnings", []) or []
        )

        if guard_warnings:
            lines.append("- Warnings:")
            lines.extend(
                f"  - {item}" for item in guard_warnings
            )
        else:
            lines.append("- Warnings: None")

        checks = quality_guards.get("checks", {}) or {}

        if checks:
            lines.append("- Checks:")

            for key, value in checks.items():
                lines.append(f"  - {key}: {value}")

        lines.extend(["", "## Source Selection Reasoning", ""])

        reasons = (
            payload.get("source_selection_reasons", []) or []
        )

        if not reasons:
            lines.append("No source selection reasoning available.")
        else:
            for item in reasons[:12]:
                if not isinstance(item, dict):
                    continue

                lines.append(
                    f"{item.get('rank', '-')}. "
                    f"{item.get('title', 'Untitled')} "
                    f"({item.get('platform', '-')}, "
                    f"{item.get('source_type', '-')}, "
                    f"difficulty={item.get('difficulty', '-')})"
                )

                selection_reason = str(
                    item.get("selection_reason", "") or ""
                )

                if selection_reason:
                    lines.append(f"   - Reason: {selection_reason}")

                url = item.get("url")

                if url:
                    lines.append(f"   - URL: {url}")

        return "\n".join(lines)

    def _render_html(
        self, payload: dict[str, Any]
    ) -> str:
        try:
            from storage.html_writer import HTMLWriter

            payload = dict(payload or {})
            payload.setdefault("kind", "pipeline")
            return HTMLWriter().render_payload(payload)
        except Exception:
            title = html.escape(
                str(
                    payload.get("corrected_topic")
                    or payload.get("query", {}).get("topic")
                    or "Research Pipeline Run"
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

    def _file_prefix(self, state: PipelineState) -> str:
        topic = ""

        if state.query is not None:
            topic = str(getattr(state.query, "topic", "") or "")

        return self._slug(topic or "pipeline_research")

    async def _emit(
        self,
        event_name: str,
        payload: dict[str, Any],
    ) -> None:
        try:
            if self._event_bus is None:
                return

            emit = getattr(self._event_bus, "emit", None)

            if not callable(emit):
                return

            result = emit(event_name, payload)

            if inspect.isawaitable(result):
                await result

        except Exception as exc:
            self._logger.warning(
                f"Failed to emit event {event_name}: {exc}"
            )

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value

        return value

    def _set_attr(
        self, obj: Any, name: str, value: Any
    ) -> None:
        try:
            setattr(obj, name, value)
        except Exception:
            try:
                object.__setattr__(obj, name, value)
            except Exception:
                pass

    def _model_to_dict(self, obj: Any) -> Any:
        if obj is None:
            return None

        if isinstance(obj, dict):
            return obj

        model_dump = getattr(obj, "model_dump", None)

        if callable(model_dump):
            try:
                return model_dump(
                    mode="json",
                    exclude_none=False,
                    by_alias=False,
                )
            except Exception:
                pass

        dict_method = getattr(obj, "dict", None)

        if callable(dict_method):
            try:
                return dict_method()
            except Exception:
                pass

        return str(obj)

    def _get_field(
        self,
        obj: Any,
        key: str,
        default: Any = None,
    ) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)

        return getattr(obj, key, default)

    def _as_list(self, value: Any) -> list[Any]:
        if value is None:
            return []

        if isinstance(value, list):
            return value

        if isinstance(value, tuple):
            return list(value)

        return [value]

    def _clean_list_strings(self, value: Any) -> list[str]:
        if value is None:
            return []

        if isinstance(value, str):
            value = [value]

        if isinstance(value, (list, tuple, set)):
            result: list[str] = []

            for item in value:
                text = str(item or "").strip()

                if text and text not in result:
                    result.append(text)

            return result

        return []

    def _count_learning_steps(
        self, learning_path: Any
    ) -> int:
        if learning_path is None:
            return 0

        steps = (
            self._get_field(learning_path, "steps", []) or []
        )
        return len(self._as_list(steps))

    def _enum_value(self, value: Any) -> str:
        if value is None:
            return ""

        return str(getattr(value, "value", value))

    def _slug(self, value: str) -> str:
        cleaned = str(value or "").lower().strip()
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

        return (cleaned or "pipeline_run")[:80]