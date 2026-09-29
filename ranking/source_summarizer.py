from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from core import constants
from core.config import get_project_root
from core.exceptions import ProviderRateLimitError
from core.models import CoreModel, Difficulty, Source
from core.schemas import LLMMessageSchema, LLMRequestSchema
from llm.guardrails import validate_llm_request, validate_llm_response
from llm.parser import parse_json_object_response
from llm.prompts import build_source_summary_prompt
from llm.provider import LLMProviderManager
from llm.router import get_fast_model_references
from utils.logger import get_logger, get_trace_logger
from utils.text import clean_text, truncate_text

_MODE_CONFIG_CACHE: dict[str, dict[str, Any]] | None = None


def _load_all_mode_configs() -> dict[str, dict[str, Any]]:
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


def _resolve_mode(mode: str) -> str:
    text = str(mode or "").strip().lower()

    if text in constants.VALID_MODES:
        return text

    return constants.DEFAULT_MODE


def _mode_config(mode: str) -> dict[str, Any]:
    configs = _load_all_mode_configs()
    resolved = _resolve_mode(mode)

    if resolved not in configs:
        resolved = constants.DEFAULT_MODE

    return dict(
        configs.get(
            resolved,
            constants.MODE_DEFAULTS[constants.DEFAULT_MODE],
        )
    )


class SourceSummary(CoreModel):
    source_id: str
    title: str
    summary: str
    difficulty: Difficulty = Difficulty.INTERMEDIATE
    key_topics: list[str] = Field(default_factory=list)
    why_useful: str = ""
    estimated_reading_minutes: int = 10
    llm_generated: bool = False


class SourceSummarizer:
    def __init__(
        self,
        use_llm: bool = True,
        max_sources: int | None = None,
        max_concurrency: int | None = None,
        batch_size: int | None = None,
        batch_pause_seconds: float | None = None,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.5,
    ) -> None:
        self._use_llm = bool(use_llm)
        self._max_sources = max(
            1,
            int(
                max_sources
                if max_sources is not None
                else constants.MAX_SOURCES_TO_SUMMARIZE
            ),
        )
        self._max_concurrency = max(
            1,
            int(
                max_concurrency
                if max_concurrency is not None
                else constants.SUMMARY_CONCURRENCY
            ),
        )
        self._batch_size = max(
            1,
            int(
                batch_size
                if batch_size is not None
                else constants.SUMMARY_BATCH_SIZE
            ),
        )
        self._batch_pause_seconds = max(
            0.0,
            float(
                batch_pause_seconds
                if batch_pause_seconds is not None
                else constants.SUMMARY_BATCH_PAUSE
            ),
        )
        self._max_retries = max(1, int(max_retries))
        self._retry_backoff_seconds = max(
            0.5, float(retry_backoff_seconds)
        )
        self._logger = get_logger("ranking.source_summarizer")
        self._trace = get_trace_logger()
        self._degraded_providers: set[str] = set()

    def reset_provider_health(self) -> None:
        self._degraded_providers.clear()

    async def summarize_sources(
        self,
        sources: list[Any],
        query: Any = None,
        mode: str = "",
        **_: Any,
    ) -> list[SourceSummary]:
        if not isinstance(sources, list):
            sources = []

        valid_sources = [
            source for source in sources if isinstance(source, Source)
        ]

        resolved_mode = _resolve_mode(mode)
        mode_values = _mode_config(resolved_mode)
        limit = self._mode_limit(resolved_mode, mode_values)
        runtime = self._mode_runtime(mode_values)

        valid_sources = valid_sources[:limit]

        if not valid_sources:
            return []

        if not self._use_llm:
            return [
                self._heuristic_summary(source)
                for source in valid_sources
            ]

        llm_candidates: list[Source] = []
        heuristic_only: list[Source] = []

        for source in valid_sources:
            abstract = clean_text(source.abstract or "")

            if abstract:
                llm_candidates.append(source)
            else:
                heuristic_only.append(source)

        self._trace.emit(
            "summarizer_started",
            total_sources=len(valid_sources),
            llm_candidates=len(llm_candidates),
            heuristic_only=len(heuristic_only),
            batch_size=self._batch_size,
            max_concurrency=runtime["concurrency"],
            batch_pause=runtime["pause"],
            mode=resolved_mode,
            degraded_providers=sorted(self._degraded_providers),
        )

        results: list[SourceSummary] = [
            self._heuristic_summary(source)
            for source in heuristic_only
        ]

        if not llm_candidates:
            self._trace.emit(
                "summarizer_completed",
                total_summaries=len(results),
                llm_summaries=0,
                heuristic_summaries=len(results),
                mode=resolved_mode,
            )
            return self._reorder_by_source_order(
                results, valid_sources
            )

        semaphore = asyncio.Semaphore(runtime["concurrency"])
        per_call_timeout = self._per_call_timeout(resolved_mode)

        async def summarize_with_limit(
            manager: LLMProviderManager,
            source: Source,
        ) -> SourceSummary:
            async with semaphore:
                try:
                    return await asyncio.wait_for(
                        self._summarize_with_retries(
                            manager, source, query
                        ),
                        timeout=per_call_timeout,
                    )
                except asyncio.TimeoutError:
                    self._trace.emit(
                        "summary_call_hard_timeout",
                        source_id=source.source_id,
                        timeout_seconds=per_call_timeout,
                        mode=resolved_mode,
                    )
                    return self._heuristic_summary(source)

        async with LLMProviderManager() as manager:
            batch_pause = runtime["pause"]

            for batch_start in range(
                0, len(llm_candidates), self._batch_size
            ):
                batch = llm_candidates[
                    batch_start : batch_start + self._batch_size
                ]

                batch_results = await asyncio.gather(
                    *(
                        summarize_with_limit(manager, source)
                        for source in batch
                    )
                )

                results.extend(batch_results)

                llm_count = sum(
                    1
                    for item in batch_results
                    if item.llm_generated
                )

                self._trace.emit(
                    "summarizer_batch_completed",
                    batch_index=(batch_start // self._batch_size) + 1,
                    batch_size=len(batch),
                    llm_summaries=llm_count,
                    heuristic_summaries=len(batch) - llm_count,
                    completed=len(results),
                    total=len(valid_sources),
                    mode=resolved_mode,
                    degraded_providers=sorted(
                        self._degraded_providers
                    ),
                )

                if (
                    batch_pause > 0.0
                    and batch_start + self._batch_size
                    < len(llm_candidates)
                ):
                    await asyncio.sleep(batch_pause)

        llm_total = sum(
            1 for item in results if item.llm_generated
        )

        self._trace.emit(
            "summarizer_completed",
            total_summaries=len(results),
            llm_summaries=llm_total,
            heuristic_summaries=len(results) - llm_total,
            mode=resolved_mode,
            degraded_providers=sorted(self._degraded_providers),
        )

        return self._reorder_by_source_order(
            results, valid_sources
        )

    async def summarize_source(
        self,
        source: Source,
        query: Any = None,
        **_: Any,
    ) -> SourceSummary:
        if not self._use_llm:
            return self._heuristic_summary(source)

        abstract = clean_text(source.abstract or "")

        if not abstract:
            return self._heuristic_summary(source)

        async with LLMProviderManager() as manager:
            return await self._summarize_with_retries(
                manager, source, query
            )

    def _mode_limit(
        self,
        mode: str,
        mode_values: dict[str, Any] | None = None,
    ) -> int:
        values = mode_values or _mode_config(mode)

        try:
            return max(
                1,
                int(
                    values.get(
                        "summarize_max", self._max_sources
                    )
                ),
            )
        except Exception:
            return self._max_sources

    def _mode_runtime(
        self,
        mode_values: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            concurrency = int(
                mode_values.get(
                    "summarize_concurrency",
                    self._max_concurrency,
                )
            )
        except Exception:
            concurrency = self._max_concurrency

        try:
            pause = float(
                mode_values.get(
                    "summarize_batch_pause",
                    self._batch_pause_seconds,
                )
            )
        except Exception:
            pause = self._batch_pause_seconds

        return {
            "concurrency": max(1, concurrency),
            "pause": max(0.0, pause),
        }
    def _per_call_timeout(self, mode: str) -> float:
        if mode == constants.MODE_FAST:
            return 15.0

        if mode == constants.MODE_BALANCED:
            return 25.0

        return 40.0
    def _reorder_by_source_order(
        self,
        summaries: list[SourceSummary],
        sources: list[Source],
    ) -> list[SourceSummary]:
        order = {
            source.source_id: index
            for index, source in enumerate(sources)
        }

        return sorted(
            summaries,
            key=lambda item: order.get(item.source_id, 1_000_000),
        )

    async def _summarize_with_retries(
        self,
        manager: LLMProviderManager,
        source: Source,
        query: Any = None,
    ) -> SourceSummary:
        last_error = ""

        for attempt in range(1, self._max_retries + 1):
            try:
                return await self._llm_summary(
                    manager, source, query
                )
            except ProviderRateLimitError as exc:
                last_error = str(exc)

                if attempt >= self._max_retries:
                    break

                delay = self._rate_limit_delay(exc, attempt)

                self._trace.emit(
                    "summary_rate_limited",
                    source_id=source.source_id,
                    attempt=attempt,
                    retry_delay_seconds=round(delay, 2),
                )

                await asyncio.sleep(delay)
            except Exception as exc:
                last_error = str(exc)

                if attempt >= self._max_retries:
                    break

                delay = min(
                    self._retry_backoff_seconds
                    * (2 ** (attempt - 1)),
                    30.0,
                )
                delay += random.uniform(0.0, 0.5)

                self._trace.emit(
                    "summary_retry",
                    source_id=source.source_id,
                    attempt=attempt,
                    error=last_error,
                    retry_delay_seconds=round(delay, 2),
                )

                await asyncio.sleep(delay)

        self._trace.emit(
            "summary_fallback_to_heuristic",
            source_id=source.source_id,
            attempts=self._max_retries,
            last_error=last_error,
        )

        return self._heuristic_summary(source)

    def _rate_limit_delay(
        self,
        exc: ProviderRateLimitError,
        attempt: int,
    ) -> float:
        if exc.retry_after is not None:
            return max(float(exc.retry_after), 1.0) + random.uniform(
                0.0, 1.0
            )

        base = self._retry_backoff_seconds * (2**attempt)

        return min(base, 30.0) + random.uniform(0.0, 1.0)

    def _provider_value(self, model_reference: Any) -> str:
        provider = getattr(model_reference, "provider", None)
        value = getattr(provider, "value", provider)
        return str(value or "").strip().lower()

    def _mark_provider_degraded(self, provider_value: str) -> None:
        normalized = str(provider_value or "").strip().lower()

        if normalized:
            self._degraded_providers.add(normalized)

    def _reorder_models(
        self,
        model_references: list[Any],
    ) -> list[Any]:
        references = list(model_references)

        if not references or not self._degraded_providers:
            return references

        healthy: list[Any] = []
        degraded: list[Any] = []

        for reference in references:
            provider_value = self._provider_value(reference)

            if (
                provider_value
                and provider_value in self._degraded_providers
            ):
                degraded.append(reference)
            else:
                healthy.append(reference)

        return healthy + degraded

    async def _llm_summary(
        self,
        manager: LLMProviderManager,
        source: Source,
        query: Any = None,
    ) -> SourceSummary:
        model_references = get_fast_model_references(limit=3)

        if not model_references:
            raise RuntimeError(
                "No usable fast models for summarization"
            )

        ordered_references = self._reorder_models(model_references)

        serialized_source = {
            "title": source.title,
            "abstract": truncate_text(
                source.abstract or "", max_length=1500
            ),
            "platform": self._enum_value(source.platform),
            "source_type": self._enum_value(source.source_type),
            "year": source.year,
            "citation_count": source.citation_count,
        }

        prompt = self._build_prompt(serialized_source, query)

        messages = [
            LLMMessageSchema(
                role="system",
                content=(
                    "You are a concise learning-resource summarizer. "
                    "Return only valid JSON."
                ),
            ),
            LLMMessageSchema(role="user", content=prompt),
        ]

        last_error: Exception | None = None

        for model_reference in ordered_references:
            provider_value = self._provider_value(model_reference)
            model_id = str(
                getattr(model_reference, "model_id", "") or ""
            )

            try:
                request = LLMRequestSchema(
                    provider=model_reference.provider,
                    model=model_reference.model_id,
                    messages=messages,
                    temperature=constants.DEFAULT_TEMPERATURE,
                    max_tokens=constants.SUMMARY_MAX_TOKENS,
                    response_format="json_object",
                )

                request = validate_llm_request(request)
                response = await manager.complete(request)
                response = validate_llm_response(response)

                payload = parse_json_object_response(
                    response.content
                )
                summary = self._parse_payload(source, payload)

                self._trace.emit(
                    "summary_llm_success",
                    source_id=source.source_id,
                    provider=provider_value,
                    model=model_id,
                )

                return summary
            except ProviderRateLimitError as exc:
                self._mark_provider_degraded(provider_value)
                last_error = exc

                self._trace.emit(
                    "summary_model_rate_limited",
                    source_id=source.source_id,
                    provider=provider_value,
                    model=model_id,
                    degraded_providers=sorted(
                        self._degraded_providers
                    ),
                )

                continue
            except Exception as exc:
                last_error = exc

                self._trace.emit(
                    "summary_model_failed",
                    source_id=source.source_id,
                    provider=provider_value,
                    model=model_id,
                    error=str(exc),
                )

                continue

        if last_error is not None:
            raise last_error

        raise RuntimeError(
            "Summarization failed with no error captured"
        )

    def _build_prompt(
        self,
        serialized_source: dict[str, Any],
        query: Any,
    ) -> str:
        query_text = self._query_text(query)

        enriched_source = dict(serialized_source)

        if query_text:
            enriched_source["learning_goal"] = query_text

        attempts = []

        if query_text:
            attempts.append(
                lambda: build_source_summary_prompt(
                    enriched_source, query_text
                )
            )
            attempts.append(
                lambda: build_source_summary_prompt(
                    enriched_source, query=query_text
                )
            )

        attempts.append(
            lambda: build_source_summary_prompt(enriched_source)
        )
        attempts.append(
            lambda: build_source_summary_prompt(serialized_source)
        )

        for attempt in attempts:
            try:
                return attempt()
            except Exception:
                continue

        return build_source_summary_prompt(serialized_source)

    def _query_text(self, query: Any) -> str:
        if query is None:
            return ""

        if isinstance(query, str):
            return clean_text(query)

        if isinstance(query, dict):
            for key in (
                "topic",
                "corrected_topic",
                "primary_concept",
                "goal",
                "query",
            ):
                text = clean_text(query.get(key, ""))

                if text:
                    return text

            return ""

        for key in (
            "topic",
            "corrected_topic",
            "primary_concept",
            "goal",
            "query",
        ):
            text = clean_text(getattr(query, key, ""))

            if text:
                return text

        return clean_text(query)

    def _heuristic_summary(self, source: Source) -> SourceSummary:
        abstract = clean_text(source.abstract or "")

        if abstract:
            summary = truncate_text(
                abstract, max_length=300, suffix="..."
            )
        else:
            summary = (
                f"{source.title} "
                f"({self._enum_value(source.source_type)} resource)."
            )

        key_topics = self._extract_key_topics(source)
        reading_minutes = self._estimate_reading_minutes(source)
        difficulty = source.difficulty or Difficulty.INTERMEDIATE

        return SourceSummary(
            source_id=source.source_id,
            title=source.title,
            summary=summary,
            difficulty=difficulty,
            key_topics=key_topics,
            why_useful=self._build_why_useful(source),
            estimated_reading_minutes=reading_minutes,
            llm_generated=False,
        )

    def _parse_payload(
        self,
        source: Source,
        payload: dict[str, Any],
    ) -> SourceSummary:
        summary = truncate_text(
            clean_text(payload.get("summary", "")),
            max_length=500,
            suffix="",
        ) or truncate_text(
            clean_text(source.abstract or ""), max_length=300
        )

        raw_difficulty = str(
            payload.get("difficulty", "")
        ).strip().lower()

        if source.difficulty is not None:
            difficulty = source.difficulty
        else:
            try:
                difficulty = Difficulty(raw_difficulty)
            except Exception:
                difficulty = Difficulty.INTERMEDIATE

        raw_topics = payload.get("key_topics", [])
        key_topics: list[str] = []

        if isinstance(raw_topics, list):
            key_topics = [
                clean_text(item)
                for item in raw_topics
                if isinstance(item, str)
            ]

        key_topics = [item for item in key_topics if item][:6]

        why_useful = truncate_text(
            clean_text(payload.get("why_useful", "")),
            max_length=300,
            suffix="",
        ) or self._build_why_useful(source)

        return SourceSummary(
            source_id=source.source_id,
            title=source.title,
            summary=summary,
            difficulty=difficulty,
            key_topics=key_topics,
            why_useful=why_useful,
            estimated_reading_minutes=self._estimate_reading_minutes(
                source
            ),
            llm_generated=True,
        )

    def _extract_key_topics(self, source: Source) -> list[str]:
        topics: list[str] = []
        metadata = (
            source.metadata
            if isinstance(source.metadata, dict)
            else {}
        )

        for key in ("topics", "tags", "categories", "pipeline_tag"):
            value = metadata.get(key)

            if isinstance(value, list):
                topics.extend(
                    clean_text(item) for item in value if item
                )
            elif isinstance(value, str) and value:
                topics.append(clean_text(value))

        seen: set[str] = set()
        unique_topics: list[str] = []

        for topic in topics:
            lowered = topic.lower()

            if lowered not in seen:
                seen.add(lowered)
                unique_topics.append(topic)

        return unique_topics[:6]

    def _build_why_useful(self, source: Source) -> str:
        source_type = self._enum_value(source.source_type)
        platform = self._enum_value(source.platform)

        return (
            f"A {source_type} from {platform} relevant to your "
            f"learning goal."
        )

    def _estimate_reading_minutes(self, source: Source) -> int:
        source_type = self._enum_value(source.source_type)
        abstract = clean_text(source.abstract or "")
        word_count = len(abstract.split())

        base = {
            "research_paper": 90,
            "repository": 120,
            "course": 180,
            "video": 45,
            "documentation": 30,
            "dataset": 60,
            "model": 60,
            "blog": 20,
        }.get(source_type, 30)

        if word_count > 300:
            base += 15

        return base

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value))