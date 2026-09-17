from __future__ import annotations

import asyncio
import random
from typing import Any

from pydantic import Field

from core import constants
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
        max_sources: int = 100,
        max_concurrency: int = 3,
        batch_size: int = 4,
        batch_pause_seconds: float = 2.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 4.0,
    ) -> None:
        self._use_llm = use_llm
        self._max_sources = max(1, max_sources)
        self._max_concurrency = max(1, max_concurrency)
        self._batch_size = max(1, batch_size)
        self._batch_pause_seconds = max(0.0, batch_pause_seconds)
        self._max_retries = max(1, max_retries)
        self._retry_backoff_seconds = max(0.5, retry_backoff_seconds)
        self._logger = get_logger("ranking.source_summarizer")
        self._trace = get_trace_logger()

    async def summarize_sources(self, sources: list[Any]) -> list[SourceSummary]:
        valid_sources = [source for source in sources if isinstance(source, Source)]
        valid_sources = valid_sources[: self._max_sources]

        if not valid_sources:
            return []

        if not self._use_llm:
            return [self._heuristic_summary(source) for source in valid_sources]

        self._trace.emit(
            "summarizer_started",
            total_sources=len(valid_sources),
            batch_size=self._batch_size,
            max_concurrency=self._max_concurrency,
        )

        semaphore = asyncio.Semaphore(self._max_concurrency)
        results: list[SourceSummary] = []

        async def summarize_with_limit(
            manager: LLMProviderManager,
            source: Source,
        ) -> SourceSummary:
            async with semaphore:
                return await self._summarize_with_retries(manager, source)

        async with LLMProviderManager() as manager:
            for batch_start in range(0, len(valid_sources), self._batch_size):
                batch = valid_sources[batch_start: batch_start + self._batch_size]

                batch_results = await asyncio.gather(
                    *(summarize_with_limit(manager, source) for source in batch)
                )
                results.extend(batch_results)

                llm_count = sum(1 for item in batch_results if item.llm_generated)
                self._trace.emit(
                    "summarizer_batch_completed",
                    batch_index=(batch_start // self._batch_size) + 1,
                    batch_size=len(batch),
                    llm_summaries=llm_count,
                    heuristic_summaries=len(batch) - llm_count,
                    completed=len(results),
                    total=len(valid_sources),
                )

                if batch_start + self._batch_size < len(valid_sources):
                    await asyncio.sleep(self._batch_pause_seconds)

        llm_total = sum(1 for item in results if item.llm_generated)
        self._trace.emit(
            "summarizer_completed",
            total_summaries=len(results),
            llm_summaries=llm_total,
            heuristic_summaries=len(results) - llm_total,
        )

        return results

    async def summarize_source(self, source: Source) -> SourceSummary:
        if not self._use_llm:
            return self._heuristic_summary(source)

        async with LLMProviderManager() as manager:
            return await self._summarize_with_retries(manager, source)

    async def _summarize_with_retries(
        self,
        manager: LLMProviderManager,
        source: Source,
    ) -> SourceSummary:
        last_error = ""

        for attempt in range(1, self._max_retries + 1):
            try:
                return await self._llm_summary(manager, source)
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
                    self._retry_backoff_seconds * (2 ** (attempt - 1)),
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

    def _rate_limit_delay(self, exc: ProviderRateLimitError, attempt: int) -> float:
        if exc.retry_after is not None:
            return max(float(exc.retry_after), 1.0) + random.uniform(0.0, 1.0)
        base = self._retry_backoff_seconds * (2 ** attempt)
        return min(base, 30.0) + random.uniform(0.0, 1.0)

    async def _llm_summary(
        self,
        manager: LLMProviderManager,
        source: Source,
    ) -> SourceSummary:
        model_references = get_fast_model_references(limit=3)
        if not model_references:
            raise RuntimeError("No usable fast models for summarization")

        serialized_source = {
            "title": source.title,
            "abstract": truncate_text(source.abstract or "", max_length=1500),
            "platform": self._enum_value(source.platform),
            "source_type": self._enum_value(source.source_type),
            "year": source.year,
            "citation_count": source.citation_count,
        }
        prompt = build_source_summary_prompt(serialized_source)
        messages = [
            LLMMessageSchema(
                role="system",
                content="You are a concise learning-resource summarizer. Return only valid JSON.",
            ),
            LLMMessageSchema(role="user", content=prompt),
        ]

        last_error: Exception | None = None
        for model_reference in model_references:
            try:
                request = LLMRequestSchema(
                    provider=model_reference.provider,
                    model=model_reference.model_id,
                    messages=messages,
                    temperature=constants.DEFAULT_TEMPERATURE,
                    max_tokens=512,
                    response_format="json_object",
                )
                request = validate_llm_request(request)
                response = await manager.complete(request)
                response = validate_llm_response(response)
                payload = parse_json_object_response(response.content)
                summary = self._parse_payload(source, payload)
                self._trace.emit(
                    "summary_llm_success",
                    source_id=source.source_id,
                    provider=model_reference.provider.value,
                    model=model_reference.model_id,
                )
                return summary
            except Exception as exc:
                last_error = exc
                self._trace.emit(
                    "summary_model_failed",
                    source_id=source.source_id,
                    provider=model_reference.provider.value,
                    model=model_reference.model_id,
                    error=str(exc),
                )
                continue

        if last_error is not None:
            raise last_error
        raise RuntimeError("Summarization failed with no error captured")

    def _heuristic_summary(self, source: Source) -> SourceSummary:
        abstract = clean_text(source.abstract or "")
        if abstract:
            summary = truncate_text(abstract, max_length=300, suffix="...")
        else:
            summary = f"{source.title} ({self._enum_value(source.source_type)} resource)."

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

    def _parse_payload(self, source: Source, payload: dict[str, Any]) -> SourceSummary:
        summary = truncate_text(
            clean_text(payload.get("summary", "")),
            max_length=500,
            suffix="",
        ) or truncate_text(clean_text(source.abstract or ""), max_length=300)

        raw_difficulty = str(payload.get("difficulty", "")).strip().lower()

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
            key_topics = [clean_text(item) for item in raw_topics if isinstance(item, str)]
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
            estimated_reading_minutes=self._estimate_reading_minutes(source),
            llm_generated=True,
        )

    def _extract_key_topics(self, source: Source) -> list[str]:
        topics: list[str] = []
        metadata = source.metadata if isinstance(source.metadata, dict) else {}
        for key in ("topics", "tags", "categories", "pipeline_tag"):
            value = metadata.get(key)
            if isinstance(value, list):
                topics.extend(clean_text(item) for item in value if item)
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
        return f"A {source_type} from {platform} relevant to your learning goal."

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