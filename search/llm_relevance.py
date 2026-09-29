from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from core import constants
from core.models import Source
from search.query_profile import QueryProfile
from utils.logger import get_logger, get_trace_logger
from utils.text import clean_text

_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "prompts"
    / "relevance_filter_prompt.txt"
)

_MARKER_CONCEPT = "<<<CONCEPT>>>"
_MARKER_QUERY = "<<<QUERY>>>"
_MARKER_LEVEL = "<<<LEVEL>>>"
_MARKER_SOURCES = "<<<SOURCES>>>"

_VALID_RELEVANCE = frozenset({"yes", "no", "maybe"})
_VALID_DIFFICULTY = frozenset({"beginner", "intermediate", "advanced"})

_JUNK_TITLE_SUFFIXES = (
    "/references",
    "/reference",
    "/bibliography",
    "/further reading",
    "/further-reading",
    "/see also",
    "/see-also",
    "/external links",
    "/external-links",
    "/glossary",
    "/index",
    " (disambiguation)",
    " (disambiguation page)",
)

_JUNK_TITLE_SUBSTRINGS = (
    "list of references",
    "bibliography of",
    "references for",
    "citation list",
)

_DEFAULT_PROMPT = """You are a strict relevance judge for a research assistant.

Concept: <<<CONCEPT>>>
Original query: <<<QUERY>>>
Expected level: <<<LEVEL>>>

For each source, return one decision.
relevant: yes | no | maybe
difficulty: beginner | intermediate | advanced
reason: one short sentence.

Rules:
- Judge against the concept, not shared words.
- Reject sources that use the concept's words but belong to a different domain.
- Accept sources that teach, explain, or research the concept.
- Reject pure bibliography pages, reference lists, citation collections, glossaries, and index pages.
- Reject disambiguation pages, stub pages, and pages whose title is only a section anchor.
- Reject list-of-X pages that contain no explanatory prose beyond names and links.
- Reject the source if its abstract is empty and its title is generic.
- A source whose title or abstract is primarily about a different subject is NOT relevant.
- If you are not confident a source is relevant, mark it "no".
- Return only JSON: {"decisions": [{"source_id": "...", "relevant": "yes", "difficulty": "beginner", "reason": "..."}]}

Sources:
<<<SOURCES>>>
"""


class LLMRelevanceFilter:
    def __init__(
        self,
        enabled: bool = False,
        batch_size: int = 10,
        max_sources: int = 40,
        min_sources: int = 5,
        model_limit: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        self._enabled = bool(enabled)
        self._batch_size = max(1, int(batch_size))
        self._max_sources = max(1, int(max_sources))
        self._min_sources = max(0, int(min_sources))
        self._model_limit = max(1, int(model_limit))
        self._temperature = max(0.0, min(1.0, float(temperature)))
        self._max_tokens = max(256, int(max_tokens))
        self._logger = get_logger("search.llm_relevance")
        self._trace = get_trace_logger()
        self._prompt_template = self._load_prompt()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _load_prompt(self) -> str:
        try:
            if _PROMPT_PATH.exists():
                content = _PROMPT_PATH.read_text(encoding="utf-8")

                if content.strip():
                    return content
        except Exception:
            pass

        return _DEFAULT_PROMPT

    def _resolve_mode(self, mode: str) -> str:
        text = str(mode or "").strip().lower()

        if text in constants.VALID_MODES:
            return text

        return constants.DEFAULT_MODE

    def _mode_allows_filter(self, mode: str) -> bool:
        return self._resolve_mode(mode) == constants.MODE_DEEP

    async def filter_sources(
        self,
        sources: list[Source],
        profile: QueryProfile,
        mode: str = "",
    ) -> list[Source]:
        if not self._enabled:
            return sources

        if not sources:
            return sources

        resolved_mode = self._resolve_mode(mode)

        if not self._mode_allows_filter(resolved_mode):
            self._trace.emit(
                "llm_relevance_skipped",
                reason="mode_not_deep",
                mode=resolved_mode,
                source_count=len(sources),
            )
            return sources

        if len(sources) < self._min_sources:
            self._trace.emit(
                "llm_relevance_skipped",
                reason="too_few_sources",
                source_count=len(sources),
                min_required=self._min_sources,
                mode=resolved_mode,
            )
            return sources

        try:
            from core.schemas import LLMMessageSchema, LLMRequestSchema
            from llm.guardrails import validate_llm_request
            from llm.parser import parse_json_object_response
            from llm.provider import LLMProviderManager
            from llm.router import get_fast_model_references
        except Exception as exc:
            self._logger.warning(
                f"LLM relevance layer unavailable: {exc}"
            )
            return sources

        model_references = get_fast_model_references(
            limit=self._model_limit
        )

        if not model_references:
            self._trace.emit(
                "llm_relevance_skipped",
                reason="no_models_available",
                mode=resolved_mode,
            )
            return sources

        candidates: list[Source] = []
        dropped_ids: set[str] = set()
        passed_junk_filter: list[Source] = []

        for source in sources[: self._max_sources]:
            if self._is_obvious_junk(source):
                dropped_ids.add(source.source_id)
                self._trace.emit(
                    "llm_relevance_prefilter_drop",
                    source_id=source.source_id,
                    title=str(getattr(source, "title", ""))[:120],
                    reason="junk_title_pattern",
                )
                continue

            passed_junk_filter.append(source)

            if not self._shares_concept_tokens(source, profile):
                dropped_ids.add(source.source_id)
                self._trace.emit(
                    "llm_relevance_prefilter_drop",
                    source_id=source.source_id,
                    title=str(getattr(source, "title", ""))[:120],
                    reason="no_concept_token_overlap",
                )
                continue

            candidates.append(source)

        if not candidates and passed_junk_filter:
            rescue_count = min(3, len(passed_junk_filter))

            for source in passed_junk_filter[:rescue_count]:
                dropped_ids.discard(source.source_id)
                candidates.append(source)

            self._trace.emit(
                "llm_relevance_prefilter_rescue",
                rescued=rescue_count,
                total_passed_junk_filter=len(passed_junk_filter),
                concept_tokens=list(profile.concept_tokens or [])[:10],
            )

        batches = [
            candidates[index : index + self._batch_size]
            for index in range(0, len(candidates), self._batch_size)
        ]

        batch_reports: list[dict[str, Any]] = []

        if not candidates:
            self._trace.emit(
                "llm_relevance_completed",
                input_count=len(sources),
                output_count=0,
                dropped_count=len(dropped_ids),
                batches=0,
                model_count=0,
                mode=resolved_mode,
                note="all_dropped_by_prefilter",
            )
            return []

        async with LLMProviderManager() as manager:
            for batch_index, batch in enumerate(batches):
                report = await self._judge_batch(
                    batch=batch,
                    batch_index=batch_index,
                    profile=profile,
                    manager=manager,
                    model_references=model_references,
                    LLMMessageSchema=LLMMessageSchema,
                    LLMRequestSchema=LLMRequestSchema,
                    validate_llm_request=validate_llm_request,
                    parse_json_object_response=parse_json_object_response,
                )

                batch_reports.append(report)
                dropped_ids.update(report.get("drop", set()))

        if not dropped_ids:
            self._trace.emit(
                "llm_relevance_completed",
                input_count=len(sources),
                output_count=len(sources),
                dropped_count=0,
                batches=len(batches),
                model_count=len(model_references),
                mode=resolved_mode,
                note="no_drops",
            )
            return sources

        result = [
            source
            for source in sources
            if source.source_id not in dropped_ids
        ]

        if not result:
            self._trace.emit(
                "llm_relevance_all_dropped",
                input_count=len(sources),
                dropped_count=len(dropped_ids),
                action="passing_original_sources_through",
                batch_reports=batch_reports,
            )
            return sources

        rescue_target = self._rescue_target(resolved_mode)

        if len(result) < rescue_target:
            rescued = self._rescue_sources(
                sources=sources,
                kept_ids={s.source_id for s in result},
                target=rescue_target,
            )

            if rescued:
                result = result + rescued

                self._trace.emit(
                    "llm_relevance_rescued",
                    input_count=len(sources),
                    output_count=len(result),
                    target=rescue_target,
                    rescued=len(rescued),
                    mode=resolved_mode,
                )

        self._trace.emit(
            "llm_relevance_completed",
            input_count=len(sources),
            output_count=len(result),
            dropped_count=len(dropped_ids),
            batches=len(batches),
            model_count=len(model_references),
            mode=resolved_mode,
            batch_reports=batch_reports,
        )

        return result

    @staticmethod
    def _shares_concept_tokens(
        source: Source,
        profile: QueryProfile,
    ) -> bool:
        try:
            from search.query_tokens import tokens_match
        except Exception:
            return True

        concept_tokens = list(profile.concept_tokens or [])

        if not concept_tokens:
            return True

        title = str(getattr(source, "title", "") or "").lower()
        abstract = str(getattr(source, "abstract", "") or "").lower()

        if not title and not abstract:
            return True

        haystack_tokens = set(
            token
            for token in (
                title.replace("-", " ")
                .replace("/", " ")
                .replace("_", " ")
                .split()
            )
            if token
        )

        if abstract:
            haystack_tokens.update(
                token for token in abstract.split() if token
            )

        for concept in concept_tokens:
            concept_text = str(concept or "").strip().lower()

            if not concept_text or len(concept_text) < 3:
                continue

            for candidate in haystack_tokens:
                if tokens_match(concept_text, candidate):
                    return True

        return False

    @staticmethod
    def _is_obvious_junk(source: Source) -> bool:
        title = clean_text(getattr(source, "title", "") or "").lower()

        if not title:
            return False

        if any(
            title.endswith(suffix) for suffix in _JUNK_TITLE_SUFFIXES
        ):
            return True

        if any(
            substring in title for substring in _JUNK_TITLE_SUBSTRINGS
        ):
            return True

        return False
    @staticmethod
    def _rescue_target(mode: str) -> int:
        if mode == constants.MODE_FAST:
            return 5

        if mode == constants.MODE_BALANCED:
            return 8

        return 10

    def _rescue_sources(
        self,
        sources: list[Source],
        kept_ids: set[str],
        target: int,
    ) -> list[Source]:
        needed = max(0, target - len(kept_ids))

        if needed <= 0:
            return []

        candidates: list[Source] = []

        for source in sources:
            source_id = str(getattr(source, "source_id", "") or "")

            if not source_id or source_id in kept_ids:
                continue

            if self._is_obvious_junk(source):
                continue

            candidates.append(source)

        candidates.sort(
            key=self._rescue_score,
            reverse=True,
        )

        return candidates[:needed]

    @staticmethod
    def _rescue_score(source: Source) -> float:
        score = 0.0

        abstract = str(getattr(source, "abstract", "") or "")
        title = str(getattr(source, "title", "") or "")

        score += min(len(abstract) / 400.0, 2.0)
        score += min(len(title) / 40.0, 1.0)

        citations = getattr(source, "citation_count", None)

        if isinstance(citations, int):
            score += min(citations / 500.0, 2.0)

        year = getattr(source, "year", None)

        if isinstance(year, int) and year >= 2015:
            score += 0.5

        return score
    async def _judge_batch(
        self,
        batch: list[Source],
        batch_index: int,
        profile: QueryProfile,
        manager: Any,
        model_references: list[Any],
        LLMMessageSchema: Any,
        LLMRequestSchema: Any,
        validate_llm_request: Any,
        parse_json_object_response: Any,
    ) -> dict[str, Any]:
        report: dict[str, Any] = {
            "batch_index": batch_index,
            "sources_in": len(batch),
            "sources_kept": len(batch),
            "sources_dropped": 0,
            "model_used": None,
            "latency_ms": 0.0,
            "error": None,
            "keep": set(),
            "drop": set(),
        }

        prompt = self._build_prompt(batch, profile)

        messages = [
            LLMMessageSchema(
                role="system",
                content="You are a relevance judge. Return only JSON.",
            ),
            LLMMessageSchema(role="user", content=prompt),
        ]

        batch_started = time.perf_counter()

        for model_reference in model_references:
            model_label = (
                f"{getattr(model_reference.provider, 'value', model_reference.provider)}"
                f"/{getattr(model_reference, 'model_id', '?')}"
            )

            try:
                request = LLMRequestSchema(
                    provider=model_reference.provider,
                    model=model_reference.model_id,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    response_format="json_object",
                )

                request = validate_llm_request(request)
                response = await manager.complete(request)

                if response is None:
                    continue

                payload = parse_json_object_response(response.content)

                if not isinstance(payload, dict):
                    continue

                decisions = payload.get("decisions", [])

                if not isinstance(decisions, list) or not decisions:
                    continue

                keep: set[str] = set()
                drop: set[str] = set()

                valid_ids = {source.source_id for source in batch}

                for decision in decisions:
                    if not isinstance(decision, dict):
                        continue

                    source_id = str(
                        decision.get("source_id", "")
                    ).strip()

                    if not source_id or source_id not in valid_ids:
                        continue

                    relevant = (
                        str(decision.get("relevant", ""))
                        .strip()
                        .lower()
                    )

                    if relevant not in _VALID_RELEVANCE:
                        relevant = "maybe"

                    difficulty = (
                        str(decision.get("difficulty", ""))
                        .strip()
                        .lower()
                    )

                    if difficulty not in _VALID_DIFFICULTY:
                        difficulty = ""

                    reason = str(
                        decision.get("reason", "")
                    ).strip()[:240]

                    if relevant == "no":
                        drop.add(source_id)
                    else:
                        keep.add(source_id)

                    self._trace.emit(
                        "llm_relevance_decision",
                        batch_index=batch_index,
                        source_id=source_id,
                        relevant=relevant,
                        difficulty=difficulty,
                        reason=reason,
                    )

                if keep or drop:
                    latency_ms = (
                        time.perf_counter() - batch_started
                    ) * 1000

                    report["model_used"] = model_label
                    report["latency_ms"] = round(latency_ms, 2)
                    report["keep"] = keep
                    report["drop"] = drop
                    report["sources_kept"] = len(batch) - len(drop)
                    report["sources_dropped"] = len(drop)

                    self._trace.emit(
                        "llm_relevance_batch",
                        batch_index=batch_index,
                        model_used=model_label,
                        latency_ms=round(latency_ms, 2),
                        sources_in=len(batch),
                        sources_kept=len(batch) - len(drop),
                        sources_dropped=len(drop),
                    )

                    return report
            except Exception as exc:
                self._logger.warning(
                    f"LLM relevance batch {batch_index} failed for "
                    f"model {model_label}: {exc}"
                )
                report["error"] = str(exc)
                continue

        latency_ms = (time.perf_counter() - batch_started) * 1000
        report["latency_ms"] = round(latency_ms, 2)

        self._trace.emit(
            "llm_relevance_batch",
            batch_index=batch_index,
            model_used=None,
            latency_ms=round(latency_ms, 2),
            sources_in=len(batch),
            sources_kept=len(batch),
            sources_dropped=0,
            note="all_models_failed_batch_kept",
        )

        return report

    def _build_prompt(
        self,
        batch: list[Source],
        profile: QueryProfile,
    ) -> str:
        payload = []

        for source in batch:
            entry = self._build_source_entry(source)
            payload.append(entry)

        concept = profile.concept_phrase or profile.original_query or ""
        query = profile.original_query or ""
        level = profile.level_profile or "mixed"
        sources_json = json.dumps(
            payload, ensure_ascii=False, indent=2
        )

        prompt = self._prompt_template
        prompt = prompt.replace(_MARKER_CONCEPT, concept)
        prompt = prompt.replace(_MARKER_QUERY, query)
        prompt = prompt.replace(_MARKER_LEVEL, level)
        prompt = prompt.replace(_MARKER_SOURCES, sources_json)

        return prompt

    def _build_source_entry(self, source: Source) -> dict[str, Any]:
        metadata = getattr(source, "metadata", None)

        if not isinstance(metadata, dict):
            metadata = {}

        entry: dict[str, Any] = {
            "source_id": source.source_id,
            "title": source.title,
            "abstract": (source.abstract or "")[:800],
            "source_type": getattr(
                source.source_type,
                "value",
                str(source.source_type),
            ),
            "platform": getattr(
                source.platform,
                "value",
                str(source.platform),
            ),
            "difficulty": getattr(
                getattr(source, "difficulty", None),
                "value",
                None,
            ),
        }

        year = getattr(source, "year", None)

        if isinstance(year, int):
            entry["year"] = year

        citation_count = getattr(source, "citation_count", None)

        if isinstance(citation_count, int) and citation_count >= 0:
            entry["citation_count"] = citation_count

        stars = metadata.get("stars")

        if isinstance(stars, int) and stars >= 0:
            entry["stars"] = stars

        downloads = metadata.get("downloads")

        if isinstance(downloads, int) and downloads >= 0:
            entry["downloads"] = downloads

        likes = metadata.get("likes")

        if isinstance(likes, int) and likes >= 0:
            entry["likes"] = likes

        venue = metadata.get("venue")

        if isinstance(venue, str) and venue.strip():
            entry["venue"] = venue.strip()

        journal = metadata.get("journal")

        if isinstance(journal, str) and journal.strip():
            entry.setdefault("venue", journal.strip())

        container_title = metadata.get("container_title")

        if isinstance(container_title, str) and container_title.strip():
            entry.setdefault("venue", container_title.strip())

        return entry