from __future__ import annotations

from typing import Any

from core import constants
from core.models import Difficulty, LearningPath, LearningStep, RankedSource
from core.schemas import LLMMessageSchema, LLMRequestSchema, SearchQuerySchema
from llm.guardrails import validate_llm_request, validate_llm_response
from llm.parser import (
    parse_json_object_response,
    parse_truncated_json_object,
)
from llm.prompts import (
    build_learning_path_prompt,
    system_learning_path_builder,
)
from llm.provider import LLMProviderManager
from llm.router import (
    get_fast_model_references,
    get_strong_model_references,
)
from ranking.difficulty_classifier import DifficultyClassifier
from utils.logger import get_logger, get_trace_logger
from utils.text import clean_text, truncate_text


PADDING_MARKER = "[padded]"
MINIMUM_STEPS = constants.PATH_MINIMUM_STEPS
MAX_TARGET_STEPS = constants.PATH_MAX_TARGET_STEPS
PROMPT_TIER = "compact"
TOKEN_FLOOR = constants.PATH_TOKEN_FLOOR
TOKEN_CEILING = constants.PATH_TOKEN_CEILING
PER_STEP_TOKEN_BUDGET = constants.PATH_PER_STEP_TOKENS
TOKEN_WRAPPER_BUDGET = constants.PATH_WRAPPER_TOKENS
TRUNCATION_TOKEN_RATIO = constants.PATH_TRUNCATION_RATIO
MIN_STEP_RESOURCE_SCORE = constants.PATH_MIN_STEP_RESOURCE_SCORE

_ADVANCED_STEP_MARKERS: tuple[str, ...] = (
    "advanced",
    "research",
    "expert",
    "deep dive",
    "in depth",
    "in-depth",
)

_BEGINNER_STEP_MARKERS: tuple[str, ...] = (
    "foundation",
    "foundations",
    "basic",
    "basics",
    "intro",
    "introduction",
    "introductory",
    "fundamentals",
    "getting started",
    "primer",
    "overview",
)

_INTERMEDIATE_STEP_MARKERS: tuple[str, ...] = (
    "core",
    "intermediate",
    "applied",
    "practice",
    "synthesis",
    "application",
    "techniques",
    "methods",
)


class LearningPathBuilder:
    def __init__(
        self,
        use_llm: bool = True,
        classifier: Any | None = None,
        max_resources_per_step: int | None = None,
    ) -> None:
        self._use_llm = bool(use_llm)
        self._classifier = classifier
        self._max_resources_per_step = max_resources_per_step
        self._logger = get_logger("ranking.learning_path_builder")
        self._trace = get_trace_logger()

        if self._classifier is None:
            try:
                self._classifier = DifficultyClassifier()
            except Exception:
                self._classifier = None

    async def build(
        self,
        ranked_sources: list[Any],
        query: SearchQuerySchema | str | dict[str, Any],
        mode: str = "",
    ) -> LearningPath:
        resolved_topic = self._extract_corrected_topic(query)
        query_schema = self._coerce_query(query, resolved_topic)

        valid_ranked = [
            item
            for item in ranked_sources
            if isinstance(item, RankedSource)
        ]

        empty_path = LearningPath(
            topic=query_schema.topic,
            level=query_schema.level,
            goal=query_schema.goal,
            steps=[],
        )

        if not valid_ranked:
            self._trace.emit(
                "learning_path_strategy",
                strategy="none",
                total_steps=0,
                reason="no_ranked_sources",
            )
            return empty_path

        resolved_mode = self._resolve_mode(mode)

        classified_ranked = self._classify_ranked(
            valid_ranked, query_schema
        )

        source_count = len(classified_ranked)
        minimum_steps = self._compute_minimum_steps(source_count)

        strategy = self._strategy_for_mode(resolved_mode)

        ai_path: LearningPath | None = None

        if self._use_llm and strategy != "heuristic":
            try:
                ai_path = await self._build_via_single_model(
                    classified_ranked,
                    query_schema,
                    strategy,
                )
            except Exception as exc:
                self._logger.warning(
                    f"LLM learning path generation failed: {exc}"
                )
                self._trace.emit(
                    "learning_path_llm_failed", error=str(exc)
                )

        if ai_path is not None and ai_path.steps:
            if (
                len(ai_path.steps) >= minimum_steps
                or source_count < minimum_steps
            ):
                aligned = self._align_path(ai_path, query_schema)
                padded = self._path_contains_padding(aligned)

                self._trace.emit(
                    "learning_path_strategy",
                    strategy=(
                        "ai_padded" if padded else "ai"
                    ),
                    total_steps=len(aligned.steps),
                    padded=padded,
                    source_count=source_count,
                    mode=resolved_mode,
                )

                return aligned

            self._trace.emit(
                "learning_path_llm_insufficient",
                llm_steps=len(ai_path.steps),
                minimum_steps=minimum_steps,
                source_count=source_count,
            )

        heuristic_path = self._heuristic_path(
            classified_ranked, query_schema
        )
        aligned_heuristic = self._align_path(
            heuristic_path, query_schema
        )

        if (
            len(aligned_heuristic.steps) >= minimum_steps
            or source_count < minimum_steps
        ):
            self._trace.emit(
                "learning_path_strategy",
                strategy="heuristic",
                total_steps=len(aligned_heuristic.steps),
                padded=False,
                source_count=source_count,
                mode=resolved_mode,
            )
            return aligned_heuristic

        used_ids: set[str] = set()

        for step in aligned_heuristic.steps:
            for resource in step.resources:
                source = getattr(resource, "source", None)

                if source is not None:
                    used_ids.add(source.source_id)

        padded_steps = self._pad_steps(
            list(aligned_heuristic.steps),
            classified_ranked,
            used_ids,
            query_schema,
        )

        final_path = aligned_heuristic.model_copy(
            update={"steps": padded_steps}
        )
        aligned = self._align_path(final_path, query_schema)

        self._trace.emit(
            "learning_path_strategy",
            strategy="padded",
            total_steps=len(aligned.steps),
            padded=True,
            source_count=source_count,
            mode=resolved_mode,
        )

        return aligned

    def _resolve_mode(self, mode: str) -> str:
        text = str(mode or "").strip().lower()

        if text in constants.VALID_MODES:
            return text

        return constants.DEFAULT_MODE

    def _strategy_for_mode(self, mode: str) -> str:
        if mode == constants.MODE_FAST:
            return "heuristic"

        if mode == constants.MODE_BALANCED:
            return "fast"

        return "strong"

    async def _build_via_single_model(
        self,
        ranked_sources: list[RankedSource],
        query_schema: SearchQuerySchema,
        strategy: str,
    ) -> LearningPath | None:
        model_reference = self._select_model(strategy)

        if model_reference is None:
            self._trace.emit(
                "learning_path_llm_no_models",
                strategy=strategy,
                reason="no_model_reference_available",
            )
            return None

        source_count = len(ranked_sources)
        max_tokens = self._compute_dynamic_max_tokens(source_count)

        async with LLMProviderManager() as manager:
            return await self._try_single(
                manager=manager,
                model_reference=model_reference,
                strategy=strategy,
                max_tokens=max_tokens,
                ranked_sources=ranked_sources,
                query_schema=query_schema,
            )

    def _select_model(self, strategy: str) -> Any:
        try:
            if strategy == "fast":
                fast_refs = get_fast_model_references(limit=1)

                if fast_refs:
                    return fast_refs[0]

                strong_refs = get_strong_model_references(limit=1)
                return strong_refs[0] if strong_refs else None

            strong_refs = get_strong_model_references(limit=1)

            if strong_refs:
                return strong_refs[0]

            fast_refs = get_fast_model_references(limit=1)
            return fast_refs[0] if fast_refs else None
        except Exception:
            return None

    def _compute_dynamic_max_tokens(
        self,
        source_count: int,
    ) -> int:
        step_target = max(
            MINIMUM_STEPS, min(MAX_TARGET_STEPS, source_count)
        )
        raw = (
            TOKEN_WRAPPER_BUDGET
            + step_target * PER_STEP_TOKEN_BUDGET
        )
        return max(TOKEN_FLOOR, min(TOKEN_CEILING, raw))

    def _timeout_for_strategy(self, strategy: str) -> float:
        if strategy == "fast":
            return 20.0

        if strategy == "strong":
            return 30.0

        return 25.0

    async def _try_single(
        self,
        manager: LLMProviderManager,
        model_reference: Any,
        strategy: str,
        max_tokens: int,
        ranked_sources: list[RankedSource],
        query_schema: SearchQuerySchema,
    ) -> LearningPath | None:
        provider_name = self._enum_value(
            getattr(model_reference, "provider", "")
        )
        model_id = clean_text(
            getattr(model_reference, "model_id", "")
        )

        if not model_id:
            model_id = clean_text(
                getattr(model_reference, "model", "")
            )

        if not model_id:
            return None

        serialized = self._serialize_ranked_sources(
            ranked_sources, tier=PROMPT_TIER
        )
        prompt = build_learning_path_prompt(
            topic=query_schema.topic,
            goal=query_schema.goal,
            level=query_schema.level,
            ranked_sources=serialized,
            tier=PROMPT_TIER,
        )

        timeout_seconds = self._timeout_for_strategy(strategy)

        self._trace.emit(
            "learning_path_llm_attempt",
            strategy=strategy,
            provider=provider_name,
            model=model_id,
            max_tokens=max_tokens,
            prompt_chars=len(prompt),
            timeout_seconds=timeout_seconds,
        )

        try:
            messages = [
                LLMMessageSchema(
                    role="system",
                    content=system_learning_path_builder(),
                ),
                LLMMessageSchema(role="user", content=prompt),
            ]

            request = LLMRequestSchema(
                provider=model_reference.provider,
                model=model_id,
                messages=messages,
                temperature=constants.DEFAULT_TEMPERATURE,
                max_tokens=max_tokens,
                response_format="json_object",
            )

            request = validate_llm_request(request)

            inner_task = asyncio.create_task(
                manager.complete(request)
            )

            try:
                response = await asyncio.wait_for(
                    inner_task,
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                inner_task.cancel()

                try:
                    await inner_task
                except Exception:
                    pass

                self._trace.emit(
                    "learning_path_llm_hard_timeout",
                    strategy=strategy,
                    provider=provider_name,
                    model=model_id,
                    timeout_seconds=timeout_seconds,
                )
                return None
            except Exception as exc:
                if not inner_task.done():
                    inner_task.cancel()

                    try:
                        await inner_task
                    except Exception:
                        pass

                self._trace.emit(
                    "learning_path_llm_error",
                    strategy=strategy,
                    provider=provider_name,
                    model=model_id,
                    error=str(exc),
                )
                return None

            if response is None:
                self._trace.emit(
                    "learning_path_llm_no_response",
                    strategy=strategy,
                    provider=provider_name,
                    model=model_id,
                )
                return None

            response = validate_llm_response(response)

            output_tokens = int(
                getattr(response, "tokens_used", 0) or 0
            )
            content = response.content or ""
            truncated = self._is_likely_truncated(
                content, output_tokens, max_tokens
            )

            self._trace.emit(
                "learning_path_llm_response",
                strategy=strategy,
                provider=provider_name,
                model=model_id,
                content_length=len(content),
                output_tokens=output_tokens,
                max_tokens=max_tokens,
                truncated=truncated,
            )

            payload: dict[str, Any] | None = None

            if truncated:
                self._trace.emit(
                    "learning_path_llm_truncated",
                    strategy=strategy,
                    provider=provider_name,
                    model=model_id,
                    output_tokens=output_tokens,
                    max_tokens=max_tokens,
                )

                payload = parse_truncated_json_object(content)

                if payload is not None:
                    self._trace.emit(
                        "learning_path_llm_repaired",
                        strategy=strategy,
                        provider=provider_name,
                        model=model_id,
                    )

            if payload is None:
                try:
                    payload = parse_json_object_response(content)
                except Exception:
                    payload = None

            if payload is None:
                self._trace.emit(
                    "learning_path_llm_unparseable",
                    strategy=strategy,
                    provider=provider_name,
                    model=model_id,
                )
                return None

            payload = self._unwrap_payload(payload)
            parsed_path = self._parse_llm_path(
                payload, ranked_sources, query_schema
            )

            self._trace.emit(
                "learning_path_llm_parsed",
                strategy=strategy,
                provider=provider_name,
                model=model_id,
                parsed_steps=len(parsed_path.steps),
            )

            if parsed_path.steps:
                return parsed_path

            return None
        except Exception as exc:
            self._trace.emit(
                "learning_path_llm_error",
                strategy=strategy,
                provider=provider_name,
                model=model_id,
                error=str(exc),
            )
            return None

    def _is_likely_truncated(
        self,
        content: str,
        output_tokens: int,
        max_tokens: int,
    ) -> bool:
        if not content:
            return False

        if max_tokens > 0:
            threshold = max_tokens * TRUNCATION_TOKEN_RATIO

            if output_tokens >= threshold:
                return True

        stripped = content.rstrip()

        if stripped.endswith((",", ":", "[", "{")):
            return True

        opens = stripped.count("{") + stripped.count("[")
        closes = stripped.count("}") + stripped.count("]")

        if opens > closes:
            return True

        return False

    def _path_contains_padding(
        self,
        learning_path: LearningPath,
    ) -> bool:
        for step in learning_path.steps:
            for resource in step.resources:
                reason = str(
                    getattr(resource, "reason", "") or ""
                )

                if reason.startswith(PADDING_MARKER):
                    return True

        return False

    def _extract_corrected_topic(self, query: Any) -> str:
        if isinstance(query, str):
            return clean_text(query)

        if isinstance(query, dict):
            corrected = clean_text(
                query.get("corrected_topic", "")
            )

            if corrected:
                return corrected

            return clean_text(query.get("topic", ""))

        corrected = clean_text(
            getattr(query, "corrected_topic", "")
        )

        if corrected:
            return corrected

        try:
            data = query.model_dump()
            corrected = clean_text(
                data.get("corrected_topic", "")
            )

            if corrected:
                return corrected

            return clean_text(data.get("topic", ""))
        except Exception:
            return clean_text(getattr(query, "topic", ""))

    def _coerce_query(
        self,
        query: SearchQuerySchema | str | dict[str, Any],
        resolved_topic: str,
    ) -> SearchQuerySchema:
        if isinstance(query, SearchQuerySchema):
            if resolved_topic:
                try:
                    return query.model_copy(
                        update={"topic": resolved_topic}
                    )
                except Exception:
                    return query

            return query

        if isinstance(query, str):
            return SearchQuerySchema(
                topic=resolved_topic or query
            )

        if isinstance(query, dict):
            data = dict(query)
            data.pop("corrected_topic", None)

            if resolved_topic:
                data["topic"] = resolved_topic

            allowed = set(SearchQuerySchema.model_fields.keys())
            clean_data = {
                key: value
                for key, value in data.items()
                if key in allowed
            }

            return SearchQuerySchema.model_validate(clean_data)

        raise ValueError("Unsupported query type")

    def _classify_ranked(
        self,
        ranked_sources: list[RankedSource],
        query_schema: SearchQuerySchema,
    ) -> list[RankedSource]:
        if self._classifier is None:
            return ranked_sources

        sources = [item.source for item in ranked_sources]

        try:
            classified_sources = self._classifier.classify_sources(
                sources, level=query_schema.level
            )
        except Exception:
            return ranked_sources

        classified_map = {
            item.source_id: item for item in classified_sources
        }
        updated: list[RankedSource] = []

        for ranked in ranked_sources:
            source = classified_map.get(
                ranked.source.source_id, ranked.source
            )

            if source is not ranked.source:
                ranked = ranked.model_copy(update={"source": source})

            updated.append(ranked)

        return updated

    def _unwrap_payload(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {"steps": []}

        if "steps" in payload:
            return payload

        for key in (
            "learning_path",
            "path",
            "result",
            "output",
            "data",
        ):
            value = payload.get(key)

            if isinstance(value, dict) and "steps" in value:
                return value

        for value in payload.values():
            if isinstance(value, dict) and "steps" in value:
                return value

            if isinstance(value, list):
                return {"steps": value}

        return payload

    def _parse_llm_path(
        self,
        payload: dict[str, Any],
        ranked_sources: list[RankedSource],
        query_schema: SearchQuerySchema,
    ) -> LearningPath:
        ranked_map = {
            item.source.source_id: item
            for item in ranked_sources
        }

        used_source_ids: set[str] = set()
        steps: list[LearningStep] = []

        raw_steps = (
            payload.get("steps", [])
            if isinstance(payload, dict)
            else []
        )

        if not isinstance(raw_steps, list):
            raw_steps = []

        for raw_step in raw_steps:
            if not isinstance(raw_step, dict):
                continue

            title = clean_text(raw_step.get("title", ""))
            objective = clean_text(raw_step.get("objective", ""))

            if not title and objective:
                title = objective

            if not title:
                continue

            resource_ids = self._extract_resource_ids(raw_step)
            resources: list[RankedSource] = []

            for raw_id in resource_ids:
                source_id = clean_text(raw_id)

                if not source_id:
                    continue

                ranked_source = ranked_map.get(source_id)

                if (
                    ranked_source is None
                    or source_id in used_source_ids
                ):
                    continue

                reason = (
                    "Selected by AI path builder for this step "
                    "objective."
                )
                resources.append(
                    ranked_source.model_copy(
                        update={"reason": reason}
                    )
                )
                used_source_ids.add(source_id)

                if (
                    self._max_resources_per_step is not None
                    and len(resources)
                    >= self._max_resources_per_step
                ):
                    break

            if not resources:
                continue

            estimated_minutes = self._parse_estimated_minutes(
                raw_step.get("estimated_minutes"), resources
            )

            steps.append(
                LearningStep(
                    step=len(steps) + 1,
                    title=title,
                    objective=objective or title,
                    estimated_minutes=estimated_minutes,
                    resources=resources,
                )
            )

        if steps:
            steps = self._maybe_swap_weak_resources(
                steps,
                ranked_sources,
                used_source_ids,
                query_schema,
            )

        return LearningPath(
            topic=query_schema.topic,
            level=query_schema.level,
            goal=query_schema.goal,
            steps=steps,
        )

    def _maybe_swap_weak_resources(
        self,
        steps: list[LearningStep],
        ranked_sources: list[RankedSource],
        used_source_ids: set[str],
        query_schema: SearchQuerySchema,
    ) -> list[LearningStep]:
        if not steps or not ranked_sources:
            return steps

        candidates: list[RankedSource] = []

        for ranked in ranked_sources:
            source_id = ranked.source.source_id

            if source_id in used_source_ids:
                continue

            if float(ranked.score) < MIN_STEP_RESOURCE_SCORE:
                continue

            candidates.append(ranked)

        if not candidates:
            return steps

        candidates.sort(
            key=lambda item: (
                -float(item.score),
                -float(item.confidence),
            )
        )

        updated_steps: list[LearningStep] = []
        swaps_performed = 0

        for step in steps:
            resources = list(getattr(step, "resources", []) or [])

            if not resources:
                updated_steps.append(step)
                continue

            weakest = min(
                resources,
                key=lambda item: float(
                    getattr(item, "score", 0.0)
                ),
            )

            weakest_score = float(
                getattr(weakest, "score", 0.0)
            )

            if weakest_score >= MIN_STEP_RESOURCE_SCORE:
                updated_steps.append(step)
                continue

            step_band = self._step_difficulty_band(step.title)

            if step_band is None:
                updated_steps.append(step)
                continue

            best_replacement = None

            for candidate in candidates:
                if self._source_in_band(
                    candidate.source, step_band
                ):
                    best_replacement = candidate
                    break

            if best_replacement is None:
                updated_steps.append(step)
                continue

            replacement_score = float(best_replacement.score)

            if replacement_score <= weakest_score:
                updated_steps.append(step)
                continue

            old_source_id = str(
                getattr(weakest.source, "source_id", "") or ""
            )
            new_source_id = str(
                getattr(best_replacement.source, "source_id", "")
                or ""
            )

            used_source_ids.discard(old_source_id)
            used_source_ids.add(new_source_id)

            new_reason = (
                "Swapped in by path builder because original pick "
                f"scored {weakest_score:.2f} below "
                f"{MIN_STEP_RESOURCE_SCORE:.2f}"
            )

            new_resources: list[RankedSource] = []

            for resource in resources:
                if resource is weakest:
                    new_resources.append(
                        best_replacement.model_copy(
                            update={"reason": new_reason}
                        )
                    )
                else:
                    new_resources.append(resource)

            updated_steps.append(
                step.model_copy(update={"resources": new_resources})
            )

            candidates = [
                c
                for c in candidates
                if c.source.source_id != new_source_id
            ]

            swaps_performed += 1

            self._trace.emit(
                "learning_path_step_resource_swapped",
                step_number=int(getattr(step, "step", 0) or 0),
                step_title=clean_text(step.title)[:120],
                step_band=step_band,
                old_source_id=old_source_id,
                new_source_id=new_source_id,
                old_score=round(weakest_score, 4),
                new_score=round(replacement_score, 4),
                reason="llm_pick_below_min_score",
            )

        if swaps_performed:
            self._trace.emit(
                "learning_path_resource_swaps_total",
                swaps_performed=swaps_performed,
                total_steps=len(updated_steps),
                min_step_resource_score=MIN_STEP_RESOURCE_SCORE,
            )

        return updated_steps

    def _step_difficulty_band(self, title: str) -> str | None:
        lowered = clean_text(title).lower()

        if not lowered:
            return None

        if any(
            marker in lowered for marker in _ADVANCED_STEP_MARKERS
        ):
            return "advanced"

        if any(
            marker in lowered for marker in _BEGINNER_STEP_MARKERS
        ):
            return "beginner"

        if any(
            marker in lowered
            for marker in _INTERMEDIATE_STEP_MARKERS
        ):
            return "intermediate"

        return None

    def _source_in_band(self, source: Any, band: str) -> bool:
        if source is None:
            return False

        difficulty = getattr(source, "difficulty", None)
        source_type = self._type_value(source)
        platform = self._enum_value(
            getattr(source, "platform", "")
        ).lower()

        if band == "advanced":
            if difficulty == Difficulty.ADVANCED:
                return True

            if source_type == "research_paper":
                return True

            if difficulty == Difficulty.INTERMEDIATE:
                return True

            return False

        if band == "beginner":
            if difficulty == Difficulty.BEGINNER:
                return True

            if self._is_beginner_source(source):
                return True

            if difficulty == Difficulty.INTERMEDIATE:
                return True

            return False

        if band == "intermediate":
            if difficulty == Difficulty.INTERMEDIATE:
                return True

            if source_type in {
                "course",
                "book",
                "documentation",
            }:
                return True

            if platform in {
                "wikipedia",
                "wikibooks",
                "openstax",
                "libretexts",
                "mit_ocw",
            }:
                return True

            return False

        return False

    def _extract_resource_ids(
        self,
        raw_step: dict[str, Any],
    ) -> list[str]:
        values: list[Any] = []

        for key in (
            "resource_source_ids",
            "resource_ids",
            "source_ids",
            "resources",
            "selected_sources",
            "sources",
        ):
            value = raw_step.get(key)

            if value is None:
                continue

            if isinstance(value, list):
                values.extend(value)
            elif isinstance(value, (str, int)):
                values.append(value)
            elif isinstance(value, dict):
                source_id = value.get("source_id") or value.get(
                    "id"
                )

                if source_id is not None:
                    values.append(source_id)

        extracted: list[str] = []

        for value in values:
            if isinstance(value, dict):
                source_id = value.get("source_id") or value.get(
                    "id"
                )

                if source_id is not None:
                    extracted.append(str(source_id))
            elif value is not None:
                extracted.append(str(value))

        return extracted

    def _parse_estimated_minutes(
        self,
        value: Any,
        resources: list[RankedSource],
    ) -> int:
        try:
            minutes = int(value)

            if minutes > 0:
                return minutes
        except Exception:
            pass

        return max(30, len(resources) * 30)

    def _serialize_ranked_sources(
        self,
        ranked_sources: list[RankedSource],
        tier: str = "compact",
    ) -> list[dict[str, Any]]:
        normalized = str(tier or "compact").strip().lower()
        serialized: list[dict[str, Any]] = []

        for ranked in ranked_sources:
            entry: dict[str, Any] = {
                "source_id": ranked.source.source_id,
                "title": ranked.source.title,
                "platform": self._enum_value(
                    ranked.source.platform
                ),
                "source_type": self._enum_value(
                    ranked.source.source_type
                ),
                "difficulty": (
                    self._enum_value(ranked.source.difficulty)
                    if ranked.source.difficulty is not None
                    else None
                ),
                "score": ranked.score,
                "rank": ranked.rank,
            }

            if normalized == "full":
                entry["url"] = ranked.source.url
                entry["abstract"] = truncate_text(
                    ranked.source.abstract or "",
                    max_length=250,
                    suffix="",
                )
                entry["reason"] = ranked.reason

            serialized.append(entry)

        return serialized

    def _heuristic_path(
        self,
        ranked_sources: list[RankedSource],
        query_schema: SearchQuerySchema,
    ) -> LearningPath:
        limit = self._max_resources_per_step or 4
        used: set[str] = set()
        steps: list[LearningStep] = []

        beginner_sources = [
            item
            for item in ranked_sources
            if self._is_beginner_source(item.source)
        ]

        beginner_ids = {
            b.source.source_id for b in beginner_sources
        }

        intermediate_sources = [
            item
            for item in ranked_sources
            if getattr(item.source, "difficulty", None)
            == Difficulty.INTERMEDIATE
            and item.source.source_id not in beginner_ids
        ]

        advanced_sources = [
            item
            for item in ranked_sources
            if getattr(item.source, "difficulty", None)
            == Difficulty.ADVANCED
        ]

        practice_sources = [
            item
            for item in ranked_sources
            if self._type_value(item.source)
            in {"repository", "model", "dataset"}
            or getattr(item.source, "has_code", False) is True
        ]

        def add_step(
            title: str,
            objective: str,
            candidates: list[RankedSource],
            reason: str,
        ) -> bool:
            available = [
                item
                for item in candidates
                if item.source.source_id not in used
            ]

            if not available:
                return False

            resources: list[RankedSource] = []

            for item in available[:limit]:
                used.add(item.source.source_id)
                resources.append(
                    item.model_copy(update={"reason": reason})
                )

            steps.append(
                LearningStep(
                    step=len(steps) + 1,
                    title=title,
                    objective=objective,
                    estimated_minutes=max(
                        30, len(resources) * 30
                    ),
                    resources=resources,
                )
            )

            return True

        if beginner_sources:
            add_step(
                "Foundations and Basics",
                "Start with beginner-friendly sources that explain "
                "the fundamentals clearly.",
                beginner_sources,
                "Selected because this source is beginner-friendly "
                "and matches the requested starting level.",
            )
        else:
            sorted_all = sorted(
                ranked_sources,
                key=lambda item: self._difficulty_order(
                    item.source
                ),
            )

            add_step(
                "Foundations",
                "Start with the easiest available sources for this "
                "topic.",
                sorted_all,
                "Selected because no explicit beginner source was "
                "found, so the easiest available source is used "
                "first.",
            )

        add_step(
            "Core Concepts",
            "Build the main concepts after the fundamentals.",
            intermediate_sources
            or [
                item
                for item in ranked_sources
                if item.source.source_id not in used
                and self._difficulty_order(item.source) <= 1
            ],
            "Selected because this source explains core concepts "
            "for this topic.",
        )

        add_step(
            "Applied Practice",
            "Apply the concepts through code, models, datasets, or "
            "practical examples.",
            practice_sources
            or [
                item
                for item in ranked_sources
                if item.source.source_id not in used
                and self._type_value(item.source)
                in {
                    "repository",
                    "model",
                    "dataset",
                    "documentation",
                }
            ],
            "Selected because this source supports applied practice "
            "or implementation.",
        )

        add_step(
            "Advanced Research",
            "Move to advanced papers and deeper technical material.",
            advanced_sources
            or [
                item
                for item in ranked_sources
                if item.source.source_id not in used
                and self._difficulty_order(item.source) >= 1
            ],
            "Selected because this source provides advanced or "
            "deeper material.",
        )

        while len(steps) < MINIMUM_STEPS:
            remaining = [
                item
                for item in ranked_sources
                if item.source.source_id not in used
            ]

            if not remaining:
                break

            add_step(
                "Deepen Understanding",
                "Continue studying additional relevant sources.",
                remaining,
                "Selected to ensure the path has enough useful "
                "sources.",
            )

        steps = steps[:MAX_TARGET_STEPS]

        renumbered: list[LearningStep] = []

        for index, step in enumerate(steps, start=1):
            objective = self._annotate_objective(
                step.objective, query_schema
            )
            renumbered.append(
                step.model_copy(
                    update={
                        "step": index,
                        "objective": objective,
                    }
                )
            )

        return LearningPath(
            topic=query_schema.topic,
            level=query_schema.level,
            goal=query_schema.goal,
            steps=renumbered,
        )

    def _pad_steps(
        self,
        steps: list[LearningStep],
        ranked_sources: list[RankedSource],
        used_ids: set[str],
        query_schema: SearchQuerySchema,
    ) -> list[LearningStep]:
        available = [
            r
            for r in ranked_sources
            if r.source.source_id not in used_ids
        ]

        step_number = len(steps)

        for ranked in available:
            if len(steps) >= MINIMUM_STEPS:
                break

            step_number += 1
            used_ids.add(ranked.source.source_id)

            reason = (
                f"{PADDING_MARKER} LLM path produced fewer than "
                f"{MINIMUM_STEPS} steps; source added to satisfy "
                f"minimum length"
            )

            steps.append(
                LearningStep(
                    step=step_number,
                    title=(
                        f"Deepen Understanding: "
                        f"{ranked.source.title[:60]}"
                    ),
                    objective=(
                        f"Study {ranked.source.title} to extend "
                        f"your knowledge."
                    ),
                    estimated_minutes=45,
                    resources=[
                        ranked.model_copy(
                            update={"reason": reason}
                        )
                    ],
                )
            )

        for index, step in enumerate(steps, start=1):
            step.step = index

        return steps

    def _align_path(
        self,
        learning_path: LearningPath,
        query_schema: SearchQuerySchema,
    ) -> LearningPath:
        steps = list(getattr(learning_path, "steps", []) or [])

        if not steps:
            return learning_path

        updated_steps: list[LearningStep] = []

        for step in steps:
            resources = list(
                getattr(step, "resources", []) or []
            )
            resources.sort(
                key=lambda item: self._difficulty_order(
                    getattr(item, "source", None)
                )
            )
            updated_steps.append(
                step.model_copy(update={"resources": resources})
            )

        indexed: list[tuple[int, LearningStep]] = list(
            enumerate(updated_steps)
        )

        indexed.sort(
            key=lambda pair: (
                self._step_title_band(pair[1].title),
                self._step_difficulty_score(pair[1]),
                pair[0],
            )
        )

        updated_steps = [step for _, step in indexed]

        renumbered: list[LearningStep] = []

        for index, step in enumerate(updated_steps, start=1):
            objective = self._annotate_objective(
                step.objective, query_schema
            )
            renumbered.append(
                step.model_copy(
                    update={
                        "step": index,
                        "objective": objective,
                    }
                )
            )

        return learning_path.model_copy(
            update={"steps": renumbered}
        )

    def _step_difficulty_score(self, step: LearningStep) -> float:
        resources = list(getattr(step, "resources", []) or [])

        if not resources:
            return 1.0

        orders = [
            self._difficulty_order(getattr(item, "source", None))
            for item in resources
        ]

        return sum(orders) / len(orders)

    def _difficulty_order(self, source: Any) -> int:
        if source is None:
            return 1

        difficulty = getattr(source, "difficulty", None)

        if difficulty == Difficulty.BEGINNER:
            return 0

        if difficulty == Difficulty.INTERMEDIATE:
            return 1

        if difficulty == Difficulty.ADVANCED:
            return 2

        if self._is_beginner_source(source):
            return 0

        if self._type_value(source) == "research_paper":
            return 2

        return 1

    def _step_title_band(self, title: str) -> int:
        lowered = clean_text(title).lower().strip()

        if not lowered:
            return 1

        beginner_markers = (
            "introduction",
            "intro ",
            "intro:",
            "intro to",
            "foundation",
            "foundations",
            " basics",
            "basic ",
            "basic:",
            "fundamental",
            "getting started",
            "overview",
            "primer",
            "first steps",
        )

        advanced_markers = (
            "advanced",
            "expert",
            "deep dive",
            "deep-dive",
            "in depth",
            "in-depth",
            "frontier",
            "state of the art",
            "state-of-the-art",
        )

        for marker in beginner_markers:
            if marker in lowered:
                return 0

        for marker in advanced_markers:
            if marker in lowered:
                return 2

        return 1

    def _is_beginner_source(self, source: Any) -> bool:
        if source is None:
            return False

        difficulty = getattr(source, "difficulty", None)

        if difficulty == Difficulty.BEGINNER:
            return True

        platform = self._enum_value(
            getattr(source, "platform", "")
        ).lower()
        source_type = self._type_value(source)

        beginner_platforms = {
            "wikipedia",
            "wikibooks",
            "wikiversity",
            "openstax",
            "libretexts",
            "mit_ocw",
            "open_library",
            "internet_archive",
        }

        beginner_source_types = {
            "documentation",
            "course",
            "book",
            "video",
        }

        if platform in beginner_platforms:
            return True

        if source_type in beginner_source_types:
            return True

        return False

    def _annotate_objective(
        self,
        objective: str,
        query_schema: SearchQuerySchema,
    ) -> str:
        objective = clean_text(objective)

        domains = self._clean_string_list(
            getattr(query_schema, "target_domains", [])
        )

        if not domains:
            domains = ["general"]

        level = (
            clean_text(getattr(query_schema, "level", ""))
            or "mixed"
        )

        annotation = (
            f"Domain: {', '.join(domains)} | "
            f"Level: {level} | "
            "Ordering: basics -> core -> applied -> advanced."
        )

        if not objective:
            return annotation

        if "Domain:" in objective:
            return objective

        return f"{objective} | {annotation}"

    def _compute_minimum_steps(self, source_count: int) -> int:
        if source_count < 3:
            return 1

        if source_count < 6:
            return 3

        if source_count < 12:
            return 4

        return 5

    def _clean_string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []

        cleaned = [
            clean_text(item)
            for item in value
            if isinstance(item, str)
        ]
        return [item for item in cleaned if item]

    def _type_value(self, source: Any) -> str:
        return self._enum_value(
            getattr(source, "source_type", "")
        ).lower()

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value) or "")