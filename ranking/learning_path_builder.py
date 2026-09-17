from __future__ import annotations
from typing import Any
from core import constants
from llm.guardrails import validate_llm_request, validate_llm_response
from llm.parser import parse_json_object_response
from llm.prompts import build_learning_path_prompt, system_learning_path_builder
from llm.provider import LLMProviderManager
from llm.router import get_fast_model_references, get_strong_model_references
from core.models import LearningPath, LearningStep, RankedSource
from core.schemas import LLMMessageSchema, LLMRequestSchema, SearchQuerySchema
from utils.logger import get_logger, get_trace_logger
from utils.text import clean_text


class LearningPathBuilder:
    def __init__(
        self,
        use_llm: bool = True,
        classifier: Any | None = None,
        max_resources_per_step: int | None = None,
    ) -> None:
        self._use_llm = bool(use_llm)
        self._classifier = classifier
        resolved_limit = None
        try:
            if max_resources_per_step is not None:
                resolved_limit = max(1, int(max_resources_per_step))
        except Exception:
            resolved_limit = None
        self._max_resources_per_step = resolved_limit
        self._logger = get_logger("ranking.learning_path_builder")
        self._trace = get_trace_logger()

    async def build(
        self,
        ranked_sources: list[Any],
        query: SearchQuerySchema | str | dict[str, Any],
    ) -> LearningPath:
        resolved_topic = self._extract_corrected_topic(query)
        query_schema = self._coerce_query(query, resolved_topic)
        valid_ranked = [item for item in ranked_sources if isinstance(item, RankedSource)]
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
        try:
            ai_path = await self._build_with_llm(valid_ranked, query_schema)
            if ai_path.steps:
                self._trace.emit(
                    "learning_path_strategy",
                    strategy="ai",
                    total_steps=len(ai_path.steps),
                )
                return ai_path
            self._trace.emit(
                "learning_path_llm_rejected",
                reason="llm_returned_no_steps",
                llm_steps=0,
            )
            return empty_path
        except Exception as exc:
            self._logger.warning(f"LLM learning path generation failed: {exc}")
            self._trace.emit(
                "learning_path_llm_failed",
                error=str(exc),
            )
            return empty_path

    def _extract_corrected_topic(self, query: Any) -> str:
        if isinstance(query, str):
            return clean_text(query)
        if isinstance(query, dict):
            corrected = clean_text(query.get("corrected_topic", ""))
            if corrected:
                return corrected
            return clean_text(query.get("topic", ""))
        corrected = clean_text(getattr(query, "corrected_topic", ""))
        if corrected:
            return corrected
        try:
            data = query.model_dump()
            corrected = clean_text(data.get("corrected_topic", ""))
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
                    return query.model_copy(update={"topic": resolved_topic})
                except Exception:
                    return query
            return query
        if isinstance(query, str):
            return SearchQuerySchema(topic=resolved_topic or query)
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

    async def _build_with_llm(
        self,
        ranked_sources: list[RankedSource],
        query_schema: SearchQuerySchema,
    ) -> LearningPath:
        model_references = self._model_references()
        if not model_references:
            raise ValueError("No model references available")
        serialized = self._serialize_ranked_sources(ranked_sources)
        base_prompt = build_learning_path_prompt(
            topic=query_schema.topic,
            goal=query_schema.goal,
            level=query_schema.level,
            ranked_sources=serialized,
        )
        last_error: Exception | None = None
        async with LLMProviderManager() as manager:
            for model_reference in model_references:
                provider_name = self._enum_value(model_reference.provider)
                model_id = clean_text(getattr(model_reference, "model_id", ""))
                if not model_id:
                    model_id = clean_text(getattr(model_reference, "model", ""))
                if not model_id:
                    continue
                for attempt in range(2):
                    try:
                        prompt = base_prompt if attempt == 0 else self._repair_prompt(base_prompt)
                        messages = [
                            LLMMessageSchema(
                                role="system",
                                content=system_learning_path_builder(),
                            ),
                            LLMMessageSchema(
                                role="user",
                                content=prompt,
                            ),
                        ]
                        request = LLMRequestSchema(
                            provider=model_reference.provider,
                            model=model_id,
                            messages=messages,
                            temperature=constants.DEFAULT_TEMPERATURE,
                            max_tokens=3000,
                            response_format="json_object",
                        )
                        request = validate_llm_request(request)
                        response = await manager.complete(request)
                        if response is None:
                            continue
                        response = validate_llm_response(response)
                        self._trace.emit(
                            "learning_path_llm_response",
                            provider=provider_name,
                            model=model_id,
                            content_length=len(response.content or ""),
                        )
                        payload = parse_json_object_response(response.content)
                        payload = self._unwrap_payload(payload)
                        parsed_path = self._parse_llm_path(
                            payload,
                            ranked_sources,
                            query_schema,
                        )
                        self._trace.emit(
                            "learning_path_llm_parsed",
                            parsed_steps=len(parsed_path.steps),
                        )
                        if parsed_path.steps:
                            return parsed_path
                        last_error = ValueError("LLM returned no learning path steps")
                    except Exception as exc:
                        last_error = exc
                        continue
        if last_error is not None:
            raise last_error
        raise ValueError("All models failed to generate learning path")

    def _model_references(self) -> list[Any]:
        references: list[Any] = []
        seen: set[tuple[str, str]] = set()
        try:
            strong_references = get_strong_model_references(limit=3)
        except Exception:
            strong_references = []
        try:
            fast_references = get_fast_model_references(limit=3)
        except Exception:
            fast_references = []
        for reference in list(strong_references) + list(fast_references):
            try:
                provider_name = self._enum_value(reference.provider)
                model_id = clean_text(getattr(reference, "model_id", ""))
                if not model_id:
                    model_id = clean_text(getattr(reference, "model", ""))
                if not provider_name or not model_id:
                    continue
                key = (provider_name, model_id)
            except Exception:
                continue
            if key in seen:
                continue
            seen.add(key)
            references.append(reference)
        return references[:5]

    def _unwrap_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {"steps": []}
        if "steps" in payload:
            return payload
        for key in ("learning_path", "path", "result", "output", "data"):
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
        raw_steps = payload.get("steps", []) if isinstance(payload, dict) else []
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
                if ranked_source is None or source_id in used_source_ids:
                    continue
                resources.append(ranked_source)
                used_source_ids.add(source_id)
                if (
                    self._max_resources_per_step is not None
                    and len(resources) >= self._max_resources_per_step
                ):
                    break
            if not resources:
                continue
            estimated_minutes = self._parse_estimated_minutes(
                raw_step.get("estimated_minutes"),
                resources,
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
        if len(steps) < 3 and len(ranked_sources) >= 3:
            steps = self._pad_steps(steps, ranked_sources, used_source_ids, query_schema)
        return LearningPath(
            topic=query_schema.topic,
            level=query_schema.level,
            goal=query_schema.goal,
            steps=steps,
        )
    def _pad_steps(
        self,
        steps: list[LearningStep],
        ranked_sources: list[RankedSource],
        used_ids: set[str],
        query_schema: SearchQuerySchema,
    ) -> list[LearningStep]:
        available = [
            r for r in ranked_sources
            if r.source.source_id not in used_ids
        ]
        step_number = len(steps)
        for ranked in available:
            if len(steps) >= 3:
                break
            step_number += 1
            used_ids.add(ranked.source.source_id)
            steps.append(
                LearningStep(
                    step=step_number,
                    title=f"Deepen Understanding: {ranked.source.title[:60]}",
                    objective=f"Study {ranked.source.title} to extend your knowledge.",
                    estimated_minutes=45,
                    resources=[ranked],
                )
            )
        for index, step in enumerate(steps, start=1):
            step.step = index
        return steps

    def _extract_resource_ids(self, raw_step: dict[str, Any]) -> list[str]:
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
                source_id = value.get("source_id") or value.get("id")
                if source_id is not None:
                    values.append(source_id)
        extracted: list[str] = []
        for value in values:
            if isinstance(value, dict):
                source_id = value.get("source_id") or value.get("id")
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

    def _repair_prompt(self, base_prompt: str) -> str:
        return (
            "The previous learning path output was empty, invalid, or did not contain steps.\n"
            "Regenerate the complete learning path now.\n"
            "Rules: return only valid JSON; steps must be a non-empty array; use only provided source IDs; "
            "if ranked_sources contains at least one source, create at least one step.\n\n"
            f"{base_prompt}"
        )

    def _serialize_ranked_sources(
        self,
        ranked_sources: list[RankedSource],
    ) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for ranked in ranked_sources:
            serialized.append(
                {
                    "source_id": ranked.source.source_id,
                    "title": ranked.source.title,
                    "url": ranked.source.url,
                    "platform": self._enum_value(ranked.source.platform),
                    "source_type": self._enum_value(ranked.source.source_type),
                    "difficulty": self._enum_value(ranked.source.difficulty)
                    if ranked.source.difficulty is not None
                    else None,
                    "rank": ranked.rank,
                    "score": ranked.score,
                    "reason": ranked.reason,
                }
            )
        return serialized

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value))