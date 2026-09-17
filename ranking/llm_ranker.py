from __future__ import annotations
from typing import Any
from core import constants
from llm.guardrails import validate_llm_request, validate_llm_response
from llm.parser import parse_json_object_response
from llm.prompts import build_source_ranking_prompt, system_research_ranker
from llm.provider import LLMProviderManager
from llm.router import get_strong_model_references
from core.models import Difficulty, ModelReference, RankedSource, Source
from ranking.difficulty_classifier import DifficultyClassifier
from ranking.scorer import HeuristicScorer
from core.schemas import LLMMessageSchema, LLMRequestSchema, SearchQuerySchema
from utils.logger import get_logger
from utils.text import clean_text, truncate_text

class LLMRanker:
    def __init__(
        self,
        scorer: HeuristicScorer | None = None,
        classifier: DifficultyClassifier | None = None,
        model_reference: ModelReference | None = None,
        max_sources: int = 20,
    ) -> None:
        self._scorer = scorer or HeuristicScorer()
        self._classifier = classifier or DifficultyClassifier()
        self._model_reference = model_reference
        self._max_sources = max(3, max_sources)
        self._logger = get_logger("ranking.llm_ranker")

    async def rank(
        self,
        sources: list[Any],
        query: SearchQuerySchema | str | dict[str, Any],
    ) -> list[RankedSource]:
        query_schema = self._coerce_query(query)
        valid_sources = [source for source in sources if isinstance(source, Source)]
        if not valid_sources:
            return []
        classified_sources = self._classifier.classify_sources(valid_sources)
        heuristic_ranked = self._scorer.score_sources(
            classified_sources,
            query=query_schema.topic,
            goal=query_schema.goal,
            level=query_schema.level,
        )
        if len(heuristic_ranked) <= 1:
            return heuristic_ranked
        candidate_sources = [item.source for item in heuristic_ranked[: self._max_sources]]
        source_map = {source.source_id: source for source in candidate_sources}
        model_reference = self._model_reference
        if model_reference is None:
            try:
                model_reference = get_strong_model_references(limit=1)[0]
            except Exception as exc:
                self._logger.warning(f"No usable LLM ranker model: {exc}")
                return heuristic_ranked
        try:
            serialized_sources = self._serialize_sources(candidate_sources)
            prompt = build_source_ranking_prompt(
                topic=query_schema.topic,
                goal=query_schema.goal,
                level=query_schema.level,
                sources=serialized_sources,
            )
            messages = [
                LLMMessageSchema(role="system", content=system_research_ranker()),
                LLMMessageSchema(role="user", content=prompt),
            ]
            request = LLMRequestSchema(
                provider=model_reference.provider,
                model=model_reference.model_id,
                messages=messages,
                temperature=constants.DEFAULT_TEMPERATURE,
                max_tokens=3000,
                response_format="json_object",
            )
            request = validate_llm_request(request)
            response: Any = None
            async with LLMProviderManager() as manager:
                response = await manager.complete(request)
            if response is None:
                return heuristic_ranked
            response = validate_llm_response(response)
            payload = self._extract_payload(response.content)
            parsed_rankings = self._parse_rankings(payload, source_map)
            if parsed_rankings:
                return self._merge_with_heuristic(
                    parsed_rankings,
                    heuristic_ranked,
                    source_map,
                )
        except Exception as exc:
            self._logger.warning(f"LLM ranking failed, using heuristic fallback: {exc}")
        return heuristic_ranked

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
        raise ValueError("Unsupported query type")

    def _serialize_sources(self, sources: list[Source]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for source in sources:
            serialized.append(
                {
                    "source_id": source.source_id,
                    "title": source.title,
                    "url": source.url,
                    "platform": self._enum_value(source.platform),
                    "source_type": self._enum_value(source.source_type),
                    "abstract": truncate_text(source.abstract or "", max_length=500),
                    "year": source.year,
                    "citation_count": source.citation_count,
                    "has_code": source.has_code,
                    "difficulty": self._enum_value(source.difficulty),
                }
            )
        return serialized

    def _extract_payload(self, content: str) -> dict[str, Any] | None:
        try:
            payload = parse_json_object_response(content)
            if "rankings" in payload:
                return payload
            for value in payload.values():
                if isinstance(value, list):
                    return {"rankings": value}
            return payload
        except Exception:
            return None

    def _parse_rankings(
        self,
        payload: dict[str, Any] | None,
        source_map: dict[str, Source],
    ) -> list[RankedSource]:
        if payload is None:
            return []
        raw_items = payload.get("rankings", [])
        if not isinstance(raw_items, list):
            return []
        parsed: list[tuple[int, RankedSource]] = []
        seen: set[str] = set()
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            source_id = str(item.get("source_id", "")).strip()
            source = source_map.get(source_id)
            if source is None or source_id in seen:
                continue
            score = self._normalize_unit_interval(item.get("score"))
            confidence = self._normalize_unit_interval(item.get("confidence"))
            reason = truncate_text(
                clean_text(item.get("reason", "")),
                max_length=300,
                suffix="",
            ) or None
            difficulty = self._parse_difficulty(item.get("difficulty"))
            updated_source = source
            if difficulty is not None:
                try:
                    updated_source = source.model_copy(update={"difficulty": difficulty})
                except Exception:
                    updated_source = source
            try:
                provided_rank = int(item.get("rank", index + 1))
            except Exception:
                provided_rank = index + 1
            ranked_source = RankedSource(
                source=updated_source,
                rank=max(1, provided_rank),
                score=0.5 if score is None else score,
                confidence=0.5 if confidence is None else confidence,
                reason=reason,
            )
            parsed.append((max(1, provided_rank), ranked_source))
            seen.add(source_id)
        parsed.sort(key=lambda item: (item[0], -item[1].score))
        return [ranked for _, ranked in parsed]

    def _merge_with_heuristic(
        self,
        parsed_rankings: list[RankedSource],
        heuristic_rankings: list[RankedSource],
        source_map: dict[str, Source],
    ) -> list[RankedSource]:
        final: list[RankedSource] = []
        seen: set[str] = set()
        for ranked in parsed_rankings:
            source_id = ranked.source.source_id
            if source_id in seen:
                continue
            seen.add(source_id)
            final.append(ranked)
        for ranked in heuristic_rankings:
            source_id = ranked.source.source_id
            if source_id in seen:
                continue
            seen.add(source_id)
            final.append(ranked)
        for index, ranked in enumerate(final, start=1):
            ranked.rank = index
        return final

    def _normalize_unit_interval(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            numeric = float(value)
        except Exception:
            return None
        if numeric < 0:
            return 0.0
        if numeric <= 1:
            return numeric
        if numeric <= 100:
            return numeric / 100
        return 1.0

    def _parse_difficulty(self, value: Any) -> Difficulty | None:
        if value is None:
            return None
        try:
            return Difficulty(str(value).strip().lower())
        except Exception:
            return Difficulty.UNKNOWN

    @staticmethod
    def _enum_value(value: Any) -> str:
        if value is None:
            return ""
        if hasattr(value, "value"):
            return str(value.value)
        return str(value)