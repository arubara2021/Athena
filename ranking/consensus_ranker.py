from __future__ import annotations

from pathlib import Path
from typing import Any

from core.config import get_settings
from core.ranking_config import get_ranking_config
from llm.aggregator import ConsensusAggregator
from llm.ensemble import EnsembleEngine
from llm.parser import parse_json_object_response
from llm.prompts import build_source_ranking_prompt, system_research_ranker
from llm.router import choose_models_for_task, get_judge_model_reference
from core.models import Difficulty, RankedSource, Source, VotingMode
from ranking.difficulty_classifier import DifficultyClassifier
from ranking.scorer import HeuristicScorer
from core.schemas import EnsembleRequestSchema, SearchQuerySchema
from search.query_tokens import concept_tokens
from utils.logger import get_logger, get_trace_logger
from utils.text import clean_text, truncate_text

_THRESHOLD_CACHE: dict[str, float] | None = None


def _load_consensus_thresholds() -> dict[str, float]:
    global _THRESHOLD_CACHE

    if _THRESHOLD_CACHE is not None:
        return dict(_THRESHOLD_CACHE)

    values = {
        "min_relevance_score": 0.20,
        "min_llm_score": 0.05,
    }

    try:
        import yaml

        config_path = Path(__file__).resolve().parent.parent / "configs" / "ranking.yaml"

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}

            section = loaded.get("consensus", {}) if isinstance(loaded, dict) else {}

            if isinstance(section, dict):
                for key in values:
                    raw = section.get(key)
                    try:
                        numeric = float(raw)
                    except Exception:
                        continue
                    values[key] = max(0.0, min(1.0, numeric))
    except Exception:
        pass

    _THRESHOLD_CACHE = dict(values)
    return dict(values)


class ConsensusRanker:
    def __init__(
        self,
        scorer: HeuristicScorer | None = None,
        classifier: DifficultyClassifier | None = None,
        max_models: int | None = None,
        max_llm_sources: int | None = None,
        use_llm: bool = True,
        min_relevance_score: float | None = None,
        min_llm_score: float | None = None,
    ) -> None:
        consensus_config = get_ranking_config().consensus_settings()
        thresholds = _load_consensus_thresholds()

        self._scorer = scorer or HeuristicScorer()
        self._classifier = classifier or DifficultyClassifier()
        self._max_models = max(1, max_models or consensus_config.max_models)
        self._max_llm_sources = max(3, max_llm_sources or consensus_config.max_llm_sources)
        self._use_llm = use_llm

        resolved_relevance = (
            min_relevance_score
            if min_relevance_score is not None
            else thresholds["min_relevance_score"]
        )
        resolved_llm_score = (
            min_llm_score
            if min_llm_score is not None
            else thresholds["min_llm_score"]
        )

        self._min_relevance_score = max(0.0, min(1.0, resolved_relevance))
        self._min_llm_score = max(0.0, min(1.0, resolved_llm_score))

        self._logger = get_logger("ranking.consensus_ranker")
        self._trace = get_trace_logger()

    async def rank(
        self,
        sources: list[Any],
        query: SearchQuerySchema | str | dict[str, Any],
        model_references: list[Any] | None = None,
        judge_model: Any | None = None,
    ) -> list[RankedSource]:
        query_schema = self._coerce_query(query)
        valid_sources = [source for source in sources if isinstance(source, Source)]

        if not valid_sources:
            self._trace.emit("ranking_no_sources")
            return []

        classified_sources = self._classifier.classify_sources(valid_sources)

        difficulty_counts: dict[str, int] = {}
        for source in classified_sources:
            difficulty_value = source.difficulty.value if source.difficulty else "unknown"
            difficulty_counts[difficulty_value] = difficulty_counts.get(difficulty_value, 0) + 1

        self._trace.emit(
            "difficulty_classification",
            total_sources=len(classified_sources),
            distribution=difficulty_counts,
        )

        heuristic_ranked = self._scorer.score_sources(
            classified_sources,
            query=query_schema,
            goal=query_schema.goal,
            level=query_schema.level,
        )

        if not heuristic_ranked:
            return []

        relevance_by_id: dict[str, float] = {}
        for item in heuristic_ranked:
            relevance_by_id[item.source.source_id] = self._scorer.relevance_factor(
                item.source,
                query_schema,
            )

        best_relevance = max(relevance_by_id.values(), default=0.0)

        if best_relevance < self._min_relevance_score:
            broadened_query = self._broaden_query(query_schema)
            broadened_ranked = self._scorer.score_sources(
                classified_sources,
                query=broadened_query,
                goal=query_schema.goal,
                level=query_schema.level,
            )

            if broadened_ranked:
                broadened_relevance_by_id: dict[str, float] = {}
                for item in broadened_ranked:
                    broadened_relevance_by_id[item.source.source_id] = (
                        self._scorer.relevance_factor(
                            item.source,
                            broadened_query,
                        )
                    )

                broadened_best = max(broadened_relevance_by_id.values(), default=0.0)

                if broadened_best >= best_relevance:
                    heuristic_ranked = broadened_ranked
                    relevance_by_id = broadened_relevance_by_id
                    best_relevance = broadened_best

            self._trace.emit(
                "ranking_relevance_gate_broadened",
                best_relevance=round(best_relevance, 4),
                threshold=self._min_relevance_score,
                total_sources=len(heuristic_ranked),
            )

        relevant_ranked = [
            item
            for item in heuristic_ranked
            if relevance_by_id.get(item.source.source_id, 0.0)
            >= self._min_relevance_score * 0.5
        ]

        if not relevant_ranked:
            relevant_ranked = heuristic_ranked
            self._trace.emit(
                "ranking_relevance_gate_used_all",
                best_relevance=round(best_relevance, 4),
                threshold=self._min_relevance_score,
                total_sources=len(heuristic_ranked),
            )
        else:
            self._trace.emit(
                "ranking_relevance_gate_passed",
                best_relevance=round(best_relevance, 4),
                threshold=self._min_relevance_score,
                total_sources=len(heuristic_ranked),
                relevant_sources=len(relevant_ranked),
                filtered_out=len(heuristic_ranked) - len(relevant_ranked),
            )

        if len(relevant_ranked) <= 1 or not self._use_llm:
            return relevant_ranked

        candidate_sources = [item.source for item in relevant_ranked[: self._max_llm_sources]]

        try:
            resolved_models = model_references or choose_models_for_task(
                "ranking",
                limit=self._max_models,
            )
        except Exception as exc:
            self._logger.warning(f"Consensus model selection failed: {exc}")
            self._trace.emit(
                "ranking_fallback",
                reason="model_selection_failed",
                error=str(exc),
                fallback="heuristic",
            )
            return relevant_ranked

        if not resolved_models:
            self._trace.emit(
                "ranking_fallback",
                reason="no_models_available",
                fallback="heuristic",
            )
            return relevant_ranked

        self._trace.emit(
            "llm_ranking_started",
            models=[
                {"provider": m.provider.value, "model_id": m.model_id}
                for m in resolved_models
            ],
            candidate_count=len(candidate_sources),
        )

        try:
            source_map = {source.source_id: source for source in candidate_sources}
            serialized_sources = self._serialize_sources(candidate_sources)

            prompt = build_source_ranking_prompt(
                topic=query_schema.topic,
                goal=query_schema.goal,
                level=query_schema.level,
                sources=serialized_sources,
            )

            prompt += (
                f"\nCRITICAL: You MUST rank ALL {len(candidate_sources)} provided sources. "
                f"Do NOT skip any source. "
                f"Return exactly {len(candidate_sources)} ranking entries in the rankings array. "
                f"Every source_id from the input MUST appear in your output.\n"
            )

            settings = get_settings()

            resolved_judge = judge_model
            if resolved_judge is None:
                try:
                    resolved_judge = get_judge_model_reference()
                except Exception:
                    resolved_judge = None

            required_output_tokens = max(4000, len(candidate_sources) * 250)

            request = EnsembleRequestSchema(
                name="source_ranking",
                prompt=prompt,
                context={
                    "system_prompt": system_research_ranker(),
                    "expect_json": True,
                    "max_tokens": required_output_tokens,
                },
                models=resolved_models,
                voting_mode=settings.ensemble_voting_mode,
                threshold=settings.ensemble_consensus_threshold,
                judge_model=resolved_judge,
            )

            ensemble_result: Any = None
            consensus_result: Any = None

            async with EnsembleEngine() as engine:
                ensemble_result = await engine.run(request)

            if ensemble_result is None:
                self._trace.emit(
                    "ranking_fallback",
                    reason="ensemble_returned_none",
                    fallback="heuristic",
                )
                return relevant_ranked

            async with ConsensusAggregator() as aggregator:
                consensus_result = await aggregator.aggregate(request, ensemble_result)

            if consensus_result is None:
                self._trace.emit(
                    "ranking_fallback",
                    reason="consensus_returned_none",
                    fallback="heuristic",
                )
                return relevant_ranked

            self._trace.emit(
                "llm_ranking_consensus",
                agreement_score=consensus_result.agreement_score,
                judge_used=consensus_result.judge_used,
                vote_count=len(consensus_result.votes),
            )

            parsed_rankings = self._parse_final_output(
                consensus_result.final_output,
                source_map,
            )

            if parsed_rankings:
                kept_rankings = [
                    ranked
                    for ranked in parsed_rankings
                    if ranked.score >= self._min_llm_score
                ]

                dropped_count = len(parsed_rankings) - len(kept_rankings)

                if dropped_count > 0:
                    self._trace.emit(
                        "llm_ranking_low_score_dropped",
                        dropped_count=dropped_count,
                        threshold=self._min_llm_score,
                    )

                llm_seen_ids = {ranked.source.source_id for ranked in parsed_rankings}

                if kept_rankings:
                    merged = self._merge_with_heuristic(
                        kept_rankings,
                        relevant_ranked,
                        source_map,
                        exclude_ids=llm_seen_ids,
                    )

                    if merged:
                        aligned = self._apply_final_alignment(merged, query_schema)
                        self._trace.emit(
                            "llm_ranking_success",
                            llm_ranked=len(kept_rankings),
                            llm_dropped=dropped_count,
                            total_merged=len(aligned),
                            judge_used=consensus_result.judge_used,
                        )
                        return aligned
                else:
                    self._trace.emit(
                        "llm_ranking_all_rejected",
                        threshold=self._min_llm_score,
                        fallback="heuristic",
                    )

            self._trace.emit(
                "ranking_fallback",
                reason="parsed_rankings_empty",
                fallback="heuristic",
            )
            return relevant_ranked

        except Exception as exc:
            self._logger.warning(
                f"Consensus ranking failed, using heuristic fallback: {exc}"
            )
            self._trace.emit(
                "ranking_fallback",
                reason="llm_exception",
                error=str(exc),
                fallback="heuristic",
            )
            return relevant_ranked

    def _broaden_query(self, query_schema: Any) -> dict[str, Any]:
        topic = clean_text(
            getattr(query_schema, "corrected_topic", "")
            or getattr(query_schema, "topic", "")
        )
        primary_concept = clean_text(
            getattr(query_schema, "primary_concept", "")
            or topic
        )
        raw_keywords = getattr(query_schema, "keywords", []) or []

        keywords: list[str] = []
        for keyword in raw_keywords:
            text = clean_text(keyword)
            if text and text not in keywords:
                keywords.append(text)

        if not keywords and topic:
            keywords = list(concept_tokens(topic))[:8]
        else:
            for token in concept_tokens(topic):
                if token not in keywords:
                    keywords.append(token)
                if len(keywords) >= 12:
                    break

        short_search_queries = [
            clean_text(item)
            for item in getattr(query_schema, "short_search_queries", []) or []
            if clean_text(item)
        ]

        return {
            "topic": topic,
            "primary_concept": primary_concept,
            "keywords": keywords[:12],
            "short_search_queries": short_search_queries[:6],
        }

    def _apply_final_alignment(
        self,
        merged: list[RankedSource],
        query_schema: SearchQuerySchema,
    ) -> list[RankedSource]:
        goal = query_schema.goal or ""
        level = query_schema.level or ""

        beginner = "beginner" in level.lower() or "basics" in goal.lower()
        advanced = "advanced" in level.lower() or "advanced" in goal.lower()

        if not beginner and not advanced:
            return merged

        adjusted: list[RankedSource] = []

        for ranked in merged:
            alignment = self._scorer.compute_alignment(ranked.source, goal, level)
            new_score = max(0.0, min(1.0, ranked.score + alignment))

            adjusted.append(
                RankedSource(
                    source=ranked.source,
                    rank=ranked.rank,
                    score=new_score,
                    confidence=ranked.confidence,
                    reason=ranked.reason,
                )
            )

        adjusted.sort(key=lambda item: item.score, reverse=True)

        for index, item in enumerate(adjusted, start=1):
            item.rank = index

        self._trace.emit(
            "ranking_final_alignment",
            level=level,
            goal=goal,
            sources_adjusted=len(adjusted),
            top_source=adjusted[0].source.title[:80] if adjusted else None,
        )

        return adjusted

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
                    "difficulty": self._enum_value(source.difficulty)
                    if source.difficulty is not None
                    else None,
                }
            )

        return serialized

    def _parse_final_output(
        self,
        final_output: Any,
        source_map: dict[str, Source],
    ) -> list[RankedSource]:
        payload = self._extract_payload(final_output)

        if payload is None:
            self._trace.emit(
                "ranking_parse_failed",
                reason="payload_extraction_failed",
                output_type=type(final_output).__name__,
            )
            return []

        raw_items = payload.get("rankings", [])

        if not isinstance(raw_items, list):
            self._trace.emit(
                "ranking_parse_failed",
                reason="rankings_not_a_list",
                rankings_type=type(raw_items).__name__,
            )
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

    def _extract_payload(self, final_output: Any) -> dict[str, Any] | None:
        if final_output is None:
            return None

        if isinstance(final_output, dict):
            if "rankings" in final_output:
                return final_output
            for value in final_output.values():
                if isinstance(value, list):
                    return {"rankings": value}
            return final_output

        if isinstance(final_output, list):
            return {"rankings": final_output}

        try:
            payload = parse_json_object_response(final_output)

            if "rankings" in payload:
                return payload

            for value in payload.values():
                if isinstance(value, list):
                    return {"rankings": value}

            return payload
        except Exception:
            return None

    def _merge_with_heuristic(
        self,
        parsed_rankings: list[RankedSource],
        heuristic_rankings: list[RankedSource],
        source_map: dict[str, Source],
        exclude_ids: set[str] | None = None,
    ) -> list[RankedSource]:
        excluded = set(exclude_ids or set())
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
            if source_id in seen or source_id in excluded:
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
            return None

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value))