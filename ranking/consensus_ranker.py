from __future__ import annotations

from pathlib import Path
from typing import Any

from core import constants
from core.config import get_settings
from core.models import Difficulty, RankedSource, Source
from core.ranking_config import get_ranking_config
from core.schemas import EnsembleRequestSchema, SearchQuerySchema
from llm.aggregator import ConsensusAggregator
from llm.ensemble import EnsembleEngine
from llm.parser import parse_json_object_response
from llm.prompts import (
    build_source_ranking_prompt,
    system_research_ranker,
)
from llm.router import choose_models_for_task, get_judge_model_reference
from ranking.difficulty_classifier import DifficultyClassifier
from ranking.scorer import HeuristicScorer
from search.query_tokens import concept_tokens, tokens_match
from utils.logger import get_logger, get_trace_logger
from utils.text import clean_text, truncate_text


_DEFAULTS = {
    "min_relevance_score": 0.20,
    "min_llm_score": 0.25,
    "quality_floor_min_score": 0.10,
    "quality_floor_min_sources": 3,
    "quality_floor_mode": "adaptive",
    "quality_floor_band_ratio": 0.35,
    "relevance_retention_ratio": 0.30,
    "relevance_absolute_floor": 0.05,
    "topic_presence_min_overlap": 1,
    "topic_presence_min_coverage": 0.15,
    "topic_presence_allow_beginner_exemption": True,
    "beginner_rescue_max": 3,
    "beginner_rescue_min_score_ratio": 0.55,
    "beginner_rescue_min_relevance": 0.15,
}

_CREDIBILITY_WARNING_MARKERS: tuple[str, ...] = (
    "artificial soul",
    "conscious identity",
    "consciousness upload",
    "collective human experience",
    "quantum mind",
    "quantum consciousness",
    "universal love",
    "universal consciousness",
    "the pinnacle",
    "pinnacle of",
    "transcendent",
    "first prototype of",
    "novel epistemological",
    "epistemological framework for",
    "soul of",
    "mystical",
    "metaphysical framework for",
)

_THRESHOLD_CACHE: dict[str, Any] | None = None


def _coerce_threshold(value: Any, fallback: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return float(fallback)


def _coerce_int(value: Any, fallback: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed >= 0 else int(fallback)
    except Exception:
        return int(fallback)


def _coerce_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        text = value.strip().lower()

        if text in {"true", "yes", "on", "1"}:
            return True

        if text in {"false", "no", "off", "0"}:
            return False

    return bool(fallback)


def _load_consensus_thresholds() -> dict[str, Any]:
    global _THRESHOLD_CACHE

    if _THRESHOLD_CACHE is not None:
        return dict(_THRESHOLD_CACHE)

    values: dict[str, Any] = dict(_DEFAULTS)

    try:
        import yaml

        config_path = (
            Path(__file__).resolve().parent.parent
            / "configs"
            / "ranking.yaml"
        )

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}

            section = (
                loaded.get("consensus", {})
                if isinstance(loaded, dict)
                else {}
            )

            if isinstance(section, dict):
                values["min_relevance_score"] = _coerce_threshold(
                    section.get("min_relevance_score"),
                    values["min_relevance_score"],
                )
                values["min_llm_score"] = _coerce_threshold(
                    section.get("min_llm_score"),
                    values["min_llm_score"],
                )

                quality = section.get("quality_floor")

                if isinstance(quality, dict):
                    values[
                        "quality_floor_min_score"
                    ] = _coerce_threshold(
                        quality.get("min_score"),
                        values["quality_floor_min_score"],
                    )
                    values[
                        "quality_floor_min_sources"
                    ] = _coerce_int(
                        quality.get("min_sources"),
                        values["quality_floor_min_sources"],
                    )

                    mode = str(quality.get("mode") or "").strip().lower()

                    if mode in {"adaptive", "strict", "proportional"}:
                        values["quality_floor_mode"] = mode

                    values["quality_floor_enabled"] = _coerce_bool(
                        quality.get("enabled"),
                        True,
                    )
                    values[
                        "quality_floor_band_ratio"
                    ] = _coerce_threshold(
                        quality.get("band_ratio"),
                        values["quality_floor_band_ratio"],
                    )
                else:
                    values["quality_floor_enabled"] = True

                retention = section.get("relevance_retention")

                if isinstance(retention, dict):
                    values[
                        "relevance_retention_ratio"
                    ] = _coerce_threshold(
                        retention.get("ratio"),
                        values["relevance_retention_ratio"],
                    )
                    values[
                        "relevance_absolute_floor"
                    ] = _coerce_threshold(
                        retention.get("absolute_floor"),
                        values["relevance_absolute_floor"],
                    )

                topic = section.get("topic_presence")

                if isinstance(topic, dict):
                    values[
                        "topic_presence_min_overlap"
                    ] = _coerce_int(
                        topic.get("min_overlap"),
                        values["topic_presence_min_overlap"],
                    )
                    values[
                        "topic_presence_min_coverage"
                    ] = _coerce_threshold(
                        topic.get("min_coverage"),
                        values["topic_presence_min_coverage"],
                    )
                    values[
                        "topic_presence_allow_beginner_exemption"
                    ] = _coerce_bool(
                        topic.get("allow_beginner_exemption"),
                        values[
                            "topic_presence_allow_beginner_exemption"
                        ],
                    )
                    values["topic_presence_enabled"] = _coerce_bool(
                        topic.get("enabled"),
                        True,
                    )
                else:
                    values["topic_presence_enabled"] = True

                rescue = section.get("beginner_rescue")

                if isinstance(rescue, dict):
                    values["beginner_rescue_max"] = _coerce_int(
                        rescue.get("max_rescues"),
                        values["beginner_rescue_max"],
                    )
                    values[
                        "beginner_rescue_min_score_ratio"
                    ] = _coerce_threshold(
                        rescue.get("min_score_ratio"),
                        values["beginner_rescue_min_score_ratio"],
                    )
                    values[
                        "beginner_rescue_min_relevance"
                    ] = _coerce_threshold(
                        rescue.get("min_relevance"),
                        values["beginner_rescue_min_relevance"],
                    )
                    values["beginner_rescue_enabled"] = _coerce_bool(
                        rescue.get("enabled"),
                        True,
                    )
                else:
                    values["beginner_rescue_enabled"] = True
    except Exception:
        pass

    if "quality_floor_enabled" not in values:
        values["quality_floor_enabled"] = True

    if "topic_presence_enabled" not in values:
        values["topic_presence_enabled"] = True

    if "beginner_rescue_enabled" not in values:
        values["beginner_rescue_enabled"] = True

    _THRESHOLD_CACHE = dict(values)
    return dict(values)


class ConsensusRanker:
    RELEVANCE_RETENTION_RATIO = 0.40
    RELEVANCE_ABSOLUTE_FLOOR = 0.10
    QUALITY_FLOOR_BAND_RATIO = 0.50
    BEGINNER_RESCUE_MIN_RELEVANCE = 0.20

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
        self._max_llm_sources = max(
            3,
            max_llm_sources or consensus_config.max_llm_sources,
        )
        self._use_llm = use_llm

        resolved_relevance = (
            min_relevance_score
            if min_relevance_score is not None
            else consensus_config.min_relevance_score
            if consensus_config.min_relevance_score > 0
            else thresholds["min_relevance_score"]
        )

        resolved_llm_score = (
            min_llm_score
            if min_llm_score is not None
            else consensus_config.min_llm_score
            if consensus_config.min_llm_score > 0
            else thresholds["min_llm_score"]
        )

        self._min_relevance_score = _coerce_threshold(
            resolved_relevance,
            thresholds["min_relevance_score"],
        )
        self._min_llm_score = _coerce_threshold(
            resolved_llm_score,
            thresholds["min_llm_score"],
        )

        self._relevance_retention_ratio = _coerce_threshold(
            thresholds.get("relevance_retention_ratio"),
            self.RELEVANCE_RETENTION_RATIO,
        )
        self._relevance_absolute_floor = _coerce_threshold(
            thresholds.get("relevance_absolute_floor"),
            self.RELEVANCE_ABSOLUTE_FLOOR,
        )

        self._quality_floor_enabled = bool(
            thresholds.get("quality_floor_enabled", True)
        )
        self._quality_floor_min_score = _coerce_threshold(
            thresholds.get("quality_floor_min_score"),
            _DEFAULTS["quality_floor_min_score"],
        )
        self._quality_floor_min_sources = _coerce_int(
            thresholds.get("quality_floor_min_sources"),
            _DEFAULTS["quality_floor_min_sources"],
        )
        self._quality_floor_mode = str(
            thresholds.get("quality_floor_mode")
            or _DEFAULTS["quality_floor_mode"]
        ).strip().lower()
        self._quality_floor_band_ratio = _coerce_threshold(
            thresholds.get("quality_floor_band_ratio"),
            self.QUALITY_FLOOR_BAND_RATIO,
        )

        self._topic_presence_enabled = bool(
            thresholds.get("topic_presence_enabled", True)
        )
        self._topic_presence_min_overlap = _coerce_int(
            thresholds.get("topic_presence_min_overlap"),
            _DEFAULTS["topic_presence_min_overlap"],
        )
        self._topic_presence_min_coverage = _coerce_threshold(
            thresholds.get("topic_presence_min_coverage"),
            _DEFAULTS["topic_presence_min_coverage"],
        )
        self._topic_presence_allow_beginner_exemption = bool(
            thresholds.get(
                "topic_presence_allow_beginner_exemption", True
            )
        )

        self._beginner_rescue_enabled = bool(
            thresholds.get("beginner_rescue_enabled", True)
        )
        self._beginner_rescue_max = _coerce_int(
            thresholds.get("beginner_rescue_max"),
            _DEFAULTS["beginner_rescue_max"],
        )
        self._beginner_rescue_min_score_ratio = _coerce_threshold(
            thresholds.get("beginner_rescue_min_score_ratio"),
            _DEFAULTS["beginner_rescue_min_score_ratio"],
        )
        self._beginner_rescue_min_relevance = _coerce_threshold(
            thresholds.get("beginner_rescue_min_relevance"),
            self.BEGINNER_RESCUE_MIN_RELEVANCE,
        )

        self._logger = get_logger("ranking.consensus_ranker")
        self._trace = get_trace_logger()

    async def rank(
        self,
        sources: list[Any],
        query: SearchQuerySchema | str | dict[str, Any],
        model_references: list[Any] | None = None,
        judge_model: Any | None = None,
        mode: str = "",
    ) -> list[RankedSource]:
        query_schema = self._coerce_query(query)
        valid_sources = [
            source for source in sources if isinstance(source, Source)
        ]

        if not valid_sources:
            self._trace.emit("ranking_no_sources")
            return []

        classified_sources = self._classifier.classify_sources(
            valid_sources,
            level=query_schema.level,
        )

        classified_sources = self._apply_beginner_platform_override(
            classified_sources
        )

        difficulty_counts: dict[str, int] = {}

        for source in classified_sources:
            difficulty_value = (
                source.difficulty.value if source.difficulty else "unknown"
            )
            difficulty_counts[difficulty_value] = (
                difficulty_counts.get(difficulty_value, 0) + 1
            )

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
            relevance_by_id[item.source.source_id] = (
                self._scorer.relevance_factor(item.source, query_schema)
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
                    broadened_relevance_by_id[
                        item.source.source_id
                    ] = self._scorer.relevance_factor(
                        item.source, broadened_query
                    )

                broadened_best = max(
                    broadened_relevance_by_id.values(), default=0.0
                )

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

        tracked_drops: list[tuple[RankedSource, str]] = []

        relevant_ranked, relevance_drops = (
            self._apply_proportional_relevance_gate(
                heuristic_ranked,
                relevance_by_id,
                best_relevance,
                query_schema=query_schema,
            )
        )
        tracked_drops.extend(relevance_drops)

        if not relevant_ranked:
            relevant_ranked = heuristic_ranked

        if self._topic_presence_enabled:
            relevant_ranked, topic_drops = (
                self._apply_topic_presence_gate(
                    relevant_ranked, query_schema
                )
            )
            tracked_drops.extend(topic_drops)

        resolved_mode = self._resolve_mode(mode)
        skip_llm = self._should_skip_llm(resolved_mode)

        if len(relevant_ranked) <= 1 or not self._use_llm or skip_llm:
            aligned = self._enforce_alignment(
                relevant_ranked,
                query_schema,
                relevance_by_id,
            )
            return self._finalize_quality(
                aligned,
                query_schema,
                tracked_drops,
                relevance_by_id,
            )

        single_model = self._is_single_model_mode(resolved_mode)

        candidate_limit = (
            self._single_model_source_limit(resolved_mode)
            if single_model
            else self._max_llm_sources
        )

        candidate_sources = [
            item.source for item in relevant_ranked[:candidate_limit]
        ]

        try:
            if model_references:
                resolved_models = list(model_references)
            else:
                resolved_models = choose_models_for_task(
                    "ranking",
                    limit=1 if single_model else self._max_models,
                )
        except Exception as exc:
            self._logger.warning(
                f"Consensus model selection failed: {exc}"
            )
            self._trace.emit(
                "ranking_fallback",
                reason="model_selection_failed",
                error=str(exc),
                fallback="heuristic",
            )
            aligned = self._enforce_alignment(
                relevant_ranked, query_schema, relevance_by_id
            )
            return self._finalize_quality(
                aligned,
                query_schema,
                tracked_drops,
                relevance_by_id,
            )

        if not resolved_models:
            self._trace.emit(
                "ranking_fallback",
                reason="no_models_available",
                fallback="heuristic",
            )
            aligned = self._enforce_alignment(
                relevant_ranked, query_schema, relevance_by_id
            )
            return self._finalize_quality(
                aligned,
                query_schema,
                tracked_drops,
                relevance_by_id,
            )

        if single_model:
            resolved_models = resolved_models[:1]

        self._trace.emit(
            "llm_ranking_started",
            models=[
                {
                    "provider": m.provider.value,
                    "model_id": m.model_id,
                }
                for m in resolved_models
            ],
            candidate_count=len(candidate_sources),
            single_model=single_model,
        )

        try:
            source_map = {
                source.source_id: source
                for source in candidate_sources
            }
            serialized_sources = self._serialize_sources(
                candidate_sources
            )

            prompt = build_source_ranking_prompt(
                topic=query_schema.topic,
                goal=query_schema.goal,
                level=query_schema.level,
                sources=serialized_sources,
            )

            prompt += (
                f"\nCRITICAL: You MUST rank ALL "
                f"{len(candidate_sources)} provided sources. Do NOT "
                f"skip any source. Return exactly "
                f"{len(candidate_sources)} ranking entries in the "
                f"rankings array. Every source_id from the input MUST "
                f"appear in your output.\n"
            )

            prompt += (
                "\nCRITICAL LEVEL ALIGNMENT RULES:\n"
                "- If the user asks for basics, beginner, "
                "introduction, tutorial, or learning from basics, "
                "beginner-friendly sources MUST outrank advanced "
                "research papers.\n"
                "- An advanced research paper MUST NEVER be rank 1 "
                "when a beginner or intermediate educational source "
                "exists.\n"
                "- Prefer OpenStax, LibreTexts, Wikipedia, Wikibooks, "
                "Wikiversity, MIT OCW, Open Library, and Internet "
                "Archive for beginner requests.\n"
                "- Advanced research papers are allowed only after "
                "beginner and intermediate sources for beginner "
                "requests.\n"
            )

            prompt += (
                "\nCRITICAL TOPIC ALIGNMENT RULES:\n"
                "- Rank sources that are directly about the primary "
                "concept highest.\n"
                "- Sources that only mention the topic in passing "
                "MUST receive low scores.\n"
                "- Do not rank tangential advanced papers above "
                "directly relevant beginner material.\n"
            )

            prompt += (
                "\nSCORE REJECTION RULE:\n"
                f"- If a source is off-topic, assign score < "
                f"{self._min_llm_score:.2f} so it is dropped.\n"
                "- Do NOT assign high scores just because a source "
                "has many citations or is well known.\n"
            )

            settings = get_settings()

            resolved_judge: Any = None

            if not single_model:
                resolved_judge = judge_model

                if resolved_judge is None:
                    try:
                        resolved_judge = get_judge_model_reference()
                    except Exception:
                        resolved_judge = None

            required_output_tokens = max(
                constants.RANKER_MAX_OUTPUT_TOKENS_FLOOR,
                len(candidate_sources)
                * constants.RANKER_MAX_OUTPUT_TOKENS_PER_SOURCE,
            )

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
                aligned = self._enforce_alignment(
                    relevant_ranked, query_schema, relevance_by_id
                )
                return self._finalize_quality(
                    aligned,
                    query_schema,
                    tracked_drops,
                    relevance_by_id,
                )

            if single_model:
                final_output = (
                    ensemble_result.final_output
                    if ensemble_result.final_output is not None
                    else self._first_successful_vote_output(
                        ensemble_result
                    )
                )

                parsed_rankings = self._parse_final_output(
                    final_output, source_map
                )

                if parsed_rankings:
                    kept_rankings = [
                        ranked
                        for ranked in parsed_rankings
                        if ranked.score >= self._min_llm_score
                    ]

                    dropped_count = (
                        len(parsed_rankings) - len(kept_rankings)
                    )

                    if dropped_count > 0:
                        self._trace.emit(
                            "llm_ranking_low_score_dropped",
                            dropped_count=dropped_count,
                            threshold=self._min_llm_score,
                        )

                    llm_seen_ids = {
                        ranked.source.source_id
                        for ranked in parsed_rankings
                    }

                    if kept_rankings:
                        merged = self._merge_with_heuristic(
                            kept_rankings,
                            relevant_ranked,
                            source_map,
                            exclude_ids=llm_seen_ids,
                        )

                        if merged:
                            aligned = self._enforce_alignment(
                                merged,
                                query_schema,
                                relevance_by_id,
                            )

                            self._trace.emit(
                                "llm_ranking_success",
                                llm_ranked=len(kept_rankings),
                                llm_dropped=dropped_count,
                                total_merged=len(aligned),
                                judge_used=False,
                                single_model=True,
                            )

                            return self._finalize_quality(
                                aligned,
                                query_schema,
                                tracked_drops,
                                relevance_by_id,
                            )

                    self._trace.emit(
                        "llm_ranking_all_rejected",
                        threshold=self._min_llm_score,
                        fallback="heuristic",
                    )

                self._trace.emit(
                    "ranking_fallback",
                    reason="single_model_parse_empty",
                    fallback="heuristic",
                )

                aligned = self._enforce_alignment(
                    relevant_ranked, query_schema, relevance_by_id
                )
                return self._finalize_quality(
                    aligned,
                    query_schema,
                    tracked_drops,
                    relevance_by_id,
                )

            async with ConsensusAggregator() as aggregator:
                consensus_result = await aggregator.aggregate(
                    request,
                    ensemble_result,
                )

            if consensus_result is None:
                self._trace.emit(
                    "ranking_fallback",
                    reason="consensus_returned_none",
                    fallback="heuristic",
                )
                aligned = self._enforce_alignment(
                    relevant_ranked, query_schema, relevance_by_id
                )
                return self._finalize_quality(
                    aligned,
                    query_schema,
                    tracked_drops,
                    relevance_by_id,
                )

            self._trace.emit(
                "llm_ranking_consensus",
                agreement_score=consensus_result.agreement_score,
                judge_used=consensus_result.judge_used,
                vote_count=len(consensus_result.votes),
            )

            parsed_rankings = self._parse_final_output(
                consensus_result.final_output, source_map
            )

            vote_candidates = self._parse_votes(
                getattr(consensus_result, "votes", []) or [],
                source_map,
            )

            if vote_candidates:
                final_alignment_score = self._alignment_score(
                    parsed_rankings, query_schema
                )

                best_vote = max(
                    vote_candidates,
                    key=lambda item: self._alignment_score(
                        item, query_schema
                    ),
                )

                best_vote_alignment = self._alignment_score(
                    best_vote, query_schema
                )

                if (
                    best_vote_alignment
                    > final_alignment_score + 0.10
                ):
                    parsed_rankings = best_vote

                    self._trace.emit(
                        "ranking_level_aligned_vote_selected",
                        final_alignment=round(
                            final_alignment_score, 4
                        ),
                        best_vote_alignment=round(
                            best_vote_alignment, 4
                        ),
                    )

            if parsed_rankings:
                kept_rankings = [
                    ranked
                    for ranked in parsed_rankings
                    if ranked.score >= self._min_llm_score
                ]

                dropped_count = (
                    len(parsed_rankings) - len(kept_rankings)
                )

                if dropped_count > 0:
                    self._trace.emit(
                        "llm_ranking_low_score_dropped",
                        dropped_count=dropped_count,
                        threshold=self._min_llm_score,
                    )

                llm_seen_ids = {
                    ranked.source.source_id
                    for ranked in parsed_rankings
                }

                if kept_rankings:
                    merged = self._merge_with_heuristic(
                        kept_rankings,
                        relevant_ranked,
                        source_map,
                        exclude_ids=llm_seen_ids,
                    )

                    if merged:
                        aligned = self._enforce_alignment(
                            merged,
                            query_schema,
                            relevance_by_id,
                        )

                        self._trace.emit(
                            "llm_ranking_success",
                            llm_ranked=len(kept_rankings),
                            llm_dropped=dropped_count,
                            total_merged=len(aligned),
                            judge_used=consensus_result.judge_used,
                        )

                        return self._finalize_quality(
                            aligned,
                            query_schema,
                            tracked_drops,
                            relevance_by_id,
                        )
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

            aligned = self._enforce_alignment(
                relevant_ranked, query_schema, relevance_by_id
            )
            return self._finalize_quality(
                aligned,
                query_schema,
                tracked_drops,
                relevance_by_id,
            )
        except Exception as exc:
            self._logger.warning(
                f"Consensus ranking failed, using heuristic "
                f"fallback: {exc}"
            )

            self._trace.emit(
                "ranking_fallback",
                reason="llm_exception",
                error=str(exc),
                fallback="heuristic",
            )

            aligned = self._enforce_alignment(
                relevant_ranked, query_schema, relevance_by_id
            )
            return self._finalize_quality(
                aligned,
                query_schema,
                tracked_drops,
                relevance_by_id,
            )

    def _resolve_mode(self, mode: str) -> str:
        text = str(mode or "").strip().lower()

        if text in constants.VALID_MODES:
            return text

        return constants.DEFAULT_MODE

    def _should_skip_llm(self, mode: str) -> bool:
        mode_defaults = constants.MODE_DEFAULTS.get(
            mode, constants.MODE_DEFAULTS[constants.DEFAULT_MODE]
        )
        return bool(mode_defaults.get("skip_ranking_llm", False))

    def _is_single_model_mode(self, mode: str) -> bool:
        return mode in (constants.MODE_FAST, constants.MODE_BALANCED)

    def _single_model_source_limit(self, mode: str) -> int:
        mode_defaults = constants.MODE_DEFAULTS.get(
            mode, constants.MODE_DEFAULTS[constants.DEFAULT_MODE]
        )

        try:
            return max(
                3, int(mode_defaults.get("top_n", 10))
            )
        except Exception:
            return 10

    def _first_successful_vote_output(
        self,
        ensemble_result: Any,
    ) -> Any:
        for vote in getattr(ensemble_result, "votes", []) or []:
            if getattr(vote, "success", False):
                return getattr(vote, "output", None)

        return None

    def _apply_proportional_relevance_gate(
        self,
        rankings: list[RankedSource],
        relevance_by_id: dict[str, float],
        best_relevance: float,
        query_schema: SearchQuerySchema | None = None,
    ) -> tuple[list[RankedSource], list[tuple[RankedSource, str]]]:
        if not rankings:
            return rankings, []

        if best_relevance <= 0:
            return rankings, []

        retention_ratio = self._relevance_retention_ratio

        if query_schema is not None:
            retention_ratio = self._effective_retention_ratio(
                query_schema
            )

        effective_floor = max(
            best_relevance * retention_ratio,
            self._relevance_absolute_floor,
        )

        passed: list[RankedSource] = []
        dropped: list[tuple[RankedSource, str]] = []

        for ranked in rankings:
            relevance = relevance_by_id.get(
                ranked.source.source_id, 0.0
            )

            if relevance >= effective_floor:
                passed.append(ranked)
            else:
                dropped.append((ranked, "relevance"))

        min_sources = max(3, self._quality_floor_min_sources)

        if len(passed) < min_sources:
            rescued = sorted(
                dropped,
                key=lambda item: -relevance_by_id.get(
                    item[0].source.source_id, 0.0
                ),
            )

            needed = min_sources - len(passed)
            rescued_batch = rescued[:needed]

            for ranked, _origin in rescued_batch:
                passed.append(ranked)

            dropped = rescued_batch[len(rescued_batch):]

            if rescued_batch:
                self._trace.emit(
                    "ranking_relevance_gate_rescue",
                    passed_before=len(passed) - len(rescued_batch),
                    rescued=len(rescued_batch),
                    effective_floor=round(effective_floor, 4),
                    min_sources=min_sources,
                )

        self._trace.emit(
            "ranking_relevance_gate_proportional",
            best_relevance=round(best_relevance, 4),
            retention_ratio=round(retention_ratio, 4),
            absolute_floor=round(
                self._relevance_absolute_floor, 4
            ),
            effective_floor=round(effective_floor, 4),
            kept_count=len(passed),
            dropped_count=len(dropped),
        )

        return passed, dropped

    def _apply_topic_presence_gate(
        self,
        rankings: list[RankedSource],
        query_schema: SearchQuerySchema,
    ) -> tuple[list[RankedSource], list[tuple[RankedSource, str]]]:
        if not rankings:
            return rankings, []

        query_tokens = self._concept_token_set(query_schema)

        if not query_tokens:
            return rankings, []

        passed: list[RankedSource] = []
        dropped: list[tuple[RankedSource, str]] = []

        for ranked in rankings:
            source = ranked.source
            title = clean_text(
                getattr(source, "title", "") or ""
            ).lower()
            text = self._combined_source_text(source)
            source_tokens = self._tokenize_for_overlap(text)

            overlap = 0

            for token in query_tokens:
                for candidate in source_tokens:
                    if tokens_match(token, candidate):
                        overlap += 1
                        break

            coverage = (
                overlap / max(1, len(query_tokens))
                if query_tokens
                else 1.0
            )

            head_position = self._title_head_position_score(
                title, query_tokens
            )

            meets_overlap = overlap >= self._topic_presence_min_overlap
            meets_coverage = (
                coverage >= self._topic_presence_min_coverage
            )
            meets_position = head_position >= 0.25

            if (
                (meets_overlap and meets_position)
                or (meets_coverage and meets_position)
            ):
                passed.append(ranked)
            else:
                dropped.append((ranked, "topic"))

        min_sources = max(3, self._quality_floor_min_sources)

        if len(passed) < min_sources:
            passed.sort(key=lambda item: -item.score)
            rescued = sorted(
                dropped, key=lambda item: -item[0].score
            )

            needed = min_sources - len(passed)
            rescued_batch = rescued[:needed]

            for ranked, _origin in rescued_batch:
                passed.append(ranked)

            dropped = rescued_batch[len(rescued_batch):]

            if rescued_batch:
                self._trace.emit(
                    "ranking_topic_gate_rescue",
                    passed=len(passed),
                    rescued=len(rescued_batch),
                    min_sources=min_sources,
                )

        self._trace.emit(
            "ranking_topic_gate_applied",
            input_count=len(rankings),
            passed_count=len(passed),
            dropped_count=len(dropped),
            min_overlap=self._topic_presence_min_overlap,
            min_coverage=self._topic_presence_min_coverage,
            query_token_count=len(query_tokens),
            uses_title_position=True,
        )

        return passed, dropped

    def _title_head_position_score(
        self,
        title: str,
        query_tokens: set[str],
    ) -> float:
        if not title or not query_tokens:
            return 0.0

        words = title.split()

        if not words:
            return 0.0

        cutoff = max(2, int(len(words) * 0.7))
        head = " ".join(words[:cutoff])

        matches = 0

        for token in query_tokens:
            if not token:
                continue

            if token in head:
                matches += 1

        return matches / max(1, len(query_tokens))

    def _finalize_quality(
        self,
        rankings: list[RankedSource],
        query_schema: SearchQuerySchema,
        tracked_drops: list[tuple[RankedSource, str]],
        relevance_by_id: dict[str, float],
    ) -> list[RankedSource]:
        if not rankings:
            return rankings

        ranked, floor_drops = self._apply_quality_floor(rankings)
        tracked_drops = tracked_drops + floor_drops

        ranked = self._maybe_rescue_beginner_sources(
            ranked,
            query_schema,
            tracked_drops,
            relevance_by_id,
        )

        return self._renumber(ranked)

    def _apply_quality_floor(
        self,
        rankings: list[RankedSource],
    ) -> tuple[list[RankedSource], list[tuple[RankedSource, str]]]:
        if not self._quality_floor_enabled or not rankings:
            return rankings, []

        min_score = self._quality_floor_min_score
        min_sources = max(1, self._quality_floor_min_sources)
        band_ratio = self._quality_floor_band_ratio

        sorted_rankings = sorted(
            rankings,
            key=lambda item: (
                -float(item.score),
                -float(item.confidence),
            ),
        )

        best_score = float(sorted_rankings[0].score)
        band_floor = best_score * band_ratio
        dynamic_floor = max(min_score, band_floor)

        keep = [
            ranked
            for ranked in sorted_rankings
            if float(ranked.score) >= dynamic_floor
        ]

        if len(keep) >= min_sources:
            dropped_ids = {
                r.source.source_id for r in rankings
            } - {r.source.source_id for r in keep}

            drops = [
                (ranked, "quality_floor")
                for ranked in rankings
                if ranked.source.source_id in dropped_ids
            ]

            self._trace.emit(
                "ranking_quality_floor_applied",
                best_score=round(best_score, 4),
                band_ratio=band_ratio,
                band_floor=round(band_floor, 4),
                min_score=min_score,
                dynamic_floor=round(dynamic_floor, 4),
                input_count=len(rankings),
                kept_count=len(keep),
                dropped_count=len(drops),
            )

            return keep, drops

        target_count = min(min_sources, len(sorted_rankings))
        kept = sorted_rankings[:target_count]

        dropped_ids = {
            r.source.source_id for r in rankings
        } - {r.source.source_id for r in kept}

        drops = [
            (ranked, "quality_floor")
            for ranked in rankings
            if ranked.source.source_id in dropped_ids
        ]

        self._trace.emit(
            "ranking_quality_floor_rescue",
            best_score=round(best_score, 4),
            band_floor=round(band_floor, 4),
            min_score=min_score,
            dynamic_floor=round(dynamic_floor, 4),
            input_count=len(rankings),
            kept_count=len(kept),
            min_sources=min_sources,
            note="rescue_below_dynamic_floor",
        )

        return kept, drops

    def _maybe_rescue_beginner_sources(
        self,
        rankings: list[RankedSource],
        query_schema: SearchQuerySchema,
        tracked_drops: list[tuple[RankedSource, str]],
        relevance_by_id: dict[str, float],
    ) -> list[RankedSource]:
        if (
            not self._beginner_rescue_enabled
            or not rankings
            or not tracked_drops
        ):
            return rankings

        beginner_request = self._is_beginner_request(query_schema)
        mixed_request = self._is_mixed_request(query_schema)

        if not beginner_request and not mixed_request:
            return rankings

        has_beginner = any(
            self._is_beginner_source(ranked.source)
            for ranked in rankings
        )

        if has_beginner:
            return rankings

        best_score = max(
            (float(item.score) for item in rankings), default=0.0
        )

        if best_score <= 0:
            return rankings

        effective_min_ratio = self._beginner_rescue_min_score_ratio

        if mixed_request and not beginner_request:
            effective_min_ratio = min(
                effective_min_ratio, 0.50
            )

        min_acceptable = best_score * effective_min_ratio

        candidates: list[RankedSource] = []
        origin_counter: dict[str, int] = {
            "relevance": 0,
            "topic": 0,
            "quality_floor": 0,
        }

        seen_ids: set[str] = set()

        for ranked, origin in tracked_drops:
            source_id = ranked.source.source_id

            if source_id in seen_ids:
                continue

            if not self._is_beginner_source(ranked.source):
                continue

            relevance = relevance_by_id.get(source_id, 0.0)

            if relevance < self._beginner_rescue_min_relevance:
                continue

            seen_ids.add(source_id)

            floored_score = max(
                float(ranked.score), min_acceptable
            )

            candidates.append(
                ranked.model_copy(update={"score": floored_score})
            )

            origin_counter[origin] = (
                origin_counter.get(origin, 0) + 1
            )

        if not candidates:
            return rankings

        candidates.sort(
            key=lambda item: (
                -float(item.score),
                -float(item.confidence),
            )
        )

        effective_max = self._beginner_rescue_max

        if mixed_request and not beginner_request:
            effective_max = max(effective_max, 3)

        rescued = candidates[:effective_max]
        combined = list(rankings) + rescued

        combined.sort(
            key=lambda item: (
                -float(item.score),
                -float(item.confidence),
            )
        )

        self._trace.emit(
            "ranking_beginner_rescue_applied",
            rescue_count=len(rescued),
            total_after_rescue=len(combined),
            best_score=round(best_score, 4),
            min_acceptable=round(min_acceptable, 4),
            rescued_from_relevance_gate=origin_counter.get(
                "relevance", 0
            ),
            rescued_from_topic_gate=origin_counter.get("topic", 0),
            rescued_from_quality_floor=origin_counter.get(
                "quality_floor", 0
            ),
        )

        return combined

    def _renumber(
        self,
        rankings: list[RankedSource],
    ) -> list[RankedSource]:
        final: list[RankedSource] = []

        for index, ranked in enumerate(rankings, start=1):
            final.append(ranked.model_copy(update={"rank": index}))

        return final

    def _concept_token_set(
        self,
        query_schema: SearchQuerySchema,
    ) -> set[str]:
        tokens: set[str] = set()

        for text in (
            getattr(query_schema, "primary_concept", "") or "",
            getattr(query_schema, "corrected_topic", "") or "",
            getattr(query_schema, "topic", "") or "",
        ):
            if not text:
                continue

            try:
                extracted = concept_tokens(text)
            except Exception:
                extracted = set()

            for token in extracted:
                token_text = str(token or "").strip().lower()

                if token_text and len(token_text) >= 3:
                    tokens.add(token_text)

        for keyword in getattr(query_schema, "keywords", []) or []:
            keyword_text = clean_text(keyword).lower()

            if not keyword_text:
                continue

            for word in keyword_text.split():
                if len(word) >= 3:
                    tokens.add(word)

        return tokens
    def _effective_retention_ratio(
        self,
        query_schema: SearchQuerySchema,
    ) -> float:
        try:
            tokens = self._concept_token_set(query_schema)
        except Exception:
            tokens = set()

        token_count = len(tokens)

        if token_count >= 5:
            return min(self._relevance_retention_ratio, 0.20)

        if token_count >= 4:
            return min(self._relevance_retention_ratio, 0.25)

        if token_count >= 3:
            return min(self._relevance_retention_ratio, 0.30)

        return min(self._relevance_retention_ratio, 0.35)
    def _combined_source_text(self, source: Source) -> str:
        parts = [
            clean_text(getattr(source, "title", "")),
            clean_text(getattr(source, "abstract", "")),
            clean_text(getattr(source, "summary", "")),
        ]

        metadata = getattr(source, "metadata", {}) or {}

        if isinstance(metadata, dict):
            for key in ("description", "topics", "tags"):
                value = metadata.get(key)

                if isinstance(value, str):
                    parts.append(clean_text(value))
                elif isinstance(value, (list, tuple, set)):
                    for item in value:
                        parts.append(clean_text(item))

        return " ".join(
            part for part in parts if part
        ).strip().lower()

    def _tokenize_for_overlap(self, text: str) -> set[str]:
        if not text:
            return set()

        import re

        raw = re.findall(r"[a-z0-9+#]+", text)
        return {token for token in raw if len(token) >= 3}

    def _broaden_query(
        self,
        query_schema: Any,
    ) -> dict[str, Any]:
        topic = clean_text(
            getattr(query_schema, "corrected_topic", "")
            or getattr(query_schema, "topic", "")
        )

        primary_concept = clean_text(
            getattr(query_schema, "primary_concept", "") or topic
        )

        raw_keywords = (
            getattr(query_schema, "keywords", []) or []
        )
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
            for item in getattr(
                query_schema, "short_search_queries", []
            )
            or []
            if clean_text(item)
        ]

        return {
            "topic": topic,
            "primary_concept": primary_concept,
            "keywords": keywords[:12],
            "short_search_queries": short_search_queries[:6],
        }

    def _credibility_penalty(self, source: Source) -> float:
        title = clean_text(getattr(source, "title", "")).lower()
        abstract = clean_text(
            getattr(source, "abstract", "")
        ).lower()
        text = f"{title} {abstract}"

        if not text.strip():
            return 1.0

        hits = 0

        for marker in _CREDIBILITY_WARNING_MARKERS:
            if marker in text:
                hits += 1

        if hits == 0:
            return 1.0

        if hits == 1:
            return 0.75

        if hits == 2:
            return 0.50

        return 0.35

    def _enforce_alignment(
        self,
        rankings: list[RankedSource],
        query_schema: SearchQuerySchema,
        relevance_by_id: dict[str, float],
    ) -> list[RankedSource]:
        if not rankings:
            return []

        adjusted: list[RankedSource] = []
        beginner_request = self._is_beginner_request(query_schema)
        advanced_request = (
            self._is_advanced_request(query_schema)
            and not beginner_request
        )

        for ranked in rankings:
            score = float(ranked.score)
            source_id = ranked.source.source_id
            relevance = relevance_by_id.get(source_id)

            if relevance is not None:
                if relevance < 0.20:
                    score *= 0.25

                if relevance < 0.05 and beginner_request:
                    score *= 0.10

            credibility = self._credibility_penalty(ranked.source)

            if credibility < 1.0:
                score *= credibility

                self._trace.emit(
                    "ranking_credibility_penalty_applied",
                    source_id=source_id,
                    title=clean_text(
                        getattr(ranked.source, "title", "")
                    )[:120],
                    penalty=round(credibility, 4),
                    score_before=round(float(ranked.score), 4),
                    score_after=round(score, 4),
                )

            score = max(0.0, min(1.0, score))
            adjusted.append(
                ranked.model_copy(update={"score": score})
            )

        if beginner_request:
            adjusted.sort(
                key=lambda item: (
                    self._beginner_band(item.source),
                    -item.score,
                    -item.confidence,
                )
            )
        elif advanced_request:
            adjusted.sort(
                key=lambda item: (
                    self._advanced_band(item.source),
                    -item.score,
                    -item.confidence,
                )
            )
        else:
            adjusted.sort(key=lambda item: -item.score)

        final: list[RankedSource] = []

        for index, item in enumerate(adjusted, start=1):
            reason = clean_text(item.reason or "")

            if beginner_request or advanced_request:
                reason = (
                    f"{reason}; level_aligned"
                    if reason
                    else "level_aligned"
                )

            final.append(
                item.model_copy(
                    update={
                        "rank": index,
                        "reason": reason or None,
                    }
                )
            )

        return final

    def _parse_votes(
        self,
        votes: list[Any],
        source_map: dict[str, Source],
    ) -> list[list[RankedSource]]:
        candidates: list[list[RankedSource]] = []

        for vote in votes or []:
            if not getattr(vote, "success", True):
                continue

            output = getattr(vote, "output", None)
            parsed = self._parse_final_output(output, source_map)

            if parsed:
                candidates.append(parsed)

        return candidates

    def _alignment_score(
        self,
        rankings: list[RankedSource],
        query_schema: SearchQuerySchema,
    ) -> float:
        if not rankings:
            return 0.0

        beginner_request = self._is_beginner_request(query_schema)
        advanced_request = self._is_advanced_request(query_schema)

        if not beginner_request and not advanced_request:
            return 0.50

        top = rankings[: min(5, len(rankings))]
        score = 0.0

        for index, ranked in enumerate(top):
            difficulty = getattr(ranked.source, "difficulty", None)
            weight = max(0.5, 1.0 - (index * 0.1))

            if beginner_request:
                if difficulty == Difficulty.BEGINNER:
                    score += 1.0 * weight
                elif difficulty == Difficulty.INTERMEDIATE:
                    score += 0.6 * weight
                elif difficulty == Difficulty.ADVANCED:
                    score += 0.0
                else:
                    score += 0.4 * weight
            else:
                if difficulty == Difficulty.ADVANCED:
                    score += 1.0 * weight
                elif difficulty == Difficulty.INTERMEDIATE:
                    score += 0.6 * weight
                elif difficulty == Difficulty.BEGINNER:
                    score += 0.1 * weight
                else:
                    score += 0.4 * weight

        return score / len(top)

    def _beginner_band(self, source: Source) -> int:
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

    def _advanced_band(self, source: Source) -> int:
        difficulty = getattr(source, "difficulty", None)

        if difficulty == Difficulty.ADVANCED:
            return 0

        if self._type_value(source) == "research_paper":
            return 0

        if difficulty == Difficulty.INTERMEDIATE:
            return 1

        if self._is_beginner_source(source):
            return 2

        return 1
    _BEGINNER_OVERRIDE_PLATFORMS = frozenset(
        {
            "wikipedia",
            "wikibooks",
            "wikiversity",
            "openstax",
            "libretexts",
            "mit_ocw",
            "open_library",
            "internet_archive",
        }
    )

    def _apply_beginner_platform_override(
        self,
        sources: list[Source],
    ) -> list[Source]:
        overridden: list[Source] = []

        for source in sources:
            platform = self._enum_value(
                getattr(source, "platform", "")
            ).lower()

            if platform not in self._BEGINNER_OVERRIDE_PLATFORMS:
                overridden.append(source)
                continue

            current = getattr(source, "difficulty", None)

            if current == Difficulty.BEGINNER:
                overridden.append(source)
                continue

            try:
                overridden.append(
                    source.model_copy(
                        update={"difficulty": Difficulty.BEGINNER}
                    )
                )
            except Exception:
                overridden.append(source)

        return overridden
    def _is_beginner_source(self, source: Source) -> bool:
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

    def _is_beginner_request(
        self,
        query_schema: SearchQuerySchema,
    ) -> bool:
        level_profile = str(
            getattr(query_schema, "level_profile", "") or ""
        ).strip().lower()

        if level_profile == "beginner":
            return True

        if level_profile == "advanced":
            return False

        level = clean_text(
            getattr(query_schema, "level", "") or ""
        ).lower()
        goal = clean_text(
            getattr(query_schema, "goal", "") or ""
        ).lower()
        topic = clean_text(
            getattr(query_schema, "topic", "") or ""
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

        text = f"{level} {goal} {topic}"

        markers = (
            "beginner",
            "basics",
            "basic",
            "fundamentals",
            "introduction",
            "intro",
            "tutorial",
            "course",
            "learn",
            "learning",
            "from basics",
            "from scratch",
            "step by step",
        )

        return any(marker in text for marker in markers)
    def _is_mixed_request(
        self,
        query_schema: SearchQuerySchema,
    ) -> bool:
        level_profile = str(
            getattr(query_schema, "level_profile", "") or ""
        ).strip().lower()

        if level_profile in {"mixed", "balanced"}:
            return True

        if level_profile == "beginner":
            return False

        if level_profile == "advanced":
            return False

        level = clean_text(
            getattr(query_schema, "level", "") or ""
        ).lower()
        goal = clean_text(
            getattr(query_schema, "goal", "") or ""
        ).lower()
        topic = clean_text(
            getattr(query_schema, "topic", "") or ""
        ).lower()

        text = f"{level} {goal} {topic}"

        mixed_markers = (
            "basics to advanced",
            "beginner to advanced",
            "fundamentals to advanced",
            "start to finish",
            "from basics",
            "from zero",
        )

        return any(marker in text for marker in mixed_markers)
    def _is_advanced_request(
        self,
        query_schema: SearchQuerySchema,
    ) -> bool:
        level_profile = str(
            getattr(query_schema, "level_profile", "") or ""
        ).strip().lower()

        if level_profile == "advanced":
            return True

        if level_profile in {"beginner", "mixed", "balanced"}:
            return False

        level = clean_text(
            getattr(query_schema, "level", "") or ""
        ).lower()
        goal = clean_text(
            getattr(query_schema, "goal", "") or ""
        ).lower()
        topic = clean_text(
            getattr(query_schema, "topic", "") or ""
        ).lower()

        text = f"{level} {goal} {topic}"

        has_beginner_marker = any(
            marker in text
            for marker in (
                "beginner",
                "basics",
                "basic",
                "fundamentals",
                "introduction",
                "intro",
                "from scratch",
                "from zero",
                "zero to hero",
            )
        )

        has_advanced_marker = any(
            marker in text
            for marker in (
                "advanced",
                "expert",
                "research paper",
                "state of the art",
                "state-of-the-art",
                "deep dive",
                "in depth",
                "in-depth",
            )
        )

        if has_beginner_marker and has_advanced_marker:
            return False

        if has_beginner_marker:
            return False

        return has_advanced_marker

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

    def _serialize_sources(
        self,
        sources: list[Source],
    ) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []

        for source in sources:
            entry: dict[str, Any] = {
                "source_id": source.source_id,
                "title": source.title,
                "url": source.url,
                "platform": self._enum_value(source.platform),
                "source_type": self._enum_value(source.source_type),
                "abstract": truncate_text(
                    source.abstract or "",
                    max_length=500,
                ),
                "year": source.year,
                "citation_count": source.citation_count,
                "has_code": source.has_code,
                "difficulty": (
                    self._enum_value(source.difficulty)
                    if source.difficulty is not None
                    else None
                ),
            }

            credibility = self._credibility_penalty(source)

            if credibility < 1.0:
                entry["credibility_warning"] = (
                    "source uses fringe or pseudoscientific language; "
                    "rank this source below sources with similar topic "
                    "coverage"
                )
                entry["credibility_penalty"] = round(credibility, 2)

            serialized.append(entry)

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

            score = self._normalize_unit_interval(
                item.get("score")
            )
            confidence = self._normalize_unit_interval(
                item.get("confidence")
            )

            reason = (
                truncate_text(
                    clean_text(item.get("reason", "")),
                    max_length=300,
                    suffix="",
                )
                or None
            )

            difficulty = self._parse_difficulty(
                item.get("difficulty")
            )
            updated_source = source

            if difficulty is not None:
                try:
                    updated_source = source.model_copy(
                        update={"difficulty": difficulty}
                    )
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
                confidence=(
                    0.5 if confidence is None else confidence
                ),
                reason=reason,
            )

            parsed.append(
                (max(1, provided_rank), ranked_source)
            )
            seen.add(source_id)

        parsed.sort(key=lambda item: (item[0], -item[1].score))

        return [ranked for _, ranked in parsed]

    def _extract_payload(
        self,
        final_output: Any,
    ) -> dict[str, Any] | None:
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

        return self._renumber(final)

    def _normalize_unit_interval(
        self,
        value: Any,
    ) -> float | None:
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

    def _parse_difficulty(
        self,
        value: Any,
    ) -> Difficulty | None:
        if value is None:
            return None

        try:
            return Difficulty(str(value).strip().lower())
        except Exception:
            return None

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value))

    def _type_value(self, source: Source) -> str:
        return self._enum_value(
            getattr(source, "source_type", "")
        ).lower()