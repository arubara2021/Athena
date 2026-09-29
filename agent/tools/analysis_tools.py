from __future__ import annotations
from typing import Any
from agent.tools.registry import ToolRegistry, ToolDefinition
from agent.tools.executor import ToolResult
from core.models import Source, SourcePlatform, SourceType, Difficulty
from core.schemas import SearchQuerySchema
from ranking.consensus_ranker import ConsensusRanker
from ranking.difficulty_classifier import DifficultyClassifier
from ranking.scorer import HeuristicScorer
from utils.logger import get_logger
from utils.text import clean_text

_logger = get_logger("agent.tools.analysis")
_ANALYSIS_TIMEOUT = 120.0
_RANK_TOKEN_COST = 8000
_CLASSIFY_TOKEN_COST = 1000


def _reconstruct_sources(sources_data: list[dict[str, Any]]) -> list[Source]:
    sources = []

    for item in sources_data:
        if not isinstance(item, dict):
            continue

        try:
            platform_str = str(item.get("platform", "web")).lower()
            try:
                platform = SourcePlatform(platform_str)
            except Exception:
                platform = SourcePlatform.WEB

            type_str = str(item.get("source_type", "other")).lower()
            try:
                source_type = SourceType(type_str)
            except Exception:
                source_type = SourceType.OTHER

            difficulty_str = item.get("difficulty")
            difficulty = None
            if difficulty_str:
                try:
                    difficulty = Difficulty(str(difficulty_str).lower())
                except Exception:
                    pass

            source = Source(
                source_id=str(item.get("source_id", "")),
                title=str(item.get("title", "")),
                url=str(item.get("url", "")),
                platform=platform,
                source_type=source_type,
                abstract=item.get("abstract"),
                year=item.get("year"),
                citation_count=item.get("citation_count"),
                has_code=item.get("has_code"),
                difficulty=difficulty,
                metadata=item.get("metadata", {}),
            )
            sources.append(source)

        except Exception as exc:
            _logger.warning(f"Failed to reconstruct source: {exc}")
            continue

    return sources


async def rank_sources(
    sources_data: list[dict[str, Any]] | None = None,
    query: str = "",
    level: str = "",
    goal: str = "",
    use_llm: bool = True,
    **_: Any,
) -> ToolResult:
    try:
        if sources_data is None:
            sources_data = []

        if not isinstance(sources_data, list):
            sources_data = []

        sources = _reconstruct_sources(sources_data)

        if not sources:
            return ToolResult(
                tool_name="rank_sources",
                success=True,
                data={
                    "total_ranked": 0,
                    "rankings": [],
                    "insufficient_sources": True,
                    "reason": "No valid sources provided for ranking",
                },
                tokens_used=0,
            )

        resolved_query = clean_text(query or goal) or "research sources"

        query_schema = SearchQuerySchema(
            topic=resolved_query,
            goal=clean_text(goal),
            level=clean_text(level),
            max_results=max(1, min(100, len(sources))),
        )

        ranker = ConsensusRanker(use_llm=use_llm)
        ranked = await ranker.rank(sources, query_schema)

        ranked_data = []
        for item in ranked:
            ranked_data.append(
                {
                    "rank": item.rank,
                    "score": item.score,
                    "confidence": item.confidence,
                    "reason": item.reason,
                    "source_id": item.source.source_id,
                    "title": item.source.title,
                    "url": item.source.url,
                    "platform": str(getattr(item.source.platform, "value", item.source.platform)),
                    "difficulty": str(getattr(item.source.difficulty, "value", item.source.difficulty)) if item.source.difficulty else None,
                }
            )

        return ToolResult(
            tool_name="rank_sources",
            success=True,
            data={
                "total_ranked": len(ranked_data),
                "rankings": ranked_data,
            },
            tokens_used=0,
        )

    except Exception as exc:
        _logger.warning(f"rank_sources failed: {exc}")
        return ToolResult(
            tool_name="rank_sources",
            success=False,
            data=None,
            error=str(exc),
        )


async def classify_difficulty(
    sources_data: list[dict[str, Any]] | None = None,
    **_: Any,
) -> ToolResult:
    try:
        if sources_data is None:
            sources_data = []

        if not isinstance(sources_data, list):
            sources_data = []

        sources = _reconstruct_sources(sources_data)

        if not sources:
            return ToolResult(
                tool_name="classify_difficulty",
                success=True,
                data={
                    "classified": [],
                    "insufficient_sources": True,
                    "reason": "No valid sources provided for classification",
                },
                tokens_used=0,
            )

        classifier = DifficultyClassifier()
        classified = classifier.classify_sources(sources)

        results = []
        for source in classified:
            results.append(
                {
                    "source_id": source.source_id,
                    "title": source.title,
                    "difficulty": str(getattr(source.difficulty, "value", source.difficulty)) if source.difficulty else "unknown",
                }
            )

        return ToolResult(
            tool_name="classify_difficulty",
            success=True,
            data={"classified": results},
            tokens_used=0,
        )

    except Exception as exc:
        _logger.warning(f"classify_difficulty failed: {exc}")
        return ToolResult(
            tool_name="classify_difficulty",
            success=False,
            data=None,
            error=str(exc),
        )


async def compare_sources(
    sources_data: list[dict[str, Any]] | None = None,
    query: str = "",
    **_: Any,
) -> ToolResult:
    try:
        if not isinstance(sources_data, list):
            sources_data = []

        sources = _reconstruct_sources(sources_data)
        cleaned_query = clean_text(query)

        if not sources:
            return ToolResult(
                tool_name="compare_sources",
                success=True,
                data={
                    "query": cleaned_query,
                    "comparisons": [],
                    "best_source": None,
                    "source_count": 0,
                    "insufficient_sources": True,
                    "reason": "No sources provided to compare",
                },
                tokens_used=0,
            )

        topic = cleaned_query or (sources[0].title if sources else "")
        if not topic:
            topic = "source comparison"
        if len(topic) < 3:
            topic = f"{topic} comparison"

        scorer = HeuristicScorer()
        query_schema = SearchQuerySchema(topic=topic, max_results=len(sources))

        comparisons = []
        for source in sources:
            score = scorer.score_source(source, query=query_schema)
            relevance = scorer.relevance_factor(source, query_schema)
            comparisons.append(
                {
                    "source_id": source.source_id,
                    "title": source.title,
                    "heuristic_score": round(score, 4),
                    "relevance_factor": round(relevance, 4),
                    "platform": str(getattr(source.platform, "value", source.platform)),
                    "year": source.year,
                    "citation_count": source.citation_count,
                }
            )

        comparisons.sort(key=lambda x: x["heuristic_score"], reverse=True)

        data = {
            "query": topic,
            "comparisons": comparisons,
            "best_source": comparisons[0]["title"] if comparisons else None,
            "source_count": len(sources),
        }

        if len(sources) < 2:
            data["insufficient_sources"] = True
            data["reason"] = "Need at least 2 sources to compare"

        return ToolResult(
            tool_name="compare_sources",
            success=True,
            data=data,
            tokens_used=0,
        )

    except Exception as exc:
        _logger.warning(f"compare_sources failed: {exc}")
        return ToolResult(
            tool_name="compare_sources",
            success=False,
            data=None,
            error=str(exc),
        )


async def check_relevance(
    title: str = "",
    abstract: str = "",
    query: str = "",
    **_: Any,
) -> ToolResult:
    try:
        cleaned_title = clean_text(title)
        cleaned_abstract = clean_text(abstract)
        cleaned_query = clean_text(query)

        if not cleaned_query:
            return ToolResult(
                tool_name="check_relevance",
                success=True,
                data={
                    "is_relevant": False,
                    "relevance_score": 0.0,
                    "title": cleaned_title,
                    "query": "",
                    "reason": "No query provided",
                },
                tokens_used=0,
            )

        if len(cleaned_query) < 3:
            cleaned_query = f"{cleaned_query} relevance"

        source = Source(
            source_id="relevance_check",
            title=cleaned_title,
            url="",
            platform=SourcePlatform.WEB,
            source_type=SourceType.OTHER,
            abstract=cleaned_abstract if cleaned_abstract else None,
        )

        scorer = HeuristicScorer()
        query_schema = SearchQuerySchema(topic=cleaned_query, max_results=1)
        relevance = scorer.relevance_factor(source, query_schema)
        is_relevant = relevance >= 0.3

        return ToolResult(
            tool_name="check_relevance",
            success=True,
            data={
                "is_relevant": is_relevant,
                "relevance_score": round(relevance, 4),
                "title": cleaned_title,
                "query": cleaned_query,
            },
            tokens_used=0,
        )

    except Exception as exc:
        _logger.warning(f"check_relevance failed: {exc}")
        return ToolResult(
            tool_name="check_relevance",
            success=False,
            data=None,
            error=str(exc),
        )


def register_analysis_tools(registry: ToolRegistry) -> None:
    registry.register(
        ToolDefinition(
            name="rank_sources",
            description="Rank sources using heuristic and AI consensus ranking",
            category="analysis",
            token_cost_est=_RANK_TOKEN_COST,
            timeout_seconds=_ANALYSIS_TIMEOUT,
            requires_llm=True,
            parameters_schema={
                "sources_data": "array",
                "query": "string",
                "level": "string",
                "goal": "string",
            },
        ),
        rank_sources,
    )

    registry.register(
        ToolDefinition(
            name="classify_difficulty",
            description="Classify the difficulty level of sources (beginner/intermediate/advanced)",
            category="analysis",
            token_cost_est=_CLASSIFY_TOKEN_COST,
            timeout_seconds=30.0,
            requires_llm=False,
            parameters_schema={
                "sources_data": "array",
            },
        ),
        classify_difficulty,
    )

    registry.register(
        ToolDefinition(
            name="compare_sources",
            description="Compare multiple sources and determine which is most relevant",
            category="analysis",
            token_cost_est=2000,
            timeout_seconds=30.0,
            requires_llm=False,
            parameters_schema={
                "sources_data": "array",
                "query": "string",
            },
        ),
        compare_sources,
    )

    registry.register(
        ToolDefinition(
            name="check_relevance",
            description="Check if a single source is relevant to the query",
            category="analysis",
            token_cost_est=500,
            timeout_seconds=15.0,
            requires_llm=False,
            parameters_schema={
                "title": "string",
                "abstract": "string",
                "query": "string",
            },
        ),
        check_relevance,
    )