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
    sources_data: list[dict[str, Any]],
    query: str,
    level: str = "",
    goal: str = "",
    use_llm: bool = True,
) -> ToolResult:
    try:
        sources = _reconstruct_sources(sources_data)
        if not sources:
            return ToolResult(
                tool_name="rank_sources",
                success=False,
                data=None,
                error="No valid sources provided for ranking",
            )

        query_schema = SearchQuerySchema(
            topic=query,
            goal=goal,
            level=level,
            max_results=len(sources),
        )

        ranker = ConsensusRanker(use_llm=use_llm)
        ranked = await ranker.rank(sources, query_schema)

        ranked_data = []
        for item in ranked:
            ranked_data.append({
                "rank": item.rank,
                "score": item.score,
                "confidence": item.confidence,
                "reason": item.reason,
                "source_id": item.source.source_id,
                "title": item.source.title,
                "url": item.source.url,
                "platform": str(getattr(item.source.platform, "value", item.source.platform)),
                "difficulty": str(getattr(item.source.difficulty, "value", item.source.difficulty)) if item.source.difficulty else None,
            })

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
    sources_data: list[dict[str, Any]],
) -> ToolResult:
    try:
        sources = _reconstruct_sources(sources_data)
        if not sources:
            return ToolResult(
                tool_name="classify_difficulty",
                success=False,
                data=None,
                error="No valid sources provided for classification",
            )

        classifier = DifficultyClassifier()
        classified = classifier.classify_sources(sources)

        results = []
        for source in classified:
            results.append({
                "source_id": source.source_id,
                "title": source.title,
                "difficulty": str(getattr(source.difficulty, "value", source.difficulty)) if source.difficulty else "unknown",
            })

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
    sources_data: list[dict[str, Any]],
    query: str,
) -> ToolResult:
    try:
        sources = _reconstruct_sources(sources_data)
        if len(sources) < 2:
            return ToolResult(
                tool_name="compare_sources",
                success=False,
                data=None,
                error="Need at least 2 sources to compare",
            )

        scorer = HeuristicScorer()
        query_schema = SearchQuerySchema(topic=query, max_results=len(sources))

        comparisons = []
        for source in sources:
            score = scorer.score_source(source, query=query_schema)
            relevance = scorer.relevance_factor(source, query_schema)
            comparisons.append({
                "source_id": source.source_id,
                "title": source.title,
                "heuristic_score": round(score, 4),
                "relevance_factor": round(relevance, 4),
                "platform": str(getattr(source.platform, "value", source.platform)),
                "year": source.year,
                "citation_count": source.citation_count,
            })

        comparisons.sort(key=lambda x: x["heuristic_score"], reverse=True)

        return ToolResult(
            tool_name="compare_sources",
            success=True,
            data={
                "query": query,
                "comparisons": comparisons,
                "best_source": comparisons[0]["title"] if comparisons else None,
            },
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
    title: str,
    abstract: str,
    query: str,
) -> ToolResult:
    try:
        source = Source(
            source_id="relevance_check",
            title=clean_text(title),
            url="",
            platform=SourcePlatform.WEB,
            source_type=SourceType.OTHER,
            abstract=clean_text(abstract) if abstract else None,
        )

        scorer = HeuristicScorer()
        query_schema = SearchQuerySchema(topic=query, max_results=1)
        relevance = scorer.relevance_factor(source, query_schema)

        is_relevant = relevance >= 0.3

        return ToolResult(
            tool_name="check_relevance",
            success=True,
            data={
                "is_relevant": is_relevant,
                "relevance_score": round(relevance, 4),
                "title": clean_text(title),
                "query": query,
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
        ),
        check_relevance,
    )