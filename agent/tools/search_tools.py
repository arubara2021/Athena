from __future__ import annotations

from typing import Any

from agent.tools.registry import ToolRegistry, ToolDefinition
from agent.tools.executor import ToolResult
from core.models import SourcePlatform
from core.schemas import SearchQuerySchema
from search.orchestrator import SearchOrchestrator
from utils.logger import get_logger


_logger = get_logger("agent.tools.search")

_SEARCH_TOOL_TIMEOUT = 90.0
_SEARCH_TOKEN_COST = 500


async def search_academic(
    query: str,
    max_results: int = 10,
    level: str = "",
    goal: str = "",
) -> ToolResult:
    try:
        platforms = [
            SourcePlatform.ARXIV,
            SourcePlatform.OPENALEX,
            SourcePlatform.SEMANTIC_SCHOLAR,
        ]
        query_schema = SearchQuerySchema(
            topic=query,
            goal=goal,
            level=level,
            max_results=max_results,
            platforms=platforms,
        )
        async with SearchOrchestrator(use_llm_expansion=False) as orchestrator:
            result = await orchestrator.run(query_schema)

        sources_data = []
        for source in result.sources:
            sources_data.append({
                "source_id": source.source_id,
                "title": source.title,
                "url": source.url,
                "platform": str(getattr(source.platform, "value", source.platform)),
                "source_type": str(getattr(source.source_type, "value", source.source_type)),
                "abstract": (source.abstract or "")[:500],
                "year": source.year,
                "citation_count": source.citation_count,
                "difficulty": str(getattr(source.difficulty, "value", source.difficulty)) if source.difficulty else None,
            })

        return ToolResult(
            tool_name="search_academic",
            success=True,
            data={
                "total_found": result.total_found,
                "total_valid": result.total_valid,
                "sources": sources_data,
                "corrected_topic": result.corrected_topic,
                "keywords": result.keywords,
            },
            tokens_used=0,
        )
    except Exception as exc:
        _logger.warning(f"search_academic failed: {exc}")
        return ToolResult(
            tool_name="search_academic",
            success=False,
            data=None,
            error=str(exc),
        )


async def search_web(
    query: str,
    max_results: int = 8,
    level: str = "",
    goal: str = "",
) -> ToolResult:
    try:
        platforms = [
            SourcePlatform.WEB,
            SourcePlatform.TAVILY,
            SourcePlatform.EXA,
            SourcePlatform.SERPER,
        ]
        query_schema = SearchQuerySchema(
            topic=query,
            goal=goal,
            level=level,
            max_results=max_results,
            platforms=platforms,
        )
        async with SearchOrchestrator(use_llm_expansion=False) as orchestrator:
            result = await orchestrator.run(query_schema)

        sources_data = []
        for source in result.sources:
            sources_data.append({
                "source_id": source.source_id,
                "title": source.title,
                "url": source.url,
                "platform": str(getattr(source.platform, "value", source.platform)),
                "abstract": (source.abstract or "")[:500],
            })

        return ToolResult(
            tool_name="search_web",
            success=True,
            data={
                "total_found": result.total_found,
                "sources": sources_data,
            },
            tokens_used=0,
        )
    except Exception as exc:
        _logger.warning(f"search_web failed: {exc}")
        return ToolResult(
            tool_name="search_web",
            success=False,
            data=None,
            error=str(exc),
        )


async def search_github(
    query: str,
    max_results: int = 8,
) -> ToolResult:
    try:
        platforms = [SourcePlatform.GITHUB]
        query_schema = SearchQuerySchema(
            topic=query,
            max_results=max_results,
            platforms=platforms,
        )
        async with SearchOrchestrator(use_llm_expansion=False) as orchestrator:
            result = await orchestrator.run(query_schema)

        sources_data = []
        for source in result.sources:
            metadata = source.metadata if isinstance(source.metadata, dict) else {}
            sources_data.append({
                "source_id": source.source_id,
                "title": source.title,
                "url": source.url,
                "abstract": (source.abstract or "")[:300],
                "stars": metadata.get("stars"),
                "language": metadata.get("language"),
            })

        return ToolResult(
            tool_name="search_github",
            success=True,
            data={"sources": sources_data},
            tokens_used=0,
        )
    except Exception as exc:
        _logger.warning(f"search_github failed: {exc}")
        return ToolResult(
            tool_name="search_github",
            success=False,
            data=None,
            error=str(exc),
        )


async def search_wikipedia(
    query: str,
    max_results: int = 5,
) -> ToolResult:
    try:
        platforms = [SourcePlatform.WIKIPEDIA]
        query_schema = SearchQuerySchema(
            topic=query,
            max_results=max_results,
            platforms=platforms,
        )
        async with SearchOrchestrator(use_llm_expansion=False) as orchestrator:
            result = await orchestrator.run(query_schema)

        sources_data = []
        for source in result.sources:
            sources_data.append({
                "source_id": source.source_id,
                "title": source.title,
                "url": source.url,
                "abstract": (source.abstract or "")[:500],
            })

        return ToolResult(
            tool_name="search_wikipedia",
            success=True,
            data={"sources": sources_data},
            tokens_used=0,
        )
    except Exception as exc:
        _logger.warning(f"search_wikipedia failed: {exc}")
        return ToolResult(
            tool_name="search_wikipedia",
            success=False,
            data=None,
            error=str(exc),
        )


def register_search_tools(registry: ToolRegistry) -> None:
    registry.register(
        ToolDefinition(
            name="search_academic",
            description="Search academic platforms (arXiv, OpenAlex, Semantic Scholar) for research papers",
            category="search",
            token_cost_est=_SEARCH_TOKEN_COST,
            timeout_seconds=_SEARCH_TOOL_TIMEOUT,
            requires_llm=False,
        ),
        search_academic,
    )

    registry.register(
        ToolDefinition(
            name="search_web",
            description="Search the web using Tavily, Exa, Serper for general content",
            category="search",
            token_cost_est=_SEARCH_TOKEN_COST,
            timeout_seconds=_SEARCH_TOOL_TIMEOUT,
            requires_llm=False,
        ),
        search_web,
    )

    registry.register(
        ToolDefinition(
            name="search_github",
            description="Search GitHub for code repositories related to the topic",
            category="search",
            token_cost_est=_SEARCH_TOKEN_COST,
            timeout_seconds=_SEARCH_TOOL_TIMEOUT,
            requires_llm=False,
        ),
        search_github,
    )

    registry.register(
        ToolDefinition(
            name="search_wikipedia",
            description="Search Wikipedia for documentation and introductory content",
            category="search",
            token_cost_est=_SEARCH_TOKEN_COST,
            timeout_seconds=_SEARCH_TOOL_TIMEOUT,
            requires_llm=False,
        ),
        search_wikipedia,
    )