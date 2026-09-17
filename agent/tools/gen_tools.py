from __future__ import annotations

from typing import Any

from agent.tools.registry import ToolRegistry, ToolDefinition
from agent.tools.executor import ToolResult
from core.models import Source, RankedSource, SourcePlatform, SourceType, Difficulty
from core.schemas import SearchQuerySchema
from ranking.learning_path_builder import LearningPathBuilder
from ranking.path_enhancer import PathEnhancer
from ranking.source_summarizer import SourceSummarizer
from utils.logger import get_logger
from utils.text import clean_text


_logger = get_logger("agent.tools.gen")

_GEN_TIMEOUT = 120.0
_PATH_TOKEN_COST = 6000
_SUMMARY_TOKEN_COST = 1000


def _reconstruct_ranked_sources(sources_data: list[dict[str, Any]]) -> list[RankedSource]:
    ranked = []
    for index, item in enumerate(sources_data):
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
                difficulty=difficulty,
                metadata=item.get("metadata", {}),
            )

            ranked_source = RankedSource(
                source=source,
                rank=item.get("rank", index + 1),
                score=float(item.get("score", 0.5)),
                confidence=float(item.get("confidence", 0.5)),
                reason=item.get("reason"),
            )
            ranked.append(ranked_source)
        except Exception as exc:
            _logger.warning(f"Failed to reconstruct ranked source: {exc}")
            continue
    return ranked


async def generate_learning_path(
    sources_data: list[dict[str, Any]],
    query: str,
    level: str = "",
    goal: str = "",
    use_llm: bool = True,
) -> ToolResult:
    try:
        ranked_sources = _reconstruct_ranked_sources(sources_data)
        if not ranked_sources:
            return ToolResult(
                tool_name="generate_learning_path",
                success=False,
                data=None,
                error="No valid ranked sources provided",
            )

        query_schema = SearchQuerySchema(
            topic=query,
            goal=goal,
            level=level,
            max_results=len(ranked_sources),
        )

        builder = LearningPathBuilder(use_llm=use_llm)
        learning_path = await builder.build(ranked_sources, query_schema)

        steps_data = []
        for step in learning_path.steps:
            resources = []
            for res in step.resources:
                resources.append({
                    "title": res.source.title,
                    "url": res.source.url,
                    "rank": res.rank,
                })
            steps_data.append({
                "step": step.step,
                "title": step.title,
                "objective": step.objective,
                "estimated_minutes": step.estimated_minutes,
                "resources": resources,
            })

        return ToolResult(
            tool_name="generate_learning_path",
            success=True,
            data={
                "topic": learning_path.topic,
                "level": learning_path.level,
                "goal": learning_path.goal,
                "total_steps": len(steps_data),
                "steps": steps_data,
            },
            tokens_used=0,
        )
    except Exception as exc:
        _logger.warning(f"generate_learning_path failed: {exc}")
        return ToolResult(
            tool_name="generate_learning_path",
            success=False,
            data=None,
            error=str(exc),
        )


async def enhance_path(
    path_data: dict[str, Any],
    query: str,
    level: str = "",
    use_llm: bool = True,
) -> ToolResult:
    try:
        from core.models import LearningPath, LearningStep

        steps = []
        for step_item in path_data.get("steps", []):
            resources = []
            for res_item in step_item.get("resources", []):
                source = Source(
                    source_id=res_item.get("source_id", ""),
                    title=res_item.get("title", ""),
                    url=res_item.get("url", ""),
                    platform=SourcePlatform.WEB,
                    source_type=SourceType.OTHER,
                )
                resources.append(RankedSource(source=source, rank=1, score=0.5))

            steps.append(LearningStep(
                step=step_item.get("step", 1),
                title=step_item.get("title", ""),
                objective=step_item.get("objective", ""),
                estimated_minutes=step_item.get("estimated_minutes"),
                resources=resources,
            ))

        learning_path = LearningPath(
            topic=path_data.get("topic", query),
            level=path_data.get("level", level),
            goal=path_data.get("goal", ""),
            steps=steps,
        )

        query_schema = SearchQuerySchema(topic=query, level=level)
        enhancer = PathEnhancer(use_llm=use_llm)
        enhanced = await enhancer.enhance(learning_path, query_schema)

        enhancements_data = []
        for enh in enhanced.enhancements:
            enhancements_data.append({
                "step": enh.step,
                "prerequisites": enh.prerequisites,
                "suggested_projects": enh.suggested_projects,
                "key_concepts": enh.key_concepts,
            })

        return ToolResult(
            tool_name="enhance_path",
            success=True,
            data={
                "total_enhancements": len(enhancements_data),
                "enhancements": enhancements_data,
            },
            tokens_used=0,
        )
    except Exception as exc:
        _logger.warning(f"enhance_path failed: {exc}")
        return ToolResult(
            tool_name="enhance_path",
            success=False,
            data=None,
            error=str(exc),
        )


async def summarize_source(
    title: str,
    abstract: str,
    url: str = "",
    use_llm: bool = True,
) -> ToolResult:
    try:
        source = Source(
            source_id="summarize_target",
            title=clean_text(title),
            url=clean_text(url),
            platform=SourcePlatform.WEB,
            source_type=SourceType.OTHER,
            abstract=clean_text(abstract) if abstract else None,
        )

        summarizer = SourceSummarizer(use_llm=use_llm)
        summaries = await summarizer.summarize_sources([source])

        if not summaries:
            return ToolResult(
                tool_name="summarize_source",
                success=False,
                data=None,
                error="Failed to generate summary",
            )

        summary = summaries[0]
        summary_data = {
            "title": getattr(summary, "title", title),
            "summary": getattr(summary, "summary", ""),
            "difficulty": str(getattr(summary, "difficulty", "")),
            "key_topics": getattr(summary, "key_topics", []),
            "why_useful": getattr(summary, "why_useful", ""),
        }

        return ToolResult(
            tool_name="summarize_source",
            success=True,
            data=summary_data,
            tokens_used=0,
        )
    except Exception as exc:
        _logger.warning(f"summarize_source failed: {exc}")
        return ToolResult(
            tool_name="summarize_source",
            success=False,
            data=None,
            error=str(exc),
        )


async def generate_report(
    query: str,
    sources_data: list[dict[str, Any]],
    learning_path_data: dict[str, Any] | None = None,
) -> ToolResult:
    try:
        report = {
            "query": query,
            "total_sources": len(sources_data),
            "sources": sources_data[:10],
            "learning_path": learning_path_data,
            "generated_by": "autonomous_agent",
        }

        return ToolResult(
            tool_name="generate_report",
            success=True,
            data=report,
            tokens_used=0,
        )
    except Exception as exc:
        _logger.warning(f"generate_report failed: {exc}")
        return ToolResult(
            tool_name="generate_report",
            success=False,
            data=None,
            error=str(exc),
        )


def register_gen_tools(registry: ToolRegistry) -> None:
    registry.register(
        ToolDefinition(
            name="generate_learning_path",
            description="Generate a structured learning path from ranked sources",
            category="generation",
            token_cost_est=_PATH_TOKEN_COST,
            timeout_seconds=_GEN_TIMEOUT,
            requires_llm=True,
        ),
        generate_learning_path,
    )

    registry.register(
        ToolDefinition(
            name="enhance_path",
            description="Enhance a learning path with prerequisites, projects, and key concepts",
            category="generation",
            token_cost_est=4000,
            timeout_seconds=_GEN_TIMEOUT,
            requires_llm=True,
        ),
        enhance_path,
    )

    registry.register(
        ToolDefinition(
            name="summarize_source",
            description="Generate a concise summary of a single source",
            category="generation",
            token_cost_est=_SUMMARY_TOKEN_COST,
            timeout_seconds=60.0,
            requires_llm=True,
        ),
        summarize_source,
    )

    registry.register(
        ToolDefinition(
            name="generate_report",
            description="Compile all findings into a structured report",
            category="generation",
            token_cost_est=3000,
            timeout_seconds=30.0,
            requires_llm=False,
        ),
        generate_report,
    )