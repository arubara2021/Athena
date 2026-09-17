from __future__ import annotations

from typing import Any

from pydantic import Field

from core.models import CoreModel, RankedSource, Source, SourcePlatform, SourceType
from utils.logger import get_logger
from utils.text import clean_text


DEFAULT_MAX_OBSERVE_CONTENT_LENGTH = 3000
DEFAULT_MIN_FINDINGS_FOR_SUFFICIENCY = 3


class ObserveResult(CoreModel):
    success: bool = False
    data_processed: bool = False
    findings_added: int = Field(default=0, ge=0)
    ranked_updated: bool = False
    learning_path_generated: bool = False
    is_sufficient: bool = False
    content_summary: str = ""
    notes: list[str] = Field(default_factory=list)


class ObservePhase:
    def __init__(
        self,
        max_content_length: int | None = None,
        min_findings_for_sufficiency: int | None = None,
    ) -> None:
        self._max_content_length = (
            max_content_length or DEFAULT_MAX_OBSERVE_CONTENT_LENGTH
        )
        self._min_findings = (
            min_findings_for_sufficiency or DEFAULT_MIN_FINDINGS_FOR_SUFFICIENCY
        )
        self._logger = get_logger("agent.loop.observe")

    def observe(self, state: Any, act_result: Any) -> ObserveResult:
        if act_result.skipped:
            return ObserveResult(
                success=True,
                data_processed=False,
                is_sufficient=False,
                notes=[f"Skipped: {act_result.skip_reason}"],
            )

        if not act_result.success:
            return ObserveResult(
                success=False,
                data_processed=False,
                is_sufficient=False,
                notes=[f"Action failed: {act_result.error}"],
            )

        data = act_result.data
        tool_name = act_result.tool_name

        if tool_name.startswith("search"):
            return self._observe_search_results(state, data)

        if tool_name == "rank_sources":
            return self._observe_ranking_results(state, data)

        if tool_name == "generate_learning_path":
            return self._observe_learning_path(state, data)

        if tool_name.startswith("read"):
            return self._observe_read_result(state, data)

        if tool_name.startswith("summarize"):
            return self._observe_summary(state, data)

        return self._observe_generic(state, data)

    def _observe_search_results(self, state: Any, data: Any) -> ObserveResult:
        if not isinstance(data, dict):
            return ObserveResult(
                success=False,
                notes=["Search result is not a valid dictionary"],
            )

        sources_data = data.get("sources", [])
        if not isinstance(sources_data, list):
            return ObserveResult(
                success=False,
                notes=["No sources list in search result"],
            )

        findings_added = 0
        for item in sources_data:
            if not isinstance(item, dict):
                continue
            source = self._dict_to_source(item)
            if source is not None:
                state.add_finding(source)
                findings_added += 1

        is_sufficient = len(state.findings) >= self._min_findings

        return ObserveResult(
            success=True,
            data_processed=True,
            findings_added=findings_added,
            is_sufficient=is_sufficient,
            content_summary=f"Added {findings_added} sources, total: {len(state.findings)}",
            notes=[
                f"Total findings: {len(state.findings)}",
                f"Sufficiency threshold: {self._min_findings}",
            ],
        )

    def _observe_ranking_results(self, state: Any, data: Any) -> ObserveResult:
        if not isinstance(data, dict):
            return ObserveResult(
                success=False,
                notes=["Ranking result is not a valid dictionary"],
            )

        rankings = data.get("rankings", [])
        if not isinstance(rankings, list) or not rankings:
            return ObserveResult(
                success=False,
                notes=["No rankings in ranking result"],
            )

        ranked_sources = []
        source_map = {s.source_id: s for s in state.findings}

        for item in rankings:
            if not isinstance(item, dict):
                continue
            source_id = str(item.get("source_id", ""))
            source = source_map.get(source_id)
            if source is None:
                continue

            try:
                ranked = RankedSource(
                    source=source,
                    rank=int(item.get("rank", len(ranked_sources) + 1)),
                    score=float(item.get("score", 0.5)),
                    confidence=float(item.get("confidence", 0.5)),
                    reason=clean_text(item.get("reason", "")) or None,
                )
                ranked_sources.append(ranked)
            except Exception:
                continue

        if ranked_sources:
            state.ranked_sources = ranked_sources

        return ObserveResult(
            success=True,
            data_processed=True,
            ranked_updated=bool(ranked_sources),
            is_sufficient=len(ranked_sources) >= 3,
            content_summary=f"Ranked {len(ranked_sources)} sources",
            notes=[f"Top source: {ranked_sources[0].source.title[:80]}" if ranked_sources else "No rankings"],
        )

    def _observe_learning_path(self, state: Any, data: Any) -> ObserveResult:
        if not isinstance(data, dict):
            return ObserveResult(
                success=False,
                notes=["Learning path result is not a valid dictionary"],
            )

        steps = data.get("steps", [])
        if not isinstance(steps, list) or not steps:
            return ObserveResult(
                success=False,
                notes=["No steps in learning path"],
            )

        return ObserveResult(
            success=True,
            data_processed=True,
            learning_path_generated=True,
            is_sufficient=True,
            content_summary=f"Learning path generated with {len(steps)} steps",
            notes=[f"Steps: {len(steps)}"],
        )

    def _observe_read_result(self, state: Any, data: Any) -> ObserveResult:
        if not isinstance(data, dict):
            return ObserveResult(
                success=False,
                notes=["Read result is not a valid dictionary"],
            )

        content = clean_text(data.get("content_preview", "") or data.get("full_content", ""))
        content = content[: self._max_content_length]

        return ObserveResult(
            success=True,
            data_processed=True,
            is_sufficient=bool(content),
            content_summary=content[:200] if content else "Empty content",
            notes=[f"Content length: {len(content)}"],
        )

    def _observe_summary(self, state: Any, data: Any) -> ObserveResult:
        if not isinstance(data, dict):
            return ObserveResult(
                success=False,
                notes=["Summary result is not a valid dictionary"],
            )

        summary = clean_text(data.get("summary", ""))
        return ObserveResult(
            success=True,
            data_processed=True,
            is_sufficient=bool(summary),
            content_summary=summary[:200] if summary else "Empty summary",
        )

    def _observe_generic(self, state: Any, data: Any) -> ObserveResult:
        content = str(data)[: self._max_content_length] if data else ""
        return ObserveResult(
            success=True,
            data_processed=bool(data),
            is_sufficient=bool(data),
            content_summary=content[:200] if content else "No data",
        )

    def _dict_to_source(self, item: dict[str, Any]) -> Source | None:
        try:
            title = clean_text(item.get("title", ""))
            url = clean_text(item.get("url", ""))
            if not title or not url:
                return None

            platform_str = str(item.get("platform", "web")).lower()
            try:
                platform = SourcePlatform(platform_str)
            except ValueError:
                platform = SourcePlatform.WEB

            type_str = str(item.get("source_type", "other")).lower()
            try:
                source_type = SourceType(type_str)
            except ValueError:
                source_type = SourceType.OTHER

            abstract = clean_text(item.get("abstract", "")) or None

            return Source(
                source_id=str(item.get("source_id", "")) or f"observed_{title[:50]}",
                title=title,
                url=url,
                platform=platform,
                source_type=source_type,
                abstract=abstract,
                year=item.get("year"),
                citation_count=item.get("citation_count"),
                metadata=item.get("metadata", {}),
            )
        except Exception as exc:
            self._logger.warning(f"Failed to convert dict to Source: {exc}")
            return None