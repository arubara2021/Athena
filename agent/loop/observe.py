from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from core.models import (
    CoreModel,
    Difficulty,
    RankedSource,
    Source,
    SourcePlatform,
    SourceType,
)
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
    summary_updated: bool = False
    comparison_completed: bool = False
    soft_skip: bool = False
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
        if getattr(act_result, "skipped", False):
            return ObserveResult(
                success=True,
                data_processed=False,
                soft_skip=True,
                is_sufficient=True,
                notes=[
                    f"Skipped: {clean_text(getattr(act_result, 'skip_reason', ''))}"
                ],
            )

        if not act_result.success:
            return ObserveResult(
                success=False,
                data_processed=False,
                is_sufficient=False,
                notes=[f"Action failed: {clean_text(act_result.error or '')}"],
            )

        data = act_result.data
        tool_name = clean_text(act_result.tool_name)

        if tool_name.startswith("search"):
            return self._observe_search_results(state, data)

        if tool_name == "rank_sources":
            return self._observe_ranking_results(state, data)

        if tool_name == "generate_learning_path":
            return self._observe_learning_path(state, data)

        if tool_name == "extract_source_summary":
            return self._observe_extract_summary(state, data)

        if tool_name == "compare_sources":
            return self._observe_compare_sources(state, data)

        if tool_name == "classify_difficulty":
            return self._observe_classify_difficulty(state, data)

        if tool_name == "check_relevance":
            return self._observe_check_relevance(state, data)

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

        ranked_sources: list[RankedSource] = []
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
                ranked_updated=True,
                is_sufficient=len(ranked_sources) >= 3,
                content_summary=f"Ranked {len(ranked_sources)} sources",
                notes=[
                    f"Top source: {ranked_sources[0].source.title[:80]}"
                    if ranked_sources
                    else "No rankings"
                ],
            )

        return ObserveResult(
            success=False,
            notes=["No valid ranked sources could be reconstructed"],
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

    def _observe_extract_summary(self, state: Any, data: Any) -> ObserveResult:
        if not isinstance(data, dict):
            return ObserveResult(
                success=False,
                notes=["Extract summary result is not a valid dictionary"],
            )

        if bool(data.get("skipped")):
            reason = (
                clean_text(data.get("reason", ""))
                or "No source available for summary extraction"
            )
            return ObserveResult(
                success=True,
                data_processed=False,
                soft_skip=True,
                is_sufficient=True,
                content_summary=reason,
                notes=[reason],
            )

        summary = clean_text(data.get("summary", ""))
        title = clean_text(data.get("title", ""))
        source_id = clean_text(data.get("source_id", ""))
        updated = self._update_source_summary(state, source_id, title, summary)

        return ObserveResult(
            success=True,
            data_processed=bool(summary),
            summary_updated=updated,
            is_sufficient=bool(summary),
            content_summary=summary[:200] if summary else "No summary",
            notes=[f"Summary length: {len(summary)}"],
        )

    def _observe_compare_sources(self, state: Any, data: Any) -> ObserveResult:
        if not isinstance(data, dict):
            return ObserveResult(
                success=False,
                notes=["Compare sources result is not a valid dictionary"],
            )

        comparisons = data.get("comparisons", [])
        if not isinstance(comparisons, list):
            comparisons = []

        source_count = data.get("source_count", len(comparisons))
        try:
            source_count = int(source_count)
        except Exception:
            source_count = len(comparisons)

        if bool(data.get("insufficient_sources")) or source_count < 2:
            reason = (
                clean_text(data.get("reason", ""))
                or "Insufficient sources for comparison"
            )
            return ObserveResult(
                success=True,
                data_processed=False,
                soft_skip=True,
                is_sufficient=True,
                content_summary=reason,
                notes=[reason],
            )

        best_source = clean_text(data.get("best_source", ""))
        return ObserveResult(
            success=True,
            data_processed=bool(comparisons),
            comparison_completed=bool(comparisons),
            is_sufficient=len(comparisons) >= 2,
            content_summary=(
                f"Compared {len(comparisons)} sources. Best: {best_source}"[:200]
            ),
            notes=[f"Comparison count: {len(comparisons)}"],
        )

    def _observe_classify_difficulty(self, state: Any, data: Any) -> ObserveResult:
        if not isinstance(data, dict):
            return ObserveResult(
                success=False,
                notes=["Classify difficulty result is not a valid dictionary"],
            )

        classified = data.get("classified", [])
        if not isinstance(classified, list):
            classified = []

        return ObserveResult(
            success=True,
            data_processed=bool(classified),
            is_sufficient=bool(classified),
            content_summary=f"Classified {len(classified)} sources",
            notes=[f"Classified count: {len(classified)}"],
        )

    def _observe_check_relevance(self, state: Any, data: Any) -> ObserveResult:
        if not isinstance(data, dict):
            return ObserveResult(
                success=False,
                notes=["Check relevance result is not a valid dictionary"],
            )

        is_relevant = bool(data.get("is_relevant"))
        score = data.get("relevance_score", 0.0)

        return ObserveResult(
            success=True,
            data_processed=True,
            is_sufficient=True,
            content_summary=(
                f"Relevance score {score}: "
                f"{'relevant' if is_relevant else 'not relevant'}"
            ),
            notes=[f"is_relevant={is_relevant}"],
        )

    def _observe_read_result(self, state: Any, data: Any) -> ObserveResult:
        if not isinstance(data, dict):
            return ObserveResult(
                success=False,
                notes=["Read result is not a valid dictionary"],
            )

        content = clean_text(
            data.get("content_preview", "") or data.get("full_content", "")
        )
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
            summary_updated=bool(summary),
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

    def _update_source_summary(
        self,
        state: Any,
        source_id: str,
        title: str,
        summary: str,
    ) -> bool:
        if not summary:
            return False

        matched = False

        for source in getattr(state, "findings", []) or []:
            if (
                source_id
                and getattr(source, "source_id", "") == source_id
            ) or (
                title
                and getattr(source, "title", "") == title
            ):
                try:
                    source.summary = summary
                    matched = True
                except Exception:
                    pass

        for item in getattr(state, "ranked_sources", []) or []:
            source = getattr(item, "source", None)
            if source is None:
                continue

            if (
                source_id
                and getattr(source, "source_id", "") == source_id
            ) or (
                title
                and getattr(source, "title", "") == title
            ):
                try:
                    source.summary = summary
                    matched = True
                except Exception:
                    pass

        return matched

    def _dict_to_source(self, item: dict[str, Any]) -> Source | None:
        try:
            title = clean_text(item.get("title", ""))
            url = clean_text(item.get("url", ""))

            if not title or not url:
                return None

            source_id = str(item.get("source_id", "") or "").strip()
            if not source_id:
                source_id = f"observed_{title[:50]}"

            abstract = clean_text(item.get("abstract", "")) or None
            summary = clean_text(item.get("summary", "")) or None

            return Source(
                source_id=source_id,
                title=title,
                url=url,
                platform=self._parse_platform(item.get("platform")),
                source_type=self._parse_source_type(item.get("source_type")),
                abstract=abstract,
                summary=summary,
                authors=self._parse_authors(item.get("authors")),
                published_at=self._parse_published_at(item.get("published_at")),
                year=self._parse_int(item.get("year")),
                citation_count=self._parse_int(item.get("citation_count")),
                has_code=self._parse_bool(item.get("has_code")),
                difficulty=self._parse_difficulty(item.get("difficulty")),
                score=self._parse_float(item.get("score")),
                metadata=self._parse_metadata(item.get("metadata")),
            )
        except Exception as exc:
            self._logger.warning(f"Failed to convert dict to Source: {exc}")
            return None

    def _parse_platform(self, value: Any) -> SourcePlatform:
        if isinstance(value, SourcePlatform):
            return value

        text = str(value or "").strip().lower()
        if not text:
            return SourcePlatform.WEB

        try:
            return SourcePlatform(text)
        except ValueError:
            return SourcePlatform.WEB

    def _parse_source_type(self, value: Any) -> SourceType:
        if isinstance(value, SourceType):
            return value

        text = str(value or "").strip().lower()
        if not text:
            return SourceType.OTHER

        try:
            return SourceType(text)
        except ValueError:
            return SourceType.OTHER

    def _parse_difficulty(self, value: Any) -> Difficulty | None:
        if value is None:
            return None

        if isinstance(value, Difficulty):
            return value

        text = str(value).strip().lower()
        if not text:
            return None

        try:
            return Difficulty(text)
        except (ValueError, TypeError):
            return Difficulty.UNKNOWN

    def _parse_published_at(self, value: Any) -> datetime | None:
        if value is None:
            return None

        if isinstance(value, datetime):
            return value

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None

            try:
                return datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                return None

        return None

    def _parse_authors(self, value: Any) -> list[str]:
        if not isinstance(value, (list, tuple, set)):
            return []

        result: list[str] = []
        seen: set[str] = set()

        for item in value:
            text = clean_text(item)
            if not text:
                continue

            key = text.lower()
            if key in seen:
                continue

            seen.add(key)
            result.append(text)

        return result

    def _parse_metadata(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)

        return {}

    def _parse_int(self, value: Any) -> int | None:
        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _parse_float(self, value: Any) -> float | None:
        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _parse_bool(self, value: Any) -> bool | None:
        if value is None:
            return None

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

        return None