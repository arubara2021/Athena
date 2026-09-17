from __future__ import annotations

from typing import Any

from storage.file_manager import FileManager
from utils.text import clean_text, normalize_newlines, truncate_text


class MarkdownWriter:
    def __init__(self, file_manager: FileManager) -> None:
        self._file_manager = file_manager

    def write(
        self,
        content: str,
        filename: str,
        subdirectory: str = "markdown",
    ) -> Any:
        path = self._file_manager.build_path(filename, subdirectory=subdirectory)
        return self._file_manager.write_text(path, normalize_newlines(content))

    def write_state(
        self,
        state: Any,
        filename: str,
        subdirectory: str = "markdown",
    ) -> Any:
        content = self.render_state(state)
        return self.write(content, filename, subdirectory=subdirectory)

    def render_state(self, state: Any) -> str:
        lines: list[str] = []

        query = getattr(state, "query", None)
        topic = self._query_field(query, "topic", "Research Report")

        lines.append(f"# Research Report: {topic}")
        lines.append("")

        lines.extend(self._render_overview(state, query))
        lines.extend(self._render_errors(state))
        lines.extend(self._render_ranked_sources(state))
        lines.extend(self._render_source_summaries(state))
        lines.extend(self._render_learning_path(state))
        lines.extend(self._render_saved_files(state))

        return "\n".join(lines)

    def render_result(self, result: Any) -> str:
        return self.render_state(result)

    def _render_overview(self, state: Any, query: Any) -> list[str]:
        status = self._enum_value(getattr(state, "status", None)) or "-"
        request_id = str(getattr(state, "request_id", "") or "-")
        goal = self._query_field(query, "goal", "-")
        level = self._query_field(query, "level", "-")

        sources = list(getattr(state, "sources", []) or [])
        ranked_sources = list(getattr(state, "ranked_sources", []) or [])
        learning_path = getattr(state, "learning_path", None)
        steps = list(getattr(learning_path, "steps", []) or []) if learning_path else []
        errors = list(getattr(state, "errors", []) or [])
        summaries = list(getattr(state, "source_summaries", []) or [])
        enhanced_learning_path = getattr(state, "enhanced_learning_path", None)
        rag_documents_indexed = getattr(state, "rag_documents_indexed", 0)

        latency = self._format_float(getattr(state, "latency_ms", None))

        lines: list[str] = []
        lines.append("## Overview")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| Goal | {self._escape_cell(goal)} |")
        lines.append(f"| Level | {self._escape_cell(level)} |")
        lines.append(f"| Request ID | {self._escape_cell(request_id)} |")
        lines.append(f"| Status | {self._escape_cell(status.upper())} |")
        lines.append(f"| Latency | {self._escape_cell(latency)} |")
        lines.append(f"| Sources | {len(sources)} |")
        lines.append(f"| Ranked Sources | {len(ranked_sources)} |")
        lines.append(f"| Source Summaries | {len(summaries)} |")
        lines.append(f"| Learning Steps | {len(steps)} |")
        lines.append(f"| Enhanced Learning Path | {'Yes' if enhanced_learning_path else 'No'} |")
        lines.append(f"| RAG Documents Indexed | {rag_documents_indexed} |")
        lines.append(f"| Errors | {len(errors)} |")
        lines.append("")

        return lines

    def _render_errors(self, state: Any) -> list[str]:
        errors = list(getattr(state, "errors", []) or [])

        if not errors:
            return []

        lines: list[str] = []
        lines.append("## Errors")
        lines.append("")

        for error in errors:
            lines.append(f"- {clean_text(error)}")

        lines.append("")
        return lines

    def _render_ranked_sources(self, state: Any) -> list[str]:
        ranked_sources = list(getattr(state, "ranked_sources", []) or [])

        if not ranked_sources:
            sources = list(getattr(state, "sources", []) or [])

            if not sources:
                return ["## Sources", "", "No sources were found.", ""]

            lines: list[str] = []
            lines.append("## Sources")
            lines.append("")
            lines.append("| # | Title | Platform | Type | Year |")
            lines.append("| ---: | --- | --- | --- | ---: |")

            for index, source in enumerate(sources, start=1):
                title_cell = self._source_title_cell(source)
                platform = self._enum_value(getattr(source, "platform", None)) or "-"
                source_type = self._enum_value(getattr(source, "source_type", None)) or "-"
                year = str(getattr(source, "year", None) or "-")

                lines.append(
                    f"| {index} | {title_cell} | {self._escape_cell(platform)} | "
                    f"{self._escape_cell(source_type)} | {self._escape_cell(year)} |"
                )

            lines.append("")
            return lines

        summary_map = self._summary_map(state)

        lines: list[str] = []
        lines.append("## Ranked Sources")
        lines.append("")
        lines.append(
            "| # | Title | Platform | Type | Difficulty | Score | Confidence | Year | Summary |"
        )
        lines.append("| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |")

        for ranked in ranked_sources:
            source = getattr(ranked, "source", None)

            if source is None:
                continue

            rank = str(getattr(ranked, "rank", "-"))
            title_cell = self._source_title_cell(source)
            platform = self._enum_value(getattr(source, "platform", None)) or "-"
            source_type = self._enum_value(getattr(source, "source_type", None)) or "-"
            difficulty = self._enum_value(getattr(source, "difficulty", None)) or "-"
            score = self._format_float(getattr(ranked, "score", None))
            confidence = self._format_float(getattr(ranked, "confidence", None))
            year = str(getattr(source, "year", None) or "-")

            summary_text = self._summary_text(source, summary_map)
            summary_cell = "-"

            if summary_text:
                summary_cell = self._escape_cell(
                    truncate_text(summary_text, max_length=220, suffix="...")
                )

            lines.append(
                f"| {self._escape_cell(rank)} | {title_cell} | {self._escape_cell(platform)} | "
                f"{self._escape_cell(source_type)} | {self._escape_cell(difficulty)} | "
                f"{self._escape_cell(score)} | {self._escape_cell(confidence)} | "
                f"{self._escape_cell(year)} | {summary_cell} |"
            )

        lines.append("")
        return lines

    def _render_source_summaries(self, state: Any) -> list[str]:
        summaries = list(getattr(state, "source_summaries", []) or [])

        if not summaries:
            return []

        lines: list[str] = []
        lines.append("## Source Summaries")
        lines.append("")

        for summary_item in summaries:
            title = clean_text(self._get_field(summary_item, "title", "Untitled"))
            difficulty = self._enum_value(self._get_field(summary_item, "difficulty", None)) or "-"
            summary_text = clean_text(self._get_field(summary_item, "summary", ""))
            why_useful = clean_text(self._get_field(summary_item, "why_useful", ""))
            key_topics = self._list_field(summary_item, "key_topics")

            lines.append(f"### {title}")
            lines.append("")
            lines.append(f"Difficulty: {difficulty}")
            lines.append("")

            if summary_text:
                lines.append(summary_text)
                lines.append("")

            if why_useful:
                lines.append(f"Why useful: {why_useful}")
                lines.append("")

            if key_topics:
                lines.append(f"Key topics: {', '.join(key_topics)}")
                lines.append("")

        return lines

    def _render_learning_path(self, state: Any) -> list[str]:
        learning_path = getattr(state, "learning_path", None)

        if learning_path is None:
            return []

        steps = list(getattr(learning_path, "steps", []) or [])

        if not steps:
            return []

        enhancement_map = self._enhancement_map(state)

        lines: list[str] = []
        lines.append("## Learning Path")
        lines.append("")

        for step in steps:
            step_number = getattr(step, "step", "-")
            title = clean_text(getattr(step, "title", ""))
            objective = clean_text(getattr(step, "objective", ""))
            estimated_minutes = getattr(step, "estimated_minutes", None)
            resources = list(getattr(step, "resources", []) or [])

            lines.append(f"### Step {step_number}: {title}")
            lines.append("")

            if objective:
                lines.append(f"Objective: {objective}")
                lines.append("")

            if estimated_minutes is not None:
                lines.append(f"Estimated time: {estimated_minutes} minutes")
                lines.append("")

            enhancement = enhancement_map.get(step_number)

            if enhancement is not None:
                prerequisites = self._list_field(enhancement, "prerequisites")
                suggested_projects = self._list_field(enhancement, "suggested_projects")
                key_concepts = self._list_field(enhancement, "key_concepts")

                if prerequisites:
                    lines.append("Prerequisites:")
                    lines.append("")
                    for item in prerequisites:
                        lines.append(f"- {item}")
                    lines.append("")

                if suggested_projects:
                    lines.append("Suggested projects:")
                    lines.append("")
                    for item in suggested_projects:
                        lines.append(f"- {item}")
                    lines.append("")

                if key_concepts:
                    lines.append("Key concepts:")
                    lines.append("")
                    for item in key_concepts:
                        lines.append(f"- {item}")
                    lines.append("")

            if resources:
                lines.append("Resources:")
                lines.append("")

                for ranked in resources:
                    source = getattr(ranked, "source", None)

                    if source is None:
                        continue

                    source_title = clean_text(getattr(source, "title", "Untitled"))
                    source_url = str(getattr(source, "url", "") or "")

                    if source_url:
                        safe_url = source_url.replace("|", "%7C")
                        lines.append(f"- [{source_title}]({safe_url})")
                    else:
                        lines.append(f"- {source_title}")

                lines.append("")

        return lines

    def _render_saved_files(self, state: Any) -> list[str]:
        saved_files = getattr(state, "saved_files", {}) or {}

        if not saved_files:
            return []

        lines: list[str] = []
        lines.append("## Saved Files")
        lines.append("")
        lines.append("| Type | Path |")
        lines.append("| --- | --- |")

        for file_type, file_path in saved_files.items():
            lines.append(
                f"| {self._escape_cell(file_type)} | {self._escape_cell(file_path)} |"
            )

        lines.append("")
        return lines

    def _source_title_cell(self, source: Any) -> str:
        title = clean_text(getattr(source, "title", "Untitled")) or "Untitled"
        url = str(getattr(source, "url", "") or "")

        escaped_title = self._escape_cell(title)

        if not url:
            return escaped_title

        safe_url = url.replace("|", "%7C")
        return f"[{escaped_title}]({safe_url})"

    def _summary_map(self, state: Any) -> dict[str, Any]:
        summaries = list(getattr(state, "source_summaries", []) or [])
        summary_map: dict[str, Any] = {}

        for item in summaries:
            source_id = self._get_field(item, "source_id", None)

            if source_id:
                summary_map[str(source_id)] = item

        return summary_map

    def _summary_text(self, source: Any, summary_map: dict[str, Any]) -> str:
        source_summary = getattr(source, "summary", None)

        if source_summary:
            return clean_text(source_summary)

        source_id = str(getattr(source, "source_id", "") or "")
        summary_item = summary_map.get(source_id)

        if summary_item is None:
            return ""

        return clean_text(self._get_field(summary_item, "summary", ""))

    def _enhancement_map(self, state: Any) -> dict[Any, Any]:
        enhanced_learning_path = getattr(state, "enhanced_learning_path", None)

        if enhanced_learning_path is None:
            return {}

        enhancements = self._get_field(enhanced_learning_path, "enhancements", [])

        if not isinstance(enhancements, list):
            return {}

        enhancement_map: dict[Any, Any] = {}

        for enhancement in enhancements:
            step_number = self._get_field(enhancement, "step", None)

            if step_number is not None:
                enhancement_map[step_number] = enhancement

        return enhancement_map

    def _query_field(self, query: Any, field: str, default: str) -> str:
        if query is None:
            return default

        value = getattr(query, field, None)

        if value is None:
            return default

        return str(value)

    def _get_field(self, item: Any, field: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(field, default)

        return getattr(item, field, default)

    def _list_field(self, item: Any, field: str) -> list[str]:
        value = self._get_field(item, field, [])

        if not isinstance(value, list):
            return []

        cleaned: list[str] = []

        for raw_item in value:
            text = clean_text(raw_item)

            if text:
                cleaned.append(text)

        return cleaned

    def _escape_cell(self, value: Any) -> str:
        text = clean_text(value)
        text = text.replace("|", "\\|")
        text = text.replace("\n", " ")
        return text

    def _format_float(self, value: Any) -> str:
        if value is None:
            return "-"

        try:
            return f"{float(value):.2f}"
        except Exception:
            return str(value)

    @staticmethod
    def _enum_value(value: Any) -> str:
        if value is None:
            return ""

        return str(getattr(value, "value", value))