from __future__ import annotations

import html
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import get_reports_output_directory
from utils.logger import get_logger
from utils.text import clean_text, slugify


_CSS = """
:root {
    --bg: #0b0f17;
    --surface: #121826;
    --surface-2: #1a2233;
    --border: #232d42;
    --text: #e6ebf5;
    --muted: #8a94ab;
    --accent: #4f8cff;
    --green: #34d399;
    --yellow: #fbbf24;
    --red: #f87171;
    --blue: #60a5fa;
}
* { box-sizing: border-box; }
body {
    margin: 0;
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
}
.container { max-width: 1100px; margin: 0 auto; padding: 24px; }
.header {
    background: linear-gradient(135deg, #16213a, #0f1524);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 24px;
}
.header h1 { margin: 0 0 8px; font-size: 26px; }
.meta-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin-top: 16px;
}
.meta-item { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px; }
.meta-label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
.meta-value { font-size: 15px; margin-top: 4px; word-break: break-word; }
.card-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px; margin-bottom: 24px; }
.stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 18px; text-align: center; }
.stat-number { font-size: 28px; font-weight: 700; color: var(--accent); }
.stat-label { color: var(--muted); font-size: 13px; margin-top: 4px; }
.section-title { font-size: 20px; margin: 28px 0 14px; border-left: 4px solid var(--accent); padding-left: 10px; }
table { width: 100%; border-collapse: collapse; background: var(--surface); border-radius: 12px; overflow: hidden; }
th, td { padding: 12px 14px; text-align: left; border-bottom: 1px solid var(--border); font-size: 14px; }
th { background: var(--surface-2); color: var(--muted); text-transform: uppercase; font-size: 12px; letter-spacing: 0.5px; }
tr:last-child td { border-bottom: none; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
.badge-beginner { background: rgba(52, 211, 153, 0.15); color: var(--green); }
.badge-intermediate { background: rgba(251, 191, 36, 0.15); color: var(--yellow); }
.badge-advanced { background: rgba(248, 113, 113, 0.15); color: var(--red); }
.badge-neutral { background: rgba(138, 148, 171, 0.15); color: var(--muted); }
.badge-platform { background: rgba(96, 165, 250, 0.15); color: var(--blue); }
.score-bar { width: 110px; height: 8px; background: var(--surface-2); border-radius: 999px; overflow: hidden; }
.score-fill { height: 100%; border-radius: 999px; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.step-card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 20px; margin-bottom: 16px; }
.step-header { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
.step-number {
    background: var(--accent); color: #fff; width: 32px; height: 32px;
    border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; flex-shrink: 0;
}
.step-title { font-size: 17px; font-weight: 600; }
.step-objective { color: var(--muted); margin: 6px 0 12px; }
.resource-list { list-style: none; padding: 0; margin: 0; }
.resource-list li { padding: 8px 0; border-bottom: 1px solid var(--border); font-size: 14px; }
.resource-list li:last-child { border-bottom: none; }
.enhancement-block { margin-top: 14px; padding: 12px; background: var(--surface-2); border-radius: 10px; }
.enhancement-title { font-size: 13px; color: var(--muted); text-transform: uppercase; margin-bottom: 6px; }
.enhancement-block ul { margin: 0; padding-left: 18px; font-size: 13px; }
.error-panel { background: rgba(248, 113, 113, 0.1); border: 1px solid var(--red); border-radius: 12px; padding: 16px; margin-bottom: 20px; }
.error-panel pre { margin: 0; white-space: pre-wrap; font-size: 13px; color: var(--red); }
.footer { color: var(--muted); text-align: center; font-size: 12px; margin-top: 32px; }
"""


class HTMLWriter:
    def __init__(self, file_manager: Any = None) -> None:
        self._file_manager = file_manager
        self._logger = get_logger("storage.html_writer")

    def write_state(
        self,
        state: Any,
        filename: str | None = None,
        subdirectory: str = "reports",
    ) -> Path:
        html_content = self.render_state(state)

        if filename is None:
            prefix = self._file_prefix(state)
            filename = self._timestamped_filename(prefix, "html")

        output_dir = get_reports_output_directory()
        target_path = output_dir / filename

        self._atomic_write(target_path, html_content)
        return target_path

    def render_state(self, state: Any) -> str:
        topic = self._query_field(state, "topic", "Research Report")
        goal = self._query_field(state, "goal", "-")
        level = self._query_field(state, "level", "-")

        request_id = str(getattr(state, "request_id", "") or "")
        status = self._enum_value(getattr(state, "status", None)) or "unknown"
        latency_ms = getattr(state, "latency_ms", None)
        errors = list(getattr(state, "errors", []) or [])
        ranked_sources = list(getattr(state, "ranked_sources", []) or [])
        learning_path = getattr(state, "learning_path", None)
        generated_at = datetime.now(timezone.utc)

        total_sources = len(list(getattr(state, "sources", []) or []))
        total_ranked = len(ranked_sources)
        total_steps = len(getattr(learning_path, "steps", []) or []) if learning_path else 0

        body_parts: list[str] = []
        body_parts.append(self._render_header(topic, goal, level, request_id, status, latency_ms))
        body_parts.append(self._render_stat_cards(total_sources, total_ranked, total_steps, len(errors)))

        if errors:
            body_parts.append(self._render_errors(errors))

        if ranked_sources:
            body_parts.append(self._render_ranked_sources(ranked_sources))

        if learning_path is not None:
            body_parts.append(self._render_learning_path(learning_path))

        body_parts.append(self._render_footer(generated_at))

        return self._wrap_document(topic, "".join(body_parts))

    def render_result(self, result: Any) -> str:
        return self.render_state(result)

    def _wrap_document(self, title: str, body: str) -> str:
        escaped_title = html.escape(clean_text(title) or "Research Report")
        return (
            "<!DOCTYPE html>\n"
            "<html lang=\"en\">\n"
            "<head>\n"
            "<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"<title>{escaped_title}</title>\n"
            f"<style>{_CSS}</style>\n"
            "</head>\n"
            "<body>\n"
            "<div class=\"container\">\n"
            f"{body}\n"
            "</div>\n"
            "</body>\n"
            "</html>\n"
        )

    def _render_header(
        self,
        topic: str,
        goal: str,
        level: str,
        request_id: str,
        status: str,
        latency_ms: Any,
    ) -> str:
        status_badge = self._status_badge(status)
        latency_text = f"{float(latency_ms):.0f} ms" if latency_ms is not None else "-"

        return (
            "<div class=\"header\">\n"
            f"<h1>{html.escape(clean_text(topic))}</h1>\n"
            f"<div>{status_badge}</div>\n"
            "<div class=\"meta-grid\">\n"
            f"<div class=\"meta-item\"><div class=\"meta-label\">Goal</div><div class=\"meta-value\">{html.escape(clean_text(goal))}</div></div>\n"
            f"<div class=\"meta-item\"><div class=\"meta-label\">Level</div><div class=\"meta-value\">{html.escape(clean_text(level))}</div></div>\n"
            f"<div class=\"meta-item\"><div class=\"meta-label\">Request ID</div><div class=\"meta-value\">{html.escape(request_id)}</div></div>\n"
            f"<div class=\"meta-item\"><div class=\"meta-label\">Latency</div><div class=\"meta-value\">{html.escape(latency_text)}</div></div>\n"
            "</div>\n"
            "</div>\n"
        )

    def _render_stat_cards(
        self,
        total_sources: int,
        total_ranked: int,
        total_steps: int,
        total_errors: int,
    ) -> str:
        cards = [
            ("Sources", total_sources),
            ("Ranked", total_ranked),
            ("Learning Steps", total_steps),
            ("Errors", total_errors),
        ]

        parts = ["<div class=\"card-row\">\n"]
        for label, value in cards:
            parts.append(
                "<div class=\"stat-card\">\n"
                f"<div class=\"stat-number\">{value}</div>\n"
                f"<div class=\"stat-label\">{html.escape(label)}</div>\n"
                "</div>\n"
            )
        parts.append("</div>\n")
        return "".join(parts)

    def _render_errors(self, errors: list[str]) -> str:
        error_text = html.escape("\n".join(str(error) for error in errors))
        return (
            "<div class=\"error-panel\">\n"
            "<strong>Errors</strong>\n"
            f"<pre>{error_text}</pre>\n"
            "</div>\n"
        )

    def _render_ranked_sources(self, ranked_sources: list[Any]) -> str:
        rows: list[str] = []

        for ranked in ranked_sources:
            source = getattr(ranked, "source", None)
            if source is None:
                continue

            title = clean_text(getattr(source, "title", "Untitled"))
            url = str(getattr(source, "url", "") or "")
            platform = self._enum_value(getattr(source, "platform", None))
            source_type = self._enum_value(getattr(source, "source_type", None))
            difficulty = self._enum_value(getattr(source, "difficulty", None)) or "unknown"
            score = float(getattr(ranked, "score", 0.0) or 0.0)
            confidence = float(getattr(ranked, "confidence", 0.0) or 0.0)
            rank = int(getattr(ranked, "rank", 0) or 0)

            title_cell = (
                f"<a href=\"{html.escape(url)}\" target=\"_blank\">{html.escape(title)}</a>"
                if url
                else html.escape(title)
            )

            rows.append(
                "<tr>\n"
                f"<td>{rank}</td>\n"
                f"<td>{title_cell}</td>\n"
                f"<td><span class=\"badge badge-platform\">{html.escape(platform)}</span></td>\n"
                f"<td>{html.escape(source_type)}</td>\n"
                f"<td>{self._difficulty_badge(difficulty)}</td>\n"
                f"<td>{self._score_bar(score)}</td>\n"
                f"<td>{confidence:.2f}</td>\n"
                "</tr>\n"
            )

        return (
            "<div class=\"section-title\">Ranked Sources</div>\n"
            "<table>\n"
            "<thead>\n"
            "<tr><th>#</th><th>Title</th><th>Platform</th><th>Type</th><th>Difficulty</th><th>Score</th><th>Confidence</th></tr>\n"
            "</thead>\n"
            "<tbody>\n"
            + "".join(rows) +
            "</tbody>\n"
            "</table>\n"
        )

    def _render_learning_path(self, learning_path: Any) -> str:
        steps = list(getattr(learning_path, "steps", []) or [])
        if not steps:
            return ""

        parts = ["<div class=\"section-title\">Learning Path</div>\n"]

        for step in steps:
            step_number = int(getattr(step, "step", 0) or 0)
            title = clean_text(getattr(step, "title", ""))
            objective = clean_text(getattr(step, "objective", ""))
            estimated_minutes = getattr(step, "estimated_minutes", None)
            resources = list(getattr(step, "resources", []) or [])

            enhancements = getattr(step, "prerequisites", None)

            resource_items: list[str] = []
            for resource in resources:
                resource_source = getattr(resource, "source", None)
                if resource_source is None:
                    continue
                resource_title = clean_text(getattr(resource_source, "title", "Untitled"))
                resource_url = str(getattr(resource_source, "url", "") or "")
                if resource_url:
                    resource_items.append(
                        f"<li><a href=\"{html.escape(resource_url)}\" target=\"_blank\">{html.escape(resource_title)}</a></li>\n"
                    )
                else:
                    resource_items.append(f"<li>{html.escape(resource_title)}</li>\n")

            time_text = f"{estimated_minutes} minutes" if estimated_minutes else "-"

            parts.append(
                "<div class=\"step-card\">\n"
                "<div class=\"step-header\">\n"
                f"<div class=\"step-number\">{step_number}</div>\n"
                f"<div class=\"step-title\">{html.escape(title)}</div>\n"
                "</div>\n"
                f"<div class=\"step-objective\">{html.escape(objective)}</div>\n"
                f"<div class=\"meta-label\">Estimated Time: {html.escape(time_text)}</div>\n"
                "<ul class=\"resource-list\">\n" + "".join(resource_items) + "</ul>\n"
            )

            prerequisites = list(getattr(step, "prerequisites", []) or [])
            projects = list(getattr(step, "suggested_projects", []) or [])

            if prerequisites:
                parts.append(self._render_enhancement("Prerequisites", prerequisites))
            if projects:
                parts.append(self._render_enhancement("Suggested Projects", projects))

            parts.append("</div>\n")

        return "".join(parts)

    def _render_enhancement(self, title: str, items: list[str]) -> str:
        item_html = "".join(f"<li>{html.escape(clean_text(item))}</li>" for item in items)
        return (
            "<div class=\"enhancement-block\">\n"
            f"<div class=\"enhancement-title\">{html.escape(title)}</div>\n"
            f"<ul>{item_html}</ul>\n"
            "</div>\n"
        )

    def _render_footer(self, generated_at: datetime) -> str:
        timestamp = generated_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        return f"<div class=\"footer\">Generated by Research Agent on {timestamp}</div>\n"

    def _difficulty_badge(self, difficulty: str) -> str:
        normalized = difficulty.lower()
        if normalized in {"beginner", "intermediate", "advanced"}:
            return f"<span class=\"badge badge-{normalized}\">{html.escape(difficulty)}</span>"
        return f"<span class=\"badge badge-neutral\">{html.escape(difficulty)}</span>"

    def _status_badge(self, status: str) -> str:
        normalized = status.lower()
        color_map = {
            "success": "badge-beginner",
            "partial": "badge-intermediate",
            "failed": "badge-advanced",
        }
        badge_class = color_map.get(normalized, "badge-neutral")
        return f"<span class=\"badge {badge_class}\">{html.escape(status.upper())}</span>"

    def _score_bar(self, score: float) -> str:
        percentage = max(0, min(100, int(score * 100)))
        color = self._score_color(score)
        return (
            "<div class=\"score-bar\">\n"
            f"<div class=\"score-fill\" style=\"width: {percentage}%; background: {color};\"></div>\n"
            "</div>\n"
        )

    def _score_color(self, score: float) -> str:
        if score >= 0.75:
            return "#34d399"
        if score >= 0.5:
            return "#fbbf24"
        return "#f87171"

    def _query_field(self, state: Any, field: str, default: str) -> str:
        query = getattr(state, "query", None)
        if query is None:
            return default
        value = getattr(query, field, None)
        if value is None:
            return default
        return str(value)

    def _file_prefix(self, state: Any) -> str:
        topic = self._query_field(state, "topic", "research")
        cleaned = slugify(topic, max_length=60)
        return cleaned or "research"

    def _timestamped_filename(self, prefix: str, extension: str) -> str:
        if self._file_manager is not None and hasattr(self._file_manager, "timestamped_filename"):
            try:
                return self._file_manager.timestamped_filename(prefix, extension)
            except Exception:
                pass

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}.{extension}"

    def _atomic_write(self, target_path: Path, content: str) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target_path.with_suffix(target_path.suffix + ".tmp")

        with open(temp_path, "w", encoding="utf-8") as handle:
            handle.write(content)

        os.replace(temp_path, target_path)

    @staticmethod
    def _enum_value(value: Any) -> str:
        if value is None:
            return ""
        return str(getattr(value, "value", value))