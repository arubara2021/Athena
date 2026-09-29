from __future__ import annotations

import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import get_reports_output_directory
from utils.logger import get_logger
from utils.text import clean_text, slugify


_CSS = """
*, *::before, *::after { box-sizing: border-box; }
:root {
    --bg: #0a0e1a;
    --bg-2: #0d1220;
    --surface: #121826;
    --surface-2: #182135;
    --surface-3: #1f2942;
    --border: #26314d;
    --border-2: #334263;
    --text: #e8eefb;
    --text-2: #c3ccdf;
    --muted: #7d8aa6;
    --primary: #a855f7;
    --primary-2: #c084fc;
    --accent: #22d3ee;
    --accent-2: #67e8f9;
    --success: #34d399;
    --warning: #fbbf24;
    --danger: #f87171;
    --info: #60a5fa;
    --shadow: 0 10px 30px rgba(0,0,0,0.35);
    --radius: 16px;
    --radius-sm: 10px;
    --mono: ui-monospace, SFMono-Regular, "JetBrains Mono", Menlo, Consolas, monospace;
}
html[data-theme="light"] {
    --bg: #f6f7fb;
    --bg-2: #eef1f8;
    --surface: #ffffff;
    --surface-2: #f3f5fa;
    --surface-3: #e8ecf5;
    --border: #d8deeb;
    --border-2: #bfc8db;
    --text: #111827;
    --text-2: #374151;
    --muted: #6b7280;
    --shadow: 0 10px 30px rgba(15,23,42,0.08);
}
html, body { margin: 0; padding: 0; }
body {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI",
        Roboto, "Helvetica Neue", Arial, sans-serif;
    background:
        radial-gradient(1200px 600px at 10% -10%, rgba(168,85,247,0.10), transparent 60%),
        radial-gradient(900px 500px at 110% 10%, rgba(34,211,238,0.08), transparent 60%),
        var(--bg);
    color: var(--text);
    line-height: 1.6;
    min-height: 100vh;
    -webkit-font-smoothing: antialiased;
}
a { color: var(--accent); text-decoration: none; transition: color 0.15s ease; }
a:hover { color: var(--accent-2); text-decoration: underline; }
.page { max-width: 1180px; margin: 0 auto; padding: 32px 24px 64px; }

.top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 4px 24px;
}
.brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 700;
    letter-spacing: 0.14em;
    font-size: 13px;
    color: var(--text-2);
}
.brand-mark {
    width: 26px;
    height: 26px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 8px;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    color: #0a0e1a;
    font-size: 14px;
}
.theme-toggle {
    appearance: none;
    background: var(--surface);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 8px 14px;
    font: inherit;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s ease;
}
.theme-toggle:hover { border-color: var(--border-2); background: var(--surface-2); }

.hero {
    background: linear-gradient(140deg, var(--surface), var(--surface-2));
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 32px 32px 28px;
    margin-bottom: 24px;
    box-shadow: var(--shadow);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(600px 200px at 100% 0%, rgba(168,85,247,0.14), transparent 70%);
    pointer-events: none;
}
.hero-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.14em;
    margin-bottom: 14px;
}
.hero-status.success { background: rgba(52,211,153,0.16); color: var(--success); border: 1px solid rgba(52,211,153,0.35); }
.hero-status.partial { background: rgba(251,191,36,0.16); color: var(--warning); border: 1px solid rgba(251,191,36,0.35); }
.hero-status.failed  { background: rgba(248,113,113,0.16); color: var(--danger);  border: 1px solid rgba(248,113,113,0.35); }
.hero-status.unknown { background: rgba(125,138,166,0.16); color: var(--muted);   border: 1px solid var(--border); }
.hero-title {
    font-size: 34px;
    line-height: 1.15;
    margin: 0 0 10px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text);
    word-break: break-word;
}
.hero-goal {
    font-size: 15px;
    color: var(--text-2);
    margin: 0 0 22px;
    max-width: 720px;
}
.hero-meta {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 14px;
    position: relative;
    z-index: 1;
}
.meta-item {
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 12px 14px;
}
html[data-theme="light"] .meta-item { background: var(--surface); }
.meta-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.14em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 4px;
}
.meta-value {
    font-size: 14px;
    color: var(--text);
    word-break: break-word;
    font-family: var(--mono);
}

.stat-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 14px;
    margin-bottom: 24px;
}
.stat {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 18px 20px;
    text-align: left;
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.stat:hover { transform: translateY(-2px); border-color: var(--border-2); }
.stat-num {
    font-size: 28px;
    font-weight: 700;
    color: var(--primary-2);
    font-family: var(--mono);
    display: block;
    line-height: 1;
    margin-bottom: 6px;
}
.stat-label {
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 600;
}

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px 26px;
    margin-bottom: 20px;
    box-shadow: var(--shadow);
}
.section-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 18px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border);
}
.section-title {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: var(--text);
    margin: 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title::before {
    content: "";
    width: 4px;
    height: 18px;
    border-radius: 2px;
    background: linear-gradient(180deg, var(--primary), var(--accent));
    display: inline-block;
}
.section-count {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.14em;
    color: var(--muted);
    font-family: var(--mono);
    text-transform: uppercase;
}

.funnel {
    display: grid;
    gap: 10px;
}
.funnel-row {
    display: grid;
    grid-template-columns: 120px 60px 1fr 70px;
    align-items: center;
    gap: 14px;
    font-size: 13px;
}
.funnel-label { color: var(--muted); text-transform: uppercase; letter-spacing: 0.14em; font-size: 11px; font-weight: 700; }
.funnel-value { color: var(--text); font-family: var(--mono); text-align: right; font-weight: 600; }
.funnel-bar { height: 8px; border-radius: 999px; background: var(--surface-3); overflow: hidden; }
.funnel-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--primary), var(--accent)); transition: width 0.3s ease; }
.funnel-share { color: var(--muted); font-family: var(--mono); text-align: right; font-size: 12px; }
.funnel-hint {
    margin-top: 14px;
    padding-top: 14px;
    border-top: 1px dashed var(--border);
    font-size: 13px;
    color: var(--muted);
    font-style: italic;
}

.src-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 14px; }
.src-table thead th {
    text-align: left;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 0 12px 12px;
    border-bottom: 1px solid var(--border);
}
.src-table tbody td {
    padding: 14px 12px;
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
    color: var(--text-2);
}
.src-table tbody tr:last-child td { border-bottom: none; }
.src-table tbody tr:hover { background: rgba(255,255,255,0.02); }
html[data-theme="light"] .src-table tbody tr:hover { background: var(--surface-2); }
.src-rank { font-family: var(--mono); color: var(--muted); text-align: right; width: 34px; }
.src-title { font-weight: 600; color: var(--text); }
.src-title a { color: var(--text); }
.src-title a:hover { color: var(--accent); }
.src-platform { font-size: 12px; color: var(--muted); }

.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    white-space: nowrap;
}
.badge.beginner     { background: rgba(52,211,153,0.15); color: var(--success); }
.badge.intermediate { background: rgba(251,191,36,0.15); color: var(--warning); }
.badge.advanced     { background: rgba(248,113,113,0.15); color: var(--danger); }
.badge.unknown      { background: rgba(125,138,166,0.15); color: var(--muted); }
.badge.platform     { background: rgba(96,165,250,0.15); color: var(--info); }
.badge.type         { background: rgba(168,85,247,0.15); color: var(--primary-2); }

.score-cell { display: flex; align-items: center; gap: 10px; }
.score-bar { flex: 1; height: 6px; border-radius: 999px; background: var(--surface-3); overflow: hidden; min-width: 60px; }
.score-fill { height: 100%; border-radius: 999px; }
.score-num { font-family: var(--mono); font-size: 12px; color: var(--text); min-width: 40px; text-align: right; }

.summary-grid { display: grid; gap: 16px; }
.summary-card {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--primary);
    border-radius: var(--radius-sm);
    padding: 18px 20px;
}
.summary-head {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 10px;
    flex-wrap: wrap;
}
.summary-title { font-size: 15px; font-weight: 600; color: var(--text); margin: 0; }
.summary-body { font-size: 14px; color: var(--text-2); margin: 0 0 10px; }
.summary-why {
    font-size: 13px;
    color: var(--muted);
    margin: 0;
    padding-top: 10px;
    border-top: 1px dashed var(--border);
}
.summary-why strong { color: var(--text-2); }
.summary-topics { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 6px; }
.topic-chip {
    font-size: 11px;
    padding: 3px 9px;
    background: rgba(34,211,238,0.10);
    color: var(--accent);
    border: 1px solid rgba(34,211,238,0.25);
    border-radius: 999px;
    font-family: var(--mono);
}

.timeline { position: relative; margin: 0; padding: 4px 0 4px 44px; list-style: none; }
.timeline::before {
    content: "";
    position: absolute;
    left: 15px;
    top: 8px;
    bottom: 8px;
    width: 2px;
    background: linear-gradient(180deg, var(--primary), var(--accent), transparent);
    border-radius: 2px;
}
.step {
    position: relative;
    padding: 16px 0 20px;
    border-bottom: 1px dashed var(--border);
}
.step:last-child { border-bottom: none; }
.step::before {
    content: attr(data-step);
    position: absolute;
    left: -44px;
    top: 16px;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    color: #0a0e1a;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 14px;
    font-family: var(--mono);
    box-shadow: 0 0 0 4px var(--surface), 0 0 0 5px var(--border);
}
.step-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
    margin-bottom: 8px;
    flex-wrap: wrap;
}
.step-title { font-size: 16px; font-weight: 700; color: var(--text); margin: 0; }
.step-minutes {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    font-family: var(--mono);
    white-space: nowrap;
}
.step-objective { font-size: 14px; color: var(--text-2); margin: 6px 0 12px; }
.resource-list { list-style: none; padding: 0; margin: 0; display: grid; gap: 6px; }
.resource-item {
    padding: 8px 12px;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 13px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.resource-item::before {
    content: "→";
    color: var(--accent);
    font-family: var(--mono);
    flex-shrink: 0;
}
.enh-block {
    margin-top: 12px;
    padding: 12px 14px;
    background: var(--surface-3);
    border-radius: 8px;
    font-size: 13px;
}
.enh-block-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
}
.enh-list { margin: 0; padding-left: 18px; color: var(--text-2); }

.diag-list { display: grid; gap: 8px; }
.diag-item {
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 13px;
    font-family: var(--mono);
    word-break: break-word;
}
.diag-item.error   { background: rgba(248,113,113,0.10); color: var(--danger); border-left: 3px solid var(--danger); }
.diag-item.warning { background: rgba(251,191,36,0.10); color: var(--warning); border-left: 3px solid var(--warning); }
.diag-item.ok      { background: rgba(52,211,153,0.10); color: var(--success); border-left: 3px solid var(--success); }

.file-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.file-table th {
    text-align: left;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 0 12px 12px;
    border-bottom: 1px solid var(--border);
}
.file-table td {
    padding: 12px;
    border-bottom: 1px solid var(--border);
    color: var(--text-2);
    font-family: var(--mono);
    font-size: 12px;
    word-break: break-all;
}
.file-table tr:last-child td { border-bottom: none; }
.file-kind { font-family: inherit; font-weight: 700; color: var(--primary-2); text-transform: uppercase; font-size: 11px; letter-spacing: 0.14em; }

.footer {
    text-align: center;
    color: var(--muted);
    font-size: 12px;
    padding-top: 32px;
    letter-spacing: 0.06em;
}

@media (max-width: 720px) {
    .hero { padding: 22px; }
    .hero-title { font-size: 24px; }
    .card { padding: 18px; }
    .funnel-row { grid-template-columns: 90px 50px 1fr 60px; gap: 8px; font-size: 12px; }
    .src-table thead { display: none; }
    .src-table tbody td { display: block; padding: 6px 0; border: none; }
    .src-table tbody tr { display: block; padding: 14px 0; border-bottom: 1px solid var(--border); }
}

@media print {
    body { background: #fff; color: #000; }
    .theme-toggle { display: none; }
    .card { box-shadow: none; border: 1px solid #ccc; page-break-inside: avoid; }
    .hero { box-shadow: none; border: 1px solid #ccc; }
    a { color: #000; text-decoration: underline; }
}
"""

_JS = """
(function() {
    var root = document.documentElement;
    var stored = null;
    try { stored = localStorage.getItem('athena-theme'); } catch(e) {}
    if (stored === 'light') { root.setAttribute('data-theme', 'light'); }
    window.toggleTheme = function() {
        var now = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
        root.setAttribute('data-theme', now);
        try { localStorage.setItem('athena-theme', now); } catch(e) {}
    };
})();
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
        payload = self._state_to_payload(state)
        return self.render_payload(payload)

    def render_result(self, result: Any) -> str:
        return self.render_state(result)

    def render_payload(self, payload: dict[str, Any]) -> str:
        try:
            data = self._normalize_payload(payload)
        except Exception:
            data = {"topic": "Research Report", "status": "unknown"}

        title = data.get("topic") or "Research Report"
        body_parts = [
            self._render_topbar(),
            self._render_hero(data),
            self._render_stats(data),
        ]

        if data.get("source_count", 0) > 0:
            body_parts.append(self._render_funnel(data))

        if data.get("token_budget", 0) > 0 or data.get("tokens_used", 0) > 0:
            body_parts.append(self._render_tokens(data))

        if data.get("ranked_sources"):
            body_parts.append(self._render_sources(data))

        if data.get("source_summaries"):
            body_parts.append(self._render_summaries(data))

        if data.get("learning_path"):
            body_parts.append(self._render_learning_path(data))

        if data.get("platform_counts"):
            body_parts.append(self._render_platforms(data))

        body_parts.append(self._render_diagnostics(data))

        if data.get("saved_files"):
            body_parts.append(self._render_saved_files(data))

        body_parts.append(self._render_footer())

        return self._wrap_document(title, "".join(body_parts))

    def _state_to_payload(self, state: Any) -> dict[str, Any]:
        try:
            model_dump = getattr(state, "model_dump", None)
            if callable(model_dump):
                return model_dump(mode="json", exclude_none=False, by_alias=False)
        except Exception:
            pass

        return {
            "request_id": getattr(state, "request_id", ""),
            "status": self._enum_value(getattr(state, "status", "")),
            "query": {
                "topic": getattr(getattr(state, "query", None), "topic", ""),
                "goal": getattr(getattr(state, "query", None), "goal", ""),
                "level": getattr(getattr(state, "query", None), "level", ""),
            },
            "ranked_sources": [
                self._model_to_dict(item)
                for item in (getattr(state, "ranked_sources", []) or [])
            ],
            "source_summaries": [
                self._model_to_dict(item)
                for item in (getattr(state, "source_summaries", []) or [])
            ],
            "learning_path": self._model_to_dict(
                getattr(state, "learning_path", None)
            ),
            "errors": list(getattr(state, "errors", []) or []),
            "warnings": list(getattr(state, "warnings", []) or []),
            "saved_files": dict(getattr(state, "saved_files", {}) or {}),
            "latency_ms": getattr(state, "latency_ms", None),
            "rag_documents_indexed": getattr(state, "rag_documents_indexed", 0),
        }

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {"topic": "Research Report", "status": "unknown"}

        query = payload.get("query") or {}
        if not isinstance(query, dict):
            query = {}

        topic = (
            payload.get("topic")
            or query.get("topic")
            or payload.get("corrected_topic")
            or payload.get("goal")
            or "Research Report"
        )
        goal = payload.get("goal") or query.get("goal") or ""
        level = payload.get("level") or query.get("level") or ""
        request_id = (
            payload.get("request_id")
            or payload.get("agent_id")
            or ""
        )
        kind = (
            payload.get("kind")
            or ("agent" if payload.get("agent_id") else "pipeline")
        )

        sources = payload.get("sources") or payload.get("findings") or []
        ranked = payload.get("ranked_sources") or []
        summaries = (
            payload.get("source_summaries")
            or payload.get("summaries")
            or []
        )

        learning_path = payload.get("learning_path")
        enhanced = payload.get("enhanced_learning_path")

        platform_counts: dict[str, int] = {}
        for item in (ranked or sources):
            src = item.get("source") if isinstance(item, dict) else None
            if src is None and isinstance(item, dict):
                src = item
            if not isinstance(src, dict):
                continue
            platform = str(src.get("platform") or "unknown").lower()
            platform_counts[platform] = platform_counts.get(platform, 0) + 1

        budget = self._to_int(payload.get("total_budget"), 0)
        used = self._to_int(payload.get("total_tokens_used"), 0)
        remaining = self._to_int(payload.get("tokens_remaining"), 0)

        return {
            "kind": kind,
            "topic": topic,
            "goal": goal,
            "level": level,
            "request_id": request_id,
            "status": str(payload.get("status") or "unknown").lower(),
            "mode": payload.get("mode") or "",
            "latency_ms": payload.get("latency_ms"),
            "source_count": len(sources),
            "ranked_count": len(ranked),
            "summary_count": len(summaries),
            "learning_step_count": self._count_steps(learning_path),
            "path_resource_count": self._count_resources(learning_path),
            "rag_documents_indexed": self._to_int(
                payload.get("rag_documents_indexed"), 0
            ),
            "ranked_sources": ranked,
            "source_summaries": summaries,
            "learning_path": learning_path,
            "enhanced_learning_path": enhanced,
            "errors": list(payload.get("errors") or []),
            "warnings": list(payload.get("warnings") or []),
            "saved_files": dict(payload.get("saved_files") or {}),
            "platform_counts": platform_counts,
            "total_iterations": self._to_int(payload.get("total_iterations"), 0),
            "tokens_used": used,
            "token_budget": budget,
            "tokens_remaining": remaining,
            "target_domains": list(payload.get("target_domains") or []),
            "detected_level": payload.get("detected_level") or "",
            "domain_confidence": payload.get("domain_confidence") or 0.0,
            "expanded_queries": list(payload.get("expanded_queries") or []),
            "keywords": list(payload.get("keywords") or []),
        }

    def _wrap_document(self, title: str, body: str) -> str:
        escaped_title = html.escape(clean_text(title) or "Research Report")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en" data-theme="dark">\n'
            "<head>\n"
            '<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<meta name="description" content="ATHENA research report for {escaped_title}">\n'
            f"<title>{escaped_title} · ATHENA</title>\n"
            f"<style>{_CSS}</style>\n"
            f"<script>{_JS}</script>\n"
            "</head>\n"
            "<body>\n"
            '<div class="page">\n'
            f"{body}\n"
            "</div>\n"
            "</body>\n"
            "</html>\n"
        )

    def _render_topbar(self) -> str:
        return (
            '<div class="top-bar">\n'
            '  <div class="brand"><span class="brand-mark">◆</span>'
            '<span>ATHENA · REPORT</span></div>\n'
            '  <button class="theme-toggle" onclick="toggleTheme()">'
            'Toggle theme</button>\n'
            '</div>\n'
        )

    def _render_hero(self, data: dict[str, Any]) -> str:
        status = str(data.get("status", "unknown")).lower()
        status_class = status if status in (
            "success", "partial", "failed"
        ) else "unknown"
        status_label = {
            "success": "SUCCESS",
            "partial": "PARTIAL",
            "failed": "FAILED",
        }.get(status, status.upper() if status else "UNKNOWN")

        topic = clean_text(data.get("topic") or "Research Report")
        goal = clean_text(data.get("goal") or "")
        level = clean_text(data.get("level") or data.get("detected_level") or "-")
        request_id = clean_text(data.get("request_id") or "-")
        latency = data.get("latency_ms")
        latency_text = self._format_latency(latency)

        meta_items: list[tuple[str, str]] = [
            ("Level", level),
            ("Duration", latency_text),
            ("Request", request_id),
        ]

        if data.get("kind") == "agent":
            meta_items.append(("Iterations", str(data.get("total_iterations") or 0)))

        if data.get("detected_level") and data.get("detected_level") != level:
            meta_items.append(("Detected", str(data.get("detected_level"))))

        meta_html = "".join(
            f'<div class="meta-item">'
            f'<div class="meta-label">{html.escape(k)}</div>'
            f'<div class="meta-value">{html.escape(v)}</div>'
            f'</div>'
            for k, v in meta_items
        )

        goal_block = (
            f'<p class="hero-goal">{html.escape(goal)}</p>'
            if goal else ""
        )

        return (
            '<section class="hero">\n'
            f'  <span class="hero-status {status_class}">{html.escape(status_label)}</span>\n'
            f'  <h1 class="hero-title">{html.escape(topic)}</h1>\n'
            f'  {goal_block}\n'
            f'  <div class="hero-meta">{meta_html}</div>\n'
            '</section>\n'
        )

    def _render_stats(self, data: dict[str, Any]) -> str:
        cards = [
            ("Sources", data.get("source_count", 0)),
            ("Ranked", data.get("ranked_count", 0)),
            ("Summaries", data.get("summary_count", 0)),
            ("Steps", data.get("learning_step_count", 0)),
        ]

        if data.get("rag_documents_indexed"):
            cards.append(("Indexed", data.get("rag_documents_indexed")))

        if data.get("kind") == "agent" and data.get("tokens_used"):
            cards.append(("Tokens", data.get("tokens_used")))

        return (
            '<section class="stat-row">\n'
            + "".join(
                f'  <div class="stat"><span class="stat-num">{html.escape(str(v))}</span>'
                f'<span class="stat-label">{html.escape(k)}</span></div>\n'
                for k, v in cards
            )
            + '</section>\n'
        )

    def _render_funnel(self, data: dict[str, Any]) -> str:
        stages = [
            ("Retrieved", int(data.get("source_count", 0) or 0)),
            ("Ranked", int(data.get("ranked_count", 0) or 0)),
            ("Summarized", int(data.get("summary_count", 0) or 0)),
            ("Path used", int(data.get("path_resource_count", 0) or 0)),
        ]
        max_val = max((v for _, v in stages), default=0) or 1

        rows = []
        for label, value in stages:
            share = (value / max_val) * 100.0 if max_val else 0.0
            rows.append(
                '<div class="funnel-row">'
                f'<div class="funnel-label">{html.escape(label)}</div>'
                f'<div class="funnel-value">{value}</div>'
                f'<div class="funnel-bar"><div class="funnel-fill" style="width:{share:.1f}%"></div></div>'
                f'<div class="funnel-share">{share:5.1f}%</div>'
                '</div>'
            )

        hint = ""
        retrieved = stages[0][1]
        ranked = stages[1][1]
        summarized = stages[2][1]
        if retrieved > 0 and ranked < retrieved:
            hint = (
                f"Relevance and topic-presence gates dropped "
                f"{retrieved - ranked} of {retrieved} retrieved sources "
                f"before ranking."
            )
        elif retrieved > 0 and ranked == retrieved and summarized < ranked:
            hint = (
                f"Summarizer capped output at {summarized}; "
                f"{ranked - summarized} ranked sources were not summarized."
            )
        elif retrieved > 0 and all(v == retrieved for _, v in stages):
            hint = "No shrinkage: every stage kept the same number of items."

        hint_html = (
            f'<div class="funnel-hint">{html.escape(hint)}</div>'
            if hint else ""
        )

        return (
            '<section class="card">\n'
            '  <div class="section-head">'
            '<h2 class="section-title">Research Funnel</h2>'
            '</div>\n'
            '  <div class="funnel">'
            + "".join(rows)
            + '</div>\n'
            f'  {hint_html}\n'
            '</section>\n'
        )

    def _render_tokens(self, data: dict[str, Any]) -> str:
        budget = int(data.get("token_budget", 0) or 0)
        used = int(data.get("tokens_used", 0) or 0)
        remaining = int(data.get("tokens_remaining", 0) or 0)
        percent = (used / budget) * 100.0 if budget > 0 else 0.0

        return (
            '<section class="card">\n'
            '  <div class="section-head">'
            '<h2 class="section-title">Token Economics</h2>'
            f'<span class="section-count">{used} / {budget}</span>'
            '</div>\n'
            '  <div class="funnel">\n'
            f'    <div class="funnel-row"><div class="funnel-label">Used</div>'
            f'<div class="funnel-value">{used}</div>'
            f'<div class="funnel-bar"><div class="funnel-fill" style="width:{percent:.1f}%"></div></div>'
            f'<div class="funnel-share">{percent:5.1f}%</div></div>\n'
            f'    <div class="funnel-row"><div class="funnel-label">Remaining</div>'
            f'<div class="funnel-value">{remaining}</div>'
            f'<div class="funnel-bar"></div><div class="funnel-share"></div></div>\n'
            '  </div>\n'
            '</section>\n'
        )

    def _render_sources(self, data: dict[str, Any]) -> str:
        rows = []
        for index, item in enumerate(data.get("ranked_sources") or [], start=1):
            if not isinstance(item, dict):
                continue
            src = item.get("source") if isinstance(item.get("source"), dict) else item
            if not isinstance(src, dict):
                continue

            rank = item.get("rank") or index
            title = clean_text(src.get("title") or "Untitled")
            url = str(src.get("url") or "").strip()
            platform = str(src.get("platform") or "-").lower()
            stype = str(src.get("source_type") or "-").replace("_", " ")
            difficulty = str(src.get("difficulty") or "unknown").lower()
            score = self._to_float(item.get("score"), 0.0)
            confidence = self._to_float(item.get("confidence"), 0.0)
            year = src.get("year")

            title_html = (
                f'<a href="{html.escape(url)}" target="_blank" rel="noopener">'
                f'{html.escape(title)}</a>'
                if url else html.escape(title)
            )

            score_color = self._score_color(score)
            score_pct = max(0, min(100, int(score * 100)))

            year_html = f'<span class="src-platform">{html.escape(str(year))}</span>' if year else ""

            rows.append(
                '<tr>'
                f'<td class="src-rank">{html.escape(str(rank))}</td>'
                f'<td><div class="src-title">{title_html}</div>'
                f'<div class="src-platform">{html.escape(platform)} · {html.escape(stype)} {year_html}</div></td>'
                f'<td><span class="badge platform">{html.escape(platform)}</span></td>'
                f'<td><span class="badge {html.escape(difficulty)}">{html.escape(difficulty)}</span></td>'
                f'<td><div class="score-cell">'
                f'<div class="score-bar"><div class="score-fill" style="width:{score_pct}%;background:{score_color}"></div></div>'
                f'<span class="score-num">{score:.2f}</span></div></td>'
                f'<td><span class="score-num">{confidence:.2f}</span></td>'
                '</tr>'
            )

        return (
            '<section class="card">\n'
            f'  <div class="section-head"><h2 class="section-title">Ranked Sources</h2>'
            f'<span class="section-count">{len(rows)} items</span></div>\n'
            '  <table class="src-table">\n'
            '    <thead><tr>'
            '<th>#</th><th>Title</th><th>Platform</th><th>Difficulty</th>'
            '<th>Score</th><th>Conf</th>'
            '</tr></thead>\n'
            '    <tbody>'
            + "".join(rows) +
            '</tbody>\n'
            '  </table>\n'
            '</section>\n'
        )

    def _render_summaries(self, data: dict[str, Any]) -> str:
        cards = []
        for item in (data.get("source_summaries") or [])[:12]:
            if not isinstance(item, dict):
                continue

            title = clean_text(item.get("title") or "Untitled")
            difficulty = str(item.get("difficulty") or "unknown").lower()
            summary_text = clean_text(item.get("summary") or "")
            why = clean_text(item.get("why_useful") or "")
            topics = item.get("key_topics") or []

            topic_chips = "".join(
                f'<span class="topic-chip">{html.escape(str(t))}</span>'
                for t in (topics[:8] if isinstance(topics, list) else [])
            )
            topics_html = (
                f'<div class="summary-topics">{topic_chips}</div>'
                if topic_chips else ""
            )

            why_html = (
                f'<p class="summary-why"><strong>Why useful:</strong> '
                f'{html.escape(why)}</p>'
                if why else ""
            )

            summary_html = (
                f'<p class="summary-body">{html.escape(summary_text)}</p>'
                if summary_text else ""
            )

            cards.append(
                '<div class="summary-card">'
                '<div class="summary-head">'
                f'<h3 class="summary-title">{html.escape(title)}</h3>'
                f'<span class="badge {html.escape(difficulty)}">{html.escape(difficulty)}</span>'
                '</div>'
                f'{summary_html}'
                f'{why_html}'
                f'{topics_html}'
                '</div>'
            )

        return (
            '<section class="card">\n'
            f'  <div class="section-head"><h2 class="section-title">Source Summaries</h2>'
            f'<span class="section-count">{len(cards)} summaries</span></div>\n'
            '  <div class="summary-grid">'
            + "".join(cards) +
            '</div>\n'
            '</section>\n'
        )

    def _render_learning_path(self, data: dict[str, Any]) -> str:
        path = data.get("learning_path") or {}
        if not isinstance(path, dict):
            return ""
        steps = path.get("steps") or []
        if not isinstance(steps, list) or not steps:
            return ""

        enhanced = data.get("enhanced_learning_path")
        enhancements = {}
        if isinstance(enhanced, dict):
            for entry in (enhanced.get("enhancements") or []):
                if isinstance(entry, dict) and entry.get("step") is not None:
                    enhancements[entry["step"]] = entry

        items = []
        total_minutes = 0
        for step in steps:
            if not isinstance(step, dict):
                continue
            number = step.get("step") or 0
            title = clean_text(step.get("title") or "")
            objective = clean_text(step.get("objective") or "")
            minutes = int(step.get("estimated_minutes") or 0)
            total_minutes += minutes

            resources = step.get("resources") or []
            resource_html = ""
            if isinstance(resources, list) and resources:
                resource_items = []
                for res in resources:
                    if not isinstance(res, dict):
                        continue
                    src = res.get("source") if isinstance(res.get("source"), dict) else res
                    if not isinstance(src, dict):
                        continue
                    res_title = clean_text(src.get("title") or "Untitled")
                    res_url = str(src.get("url") or "").strip()
                    if res_url:
                        resource_items.append(
                            f'<li class="resource-item">'
                            f'<a href="{html.escape(res_url)}" target="_blank" rel="noopener">'
                            f'{html.escape(res_title)}</a></li>'
                        )
                    else:
                        resource_items.append(
                            f'<li class="resource-item">{html.escape(res_title)}</li>'
                        )
                if resource_items:
                    resource_html = (
                        f'<ul class="resource-list">{"".join(resource_items)}</ul>'
                    )

            enhancement = enhancements.get(number)
            enh_html = ""
            if isinstance(enhancement, dict):
                blocks = []
                for label, key in (
                    ("Prerequisites", "prerequisites"),
                    ("Key concepts", "key_concepts"),
                    ("Suggested projects", "suggested_projects"),
                ):
                    vals = enhancement.get(key) or []
                    if not isinstance(vals, list) or not vals:
                        continue
                    li = "".join(
                        f'<li>{html.escape(clean_text(v))}</li>' for v in vals[:8]
                    )
                    blocks.append(
                        f'<div class="enh-block">'
                        f'<div class="enh-block-label">{html.escape(label)}</div>'
                        f'<ul class="enh-list">{li}</ul></div>'
                    )
                if blocks:
                    enh_html = "".join(blocks)

            minutes_html = (
                f'<span class="step-minutes">{minutes} min</span>'
                if minutes else ""
            )
            objective_html = (
                f'<p class="step-objective">{html.escape(objective)}</p>'
                if objective and objective != title else ""
            )

            items.append(
                f'<li class="step" data-step="{html.escape(str(number))}">'
                '<div class="step-header">'
                f'<h3 class="step-title">{html.escape(title)}</h3>'
                f'{minutes_html}'
                '</div>'
                f'{objective_html}'
                f'{resource_html}'
                f'{enh_html}'
                '</li>'
            )

        total_html = (
            f'<span class="section-count">{total_minutes} min total</span>'
            if total_minutes else ""
        )

        return (
            '<section class="card">\n'
            f'  <div class="section-head"><h2 class="section-title">Learning Path</h2>'
            f'{total_html}</div>\n'
            f'  <ol class="timeline">{"".join(items)}</ol>\n'
            '</section>\n'
        )

    def _render_platforms(self, data: dict[str, Any]) -> str:
        counts = data.get("platform_counts") or {}
        if not isinstance(counts, dict) or len(counts) < 2:
            return ""

        total = sum(int(v or 0) for v in counts.values()) or 1
        max_count = max(int(v or 0) for v in counts.values()) or 1

        rows = []
        for name, count in sorted(
            counts.items(),
            key=lambda kv: -int(kv[1] or 0),
        )[:12]:
            share = (int(count or 0) / total) * 100.0
            width = (int(count or 0) / max_count) * 100.0
            rows.append(
                '<div class="funnel-row">'
                f'<div class="funnel-label">{html.escape(str(name))}</div>'
                f'<div class="funnel-value">{int(count or 0)}</div>'
                f'<div class="funnel-bar"><div class="funnel-fill" style="width:{width:.1f}%"></div></div>'
                f'<div class="funnel-share">{share:5.1f}%</div>'
                '</div>'
            )

        return (
            '<section class="card">\n'
            '  <div class="section-head"><h2 class="section-title">Platform Distribution</h2></div>\n'
            '  <div class="funnel">'
            + "".join(rows) +
            '</div>\n'
            '</section>\n'
        )

    def _render_diagnostics(self, data: dict[str, Any]) -> str:
        errors = data.get("errors") or []
        warnings = data.get("warnings") or []

        if not errors and not warnings:
            return (
                '<section class="card">\n'
                '  <div class="section-head"><h2 class="section-title">Diagnostics</h2></div>\n'
                '  <div class="diag-list">'
                '<div class="diag-item ok">No diagnostic issues detected.</div>'
                '</div>\n'
                '</section>\n'
            )

        items = []
        for err in errors[:12]:
            items.append(f'<div class="diag-item error">{html.escape(str(err))}</div>')
        for warn in warnings[:12]:
            items.append(f'<div class="diag-item warning">{html.escape(str(warn))}</div>')

        return (
            '<section class="card">\n'
            f'  <div class="section-head"><h2 class="section-title">Diagnostics</h2>'
            f'<span class="section-count">{len(errors)} error · {len(warnings)} warn</span></div>\n'
            '  <div class="diag-list">'
            + "".join(items) +
            '</div>\n'
            '</section>\n'
        )

    def _render_saved_files(self, data: dict[str, Any]) -> str:
        saved = data.get("saved_files") or {}
        if not isinstance(saved, dict) or not saved:
            return ""

        rows = []
        for key, value in saved.items():
            path = Path(str(value))
            size_text = "-"
            modified_text = "-"
            if path.exists():
                try:
                    stat = path.stat()
                    size_text = self._format_bytes(stat.st_size)
                    modified_text = datetime.fromtimestamp(
                        stat.st_mtime
                    ).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    pass

            rows.append(
                '<tr>'
                f'<td><span class="file-kind">{html.escape(str(key))}</span></td>'
                f'<td>{html.escape(str(value))}</td>'
                f'<td>{html.escape(size_text)}</td>'
                f'<td>{html.escape(modified_text)}</td>'
                '</tr>'
            )

        return (
            '<section class="card">\n'
            f'  <div class="section-head"><h2 class="section-title">Saved Files</h2>'
            f'<span class="section-count">{len(rows)} files</span></div>\n'
            '  <table class="file-table">\n'
            '    <thead><tr><th>Type</th><th>Path</th><th>Size</th><th>Modified</th></tr></thead>\n'
            '    <tbody>' + "".join(rows) + '</tbody>\n'
            '  </table>\n'
            '</section>\n'
        )

    def _render_footer(self) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        return (
            f'<div class="footer">Generated by ATHENA · {html.escape(stamp)}</div>\n'
        )

    def _score_color(self, score: float) -> str:
        if score >= 0.75:
            return "#34d399"
        if score >= 0.50:
            return "#fbbf24"
        if score >= 0.25:
            return "#f87171"
        return "#dc2626"

    def _format_latency(self, value: Any) -> str:
        try:
            ms = float(value)
        except (TypeError, ValueError):
            return "-"
        if ms < 1:
            return "<1 ms"
        if ms < 1000:
            return f"{ms:.0f} ms"
        if ms < 60000:
            return f"{ms / 1000.0:.2f} s"
        minutes = int(ms // 60000)
        seconds = (ms % 60000) / 1000.0
        return f"{minutes}m {seconds:04.1f}s"

    def _format_bytes(self, value: Any) -> str:
        try:
            size = float(value)
        except (TypeError, ValueError):
            return "-"
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024.0:
                return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
            size /= 1024.0
        return f"{size:.1f} PB"

    def _to_int(self, value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _to_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _count_steps(self, path: Any) -> int:
        if not isinstance(path, dict):
            return 0
        return len(path.get("steps") or [])

    def _count_resources(self, path: Any) -> int:
        if not isinstance(path, dict):
            return 0
        total = 0
        for step in path.get("steps") or []:
            if isinstance(step, dict):
                total += len(step.get("resources") or [])
        return total

    def _model_to_dict(self, obj: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj
        dump = getattr(obj, "model_dump", None)
        if callable(dump):
            try:
                return dump(mode="json", exclude_none=False, by_alias=False)
            except Exception:
                pass
        return str(obj)

    def _file_prefix(self, state: Any) -> str:
        query = getattr(state, "query", None)
        topic = str(getattr(query, "topic", "") or "research") if query else "research"
        return slugify(topic, max_length=60) or "research"

    def _timestamped_filename(self, prefix: str, extension: str) -> str:
        if self._file_manager is not None and hasattr(
            self._file_manager, "timestamped_filename"
        ):
            try:
                return self._file_manager.timestamped_filename(prefix, extension)
            except Exception:
                pass
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{stamp}.{extension}"

    def _atomic_write(self, target_path: Path, content: str) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = target_path.with_suffix(target_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp, target_path)

    @staticmethod
    def _enum_value(value: Any) -> str:
        if value is None:
            return ""
        return str(getattr(value, "value", value))