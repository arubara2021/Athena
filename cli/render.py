from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from cli import components as comp
from cli import format as fmt
from cli import style
from cli import theme as theme_module


def _console() -> Console:
    return theme_module.get_console()


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_report_data(result: Any, kind: str = "pipeline") -> dict[str, Any]:
    status = _enum_value(_get(result, "status", "")) or "unknown"
    topic = str(_get(result, "topic", "") or "")
    goal = str(_get(result, "goal", "") or "")
    level = str(_get(result, "level", "") or "")
    request_id = str(
        _get(result, "request_id", "")
        or _get(result, "agent_id", "")
        or ""
    )

    query = _get(result, "query", None)
    if query is not None:
        topic = topic or str(_get(query, "topic", "") or "")
        goal = goal or str(_get(query, "goal", "") or "")
        level = level or str(_get(query, "level", "") or "")

    corrected_topic = str(_get(result, "corrected_topic", "") or "")
    if corrected_topic:
        topic = corrected_topic

    sources = _as_list(
        _get(result, "sources", None)
        or _get(result, "findings", None)
    )
    ranked = _as_list(_get(result, "ranked_sources", None))
    summaries = _as_list(
        _get(result, "source_summaries", None)
        or _get(result, "summaries", None)
    )
    learning_path = _get(result, "learning_path", None)
    enhanced = _get(result, "enhanced_learning_path", None)
    errors = [str(item) for item in _as_list(_get(result, "errors", None))]
    warnings = [str(item) for item in _as_list(_get(result, "warnings", None))]
    saved_files = dict(_get(result, "saved_files", {}) or {})

    step_count = 0
    resource_count = 0
    if learning_path is not None:
        steps = _as_list(_get(learning_path, "steps", None))
        step_count = len(steps)
        for step in steps:
            resource_count += len(_as_list(_get(step, "resources", None)))

    platform_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    counted = 0

    if ranked:
        for item in ranked:
            source = _get(item, "source", None)
            if source is None:
                continue
            counted += 1
            platform_counts[
                _enum_value(_get(source, "platform", "")) or "unknown"
            ] += 1
            type_counts[
                _enum_value(_get(source, "source_type", "")) or "unknown"
            ] += 1
            difficulty_counts[
                _enum_value(_get(source, "difficulty", "")) or "unknown"
            ] += 1
    else:
        for source in sources:
            counted += 1
            platform_counts[
                _enum_value(_get(source, "platform", "")) or "unknown"
            ] += 1
            type_counts[
                _enum_value(_get(source, "source_type", "")) or "unknown"
            ] += 1

    budget = _to_int(_get(result, "total_budget", 0), 0)
    used = _to_int(_get(result, "total_tokens_used", 0), 0)
    remaining_attr = _to_int(_get(result, "tokens_remaining", 0), 0)

    if budget <= 0 and used > 0:
        budget = max(40000, used)

    if budget > 0:
        computed_remaining = max(0, budget - used)
    else:
        computed_remaining = remaining_attr

    if (
        remaining_attr <= 0
        or (budget > 0 and remaining_attr > budget)
        or abs(remaining_attr - computed_remaining) > 100
    ):
        remaining = computed_remaining
    else:
        remaining = remaining_attr

    return {
        "kind": kind,
        "status": status,
        "topic": topic,
        "goal": goal,
        "level": level,
        "request_id": request_id,
        "detected_level": str(_get(result, "detected_level", "") or ""),
        "target_domains": _as_list(_get(result, "target_domains", None)),
        "domain_confidence": _get(result, "domain_confidence", 0.0),
        "sources": sources,
        "ranked_sources": ranked,
        "summaries": summaries,
        "learning_path": learning_path,
        "enhanced_learning_path": enhanced,
        "errors": errors,
        "warnings": warnings,
        "saved_files": saved_files,
        "latency_ms": _get(result, "latency_ms", None),
        "iterations": _get(result, "total_iterations", None),
        "source_count": len(sources),
        "ranked_count": len(ranked),
        "summary_count": len(summaries),
        "learning_step_count": step_count,
        "path_resource_count": resource_count,
        "rag_docs": _to_int(_get(result, "rag_documents_indexed", 0), 0),
        "tokens_used": used,
        "tokens_remaining": remaining,
        "budget": budget,
        "platform_counts": platform_counts,
        "type_counts": type_counts,
        "difficulty_counts": difficulty_counts,
        "counted_total": counted,
        "expanded_queries": _as_list(_get(result, "expanded_queries", None)),
        "keywords": _as_list(_get(result, "keywords", None)),
        "quality_guards": dict(_get(result, "quality_guards", {}) or {}),
        "quality_gate_passed": bool(
            _get(result, "quality_gate_passed", True)
        ),
    }


def render_report(
    result: Any,
    kind: str = "pipeline",
    limit: int = 10,
    console: Console | None = None,
) -> None:
    con = console or _console()
    try:
        data = build_report_data(result, kind)
    except Exception:
        return

    for section in (
        _render_header,
        _render_executive,
        _render_funnel,
        _render_tokens,
        _render_sources,
        _render_summaries,
        _render_learning_path,
        _render_distribution,
        _render_diagnostics,
        _render_saved_files,
    ):
        try:
            section(con, data, limit)
        except Exception:
            continue


def _render_header(console: Console, data: dict[str, Any], limit: int) -> None:
    topic = str(data.get("topic") or data.get("goal") or "")
    label = f"REPORT · {fmt.ellipsis(topic, 90)}" if topic else "REPORT"
    console.print()
    console.print(
        Rule(
            Text(f"  {label}  ", style="athena.heading"),
            style="athena.border",
            characters="─",
        )
    )


def _render_executive(
    console: Console, data: dict[str, Any], limit: int
) -> None:
    status = str(data.get("status", "")).lower()
    border = {
        "success": "athena.success",
        "completed": "athena.success",
        "done": "athena.success",
        "partial": "athena.warning",
        "failed": "athena.danger",
        "error": "athena.danger",
    }.get(status, "athena.border")

    left_rows = [
        ("MODE", str(data.get("kind", "pipeline")).upper()),
        ("RUN ID", data.get("request_id") or "-"),
        ("GOAL", data.get("goal") or "-"),
        ("TOPIC", data.get("topic") or "-"),
        ("LEVEL", data.get("level") or data.get("detected_level") or "-"),
        ("DURATION", fmt.format_latency(data.get("latency_ms"))),
    ]
    right_rows = [
        ("SOURCES", str(data.get("source_count", 0))),
        ("RANKED", str(data.get("ranked_count", 0))),
        ("SUMMARIES", str(data.get("summary_count", 0))),
        ("STEPS", str(data.get("learning_step_count", 0))),
        ("ENHANCED", "yes" if data.get("enhanced_learning_path") else "no"),
        ("RAG DOCS", str(data.get("rag_docs", 0))),
        ("ERRORS", str(len(data.get("errors") or []))),
        ("WARNINGS", str(len(data.get("warnings") or []))),
    ]

    grid = Table.grid(padding=(0, 4), expand=True)
    grid.add_column(style="athena.label", justify="right", no_wrap=True)
    grid.add_column(style="athena.value", overflow="fold")
    grid.add_column(style="athena.label", justify="right", no_wrap=True)
    grid.add_column(style="athena.value", overflow="fold")

    rows = max(len(left_rows), len(right_rows))
    for index in range(rows):
        left_label, left_value = (
            left_rows[index] if index < len(left_rows) else ("", "")
        )
        right_label, right_value = (
            right_rows[index] if index < len(right_rows) else ("", "")
        )
        grid.add_row(
            Text(str(left_label).upper(), style="athena.label")
            if left_label
            else Text(""),
            Text(str(left_value), style="athena.value")
            if left_value
            else Text(""),
            Text(str(right_label).upper(), style="athena.label")
            if right_label
            else Text(""),
            Text(str(right_value), style="athena.value")
            if right_value
            else Text(""),
        )

    header = Text()
    header.append("STATUS  ", style="athena.label")
    header.append_text(comp.status_badge(status))

    body = Table.grid(padding=(0, 0), expand=True)
    body.add_column(overflow="fold")
    body.add_row(header)
    body.add_row(Text(""))
    body.add_row(grid)

    console.print(
        comp.panel(
            body,
            title="EXECUTIVE SUMMARY",
            border=border,
            padding=(1, 2),
        )
    )


def _funnel_hint(stages: list[tuple[str, int]]) -> str:
    retrieved = stages[0][1] if len(stages) > 0 else 0
    ranked = stages[1][1] if len(stages) > 1 else 0
    summarized = stages[2][1] if len(stages) > 2 else 0

    if retrieved > 0 and ranked < retrieved:
        dropped = retrieved - ranked
        return (
            f"Relevance and topic-presence gates dropped {dropped} of "
            f"{retrieved} retrieved sources before ranking."
        )
    if retrieved > 0 and ranked == retrieved and summarized < ranked:
        dropped = ranked - summarized
        return (
            f"The summarizer capped output at {summarized}; {dropped} "
            f"ranked sources were not summarized."
        )
    if retrieved > 0 and all(value == retrieved for _, value in stages):
        return (
            "No shrinkage in this run: every stage kept the same "
            "number of items."
        )
    return ""


def _render_funnel(
    console: Console, data: dict[str, Any], limit: int
) -> None:
    stages = [
        ("RETRIEVED", int(data.get("source_count", 0) or 0)),
        ("RANKED", int(data.get("ranked_count", 0) or 0)),
        ("SUMMARIZED", int(data.get("summary_count", 0) or 0)),
        ("PATH USED", int(data.get("path_resource_count", 0) or 0)),
    ]
    max_value = max((value for _, value in stages), default=0)
    if max_value <= 0:
        max_value = 1

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="athena.label", justify="right", no_wrap=True)
    grid.add_column(style="athena.metric", justify="right", no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_column(style="athena.muted", justify="right", no_wrap=True)

    for label, value in stages:
        share = (value / max_value) * 100.0 if max_value else 0.0
        grid.add_row(
            label,
            str(value),
            comp.mini_bar(value, max_value, width=20),
            f"{share:5.1f}%",
        )

    body = Table.grid(padding=(0, 0), expand=False)
    body.add_column(overflow="fold")
    body.add_row(grid)

    hint = _funnel_hint(stages)
    if hint:
        body.add_row(Text(""))
        body.add_row(Text(f"  {hint}", style="athena.muted"))

    console.print(
        comp.panel(
            body, title="RESEARCH FUNNEL", border="athena.border"
        )
    )


def _render_tokens(
    console: Console, data: dict[str, Any], limit: int
) -> None:
    budget = int(data.get("budget", 0) or 0)
    used = int(data.get("tokens_used", 0) or 0)
    if budget <= 0 and used <= 0:
        return
    remaining = int(data.get("tokens_remaining", 0) or 0)
    percent = (used / budget) * 100.0 if budget > 0 else 0.0

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="athena.label", justify="right", no_wrap=True)
    grid.add_column(style="athena.metric", justify="left", no_wrap=True)
    grid.add_row("BUDGET", str(budget))
    grid.add_row("USED", str(used))
    grid.add_row("REMAINING", str(remaining))
    grid.add_row("UTILIZATION", comp.bar(percent))

    console.print(
        comp.panel(grid, title="TOKEN ECONOMICS", border="athena.border")
    )


def _render_sources(
    console: Console, data: dict[str, Any], limit: int
) -> None:
    ranked = data.get("ranked_sources") or []
    sources = data.get("sources") or []

    if not ranked and not sources:
        console.print(comp.empty_panel("No sources were found.", "SOURCES"))
        return

    items = ranked if ranked else sources
    try:
        cap = max(1, int(limit))
    except (TypeError, ValueError):
        cap = 10

    rows: list[list[Any]] = []
    for index, item in enumerate(items[:cap], start=1):
        source = _get(item, "source", None) or item
        rank = _get(item, "rank", index) or index
        title = str(_get(source, "title", "") or "Untitled")
        url = str(_get(source, "url", "") or "")
        platform = _enum_value(_get(source, "platform", "")) or "-"
        source_type = _enum_value(_get(source, "source_type", "")) or "-"
        score = _get(item, "score", None)
        confidence = _get(item, "confidence", None)
        year = _get(source, "year", None) or "-"

        rows.append(
            [
                str(rank),
                comp.source_title(title, url, limit=110),
                platform,
                source_type,
                comp.score_badge(score),
                comp.confidence_badge(confidence),
                str(year),
            ]
        )

    table = comp.data_table(
        columns=(
            ("rank", "#"),
            ("title", "TITLE"),
            ("platform", "PLATFORM"),
            ("source_type", "TYPE"),
            ("score", "SCORE"),
            ("confidence", "CONF"),
            ("year", "YEAR"),
        ),
        rows=rows,
        title="RANKED SOURCES" if ranked else "SOURCES",
        expand=True,
    )
    console.print(table)


def _render_summaries(
    console: Console, data: dict[str, Any], limit: int
) -> None:
    summaries = data.get("summaries") or []
    if not summaries:
        return

    grid = Table.grid(padding=(0, 2), expand=True)
    grid.add_column(overflow="fold")

    for index, item in enumerate(summaries[:8], start=1):
        title = str(_get(item, "title", "") or "Untitled")
        difficulty = _enum_value(_get(item, "difficulty", "")) or "unknown"
        summary_text = str(_get(item, "summary", "") or "")
        why = str(_get(item, "why_useful", "") or "")
        key_topics = _as_list(_get(item, "key_topics", None))

        block = Table.grid(padding=(0, 2))
        block.add_column(overflow="fold")
        header = Text()
        header.append(f" {index}  ", style="athena.metric")
        header.append(title, style="athena.value")
        header.append("   ")
        header.append_text(comp.difficulty_badge(difficulty))
        block.add_row(header)
        if summary_text:
            block.add_row(
                Text(fmt.ellipsis(summary_text, 480), style="athena.text")
            )
        if why:
            block.add_row(Text(f"why: {why}", style="athena.muted"))
        if key_topics:
            topics_text = Text()
            topics_text.append("topics: ", style="athena.muted")
            topics_text.append(
                ", ".join(str(item) for item in key_topics[:8]),
                style="athena.text",
            )
            block.add_row(topics_text)
        if index < len(summaries[:8]):
            block.add_row(Text(""))
        grid.add_row(block)

    console.print(
        comp.panel(grid, title="SOURCE SUMMARIES", border="athena.border")
    )


def _render_learning_path(
    console: Console, data: dict[str, Any], limit: int
) -> None:
    path = data.get("learning_path")
    if path is None:
        return
    steps = _as_list(_get(path, "steps", None))
    if not steps:
        return

    enhanced = data.get("enhanced_learning_path")
    enhancement_map: dict[Any, Any] = {}
    if enhanced is not None:
        enhancements = _as_list(_get(enhanced, "enhancements", None))
        for entry in enhancements:
            step_num = _get(entry, "step", None)
            if step_num is not None:
                enhancement_map[step_num] = entry

    total_minutes = 0
    tree = Tree(
        Text("LEARNING PATH", style="athena.heading"),
        guide_style="athena.border",
    )

    for step in steps:
        number = _to_int(_get(step, "step", 0), 0)
        title = str(_get(step, "title", "") or "")
        objective = str(_get(step, "objective", "") or "")
        minutes = _to_int(_get(step, "estimated_minutes", 0), 0)
        total_minutes += minutes

        label = Text()
        label.append(f" STEP {number} ", style="bold black on #c084fc")
        label.append("   ")
        label.append(title, style="athena.value")
        if minutes:
            label.append(f"   {minutes} min", style="athena.muted")

        branch = tree.add(label)

        if objective and objective != title:
            branch.add(Text(objective, style="athena.text"))

        enhancement = enhancement_map.get(number)
        prerequisites = _as_list(_get(enhancement, "prerequisites", None))
        projects = _as_list(_get(enhancement, "suggested_projects", None))
        concepts = _as_list(_get(enhancement, "key_concepts", None))

        if prerequisites:
            node = branch.add(
                Text("prerequisites", style="athena.label")
            )
            for item in prerequisites[:8]:
                node.add(Text(str(item), style="athena.text"))

        if concepts:
            node = branch.add(Text("key concepts", style="athena.label"))
            for item in concepts[:8]:
                node.add(Text(str(item), style="athena.text"))

        if projects:
            node = branch.add(
                Text("suggested projects", style="athena.label")
            )
            for item in projects[:8]:
                node.add(Text(str(item), style="athena.text"))

        resources = _as_list(_get(step, "resources", None))
        if resources:
            node = branch.add(
                Text(
                    f"resources ({len(resources)})",
                    style="athena.label",
                )
            )
            for resource in resources[:10]:
                source = _get(resource, "source", None) or resource
                res_title = str(_get(source, "title", "") or "Untitled")
                res_url = str(_get(source, "url", "") or "")
                node.add(comp.source_title(res_title, res_url, limit=100))

    if total_minutes:
        tree.add(
            Text(
                f"total estimated time: {total_minutes} minutes",
                style="athena.muted",
            )
        )

    console.print(
        comp.panel(tree, title="LEARNING PATH", border="athena.border")
    )


def _render_distribution(
    console: Console, data: dict[str, Any], limit: int
) -> None:
    counts: Counter[str] = data.get("platform_counts") or Counter()
    if not isinstance(counts, Counter):
        try:
            counts = Counter(counts)
        except Exception:
            return
    if len(counts) < 2:
        return

    total = int(data.get("counted_total", 0) or 0)
    if total <= 0:
        total = sum(counts.values()) or 1

    max_count = max(counts.values() or [1])

    table = Table(
        box=None,
        show_header=False,
        padding=(0, 2),
        expand=False,
    )
    table.add_column(style="athena.value", no_wrap=True)
    table.add_column(style="athena.metric", justify="right", no_wrap=True)
    table.add_column(no_wrap=True)
    table.add_column(style="athena.muted", justify="right", no_wrap=True)

    for name, count in counts.most_common(12):
        share = (count / total) * 100.0
        table.add_row(
            str(name)[:24],
            str(count),
            comp.mini_bar(count, max_count, width=18),
            f"{share:5.1f}%",
        )

    console.print(
        comp.panel(
            table, title="PLATFORM DISTRIBUTION", border="athena.border"
        )
    )


def _render_diagnostics(
    console: Console, data: dict[str, Any], limit: int
) -> None:
    errors = data.get("errors") or []
    warnings = data.get("warnings") or []

    if not errors and not warnings:
        console.print(
            comp.panel(
                Text(
                    "No diagnostic issues detected.", style="athena.text"
                ),
                title="DIAGNOSTICS",
                border="athena.success",
            )
        )
        return

    body = Text()
    for index, item in enumerate(errors[:10]):
        if index:
            body.append("\n")
        body.append("  ERROR  ", style="athena.danger")
        body.append(fmt.ellipsis(str(item), 220), style="athena.text")
    for index, item in enumerate(warnings[:10]):
        if errors or index:
            body.append("\n")
        body.append("  WARN   ", style="athena.warning")
        body.append(fmt.ellipsis(str(item), 220), style="athena.text")

    border = "athena.danger" if errors else "athena.warning"
    console.print(comp.panel(body, title="DIAGNOSTICS", border=border))


def _render_saved_files(
    console: Console, data: dict[str, Any], limit: int
) -> None:
    saved = data.get("saved_files") or {}
    if not saved:
        return

    rows: list[list[Any]] = []
    for key, value in saved.items():
        path = Path(str(value))
        size = "-"
        modified = "-"
        if path.exists():
            try:
                stat = path.stat()
                size = fmt.format_bytes(stat.st_size)
                from datetime import datetime
                modified = datetime.fromtimestamp(stat.st_mtime).strftime(
                    "%Y-%m-%d %H:%M"
                )
            except Exception:
                pass
        rows.append([str(key).upper(), str(value), size, modified])

    table = comp.data_table(
        columns=(
            ("kind", "TYPE"),
            ("path", "PATH"),
            ("size", "SIZE"),
            ("modified", "MODIFIED"),
        ),
        rows=rows,
        title="SAVED FILES",
        expand=True,
    )
    console.print(table)
    console.print()