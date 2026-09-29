from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.text import Text

from cli import banner as banner_module
from cli import components as comp
from cli import errors as errors_module
from cli import format as fmt
from cli import lazy
from cli import theme as theme_module


def _console() -> Console:
    return theme_module.get_console()


runs_app = typer.Typer(help="Inspect saved pipeline runs.")


def _load_store() -> Any:
    store = lazy.get_sqlite_store()
    return store


async def _list_runs(limit: int) -> list[dict[str, Any]]:
    store = _load_store()
    await store.initialize()
    return await store.list_runs(limit)


async def _get_run(request_id: str) -> dict[str, Any] | None:
    store = _load_store()
    await store.initialize()
    return await store.get_run(request_id)


@runs_app.command("list", help="List saved runs.")
def list_runs(
    limit: int = typer.Option(20, "--limit", "-n", min=1, max=500),
) -> None:
    console = _console()
    banner_module.print_title("runs", subtitle="list")

    try:
        rows = asyncio.run(_list_runs(limit))
    except Exception as exc:
        errors_module.print_error(exc)
        raise typer.Exit(code=1)

    if not rows:
        console.print(comp.empty_panel("No saved runs yet.", "RUNS"))
        banner_module.print_closing("Nothing to show.")
        return

    table_rows: list[list[Any]] = []
    for item in rows:
        request_id = str(item.get("request_id", "") or "")
        topic = str(item.get("topic", "") or "untitled")
        status = str(item.get("status", "") or "unknown")
        sources = _to_int(item.get("total_sources"), 0)
        ranked = _to_int(item.get("total_ranked"), 0)
        steps = _to_int(item.get("total_steps"), 0)
        latency = fmt.format_latency(item.get("latency_ms"))
        created = fmt.format_short_timestamp(
            item.get("created_at") or item.get("finished_at")
        )
        table_rows.append(
            [
                Text(request_id[:12], style="athena.muted"),
                Text(fmt.ellipsis(topic, 60), style="athena.value"),
                comp.status_badge(status),
                str(sources),
                str(ranked),
                str(steps),
                latency,
                Text(created, style="athena.muted"),
            ]
        )

    console.print(
        comp.data_table(
            columns=(
                ("id", "REQUEST ID"),
                ("topic", "TOPIC"),
                ("status", "STATUS"),
                ("sources", "SRC"),
                ("ranked", "RANK"),
                ("steps", "STEPS"),
                ("latency", "LATENCY"),
                ("created", "CREATED AT"),
            ),
            rows=table_rows,
            title="SAVED RUNS",
            expand=True,
        )
    )
    banner_module.print_closing(
        "Runs listed.", subtitle=f"{len(table_rows)} rows"
    )


@runs_app.command("show", help="Show one saved run in detail.")
def show_run(
    request_id: str = typer.Argument(..., help="Request ID of the run."),
) -> None:
    console = _console()
    banner_module.print_title("runs", subtitle=f"show {fmt.ellipsis(request_id, 20)}")

    try:
        run = asyncio.run(_get_run(request_id))
    except Exception as exc:
        errors_module.print_error(exc)
        raise typer.Exit(code=1)

    if run is None:
        console.print(
            comp.warning_panel(
                f"Run '{request_id}' was not found.", title="NOT FOUND"
            )
        )
        raise typer.Exit(code=1)

    status = str(run.get("status", "") or "unknown")
    query = _extract_query(run)
    topic = str(run.get("topic") or query.get("topic") or "-")
    goal = str(run.get("goal") or query.get("goal") or "-")
    level = str(run.get("level") or query.get("level") or "-")

    summary_rows = [
        ("Run ID", request_id),
        ("Status", status.upper()),
        ("Topic", topic),
        ("Goal", goal),
        ("Level", level),
        ("Sources", str(_to_int(run.get("total_sources"), 0))),
        ("Ranked", str(_to_int(run.get("total_ranked"), 0))),
        ("Steps", str(_to_int(run.get("total_steps"), 0))),
        ("Latency", fmt.format_latency(run.get("latency_ms"))),
        ("Created", fmt.format_short_timestamp(run.get("created_at"))),
        ("Finished", fmt.format_short_timestamp(run.get("finished_at"))),
    ]

    border = {
        "success": "athena.success",
        "completed": "athena.success",
        "done": "athena.success",
        "partial": "athena.warning",
        "failed": "athena.danger",
        "error": "athena.danger",
    }.get(status.lower(), "athena.border")

    console.print(
        comp.panel(
            comp.kv_grid(summary_rows),
            title="RUN SUMMARY",
            border=border,
            padding=(1, 2),
        )
    )

    sources = list(run.get("sources", []) or [])
    if sources:
        source_rows: list[list[Any]] = []
        for item in sources[:50]:
            rank = _to_int(item.get("rank"), 0)
            title = str(item.get("title") or "Untitled")
            url = str(item.get("url") or "")
            platform = str(item.get("platform") or "-")
            source_type = str(item.get("source_type") or "-")
            difficulty = str(item.get("difficulty") or "unknown")
            score = item.get("score")
            confidence = item.get("confidence")
            source_rows.append(
                [
                    str(rank),
                    comp.source_title(title, url, limit=90),
                    platform,
                    source_type,
                    comp.difficulty_badge(difficulty),
                    comp.score_badge(score),
                    comp.confidence_badge(confidence),
                ]
            )
        console.print(
            comp.data_table(
                columns=(
                    ("rank", "#"),
                    ("title", "TITLE"),
                    ("platform", "PLATFORM"),
                    ("source_type", "TYPE"),
                    ("difficulty", "DIFF"),
                    ("score", "SCORE"),
                    ("confidence", "CONF"),
                ),
                rows=source_rows,
                title="SAVED SOURCES",
                expand=True,
            )
        )

    steps = list(run.get("learning_steps", []) or [])
    if steps:
        step_rows: list[list[Any]] = []
        for step in steps:
            step_rows.append(
                [
                    str(_to_int(step.get("step"), 0)),
                    str(step.get("title") or ""),
                    fmt.ellipsis(str(step.get("objective") or ""), 100),
                    str(_to_int(step.get("estimated_minutes"), 0) or "-"),
                ]
            )
        console.print(
            comp.data_table(
                columns=(
                    ("step", "STEP"),
                    ("title", "TITLE"),
                    ("objective", "OBJECTIVE"),
                    ("minutes", "MIN"),
                ),
                rows=step_rows,
                title="LEARNING STEPS",
                expand=True,
            )
        )

    banner_module.print_closing("Run shown.")


def _extract_query(run: dict[str, Any]) -> dict[str, Any]:
    payload = run.get("payload")
    if isinstance(payload, dict):
        query = payload.get("query")
        if isinstance(query, dict):
            return query
    return {}


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def register(app: typer.Typer) -> None:
    app.add_typer(runs_app, name="runs")