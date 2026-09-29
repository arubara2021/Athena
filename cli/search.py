from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from typing import Any

import typer
from rich.console import Console

from cli import banner as banner_module
from cli import components as comp
from cli import errors as errors_module
from cli import format as fmt
from cli import lazy
from cli import progress as progress_module
from cli import render as render_module
from cli import theme as theme_module


def _console() -> Console:
    return theme_module.get_console()


def _defaults() -> dict[str, Any]:
    return {
        "goal": "Learn from basics to advanced",
        "level": "beginner to advanced",
        "max_results": 20,
        "display_limit": 10,
        "mode": "balanced",
        "use_llm_expansion": False,
        "use_llm_ranking": True,
        "use_llm_learning_path": False,
        "enable_rag": False,
    }


def _resolve_mode(value: str) -> str:
    text = str(value or "").strip().lower()
    constants = None
    try:
        constants = lazy.get_constants()
    except Exception:
        constants = None
    valid = set(getattr(constants, "VALID_MODES", ()) or ())
    default = str(getattr(constants, "DEFAULT_MODE", "balanced") or "balanced")
    if text in valid:
        return text
    return default


def _build_query(
    topic: str,
    goal: str,
    level: str,
    max_results: int,
    platforms: list[str] | None,
    source_types: list[str] | None,
) -> Any:
    schema_cls = lazy.get_search_query_schema()
    payload: dict[str, Any] = {
        "topic": topic,
        "goal": goal,
        "level": level,
        "max_results": max_results,
    }
    if platforms:
        enum_cls = lazy.get_source_platform_enum()
        values = []
        for item in platforms:
            try:
                values.append(enum_cls(str(item).strip().lower()).value)
            except Exception:
                continue
        if values:
            payload["platforms"] = values
    if source_types:
        enum_cls = lazy.get_source_type_enum()
        values = []
        for item in source_types:
            try:
                values.append(enum_cls(str(item).strip().lower()).value)
            except Exception:
                continue
        if values:
            payload["source_types"] = values
    try:
        return schema_cls.model_validate(payload)
    except Exception as exc:
        raise typer.BadParameter(str(exc))


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


async def _execute_pipeline(
    query: Any,
    use_llm_expansion: bool,
    use_llm_ranking: bool,
    use_llm_learning_path: bool,
    enable_rag: bool,
    mode: str,
    show_progress: bool,
    title: str,
    target_sources: int,
) -> tuple[Any, Any]:
    pipeline = lazy.get_pipeline(
        use_llm_expansion=use_llm_expansion,
        use_llm_ranking=use_llm_ranking,
        use_llm_learning_path=use_llm_learning_path,
        enable_rag=enable_rag,
    )

    async with pipeline:
        if not show_progress:
            result = await pipeline.run(query, mode=mode)
            return result, None

        dashboard = progress_module.ProgressDashboard(
            mode="pipeline",
            title=title,
            target_sources=target_sources,
        )
        subscribe = getattr(pipeline, "subscribe", None)
        if callable(subscribe):
            try:
                subscribe("*", dashboard.handle_event)
            except Exception:
                pass

        live = progress_module.build_live(dashboard)
        try:
            with live:
                result = await pipeline.run(query, mode=mode)
                dashboard.absorb_result(result)
        except Exception:
            try:
                dashboard.absorb_result(result)
            except Exception:
                pass
            raise
        return result, dashboard


def _open_path(path: str) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        pass


def _print_minimal(result: Any) -> None:
    console = _console()
    request_id = str(getattr(result, "request_id", "") or "")
    status = _enum_value(getattr(result, "status", "")) or "unknown"
    console.print(f"{request_id}  {status}")
    saved = dict(getattr(result, "saved_files", {}) or {})
    for key, value in saved.items():
        console.print(f"{key}: {value}")


def _run_search(
    topic: str,
    goal: str,
    level: str,
    max_results: int,
    platforms: list[str] | None,
    source_types: list[str] | None,
    use_llm_expansion: bool,
    use_llm_ranking: bool,
    use_llm_learning_path: bool,
    enable_rag: bool,
    display_limit: int,
    mode: str,
    json_output: bool,
    markdown_output: bool,
    quiet: bool,
    open_output: bool,
) -> int:
    console = _console()
    banner_text = fmt.ellipsis(topic, 60)
    banner_module.print_title("search", subtitle=banner_text)

    query = _build_query(
        topic=topic,
        goal=goal,
        level=level,
        max_results=max_results,
        platforms=platforms,
        source_types=source_types,
    )

    show_progress = not (json_output or markdown_output or quiet)

    try:
        result, _ = asyncio.run(
            _execute_pipeline(
                query=query,
                use_llm_expansion=use_llm_expansion,
                use_llm_ranking=use_llm_ranking,
                use_llm_learning_path=use_llm_learning_path,
                enable_rag=enable_rag,
                mode=mode,
                show_progress=show_progress,
                title=banner_text,
                target_sources=max_results,
            )
        )
    except KeyboardInterrupt:
        console.print(comp.warning_panel("Run cancelled.", title="CANCELLED"))
        return 1
    except Exception as exc:
        errors_module.print_error(exc)
        return 1

    if json_output:
        try:
            payload = result.model_dump(mode="json")
        except Exception:
            payload = _to_dict(result)
        typer.echo(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        )
    elif markdown_output:
        try:
            writer = lazy.get_markdown_writer()
            typer.echo(writer.render_state(result))
        except Exception as exc:
            errors_module.print_error(exc)
            return 1
    elif quiet:
        _print_minimal(result)
    else:
        render_module.render_report(
            result, kind="pipeline", limit=display_limit, console=console
        )

    if open_output:
        saved = dict(getattr(result, "saved_files", {}) or {})
        markdown_path = saved.get("markdown")
        if markdown_path and os.path.exists(markdown_path):
            _open_path(markdown_path)

    status = _enum_value(getattr(result, "status", "")).lower()
    if status == "failed":
        return 1
    return 0


def _to_dict(result: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for attr in (
        "request_id",
        "status",
        "topic",
        "goal",
        "level",
        "corrected_topic",
        "primary_concept",
        "keywords",
        "expanded_queries",
        "sources",
        "ranked_sources",
        "source_summaries",
        "learning_path",
        "errors",
        "warnings",
        "saved_files",
        "latency_ms",
    ):
        try:
            value = getattr(result, attr, None)
        except Exception:
            value = None
        if value is None:
            payload[attr] = None
        elif isinstance(value, (list, dict, str, int, float, bool)):
            payload[attr] = value
        else:
            value_str = _enum_value(value)
            payload[attr] = value_str or str(value)
    return payload


def search(
    topic: str = typer.Argument(..., help="Topic to research."),
    goal: str = typer.Option("", "--goal", "-g", help="Learning goal."),
    level: str = typer.Option("", "--level", "-l", help="Target learner level."),
    max_results: int = typer.Option(
        20, "--max", "-m", min=1, max=100, help="Maximum final sources."
    ),
    platform: list[str] = typer.Option(None, "--platform", help="Platform filter."),
    source_type: list[str] = typer.Option(
        None, "--source-type", help="Source type filter."
    ),
    use_llm_expansion: bool = typer.Option(
        False, "--llm-expansion/--no-llm-expansion", help="LLM query expansion."
    ),
    use_llm_ranking: bool = typer.Option(
        True, "--llm-ranking/--no-llm-ranking", help="LLM consensus ranking."
    ),
    use_llm_learning_path: bool = typer.Option(
        False, "--llm-path/--no-llm-path", help="LLM learning path generation."
    ),
    enable_rag: bool = typer.Option(
        False, "--rag/--no-rag", help="Index results for chat."
    ),
    display_limit: int = typer.Option(
        10, "--display", "-d", min=1, max=100, help="Rows in the final report."
    ),
    mode: str = typer.Option("", "--mode", help="fast, balanced, deep."),
    json_output: bool = typer.Option(False, "--json/--no-json"),
    markdown_output: bool = typer.Option(False, "--markdown/--no-markdown"),
    quiet: bool = typer.Option(False, "--quiet/--no-quiet"),
    open_output: bool = typer.Option(False, "--open/--no-open"),
) -> None:
    defaults = _defaults()
    resolved_goal = goal or defaults["goal"]
    resolved_level = level or defaults["level"]
    resolved_mode = _resolve_mode(mode or defaults["mode"])

    code = _run_search(
        topic=topic,
        goal=resolved_goal,
        level=resolved_level,
        max_results=max_results,
        platforms=platform or None,
        source_types=source_type or None,
        use_llm_expansion=use_llm_expansion,
        use_llm_ranking=use_llm_ranking,
        use_llm_learning_path=use_llm_learning_path,
        enable_rag=enable_rag,
        display_limit=display_limit,
        mode=resolved_mode,
        json_output=json_output,
        markdown_output=markdown_output,
        quiet=quiet,
        open_output=open_output,
    )
    raise typer.Exit(code=code)


def ask(
    topic: str = typer.Argument(..., help="Topic to research."),
    mode: str = typer.Option("", "--mode", help="fast, balanced, deep."),
    quiet: bool = typer.Option(False, "--quiet/--no-quiet"),
    json_output: bool = typer.Option(False, "--json/--no-json"),
    markdown_output: bool = typer.Option(False, "--markdown/--no-markdown"),
) -> None:
    defaults = _defaults()
    resolved_mode = _resolve_mode(mode or defaults["mode"])

    code = _run_search(
        topic=topic,
        goal=defaults["goal"],
        level=defaults["level"],
        max_results=defaults["max_results"],
        platforms=None,
        source_types=None,
        use_llm_expansion=defaults["use_llm_expansion"],
        use_llm_ranking=defaults["use_llm_ranking"],
        use_llm_learning_path=defaults["use_llm_learning_path"],
        enable_rag=defaults["enable_rag"],
        display_limit=defaults["display_limit"],
        mode=resolved_mode,
        json_output=json_output,
        markdown_output=markdown_output,
        quiet=quiet,
        open_output=False,
    )
    raise typer.Exit(code=code)


def register(app: typer.Typer) -> None:
    app.command(name="search", help="Run the full research pipeline.")(search)
    app.command(name="ask", help="Run the pipeline with default settings.")(ask)