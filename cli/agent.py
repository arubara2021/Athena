from __future__ import annotations

import asyncio
import json
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


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


async def _execute_agent(
    goal: str,
    topic: str,
    level: str,
    max_results: int,
    budget: int,
    max_iterations: int,
    mode: str,
    show_progress: bool,
    title: str,
) -> tuple[Any, Any]:
    config_cls = lazy.get_agent_config_class()
    config = None
    try:
        config = config_cls(
            max_iterations=max_iterations,
            token_budget=budget,
            memory_enabled=True,
            reflection_enabled=True,
            mode=mode,
        )
    except Exception:
        try:
            config = config_cls()
        except Exception:
            config = None

    agent_obj = lazy.get_agent(agent_config=config) if config is not None else lazy.get_agent()

    async with agent_obj:
        if not show_progress:
            result = await agent_obj.run(
                goal=goal,
                level=level,
                topic=topic,
                max_results=max_results,
                mode=mode,
            )
            return result, None

        dashboard = progress_module.ProgressDashboard(
            mode="agent",
            title=title,
            goal=goal,
            target_sources=max_results,
        )
        subscribe = getattr(agent_obj, "subscribe", None)
        if callable(subscribe):
            try:
                subscribe("*", dashboard.handle_event)
            except Exception:
                pass

        live = progress_module.build_live(dashboard)
        try:
            with live:
                result = await agent_obj.run(
                    goal=goal,
                    level=level,
                    topic=topic,
                    max_results=max_results,
                    mode=mode,
                )
                dashboard.absorb_result(result)
        except Exception:
            try:
                dashboard.absorb_result(result)
            except Exception:
                pass
            raise
        return result, dashboard


def _print_minimal(result: Any) -> None:
    console = _console()
    agent_id = str(getattr(result, "agent_id", "") or "")
    status = _enum_value(getattr(result, "status", "")) or "unknown"
    iterations = getattr(result, "total_iterations", 0)
    tokens_used = getattr(result, "total_tokens_used", 0)
    tokens_budget = getattr(result, "total_budget", 0)
    console.print(f"{agent_id}  {status}")
    console.print(f"iterations: {iterations}")
    console.print(f"tokens: {tokens_used}/{tokens_budget}")
    saved = dict(getattr(result, "saved_files", {}) or {})
    for key, value in saved.items():
        console.print(f"{key}: {value}")


def _to_dict(result: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for attr in (
        "agent_id",
        "status",
        "goal",
        "topic",
        "level",
        "total_iterations",
        "total_tokens_used",
        "total_budget",
        "tokens_remaining",
        "ranked_sources",
        "learning_path",
        "summaries",
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
            payload[attr] = _enum_value(value) or str(value)
    return payload


def agent(
    goal: str = typer.Argument(..., help="Research goal for the agent."),
    topic: str = typer.Option("", "--topic", "-t", help="Topic override."),
    level: str = typer.Option(
        "beginner to advanced", "--level", "-l", help="Target learner level."
    ),
    max_results: int = typer.Option(
        20, "--max", "-m", min=1, max=100, help="Maximum sources."
    ),
    budget: int = typer.Option(
        40000, "--budget", "-b", min=1000, max=500000, help="Token budget."
    ),
    max_iterations: int = typer.Option(
        15, "--iterations", "-i", min=1, max=50, help="Iteration ceiling."
    ),
    display_limit: int = typer.Option(
        10, "--display", "-d", min=1, max=100, help="Rows in the final report."
    ),
    mode: str = typer.Option("", "--mode", help="fast, balanced, deep."),
    json_output: bool = typer.Option(False, "--json/--no-json"),
    quiet: bool = typer.Option(False, "--quiet/--no-quiet"),
) -> None:
    console = _console()
    resolved_topic = topic or goal
    resolved_mode = _resolve_mode(mode)
    banner_text = fmt.ellipsis(goal, 60)

    banner_module.print_title("agent", subtitle=banner_text)

    show_progress = not (json_output or quiet)

    try:
        result, _ = asyncio.run(
            _execute_agent(
                goal=goal,
                topic=resolved_topic,
                level=level,
                max_results=max_results,
                budget=budget,
                max_iterations=max_iterations,
                mode=resolved_mode,
                show_progress=show_progress,
                title=banner_text,
            )
        )
    except KeyboardInterrupt:
        console.print(comp.warning_panel("Agent cancelled.", title="CANCELLED"))
        raise typer.Exit(code=1)
    except Exception as exc:
        errors_module.print_error(exc)
        raise typer.Exit(code=1)

    if json_output:
        try:
            payload = result.model_dump(mode="json")
        except Exception:
            payload = _to_dict(result)
        typer.echo(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        )
    elif quiet:
        _print_minimal(result)
    else:
        render_module.render_report(
            result, kind="agent", limit=display_limit, console=console
        )

    status = _enum_value(getattr(result, "status", "")).lower()
    if status in ("failed", "error"):
        raise typer.Exit(code=1)


def register(app: typer.Typer) -> None:
    app.command(name="agent", help="Run the autonomous agent.")(agent)