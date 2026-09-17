from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

import typer
from rich import box
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
from rich.columns import Columns

from core import constants
from core.config import get_data_directory, get_output_directory, get_settings
from core.events import (
    LEARNING_PATH_COMPLETED,
    PATH_ENHANCED,
    PIPELINE_COMPLETED,
    PIPELINE_FAILED,
    PIPELINE_STARTED,
    RAG_INDEXED,
    RAG_INDEXING_STARTED,
    RANKING_COMPLETED,
    SEARCH_COMPLETED,
    SOURCE_SUMMARIES_COMPLETED,
    STAGE_FAILED,
)
from core.pipeline import PipelineResult, ResearchPipeline
from core.models import SourcePlatform, SourceType
from ranking.consensus_ranker import ConsensusRanker
from ranking.learning_path_builder import LearningPathBuilder
from core.schemas import SearchQuerySchema
from search.orchestrator import SearchOrchestrator
from storage.file_manager import FileManager
from storage.markdown_writer import MarkdownWriter
from storage.sqlite_store import SQLiteStore
from utils.logger import setup_logger

app = typer.Typer(
    help="Autonomous Research Agent CLI",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
runs_app = typer.Typer(help="Inspect saved pipeline runs")
app.add_typer(runs_app, name="runs")

console = Console(highlight=False)

STYLE_LABEL = "bold cyan"
STYLE_VALUE = "bold white"
STYLE_ACCENT = "bold magenta"
STYLE_OK = "bold green"
STYLE_WARN = "bold yellow"
STYLE_ERR = "bold red"
STYLE_DIM = "dim"
STYLE_HEADER = "bold white on rgb(0,51,102)"
STYLE_STEP = "bold white on magenta"
STYLE_AGENT = "bold bright_cyan"

_APP_TITLE = "RESEARCH AGENT"
_APP_SUBTITLE = "Autonomous Research  ·  AI Ranking  ·  Learning Paths"

_CLI_DEFAULTS_CACHE: dict[str, Any] | None = None

def _load_cli_defaults() -> dict[str, Any]:
    global _CLI_DEFAULTS_CACHE
    if _CLI_DEFAULTS_CACHE is not None:
        return _CLI_DEFAULTS_CACHE

    defaults: dict[str, Any] = {
        "goal": "Learn from basics to advanced",
        "level": "beginner to advanced",
        "max_results": 20,
        "display_limit": 10,
        "use_llm_expansion": False,
        "use_llm_ranking": True,
        "use_llm_learning_path": False,
        "enable_rag": False,
    }

    try:
        import yaml
        config_path = Path(__file__).resolve().parent / "configs" / "settings.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            if isinstance(loaded, dict):
                section = loaded.get("cli", {})
                if isinstance(section, dict):
                    for key in defaults:
                        if section.get(key) is not None:
                            defaults[key] = section[key]
                search_section = loaded.get("search", {})
                if (
                    isinstance(search_section, dict)
                    and search_section.get("default_max_results") is not None
                    and not (
                        isinstance(section, dict)
                        and section.get("max_results") is not None
                    )
                ):
                    defaults["max_results"] = search_section["default_max_results"]
    except Exception:
        pass

    try:
        defaults["max_results"] = max(1, min(100, int(defaults["max_results"])))
    except Exception:
        defaults["max_results"] = 20
    try:
        defaults["display_limit"] = max(1, min(100, int(defaults["display_limit"])))
    except Exception:
        defaults["display_limit"] = 10

    defaults["use_llm_expansion"] = bool(defaults["use_llm_expansion"])
    defaults["use_llm_ranking"] = bool(defaults["use_llm_ranking"])
    defaults["use_llm_learning_path"] = bool(defaults["use_llm_learning_path"])
    defaults["enable_rag"] = bool(defaults["enable_rag"])

    _CLI_DEFAULTS_CACHE = defaults
    return defaults

_DEFAULTS = _load_cli_defaults()

@app.callback()
def root(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    )
) -> None:
    setup_logger(level=logging.DEBUG if verbose else logging.INFO)

def _print_banner() -> None:
    banner_art = Text.assemble(
        (r"  ____  _____ ____  _   _ ____  _____ ____  _   _   _   _ _____ ____  _ " , "bold bright_cyan"),
        (r" |  _ \| ____/ ___|| | | / ___|| ____|  _ \| | | | / \ |_   _|  _ \| |", "bold bright_cyan"),
        (r" | |_) |  _| \___ \| | | \___ \|  _| | |_) | | | |/ _ \  | | | |_) | |", "bold bright_cyan"),
        (r" |  _ <| |___ ___) | |_| |___) | |___|  _ <| |_| / ___ \ | | |  _ <| |___", "bold bright_cyan"),
        (r" |_| \_\_____|____/ \___/|____/|_____|_| \_\\___/_/   \_\___|_| \_\_____|", "bold bright_cyan"),
    )

    subtitle = Table.grid(padding=(0, 2))
    subtitle.add_column(justify="right", style="bold magenta")
    subtitle.add_column(style="bold white")
    subtitle.add_row("VERSION", Text(constants.APP_VERSION))
    subtitle.add_row("MODE", Text("AUTONOMOUS RESEARCH"))
    subtitle.add_row("EXIT", Text("type 'exit' at any prompt"))

    console.print()
    console.print(
        Panel(
            Group(
                banner_art,
                Text("") ,
                Text(_APP_SUBTITLE, style="bold magenta"),
                subtitle,
            ),
            box=box.HEAVY,
            border_style="bold bright_cyan",
            padding=(1, 2),
        )
    )

@app.command()
def search(
    topic: str = typer.Argument(..., help="Topic to research"),
    goal: str = typer.Option(_DEFAULTS["goal"], help="Learning goal"),
    level: str = typer.Option(_DEFAULTS["level"], help="Target learner level"),
    max_results: int = typer.Option(
        _DEFAULTS["max_results"],
        min=1,
        max=100,
        help="Maximum number of final sources",
    ),
    platforms: Optional[List[str]] = typer.Option(
        None,
        "--platform",
        help="Search platforms: arxiv, semantic_scholar, openalex, github, wikipedia, huggingface, web",
    ),
    source_types: Optional[List[str]] = typer.Option(
        None,
        "--source-type",
        help="Source types: research_paper, repository, course, video, blog, documentation, dataset, model, other",
    ),
    use_llm_expansion: bool = typer.Option(
        _DEFAULTS["use_llm_expansion"],
        "--llm-expansion/--no-llm-expansion",
        help="Use LLM query expansion",
    ),
    use_llm_ranking: bool = typer.Option(
        _DEFAULTS["use_llm_ranking"],
        "--llm-ranking/--no-llm-ranking",
        help="Use multi-model consensus ranking",
    ),
    use_llm_learning_path: bool = typer.Option(
        _DEFAULTS["use_llm_learning_path"],
        "--llm-learning-path/--no-llm-learning-path",
        help="Use LLM learning path generation",
    ),
    enable_rag: bool = typer.Option(
        _DEFAULTS["enable_rag"],
        "--rag/--no-rag",
        help="Enable RAG indexing for chat",
    ),
    display_limit: int = typer.Option(
        _DEFAULTS["display_limit"],
        "--limit",
        min=1,
        max=100,
        help="Number of ranked sources to display",
    ),
    json_output: bool = typer.Option(
        False,
        "--json/--no-json",
        help="Print raw JSON output",
    ),
    markdown_output: bool = typer.Option(
        False,
        "--markdown/--no-markdown",
        help="Print Markdown output",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet/--no-quiet",
        help="Minimal output",
    ),
    open_output: bool = typer.Option(
        False,
        "--open/--no-open",
        help="Open saved Markdown report after completion",
    ),
) -> None:
    if json_output or markdown_output:
        setup_logger(level=logging.CRITICAL)

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
        result = asyncio.run(
            _execute_pipeline(
                query=query,
                use_llm_expansion=use_llm_expansion,
                use_llm_ranking=use_llm_ranking,
                use_llm_learning_path=use_llm_learning_path,
                enable_rag=enable_rag,
                show_progress=show_progress,
            )
        )
    except KeyboardInterrupt:
        console.print(Panel("[bold yellow]INTERRUPTED[/bold yellow]", border_style="yellow"))
        raise typer.Exit(code=1)
    except typer.BadParameter:
        raise
    except Exception as exc:
        console.print(
            Panel(
                escape(str(exc)),
                title="[bold red]ERROR[/bold red]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    if json_output:
        typer.echo(
            json.dumps(
                result.model_dump(mode="json"),
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
    elif markdown_output:
        markdown_writer = MarkdownWriter(FileManager(get_output_directory()))
        typer.echo(markdown_writer.render_state(result))
    elif quiet:
        _render_minimal(result)
    else:
        _render_result(result, display_limit)

    if open_output:
        markdown_path = result.saved_files.get("markdown")
        if markdown_path and os.path.exists(markdown_path):
            _open_path(markdown_path)

    if _enum_value(result.status) == "failed":
        raise typer.Exit(code=1)

@app.command()
def agent(
    goal: str = typer.Argument(..., help="Research goal for the autonomous agent"),
    topic: str = typer.Option("", help="Topic override (defaults to goal)"),
    level: str = typer.Option(_DEFAULTS["level"], help="Target learner level"),
    max_results: int = typer.Option(
        _DEFAULTS["max_results"],
        min=1,
        max=100,
        help="Maximum number of sources",
    ),
    budget: int = typer.Option(
        40000,
        min=1000,
        max=500000,
        help="Token budget for the agent run",
    ),
    max_iterations: int = typer.Option(
        15,
        min=1,
        max=50,
        help="Maximum loop iterations",
    ),
    display_limit: int = typer.Option(
        _DEFAULTS["display_limit"],
        "--limit",
        min=1,
        max=100,
        help="Number of results to display",
    ),
    json_output: bool = typer.Option(
        False,
        "--json/--no-json",
        help="Print raw JSON output",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet/--no-quiet",
        help="Minimal output",
    ),
) -> None:

    if json_output:
        setup_logger(level=logging.CRITICAL)

    resolved_topic = topic or goal

    try:
        result = asyncio.run(
            _execute_agent(
                goal=goal,
                topic=resolved_topic,
                level=level,
                max_results=max_results,
                budget=budget,
                max_iterations=max_iterations,
                show_progress=not (json_output or quiet),
            )
        )
    except KeyboardInterrupt:
        console.print(Panel("[bold yellow]AGENT INTERRUPTED[/bold yellow]", border_style="yellow"))
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(
            Panel(
                escape(str(exc)),
                title="[bold red]AGENT ERROR[/bold red]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    if json_output:
        typer.echo(
            json.dumps(
                result.model_dump(mode="json"),
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
    elif quiet:
        _render_agent_minimal(result)
    else:
        _render_agent_result(result, display_limit)

    if not result.success:
        raise typer.Exit(code=1)

async def _execute_agent(
    goal: str,
    topic: str,
    level: str,
    max_results: int,
    budget: int,
    max_iterations: int,
    show_progress: bool,
) -> Any:
    from agent.controller import AutonomousAgent, AgentResult, AgentConfig

    config = AgentConfig(
        max_iterations=max_iterations,
        token_budget=budget,
        memory_enabled=True,
        reflection_enabled=True,
    )

    agent_instance = AutonomousAgent(agent_config=config)

    async with agent_instance:
        if not show_progress:
            return await agent_instance.run(
                goal=goal,
                level=level,
                topic=topic,
                max_results=max_results,
            )

        progress = Progress(
            SpinnerColumn(style="bold cyan"),
            TextColumn("[bold cyan]{task.description}[/bold cyan]"),
            BarColumn(bar_width=30, style="cyan", complete_style="green"),
            TextColumn("[bold white]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )

        with progress:
            task = progress.add_task(
                "[bold white]Initializing autonomous agent[/bold white]",
                total=100,
            )

            async def handle_agent_event(event: Any) -> None:
                payload = event.payload or {}
                name = event.name

                if name == "agent.started":
                    progress.update(
                        task,
                        description="[bold cyan]>> Agent started - planning research strategy[/bold cyan]",
                        completed=5,
                    )
                elif name == "agent.memory_recalled":
                    ctx_len = payload.get("context_length", 0)
                    progress.update(
                        task,
                        description=f"[bold cyan]:: Memory recalled ({ctx_len} chars)[/bold cyan]",
                        completed=10,
                    )
                elif name == "agent.plan_created":
                    progress.update(
                        task,
                        description="[bold cyan]>> Plan created - executing research steps[/bold cyan]",
                        completed=15,
                    )
                elif name == "agent.loop_started":
                    progress.update(
                        task,
                        description="[bold cyan]:: Agentic loop started[/bold cyan]",
                        completed=20,
                    )
                elif name == "agent.iteration":
                    iteration = payload.get("iteration_count", 0)
                    max_iter = payload.get("max_iterations", max_iterations)
                    findings = payload.get("findings_count", 0)
                    tokens = payload.get("tokens_used", 0)
                    pct = min(85, 20 + int((iteration / max(max_iter, 1)) * 65))
                    progress.update(
                        task,
                        description=(
                            f"[bold cyan]:: Iteration {iteration}/{max_iter} "
                            f"• {findings} sources • {tokens} tokens[/bold cyan]"
                        ),
                        completed=pct,
                    )
                elif name == "agent.step_completed":
                    step_idx = payload.get("step_index", 0)
                    progress.update(
                        task,
                        description=f"[bold green]>> Step {step_idx} completed[/bold green]",
                        advance=5,
                    )
                elif name == "agent.budget_warning":
                    progress.update(
                        task,
                        description="[bold yellow]!! Budget getting low - wrapping up[/bold yellow]",
                    )
                elif name == "agent.loop_completed":
                    progress.update(
                        task,
                        description="[bold green]>> Loop completed - generating outputs[/bold green]",
                        completed=90,
                    )
                elif name == "agent.summaries_completed":
                    progress.update(
                        task,
                        description="[bold green]>> Summaries generated[/bold green]",
                        completed=93,
                    )
                elif name == "agent.learning_path_completed":
                    progress.update(
                        task,
                        description="[bold green]>> Learning path built[/bold green]",
                        completed=96,
                    )
                elif name == "agent.path_enhanced":
                    progress.update(
                        task,
                        description="[bold green]>> Path enhanced[/bold green]",
                        completed=98,
                    )
                elif name == "agent.completed":
                    progress.update(
                        task,
                        description="[bold green]>> Agent completed successfully[/bold green]",
                        completed=100,
                    )
                elif name == "agent.failed":
                    progress.update(
                        task,
                        description="[bold red]>> Agent failed[/bold red]",
                        completed=100,
                    )
                elif name == "agent.stage_failed":
                    stage = payload.get("stage", "unknown")
                    progress.update(
                        task,
                        description=f"[bold yellow]!! Stage degraded: {stage}[/bold yellow]",
                    )

            agent_instance.subscribe("*", handle_agent_event)

            result = await agent_instance.run(
                goal=goal,
                level=level,
                topic=topic,
                max_results=max_results,
            )

            progress.update(task, description="[bold white]Finished[/bold white]", completed=100)

        return result

@app.command()
def interactive() -> None:
    _print_banner()

    while True:
        try:
            console.print()
            console.print(Rule("[bold]NEW RESEARCH SESSION[/bold]", style="bold cyan"))
            topic = Prompt.ask(f"[{STYLE_LABEL}]Topic[/{STYLE_LABEL}]").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not topic:
            continue
        if topic.lower() in {"exit", "quit", "q"}:
            break

        try:
            goal = Prompt.ask(
                f"[{STYLE_LABEL}]Goal[/{STYLE_LABEL}]",
                default=_DEFAULTS["goal"],
            )
            level = Prompt.ask(
                f"[{STYLE_LABEL}]Level[/{STYLE_LABEL}]",
                default=_DEFAULTS["level"],
            )
            max_results = IntPrompt.ask(
                f"[{STYLE_LABEL}]Max results[/{STYLE_LABEL}]",
                default=_DEFAULTS["max_results"],
            )
            use_llm_expansion = Confirm.ask(
                f"[{STYLE_LABEL}]Use LLM query expansion?[/{STYLE_LABEL}]",
                default=_DEFAULTS["use_llm_expansion"],
            )
            use_llm_ranking = Confirm.ask(
                f"[{STYLE_LABEL}]Use LLM consensus ranking?[/{STYLE_LABEL}]",
                default=_DEFAULTS["use_llm_ranking"],
            )
            use_llm_learning_path = Confirm.ask(
                f"[{STYLE_LABEL}]Use LLM learning path generation?[/{STYLE_LABEL}]",
                default=_DEFAULTS["use_llm_learning_path"],
            )
            enable_rag = Confirm.ask(
                f"[{STYLE_LABEL}]Enable RAG indexing?[/{STYLE_LABEL}]",
                default=_DEFAULTS["enable_rag"],
            )

            query = _build_query(
                topic=topic,
                goal=goal,
                level=level,
                max_results=max_results,
                platforms=None,
                source_types=None,
            )

            result = asyncio.run(
                _execute_pipeline(
                    query=query,
                    use_llm_expansion=use_llm_expansion,
                    use_llm_ranking=use_llm_ranking,
                    use_llm_learning_path=use_llm_learning_path,
                    enable_rag=enable_rag,
                    show_progress=True,
                )
            )

            _render_result(result, _DEFAULTS["display_limit"])

        except KeyboardInterrupt:
            console.print(
                Panel(
                    "[bold yellow]INTERRUPTED[/bold yellow]",
                    border_style="yellow",
                )
            )
            break
        except Exception as exc:
            console.print(
                Panel(
                    escape(str(exc)),
                    title="[bold red]ERROR[/bold red]",
                    border_style="red",
                )
            )

        try:
            if not Confirm.ask(
                f"[{STYLE_ACCENT}]Run another search?[/{STYLE_ACCENT}]",
                default=True,
            ):
                break
        except (EOFError, KeyboardInterrupt):
            break

    console.print()
    console.print(
        Panel(
            "[bold white]SESSION CLOSED[/bold white]  -  [dim]all runs saved locally[/dim]",
            box=box.HEAVY,
            border_style="bold bright_cyan",
        )
    )

@app.command()
def chat(
    question: str = typer.Argument(..., help="Question to ask over saved research"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Limit chat to one saved run"),
    top_k: int = typer.Option(5, min=1, max=20, help="Number of sources to retrieve"),
) -> None:
    try:
        from rag.chat_engine import RAGChatEngine
    except Exception:
        console.print(
            Panel(
                "[bold red]RAG CHAT UNAVAILABLE[/bold red]\n"
                "Create rag/chat_engine.py, rag/vector_store.py, and rag/embedder.py first.",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    async def _run_chat() -> Any:
        async with RAGChatEngine(top_k=top_k) as engine:
            return await engine.ask(question, run_id=run_id, top_k=top_k)

    try:
        answer = asyncio.run(_run_chat())
    except KeyboardInterrupt:
        console.print(
            Panel(
                "[bold yellow]INTERRUPTED[/bold yellow]",
                border_style="yellow",
            )
        )
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(
            Panel(
                escape(str(exc)),
                title="[bold red]ERROR[/bold red]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    console.print(
        Panel(
            escape(str(getattr(answer, "answer", ""))),
            title="[bold bright_cyan on rgb(0,30,60)] >> ANSWER << [/bold bright_cyan on rgb(0,30,60)]",
            border_style="bold blue",
            box=box.ROUNDED,
        )
    )

    citations = list(getattr(answer, "citations", []) or [])
    if citations:
        console.print(_citations_table(citations))

@app.command()
def config() -> None:
    settings = get_settings()

    console.print()
    console.print(
        Panel(
            "[bold bright_cyan on rgb(0,30,60)] >> CONFIGURATION << [/bold bright_cyan on rgb(0,30,60)]",
            box=box.HEAVY,
            border_style="bold bright_cyan",
        )
    )

    provider_table = Table(
        title=Text("LLM PROVIDERS", style="bold bright_cyan"),
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    provider_table.add_column("PROVIDER", style="bold white")
    provider_table.add_column("KEY", style="dim")
    provider_table.add_column("AVAILABLE", justify="center")

    available_providers = {
        provider.value for provider in settings.available_llm_providers
    }

    for provider_name, masked_key in settings.masked_provider_keys.items():
        available = provider_name in available_providers
        provider_table.add_row(
            Text(provider_name),
            Text(masked_key or "-"),
            Text("YES", style=STYLE_OK) if available else Text("NO", style=STYLE_ERR),
        )

    console.print(provider_table)

    model_table = Table(
        title=Text("MODEL CONFIGURATION", style="bold bright_cyan"),
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    model_table.add_column("ROLE", style="bold white")
    model_table.add_column("MODEL", overflow="fold", style="bold cyan")
    model_table.add_row("Primary Fast", Text(settings.primary_fast_model))
    model_table.add_row("Primary Strong", Text(settings.primary_strong_model))
    model_table.add_row("Judge", Text(settings.ensemble_judge_model))
    model_table.add_row("Code", Text(settings.code_model))
    model_table.add_row("Embedding", Text(settings.embedding_model))
    model_table.add_row("Strong Fallbacks", Text("\n".join(settings.strong_model_chain)))
    model_table.add_row("Fast Fallbacks", Text("\n".join(settings.fast_model_chain)))
    console.print(model_table)

    search_settings_rows: list[tuple[str, Any]] = []
    default_platforms = getattr(settings, "default_search_platforms", None)
    if default_platforms:
        search_settings_rows.append(
            (
                "Default Platforms",
                ", ".join(_enum_value(item) for item in default_platforms),
            )
        )
    search_settings_rows.extend(
        [
            ("Search Concurrency", getattr(settings, "search_concurrency", None)),
            ("Max Query Variants", getattr(settings, "search_max_query_variants", None)),
            ("Request Timeout Seconds", getattr(settings, "search_request_timeout_seconds", None)),
            ("Source Max Retries", getattr(settings, "source_max_retries", None)),
        ]
    )

    visible_rows = [
        (label, value)
        for label, value in search_settings_rows
        if value is not None
    ]

    if visible_rows:
        search_table = Table(
            title=Text("SEARCH SETTINGS", style="bold bright_cyan"),
            box=box.ROUNDED,
            header_style="bold cyan",
        )
        search_table.add_column("SETTING", style="bold white")
        search_table.add_column("VALUE", overflow="fold", style="bold cyan")
        for label, value in visible_rows:
            search_table.add_row(Text(label), Text(str(value)))
        console.print(search_table)

    directory_table = Table(
        title=Text("DIRECTORIES", style="bold bright_cyan"),
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    directory_table.add_column("PURPOSE", style="bold white")
    directory_table.add_column("PATH", overflow="fold", style="dim")
    directory_table.add_row("Output", Text(str(settings.output_dir)))
    directory_table.add_row("Cache", Text(str(settings.cache_dir)))
    directory_table.add_row("Data", Text(str(settings.data_dir)))
    directory_table.add_row("Logs", Text(str(settings.logs_dir)))
    console.print(directory_table)

@app.command()
def version() -> None:
    banner_art = Text.assemble(
        (r"  ____  _____ ____  _   _ ____  _____ ____  _   _   _   _ _____ ____  _ " , "bold bright_cyan"),
        (r" |  _ \| ____/ ___|| | | / ___|| ____|  _ \| | | | / \ |_   _|  _ \| |", "bold bright_cyan"),
        (r" | |_) |  _| \___ \| | | \___ \|  _| | |_) | | | |/ _ \  | | | |_) | |", "bold bright_cyan"),
        (r" |  _ <| |___ ___) | |_| |___) | |___|  _ <| |_| / ___ \ | | |  _ <| |___", "bold bright_cyan"),
        (r" |_| \_\_____|____/ \___/|____/|_____|_| \_\\___/_/   \_\___|_| \_\_____|", "bold bright_cyan"),
    )

    console.print()
    console.print(
        Panel(
            Group(
                banner_art,
                Text("") ,
                Text(_APP_SUBTITLE, style="bold magenta"),
                Text("") ,
                Text(
                    f"VERSION {constants.APP_VERSION}   -   PYTHON {sys.version.split()[0]}",
                    style="bold white",
                ),
            ),
            title="[bold bright_cyan on rgb(0,30,60)] >> RESEARCH AGENT << [/bold bright_cyan on rgb(0,30,60)]",
            box=box.HEAVY,
            border_style="bold bright_cyan",
            padding=(1, 2),
        )
    )

@runs_app.command("list")
def list_runs(
    limit: int = typer.Option(20, min=1, max=200, help="Maximum number of runs to show"),
) -> None:
    store = SQLiteStore(_default_database_path())
    try:
        rows = asyncio.run(store.list_runs(limit))
    except Exception as exc:
        console.print(
            Panel(
                escape(str(exc)),
                title="[bold red]ERROR[/bold red]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    if not rows:
        console.print(
            Panel(
                "[bold yellow]NO SAVED RUNS FOUND[/bold yellow]",
                border_style="yellow",
            )
        )
        return

    table = Table(
        title=Text("SAVED RUNS", style="bold bright_cyan"),
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("REQUEST ID", overflow="fold", style="dim")
    table.add_column("TOPIC", overflow="fold", style="bold white")
    table.add_column("STATUS", justify="center")
    table.add_column("SOURCES", justify="right", style="bold")
    table.add_column("RANKED", justify="right", style="bold")
    table.add_column("STEPS", justify="right", style="bold")
    table.add_column("LATENCY", justify="right", style="dim")
    table.add_column("CREATED AT", overflow="fold", style="dim")

    for row in rows:
        status = str(row.get("status", ""))
        status_text = Text(status.upper(), style=_status_style(status))
        table.add_row(
            Text(str(row.get("request_id", ""))),
            Text(_ellipsis(row.get("topic", ""), 60)),
            status_text,
            Text(str(row.get("total_sources", 0))),
            Text(str(row.get("total_ranked", 0))),
            Text(str(row.get("total_steps", 0))),
            Text(_format_latency(row.get("latency_ms"))),
            Text(_short_timestamp(row.get("created_at"))),
        )

    console.print(table)

@runs_app.command("show")
def show_run(
    request_id: str = typer.Argument(..., help="Request ID of the saved run"),
) -> None:
    store = SQLiteStore(_default_database_path())
    try:
        run = asyncio.run(store.get_run(request_id))
    except Exception as exc:
        console.print(
            Panel(
                escape(str(exc)),
                title="[bold red]ERROR[/bold red]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    if run is None:
        console.print(
            Panel(
                f"[bold red]RUN NOT FOUND[/bold red]  {escape(request_id)}",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    _render_run(run)

async def _execute_pipeline(
    query: SearchQuerySchema,
    use_llm_expansion: bool,
    use_llm_ranking: bool,
    use_llm_learning_path: bool,
    enable_rag: bool,
    show_progress: bool,
) -> PipelineResult:
    pipeline = ResearchPipeline(
        search_orchestrator=SearchOrchestrator(use_llm_expansion=use_llm_expansion),
        consensus_ranker=ConsensusRanker(use_llm=use_llm_ranking),
        learning_path_builder=LearningPathBuilder(use_llm=use_llm_learning_path),
        enable_rag=enable_rag,
    )

    result: PipelineResult | None = None

    async with pipeline:
        if not show_progress:
            result = await pipeline.run(query)
        else:
            progress = Progress(
                SpinnerColumn(style="bold cyan"),
                TextColumn("[bold cyan]{task.description}[/bold cyan]"),
                BarColumn(bar_width=30, style="cyan", complete_style="green"),
                TextColumn("[bold white]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            )

            with progress:
                task = progress.add_task(
                    "[bold white]Starting research pipeline[/bold white]",
                    total=100,
                )

                async def handle_event(event: Any) -> None:
                    payload = event.payload or {}
                    if event.name == PIPELINE_STARTED:
                        progress.update(
                            task,
                            description="[bold cyan]>> Pipeline started[/bold cyan]",
                            completed=5,
                        )
                    elif event.name == SEARCH_COMPLETED:
                        total = payload.get("total_sources", 0)
                        progress.update(
                            task,
                            description=f"[bold green]>> Search completed: {total} sources[/bold green]",
                            completed=30,
                        )
                    elif event.name == RANKING_COMPLETED:
                        total = payload.get("total_ranked", 0)
                        progress.update(
                            task,
                            description=f"[bold green]>> Ranking completed: {total} ranked[/bold green]",
                            completed=50,
                        )
                    elif event.name == SOURCE_SUMMARIES_COMPLETED:
                        total = payload.get("total_summaries", 0)
                        progress.update(
                            task,
                            description=f"[bold green]>> Summaries completed: {total}[/bold green]",
                            completed=65,
                        )
                    elif event.name == LEARNING_PATH_COMPLETED:
                        total = payload.get("total_steps", 0)
                        progress.update(
                            task,
                            description=f"[bold green]>> Learning path: {total} steps[/bold green]",
                            completed=80,
                        )
                    elif event.name == PATH_ENHANCED:
                        progress.update(
                            task,
                            description="[bold green]>> Path enhanced[/bold green]",
                            completed=85,
                        )
                    elif event.name == RAG_INDEXING_STARTED:
                        progress.update(
                            task,
                            description="[bold white]:: RAG indexing...[/bold white]",
                            completed=88,
                        )
                    elif event.name == RAG_INDEXED:
                        progress.update(
                            task,
                            description="[bold green]>> RAG indexed[/bold green]",
                            completed=92,
                        )
                    elif event.name == STAGE_FAILED:
                        stage = payload.get("stage", "unknown")
                        progress.update(
                            task,
                            description=f"[bold yellow]!! Stage degraded: {stage}[/bold yellow]",
                        )
                    elif event.name == PIPELINE_COMPLETED:
                        progress.update(
                            task,
                            description="[bold green]>> Pipeline completed[/bold green]",
                            completed=100,
                        )
                    elif event.name == PIPELINE_FAILED:
                        progress.update(
                            task,
                            description="[bold red]>> Pipeline failed[/bold red]",
                            completed=100,
                        )

                pipeline.subscribe("*", handle_event)
                result = await pipeline.run(query)
                progress.update(task, description="[bold white]Finished[/bold white]", completed=100)

    if result is None:
        raise RuntimeError("Pipeline execution completed without result")

    return result

def _build_query(
    topic: str,
    goal: str,
    level: str,
    max_results: int,
    platforms: Optional[List[str]],
    source_types: Optional[List[str]],
) -> SearchQuerySchema:
    payload: dict[str, Any] = {
        "topic": topic,
        "goal": goal,
        "level": level,
        "max_results": max_results,
    }
    if platforms:
        payload["platforms"] = _parse_enum_values(platforms, SourcePlatform, "platform")
    if source_types:
        payload["source_types"] = _parse_enum_values(source_types, SourceType, "source-type")
    try:
        return SearchQuerySchema.model_validate(payload)
    except Exception as exc:
        raise typer.BadParameter(str(exc))

def _parse_enum_values(values: List[str], enum_class: Any, field_name: str) -> list[str]:
    parsed: list[str] = []
    for value in values:
        cleaned = str(value).strip().lower()
        try:
            parsed.append(enum_class(cleaned).value)
        except Exception:
            valid_values = ", ".join(item.value for item in enum_class)
            raise typer.BadParameter(
                f"Invalid {field_name}: {value}. Valid values: {valid_values}"
            )
    return parsed

def _render_result(result: PipelineResult, limit: int) -> None:
    console.print()
    topic = result.query.topic if result.query else ""
    status = _enum_value(result.status)
    border_style = _status_style(status)

    console.print(
        Rule(
            f"[bold]REPORT  //  {escape(topic)}[/bold]",
            style=border_style,
            characters="─",
        )
    )

    console.print(_header_panel(result))

    ai_panel = _ai_panel(result)
    if ai_panel is not None:
        console.print(ai_panel)

    console.print(_stats_panel(result))

    for issue_panel in _issue_panels(result):
        console.print(issue_panel)

    platform_panel = _platform_breakdown_panel(result)
    if platform_panel is not None:
        console.print(platform_panel)

    expanded_panel = _expanded_queries_panel(result)
    if expanded_panel is not None:
        console.print(expanded_panel)

    if result.ranked_sources:
        console.print(_ranked_table(result.ranked_sources, limit))
    elif result.sources:
        console.print(_sources_table(result.sources, limit))
    else:
        console.print(_no_sources_panel())

    source_summaries = list(getattr(result, "source_summaries", []) or [])
    if source_summaries:
        console.print(_source_summaries_table(source_summaries, limit))

    if result.learning_path and result.learning_path.steps:
        _render_learning_path(
            result.learning_path,
            getattr(result, "enhanced_learning_path", None),
        )

    if result.saved_files:
        console.print(_saved_files_table(result.saved_files))

    console.print(Rule(style=border_style, characters="─"))

def _render_agent_result(result: Any, limit: int) -> None:
    console.print()

    status = result.status
    border_style = _status_style(status)

    console.print(
        Rule(
            f"[bold]AGENT REPORT  //  {escape(result.topic or result.goal)}[/bold]",
            style=border_style,
            characters="─",
        )
    )

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=STYLE_LABEL, justify="right", no_wrap=True)
    grid.add_column(overflow="fold")
    grid.add_row("GOAL", Text(result.goal, style=STYLE_VALUE))
    grid.add_row("TOPIC", Text(result.topic, style=STYLE_VALUE))
    grid.add_row("LEVEL", Text(result.level or "-", style=STYLE_VALUE))
    grid.add_row("AGENT ID", Text(result.agent_id, style=STYLE_DIM))
    grid.add_row("STATUS", Text(status.upper(), style=_status_badge_style(status)))
    grid.add_row("ITERATIONS", Text(str(result.total_iterations), style=STYLE_VALUE))
    grid.add_row("TOKENS USED", Text(str(result.total_tokens_used), style=STYLE_VALUE))
    grid.add_row("TOKENS LEFT", Text(str(result.tokens_remaining), style=STYLE_VALUE))
    grid.add_row("BUDGET", Text(str(result.total_budget), style=STYLE_VALUE))
    grid.add_row("LATENCY", Text(_format_latency(result.latency_ms), style=STYLE_VALUE))

    console.print(
        Panel(
            grid,
            title="[bold bright_cyan on rgb(0,30,60)] >> AUTONOMOUS AGENT REPORT << [/bold bright_cyan on rgb(0,30,60)]",
            subtitle=escape(result.goal or ""),
            box=box.HEAVY,
            border_style=border_style,
            padding=(1, 2),
        )
    )

    token_pct = 0
    if result.total_budget > 0:
        token_pct = (result.total_tokens_used / result.total_budget) * 100

    token_bar = Table.grid(padding=(0, 1))
    token_bar.add_column(style="bold white")
    token_bar.add_column()
    filled = int(token_pct / 5)
    bar_str = "█" * filled + "░" * (20 - filled)
    token_bar.add_row(
        "TOKEN BUDGET",
        Text(f"[{bar_str}] {token_pct:.1f}%", style="bold cyan"),
    )

    console.print(
        Panel(
            token_bar,
            title="[bold bright_cyan on rgb(0,30,60)] >> TOKEN USAGE << [/bold bright_cyan on rgb(0,30,60)]",
            border_style="blue",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )

    stats_grid = Table.grid(padding=(0, 2))
    stats_grid.add_column(style=STYLE_LABEL, justify="right", no_wrap=True)
    stats_grid.add_column(overflow="fold")
    stats_grid.add_row("FINDINGS", Text(str(len(result.findings)), style="bold white"))
    stats_grid.add_row("RANKED", Text(str(len(result.ranked_sources)), style="bold white"))
    stats_grid.add_row("SUMMARIES", Text(str(len(result.source_summaries)), style="bold white"))

    learning_steps = 0
    if result.learning_path and hasattr(result.learning_path, "steps"):
        learning_steps = len(result.learning_path.steps)
    stats_grid.add_row("LEARNING STEPS", Text(str(learning_steps), style="bold white"))

    enhanced = "YES" if result.enhanced_learning_path else "NO"
    stats_grid.add_row("ENHANCED PATH", Text(enhanced, style=STYLE_ACCENT))
    stats_grid.add_row(
        "ERRORS",
        Text(str(len(result.errors)), style=STYLE_ERR if result.errors else STYLE_OK),
    )
    stats_grid.add_row(
        "WARNINGS",
        Text(str(len(result.warnings)), style=STYLE_WARN if result.warnings else STYLE_OK),
    )

    console.print(
        Panel(
            stats_grid,
            title="[bold bright_cyan on rgb(0,30,60)] >> AGENT METRICS << [/bold bright_cyan on rgb(0,30,60)]",
            border_style="blue",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )

    if result.errors:
        console.print(
            _bullet_panel(
                result.errors,
                "[bold red] >> ERRORS << [/bold red]",
                "red",
                "red",
            )
        )

    if result.warnings:
        console.print(
            _bullet_panel(
                result.warnings,
                "[bold yellow] >> WARNINGS << [/bold yellow]",
                "yellow",
                "yellow",
            )
        )

    if result.ranked_sources:
        console.print(_ranked_table(result.ranked_sources, limit))

    source_summaries = list(result.source_summaries or [])
    if source_summaries:
        console.print(_source_summaries_table(source_summaries, limit))

    if result.learning_path and hasattr(result.learning_path, "steps") and result.learning_path.steps:
        _render_learning_path(
            result.learning_path,
            result.enhanced_learning_path,
        )

    if result.saved_files:
        console.print(_saved_files_table(result.saved_files))

    console.print(Rule(style=border_style, characters="─"))

def _render_agent_minimal(result: Any) -> None:
    console.print(result.agent_id)
    console.print(result.status)
    console.print(f"iterations: {result.total_iterations}")
    console.print(f"tokens: {result.total_tokens_used}/{result.total_budget}")
    for key, value in result.saved_files.items():
        console.print(f"{key}: {value}")

def _render_minimal(result: PipelineResult) -> None:
    console.print(result.request_id)
    console.print(_enum_value(result.status))
    for key, value in result.saved_files.items():
        console.print(f"{key}: {value}")

def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def _get_ai_fields(result: PipelineResult) -> dict[str, Any]:
    query = result.query
    original = ""
    corrected = ""
    primary = ""
    keywords: list[str] = []
    intent = ""
    detected_level = ""
    short_queries: list[str] = []

    if query is not None:
        original = str(getattr(query, "topic", "") or "")
        corrected = str(getattr(query, "corrected_topic", "") or "")
        primary = str(getattr(query, "primary_concept", "") or "")
        keywords = _clean_list_strings(getattr(query, "keywords", []) or [])
        intent = str(getattr(query, "intent", "") or "")
        detected_level = str(
            getattr(query, "detected_level", "")
            or getattr(query, "level", "")
            or ""
        )
        short_queries = _clean_list_strings(
            getattr(query, "short_search_queries", []) or []
        )

    result_corrected = getattr(result, "corrected_topic", None)
    if result_corrected:
        corrected = str(result_corrected)
    result_primary = getattr(result, "primary_concept", None)
    if result_primary:
        primary = str(result_primary)
    result_keywords = getattr(result, "keywords", None)
    if result_keywords:
        keywords = _clean_list_strings(result_keywords)
    result_intent = getattr(result, "intent", None)
    if result_intent:
        intent = str(result_intent)
    result_detected_level = getattr(result, "detected_level", None)
    if result_detected_level:
        detected_level = str(result_detected_level)
    result_short_queries = getattr(result, "short_search_queries", None)
    if result_short_queries:
        short_queries = _clean_list_strings(result_short_queries)

    if not corrected:
        corrected = original
    if not primary:
        primary = corrected

    expanded_queries = _clean_list_strings(getattr(result, "expanded_queries", []) or [])

    return {
        "original": original,
        "corrected": corrected,
        "primary": primary,
        "keywords": keywords,
        "intent": intent,
        "detected_level": detected_level,
        "short_search_queries": short_queries,
        "expanded_queries": expanded_queries,
    }

def _header_panel(result: PipelineResult) -> Panel:
    status = _enum_value(result.status)
    ai = _get_ai_fields(result)

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=STYLE_LABEL, justify="right", no_wrap=True)
    grid.add_column(overflow="fold")
    grid.add_row("TOPIC", Text(ai["original"] or "-", style=STYLE_VALUE))

    goal = "-"
    level = "-"
    if result.query is not None:
        goal = str(getattr(result.query, "goal", "") or "-")
        level = str(getattr(result.query, "level", "") or "-")

    grid.add_row("GOAL", Text(goal, style=STYLE_VALUE))
    grid.add_row("LEVEL", Text(level, style=STYLE_VALUE))
    grid.add_row("REQUEST ID", Text(result.request_id, style=STYLE_DIM))
    grid.add_row("STATUS", Text(status.upper(), style=_status_badge_style(status)))
    grid.add_row("LATENCY", Text(_format_latency(result.latency_ms), style=STYLE_VALUE))

    return Panel(
        grid,
        title="[bold bright_cyan on rgb(0,30,60)] >> RESEARCH AGENT REPORT << [/bold bright_cyan on rgb(0,30,60)]",
        subtitle=escape(ai["original"] or ""),
        box=box.HEAVY,
        border_style=_status_style(status),
        padding=(1, 2),
    )

def _ai_panel(result: PipelineResult) -> Panel | None:
    ai = _get_ai_fields(result)
    if not ai["original"] and not ai["corrected"]:
        return None

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=STYLE_LABEL, justify="right", no_wrap=True)
    grid.add_column(overflow="fold")
    grid.add_row("ORIGINAL", Text(ai["original"] or "-", style=STYLE_VALUE))

    if ai["corrected"] and ai["corrected"] != ai["original"]:
        grid.add_row("AI CORRECTED", Text(ai["corrected"], style="bold green"))
    else:
        grid.add_row("AI CORRECTED", Text(ai["corrected"] or "-", style=STYLE_VALUE))

    grid.add_row("PRIMARY CONCEPT", Text(ai["primary"] or "-", style="bold magenta"))

    if ai["keywords"]:
        grid.add_row("KEYWORDS", Text(", ".join(ai["keywords"][:12]), style="bold white"))
    if ai["intent"]:
        grid.add_row("INTENT", Text(ai["intent"].upper(), style="bold cyan"))
    if ai["detected_level"]:
        grid.add_row("DETECTED LEVEL", Text(ai["detected_level"].title(), style="bold yellow"))
    if ai["short_search_queries"]:
        grid.add_row(
            "AI SEARCH QUERIES",
            Text(
                "\n".join(f"- {item}" for item in ai["short_search_queries"][:8]),
                style="dim white",
            ),
        )

    return Panel(
        grid,
        title="[bold bright_cyan on rgb(0,30,60)] >> AI QUERY UNDERSTANDING << [/bold bright_cyan on rgb(0,30,60)]",
        border_style="blue",
        box=box.ROUNDED,
        padding=(1, 2),
    )

def _stats_panel(result: PipelineResult) -> Panel:
    source_count = len(result.sources or [])
    ranked_count = len(result.ranked_sources or [])
    summary_count = len(list(getattr(result, "source_summaries", []) or []))
    learning_steps = len(result.learning_path.steps) if result.learning_path else 0
    enhanced_path = "YES" if getattr(result, "enhanced_learning_path", None) else "NO"
    rag_documents = int(getattr(result, "rag_documents_indexed", 0) or 0)
    error_count = len(result.errors or [])
    warning_count = len(list(getattr(result, "warnings", []) or []))

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=STYLE_LABEL, justify="right", no_wrap=True)
    grid.add_column(overflow="fold")
    grid.add_row("SOURCES", Text(str(source_count), style="bold white"))
    grid.add_row("RANKED", Text(str(ranked_count), style="bold white"))
    grid.add_row("SUMMARIES", Text(str(summary_count), style="bold white"))
    grid.add_row("LEARNING STEPS", Text(str(learning_steps), style="bold white"))
    grid.add_row("ENHANCED PATH", Text(enhanced_path, style=STYLE_ACCENT))
    grid.add_row("RAG DOCUMENTS", Text(str(rag_documents), style="bold white"))
    grid.add_row("ERRORS", Text(str(error_count), style=STYLE_ERR if error_count else STYLE_OK))
    grid.add_row("WARNINGS", Text(str(warning_count), style=STYLE_WARN if warning_count else STYLE_OK))

    return Panel(
        grid,
        title="[bold bright_cyan on rgb(0,30,60)] >> PIPELINE METRICS << [/bold bright_cyan on rgb(0,30,60)]",
        border_style="blue",
        box=box.ROUNDED,
        padding=(1, 2),
    )

def _platform_breakdown_panel(result: PipelineResult) -> Panel | None:
    sources = list(result.sources or [])
    if not sources and result.ranked_sources:
        sources = [
            ranked.source
            for ranked in result.ranked_sources
            if getattr(ranked, "source", None) is not None
        ]
    if not sources:
        return None

    counts: dict[str, int] = {}
    for source in sources:
        platform = _enum_value(getattr(source, "platform", "")) or "unknown"
        counts[platform] = counts.get(platform, 0) + 1

    table = Table(
        box=box.ROUNDED,
        header_style="bold cyan",
        show_edge=False,
    )
    table.add_column("PLATFORM", style="bold white")
    table.add_column("SOURCES", justify="right", style="bold cyan")

    for platform, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        table.add_row(Text(platform), Text(str(count)))

    return Panel(
        table,
        title="[bold bright_cyan on rgb(0,30,60)] >> PLATFORM COVERAGE << [/bold bright_cyan on rgb(0,30,60)]",
        border_style="cyan",
        box=box.ROUNDED,
    )

def _issue_panels(result: PipelineResult) -> list[Panel]:
    warnings = _clean_list_strings(getattr(result, "warnings", []) or [])
    errors = _clean_list_strings(result.errors or [])

    diagnoses: list[str] = []
    for item in warnings + errors:
        if item.startswith("no_sources_reason"):
            diagnoses.append(item)

    platform_prefixes = (
        "platform_unavailable:",
        "platform_skipped",
        "platform_degraded:",
        "platform_error:",
        "search_platforms_failed",
        "no_sources_reason_platform_failure",
    )

    platform_issues = [
        item
        for item in warnings
        if item.startswith(platform_prefixes) and item not in diagnoses
    ]

    other_warnings = [
        item
        for item in warnings
        if item not in diagnoses and item not in platform_issues
    ]

    other_errors = [
        item
        for item in errors
        if item not in diagnoses
    ]

    panels: list[Panel] = []

    if diagnoses:
        panels.append(
            _bullet_panel(
                _unique_items(diagnoses),
                "[bold white on red] >> NO SOURCES DIAGNOSIS << [/bold white on red]",
                "red",
                "red",
            )
        )

    if platform_issues:
        panels.append(
            _bullet_panel(
                _unique_items(platform_issues),
                "[bold black on yellow] >> PLATFORM ISSUES << [/bold black on yellow]",
                "yellow",
                "yellow",
            )
        )

    if other_warnings:
        panels.append(
            _bullet_panel(
                _unique_items(other_warnings),
                "[bold yellow] >> WARNINGS << [/bold yellow]",
                "yellow",
                "yellow",
            )
        )

    if other_errors:
        panels.append(
            _bullet_panel(
                _unique_items(other_errors),
                "[bold red] >> ERRORS << [/bold red]",
                "red",
                "red",
            )
        )

    return panels

def _expanded_queries_panel(result: PipelineResult) -> Panel | None:
    ai = _get_ai_fields(result)
    queries = ai["expanded_queries"]
    if not queries:
        return None

    body = "\n".join(
        f"[bold cyan]-[/bold cyan] {escape(item)}"
        for item in queries[:12]
    )

    return Panel(
        body,
        title="[bold bright_cyan on rgb(0,30,60)] >> SEARCH QUERIES << [/bold bright_cyan on rgb(0,30,60)]",
        border_style="cyan",
        box=box.ROUNDED,
    )

def _no_sources_panel() -> Panel:
    return Panel(
        "[bold yellow]NO SOURCES WERE FOUND[/bold yellow]",
        border_style="yellow",
        box=box.ROUNDED,
    )

def _ranked_table(ranked_sources: list[Any], limit: int) -> Table:
    table = Table(
        title=Text("RANKED SOURCES", style="bold bright_cyan"),
        box=box.ROUNDED,
        header_style="bold bright_cyan on rgb(0,30,60)",
        show_lines=True,
    )
    table.add_column("#", justify="right", style="bold white")
    table.add_column("TITLE", overflow="fold")
    table.add_column("PLATFORM", style="bold cyan")
    table.add_column("TYPE", style="bold magenta")
    table.add_column("DIFFICULTY", justify="center")
    table.add_column("SCORE", justify="right")
    table.add_column("CONF", justify="right")
    table.add_column("YEAR", justify="right", style="dim")

    for ranked in ranked_sources[:limit]:
        source = getattr(ranked, "source", None)
        if source is None:
            continue

        title_text = _title_text(
            getattr(source, "title", ""),
            getattr(source, "url", ""),
        )
        platform = _enum_value(getattr(source, "platform", ""))
        source_type = _enum_value(getattr(source, "source_type", ""))
        difficulty = _enum_value(getattr(source, "difficulty", None)) or "-"
        difficulty_text = Text(difficulty, style=_difficulty_style(difficulty))
        score_text = Text(
            _format_float(getattr(ranked, "score", None)),
            style=f"{_score_style(getattr(ranked, 'score', None))} bold",
        )
        confidence_text = Text(
            _format_float(getattr(ranked, "confidence", None)),
            style=_score_style(getattr(ranked, "confidence", None)),
        )
        year = str(getattr(source, "year", "") or "-")
        row_style = "bold" if int(getattr(ranked, "rank", 999)) <= 3 else None

        table.add_row(
            Text(str(getattr(ranked, "rank", "-"))),
            title_text,
            Text(platform),
            Text(source_type),
            difficulty_text,
            score_text,
            confidence_text,
            Text(year),
            style=row_style,
        )

    return table

def _sources_table(sources: list[Any], limit: int) -> Table:
    table = Table(
        title=Text("SOURCES", style="bold bright_cyan"),
        box=box.ROUNDED,
        header_style="bold bright_cyan on rgb(0,30,60)",
        show_lines=True,
    )
    table.add_column("#", justify="right", style="bold white")
    table.add_column("TITLE", overflow="fold")
    table.add_column("PLATFORM", style="bold cyan")
    table.add_column("TYPE", style="bold magenta")
    table.add_column("YEAR", justify="right", style="dim")

    for index, source in enumerate(sources[:limit], start=1):
        table.add_row(
            Text(str(index)),
            _title_text(getattr(source, "title", ""), getattr(source, "url", "")),
            Text(_enum_value(getattr(source, "platform", ""))),
            Text(_enum_value(getattr(source, "source_type", ""))),
            Text(str(getattr(source, "year", "") or "-")),
        )

    return table

def _source_summaries_table(source_summaries: list[Any], limit: int) -> Table:
    table = Table(
        title=Text("SOURCE SUMMARIES", style="bold bright_cyan"),
        box=box.ROUNDED,
        header_style="bold bright_cyan on rgb(0,30,60)",
        show_lines=True,
    )
    table.add_column("#", justify="right", style="bold white")
    table.add_column("TITLE", overflow="fold")
    table.add_column("DIFFICULTY", justify="center")
    table.add_column("SUMMARY", overflow="fold")
    table.add_column("WHY USEFUL", overflow="fold")

    for index, summary in enumerate(source_summaries[:limit], start=1):
        title = str(_get_field(summary, "title", "") or "Untitled")
        difficulty = _enum_value(_get_field(summary, "difficulty", None)) or "-"
        summary_text = str(_get_field(summary, "summary", "") or "-")
        why_useful = str(_get_field(summary, "why_useful", "") or "-")

        table.add_row(
            Text(str(index)),
            Text(_ellipsis(title, 90), style="bold white", overflow="fold"),
            Text(difficulty, style=_difficulty_style(difficulty)),
            Text(_ellipsis(summary_text, 480), overflow="fold"),
            Text(_ellipsis(why_useful, 280), overflow="fold"),
        )

    return table

def _citations_table(citations: list[Any]) -> Table:
    table = Table(
        title=Text("CITATIONS", style="bold bright_cyan"),
        box=box.ROUNDED,
        header_style="bold bright_cyan on rgb(0,30,60)",
        show_lines=True,
    )
    table.add_column("SOURCE ID", overflow="fold", style="dim")
    table.add_column("TITLE", overflow="fold", style="bold white")
    table.add_column("SCORE", justify="right", style="bold")

    for citation in citations:
        source_id = str(_get_field(citation, "source_id", "") or "-")
        title = str(_get_field(citation, "title", "") or "Untitled")
        url = _get_field(citation, "url", None)
        score = _get_field(citation, "score", None)

        table.add_row(
            Text(source_id),
            _title_text(title, url),
            Text(_format_float(score)),
        )

    return table

def _saved_files_table(saved_files: dict[str, str]) -> Table:
    table = Table(
        title=Text("SAVED FILES", style="bold bright_cyan"),
        box=box.SIMPLE_HEAVY,
        header_style="bold bright_cyan on rgb(0,30,60)",
    )
    table.add_column("TYPE", style="bold cyan")
    table.add_column("PATH", overflow="fold", style="bold white")

    styles = {
        "json": "bold cyan",
        "markdown": "bold magenta",
        "html": "bold blue",
        "sqlite": "bold yellow",
    }

    for key, value in saved_files.items():
        table.add_row(
            Text(key.upper(), style=styles.get(key, "bold white")),
            Text(str(value)),
        )

    return table

def _render_learning_path(
    learning_path: Any,
    enhanced_learning_path: Any = None,
) -> None:
    console.print()

    tree = Tree(
        Text("LEARNING PATH", style=STYLE_HEADER),
        guide_style="bold cyan",
    )

    steps = list(_get_field(learning_path, "steps", []) or [])

    for step in steps:
        step_number = int(_get_field(step, "step", 0) or 0)
        step_title = str(_get_field(step, "title", "") or "")
        objective = str(_get_field(step, "objective", "") or "")
        estimated_minutes = _get_field(step, "estimated_minutes", None)

        step_label = Text.assemble(
            (f" STEP {step_number} ", STYLE_STEP),
            ("  ", ""),
            (step_title, "bold cyan"),
        )

        step_branch = tree.add(step_label)

        if objective and objective != step_title:
            step_branch.add(Text(objective, style="italic white"))

        if estimated_minutes is not None:
            step_branch.add(
                Text(
                    f"ESTIMATED TIME  {estimated_minutes} minutes",
                    style="bold yellow",
                )
            )

        resources = list(_get_field(step, "resources", []) or [])
        if resources:
            resources_branch = step_branch.add(Text("RESOURCES", style="bold green"))
            for ranked in resources:
                source = _get_field(ranked, "source", None)
                if source is None:
                    continue
                resources_branch.add(
                    _title_text(
                        _get_field(source, "title", ""),
                        _get_field(source, "url", ""),
                    )
                )

        enhancement = _find_step_enhancement(enhanced_learning_path, step_number)
        if enhancement is None:
            continue

        prerequisites = list(_get_field(enhancement, "prerequisites", []) or [])
        suggested_projects = list(_get_field(enhancement, "suggested_projects", []) or [])
        key_concepts = list(_get_field(enhancement, "key_concepts", []) or [])

        if prerequisites:
            branch = step_branch.add(Text("PREREQUISITES", style="bold yellow"))
            for item in prerequisites:
                branch.add(Text(f"- {item}", style="dim white"))

        if suggested_projects:
            branch = step_branch.add(Text("PROJECTS", style="bold magenta"))
            for item in suggested_projects:
                branch.add(Text(f"- {item}", style="dim white"))

        if key_concepts:
            branch = step_branch.add(Text("KEY CONCEPTS", style="bold cyan"))
            for item in key_concepts:
                branch.add(Text(f"- {item}", style="dim white"))

    console.print(
        Panel(
            tree,
            box=box.ROUNDED,
            border_style="bold blue",
            padding=(1, 2),
        )
    )

def _render_run(run: dict[str, Any]) -> None:
    status = str(run.get("status", ""))
    border_style = _status_style(status)
    payload = _parse_json_safe(run.get("payload"), {})
    query_payload = payload.get("query") if isinstance(payload, dict) else {}
    if not isinstance(query_payload, dict):
        query_payload = {}

    topic = str(run.get("topic", "") or query_payload.get("topic", "") or "")
    corrected_topic = str(query_payload.get("corrected_topic", "") or "")
    primary_concept = str(query_payload.get("primary_concept", "") or corrected_topic)
    keywords = _clean_list_strings(query_payload.get("keywords", []))
    intent = str(query_payload.get("intent", "") or "")
    detected_level = str(
        query_payload.get("detected_level", "")
        or query_payload.get("level", "")
        or run.get("level", "")
        or ""
    )

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=STYLE_LABEL, justify="right", no_wrap=True)
    grid.add_column(overflow="fold")
    grid.add_row("REQUEST ID", Text(str(run.get("request_id", "")), style=STYLE_DIM))
    grid.add_row("TOPIC", Text(topic or "-", style=STYLE_VALUE))
    grid.add_row("GOAL", Text(str(run.get("goal", "") or "-"), style=STYLE_VALUE))
    grid.add_row("LEVEL", Text(str(run.get("level", "") or "-"), style=STYLE_VALUE))
    grid.add_row("STATUS", Text(status.upper(), style=_status_badge_style(status)))
    grid.add_row("SOURCES", Text(str(run.get("total_sources", 0)), style="bold white"))
    grid.add_row("RANKED", Text(str(run.get("total_ranked", 0)), style="bold white"))
    grid.add_row("STEPS", Text(str(run.get("total_steps", 0)), style="bold white"))
    grid.add_row("LATENCY", Text(_format_latency(run.get("latency_ms")), style=STYLE_VALUE))
    grid.add_row("CREATED AT", Text(_short_timestamp(run.get("created_at")), style=STYLE_DIM))
    grid.add_row("FINISHED AT", Text(_short_timestamp(run.get("finished_at")), style=STYLE_DIM))

    console.print()
    console.print(
        Panel(
            grid,
            title="[bold bright_cyan on rgb(0,30,60)] >> SAVED RUN << [/bold bright_cyan on rgb(0,30,60)]",
            box=box.HEAVY,
            border_style=border_style,
            padding=(1, 2),
        )
    )

    if corrected_topic or primary_concept or keywords:
        ai_grid = Table.grid(padding=(0, 2))
        ai_grid.add_column(style=STYLE_LABEL, justify="right", no_wrap=True)
        ai_grid.add_column(overflow="fold")
        ai_grid.add_row("ORIGINAL", Text(topic or "-", style=STYLE_VALUE))

        if corrected_topic and corrected_topic != topic:
            ai_grid.add_row("AI CORRECTED", Text(corrected_topic, style="bold green"))
        else:
            ai_grid.add_row("AI CORRECTED", Text(corrected_topic or "-", style=STYLE_VALUE))

        ai_grid.add_row("PRIMARY CONCEPT", Text(primary_concept or "-", style="bold magenta"))

        if keywords:
            ai_grid.add_row("KEYWORDS", Text(", ".join(keywords[:12]), style="bold white"))
        if intent:
            ai_grid.add_row("INTENT", Text(intent.upper(), style="bold cyan"))
        if detected_level:
            ai_grid.add_row("DETECTED LEVEL", Text(detected_level.title(), style="bold yellow"))

        console.print(
            Panel(
                ai_grid,
                title="[bold bright_cyan on rgb(0,30,60)] >> AI QUERY UNDERSTANDING << [/bold bright_cyan on rgb(0,30,60)]",
                border_style="blue",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

    errors = _clean_list_strings(payload.get("errors", []))
    warnings = _clean_list_strings(payload.get("warnings", []))
    expanded_queries = _clean_list_strings(payload.get("expanded_queries", []))

    if warnings:
        console.print(
            _bullet_panel(
                warnings,
                "[bold yellow] >> WARNINGS << [/bold yellow]",
                "yellow",
                "yellow",
            )
        )

    if errors:
        console.print(
            _bullet_panel(
                errors,
                "[bold red] >> ERRORS << [/bold red]",
                "red",
                "red",
            )
        )

    if expanded_queries:
        console.print(
            Panel(
                "\n".join(
                    f"[bold cyan]-[/bold cyan] {escape(item)}"
                    for item in expanded_queries[:12]
                ),
                title="[bold bright_cyan on rgb(0,30,60)] >> EXPANDED QUERIES << [/bold bright_cyan on rgb(0,30,60)]",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )

    sources = run.get("sources", [])
    if sources:
        table = Table(
            title=Text("SAVED RANKED SOURCES", style="bold bright_cyan"),
            box=box.ROUNDED,
            header_style="bold bright_cyan on rgb(0,30,60)",
            show_lines=True,
        )
        table.add_column("#", justify="right", style="bold white")
        table.add_column("TITLE", overflow="fold")
        table.add_column("PLATFORM", style="bold cyan")
        table.add_column("TYPE", style="bold magenta")
        table.add_column("DIFFICULTY", justify="center")
        table.add_column("SCORE", justify="right")
        table.add_column("CONF", justify="right")

        for source in sources:
            title_text = _title_text(source.get("title"), source.get("url"))
            difficulty = str(source.get("difficulty") or "-")
            table.add_row(
                Text(str(source.get("rank") or "-")),
                title_text,
                Text(str(source.get("platform") or "-")),
                Text(str(source.get("source_type") or "-")),
                Text(difficulty, style=_difficulty_style(difficulty)),
                Text(_format_float(source.get("score")), style="bold"),
                Text(_format_float(source.get("confidence"))),
            )

        console.print(table)

    steps = run.get("learning_steps", [])
    if steps:
        table = Table(
            title=Text("SAVED LEARNING STEPS", style="bold bright_cyan"),
            box=box.ROUNDED,
            header_style="bold bright_cyan on rgb(0,30,60)",
            show_lines=True,
        )
        table.add_column("STEP", justify="right", style="bold white")
        table.add_column("TITLE", overflow="fold", style="bold cyan")
        table.add_column("OBJECTIVE", overflow="fold")
        table.add_column("MINUTES", justify="right", style="bold yellow")
        table.add_column("RESOURCES", justify="right", style="bold")

        for step in steps:
            resources = _parse_json_safe(step.get("resources"), [])
            table.add_row(
                Text(str(step.get("step") or "-")),
                Text(str(step.get("title") or "")),
                Text(_ellipsis(step.get("objective") or "", 200)),
                Text(str(step.get("estimated_minutes") or "-")),
                Text(str(len(resources) if isinstance(resources, list) else 0)),
            )

        console.print(table)

def _find_step_enhancement(enhanced_learning_path: Any, step_number: int) -> Any:
    if enhanced_learning_path is None:
        return None
    enhancements = list(_get_field(enhanced_learning_path, "enhancements", []) or [])
    for enhancement in enhancements:
        if int(_get_field(enhancement, "step", -1) or -1) == step_number:
            return enhancement
    return None

def _bullet_panel(
    items: list[str],
    title: str,
    border_style: str,
    bullet_style: str,
) -> Panel:
    body = "\n".join(
        f"[{bullet_style}]-[/{bullet_style}] {escape(item)}"
        for item in items
    )
    return Panel(
        body,
        title=title,
        border_style=border_style,
        box=box.ROUNDED,
    )

def _clean_list_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text and text not in result:
                result.append(text)
        return result
    return []

def _unique_items(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = str(item)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result

def _ellipsis(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."

def _title_text(title: Any, url: Any) -> Text:
    clean_title = _ellipsis(str(title or "Untitled"), 140)
    text = Text(clean_title, style="bold white")
    text.overflow = "fold"

    cleaned_url = str(url or "").strip()
    if cleaned_url and " " not in cleaned_url:
        try:
            text.stylize(f"link={cleaned_url}")
        except Exception:
            pass

    return text

def _status_style(status: str) -> str:
    normalized = status.lower()
    if normalized == "success":
        return "green"
    if normalized == "partial":
        return "yellow"
    if normalized == "failed":
        return "red"
    if normalized == "running":
        return "blue"
    return "dim"

def _status_badge_style(status: str) -> str:
    normalized = status.lower()
    if normalized == "success":
        return "bold white on green"
    if normalized == "partial":
        return "bold black on yellow"
    if normalized == "failed":
        return "bold white on red"
    if normalized == "running":
        return "bold white on blue"
    if normalized == "pending":
        return "bold white on magenta"
    return "bold white on blue"

def _difficulty_style(difficulty: str) -> str:
    normalized = difficulty.lower()
    if normalized == "beginner":
        return "bold green"
    if normalized == "intermediate":
        return "bold yellow"
    if normalized == "advanced":
        return "bold red"
    return "dim"

def _score_style(value: Any) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "dim"
    if numeric >= 0.75:
        return "green"
    if numeric >= 0.5:
        return "yellow"
    return "red"

def _format_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)

def _format_latency(value: Any) -> str:
    if value is None:
        return "-"
    try:
        ms = float(value)
    except Exception:
        return str(value)
    if ms >= 60000:
        minutes = int(ms // 60000)
        seconds = (ms % 60000) / 1000
        return f"{minutes}m {seconds:04.1f}s"
    if ms >= 1000:
        return f"{ms / 1000:.1f}s"
    return f"{ms:.0f} ms"

def _short_timestamp(value: Any) -> str:
    text = str(value or "")
    if not text:
        return "-"
    return text[:19].replace("T", " ")

def _parse_json_safe(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default

def _default_database_path() -> Any:
    return get_data_directory() / "research_agent.db"

def _enum_value(value: Any) -> str:
    if value is None:
        return ""
    return str(getattr(value, "value", value))

def _open_path(path: str) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        console.print(f"Could not open: {path}")

if __name__ == "__main__":
    app()