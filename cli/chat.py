from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Iterable

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.rule import Rule
from rich.status import Status
from rich.table import Table
from rich.text import Text

from cli import banner as banner_module
from cli import components as comp
from cli import errors as errors_module
from cli import format as fmt
from cli import lazy
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
    return [value]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _state_path() -> Path:
    return Path(str(lazy.get_data_directory())) / "chat_state.json"


def _load_last_run_id() -> str | None:
    path = _state_path()
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    value = raw.get("last_run_id")
    return str(value) if value else None


def _save_last_run_id(run_id: str) -> None:
    path = _state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"last_run_id": str(run_id)}, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _coerce_source(item: Any) -> Any:
    from core.models import Source
    if not isinstance(item, dict):
        return None
    nested = item.get("source")
    if isinstance(nested, dict):
        item = nested
    try:
        return Source.model_validate(item)
    except Exception:
        return None


async def _rebuild_index(store: Any) -> int:
    from rag.embedder import Embedder as _Embedder

    output_dir = Path(str(lazy.get_output_directory()))
    json_dir = output_dir / "json"
    if not json_dir.exists():
        return 0

    try:
        files = sorted(
            json_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:100]
    except Exception:
        return 0

    sources: list[Any] = []
    run_ids: list[str] = []
    seen: set[str] = set()

    for file in files:
        try:
            data = json.loads(
                file.read_text(encoding="utf-8", errors="replace")
            )
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        run_id = str(
            data.get("request_id") or data.get("agent_id") or file.stem
        )
        for item in (
            _as_list(data.get("sources"))
            + _as_list(data.get("ranked_sources"))
        ):
            source = _coerce_source(item)
            if source is None:
                continue
            sid = str(getattr(source, "source_id", "") or "")
            if not sid or sid in seen:
                continue
            seen.add(sid)
            sources.append(source)
            run_ids.append(run_id)

    if not sources:
        return 0

    async with _Embedder() as embedder:
        embeddings = await embedder.embed_sources(sources)

    store.clear()

    grouped_sources: dict[str, list[Any]] = {}
    grouped_embeddings: dict[str, list[list[float]]] = {}
    for index, source in enumerate(sources):
        rid = run_ids[index] if index < len(run_ids) else "unknown"
        emb = embeddings[index] if index < len(embeddings) else []
        grouped_sources.setdefault(rid, []).append(source)
        grouped_embeddings.setdefault(rid, []).append(emb)

    total = 0
    for rid, run_sources in grouped_sources.items():
        total += store.add_sources(
            run_sources, grouped_embeddings[rid], run_id=rid
        )
    return total


def _answer_to_dict(answer: Any) -> dict[str, Any]:
    citations: list[dict[str, Any]] = []
    for item in _as_list(getattr(answer, "citations", None)):
        citations.append(
            {
                "source_id": str(getattr(item, "source_id", "") or ""),
                "title": str(getattr(item, "title", "") or ""),
                "url": getattr(item, "url", None),
                "score": _to_float(getattr(item, "score", 0.0)),
                "snippet": str(getattr(item, "snippet", "") or ""),
            }
        )

    ladder: list[dict[str, Any]] = []
    for step in _as_list(getattr(answer, "fallback_ladder", None)):
        ladder.append(
            {
                "stage": str(getattr(step, "stage", "") or ""),
                "reason": str(getattr(step, "reason", "") or ""),
                "action": str(getattr(step, "action", "") or ""),
                "severity": str(getattr(step, "severity", "info") or "info"),
            }
        )

    return {
        "question": str(getattr(answer, "question", "") or ""),
        "answer": str(getattr(answer, "answer", "") or ""),
        "citations": citations,
        "confidence": _to_float(getattr(answer, "confidence", 0.0)),
        "fallback": bool(getattr(answer, "fallback", False)),
        "llm_used": bool(getattr(answer, "llm_used", False)),
        "strategy": str(getattr(answer, "strategy", "") or ""),
        "session_id": str(getattr(answer, "session_id", "") or ""),
        "answer_style": str(getattr(answer, "answer_style", "") or ""),
        "diagnostics": dict(getattr(answer, "diagnostics", {}) or {}),
        "fallback_ladder": ladder,
    }


def _chunk_text(text: str, size: int) -> Iterable[str]:
    total = len(text)
    index = 0
    while index < total:
        yield text[index:index + size]
        index += size


def _animate_reveal(
    text: str,
    console: Console,
    style: str = "athena.text",
) -> None:
    payload = str(text or "")
    if not payload:
        console.print()
        return
    total = len(payload)
    size = max(2, total // 400)
    raw_delay = 1.5 / max(1.0, total / size)
    delay = max(0.001, min(0.015, raw_delay))
    try:
        for chunk in _chunk_text(payload, size):
            console.print(Text(chunk, style=style), end="", soft_wrap=True)
            time.sleep(delay)
    except KeyboardInterrupt:
        console.print()
        raise
    console.print()


def _render_citations(citations: list[Any]) -> None:
    console = _console()
    if not citations:
        return
    console.print(Rule(style="athena.border"))
    console.print()
    table = Table(
        box=None,
        show_header=True,
        header_style="athena.heading",
        padding=(0, 2),
        expand=False,
    )
    table.add_column("#", style="athena.muted", justify="right")
    table.add_column("TITLE", style="athena.value", overflow="fold")
    table.add_column("SCORE", style="athena.metric", justify="right")
    for index, item in enumerate(citations[:8], start=1):
        title = str(getattr(item, "title", "") or "Untitled")
        score = _to_float(getattr(item, "score", 0.0))
        table.add_row(str(index), fmt.ellipsis(title, 90), f"{score:.2f}")
    console.print(table)


def _render_meta(answer: Any) -> None:
    console = _console()
    confidence = _to_float(getattr(answer, "confidence", 0.0))
    fallback = bool(getattr(answer, "fallback", False))
    llm_used = bool(getattr(answer, "llm_used", False))
    strategy = str(getattr(answer, "strategy", "") or "")
    session_id = str(getattr(answer, "session_id", "") or "")

    console.print(Rule(style="athena.border"))
    console.print()

    line = Text()
    line.append("  ")
    line.append("confidence ", style="athena.label")
    line.append(f"{confidence:.2f}", style="athena.metric")
    line.append("   ")
    line.append("llm ", style="athena.label")
    line.append(
        "yes" if llm_used else "no",
        style="athena.success" if llm_used else "athena.muted",
    )
    if fallback:
        line.append("   ")
        line.append("fallback", style="athena.warning")
    if strategy:
        line.append("   ")
        line.append(strategy, style="athena.muted")
    if session_id:
        line.append("   ")
        line.append(session_id, style="athena.muted")
    console.print(line)

    if strategy in ("catalog", "catalog_snapshot"):
        console.print(
            Text(
                "  sourced from catalog metadata, not document passages",
                style="athena.muted",
            )
        )


def _render_diagnostics(answer: Any) -> None:
    console = _console()
    diagnostics = dict(getattr(answer, "diagnostics", {}) or {})
    if not diagnostics:
        return

    console.print()
    console.print(Rule(style="athena.border"))
    console.print()

    table = Table(
        box=None,
        show_header=False,
        padding=(0, 2),
        expand=False,
    )
    table.add_column(style="athena.label", justify="right", no_wrap=True)
    table.add_column(style="athena.value", overflow="fold")

    for key, value in diagnostics.items():
        if key in ("store_path", "engine_diagnostics", "topic"):
            if isinstance(value, dict):
                text = json.dumps(value, indent=2, default=str)
                if len(text) > 400:
                    text = text[:400] + "..."
                table.add_row(str(key).upper(), text)
            continue
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False, default=str)
            if len(text) > 300:
                text = text[:300] + "..."
            table.add_row(str(key).upper(), text)
        else:
            table.add_row(str(key).upper(), str(value))

    console.print(
        comp.panel(table, title="DIAGNOSTICS", border="athena.border")
    )

    ladder = list(getattr(answer, "fallback_ladder", []) or [])
    fallback_panel = comp.fallback_panel(ladder)
    if fallback_panel is not None:
        console.print(fallback_panel)


def _render_answer(answer: Any, stream: bool, explain: bool) -> None:
    console = _console()
    text = str(getattr(answer, "answer", "") or "")
    if not text:
        console.print(comp.empty_panel("No answer was produced.", "ANSWER"))
        return

    console.print()
    if stream:
        _animate_reveal(text, console)
    else:
        console.print(text)

    citations = list(getattr(answer, "citations", []) or [])
    if citations:
        _render_citations(citations)

    _render_meta(answer)

    if explain:
        _render_diagnostics(answer)


def _print_help() -> None:
    console = _console()
    rows = [
        ("/help", "show this help"),
        ("/new, /reset", "start a fresh session"),
        ("/clear", "clear the current session history"),
        ("/history", "show turns in this session"),
        ("/sessions", "list sessions on disk"),
        ("/save PATH", "export the session to a file"),
        ("/rebuild", "rebuild the vector index from output JSON"),
        ("/explain", "toggle diagnostics on every answer"),
        ("/quit, /exit", "leave the chat"),
    ]
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style="athena.metric", justify="right", no_wrap=True)
    grid.add_column(style="athena.muted")
    for cmd, desc in rows:
        grid.add_row(cmd, desc)
    console.print(
        comp.panel(grid, title="CHAT COMMANDS", border="athena.primary")
    )


def _print_repl_banner(session_id: str, run_id: str | None) -> None:
    console = _console()
    rows = [
        ("SESSION", session_id),
        ("RUN FILTER", run_id or "all runs"),
        ("COMMANDS", "/help  ·  /quit"),
    ]
    console.print(
        comp.panel(
            comp.kv_grid(rows),
            title="INTERACTIVE CHAT",
            border="athena.primary",
        )
    )


async def _open_engine(
    store: Any,
    top_k: int,
    session_id: str | None,
    source_diversity: int,
) -> Any:
    engine = lazy.get_chat_engine(
        vector_store=store,
        top_k=top_k,
        session_id=session_id,
        source_diversity=source_diversity,
    )
    await engine.__aenter__()
    return engine


async def _ask_with_status(
    engine: Any,
    question: str,
    run_id: str | None,
    top_k: int,
) -> Any:
    console = _console()
    status = Status(
        "[athena.accent]athena is thinking...[/]",
        console=console,
        spinner="dots12",
        spinner_style="athena.accent",
    )
    try:
        status.start()
    except Exception:
        status = None
    try:
        return await engine.ask_safe(
            question, run_id=run_id, top_k=top_k
        )
    finally:
        if status is not None:
            try:
                status.stop()
            except Exception:
                pass


async def _run_single(
    question: str,
    run_id: str | None,
    top_k: int,
    rebuild: bool,
    source_diversity: int,
    session_id: str | None,
    explain: bool,
    stream: bool,
    json_output: bool,
) -> int:
    console = _console()
    store = None
    engine = None
    try:
        store = lazy.get_vector_store()
        if rebuild:
            rebuilt = await _rebuild_index(store)
            console.print(
                comp.success_panel(
                    f"Index rebuilt with {rebuilt} documents.",
                    title="REBUILD",
                )
            )

        engine = await _open_engine(store, top_k, session_id, source_diversity)
        answer = await _ask_with_status(
            engine, question, run_id, top_k
        )

        if json_output:
            typer.echo(
                json.dumps(
                    _answer_to_dict(answer),
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )
        else:
            _render_answer(answer, stream=stream, explain=explain)

        return 0
    except KeyboardInterrupt:
        console.print(comp.warning_panel("Cancelled.", title="CANCELLED"))
        return 1
    except Exception as exc:
        errors_module.print_error(exc)
        return 1
    finally:
        if engine is not None:
            try:
                await engine.aclose()
            except Exception:
                pass
        if store is not None:
            try:
                store.close()
            except Exception:
                pass


async def _handle_slash(
    raw: str,
    engine: Any,
    store: Any,
    top_k: int,
    source_diversity: int,
    explain: bool,
    json_output: bool,
) -> tuple[str, Any, bool]:
    parts = raw.strip().split(maxsplit=1)
    command = parts[0].lower().strip()
    argument = parts[1].strip() if len(parts) > 1 else ""
    console = _console()

    if command in ("/quit", "/exit", "/q"):
        return "quit", engine, explain

    if command in ("/new", "/reset"):
        try:
            await engine.aclose()
        except Exception:
            pass
        engine = await _open_engine(
            store, top_k, None, source_diversity
        )
        console.print(
            comp.success_panel(
                f"New session: {engine.session_id}", title="SESSION"
            )
        )
        return "ok", engine, explain

    if command == "/clear":
        try:
            engine.conversation.clear()
            console.print(
                comp.success_panel("Session history cleared.", title="CLEAR")
            )
        except Exception as exc:
            errors_module.print_error(exc)
        return "ok", engine, explain

    if command == "/history":
        try:
            turns = engine.conversation.get_turns()
        except Exception:
            turns = []
        if not turns:
            console.print(comp.empty_panel("No history yet.", "HISTORY"))
            return "ok", engine, explain
        table = Table(
            box=None,
            show_header=True,
            header_style="athena.heading",
            padding=(0, 2),
        )
        table.add_column("#", style="athena.muted", justify="right")
        table.add_column("ROLE", style="athena.label")
        table.add_column("CONTENT", style="athena.value", overflow="fold")
        for index, turn in enumerate(turns, start=1):
            table.add_row(
                str(index),
                str(getattr(turn, "role", "")).upper(),
                fmt.ellipsis(str(getattr(turn, "content", "") or ""), 200),
            )
        console.print(table)
        return "ok", engine, explain

    if command == "/sessions":
        try:
            from rag.conversation import list_sessions
            sessions = list_sessions()
        except Exception:
            sessions = []
        if not sessions:
            console.print(comp.empty_panel("No sessions on disk.", "SESSIONS"))
            return "ok", engine, explain
        table = Table(
            box=None,
            show_header=True,
            header_style="athena.heading",
            padding=(0, 2),
        )
        table.add_column("SESSION", style="athena.value")
        table.add_column("TURNS", style="athena.metric", justify="right")
        table.add_column("UPDATED", style="athena.muted")
        for session in sessions[:20]:
            table.add_row(
                str(session.get("session_id", "")),
                str(session.get("turn_count", 0)),
                fmt.format_short_timestamp(
                    session.get("updated_at") or session.get("created_at")
                ),
            )
        console.print(table)
        return "ok", engine, explain

    if command == "/save":
        target = argument or f"chat_{engine.session_id}.json"
        try:
            path = engine.conversation.export(target)
            console.print(
                comp.success_panel(f"Saved to {path}", title="SAVE")
            )
        except Exception as exc:
            errors_module.print_error(exc)
        return "ok", engine, explain

    if command == "/rebuild":
        try:
            rebuilt = await _rebuild_index(store)
            console.print(
                comp.success_panel(
                    f"Index rebuilt with {rebuilt} documents.",
                    title="REBUILD",
                )
            )
        except Exception as exc:
            errors_module.print_error(exc)
        return "ok", engine, explain

    if command == "/explain":
        new_state = not explain
        console.print(
            comp.success_panel(
                f"Explain mode {'on' if new_state else 'off'}.",
                title="EXPLAIN",
            )
        )
        return "ok", engine, new_state

    if command == "/help":
        _print_help()
        return "ok", engine, explain

    console.print(
        comp.warning_panel(f"Unknown command: {command}", title="UNKNOWN")
    )
    return "ok", engine, explain


async def _run_interactive(
    run_id: str | None,
    top_k: int,
    rebuild: bool,
    source_diversity: int,
    session_id: str | None,
    explain: bool,
    stream: bool,
    json_output: bool,
    all_runs: bool,
) -> int:
    console = _console()
    store = None
    engine = None
    resolved_run_id = None if all_runs else (run_id or _load_last_run_id())
    try:
        store = lazy.get_vector_store()
        if rebuild:
            rebuilt = await _rebuild_index(store)
            console.print(
                comp.success_panel(
                    f"Index rebuilt with {rebuilt} documents.",
                    title="REBUILD",
                )
            )

        engine = await _open_engine(store, top_k, session_id, source_diversity)
        _print_repl_banner(engine.session_id, resolved_run_id)

        empty_streak = 0
        while True:
            try:
                raw = Prompt.ask("[bold #c084fc]you[/]")
            except KeyboardInterrupt:
                break
            except EOFError:
                empty_streak += 1
                if empty_streak >= 3:
                    break
                continue

            text = str(raw or "").strip()
            if not text:
                empty_streak += 1
                if empty_streak >= 3:
                    console.print(
                        Text(
                            "Type a question, /help, or /quit.",
                            style="athena.muted",
                        )
                    )
                    empty_streak = 0
                continue

            empty_streak = 0

            if text.startswith("/"):
                action, engine, explain = await _handle_slash(
                    text,
                    engine,
                    store,
                    top_k,
                    source_diversity,
                    explain,
                    json_output,
                )
                if action == "quit":
                    break
                continue

            try:
                answer = await _ask_with_status(
                    engine, text, resolved_run_id, top_k
                )
            except KeyboardInterrupt:
                break
            except Exception as exc:
                errors_module.print_error(exc, compact=True)
                continue

            if json_output:
                typer.echo(
                    json.dumps(
                        _answer_to_dict(answer),
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )
                )
            else:
                _render_answer(answer, stream=stream, explain=explain)

            if resolved_run_id:
                _save_last_run_id(resolved_run_id)

        console.print()
        console.print(
            comp.success_panel(
                "Session closed. All runs saved.", title="ATHENA"
            )
        )
        return 0
    except KeyboardInterrupt:
        console.print(comp.warning_panel("Cancelled.", title="CANCELLED"))
        return 1
    except Exception as exc:
        errors_module.print_error(exc)
        return 1
    finally:
        if engine is not None:
            try:
                await engine.aclose()
            except Exception:
                pass
        if store is not None:
            try:
                store.close()
            except Exception:
                pass


def chat(
    question: str = typer.Argument(
        None, help="Question to ask over saved research."
    ),
    run_id: str = typer.Option(
        "", "--run-id", help="Limit chat to one saved run."
    ),
    all_runs: bool = typer.Option(False, "--all-runs/--no-all-runs"),
    top_k: int = typer.Option(5, "--top-k", "-k", min=1, max=20),
    source_diversity: int = typer.Option(
        0, "--diversity", "-d", min=0, max=10
    ),
    session_id: str = typer.Option("", "--session-id"),
    rebuild_index: bool = typer.Option(
        False, "--rebuild-index/--no-rebuild-index"
    ),
    stream: bool = typer.Option(True, "--stream/--no-stream"),
    explain: bool = typer.Option(False, "--explain/--no-explain"),
    interactive: bool = typer.Option(
        False, "--interactive/--no-interactive"
    ),
    json_output: bool = typer.Option(False, "--json/--no-json"),
) -> None:
    console = _console()
    resolved_run_id = run_id.strip() or None
    resolved_session = session_id.strip() or None

    if interactive:
        banner_module.print_title("chat", subtitle="interactive")
        code = asyncio.run(
            _run_interactive(
                run_id=resolved_run_id,
                top_k=top_k,
                rebuild=rebuild_index,
                source_diversity=source_diversity,
                session_id=resolved_session,
                explain=explain,
                stream=stream,
                json_output=json_output,
                all_runs=all_runs,
            )
        )
        raise typer.Exit(code=code)

    if not question:
        console.print(
            comp.warning_panel(
                "Provide a question, or run with --interactive.",
                title="NO QUESTION",
            )
        )
        raise typer.Exit(code=1)

    banner_module.print_title("chat", subtitle=fmt.ellipsis(question, 60))

    resolved_for_single = (
        None if all_runs else (resolved_run_id or _load_last_run_id())
    )

    code = asyncio.run(
        _run_single(
            question=question,
            run_id=resolved_for_single,
            top_k=top_k,
            rebuild=rebuild_index,
            source_diversity=source_diversity,
            session_id=resolved_session,
            explain=explain,
            stream=stream,
            json_output=json_output,
        )
    )

    if resolved_for_single and not all_runs:
        _save_last_run_id(resolved_for_single)

    raise typer.Exit(code=code)


def register(app: typer.Typer) -> None:
    app.command(
        name="chat", help="Ask questions over saved research."
    )(chat)