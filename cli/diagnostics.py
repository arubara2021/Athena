from __future__ import annotations

import asyncio
import importlib
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import typer
from rich.console import Console
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


def _safe(fn: Callable[[], Any], default: Any = None) -> Any:
    try:
        return fn()
    except Exception:
        return default


def _status_style(status: str) -> str:
    text = str(status or "").upper()
    if text == "OK":
        return "athena.success"
    if text == "WARN":
        return "athena.warning"
    if text == "FAIL":
        return "athena.danger"
    return "athena.info"


def _status_glyph(status: str) -> str:
    text = str(status or "").upper()
    if text == "OK":
        return "✓"
    if text == "WARN":
        return "!"
    if text == "FAIL":
        return "✗"
    return "i"


def _check_row(label: str, value: str, status: str) -> tuple[Text, Text, Text]:
    glyph = _status_glyph(status)
    label_cell = Text(str(label), style="athena.value")
    value_cell = Text(str(value), style="athena.muted")
    status_cell = Text(f" {glyph}  {status} ", style=_status_style(status))
    return label_cell, value_cell, status_cell


def _count_tools() -> int:
    registry = _safe(lazy.get_tool_registry)
    if registry is None:
        return 0
    for module_name, function_name in _safe(lazy.get_tool_registrations, ()) or ():
        try:
            module = importlib.import_module(module_name)
            func = getattr(module, function_name, None)
            if callable(func):
                func(registry)
        except Exception:
            continue
    try:
        return int(getattr(registry, "tool_count", 0) or 0)
    except Exception:
        return 0


def version() -> None:
    console = _console()
    banner_module.print_title("version", subtitle="system information")
    constants = _safe(lazy.get_constants)
    version_text = (
        str(getattr(constants, "APP_VERSION", "") or "unknown")
        if constants is not None
        else "unknown"
    )
    rows = [
        ("Athena", version_text),
        ("Python", sys.version.split()[0]),
        ("Executable", sys.executable),
        ("Platform", sys.platform),
    ]
    console.print(
        comp.panel(
            comp.kv_grid(rows),
            title="VERSION",
            border="athena.primary",
            padding=(1, 2),
        )
    )
    banner_module.print_closing("Version reported.")


def doctor() -> None:
    console = _console()
    banner_module.print_title("doctor", subtitle="environment checks")

    checks: list[tuple[str, str, str]] = []

    py = sys.version_info
    py_ok = py >= (3, 10)
    checks.append(
        ("PYTHON", f"{py.major}.{py.minor}.{py.micro}", "OK" if py_ok else "WARN")
    )

    settings = _safe(lazy.get_settings)
    if settings is None:
        checks.append(("SETTINGS", "failed to load", "FAIL"))
    else:
        checks.append(("SETTINGS", "loaded", "OK"))
        for label, attr in (
            ("OUTPUT DIR", "output_dir"),
            ("CACHE DIR", "cache_dir"),
            ("DATA DIR", "data_dir"),
            ("LOGS DIR", "logs_dir"),
        ):
            path_value = getattr(settings, attr, None)
            if path_value is None:
                checks.append((label, "-", "WARN"))
                continue
            target = Path(str(path_value))
            exists = target.exists()
            writable = False
            if exists:
                try:
                    probe = target / ".athena_write_test"
                    probe.write_text("ok", encoding="utf-8")
                    probe.unlink(missing_ok=True)
                    writable = True
                except Exception:
                    writable = False
            status = "OK" if exists and writable else "WARN"
            checks.append((label, str(target), status))

        try:
            provider_count = len(
                list(getattr(settings, "available_llm_providers", []) or [])
            )
        except Exception:
            provider_count = 0
        checks.append(
            (
                "LLM PROVIDERS",
                str(provider_count),
                "OK" if provider_count > 0 else "WARN",
            )
        )

        data_dir = Path(str(getattr(settings, "data_dir", "")))
        db_path = data_dir / "research_agent.db"
        checks.append(("DATABASE", str(db_path), "OK" if db_path.exists() else "INFO"))

        vector_path = data_dir / "vector_store.db"
        if vector_path.exists():
            try:
                size = vector_path.stat().st_size
                label = f"{vector_path} ({fmt.format_bytes(size)})"
                checks.append(("VECTOR STORE", label, "OK" if size > 0 else "WARN"))
            except Exception:
                checks.append(("VECTOR STORE", str(vector_path), "WARN"))
        else:
            checks.append(("VECTOR STORE", str(vector_path), "INFO"))

    try:
        import rich as _rich
        checks.append(("RICH", getattr(_rich, "__version__", "installed"), "OK"))
    except Exception as exc:
        checks.append(("RICH", str(exc), "FAIL"))

    try:
        import typer as _typer
        checks.append(("TYPER", getattr(_typer, "__version__", "installed"), "OK"))
    except Exception as exc:
        checks.append(("TYPER", str(exc), "FAIL"))

    tool_count = _count_tools()
    checks.append(
        (
            "TOOL REGISTRY",
            str(tool_count),
            "OK" if tool_count > 0 else "WARN",
        )
    )

    table = Table(
        box=None,
        show_header=True,
        header_style="athena.heading",
        padding=(0, 2),
        expand=True,
    )
    table.add_column("CHECK", style="athena.value", no_wrap=True)
    table.add_column("VALUE", style="athena.muted", overflow="fold")
    table.add_column("STATUS", justify="right", no_wrap=True)

    for label, value, status in checks:
        row = _check_row(label, value, status)
        table.add_row(*row)

    console.print(
        comp.panel(table, title="SYSTEM DOCTOR", border="athena.primary")
    )

    failures = [item for item in checks if item[2] == "FAIL"]
    warnings = [item for item in checks if item[2] == "WARN"]

    if failures:
        banner_module.print_closing(
            "Doctor found failures.", subtitle=f"{len(failures)} fail"
        )
    elif warnings:
        banner_module.print_closing(
            "Doctor finished with warnings.",
            subtitle=f"{len(warnings)} warn",
        )
    else:
        banner_module.print_closing("Doctor finished cleanly.")


def config() -> None:
    console = _console()
    banner_module.print_title("config", subtitle="active configuration")

    settings = _safe(lazy.get_settings)
    if settings is None:
        errors_module.print_error(RuntimeError("Settings could not be loaded."))
        raise typer.Exit(code=1)

    provider_rows: list[list[Any]] = []
    try:
        masked = dict(getattr(settings, "masked_provider_keys", {}) or {})
    except Exception:
        masked = {}
    try:
        available = {
            str(getattr(item, "value", item))
            for item in (getattr(settings, "available_llm_providers", []) or [])
        }
    except Exception:
        available = set()

    for provider, key in masked.items():
        is_available = str(provider) in available
        badge = (
            Text(" available ", style="bold black on #34d399")
            if is_available
            else Text(" missing ", style="bold white on #4b5563")
        )
        provider_rows.append(
            [str(provider), str(key or "-"), badge]
        )

    if provider_rows:
        console.print(
            comp.data_table(
                columns=(
                    ("provider", "PROVIDER"),
                    ("key", "KEY"),
                    ("status", "STATUS"),
                ),
                rows=provider_rows,
                title="LLM PROVIDERS",
            )
        )

    model_rows = [
        ("Primary Fast", getattr(settings, "primary_fast_model", "-")),
        ("Primary Strong", getattr(settings, "primary_strong_model", "-")),
        ("Judge", getattr(settings, "ensemble_judge_model", "-")),
        ("Code", getattr(settings, "code_model", "-")),
        ("Embedding", getattr(settings, "embedding_model", "-")),
    ]
    console.print(
        comp.panel(comp.kv_grid(model_rows), title="MODELS", border="athena.primary")
    )

    fast_chain = list(getattr(settings, "fast_model_chain", []) or [])
    strong_chain = list(getattr(settings, "strong_model_chain", []) or [])

    chain_rows: list[list[Any]] = []
    if fast_chain:
        for index, model in enumerate(fast_chain, start=1):
            chain_rows.append(["fast", str(index), str(model)])
    if strong_chain:
        for index, model in enumerate(strong_chain, start=1):
            chain_rows.append(["strong", str(index), str(model)])

    if chain_rows:
        console.print(
            comp.data_table(
                columns=(
                    ("tier", "TIER"),
                    ("rank", "#"),
                    ("model", "MODEL"),
                ),
                rows=chain_rows,
                title="FALLBACK CHAINS",
            )
        )

    dir_rows = [
        ("Output", getattr(settings, "output_dir", "-")),
        ("Cache", getattr(settings, "cache_dir", "-")),
        ("Data", getattr(settings, "data_dir", "-")),
        ("Logs", getattr(settings, "logs_dir", "-")),
    ]
    console.print(
        comp.panel(comp.kv_grid(dir_rows), title="DIRECTORIES", border="athena.primary")
    )

    banner_module.print_closing("Configuration loaded.")


def tools() -> None:
    console = _console()
    banner_module.print_title("tools", subtitle="registered capabilities")

    registry = _safe(lazy.get_tool_registry)
    if registry is None:
        errors_module.print_error(
            RuntimeError("Tool registry could not be imported.")
        )
        raise typer.Exit(code=1)

    skipped: list[str] = []
    for module_name, function_name in _safe(lazy.get_tool_registrations, ()) or ():
        try:
            module = importlib.import_module(module_name)
            func = getattr(module, function_name, None)
            if callable(func):
                func(registry)
            else:
                skipped.append(f"{module_name}: {function_name} not found")
        except Exception as exc:
            skipped.append(f"{module_name}: {exc}")

    try:
        summary = list(registry.to_summary())
    except Exception:
        summary = []

    if not summary:
        console.print(comp.empty_panel("No tools registered.", "TOOLS"))
        raise typer.Exit(code=0)

    rows: list[list[Any]] = []
    for item in summary:
        name = str(item.get("name", "") or "")
        category = str(item.get("category", "") or "")
        description = fmt.ellipsis(str(item.get("description", "") or ""), 100)
        tokens = str(item.get("token_cost_est", 0))
        requires_llm = bool(item.get("requires_llm"))
        llm_cell = (
            Text(" yes ", style="bold black on #c084fc")
            if requires_llm
            else Text(" no  ", style="dim")
        )
        rows.append([name, category, description, tokens, llm_cell])

    console.print(
        comp.data_table(
            columns=(
                ("name", "TOOL"),
                ("category", "CATEGORY"),
                ("description", "DESCRIPTION"),
                ("tokens", "TOKENS"),
                ("llm", "LLM"),
            ),
            rows=rows,
            title="REGISTERED TOOLS",
        )
    )

    if skipped:
        body = Text()
        for index, entry in enumerate(skipped[:8]):
            if index:
                body.append("\n")
            body.append("  • ", style="athena.warning")
            body.append(fmt.ellipsis(entry, 140), style="athena.muted")
        console.print(
            comp.panel(body, title="SKIPPED MODULES", border="athena.warning")
        )

    banner_module.print_closing(
        "Tool registry loaded.", subtitle=f"{len(rows)} tools"
    )


def stats(limit: int = typer.Option(100, "--limit", "-n", min=1, max=500)) -> None:
    console = _console()
    banner_module.print_title("stats", subtitle="recent run statistics")

    store = _safe(lazy.get_sqlite_store)
    if store is None:
        errors_module.print_error(RuntimeError("SQLite store unavailable."))
        raise typer.Exit(code=1)

    async def _load() -> list[dict[str, Any]]:
        await store.initialize()
        return await store.list_runs(limit)

    try:
        rows = asyncio.run(_load())
    except Exception as exc:
        errors_module.print_error(exc)
        raise typer.Exit(code=1)

    if not rows:
        console.print(comp.empty_panel("No saved runs yet.", "STATS"))
        banner_module.print_closing("Nothing to analyze.")
        return

    status_counter: Counter[str] = Counter(
        str(item.get("status", "unknown")).lower() for item in rows
    )
    total_sources = sum(_to_int(item.get("total_sources"), 0) for item in rows)
    total_ranked = sum(_to_int(item.get("total_ranked"), 0) for item in rows)
    total_steps = sum(_to_int(item.get("total_steps"), 0) for item in rows)
    latencies = [
        _to_float(item.get("latency_ms"), 0.0)
        for item in rows
        if item.get("latency_ms") is not None
    ]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    metric_rows = [
        ("Total Runs", len(rows)),
        ("Success", status_counter.get("success", 0)),
        ("Partial", status_counter.get("partial", 0)),
        ("Failed", status_counter.get("failed", 0)),
        ("Total Sources", total_sources),
        ("Total Ranked", total_ranked),
        ("Total Steps", total_steps),
        ("Avg Latency", fmt.format_latency(avg_latency)),
    ]
    console.print(
        comp.panel(
            comp.kv_grid(metric_rows),
            title="AGGREGATE STATS",
            border="athena.primary",
        )
    )

    if status_counter:
        max_count = max(status_counter.values() or [1])
        rows_status: list[list[Any]] = []
        for status, count in status_counter.most_common():
            rows_status.append(
                [status.upper(), str(count), comp.mini_bar(count, max_count)]
            )
        console.print(
            comp.data_table(
                columns=(
                    ("status", "STATUS"),
                    ("count", "COUNT"),
                    ("bar", "SHARE"),
                ),
                rows=rows_status,
                title="STATUS DISTRIBUTION",
            )
        )

    banner_module.print_closing(
        "Statistics computed.", subtitle=f"{len(rows)} runs"
    )


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def register(app: typer.Typer) -> None:
    app.command(name="version", help="Show Athena version and system info.")(version)
    app.command(name="doctor", help="Check environment, folders, and providers.")(doctor)
    app.command(name="config", help="Show active configuration.")(config)
    app.command(name="tools", help="List registered agent tools.")(tools)
    app.command(name="stats", help="Aggregate statistics for recent runs.")(stats)