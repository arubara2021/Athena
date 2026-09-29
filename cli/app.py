from __future__ import annotations

import logging
from typing import Any

import typer

from cli import banner as banner_module
from cli import commands as commands_module
from cli import errors as errors_module
from cli import runtime as runtime_module
from cli import theme as theme_module


app = typer.Typer(
    name="athena",
    help="ATHENA — Autonomous Research Agent CLI",
    no_args_is_help=False,
    rich_markup_mode="rich",
    add_completion=False,
)

commands_module.register(app)


def _apply_verbose() -> None:
    try:
        from utils.logger import setup_logger
        setup_logger(level=logging.DEBUG)
    except Exception:
        pass


def _apply_theme(name: str) -> None:
    try:
        theme_module.set_theme(name)
    except Exception:
        pass


def _apply_no_color(enabled: bool) -> None:
    if not enabled:
        return
    try:
        theme_module.set_no_color(True)
    except Exception:
        pass


def _show_root_help(ctx: typer.Context) -> None:
    banner_module.print_banner()
    console = theme_module.get_console()
    try:
        console.print(ctx.get_help())
    except Exception:
        typer.echo(ctx.get_help())


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    theme: str = typer.Option(
        "", "--theme", help="Theme: midnight, paper, mono."
    ),
    no_color: bool = typer.Option(
        False, "--no-color", help="Disable color output."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Verbose logging."
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Show tracebacks on error."
    ),
) -> None:
    if theme:
        _apply_theme(theme)
    _apply_no_color(no_color)

    if verbose:
        _apply_verbose()
    else:
        runtime_module.silence_console_logger()

    if debug:
        errors_module.set_debug(True)

    if ctx.invoked_subcommand is None:
        _show_root_help(ctx)
        raise typer.Exit(code=0)


def get_app() -> typer.Typer:
    return app


def run() -> None:
    try:
        app()
    except KeyboardInterrupt:
        try:
            theme_module.get_console().print()
        except Exception:
            pass
        raise typer.Exit(code=130)