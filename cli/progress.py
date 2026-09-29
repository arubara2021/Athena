from __future__ import annotations

import threading
import time
from typing import Any

from rich.console import Group
from rich.table import Table
from rich.text import Text

from cli import components as comp
from cli import format as fmt


_STAGES: tuple[str, ...] = (
    "PLAN",
    "SEARCH",
    "RANK",
    "SUMMARIZE",
    "PATH",
    "ENHANCE",
    "RAG",
    "DONE",
)

_EVENT_STAGE: dict[str, str] = {
    "pipeline.started": "PLAN",
    "agent.started": "PLAN",
    "agent.plan_created": "PLAN",
    "agent.thinking": "PLAN",
    "agent.acting": "PLAN",
    "agent.observing": "PLAN",
    "agent.reflecting": "PLAN",
    "pipeline.search_completed": "SEARCH",
    "agent.loop_started": "SEARCH",
    "agent.step_completed": "SEARCH",
    "agent.tool_called": "SEARCH",
    "agent.tool_completed": "SEARCH",
    "agent.tool_failed": "SEARCH",
    "pipeline.ranking_completed": "RANK",
    "pipeline.source_summaries_completed": "SUMMARIZE",
    "agent.summaries_completed": "SUMMARIZE",
    "pipeline.learning_path_completed": "PATH",
    "agent.learning_path_completed": "PATH",
    "pipeline.path_enhanced": "ENHANCE",
    "agent.path_enhanced": "ENHANCE",
    "pipeline.rag_indexing_started": "RAG",
    "pipeline.rag_indexed": "RAG",
    "pipeline.completed": "DONE",
    "pipeline.failed": "DONE",
    "agent.synthesizing": "DONE",
    "agent.completed": "DONE",
    "agent.failed": "DONE",
}

_FAIL_EVENTS = frozenset({"pipeline.failed", "agent.failed"})
_DONE_EVENTS = frozenset({"pipeline.completed", "agent.completed"})

_SPIN_FRAMES: tuple[str, ...] = ("|", "/", "-", "\\")


def _spinner_char() -> str:
    try:
        frame = int(time.monotonic() * 10) % len(_SPIN_FRAMES)
    except Exception:
        frame = 0
    return _SPIN_FRAMES[frame]


class ProgressDashboard:
    def __init__(
        self,
        mode: str = "pipeline",
        title: str = "",
        goal: str = "",
        target_sources: int = 20,
    ) -> None:
        self._mode = str(mode or "pipeline").upper()
        self._title = str(title or "")
        self._goal = str(goal or "")
        try:
            self._target_sources = max(1, int(target_sources or 20))
        except (TypeError, ValueError):
            self._target_sources = 20
        self._start = time.perf_counter()
        self._lock = threading.RLock()
        self._stage = "PLAN"
        self._stage_history: list[str] = ["PLAN"]
        self._status = "RUNNING"
        self._sources = 0
        self._ranked = 0
        self._summaries = 0
        self._steps = 0
        self._tokens = 0
        self._budget = 0
        self._errors = 0
        self._warnings = 0
        self._iteration = 0
        self._max_iterations = 0
        self._last_detail = ""
        self._last_event = ""

    @property
    def stage(self) -> str:
        with self._lock:
            return self._stage

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def elapsed(self) -> float:
        return max(0.0, time.perf_counter() - self._start)

    def update(self, name: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._apply_event(name, payload or {})

    async def handle_event(self, event: Any) -> None:
        try:
            name = str(getattr(event, "name", "") or "")
            payload = getattr(event, "payload", None)
            if not isinstance(payload, dict):
                payload = {}
            if not name:
                return
            self.update(name, payload)
        except Exception:
            return

    def absorb_result(self, result: Any) -> None:
        with self._lock:
            try:
                status_value = getattr(result, "status", None)
                status_text = str(
                    getattr(status_value, "value", status_value) or ""
                ).strip().lower()
            except Exception:
                status_text = ""

            if status_text in ("success", "completed", "done"):
                self._status = "DONE"
            elif status_text in ("failed", "error"):
                self._status = "FAILED"
            else:
                self._status = "DONE"

            ranked = getattr(result, "ranked_sources", None)
            if ranked is not None:
                try:
                    self._ranked = len(list(ranked))
                except Exception:
                    pass

            summaries = getattr(result, "source_summaries", None)
            if summaries is None:
                summaries = getattr(result, "summaries", None)
            if summaries is not None:
                try:
                    self._summaries = len(list(summaries))
                except Exception:
                    pass

            sources = getattr(result, "sources", None)
            if sources is None:
                sources = getattr(result, "findings", None)
            if sources is not None:
                try:
                    self._sources = len(list(sources))
                except Exception:
                    pass

            path = getattr(result, "learning_path", None)
            if path is not None:
                try:
                    self._steps = len(
                        list(getattr(path, "steps", []) or [])
                    )
                except Exception:
                    pass

            used = getattr(result, "total_tokens_used", None)
            if used is not None:
                try:
                    self._tokens = int(used)
                except (TypeError, ValueError):
                    pass

            budget = getattr(result, "total_budget", None)
            if budget is not None:
                try:
                    self._budget = int(budget)
                except (TypeError, ValueError):
                    pass

            errors = getattr(result, "errors", None)
            if errors is not None:
                try:
                    self._errors = len(list(errors))
                except Exception:
                    pass

            warnings = getattr(result, "warnings", None)
            if warnings is not None:
                try:
                    self._warnings = len(list(warnings))
                except Exception:
                    pass

            self._stage = "DONE"
            if "DONE" not in self._stage_history:
                self._stage_history.append("DONE")

    def _apply_event(self, name: str, payload: dict[str, Any]) -> None:
        self._last_event = name

        target = _EVENT_STAGE.get(name)
        if target is not None:
            if target not in self._stage_history:
                self._stage_history.append(target)
            self._stage = target

        if name in _FAIL_EVENTS:
            self._status = "FAILED"
            detail = str(payload.get("error") or "").strip()
            if detail:
                self._last_detail = fmt.ellipsis(detail, 140)

        if name in _DONE_EVENTS:
            self._status = "DONE"
            self._stage = "DONE"

        if name == "pipeline.search_completed":
            try:
                self._sources = int(
                    payload.get("total_sources") or self._sources
                )
            except (TypeError, ValueError):
                pass
        elif name == "pipeline.ranking_completed":
            try:
                self._ranked = int(
                    payload.get("total_ranked") or self._ranked
                )
            except (TypeError, ValueError):
                pass
        elif name in (
            "pipeline.source_summaries_completed",
            "agent.summaries_completed",
        ):
            try:
                self._summaries = int(
                    payload.get("total_summaries") or self._summaries
                )
            except (TypeError, ValueError):
                pass
        elif name in (
            "pipeline.learning_path_completed",
            "agent.learning_path_completed",
        ):
            try:
                self._steps = int(
                    payload.get("total_steps") or self._steps
                )
            except (TypeError, ValueError):
                pass
        elif name == "agent.loop_started":
            try:
                self._max_iterations = int(
                    payload.get("max_iterations") or self._max_iterations
                )
            except (TypeError, ValueError):
                pass
        elif name in ("agent.step_completed", "agent.iteration"):
            try:
                self._iteration = int(
                    payload.get("iteration_count")
                    or payload.get("iteration")
                    or self._iteration
                )
            except (TypeError, ValueError):
                pass
            try:
                self._sources = int(
                    payload.get("findings_count")
                    or payload.get("findings")
                    or self._sources
                )
            except (TypeError, ValueError):
                pass
            try:
                self._tokens = int(
                    payload.get("tokens_used") or self._tokens
                )
            except (TypeError, ValueError):
                pass
        elif name == "agent.budget_warning":
            try:
                self._tokens = int(
                    payload.get("tokens_used") or self._tokens
                )
            except (TypeError, ValueError):
                pass
            try:
                self._budget = int(
                    payload.get("total_budget") or self._budget
                )
            except (TypeError, ValueError):
                pass
        elif name == "agent.tool_failed":
            self._errors += 1
            detail = str(payload.get("error") or "").strip()
            if detail:
                self._last_detail = fmt.ellipsis(detail, 140)
        elif name == "pipeline.stage_failed":
            self._warnings += 1
            detail = str(payload.get("error") or "").strip()
            if detail:
                self._last_detail = fmt.ellipsis(detail, 140)
        elif name == "pipeline.rag_indexed":
            try:
                self._sources = int(
                    payload.get("documents_indexed") or self._sources
                )
            except (TypeError, ValueError):
                pass

    def _panel_title(self) -> Text:
        label = {
            "DONE": "COMPLETE",
            "FAILED": "FAILED",
        }.get(self._status, "RUNNING")
        text = Text()
        text.append(f" {label} ", style=f"bold black on {self._title_bg()}")
        text.append("  ")
        text.append(self._mode, style="athena.heading")
        text.append(" ")
        return text

    def _panel_border(self) -> str:
        if self._status == "DONE":
            return "athena.success"
        if self._status == "FAILED":
            return "athena.danger"
        return "athena.primary"

    def _title_bg(self) -> str:
        if self._status == "DONE":
            return "#34d399"
        if self._status == "FAILED":
            return "#dc2626"
        return "#c084fc"

    def _render_header(self) -> Text:
        elapsed = self.elapsed
        text = Text()
        text.append("  ")
        text.append(f" {self._mode} ", style=f"bold black on {self._title_bg()}")
        text.append("   ")

        if self._status == "RUNNING":
            text.append(_spinner_char(), style="athena.accent")
        elif self._status == "DONE":
            text.append("✓", style="athena.success")
        elif self._status == "FAILED":
            text.append("✗", style="athena.danger")

        text.append("  ")
        text.append(self._title or "athena", style="athena.heading")
        text.append("   ")
        text.append("·", style="athena.muted")
        text.append("   ")

        status_style = {
            "RUNNING": "athena.accent",
            "DONE": "athena.success",
            "FAILED": "athena.danger",
        }.get(self._status, "athena.muted")
        text.append(self._status, style=status_style)
        text.append("   ")
        text.append("·", style="athena.muted")
        text.append("   ")
        text.append(f"{elapsed:6.2f}s", style="athena.value")
        return text

    def _render_stages(self) -> Text:
        text = Text()
        text.append("  ")
        for index, stage in enumerate(_STAGES):
            if index:
                text.append("   ")
            reached = stage in self._stage_history
            active = stage == self._stage and self._status == "RUNNING"
            if stage == "DONE" and self._status == "FAILED":
                text.append(stage, style="bold white on #dc2626")
            elif stage == "DONE" and self._status == "DONE":
                text.append(stage, style="bold black on #34d399")
            elif active:
                text.append(stage, style="bold black on #c084fc")
            elif reached:
                text.append(stage, style="athena.success")
            else:
                text.append(stage, style="athena.muted")
        return text

    def _render_counters(self) -> Table:
        grid = Table.grid(padding=(0, 3), expand=False)
        grid.add_column(style="athena.label", justify="right", no_wrap=True)
        grid.add_column(style="athena.metric", justify="left", no_wrap=True)
        grid.add_column(style="athena.label", justify="right", no_wrap=True)
        grid.add_column(style="athena.metric", justify="left", no_wrap=True)
        grid.add_column(style="athena.label", justify="right", no_wrap=True)
        grid.add_column(style="athena.metric", justify="left", no_wrap=True)
        grid.add_row(
            "SOURCES", str(self._sources),
            "RANKED", str(self._ranked),
            "STEPS", str(self._steps),
        )
        grid.add_row(
            "SUMMARIES", str(self._summaries),
            "TOKENS", fmt.format_count(self._tokens),
            "ERRORS", str(self._errors),
        )
        return grid

    def __rich__(self) -> Any:
        with self._lock:
            header = self._render_header()
            stages = self._render_stages()
            counters = self._render_counters()

            rule = Text("  " + "─" * 72, style="athena.border")

            parts: list[Any] = [
                header,
                Text(""),
                stages,
                Text(""),
                rule,
                Text(""),
                counters,
            ]

            if self._last_detail:
                parts.append(Text(""))
                parts.append(
                    Text(f"  {self._last_detail}", style="athena.muted")
                )

            return comp.panel(
                Group(*parts),
                title=self._panel_title(),
                border=self._panel_border(),
                padding=(1, 2),
            )


def build_live(dashboard: ProgressDashboard) -> Any:
    from rich.live import Live
    from cli import theme as theme_module
    return Live(
        dashboard,
        console=theme_module.get_console(),
        refresh_per_second=10,
        transient=False,
    )