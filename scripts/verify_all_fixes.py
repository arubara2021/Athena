from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


console = Console()


def _check(condition: bool, label: str, detail: str = "") -> tuple[bool, str, str]:
    return condition, label, detail


async def run_verification() -> int:
    results: list[tuple[bool, str, str]] = []

    results.append(_verify_observe())
    results.append(_verify_pipeline_guard())
    results.append(await _verify_shared_manager())
    results.append(_verify_cost_tracker())
    results.append(_verify_ensemble())
    results.append(_verify_path_padding())
    results.append(_verify_query_stripping())
    results.append(await _verify_live_pipeline())

    return _render_report(results)


def _verify_observe() -> tuple[bool, str, str]:
    try:
        from agent.loop.observe import ObservePhase
        from core.models import Difficulty

        phase = ObservePhase()
        item = {
            "source_id": "x",
            "title": "Test",
            "url": "https://example.com/x",
            "platform": "wikipedia",
            "source_type": "documentation",
            "difficulty": "beginner",
            "abstract": "abs",
            "year": 2023,
            "citation_count": 5,
            "authors": ["A", "B"],
            "metadata": {"k": "v"},
        }
        source = phase._dict_to_source(item)
        ok = (
            source is not None
            and source.difficulty == Difficulty.BEGINNER
            and source.year == 2023
            and source.citation_count == 5
            and source.authors == ["A", "B"]
            and source.metadata == {"k": "v"}
        )
        return _check(ok, "BUG 1 — ObservePhase preserves fields", "OK" if ok else "FAILED")
    except Exception as exc:
        return _check(False, "BUG 1 — ObservePhase preserves fields", str(exc))


def _verify_pipeline_guard() -> tuple[bool, str, str]:
    try:
        from core.pipeline import ResearchPipeline

        pipeline = ResearchPipeline.__new__(ResearchPipeline)

        class _Q:
            level_profile = "mixed"
            level = "beginner to advanced"
            goal = "learn"
            topic = "x"

        result = pipeline._is_beginner_request(_Q())
        ok = result is False
        return _check(ok, "BUG 2 — Pipeline guard respects level_profile", "OK" if ok else "FAILED")
    except Exception as exc:
        return _check(False, "BUG 2 — Pipeline guard respects level_profile", str(exc))


async def _verify_shared_manager() -> tuple[bool, str, str]:
    try:
        from llm.provider import LLMProviderManager

        LLMProviderManager._shared_instance = None
        m1 = LLMProviderManager.get_shared()
        m2 = LLMProviderManager.get_shared()
        ok = m1 is m2

        await LLMProviderManager.close_shared()

        return _check(
            ok,
            "BUG 12 — Shared manager singleton",
            "OK" if ok else "FAILED",
        )
    except Exception as exc:
        return _check(
            False,
            "BUG 12 — Shared manager singleton",
            str(exc),
        )


def _verify_cost_tracker() -> tuple[bool, str, str]:
    try:
        from agent.token.cost_tracker import CostTracker

        tracker = CostTracker(task_id="verify", budget=10000)
        tracker.on_token_usage(
            provider="x",
            model="y",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_ms=100.0,
        )
        ok = tracker.total_tokens_used == 150
        return _check(ok, "BUG 3 — CostTracker hook records tokens", "OK" if ok else "FAILED")
    except Exception as exc:
        return _check(False, "BUG 3 — CostTracker hook records tokens", str(exc))


def _verify_ensemble() -> tuple[bool, str, str]:
    try:
        from llm.ensemble import EnsembleEngine

        engine = EnsembleEngine.__new__(EnsembleEngine)

        a = {"rankings": [{"source_id": "s1", "score": 0.9}]}
        b = {"rankings": [{"source_id": "s1", "score": 0.3}]}

        fp_a = engine._vote_fingerprint(a)
        fp_b = engine._vote_fingerprint(b)

        ok = fp_a.strict != fp_b.strict and fp_a.loose == fp_b.loose
        return _check(ok, "BUG 11 — Fingerprint separates strict vs loose", "OK" if ok else "FAILED")
    except Exception as exc:
        return _check(False, "BUG 11 — Fingerprint separates strict vs loose", str(exc))


def _verify_path_padding() -> tuple[bool, str, str]:
    try:
        from ranking.learning_path_builder import MINIMUM_STEPS, PADDING_MARKER

        ok = MINIMUM_STEPS == 3 and PADDING_MARKER == "[padded]"
        return _check(ok, "BUG 4 — Padding marker exported", "OK" if ok else "FAILED")
    except Exception as exc:
        return _check(False, "BUG 4 — Padding marker exported", str(exc))


def _verify_query_stripping() -> tuple[bool, str, str]:
    try:
        from search.query_intelligence import (
            QueryAnalysis,
            QueryIntent,
            QueryIntelligence,
        )

        qi = QueryIntelligence(use_llm=False)
        analysis = QueryAnalysis(
            original_topic="basic chemistry",
            cleaned_topic="basic chemistry",
            corrected_topic="chemistry basics advanced",
            primary_concept="chemistry fundamentals",
            keywords=["chemistry", "advanced"],
            intent=QueryIntent.LEARN,
            level="mixed",
            learner_level="mixed",
            level_priority=["beginner", "intermediate", "advanced"],
        )
        cleaned = qi._strip_llm_injected_level_words(analysis)
        ok = (
            "advanced" not in cleaned.corrected_topic
            and "fundamentals" not in cleaned.primary_concept
        )
        return _check(ok, "BUG 7 — LLM level words stripped", "OK" if ok else "FAILED")
    except Exception as exc:
        return _check(False, "BUG 7 — LLM level words stripped", str(exc))


async def _verify_live_pipeline() -> tuple[bool, str, str]:
    try:
        from core.pipeline import ResearchPipeline
        from core.schemas import SearchQuerySchema
        from ranking.consensus_ranker import ConsensusRanker
        from ranking.learning_path_builder import LearningPathBuilder
        from search.orchestrator import SearchOrchestrator

        pipeline = ResearchPipeline(
            search_orchestrator=SearchOrchestrator(
                use_llm_expansion=False,
            ),
            consensus_ranker=ConsensusRanker(use_llm=False),
            learning_path_builder=LearningPathBuilder(use_llm=False),
            enable_rag=False,
        )

        query = SearchQuerySchema(
            topic="basic chemistry",
            goal="Learn basics",
            level="beginner to advanced",
            max_results=5,
        )

        async with pipeline:
            result = await pipeline.run(query)

        ok = (
            result.status.value in {"success", "partial"}
            and len(result.sources) > 0
        )
        return _check(
            ok,
            "E2E — Pipeline survives mixed-profile query",
            f"sources={len(result.sources)} status={result.status.value}",
        )
    except Exception as exc:
        return _check(False, "E2E — Pipeline survives mixed-profile query", str(exc))


def _render_report(results: list[tuple[bool, str, str]]) -> int:
    table = Table(
        title="ATHENA FIX VERIFICATION",
        show_lines=True,
    )
    table.add_column("STATUS", justify="center", style="bold")
    table.add_column("CHECK", style="white")
    table.add_column("DETAIL", style="dim")

    passed = 0
    for ok, label, detail in results:
        status = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
        table.add_row(status, label, detail)
        if ok:
            passed += 1

    console.print(table)

    total = len(results)
    summary = f"{passed}/{total} checks passed"

    if passed == total:
        console.print(
            Panel(
                f"[bold green]ALL CHECKS PASSED[/bold green]  ·  {summary}",
                border_style="green",
            )
        )
        return 0

    console.print(
        Panel(
            f"[bold red]SOME CHECKS FAILED[/bold red]  ·  {summary}",
            border_style="red",
        )
    )
    return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_verification())
    sys.exit(exit_code)