from __future__ import annotations

from collections import Counter

import pytest

from cli import _distribution_panel, _extract_report_data
from core.models import (
    Difficulty,
    RankedSource,
    Source,
    SourcePlatform,
    SourceType,
    TaskStatus,
)


class _FakeResult:
    def __init__(self, sources: list[Source]) -> None:
        self.status = TaskStatus.SUCCESS
        self.request_id = "test"
        self.query = None
        self.corrected_topic = ""
        self.primary_concept = ""
        self.keywords = []
        self.expanded_queries = []
        self.sources = sources
        self.ranked_sources = [
            RankedSource(source=s, rank=i + 1, score=0.8 - i * 0.01)
            for i, s in enumerate(sources)
        ]
        self.source_summaries = []
        self.learning_path = None
        self.enhanced_learning_path = None
        self.errors = []
        self.warnings = []
        self.saved_files = {}
        self.rag_documents_indexed = 0
        self.latency_ms = 1000.0
        self.target_domains = []
        self.target_formats = []
        self.domain_confidence = 0.0
        self.detected_level = ""
        self.domain_routed = False
        self.domain_expanded = False
        self.domain_platforms = []
        self.source_selection_reasons = []
        self.quality_guards = {}
        self.quality_gate_passed = True


def _make_source(i: int, platform: SourcePlatform) -> Source:
    return Source(
        source_id=f"s{i}",
        title=f"Source {i}",
        url=f"https://example.com/{i}",
        platform=platform,
        source_type=SourceType.DOCUMENTATION,
        difficulty=Difficulty.BEGINNER,
    )


def test_distribution_not_double_counted() -> None:
    sources = [
        _make_source(1, SourcePlatform.WIKIBOOKS),
        _make_source(2, SourcePlatform.WIKIBOOKS),
        _make_source(3, SourcePlatform.WIKIPEDIA),
    ]
    result = _FakeResult(sources)
    data = _extract_report_data(result, "pipeline")

    assert data["counted_total"] == 3
    assert data["platform_counts"][SourcePlatform.WIKIBOOKS.value] == 2
    assert data["platform_counts"][SourcePlatform.WIKIPEDIA.value] == 1
    assert sum(data["platform_counts"].values()) == 3
    assert sum(data["type_counts"].values()) == 3


def test_distribution_shares_sum_to_100() -> None:
    sources = [
        _make_source(1, SourcePlatform.WIKIBOOKS),
        _make_source(2, SourcePlatform.WIKIBOOKS),
        _make_source(3, SourcePlatform.WIKIPEDIA),
        _make_source(4, SourcePlatform.WIKIPEDIA),
    ]
    result = _FakeResult(sources)
    data = _extract_report_data(result, "pipeline")

    denominator = max(
        data["counted_total"],
        sum(data["platform_counts"].values()),
        1,
    )
    total_share = sum(
        count / denominator
        for count in data["platform_counts"].values()
    )
    assert total_share == pytest.approx(1.0)


def test_distribution_panel_handles_zero_total() -> None:
    panel = _distribution_panel("Test", Counter(), 0)
    assert panel is None


def test_distribution_panel_handles_stale_total() -> None:
    counter = Counter({"a": 5, "b": 3})
    panel = _distribution_panel("Test", counter, 0)
    assert panel is not None