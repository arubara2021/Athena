from __future__ import annotations

import pytest

from core.models import Difficulty, Source, SourcePlatform, SourceType
from search.dedupe import SourceDeduplicator


@pytest.fixture
def deduplicator() -> SourceDeduplicator:
    return SourceDeduplicator()


def test_dedupe_removes_exact_url_duplicates(deduplicator: SourceDeduplicator) -> None:
    source_a = Source(
        source_id="id_a",
        title="Test Paper",
        url="https://example.com/paper.pdf",
        platform=SourcePlatform.ARXIV,
        source_type=SourceType.PAPER,
    )
    source_b = Source(
        source_id="id_b",
        title="Test Paper Different ID",
        url="https://example.com/paper.pdf",
        platform=SourcePlatform.SEMANTIC_SCHOLAR,
        source_type=SourceType.PAPER,
    )

    result = deduplicator.deduplicate([source_a, source_b])

    assert len(result) == 1


def test_dedupe_removes_title_duplicates_same_type(
    deduplicator: SourceDeduplicator,
) -> None:
    source_a = Source(
        source_id="id_a",
        title="Autonomous AI Agents Survey",
        url="https://arxiv.org/abs/001",
        platform=SourcePlatform.ARXIV,
        source_type=SourceType.PAPER,
    )
    source_b = Source(
        source_id="id_b",
        title="Autonomous AI Agents Survey",
        url="https://semanticscholar.org/paper/002",
        platform=SourcePlatform.SEMANTIC_SCHOLAR,
        source_type=SourceType.PAPER,
    )

    result = deduplicator.deduplicate([source_a, source_b])

    assert len(result) == 1


def test_dedupe_keeps_different_types_with_same_title(
    deduplicator: SourceDeduplicator,
) -> None:
    paper = Source(
        source_id="paper_001",
        title="CrewAI Framework",
        url="https://arxiv.org/abs/001",
        platform=SourcePlatform.ARXIV,
        source_type=SourceType.PAPER,
    )
    repo = Source(
        source_id="repo_001",
        title="CrewAI Framework",
        url="https://github.com/crewai/crewai",
        platform=SourcePlatform.GITHUB,
        source_type=SourceType.REPOSITORY,
    )

    result = deduplicator.deduplicate([paper, repo])

    assert len(result) == 2


def test_dedupe_preserves_order(deduplicator: SourceDeduplicator) -> None:
    sources = [
        Source(
            source_id=f"id_{i}",
            title=f"Unique Paper {i}",
            url=f"https://example.com/{i}",
            platform=SourcePlatform.ARXIV,
            source_type=SourceType.PAPER,
        )
        for i in range(5)
    ]

    result = deduplicator.deduplicate(sources)

    assert len(result) == 5
    for index, source in enumerate(result):
        assert source.title == f"Unique Paper {index}"


def test_dedupe_merges_metadata_and_authors(
    deduplicator: SourceDeduplicator,
) -> None:
    source_a = Source(
        source_id="id_a",
        title="Shared Title Paper",
        url="https://arxiv.org/abs/001",
        platform=SourcePlatform.ARXIV,
        source_type=SourceType.PAPER,
        authors=["Alice"],
        citation_count=10,
        metadata={"arxiv_id": "001"},
    )
    source_b = Source(
        source_id="id_b",
        title="Shared Title Paper",
        url="https://semanticscholar.org/paper/002",
        platform=SourcePlatform.SEMANTIC_SCHOLAR,
        source_type=SourceType.PAPER,
        authors=["Bob"],
        citation_count=25,
        metadata={"doi": "10.1234/test"},
    )

    result = deduplicator.deduplicate([source_a, source_b])

    assert len(result) == 1
    merged = result[0]
    assert "Alice" in merged.authors
    assert "Bob" in merged.authors
    assert merged.citation_count == 25


def test_dedupe_empty_input(deduplicator: SourceDeduplicator) -> None:
    result = deduplicator.deduplicate([])
    assert result == []