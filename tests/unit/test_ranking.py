from __future__ import annotations

import pytest

from core.models import Difficulty, Source, SourcePlatform, SourceType
from ranking.scorer import HeuristicScorer


@pytest.fixture
def scorer() -> HeuristicScorer:
    return HeuristicScorer()


def test_scorer_returns_ranked_sources(scorer: HeuristicScorer) -> None:
    sources = [
        Source(
            source_id="s1",
            title="Intro to AI",
            url="https://example.com/1",
            platform=SourcePlatform.WIKIPEDIA,
            source_type=SourceType.DOCUMENTATION,
            difficulty=Difficulty.BEGINNER,
        ),
        Source(
            source_id="s2",
            title="Advanced Autonomous Agents Survey",
            url="https://arxiv.org/abs/001",
            platform=SourcePlatform.ARXIV,
            source_type=SourceType.PAPER,
            citation_count=100,
            authors=["A", "B", "C"],
            difficulty=Difficulty.ADVANCED,
        ),
    ]

    ranked = scorer.score_sources(sources)

    assert len(ranked) == 2
    assert all(r.score >= 0.0 for r in ranked)
    assert all(r.confidence >= 0.0 for r in ranked)
    assert ranked[0].rank == 1
    assert ranked[1].rank == 2


def test_scorer_with_query_boosts_relevance(scorer: HeuristicScorer) -> None:
    relevant = Source(
        source_id="rel",
        title="Autonomous AI Agent Architectures",
        url="https://example.com/rel",
        platform=SourcePlatform.ARXIV,
        source_type=SourceType.PAPER,
    )
    irrelevant = Source(
        source_id="irrel",
        title="Cooking Recipes for Beginners",
        url="https://example.com/irrel",
        platform=SourcePlatform.WIKIPEDIA,
        source_type=SourceType.DOCUMENTATION,
    )

    ranked = scorer.score_sources([relevant, irrelevant], query="autonomous AI agents")

    assert len(ranked) == 2
    relevant_score = next(r.score for r in ranked if r.source.source_id == "rel")
    irrelevant_score = next(r.score for r in ranked if r.source.source_id == "irrel")
    assert relevant_score > irrelevant_score


def test_scorer_high_citation_scores_higher(scorer: HeuristicScorer) -> None:
    high_cite = Source(
        source_id="high",
        title="Landmark Paper",
        url="https://example.com/high",
        platform=SourcePlatform.SEMANTIC_SCHOLAR,
        source_type=SourceType.PAPER,
        citation_count=500,
    )
    low_cite = Source(
        source_id="low",
        title="New Paper",
        url="https://example.com/low",
        platform=SourcePlatform.SEMANTIC_SCHOLAR,
        source_type=SourceType.PAPER,
        citation_count=2,
    )

    ranked = scorer.score_sources([high_cite, low_cite])

    high_score = next(r.score for r in ranked if r.source.source_id == "high")
    low_score = next(r.score for r in ranked if r.source.source_id == "low")
    assert high_score > low_score


def test_scorer_empty_sources_returns_empty(scorer: HeuristicScorer) -> None:
    ranked = scorer.score_sources([])
    assert ranked == []


def test_scorer_assigns_sequential_ranks(scorer: HeuristicScorer) -> None:
    sources = [
        Source(
            source_id=f"s{i}",
            title=f"Paper {i}",
            url=f"https://example.com/{i}",
            platform=SourcePlatform.ARXIV,
            source_type=SourceType.PAPER,
        )
        for i in range(5)
    ]

    ranked = scorer.score_sources(sources)

    ranks = [r.rank for r in ranked]
    assert ranks == list(range(1, 6))


def test_scorer_confidence_reflects_completeness(scorer: HeuristicScorer) -> None:
    complete = Source(
        source_id="complete",
        title="Complete Paper",
        url="https://example.com/complete",
        platform=SourcePlatform.ARXIV,
        source_type=SourceType.PAPER,
        abstract="A detailed abstract about this paper.",
        authors=["Author One", "Author Two"],
        year=2024,
        citation_count=50,
    )
    minimal = Source(
        source_id="minimal",
        title="Minimal",
        url="https://example.com/minimal",
        platform=SourcePlatform.WEB,
        source_type=SourceType.BLOG,
    )

    ranked = scorer.score_sources([complete, minimal])

    complete_conf = next(r.confidence for r in ranked if r.source.source_id == "complete")
    minimal_conf = next(r.confidence for r in ranked if r.source.source_id == "minimal")
    assert complete_conf > minimal_conf