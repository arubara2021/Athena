from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.loop.observe import ObservePhase
from agent.token.cost_tracker import CostTracker
from core.models import Difficulty, Source, SourcePlatform, SourceType
from core.schemas import SearchQuerySchema
from llm.ensemble import EnsembleEngine, VoteFingerprint
from llm.provider import LLMProviderManager
from ranking.learning_path_builder import PADDING_MARKER
from search.query_intelligence import QueryIntelligence


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("beginner", Difficulty.BEGINNER),
        ("intermediate", Difficulty.INTERMEDIATE),
        ("advanced", Difficulty.ADVANCED),
        ("BEGINNER", Difficulty.BEGINNER),
        ("Advanced", Difficulty.ADVANCED),
        ("unknown_value", Difficulty.UNKNOWN),
        (None, None),
        ("", None),
        (Difficulty.BEGINNER, Difficulty.BEGINNER),
    ],
)
def test_observe_difficulty_parsing(raw: Any, expected: Any) -> None:
    phase = ObservePhase()
    item = {
        "title": "Test",
        "url": "https://example.com/x",
        "source_id": "s1",
        "platform": "wikipedia",
        "source_type": "documentation",
        "difficulty": raw,
    }
    source = phase._dict_to_source(item)
    assert source is not None
    assert source.difficulty == expected


def test_observe_preserves_all_fields() -> None:
    phase = ObservePhase()
    item = {
        "source_id": "abc",
        "title": "Test Source",
        "url": "https://example.com/a",
        "platform": "wikibooks",
        "source_type": "documentation",
        "abstract": "An abstract.",
        "summary": "A summary.",
        "authors": ["Alice", "Bob", "Alice"],
        "year": 2023,
        "citation_count": 42,
        "has_code": True,
        "difficulty": "beginner",
        "score": 0.87,
        "metadata": {"page_id": 1, "tags": ["x", "y"]},
    }
    source = phase._dict_to_source(item)
    assert source is not None
    assert source.source_id == "abc"
    assert source.title == "Test Source"
    assert source.abstract == "An abstract."
    assert source.summary == "A summary."
    assert source.year == 2023
    assert source.citation_count == 42
    assert source.has_code is True
    assert source.difficulty == Difficulty.BEGINNER
    assert source.score == 0.87
    assert source.metadata == {"page_id": 1, "tags": ["x", "y"]}
    assert source.authors == ["Alice", "Bob"]


def test_observe_rejects_empty_title_and_url() -> None:
    phase = ObservePhase()
    assert phase._dict_to_source({}) is None
    assert phase._dict_to_source({"title": ""}) is None
    assert phase._dict_to_source({"url": ""}) is None


@dataclass
class _FakeQuery:
    level_profile: str = ""
    level: str = ""
    goal: str = ""
    topic: str = ""


@pytest.mark.parametrize(
    "profile,level,expected",
    [
        ("beginner", "beginner to advanced", True),
        ("advanced", "beginner to advanced", False),
        ("mixed", "beginner to advanced", False),
        ("balanced", "beginner to advanced", False),
        ("", "beginner", True),
        ("", "beginner to advanced", True),
        ("", "advanced", False),
        ("", "mixed", False),
    ],
)
def test_pipeline_beginner_guard(
    profile: str,
    level: str,
    expected: bool,
) -> None:
    from core.pipeline import ResearchPipeline

    pipeline = ResearchPipeline.__new__(ResearchPipeline)
    query = _FakeQuery(
        level_profile=profile,
        level=level,
        goal="learn",
        topic="chemistry",
    )
    assert pipeline._is_beginner_request(query) is expected


def test_query_intelligence_strips_llm_injected_level_words() -> None:
    qi = QueryIntelligence(use_llm=False)
    from search.query_intelligence import QueryAnalysis, QueryIntent

    analysis = QueryAnalysis(
        original_topic="basic organic chemistry",
        cleaned_topic="basic organic chemistry",
        corrected_topic="organic chemistry basics advanced",
        primary_concept="organic chemistry fundamentals",
        keywords=["organic", "chemistry", "basics", "advanced"],
        intent=QueryIntent.LEARN,
        level="mixed",
        learner_level="mixed",
        level_priority=["beginner", "intermediate", "advanced"],
        short_search_queries=[
            "organic chemistry basics advanced",
            "organic chemistry fundamentals",
        ],
    )
    cleaned = qi._strip_llm_injected_level_words(analysis)

    assert "advanced" not in cleaned.corrected_topic
    assert "fundamentals" not in cleaned.primary_concept
    assert "basics" in cleaned.corrected_topic
    assert "advanced" not in " ".join(cleaned.keywords)
    assert "advanced" not in " ".join(cleaned.short_search_queries)


def test_query_intelligence_preserves_user_typed_level_words() -> None:
    qi = QueryIntelligence(use_llm=False)
    from search.query_intelligence import QueryAnalysis, QueryIntent

    analysis = QueryAnalysis(
        original_topic="advanced python programming",
        cleaned_topic="advanced python programming",
        corrected_topic="advanced python programming tutorial",
        primary_concept="advanced python",
        keywords=["advanced", "python"],
        intent=QueryIntent.LEARN,
        level="advanced",
        learner_level="advanced",
        level_priority=["advanced"],
        short_search_queries=["advanced python programming"],
    )
    cleaned = qi._strip_llm_injected_level_words(analysis)

    assert "advanced" in cleaned.corrected_topic
    assert "advanced" in cleaned.primary_concept
    assert "advanced" in " ".join(cleaned.keywords)


def test_query_intelligence_strips_level_phrases() -> None:
    qi = QueryIntelligence(use_llm=False)
    from search.query_intelligence import QueryAnalysis, QueryIntent

    analysis = QueryAnalysis(
        original_topic="python",
        cleaned_topic="python",
        corrected_topic="python for beginners from scratch",
        primary_concept="python programming",
        keywords=["python"],
        intent=QueryIntent.LEARN,
        level="beginner",
        learner_level="beginner",
        level_priority=["beginner"],
        short_search_queries=["python for beginners"],
    )
    cleaned = qi._strip_llm_injected_level_words(analysis)

    assert "for beginners" not in cleaned.corrected_topic
    assert "from scratch" not in cleaned.corrected_topic
    assert "python" in cleaned.corrected_topic


class _DummyVote:
    def __init__(self, output: Any) -> None:
        self.output = output


def test_ensemble_fingerprint_strict_vs_loose() -> None:
    engine = EnsembleEngine.__new__(EnsembleEngine)

    vote_a = {
        "rankings": [
            {"source_id": "s1", "rank": 1, "score": 0.9},
            {"source_id": "s2", "rank": 2, "score": 0.7},
        ]
    }
    vote_b = {
        "rankings": [
            {"source_id": "s1", "rank": 1, "score": 0.5},
            {"source_id": "s2", "rank": 2, "score": 0.3},
        ]
    }

    fp_a = engine._vote_fingerprint(vote_a)
    fp_b = engine._vote_fingerprint(vote_b)

    assert fp_a.strict != fp_b.strict
    assert fp_a.loose == fp_b.loose


def test_ensemble_fingerprint_identical_votes() -> None:
    engine = EnsembleEngine.__new__(EnsembleEngine)

    vote = {
        "rankings": [
            {"source_id": "s1", "rank": 1, "score": 0.9, "reason": "x"},
            {"source_id": "s2", "rank": 2, "score": 0.7, "reason": "y"},
        ]
    }

    fp1 = engine._vote_fingerprint(vote)
    fp2 = engine._vote_fingerprint(vote)

    assert fp1.strict == fp2.strict
    assert fp1.loose == fp2.loose
    assert fp1.reasoning == fp2.reasoning
    assert fp1.has_rankings is True


def test_ensemble_fingerprint_handles_non_ranking_output() -> None:
    engine = EnsembleEngine.__new__(EnsembleEngine)

    fp = engine._vote_fingerprint("just a string")

    assert fp.strict == fp.loose
    assert fp.has_rankings is False


def test_ensemble_majority_strict_agreement() -> None:
    engine = EnsembleEngine.__new__(EnsembleEngine)

    a = {"rankings": [{"source_id": "s1", "score": 0.9}]}
    b = {"rankings": [{"source_id": "s1", "score": 0.9}]}
    c = {"rankings": [{"source_id": "s2", "score": 0.2}]}

    votes = [_DummyVote(a), _DummyVote(b), _DummyVote(c)]
    output, score, mode = engine._majority_output(votes, 0.6, 2)

    assert output is not None
    assert score == pytest.approx(2 / 3)
    assert mode == "strict"


def test_ensemble_majority_loose_agreement() -> None:
    engine = EnsembleEngine.__new__(EnsembleEngine)

    a = {"rankings": [{"source_id": "s1", "score": 0.9}]}
    b = {"rankings": [{"source_id": "s1", "score": 0.3}]}
    c = {"rankings": [{"source_id": "s2", "score": 0.5}]}

    votes = [_DummyVote(a), _DummyVote(b), _DummyVote(c)]
    output, score, mode = engine._majority_output(votes, 0.6, 2)

    assert output is not None
    assert mode == "loose"


def test_ensemble_majority_disagreement() -> None:
    engine = EnsembleEngine.__new__(EnsembleEngine)

    a = {"rankings": [{"source_id": "s1", "score": 0.9}]}
    b = {"rankings": [{"source_id": "s2", "score": 0.3}]}
    c = {"rankings": [{"source_id": "s3", "score": 0.5}]}

    votes = [_DummyVote(a), _DummyVote(b), _DummyVote(c)]
    output, score, mode = engine._majority_output(votes, 0.6, 2)

    assert output is None
    assert mode == "disagreement"


def test_shared_manager_singleton() -> None:
    LLMProviderManager._shared_instance = None

    m1 = LLMProviderManager.get_shared()
    m2 = LLMProviderManager.get_shared()

    assert m1 is m2

    asyncio.run(LLMProviderManager.close_shared())


def test_cost_tracker_on_token_usage_accumulates() -> None:
    tracker = CostTracker(task_id="t1", budget=10000)

    tracker.on_token_usage(
        provider="mistral",
        model="x",
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        latency_ms=100.0,
    )
    tracker.on_token_usage(
        provider="mistral",
        model="x",
        input_tokens=200,
        output_tokens=100,
        total_tokens=300,
        latency_ms=200.0,
    )

    assert tracker.total_tokens_used == 450


def test_cost_tracker_on_token_usage_uses_input_plus_output_fallback() -> None:
    tracker = CostTracker(task_id="t2", budget=10000)

    tracker.on_token_usage(
        provider="mistral",
        model="x",
        input_tokens=100,
        output_tokens=50,
        total_tokens=0,
        latency_ms=100.0,
    )

    assert tracker.total_tokens_used == 150


def test_cost_tracker_on_token_usage_ignores_zero() -> None:
    tracker = CostTracker(task_id="t3", budget=10000)

    tracker.on_token_usage(
        provider="mistral",
        model="x",
        total_tokens=0,
        latency_ms=0.0,
    )

    assert tracker.total_tokens_used == 0


def test_cost_tracker_on_token_usage_ignores_negative() -> None:
    tracker = CostTracker(task_id="t4", budget=10000)

    tracker.on_token_usage(
        provider="mistral",
        model="x",
        total_tokens=-5,
        latency_ms=0.0,
    )

    assert tracker.total_tokens_used == 0


def test_padding_marker_constant() -> None:
    assert PADDING_MARKER == "[padded]"