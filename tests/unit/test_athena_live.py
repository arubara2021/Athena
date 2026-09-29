from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.pipeline import ResearchPipeline
from core.schemas import SearchQuerySchema
from ranking.consensus_ranker import ConsensusRanker
from ranking.learning_path_builder import LearningPathBuilder
from search.orchestrator import SearchOrchestrator


@pytest.mark.asyncio
@pytest.mark.integration
async def test_pipeline_mixed_profile_no_wipe(tmp_path: Path) -> None:
    pipeline = ResearchPipeline(
        search_orchestrator=SearchOrchestrator(use_llm_expansion=False),
        consensus_ranker=ConsensusRanker(use_llm=False),
        learning_path_builder=LearningPathBuilder(use_llm=False),
        enable_rag=False,
        database_path=str(tmp_path / "test.db"),
    )

    query = SearchQuerySchema(
        topic="basic organic chemistry",
        goal="Learn from basics to advanced",
        level="beginner to advanced",
        max_results=10,
    )

    async with pipeline:
        result = await pipeline.run(query)

    assert result.status.value in {"success", "partial"}
    assert len(result.sources) > 0
    assert len(result.ranked_sources) > 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_pipeline_produces_learning_path(tmp_path: Path) -> None:
    pipeline = ResearchPipeline(
        search_orchestrator=SearchOrchestrator(use_llm_expansion=False),
        consensus_ranker=ConsensusRanker(use_llm=False),
        learning_path_builder=LearningPathBuilder(use_llm=False),
        enable_rag=False,
        database_path=str(tmp_path / "test.db"),
    )

    query = SearchQuerySchema(
        topic="machine learning fundamentals",
        goal="Learn from basics",
        level="beginner",
        max_results=10,
    )

    async with pipeline:
        result = await pipeline.run(query)

    if result.ranked_sources:
        assert result.learning_path is not None
        assert len(result.learning_path.steps) >= 1


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_token_ledger_records_usage(
    tmp_path: Path,
) -> None:
    from agent.controller import AutonomousAgent, AgentConfig
    from llm.provider import LLMProviderManager

    config = AgentConfig(
        max_iterations=3,
        token_budget=10000,
        memory_enabled=False,
        reflection_enabled=False,
    )

    agent = AutonomousAgent(
        agent_config=config,
        database_path=str(tmp_path / "agent.db"),
    )

    async with agent:
        result = await agent.run(
            goal="learn python basics",
            level="beginner",
            topic="python basics",
            max_results=5,
        )

    assert result.total_budget == 10000
    assert result.total_tokens_used >= 0
    assert result.tokens_remaining == 10000 - result.total_tokens_used