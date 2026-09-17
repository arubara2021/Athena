from __future__ import annotations

import pytest

from core.pipeline import PipelineResult, ResearchPipeline
from core.models import TaskStatus
from core.schemas import SearchQuerySchema


@pytest.mark.asyncio
async def test_pipeline_runs_successfully_with_real_apis() -> None:
    pipeline = ResearchPipeline()

    async with pipeline:
        result = await pipeline.run(
            SearchQuerySchema(
                topic="machine learning basics",
                goal="Learn fundamentals",
                level="beginner",
                max_results=5,
                platforms=["wikipedia"],
            )
        )

    assert isinstance(result, PipelineResult)
    assert result.request_id
    assert result.status in {TaskStatus.SUCCESS, TaskStatus.PARTIAL}
    assert result.query is not None
    assert result.query.topic == "machine learning basics"


@pytest.mark.asyncio
async def test_pipeline_returns_sources_for_wikipedia() -> None:
    pipeline = ResearchPipeline()

    async with pipeline:
        result = await pipeline.run(
            SearchQuerySchema(
                topic="artificial intelligence",
                max_results=3,
                platforms=["wikipedia"],
            )
        )

    assert len(result.sources) > 0
    assert len(result.ranked_sources) > 0
    assert all(r.rank >= 1 for r in result.ranked_sources)


@pytest.mark.asyncio
async def test_pipeline_generates_learning_path() -> None:
    pipeline = ResearchPipeline()

    async with pipeline:
        result = await pipeline.run(
            SearchQuerySchema(
                topic="deep learning",
                max_results=5,
                platforms=["wikipedia", "arxiv"],
            )
        )

    if result.sources:
        assert result.learning_path is not None
        assert len(result.learning_path.steps) > 0
        assert result.learning_path.topic == "deep learning"


@pytest.mark.asyncio
async def test_pipeline_saves_output_files() -> None:
    pipeline = ResearchPipeline()

    async with pipeline:
        result = await pipeline.run(
            SearchQuerySchema(
                topic="reinforcement learning",
                max_results=3,
                platforms=["wikipedia"],
            )
        )

    assert "json" in result.saved_files
    assert "markdown" in result.saved_files
    assert "sqlite" in result.saved_files


@pytest.mark.asyncio
async def test_pipeline_handles_invalid_query_gracefully() -> None:
    pipeline = ResearchPipeline()

    async with pipeline:
        result = await pipeline.run("")

    assert result.status == TaskStatus.FAILED
    assert len(result.errors) > 0


@pytest.mark.asyncio
async def test_pipeline_records_latency() -> None:
    pipeline = ResearchPipeline()

    async with pipeline:
        result = await pipeline.run(
            SearchQuerySchema(
                topic="neural networks",
                max_results=3,
                platforms=["wikipedia"],
            )
        )

    assert result.latency_ms is not None
    assert result.latency_ms > 0
    assert result.started_at is not None
    assert result.finished_at is not None


@pytest.mark.asyncio
async def test_pipeline_multi_platform_fetch() -> None:
    pipeline = ResearchPipeline()

    async with pipeline:
        result = await pipeline.run(
            SearchQuerySchema(
                topic="large language models",
                max_results=6,
                platforms=["arxiv", "github", "wikipedia"],
            )
        )

    assert result.status in {TaskStatus.SUCCESS, TaskStatus.PARTIAL}

    if result.sources:
        platforms_found = {s.platform for s in result.sources}
        assert len(platforms_found) >= 1