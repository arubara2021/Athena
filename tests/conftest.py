from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Generator

import pytest

from core.models import (
    Difficulty,
    LearningPath,
    LearningStep,
    RankedSource,
    Source,
    SourcePlatform,
    SourceType,
)
from core.schemas import SearchQuerySchema
from storage.file_manager import FileManager


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def file_manager(tmp_output_dir: Path) -> FileManager:
    return FileManager(tmp_output_dir)


@pytest.fixture
def sample_source_arxiv() -> Source:
    return Source(
        source_id="arxiv_test_001",
        title="Autonomous AI Agents: A Comprehensive Survey",
        url="https://arxiv.org/abs/2401.00001",
        platform=SourcePlatform.ARXIV,
        source_type=SourceType.PAPER,
        abstract="This paper surveys autonomous AI agent architectures.",
        authors=["Alice Smith", "Bob Jones"],
        published_at=datetime(2025, 3, 15, tzinfo=timezone.utc),
        year=2025,
        citation_count=42,
        has_code=True,
        difficulty=Difficulty.ADVANCED,
        score=None,
        metadata={"arxiv_id": "2401.00001", "categories": ["cs.AI"]},
    )


@pytest.fixture
def sample_source_github() -> Source:
    return Source(
        source_id="github_test_001",
        title="crewAIInc/crewAI",
        url="https://github.com/crewAIInc/crewAI",
        platform=SourcePlatform.GITHUB,
        source_type=SourceType.REPOSITORY,
        abstract="Framework for orchestrating role-playing autonomous AI agents.",
        authors=["crewAIInc"],
        published_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
        year=2025,
        citation_count=None,
        has_code=True,
        difficulty=Difficulty.INTERMEDIATE,
        score=None,
        metadata={"stars": 18500, "language": "Python"},
    )


@pytest.fixture
def sample_source_wikipedia() -> Source:
    return Source(
        source_id="wiki_test_001",
        title="Autonomous agent",
        url="https://en.wikipedia.org/wiki/Autonomous_agent",
        platform=SourcePlatform.WIKIPEDIA,
        source_type=SourceType.DOCUMENTATION,
        abstract="An autonomous agent is a system that perceives its environment and acts upon it.",
        authors=[],
        published_at=None,
        year=None,
        citation_count=None,
        has_code=None,
        difficulty=Difficulty.BEGINNER,
        score=None,
        metadata={"page_id": 12345, "word_count": 3200},
    )


@pytest.fixture
def sample_source_withdrawn() -> Source:
    return Source(
        source_id="arxiv_withdrawn_001",
        title="This paper has been withdrawn",
        url="https://arxiv.org/abs/2401.99999",
        platform=SourcePlatform.ARXIV,
        source_type=SourceType.PAPER,
        abstract="",
        authors=[],
        published_at=None,
        year=None,
        citation_count=None,
        has_code=None,
        difficulty=None,
        score=None,
        metadata={},
    )


@pytest.fixture
def sample_sources(
    sample_source_arxiv: Source,
    sample_source_github: Source,
    sample_source_wikipedia: Source,
) -> list[Source]:
    return [sample_source_arxiv, sample_source_github, sample_source_wikipedia]


@pytest.fixture
def sample_ranked_sources(sample_sources: list[Source]) -> list[RankedSource]:
    ranked: list[RankedSource] = []
    for index, source in enumerate(sample_sources, start=1):
        ranked.append(
            RankedSource(
                source=source,
                rank=index,
                score=round(1.0 - (index * 0.1), 2),
                confidence=0.9,
                reason=f"Test reason for rank {index}",
            )
        )
    return ranked


@pytest.fixture
def sample_learning_path(sample_ranked_sources: list[RankedSource]) -> LearningPath:
    return LearningPath(
        topic="Autonomous AI",
        level="beginner to advanced",
        goal="Learn from basics to advanced",
        steps=[
            LearningStep(
                step=1,
                title="Build the fundamentals",
                objective="Understand core concepts",
                estimated_minutes=90,
                resources=sample_ranked_sources[:2],
            ),
            LearningStep(
                step=2,
                title="Study advanced research",
                objective="Read higher-level papers",
                estimated_minutes=180,
                resources=sample_ranked_sources[2:],
            ),
        ],
    )


@pytest.fixture
def sample_search_query() -> SearchQuerySchema:
    return SearchQuerySchema(
        topic="Autonomous AI",
        goal="Learn from basics to advanced",
        level="beginner to advanced",
        max_results=10,
    )