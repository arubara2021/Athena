from __future__ import annotations

from typing import Any

import pytest

from core.models import (
    Difficulty,
    LearningStep,
    RankedSource,
    Source,
    SourcePlatform,
    SourceType,
)
from core.schemas import SearchQuerySchema
from ranking.learning_path_builder import (
    MINIMUM_STEPS,
    PADDING_MARKER,
    LearningPathBuilder,
)


def _make_ranked(index: int, difficulty: Difficulty) -> RankedSource:
    source = Source(
        source_id=f"s{index}",
        title=f"Source {index}",
        url=f"https://example.com/{index}",
        platform=SourcePlatform.WIKIBOOKS,
        source_type=SourceType.DOCUMENTATION,
        abstract="An abstract",
        difficulty=difficulty,
    )
    return RankedSource(
        source=source,
        rank=index,
        score=0.9 - index * 0.01,
        confidence=0.5,
    )


def test_pad_steps_adds_marker() -> None:
    builder = LearningPathBuilder(use_llm=False)
    ranked = [
        _make_ranked(i, Difficulty.BEGINNER) for i in range(1, 6)
    ]
    query = SearchQuerySchema(topic="test", level="beginner")
    steps = builder._pad_steps([], ranked, set(), query)

    assert len(steps) == MINIMUM_STEPS
    for step in steps:
        for resource in step.resources:
            assert resource.reason is not None
            assert resource.reason.startswith(PADDING_MARKER)


def test_path_contains_padding_detects_marker() -> None:
    builder = LearningPathBuilder(use_llm=False)

    from core.models import LearningPath

    ranked = _make_ranked(1, Difficulty.BEGINNER)
    padded_step = LearningStep(
        step=1,
        title="Padded Step",
        objective="obj",
        resources=[
            ranked.model_copy(
                update={"reason": f"{PADDING_MARKER} something"}
            )
        ],
    )

    path = LearningPath(
        topic="t",
        level="beginner",
        goal="g",
        steps=[padded_step],
    )

    assert builder._path_contains_padding(path) is True


def test_path_contains_padding_clean_path() -> None:
    builder = LearningPathBuilder(use_llm=False)

    from core.models import LearningPath

    ranked = _make_ranked(1, Difficulty.BEGINNER)
    step = LearningStep(
        step=1,
        title="Clean",
        objective="obj",
        resources=[
            ranked.model_copy(update={"reason": "AI selected"})
        ],
    )

    path = LearningPath(
        topic="t",
        level="beginner",
        goal="g",
        steps=[step],
    )

    assert builder._path_contains_padding(path) is False