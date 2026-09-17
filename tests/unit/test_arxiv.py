from __future__ import annotations

import pytest

from clients.arxiv_client import ArxivClient
from core.models import SourcePlatform, SourceType


@pytest.mark.asyncio
async def test_arxiv_search_returns_sources() -> None:
    client = ArxivClient()

    try:
        sources = await client.search("machine learning", max_results=3)
    finally:
        await client.close()

    assert isinstance(sources, list)
    assert len(sources) <= 3

    for source in sources:
        assert source.platform == SourcePlatform.ARXIV
        assert source.source_type == SourceType.PAPER
        assert source.title
        assert source.url
        assert source.source_id


@pytest.mark.asyncio
async def test_arxiv_search_empty_query_returns_empty() -> None:
    client = ArxivClient()

    try:
        sources = await client.search("", max_results=5)
    finally:
        await client.close()

    assert sources == []


@pytest.mark.asyncio
async def test_arxiv_search_respects_max_results() -> None:
    client = ArxivClient()

    try:
        sources = await client.search("neural networks", max_results=2)
    finally:
        await client.close()

    assert len(sources) <= 2


@pytest.mark.asyncio
async def test_arxiv_source_has_valid_metadata() -> None:
    client = ArxivClient()

    try:
        sources = await client.search("transformer architecture", max_results=1)
    finally:
        await client.close()

    if not sources:
        pytest.skip("No arXiv results returned during test")

    source = sources[0]
    assert isinstance(source.metadata, dict)
    assert "arxiv_id" in source.metadata or "categories" in source.metadata