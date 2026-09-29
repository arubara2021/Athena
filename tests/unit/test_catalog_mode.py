from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from core.models import Source, SourcePlatform, SourceType
from rag.catalog_service import CatalogAnswer, CatalogService
from rag.generator import Generator
from rag.intent_router import CatalogClassification, IntentRouter
from rag.vector_store import VectorStore


@pytest.fixture
def temp_store():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test_store.db"
        store = VectorStore(path=path)
        store.load()
        yield store
        store.close()


def _catalog_classification() -> CatalogClassification:
    return CatalogClassification(
        route="catalog",
        score=0.0,
        confidence=1.0,
        threshold=0.0,
        signals={},
        weights={},
        matched={},
        reason="test_fixture",
        is_catalog=True,
    )


def _install_fake_llm(
    monkeypatch,
    label_by_question: dict[str, str],
) -> None:
    import core.schemas as core_schemas
    import llm.guardrails as guardrails
    import llm.provider as provider_mod
    import llm.router as router_mod

    class FakeRequest:
        def __init__(
            self,
            provider=None,
            model=None,
            messages=None,
            **kwargs,
        ):
            self.provider = provider
            self.model = model
            self.messages = list(messages or [])
            self.question = (
                self.messages[-1].content if self.messages else ""
            )

    class FakeResponse:
        def __init__(self, content: str):
            self.content = content

    class FakeManager:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def complete(self, request):
            question = getattr(request, "question", "")
            label = label_by_question.get(question)

            if label is None:
                return None

            return FakeResponse(label)

    fake_ref = type(
        "Ref",
        (),
        {"provider": "fake", "model_id": "fake_model"},
    )()

    monkeypatch.setattr(core_schemas, "LLMRequestSchema", FakeRequest)
    monkeypatch.setattr(
        guardrails, "validate_llm_request", lambda x: x
    )
    monkeypatch.setattr(
        guardrails, "validate_llm_response", lambda x: x
    )
    monkeypatch.setattr(
        provider_mod, "LLMProviderManager", FakeManager
    )
    monkeypatch.setattr(
        router_mod,
        "get_fast_model_references",
        lambda limit=3: [fake_ref],
    )
    monkeypatch.setattr(
        router_mod,
        "get_strong_model_references",
        lambda limit=3: [fake_ref],
    )


def _install_no_llm(monkeypatch) -> None:
    import llm.router as router_mod

    monkeypatch.setattr(
        router_mod,
        "get_fast_model_references",
        lambda limit=3: [],
    )
    monkeypatch.setattr(
        router_mod,
        "get_strong_model_references",
        lambda limit=3: [],
    )


def _make_source(
    idx: int,
    title: str,
    abstract: str,
    platform: SourcePlatform = SourcePlatform.WIKIPEDIA,
    source_type: SourceType = SourceType.BOOK,
    domain: str | None = None,
    topic: str | None = None,
) -> Source:
    return Source(
        source_id=f"src_{idx}",
        title=title,
        url=f"https://example.com/{idx}",
        platform=platform,
        source_type=source_type,
        abstract=abstract,
        domain=domain,
        topic=topic,
    )


def test_cs_query_returns_domain_data(temp_store):
    sources = [
        _make_source(
            1,
            "Introduction to machine learning",
            "Machine learning fundamentals, models, and evaluation.",
            domain="computer_science",
            topic="machine learning",
        ),
        _make_source(
            2,
            "Algorithms and data structures",
            "Sorting, searching, and complexity analysis.",
            domain="computer_science",
            topic="algorithms",
        ),
        _make_source(
            3,
            "Basic chemistry",
            "Periodic table and chemical bonding.",
            domain="chemistry",
            topic="chemistry",
        ),
    ]

    temp_store.add_sources(
        sources,
        embeddings=[[] for _ in sources],
        run_id="run_a",
    )

    generator = Generator()
    service = CatalogService(
        vector_store=temp_store,
        generator=generator,
    )

    result = asyncio.run(
        service.answer(
            question="what computer science topics do you have",
            run_id=None,
            session_id="test",
            classification=_catalog_classification(),
        )
    )

    assert isinstance(result, CatalogAnswer)
    assert (
        result.snapshot.get("by_domain", {}).get("computer_science", 0) >= 2
    )
    assert (
        result.snapshot.get("by_topic", {}).get("machine learning", 0) >= 1
    )
    assert result.escalation is None


def test_router_maps_llm_catalog_label_to_catalog_route(monkeypatch):
    queries = (
        "what topics do you have in your corpus",
        "what sources do you have",
        "you have a library of documents",
        "you hold a large collection",
        "you've indexed many papers",
        "your corpus contains what topics",
        "our sources include what",
        "we stored many documents",
        "what computer science topics do you have",
        "give me a breakdown of everything you have amassed",
    )

    labels = {q: "catalog" for q in queries}
    _install_fake_llm(monkeypatch, labels)

    router = IntentRouter()

    for query in queries:
        result = asyncio.run(router.classify(query))

        assert result.route == "catalog", (
            f"LLM returned catalog but router mapped to {result.route}: "
            f"query={query!r} reason={result.reason}"
        )
        assert result.is_catalog is True
        assert result.reason == "llm_classified_catalog"


def test_router_maps_llm_topical_label_to_other_route(monkeypatch):
    queries = (
        "what is machine learning",
        "explain quantum mechanics",
        "how does a transformer work",
        "compare TCP and UDP",
        "why is the sky blue",
        "what are the properties of water",
    )

    labels = {q: "topical" for q in queries}
    _install_fake_llm(monkeypatch, labels)

    router = IntentRouter()

    for query in queries:
        result = asyncio.run(router.classify(query))

        assert result.route == "other", (
            f"LLM returned topical but router mapped to {result.route}: "
            f"query={query!r} reason={result.reason}"
        )
        assert result.is_catalog is False
        assert result.reason == "llm_classified_topical"


def test_router_falls_back_when_llm_unavailable(monkeypatch):
    _install_no_llm(monkeypatch)

    router = IntentRouter()
    result = asyncio.run(router.classify("what topics do you have"))

    assert result.route == "other"
    assert result.is_catalog is False
    assert result.reason == "llm_unavailable"
    assert result.confidence == 0.0


def test_classification_carries_required_fields(monkeypatch):
    _install_fake_llm(
        monkeypatch,
        {"what topics do you have": "catalog"},
    )

    router = IntentRouter()
    result = asyncio.run(router.classify("what topics do you have"))

    assert isinstance(result, CatalogClassification)

    for field in (
        "route",
        "score",
        "confidence",
        "threshold",
        "signals",
        "weights",
        "matched",
        "reason",
        "is_catalog",
    ):
        assert hasattr(result, field), f"Missing field: {field}"

    assert isinstance(result.signals, dict)
    assert isinstance(result.weights, dict)
    assert isinstance(result.matched, dict)
    assert isinstance(result.is_catalog, bool)
    assert isinstance(result.confidence, float)


def test_router_caches_repeated_questions(monkeypatch):
    call_count = {"n": 0}

    import core.schemas as core_schemas
    import llm.guardrails as guardrails
    import llm.provider as provider_mod
    import llm.router as router_mod

    class FakeRequest:
        def __init__(
            self,
            provider=None,
            model=None,
            messages=None,
            **kwargs,
        ):
            self.messages = list(messages or [])
            self.question = (
                self.messages[-1].content if self.messages else ""
            )

    class FakeResponse:
        def __init__(self, content):
            self.content = content

    class FakeManager:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def complete(self, request):
            call_count["n"] += 1
            return FakeResponse("catalog")

    fake_ref = type(
        "Ref",
        (),
        {"provider": "fake", "model_id": "fake_model"},
    )()

    monkeypatch.setattr(
        core_schemas, "LLMRequestSchema", FakeRequest
    )
    monkeypatch.setattr(
        guardrails, "validate_llm_request", lambda x: x
    )
    monkeypatch.setattr(
        guardrails, "validate_llm_response", lambda x: x
    )
    monkeypatch.setattr(
        provider_mod, "LLMProviderManager", FakeManager
    )
    monkeypatch.setattr(
        router_mod,
        "get_fast_model_references",
        lambda limit=3: [fake_ref],
    )
    monkeypatch.setattr(
        router_mod,
        "get_strong_model_references",
        lambda limit=3: [fake_ref],
    )

    router = IntentRouter()

    asyncio.run(router.classify("what topics do you have"))
    asyncio.run(router.classify("what topics do you have"))
    asyncio.run(router.classify("What Topics Do You Have?"))

    assert call_count["n"] == 1, (
        f"Cached questions should not re-call the LLM, "
        f"got {call_count['n']} calls"
    )


def test_confidence_is_derived_not_hardcoded():
    generator = Generator()

    rich_snapshot = {
        "total_documents": 100,
        "by_platform": {"wikipedia": 50, "arxiv": 50},
        "by_source_type": {"book": 50, "research_paper": 50},
        "by_domain": {"computer_science": 100},
        "by_topic": {"machine learning": 50, "algorithms": 50},
        "by_year": {"2024": 100},
        "by_run_id": {"run_a": 100},
        "titles": [{"title": "Intro to ML"}],
    }

    thin_snapshot = {"total_documents": 100}

    rich_confidence, rich_breakdown = generator._compute_catalog_confidence(
        text=(
            "The corpus contains documents about machine learning, "
            "algorithms, and related computer science topics across "
            "several platforms."
        ),
        snapshot=rich_snapshot,
        cited_dimensions=["domain", "topic", "platform"],
        classifier_score=0.8,
        classifier_confidence=0.8,
        llm_used=True,
    )

    thin_confidence, thin_breakdown = generator._compute_catalog_confidence(
        text="Short.",
        snapshot=thin_snapshot,
        cited_dimensions=[],
        classifier_score=0.8,
        classifier_confidence=0.8,
        llm_used=True,
    )

    assert rich_confidence > thin_confidence
    assert rich_confidence != 0.95
    assert thin_confidence != 0.85
    assert "snapshot_richness" in rich_breakdown
    assert "answer_substance" in rich_breakdown


def test_empty_text_yields_zero_confidence():
    generator = Generator()

    confidence, breakdown = generator._compute_catalog_confidence(
        text="",
        snapshot={"total_documents": 0},
        cited_dimensions=[],
        classifier_score=0.0,
        classifier_confidence=0.0,
        llm_used=False,
    )

    assert confidence == 0.0
    assert isinstance(breakdown, dict)


def test_refusal_caps_confidence():
    from rag.chat_engine import RAGChatEngine

    engine = RAGChatEngine.__new__(RAGChatEngine)

    refusal_texts = (
        "The provided passages do not contain information about that.",
        "I do not have enough information to answer this question.",
        "There is not enough information in the retrieved documents.",
        "The sources do not provide any details about that topic.",
    )

    for text in refusal_texts:
        assert RAGChatEngine._detect_refusal(engine, text) is True

    non_refusal_texts = (
        "The corpus contains 42 documents across three platforms.",
        "Machine learning is a subfield of computer science.",
    )

    for text in non_refusal_texts:
        assert RAGChatEngine._detect_refusal(engine, text) is False


def test_snapshot_limits_come_from_config(temp_store):
    service = CatalogService(
        vector_store=temp_store,
        generator=Generator(),
    )

    for key in (
        "max_documents",
        "max_titles",
        "max_platform_slices",
        "max_source_type_slices",
        "max_domain_slices",
        "max_topic_slices",
        "max_year_slices",
        "max_run_slices",
    ):
        assert key in service._limits
        assert isinstance(service._limits[key], int)
        assert service._limits[key] > 0

    for key in (
        "min_documents_for_topics",
        "min_documents_for_domains",
        "min_documents_for_platforms",
    ):
        assert key in service._escalation


def test_snapshot_limits_are_applied():
    service = CatalogService(
        vector_store=None,
        generator=None,
        config_override={
            "max_platform_slices": 3,
            "max_titles": 5,
        },
    )

    fake_snapshot = {
        "total_documents": 100,
        "by_platform": {
            "a": 10,
            "b": 20,
            "c": 30,
            "d": 40,
            "e": 50,
        },
        "titles": [{"title": f"t{i}"} for i in range(20)],
    }

    trimmed = service._trim_snapshot(fake_snapshot)

    assert len(trimmed["by_platform"]) <= 3
    assert len(trimmed["titles"]) <= 5
    assert len(trimmed["by_platform"]) >= 1


def test_escalation_when_dimension_missing(temp_store):
    sources = [
        _make_source(i, f"Doc {i}", f"abstract {i}") for i in range(5)
    ]

    temp_store.add_sources(
        sources,
        embeddings=[[] for _ in sources],
        run_id="run_a",
    )

    generator = Generator()
    service = CatalogService(
        vector_store=temp_store,
        generator=generator,
    )

    result = asyncio.run(
        service.answer(
            question="what domains do you have",
            run_id=None,
            session_id="test",
            classification=_catalog_classification(),
        )
    )

    assert result.escalation is not None
    assert result.escalation.startswith("dimension_")
    assert "corpus contains" in result.text.lower()