from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from core.models import CoreModel


_VALID_ANSWER_STYLES = frozenset(
    {"short", "standard", "detailed", "exhaustive"}
)
_DEFAULT_ANSWER_STYLE = "standard"


def _normalize_answer_style(value: Any) -> str:
    if value is None:
        return _DEFAULT_ANSWER_STYLE

    text = str(value).strip().lower()

    if not text:
        return _DEFAULT_ANSWER_STYLE

    if text in _VALID_ANSWER_STYLES:
        return text

    return _DEFAULT_ANSWER_STYLE


class ChatCitationSchema(CoreModel):
    source_id: str = ""
    title: str = ""
    url: str | None = None
    score: float = 0.0
    snippet: str = ""


class FallbackStepSchema(CoreModel):
    stage: str = ""
    reason: str = ""
    action: str = ""
    severity: str = "info"


class TopicProfileSchema(CoreModel):
    question: str = ""
    intent: str = "overview"
    intent_confidence: float = 0.0
    primary_topic: str = ""
    topic_tokens: list[str] = Field(default_factory=list)
    subtopics: list[str] = Field(default_factory=list)
    related_concepts: list[str] = Field(default_factory=list)
    acronyms: list[str] = Field(default_factory=list)
    quoted_phrases: list[str] = Field(default_factory=list)
    language: str = ""
    corpus_documents_seen: int = 0
    corpus_available: bool = False
    answer_style: str = _DEFAULT_ANSWER_STYLE

    @field_validator("answer_style", mode="before")
    @classmethod
    def _clean_answer_style(cls, value: Any) -> str:
        return _normalize_answer_style(value)


class ChatDiagnosticsSchema(CoreModel):
    strategy: str = ""
    fallback: bool = False
    weak_answer: bool = False
    citation_count: int = 0
    embedding_dimensions: int = 0
    keyword_hits: int = 0
    embedding_hits: int = 0
    deduped_count: int = 0
    reranked_count: int = 0
    confidence: float = 0.0
    store_total_documents: int = 0
    engine_diagnostics: dict[str, Any] = Field(default_factory=dict)
    fallback_ladder: list[FallbackStepSchema] = Field(default_factory=list)
    threshold_used: float | None = None
    threshold_floor: float | None = None
    threshold_source: str = ""
    rerank_applied: bool = False
    confidence_breakdown: dict[str, float] = Field(default_factory=dict)


class ChatAnswerSchema(CoreModel):
    question: str = ""
    answer: str = ""
    citations: list[ChatCitationSchema] = Field(default_factory=list)
    context_used: int = 0
    strategy: str = ""
    fallback: bool = False
    confidence: float = 0.0
    llm_used: bool = False
    topic: TopicProfileSchema | None = None
    expanded_queries: list[str] = Field(default_factory=list)
    session_id: str = ""
    diagnostics: ChatDiagnosticsSchema | None = None
    answer_style: str = _DEFAULT_ANSWER_STYLE

    @field_validator("answer_style", mode="before")
    @classmethod
    def _clean_answer_style(cls, value: Any) -> str:
        return _normalize_answer_style(value)