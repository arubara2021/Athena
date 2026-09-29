from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core import constants
from core.models import (
    AgentActionType,
    AgentStatus,
    Difficulty,
    ProviderName,
    SourcePlatform,
    SourceType,
    VotingMode,
)


def _clean_token_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        value = [value]

    if not isinstance(value, (list, tuple, set)):
        value = [value]

    result: list[str] = []

    for item in value:
        text = str(item or "").strip().lower()
        text = text.replace("-", "_").replace(" ", "_")
        text = "".join(ch for ch in text if ch.isalnum() or ch == "_")
        text = "_".join(part for part in text.split("_") if part)

        if text and text not in result:
            result.append(text)

    return result


def _clean_phrase_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        value = [value]

    if not isinstance(value, (list, tuple, set)):
        value = [value]

    result: list[str] = []

    for item in value:
        text = str(item or "").strip()

        if text and text not in result:
            result.append(text)

    return result


def _clean_confidence_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}

    result: dict[str, float] = {}

    for key, item in value.items():
        text = str(key or "").strip().lower()
        text = text.replace("-", "_").replace(" ", "_")
        text = "".join(ch for ch in text if ch.isalnum() or ch == "_")
        text = "_".join(part for part in text.split("_") if part)

        if not text:
            continue

        try:
            score = float(item)
        except Exception:
            continue

        result[text] = max(0.0, min(1.0, score))

    return result


def _clean_metadata_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value

    return {}


def _clean_level_profile(value: Any) -> str:
    text = str(value or "").strip().lower()

    if text in constants.VALID_LEVEL_PROFILES:
        return text

    return ""


def _clean_ratio(value: Any) -> float:
    try:
        numeric = float(value)
    except Exception:
        return 0.0

    return max(0.0, min(1.0, numeric))


def _clean_non_negative_int(value: Any) -> int:
    try:
        numeric = int(value)
    except Exception:
        return 0

    return max(0, numeric)


def _clean_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        text = value.strip().lower()

        if text in {"true", "yes", "on", "1"}:
            return True

        if text in {"false", "no", "off", "0"}:
            return False

    return False


def _clean_concept_origin(value: Any) -> str:
    text = str(value or "").strip().lower()

    if text in {"fixed", "qualified", "unknown"}:
        return text

    return ""


class BaseSchema(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        extra="ignore",
    )


class SearchQuerySchema(BaseSchema):
    topic: str = Field(
        min_length=constants.MIN_SEARCH_QUERY_LENGTH,
        max_length=constants.MAX_SEARCH_QUERY_LENGTH,
    )
    goal: str = ""
    level: str = ""
    max_results: int = Field(
        default=constants.DEFAULT_MAX_RESULTS,
        ge=constants.MIN_MAX_RESULTS,
        le=constants.MAX_MAX_RESULTS,
    )
    platforms: list[SourcePlatform] = Field(default_factory=list)
    source_types: list[SourceType] = Field(default_factory=list)
    corrected_topic: str = ""
    primary_concept: str = ""
    keywords: list[str] = Field(default_factory=list)
    intent: str = ""
    detected_level: str = ""
    short_search_queries: list[str] = Field(default_factory=list)

    target_domains: list[str] = Field(default_factory=list)
    target_formats: list[str] = Field(default_factory=list)
    domain_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    domain_confidences: dict[str, float] = Field(default_factory=dict)
    level_priority: list[str] = Field(default_factory=list)
    learner_level: str = ""
    level_alignment_score: float = Field(default=0.0, ge=0.0, le=1.0)
    source_type_metadata: dict[str, Any] = Field(default_factory=dict)

    level_profile: str = ""
    beginner_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    min_beginner_sources: int = Field(default=0, ge=0)
    exclude_research_platforms: bool = False

    preserve_tokens: list[str] = Field(default_factory=list)
    concept_origin: str = ""
    user_typed_tokens: list[str] = Field(default_factory=list)

    @field_validator("topic", mode="before")
    @classmethod
    def clean_topic(cls, value: Any) -> str:
        from utils.text import clean_text

        return clean_text(value)

    @field_validator("platforms", mode="before")
    @classmethod
    def normalize_platforms(cls, value: Any) -> list[SourcePlatform]:
        if value is None:
            return []

        if isinstance(value, str):
            value = [value]

        if not isinstance(value, (list, tuple, set)):
            return []

        result: list[SourcePlatform] = []

        for item in value:
            try:
                result.append(SourcePlatform(str(item).strip().lower()))
            except Exception:
                continue

        return result

    @field_validator("source_types", mode="before")
    @classmethod
    def normalize_source_types(cls, value: Any) -> list[SourceType]:
        if value is None:
            return []

        if isinstance(value, str):
            value = [value]

        if not isinstance(value, (list, tuple, set)):
            return []

        result: list[SourceType] = []

        for item in value:
            try:
                result.append(SourceType(str(item).strip().lower()))
            except Exception:
                continue

        return result

    @field_validator("target_domains", mode="before")
    @classmethod
    def normalize_target_domains(cls, value: Any) -> list[str]:
        return _clean_token_list(value)

    @field_validator("target_formats", mode="before")
    @classmethod
    def normalize_target_formats(cls, value: Any) -> list[str]:
        return _clean_token_list(value)

    @field_validator("domain_confidences", mode="before")
    @classmethod
    def normalize_domain_confidences(cls, value: Any) -> dict[str, float]:
        return _clean_confidence_map(value)

    @field_validator("level_priority", mode="before")
    @classmethod
    def normalize_level_priority(cls, value: Any) -> list[str]:
        return _clean_token_list(value)

    @field_validator("source_type_metadata", mode="before")
    @classmethod
    def normalize_source_type_metadata(cls, value: Any) -> dict[str, Any]:
        return _clean_metadata_dict(value)

    @field_validator("level_profile", mode="before")
    @classmethod
    def normalize_level_profile(cls, value: Any) -> str:
        return _clean_level_profile(value)

    @field_validator("beginner_ratio", mode="before")
    @classmethod
    def normalize_beginner_ratio(cls, value: Any) -> float:
        return _clean_ratio(value)

    @field_validator("min_beginner_sources", mode="before")
    @classmethod
    def normalize_min_beginner_sources(cls, value: Any) -> int:
        return _clean_non_negative_int(value)

    @field_validator("exclude_research_platforms", mode="before")
    @classmethod
    def normalize_exclude_research_platforms(cls, value: Any) -> bool:
        return _clean_bool(value)

    @field_validator("preserve_tokens", mode="before")
    @classmethod
    def normalize_preserve_tokens(cls, value: Any) -> list[str]:
        return _clean_phrase_list(value)

    @field_validator("concept_origin", mode="before")
    @classmethod
    def normalize_concept_origin(cls, value: Any) -> str:
        return _clean_concept_origin(value)


class QueryExpansionResult(BaseSchema):
    queries: list[str] = Field(default_factory=list)
    corrected_topic: str = ""
    original_topic: str = ""
    primary_concept: str = ""
    keywords: list[str] = Field(default_factory=list)
    intent: str = ""
    detected_level: str = ""
    short_search_queries: list[str] = Field(default_factory=list)

    target_domains: list[str] = Field(default_factory=list)
    target_formats: list[str] = Field(default_factory=list)
    domain_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    domain_confidences: dict[str, float] = Field(default_factory=dict)
    level_priority: list[str] = Field(default_factory=list)
    learner_level: str = ""
    level_alignment_score: float = Field(default=0.0, ge=0.0, le=1.0)
    source_type_metadata: dict[str, Any] = Field(default_factory=dict)

    level_profile: str = ""
    beginner_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    min_beginner_sources: int = Field(default=0, ge=0)
    exclude_research_platforms: bool = False

    preserve_tokens: list[str] = Field(default_factory=list)
    concept_origin: str = ""
    user_typed_tokens: list[str] = Field(default_factory=list)

    @field_validator("target_domains", mode="before")
    @classmethod
    def normalize_target_domains(cls, value: Any) -> list[str]:
        return _clean_token_list(value)

    @field_validator("target_formats", mode="before")
    @classmethod
    def normalize_target_formats(cls, value: Any) -> list[str]:
        return _clean_token_list(value)

    @field_validator("domain_confidences", mode="before")
    @classmethod
    def normalize_domain_confidences(cls, value: Any) -> dict[str, float]:
        return _clean_confidence_map(value)

    @field_validator("level_priority", mode="before")
    @classmethod
    def normalize_level_priority(cls, value: Any) -> list[str]:
        return _clean_token_list(value)

    @field_validator("source_type_metadata", mode="before")
    @classmethod
    def normalize_source_type_metadata(cls, value: Any) -> dict[str, Any]:
        return _clean_metadata_dict(value)

    @field_validator("level_profile", mode="before")
    @classmethod
    def normalize_level_profile(cls, value: Any) -> str:
        return _clean_level_profile(value)

    @field_validator("beginner_ratio", mode="before")
    @classmethod
    def normalize_beginner_ratio(cls, value: Any) -> float:
        return _clean_ratio(value)

    @field_validator("min_beginner_sources", mode="before")
    @classmethod
    def normalize_min_beginner_sources(cls, value: Any) -> int:
        return _clean_non_negative_int(value)

    @field_validator("exclude_research_platforms", mode="before")
    @classmethod
    def normalize_exclude_research_platforms(cls, value: Any) -> bool:
        return _clean_bool(value)

    @field_validator("preserve_tokens", mode="before")
    @classmethod
    def normalize_preserve_tokens(cls, value: Any) -> list[str]:
        return _clean_phrase_list(value)

    @field_validator("concept_origin", mode="before")
    @classmethod
    def normalize_concept_origin(cls, value: Any) -> str:
        return _clean_concept_origin(value)


class DomainDetectionResult(BaseSchema):
    domains: list[str] = Field(default_factory=list)
    domain_confidences: dict[str, float] = Field(default_factory=dict)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    learner_level: str = ""
    level_priority: list[str] = Field(default_factory=list)
    target_formats: list[str] = Field(default_factory=list)

    level_profile: str = ""
    beginner_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    min_beginner_sources: int = Field(default=0, ge=0)
    exclude_research_platforms: bool = False

    @field_validator("domains", mode="before")
    @classmethod
    def normalize_domains(cls, value: Any) -> list[str]:
        return _clean_token_list(value)

    @field_validator("domain_confidences", mode="before")
    @classmethod
    def normalize_domain_confidences(cls, value: Any) -> dict[str, float]:
        return _clean_confidence_map(value)

    @field_validator("level_priority", mode="before")
    @classmethod
    def normalize_level_priority(cls, value: Any) -> list[str]:
        return _clean_token_list(value)

    @field_validator("target_formats", mode="before")
    @classmethod
    def normalize_target_formats(cls, value: Any) -> list[str]:
        return _clean_token_list(value)

    @field_validator("level_profile", mode="before")
    @classmethod
    def normalize_level_profile(cls, value: Any) -> str:
        return _clean_level_profile(value)

    @field_validator("beginner_ratio", mode="before")
    @classmethod
    def normalize_beginner_ratio(cls, value: Any) -> float:
        return _clean_ratio(value)

    @field_validator("min_beginner_sources", mode="before")
    @classmethod
    def normalize_min_beginner_sources(cls, value: Any) -> int:
        return _clean_non_negative_int(value)

    @field_validator("exclude_research_platforms", mode="before")
    @classmethod
    def normalize_exclude_research_platforms(cls, value: Any) -> bool:
        return _clean_bool(value)


class SourceAlignmentMetadata(BaseSchema):
    source_id: str = ""
    domain_match_score: float = Field(default=0.0, ge=0.0, le=1.0)
    level_alignment_score: float = Field(default=0.0, ge=0.0, le=1.0)
    beginner_friendly: bool = False
    research_level: bool = False
    source_type_priority: float = Field(default=0.0, ge=0.0, le=1.0)
    source_type: str = ""
    default_difficulty: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def normalize_metadata(cls, value: Any) -> dict[str, Any]:
        return _clean_metadata_dict(value)


class SearchOrchestrationResultSchema(BaseSchema):
    request_id: str = ""
    target_domains: list[str] = Field(default_factory=list)
    target_formats: list[str] = Field(default_factory=list)
    domain_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    domain_confidences: dict[str, float] = Field(default_factory=dict)
    level_priority: list[str] = Field(default_factory=list)
    learner_level: str = ""
    level_alignment_score: float = Field(default=0.0, ge=0.0, le=1.0)
    source_type_metadata: dict[str, Any] = Field(default_factory=dict)

    level_profile: str = ""
    beginner_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    min_beginner_sources: int = Field(default=0, ge=0)
    exclude_research_platforms: bool = False

    preserve_tokens: list[str] = Field(default_factory=list)
    concept_origin: str = ""
    user_typed_tokens: list[str] = Field(default_factory=list)

    @field_validator("target_domains", mode="before")
    @classmethod
    def normalize_target_domains(cls, value: Any) -> list[str]:
        return _clean_token_list(value)

    @field_validator("target_formats", mode="before")
    @classmethod
    def normalize_target_formats(cls, value: Any) -> list[str]:
        return _clean_token_list(value)

    @field_validator("domain_confidences", mode="before")
    @classmethod
    def normalize_domain_confidences(cls, value: Any) -> dict[str, float]:
        return _clean_confidence_map(value)

    @field_validator("level_priority", mode="before")
    @classmethod
    def normalize_level_priority(cls, value: Any) -> list[str]:
        return _clean_token_list(value)

    @field_validator("source_type_metadata", mode="before")
    @classmethod
    def normalize_source_type_metadata(cls, value: Any) -> dict[str, Any]:
        return _clean_metadata_dict(value)

    @field_validator("level_profile", mode="before")
    @classmethod
    def normalize_level_profile(cls, value: Any) -> str:
        return _clean_level_profile(value)

    @field_validator("beginner_ratio", mode="before")
    @classmethod
    def normalize_beginner_ratio(cls, value: Any) -> float:
        return _clean_ratio(value)

    @field_validator("min_beginner_sources", mode="before")
    @classmethod
    def normalize_min_beginner_sources(cls, value: Any) -> int:
        return _clean_non_negative_int(value)

    @field_validator("exclude_research_platforms", mode="before")
    @classmethod
    def normalize_exclude_research_platforms(cls, value: Any) -> bool:
        return _clean_bool(value)

    @field_validator("preserve_tokens", mode="before")
    @classmethod
    def normalize_preserve_tokens(cls, value: Any) -> list[str]:
        return _clean_phrase_list(value)

    @field_validator("concept_origin", mode="before")
    @classmethod
    def normalize_concept_origin(cls, value: Any) -> str:
        return _clean_concept_origin(value)


class LLMMessageSchema(BaseSchema):
    role: str = Field(pattern=r"^(system|user|assistant)$")
    content: str


class LLMRequestSchema(BaseSchema):
    provider: ProviderName
    model: str
    messages: list[LLMMessageSchema] = Field(min_length=1)
    temperature: float = Field(
        default=constants.DEFAULT_TEMPERATURE,
        ge=constants.MIN_TEMPERATURE,
        le=constants.MAX_TEMPERATURE,
    )
    max_tokens: int = Field(
        default=constants.DEFAULT_MAX_TOKENS,
        ge=constants.MIN_MAX_TOKENS,
        le=constants.MAX_MAX_TOKENS,
    )
    top_p: float = Field(default=constants.DEFAULT_TOP_P, ge=0.0, le=1.0)
    response_format: str = "json_object"
    timeout: float | None = None

    @field_validator("model", mode="before")
    @classmethod
    def clean_model(cls, value: Any) -> str:
        return str(value or "").strip()


class LLMResponseSchema(BaseSchema):
    provider: ProviderName
    model: str
    content: str
    parsed: Any = None
    latency_ms: float = 0.0
    tokens_used: int | None = None
    raw: Any = None


class EnsembleRequestSchema(BaseSchema):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    prompt: str
    context: dict[str, Any] = Field(default_factory=dict)
    models: list[Any] = Field(default_factory=list)
    voting_mode: VotingMode = VotingMode.MAJORITY
    threshold: float = Field(
        default=constants.DEFAULT_CONSENSUS_THRESHOLD,
        ge=constants.MIN_CONSENSUS_THRESHOLD,
        le=constants.MAX_CONSENSUS_THRESHOLD,
    )
    judge_model: Any = None


class EnsembleVoteSchema(BaseSchema):
    provider: ProviderName
    model_id: str
    output: Any = None
    latency_ms: float = 0.0
    success: bool = False
    error: str | None = None


class EnsembleResultSchema(BaseSchema):
    task_id: str
    success: bool = False
    final_output: Any = None
    agreement_score: float = Field(default=0.0, ge=0.0, le=1.0)
    votes: list[EnsembleVoteSchema] = Field(default_factory=list)
    judge_used: bool = False
    reasoning: str | None = None


class ToolCallSchema(BaseSchema):
    call_id: str = Field(default_factory=lambda: str(uuid4()))
    tool_name: str
    action_type: AgentActionType = AgentActionType.NO_OP
    parameters: dict[str, Any] = Field(default_factory=dict)
    estimated_tokens: int = Field(default=0, ge=0)
    timeout_seconds: float | None = None

    @field_validator("tool_name", mode="before")
    @classmethod
    def clean_tool_name(cls, value: Any) -> str:
        return str(value or "").strip().lower()

    @field_validator("parameters", mode="before")
    @classmethod
    def ensure_dict(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}

        if isinstance(value, dict):
            return value

        return {}


class PlanStepSchema(BaseSchema):
    step_index: int = Field(ge=0)
    description: str
    action_type: AgentActionType = AgentActionType.NO_OP
    tool_name: str = ""
    tool_parameters: dict[str, Any] = Field(default_factory=dict)
    estimated_tokens: int = Field(default=0, ge=0)
    priority: int = Field(default=0, ge=0)
    depends_on: list[int] = Field(default_factory=list)

    @field_validator("description", mode="before")
    @classmethod
    def clean_description(cls, value: Any) -> str:
        from utils.text import clean_text

        return clean_text(value)


class AgentInputSchema(BaseSchema):
    agent_id: str = Field(default_factory=lambda: str(uuid4()))
    goal: str = Field(min_length=2, max_length=500)
    level: str = ""
    topic: str = ""
    max_iterations: int = Field(default=15, ge=1, le=50)
    token_budget: int = Field(default=40000, ge=1000, le=500000)
    use_memory: bool = True
    use_reflection: bool = True
    max_retries_per_step: int = Field(default=2, ge=0, le=5)
    quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("goal", mode="before")
    @classmethod
    def clean_goal(cls, value: Any) -> str:
        from utils.text import clean_text

        return clean_text(value)

    @field_validator("level", mode="before")
    @classmethod
    def normalize_level(cls, value: Any) -> str:
        text = str(value or "").strip().lower()

        if text in ("beginner", "intermediate", "advanced", "mixed"):
            return text

        return "mixed"


class AgentOutputSchema(BaseSchema):
    agent_id: str
    status: AgentStatus = AgentStatus.COMPLETED
    goal: str = ""
    final_output: Any = None
    learning_path: Any = None
    ranked_sources: list[Any] = Field(default_factory=list)
    summaries: list[Any] = Field(default_factory=list)
    total_iterations: int = Field(default=0, ge=0)
    total_tokens_used: int = Field(default=0, ge=0)
    total_budget: int = Field(default=0, ge=0)
    tokens_remaining: int = Field(default=0, ge=0)
    actions_taken: list[Any] = Field(default_factory=list)
    reflections: list[Any] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    latency_ms: float | None = None
    saved_files: dict[str, str] = Field(default_factory=dict)