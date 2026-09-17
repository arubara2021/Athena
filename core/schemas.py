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

class BaseSchema(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        extra="ignore",
    )

class SearchQuerySchema(BaseSchema):
    topic: str = Field(min_length=constants.MIN_SEARCH_QUERY_LENGTH, max_length=constants.MAX_SEARCH_QUERY_LENGTH)
    goal: str = ""
    level: str = ""
    max_results: int = Field(default=constants.DEFAULT_MAX_RESULTS, ge=constants.MIN_MAX_RESULTS, le=constants.MAX_MAX_RESULTS)
    platforms: list[SourcePlatform] = Field(default_factory=list)
    source_types: list[SourceType] = Field(default_factory=list)
    corrected_topic: str = ""
    primary_concept: str = ""
    keywords: list[str] = Field(default_factory=list)
    intent: str = ""
    detected_level: str = ""
    short_search_queries: list[str] = Field(default_factory=list)

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

class QueryExpansionResult(BaseSchema):
    queries: list[str] = Field(default_factory=list)
    corrected_topic: str = ""
    original_topic: str = ""
    primary_concept: str = ""
    keywords: list[str] = Field(default_factory=list)
    intent: str = ""
    detected_level: str = ""
    short_search_queries: list[str] = Field(default_factory=list)

class LLMMessageSchema(BaseSchema):
    role: str = Field(pattern=r"^(system|user|assistant)$")
    content: str

class LLMRequestSchema(BaseSchema):
    provider: ProviderName
    model: str
    messages: list[LLMMessageSchema] = Field(min_length=1)
    temperature: float = Field(default=constants.DEFAULT_TEMPERATURE, ge=constants.MIN_TEMPERATURE, le=constants.MAX_TEMPERATURE)
    max_tokens: int = Field(default=constants.DEFAULT_MAX_TOKENS, ge=constants.MIN_MAX_TOKENS, le=constants.MAX_MAX_TOKENS)
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
    threshold: float = Field(default=constants.DEFAULT_CONSENSUS_THRESHOLD, ge=constants.MIN_CONSENSUS_THRESHOLD, le=constants.MAX_CONSENSUS_THRESHOLD)
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