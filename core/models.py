from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

class ProviderName(str, Enum):
    SAMBANOVA = "SAMBANOVA"
    GROQ = "GROQ"
    GOOGLE = "GOOGLE"
    MISTRAL = "MISTRAL"
    NVIDIA = "NVIDIA"

class SourcePlatform(str, Enum):
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    OPENALEX = "openalex"
    GITHUB = "github"
    WIKIPEDIA = "wikipedia"
    HUGGINGFACE = "huggingface"
    WEB = "web"
    TAVILY = "tavily"
    EXA = "exa"
    CORE = "core"
    SERPER = "serper"
    SERPAPI = "serpapi"
    JINA = "jina"

class SourceType(str, Enum):
    PAPER = "research_paper"
    REPOSITORY = "repository"
    COURSE = "course"
    VIDEO = "video"
    BLOG = "blog"
    DOCUMENTATION = "documentation"
    DATASET = "dataset"
    MODEL = "model"
    OTHER = "other"

class Difficulty(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: object) -> "Difficulty | None":
        if isinstance(value, str):
            val = value.strip().lower()
            for member in cls:
                if member.value == val:
                    return member
            return cls.UNKNOWN
        return None

class ModelRole(str, Enum):
    PRIMARY_FAST = "primary_fast"
    PRIMARY_STRONG = "primary_strong"
    FALLBACK_STRONG = "fallback_strong"
    FALLBACK_FAST = "fallback_fast"
    JUDGE = "judge"
    CODE = "code"
    EMBEDDING = "embedding"

class VotingMode(str, Enum):
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    UNANIMOUS = "unanimous"
    JUDGE = "judge"

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"

class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PLANNING = "planning"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    REFLECTING = "reflecting"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"

class AgentActionType(str, Enum):
    SEARCH = "search"
    READ = "read"
    ANALYZE = "analyze"
    GENERATE = "generate"
    REFLECT = "reflect"
    PLAN = "plan"
    SYNTHESIZE = "synthesize"
    MEMORY_STORE = "memory_store"
    MEMORY_RECALL = "memory_recall"
    NO_OP = "no_op"

class CoreModel(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        extra="ignore",
    )

class ProviderHealth(CoreModel):
    provider: ProviderName
    key_present: bool = False
    working: bool = False
    model_count: int = 0
    models: list[str] = Field(default_factory=list)
    error: str | None = None
    latency_ms: float | None = None
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ModelReference(CoreModel):
    provider: ProviderName
    model_id: str
    role: ModelRole = ModelRole.FALLBACK_FAST
    priority: int = Field(default=100, ge=0)
    weight: float = Field(default=1.0, ge=0.0)

class Source(CoreModel):
    source_id: str
    title: str
    url: str
    platform: SourcePlatform
    source_type: SourceType
    abstract: str | None = None
    summary: str | None = None
    authors: list[str] = Field(default_factory=list)
    published_at: datetime | None = None
    year: int | None = None
    citation_count: int | None = None
    has_code: bool | None = None
    difficulty: Difficulty | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class RankedSource(CoreModel):
    source: Source
    rank: int = Field(ge=1)
    score: float = Field(ge=0.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str | None = None

class LearningStep(CoreModel):
    step: int = Field(ge=1)
    title: str
    objective: str
    estimated_minutes: int | None = None
    resources: list[RankedSource] = Field(default_factory=list)

class LearningPath(CoreModel):
    topic: str
    level: str
    goal: str | None = None
    steps: list[LearningStep] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class EnsembleVote(CoreModel):
    provider: ProviderName
    model_id: str
    output: Any = None
    latency_ms: float = 0.0
    success: bool = False
    error: str | None = None

class ConsensusResult(CoreModel):
    final_output: Any = None
    agreement_score: float = Field(default=0.0, ge=0.0, le=1.0)
    voting_mode: VotingMode = VotingMode.MAJORITY
    votes: list[EnsembleVote] = Field(default_factory=list)
    reasoning: str | None = None
    judge_used: bool = False

class PipelineMetrics(CoreModel):
    total_sources_found: int = 0
    total_sources_after_dedupe: int = 0
    total_sources_ranked: int = 0
    llm_calls: int = 0
    failed_llm_calls: int = 0
    total_latency_ms: float = 0.0

class ToolCall(CoreModel):
    call_id: str
    tool_name: str
    action_type: AgentActionType = AgentActionType.NO_OP
    parameters: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    tokens_used: int = 0
    latency_ms: float = 0.0
    success: bool = False
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AgentAction(CoreModel):
    action_id: str
    step_index: int = Field(default=0, ge=0)
    action_type: AgentActionType
    description: str = ""
    tool_call: ToolCall | None = None
    reasoning: str = ""
    tokens_used: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class PlanStep(CoreModel):
    step_index: int = Field(ge=0)
    description: str
    action_type: AgentActionType
    estimated_tokens: int = Field(default=0, ge=0)
    priority: int = Field(default=0, ge=0)
    status: AgentStatus = AgentStatus.IDLE
    completed: bool = False

class AgentPlan(CoreModel):
    plan_id: str
    goal: str
    steps: list[PlanStep] = Field(default_factory=list)
    total_estimated_tokens: int = Field(default=0, ge=0)
    budget_allocated: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ReflectionResult(CoreModel):
    iteration: int = Field(default=0, ge=0)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_sufficient: bool = False
    should_retry: bool = False
    should_stop: bool = False
    reasoning: str = ""
    suggestions: list[str] = Field(default_factory=list)
    tokens_used: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class MemoryEntry(CoreModel):
    entry_id: str
    memory_type: str = "episodic"
    key: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] = Field(default_factory=list)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    access_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: datetime | None = None

class AgentState(CoreModel):
    agent_id: str
    status: AgentStatus = AgentStatus.IDLE
    goal: str = ""
    level: str = ""
    plan: AgentPlan | None = None
    current_step_index: int = Field(default=0, ge=0)
    actions_taken: list[AgentAction] = Field(default_factory=list)
    findings: list[Source] = Field(default_factory=list)
    reflections: list[ReflectionResult] = Field(default_factory=list)
    short_term_memory: dict[str, Any] = Field(default_factory=dict)
    total_tokens_used: int = Field(default=0, ge=0)
    total_budget: int = Field(default=0, ge=0)
    tokens_remaining: int = Field(default=0, ge=0)
    iteration_count: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=15, ge=1)
    final_output: Any = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    latency_ms: float | None = None