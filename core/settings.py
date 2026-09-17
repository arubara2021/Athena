from __future__ import annotations

import json
from typing import Any
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core import constants
from core.exceptions import ConfigError, MissingAPIKeyError
from core.models import ProviderName, SourcePlatform, VotingMode


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=constants.ENV_FILE,
        env_file_encoding=constants.ENV_FILE_ENCODING,
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = constants.APP_NAME
    app_version: str = constants.APP_VERSION
    environment: str = constants.DEFAULT_ENVIRONMENT
    debug: bool = False

    sambanova_api_key: SecretStr = SecretStr("")
    groq_api_key: SecretStr = SecretStr("")
    google_api_key: SecretStr = SecretStr("")
    mistral_api_key: SecretStr = SecretStr("")
    nvidia_api_key: SecretStr = SecretStr("")

    github_token: SecretStr = SecretStr("")
    semantic_scholar_api_key: SecretStr = SecretStr("")
    huggingface_token: SecretStr = SecretStr("")
    tavily_api_key: SecretStr = SecretStr("")
    exa_api_key: SecretStr = SecretStr("")
    core_api_key: SecretStr = SecretStr("")
    serper_api_key: SecretStr = SecretStr("")
    serpapi_api_key: SecretStr = SecretStr("")
    jina_api_key: SecretStr = SecretStr("")

    log_level: str = constants.DEFAULT_LOG_LEVEL

    output_dir: Path = Path(constants.DEFAULT_OUTPUT_DIR)
    cache_dir: Path = Path(constants.DEFAULT_CACHE_DIR)
    data_dir: Path = Path(constants.DEFAULT_DATA_DIR)
    logs_dir: Path = Path(constants.DEFAULT_LOGS_DIR)

    llm_timeout: float = Field(default=constants.DEFAULT_TIMEOUT_SECONDS, gt=0)
    llm_connect_timeout: float = Field(
        default=constants.DEFAULT_CONNECT_TIMEOUT_SECONDS,
        gt=0,
    )
    llm_max_retries: int = Field(default=constants.DEFAULT_MAX_RETRIES, ge=0)
    llm_retry_backoff_seconds: float = Field(
        default=constants.DEFAULT_RETRY_BACKOFF_SECONDS,
        ge=0,
    )
    llm_rate_limit_per_minute: int = Field(
        default=constants.DEFAULT_RATE_LIMIT_PER_MINUTE,
        ge=1,
    )

    search_concurrency: int = Field(default=4, ge=1, le=16)
    search_max_query_variants: int = Field(default=5, ge=1, le=12)
    search_request_timeout_seconds: float = Field(default=45.0, gt=0, le=120)
    search_default_max_results: int = Field(default=20, ge=1, le=100)

    source_max_retries: int = Field(default=2, ge=0, le=5)
    source_retry_backoff_seconds: float = Field(default=1.0, ge=0, le=30)

    default_search_platforms: list[str] = Field(
        default_factory=lambda: [
            SourcePlatform.ARXIV.value,
            SourcePlatform.OPENALEX.value,
            SourcePlatform.GITHUB.value,
            SourcePlatform.WIKIPEDIA.value,
            SourcePlatform.HUGGINGFACE.value,
            SourcePlatform.TAVILY.value,
            SourcePlatform.EXA.value,
            SourcePlatform.CORE.value,
            SourcePlatform.SERPER.value,
            SourcePlatform.JINA.value,
            SourcePlatform.SERPAPI.value,
        ]
    )

    allow_semantic_scholar_without_key: bool = False

    arxiv_rate_limit_per_minute: int = Field(default=5, ge=1, le=120)
    semantic_scholar_rate_limit_per_minute: int = Field(default=3, ge=1, le=120)
    semantic_scholar_rate_limit_with_key_per_minute: int = Field(
        default=30,
        ge=1,
        le=600,
    )
    openalex_rate_limit_per_minute: int = Field(default=20, ge=1, le=300)
    github_rate_limit_per_minute: int = Field(default=10, ge=1, le=600)
    wikipedia_rate_limit_per_minute: int = Field(default=8, ge=1, le=120)
    huggingface_rate_limit_per_minute: int = Field(default=20, ge=1, le=600)
    web_search_rate_limit_per_minute: int = Field(default=5, ge=1, le=120)
    tavily_rate_limit_per_minute: int = Field(default=10, ge=1, le=120)
    exa_rate_limit_per_minute: int = Field(default=10, ge=1, le=120)
    core_rate_limit_per_minute: int = Field(default=10, ge=1, le=120)
    serper_rate_limit_per_minute: int = Field(default=10, ge=1, le=120)
    serpapi_rate_limit_per_minute: int = Field(default=10, ge=1, le=120)
    jina_rate_limit_per_minute: int = Field(default=20, ge=1, le=120)

    ensemble_consensus_threshold: float = Field(
        default=constants.DEFAULT_CONSENSUS_THRESHOLD,
        ge=constants.MIN_CONSENSUS_THRESHOLD,
        le=constants.MAX_CONSENSUS_THRESHOLD,
    )
    ensemble_voting_mode: VotingMode = VotingMode.MAJORITY
    ensemble_judge_model: str = constants.DEFAULT_JUDGE_MODEL

    primary_fast_model: str = constants.DEFAULT_PRIMARY_FAST_MODEL
    primary_strong_model: str = constants.DEFAULT_PRIMARY_STRONG_MODEL

    fallback_strong_1: str = constants.DEFAULT_FALLBACK_STRONG_1
    fallback_strong_2: str = constants.DEFAULT_FALLBACK_STRONG_2
    fallback_strong_3: str = constants.DEFAULT_FALLBACK_STRONG_3

    fallback_fast_1: str = constants.DEFAULT_FALLBACK_FAST_1
    fallback_fast_2: str = constants.DEFAULT_FALLBACK_FAST_2
    fallback_fast_3: str = constants.DEFAULT_FALLBACK_FAST_3

    code_model: str = constants.DEFAULT_CODE_MODEL
    embedding_model: str = constants.DEFAULT_EMBEDDING_MODEL

    agent_enabled: bool = True

    agent_max_iterations: int = Field(
        default=constants.AGENT_DEFAULT_MAX_ITERATIONS,
        ge=1,
        le=50,
    )

    agent_token_budget: int = Field(
        default=constants.AGENT_DEFAULT_TOKEN_BUDGET,
        ge=1000,
        le=500000,
    )

    agent_fast_model: str = ""
    agent_strong_model: str = ""
    agent_judge_model: str = ""

    agent_memory_enabled: bool = True
    agent_reflection_enabled: bool = True

    agent_max_retries_per_step: int = Field(
        default=constants.AGENT_DEFAULT_MAX_RETRIES_PER_STEP,
        ge=0,
        le=5,
    )

    agent_max_actions_per_step: int = Field(
        default=4,
        ge=1,
        le=10,
    )

    agent_quality_threshold: float = Field(
        default=constants.AGENT_DEFAULT_QUALITY_THRESHOLD,
        ge=0.0,
        le=1.0,
    )

    agent_enable_rag: bool = False

    @field_validator("environment", mode="before")
    @classmethod
    def normalize_environment(cls, value: Any) -> str:
        return str(value or constants.DEFAULT_ENVIRONMENT).strip().lower()

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: Any) -> str:
        candidate = str(value or constants.DEFAULT_LOG_LEVEL).strip().upper()

        if candidate not in constants.VALID_LOG_LEVELS:
            raise ValueError(
                f"log_level must be one of {sorted(constants.VALID_LOG_LEVELS)}"
            )

        return candidate

    @field_validator("output_dir", "cache_dir", "data_dir", "logs_dir", mode="before")
    @classmethod
    def normalize_path(cls, value: Any) -> Path:
        if isinstance(value, Path):
            return value.expanduser()

        if isinstance(value, str):
            return Path(value).expanduser()

        raise ValueError("Path value must be a string or pathlib.Path")

    @field_validator("default_search_platforms", mode="before")
    @classmethod
    def normalize_default_search_platforms(cls, value: Any) -> list[str]:
        if value is None:
            return []

        if isinstance(value, str):
            text = value.strip()

            if not text:
                return []

            if text.startswith("["):
                try:
                    value = json.loads(text)
                except Exception:
                    value = [item.strip() for item in text.split(",") if item.strip()]
            else:
                value = [item.strip() for item in text.split(",") if item.strip()]

        if isinstance(value, (list, tuple, set)):
            items = [str(item).strip().lower() for item in value if str(item).strip()]
        else:
            return []

        valid: list[str] = []

        for item in items:
            try:
                valid.append(SourcePlatform(item).value)
            except Exception:
                continue

        return valid

    @field_validator(
        "ensemble_judge_model",
        "primary_fast_model",
        "primary_strong_model",
        "fallback_strong_1",
        "fallback_strong_2",
        "fallback_strong_3",
        "fallback_fast_1",
        "fallback_fast_2",
        "fallback_fast_3",
        "code_model",
        "embedding_model",
        "agent_fast_model",
        "agent_strong_model",
        "agent_judge_model",
        mode="before",
    )
    @classmethod
    def normalize_model_name(cls, value: Any) -> str:
        return str(value or "").strip()

    @model_validator(mode="after")
    def validate_runtime_settings(self) -> Settings:
        if not self.has_llm_provider:
            raise ValueError("At least one LLM provider API key must be configured")

        return self

    @property
    def llm_provider_keys(self) -> dict[ProviderName, SecretStr]:
        return {
            ProviderName.SAMBANOVA: self.sambanova_api_key,
            ProviderName.GROQ: self.groq_api_key,
            ProviderName.GOOGLE: self.google_api_key,
            ProviderName.MISTRAL: self.mistral_api_key,
            ProviderName.NVIDIA: self.nvidia_api_key,
        }

    @property
    def available_llm_providers(self) -> tuple[ProviderName, ...]:
        return tuple(
            provider
            for provider, key in self.llm_provider_keys.items()
            if key.get_secret_value().strip()
        )

    @property
    def has_llm_provider(self) -> bool:
        return bool(self.available_llm_providers)

    @property
    def fallback_strong_models(self) -> tuple[str, ...]:
        return tuple(
            model
            for model in (
                self.fallback_strong_1,
                self.fallback_strong_2,
                self.fallback_strong_3,
            )
            if model.strip()
        )

    @property
    def fallback_fast_models(self) -> tuple[str, ...]:
        return tuple(
            model
            for model in (
                self.fallback_fast_1,
                self.fallback_fast_2,
                self.fallback_fast_3,
            )
            if model.strip()
        )

    @property
    def strong_model_chain(self) -> tuple[str, ...]:
        return (self.primary_strong_model, *self.fallback_strong_models)

    @property
    def fast_model_chain(self) -> tuple[str, ...]:
        return (self.primary_fast_model, *self.fallback_fast_models)

    @property
    def masked_provider_keys(self) -> dict[str, str]:
        return {
            provider.value: self.mask_secret(key)
            for provider, key in self.llm_provider_keys.items()
        }

    def get_provider_key(self, provider: ProviderName) -> SecretStr:
        return self.llm_provider_keys.get(provider, SecretStr(""))

    def get_provider_key_or_raise(self, provider: ProviderName) -> SecretStr:
        key = self.get_provider_key(provider)

        if not key.get_secret_value().strip():
            raise MissingAPIKeyError(
                "Required provider API key is missing",
                provider=provider.value,
                details={
                    "provider": provider.value,
                    "env_var": constants.ENV_VAR_BY_PROVIDER.get(provider.value),
                },
            )

        return key

    @staticmethod
    def mask_secret(value: SecretStr) -> str:
        raw = value.get_secret_value().strip()

        if not raw:
            return ""

        if len(raw) <= 4:
            return "****"

        return f"****{raw[-4:]}"

    def ensure_directories(self) -> None:
        directories = (
            self.output_dir,
            self.output_dir / constants.OUTPUT_JSON_DIR,
            self.output_dir / constants.OUTPUT_MARKDOWN_DIR,
            self.output_dir / constants.OUTPUT_REPORTS_DIR,
            self.cache_dir,
            self.data_dir,
            self.data_dir / constants.DATA_RAW_DIR,
            self.data_dir / constants.DATA_PROCESSED_DIR,
            self.logs_dir,
        )

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    try:
        settings = Settings()
        settings.ensure_directories()
        return settings
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(
            "Failed to load application settings",
            details={"error": str(exc)},
        ) from exc