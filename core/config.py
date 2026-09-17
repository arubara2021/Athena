from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from core import constants
from core.settings import Settings, load_settings
from core.exceptions import ConfigError
from core.models import ProviderName


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def reload_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_output_directory(subdirectory: str | None = None) -> Path:
    settings = get_settings()
    path = settings.output_dir if subdirectory is None else settings.output_dir / subdirectory
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_json_output_directory() -> Path:
    return get_output_directory(constants.OUTPUT_JSON_DIR)


def get_markdown_output_directory() -> Path:
    return get_output_directory(constants.OUTPUT_MARKDOWN_DIR)


def get_reports_output_directory() -> Path:
    return get_output_directory(constants.OUTPUT_REPORTS_DIR)


def get_cache_directory(subdirectory: str | None = None) -> Path:
    settings = get_settings()
    path = settings.cache_dir if subdirectory is None else settings.cache_dir / subdirectory
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_directory(subdirectory: str | None = None) -> Path:
    settings = get_settings()
    path = settings.data_dir if subdirectory is None else settings.data_dir / subdirectory
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_raw_data_directory() -> Path:
    return get_data_directory(constants.DATA_RAW_DIR)


def get_processed_data_directory() -> Path:
    return get_data_directory(constants.DATA_PROCESSED_DIR)


def get_logs_directory() -> Path:
    settings = get_settings()
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    return settings.logs_dir


def get_provider_api_key(provider: ProviderName) -> str:
    return get_settings().get_provider_key(provider).get_secret_value()


def get_provider_api_key_or_raise(provider: ProviderName) -> str:
    return get_settings().get_provider_key_or_raise(provider).get_secret_value()


def has_provider_api_key(provider: ProviderName) -> bool:
    return bool(get_settings().get_provider_key(provider).get_secret_value().strip())


def get_available_llm_providers() -> tuple[ProviderName, ...]:
    return get_settings().available_llm_providers


def get_primary_fast_model() -> str:
    return get_settings().primary_fast_model


def get_primary_strong_model() -> str:
    return get_settings().primary_strong_model


def get_judge_model() -> str:
    return get_settings().ensemble_judge_model


def get_code_model() -> str:
    return get_settings().code_model


def get_embedding_model() -> str:
    return get_settings().embedding_model


def get_strong_model_chain() -> tuple[str, ...]:
    settings = get_settings()
    return (settings.primary_strong_model, *settings.fallback_strong_models)


def get_fast_model_chain() -> tuple[str, ...]:
    settings = get_settings()
    return (settings.primary_fast_model, *settings.fallback_fast_models)


def get_provider_base_url(provider: ProviderName) -> str:
    try:
        return constants.PROVIDER_BASE_URLS[provider.value]
    except KeyError as exc:
        raise ConfigError(
            "Unsupported provider base URL requested",
            details={"provider": provider.value},
        ) from exc


def get_agent_config() -> dict[str, Any]:
    settings = get_settings()

    values = {
        "enabled": getattr(settings, "agent_enabled", True),
        "max_iterations": getattr(settings, "agent_max_iterations", constants.AGENT_DEFAULT_MAX_ITERATIONS),
        "max_retries_per_step": getattr(settings, "agent_max_retries_per_step", constants.AGENT_DEFAULT_MAX_RETRIES_PER_STEP),
        "max_actions_per_step": getattr(settings, "agent_max_actions_per_step", 4),
        "fast_model": getattr(settings, "agent_fast_model", "") or "",
        "strong_model": getattr(settings, "agent_strong_model", "") or "",
        "judge_model": getattr(settings, "agent_judge_model", "") or "",
        "memory_enabled": getattr(settings, "agent_memory_enabled", True),
        "reflection_enabled": getattr(settings, "agent_reflection_enabled", True),
        "token_budget": getattr(settings, "agent_token_budget", constants.AGENT_DEFAULT_TOKEN_BUDGET),
        "quality_threshold": getattr(settings, "agent_quality_threshold", constants.AGENT_DEFAULT_QUALITY_THRESHOLD),
        "enable_rag": getattr(settings, "agent_enable_rag", False),
    }

    try:
        import yaml

        config_path = get_project_root() / "configs" / "agent_config.yaml"

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}

            if isinstance(raw, dict):
                section = raw.get("agent", raw)

                if isinstance(section, dict):
                    for key in values:
                        if key in section:
                            values[key] = section[key]
    except Exception:
        pass

    try:
        values["max_iterations"] = int(values["max_iterations"])
    except Exception:
        values["max_iterations"] = constants.AGENT_DEFAULT_MAX_ITERATIONS

    try:
        values["max_retries_per_step"] = int(values["max_retries_per_step"])
    except Exception:
        values["max_retries_per_step"] = constants.AGENT_DEFAULT_MAX_RETRIES_PER_STEP

    try:
        values["max_actions_per_step"] = int(values["max_actions_per_step"])
    except Exception:
        values["max_actions_per_step"] = 4

    try:
        values["token_budget"] = int(values["token_budget"])
    except Exception:
        values["token_budget"] = constants.AGENT_DEFAULT_TOKEN_BUDGET

    try:
        values["quality_threshold"] = float(values["quality_threshold"])
    except Exception:
        values["quality_threshold"] = constants.AGENT_DEFAULT_QUALITY_THRESHOLD

    values["enabled"] = bool(values["enabled"])
    values["memory_enabled"] = bool(values["memory_enabled"])
    values["reflection_enabled"] = bool(values["reflection_enabled"])
    values["enable_rag"] = bool(values["enable_rag"])

    values["fast_model"] = str(values["fast_model"] or "").strip()
    values["strong_model"] = str(values["strong_model"] or "").strip()
    values["judge_model"] = str(values["judge_model"] or "").strip()

    if not values["fast_model"]:
        values["fast_model"] = settings.primary_fast_model

    if not values["strong_model"]:
        values["strong_model"] = settings.primary_strong_model

    if not values["judge_model"]:
        values["judge_model"] = settings.ensemble_judge_model

    return values


def get_token_budget_config() -> dict[str, Any]:
    try:
        import yaml
        config_path = get_project_root() / "configs" / "token_budgets.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
            if isinstance(raw, dict):
                return raw
    except Exception:
        pass

    return {
        "defaults": {
            "total_budget_per_task": constants.AGENT_DEFAULT_TOKEN_BUDGET,
            "wrap_up_threshold": 5000,
            "wrap_up_mode_budget": constants.AGENT_WRAP_UP_MODE_BUDGET,
            "currency": "tokens",
            "allow_burst": False,
            "burst_multiplier": 1.15,
        },
        "categories": {
            "planning": {
                "max_tokens": constants.AGENT_PLANNING_BUDGET,
                "priority": 1,
                "model_tier": "fast",
            },
            "search": {
                "max_tokens": constants.AGENT_SEARCH_BUDGET,
                "priority": 3,
                "model_tier": "fast",
            },
            "reading": {
                "max_tokens": constants.AGENT_READING_BUDGET,
                "priority": 4,
                "model_tier": "fast",
            },
            "ranking": {
                "max_tokens": constants.AGENT_RANKING_BUDGET,
                "priority": 5,
                "model_tier": "strong",
            },
            "summarization": {
                "max_tokens": constants.AGENT_SUMMARIZATION_BUDGET,
                "priority": 6,
                "model_tier": "fast",
            },
            "reflection": {
                "max_tokens": constants.AGENT_REFLECTION_BUDGET,
                "priority": 9,
                "model_tier": "fast",
            },
            "synthesis": {
                "max_tokens": constants.AGENT_SYNTHESIS_BUDGET,
                "priority": 10,
                "model_tier": "strong",
            },
        },
        "model_tiers": {
            "fast": {
                "cost_weight": 1.0,
                "max_tokens_per_call": 2048,
                "preferred_providers": ["GROQ", "MISTRAL"],
            },
            "strong": {
                "cost_weight": 2.5,
                "max_tokens_per_call": 4096,
                "preferred_providers": ["MISTRAL", "SAMBANOVA"],
            },
            "judge": {
                "cost_weight": 3.0,
                "max_tokens_per_call": 4096,
                "preferred_providers": ["MISTRAL"],
            },
        },
        "circuit_breaker": {
            "enabled": True,
            "warning_threshold_percent": constants.AGENT_WRAP_UP_THRESHOLD_PERCENT,
            "critical_threshold_percent": constants.AGENT_CRITICAL_THRESHOLD_PERCENT,
            "cooldown_seconds": 0,
            "max_consecutive_exceedances": 3,
        },
        "ledger": {
            "filename": "token_ledger.jsonl",
            "flush_interval_entries": 10,
            "max_entries_in_memory": 10000,
            "include_raw_response": False,
            "include_prompt_preview_length": 100,
        },
        "reporting": {
            "include_per_model_breakdown": True,
            "include_per_category_breakdown": True,
            "include_timeline": False,
            "save_report_to_file": True,
            "report_filename_prefix": "token_usage_report",
        },
    }