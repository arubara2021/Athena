from __future__ import annotations

from typing import Any

from core.config import get_settings
from core.models import ModelReference, ModelRole, ProviderName
from utils.logger import get_logger

_logger = get_logger("llm.router")
_trace_logger = None


def _get_trace():
    global _trace_logger
    if _trace_logger is None:
        from utils.logger import get_trace_logger
        _trace_logger = get_trace_logger()
    return _trace_logger


def _detect_provider(model_id: str) -> ProviderName:
    model_lower = model_id.lower()

    if any(k in model_lower for k in ("gemma", "gemini", "google/", "palm")):
        if "google/" in model_lower or "gemini" in model_lower:
            return ProviderName.GOOGLE

    if any(k in model_lower for k in (
        "ministral", "mistral", "codestral", "magistral",
        "open-mistral", "voxtral",
    )):
        return ProviderName.MISTRAL

    if any(k in model_lower for k in (
        "llama", "deepseek", "minimax", "gemma-4",
    )):
        return ProviderName.SAMBANOVA

    if any(k in model_lower for k in (
        "qwen", "compound", "gpt-oss", "allam",
    )):
        return ProviderName.GROQ

    if any(k in model_lower for k in ("nvidia/", "nemotron")):
        return ProviderName.NVIDIA

    return ProviderName.MISTRAL


def _build_reference(
    model_id: str,
    role: ModelRole,
    priority: int,
    weight: float = 1.0,
) -> ModelReference:
    provider = _detect_provider(model_id)
    return ModelReference(
        provider=provider,
        model_id=model_id,
        role=role,
        priority=priority,
        weight=weight,
    )


def _emit_selection(ref: ModelReference) -> None:
    trace = _get_trace()
    trace.emit(
        "model_selected",
        model_id=ref.model_id,
        provider=ref.provider.value,
        role=ref.role.value if hasattr(ref.role, "value") else str(ref.role),
        priority=ref.priority,
    )


def get_fast_model_references(limit: int = 3) -> list[ModelReference]:
    settings = get_settings()
    references: list[ModelReference] = []

    chain = settings.fast_model_chain
    for index, model_id in enumerate(chain):
        if not model_id.strip():
            continue
        if len(references) >= limit:
            break
        ref = _build_reference(
            model_id=model_id,
            role=ModelRole.FALLBACK_FAST if index > 0 else ModelRole.PRIMARY_FAST,
            priority=(index + 1) * 10,
            weight=1.0,
        )
        _emit_selection(ref)
        references.append(ref)

    return references


def get_strong_model_references(limit: int = 3) -> list[ModelReference]:
    settings = get_settings()
    references: list[ModelReference] = []

    chain = settings.strong_model_chain
    for index, model_id in enumerate(chain):
        if not model_id.strip():
            continue
        if len(references) >= limit:
            break
        ref = _build_reference(
            model_id=model_id,
            role=ModelRole.FALLBACK_STRONG if index > 0 else ModelRole.PRIMARY_STRONG,
            priority=(index + 1) * 10,
            weight=1.3 if index == 0 else 1.0,
        )
        _emit_selection(ref)
        references.append(ref)

    return references


def get_judge_model_reference() -> ModelReference:
    settings = get_settings()
    model_id = settings.ensemble_judge_model
    ref = _build_reference(
        model_id=model_id,
        role=ModelRole.JUDGE,
        priority=1,
        weight=1.5,
    )
    _emit_selection(ref)
    return ref


def get_code_model_reference() -> ModelReference:
    settings = get_settings()
    model_id = settings.code_model
    return _build_reference(
        model_id=model_id,
        role=ModelRole.CODE,
        priority=10,
        weight=1.0,
    )


def get_embedding_model_reference() -> ModelReference:
    settings = get_settings()
    model_id = settings.embedding_model
    return _build_reference(
        model_id=model_id,
        role=ModelRole.EMBEDDING,
        priority=20,
        weight=1.0,
    )


def choose_models_for_task(
    task_name: str,
    limit: int = 3,
) -> list[ModelReference]:
    normalized = task_name.strip().lower()

    if normalized in ("ranking", "learning_path", "path_enhancement"):
        return get_strong_model_references(limit=limit)

    if normalized in ("query_expansion", "query_intelligence", "summarization"):
        return get_fast_model_references(limit=limit)

    if normalized == "judge":
        return [get_judge_model_reference()]

    if normalized == "code":
        return [get_code_model_reference()]

    if normalized == "embedding":
        return [get_embedding_model_reference()]

    return get_fast_model_references(limit=limit)


def get_agent_fast_model_reference(budget_remaining: int | None = None) -> ModelReference:
    settings = get_settings()
    agent_fast = getattr(settings, "agent_fast_model", "") or ""

    if agent_fast.strip():
        ref = _build_reference(
            model_id=agent_fast,
            role=ModelRole.PRIMARY_FAST,
            priority=1,
            weight=1.0,
        )
        _emit_selection(ref)
        return ref

    if budget_remaining is not None and budget_remaining < 5000:
        fast_refs = get_fast_model_references(limit=1)
        if fast_refs:
            return fast_refs[0]

    fast_refs = get_fast_model_references(limit=1)
    if fast_refs:
        return fast_refs[0]

    return _build_reference(
        model_id=settings.primary_fast_model,
        role=ModelRole.PRIMARY_FAST,
        priority=1,
        weight=1.0,
    )


def get_agent_strong_model_reference(budget_remaining: int | None = None) -> ModelReference:
    settings = get_settings()
    agent_strong = getattr(settings, "agent_strong_model", "") or ""

    if agent_strong.strip():
        ref = _build_reference(
            model_id=agent_strong,
            role=ModelRole.PRIMARY_STRONG,
            priority=1,
            weight=1.3,
        )
        _emit_selection(ref)
        return ref

    if budget_remaining is not None and budget_remaining < 3000:
        fast_refs = get_fast_model_references(limit=1)
        if fast_refs:
            return fast_refs[0]

    strong_refs = get_strong_model_references(limit=1)
    if strong_refs:
        return strong_refs[0]

    return _build_reference(
        model_id=settings.primary_strong_model,
        role=ModelRole.PRIMARY_STRONG,
        priority=1,
        weight=1.3,
    )


def get_task_model_references(
    task_name: str,
    limit: int = 3,
) -> list[ModelReference]:
    return choose_models_for_task(task_name, limit=limit)