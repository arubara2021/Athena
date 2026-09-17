from __future__ import annotations
import asyncio
import json
import time
from pathlib import Path
from typing import Any
from core import constants
from core.exceptions import EnsembleError
from llm.guardrails import validate_llm_request, validate_llm_response
from llm.parser import parse_json_response
from llm.provider import LLMProviderManager
from core.models import EnsembleVote, ModelReference
from core.schemas import (
    EnsembleRequestSchema,
    EnsembleResultSchema,
    EnsembleVoteSchema,
    LLMMessageSchema,
    LLMRequestSchema,
)
from utils.hashing import stable_hash
from utils.logger import get_logger, get_trace_logger

_MIN_VOTES_CACHE: int | None = None

def _load_min_consensus_votes() -> int:
    global _MIN_VOTES_CACHE
    if _MIN_VOTES_CACHE is not None:
        return _MIN_VOTES_CACHE
    value = 2
    try:
        import yaml
        config_path = Path(__file__).resolve().parent.parent / "configs" / "ensemble.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
                if isinstance(loaded, dict):
                    raw = loaded.get("min_votes_for_consensus")
                    parsed = int(raw)
                    if parsed >= 1:
                        value = parsed
    except Exception:
        pass
    _MIN_VOTES_CACHE = value
    return value

class EnsembleEngine:
    def __init__(self, manager: LLMProviderManager | None = None) -> None:
        self._external_manager = manager
        self._manager = manager
        self._owns_manager = manager is None
        self._logger = get_logger("llm.ensemble")
        self._trace = get_trace_logger()

    async def __aenter__(self) -> EnsembleEngine:
        if self._manager is None:
            self._manager = LLMProviderManager()
            await self._manager.__aenter__()
            self._owns_manager = True
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self._owns_manager and self._manager is not None:
            await self._manager.__aexit__(exc_type, exc, tb)
            self._manager = self._external_manager
        return False

    async def run(self, request: EnsembleRequestSchema) -> EnsembleResultSchema:
        if self._manager is None:
            raise EnsembleError("EnsembleEngine has no active LLMProviderManager")
        validated_request = EnsembleRequestSchema.model_validate(request)
        self._trace.emit(
            "ensemble_start",
            task_id=validated_request.task_id,
            model_count=len(validated_request.models),
            models=[
                {"provider": m.provider.value, "model_id": m.model_id}
                for m in validated_request.models
            ],
        )
        tasks = [
            self._run_model(validated_request, model_reference)
            for model_reference in validated_request.models
        ]
        votes = await asyncio.gather(*tasks)
        successful_votes = [vote for vote in votes if vote.success]
        failed_votes = [vote for vote in votes if not vote.success]
        self._trace.emit(
            "ensemble_votes_collected",
            task_id=validated_request.task_id,
            total=len(votes),
            successful=len(successful_votes),
            failed=len(failed_votes),
            failures=[
                {"model_id": v.model_id, "error": v.error}
                for v in failed_votes
            ],
        )
        if not successful_votes:
            self._trace.emit(
                "ensemble_all_failed",
                task_id=validated_request.task_id,
            )
            raise EnsembleError(
                "All ensemble models failed",
                details={
                    "task_id": validated_request.task_id,
                    "errors": [vote.error for vote in votes if vote.error],
                },
            )
        min_votes = _load_min_consensus_votes()
        final_output, agreement_score = self._majority_output(
            successful_votes,
            validated_request.threshold,
            min_votes,
        )
        self._trace.emit(
            "ensemble_result",
            task_id=validated_request.task_id,
            agreement_score=agreement_score,
            consensus_reached=final_output is not None,
            threshold=validated_request.threshold,
            successful_votes=len(successful_votes),
            total_votes=len(votes),
            min_votes_required=min_votes,
        )
        return EnsembleResultSchema(
            task_id=validated_request.task_id,
            success=True,
            final_output=final_output,
            agreement_score=agreement_score,
            votes=[
                EnsembleVoteSchema(
                    provider=vote.provider,
                    model_id=vote.model_id,
                    output=vote.output,
                    latency_ms=vote.latency_ms,
                    success=vote.success,
                    error=vote.error,
                )
                for vote in votes
            ],
            judge_used=False,
            reasoning=None,
        )

    async def _run_model(
        self,
        request: EnsembleRequestSchema,
        model_reference: ModelReference,
    ) -> EnsembleVote:
        start = time.perf_counter()
        if self._manager is None:
            return EnsembleVote(
                provider=model_reference.provider,
                model_id=model_reference.model_id,
                output=None,
                latency_ms=0.0,
                success=False,
                error="LLMProviderManager is not initialized",
            )
        try:
            messages = self._build_messages(request)
            llm_request = LLMRequestSchema(
                provider=model_reference.provider,
                model=model_reference.model_id,
                messages=messages,
                temperature=constants.DEFAULT_TEMPERATURE,
                max_tokens=int(request.context.get("max_tokens", constants.DEFAULT_MAX_TOKENS)),
                response_format=request.context.get("response_format", "json_object"),
                timeout=request.context.get("timeout"),
            )
            llm_request = validate_llm_request(llm_request)
            response = await self._manager.complete(llm_request)
            response = validate_llm_response(response)
            output = self._parse_output(response.content, request)
            latency_ms = response.latency_ms or (time.perf_counter() - start) * 1000
            return EnsembleVote(
                provider=model_reference.provider,
                model_id=model_reference.model_id,
                output=output,
                latency_ms=latency_ms,
                success=True,
                error=None,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return EnsembleVote(
                provider=model_reference.provider,
                model_id=model_reference.model_id,
                output=None,
                latency_ms=latency_ms,
                success=False,
                error=str(exc),
            )

    def _build_messages(self, request: EnsembleRequestSchema) -> list[LLMMessageSchema]:
        system_prompt = str(request.context.get("system_prompt", "")).strip()
        excluded_context_keys = {
            "system_prompt",
            "expect_json",
            "max_tokens",
            "timeout",
            "response_format",
        }
        extra_context = {
            key: value
            for key, value in request.context.items()
            if key not in excluded_context_keys
        }
        user_content = request.prompt
        if extra_context:
            user_content = (
                f"{request.prompt}\n"
                f"Context:\n"
                f"{json.dumps(extra_context, ensure_ascii=False, indent=2, default=str)}"
            )
        messages: list[LLMMessageSchema] = []
        if system_prompt:
            messages.append(LLMMessageSchema(role="system", content=system_prompt))
        messages.append(LLMMessageSchema(role="user", content=user_content))
        return messages

    def _parse_output(self, content: str, request: EnsembleRequestSchema) -> Any:
        expect_json = bool(request.context.get("expect_json", True))
        if not expect_json:
            return content
        try:
            return parse_json_response(content)
        except Exception:
            return content

    def _majority_output(
        self,
        votes: list[EnsembleVote],
        threshold: float,
        min_votes: int,
    ) -> tuple[Any, float]:
        if not votes:
            return None, 0.0
        groups: dict[str, dict[str, Any]] = {}
        for vote in votes:
            fingerprint = self._vote_fingerprint(vote.output)
            entry = groups.get(fingerprint)
            if entry is None:
                groups[fingerprint] = {
                    "count": 1,
                    "output": vote.output,
                }
            else:
                entry["count"] += 1
        best = max(groups.values(), key=lambda item: item["count"])
        agreement_score = best["count"] / len(votes)
        if len(votes) < min_votes:
            return None, agreement_score
        if agreement_score >= threshold:
            return best["output"], agreement_score
        return None, agreement_score

    def _vote_fingerprint(self, output: Any) -> str:
        rankings: Any = None
        if hasattr(output, "model_dump"):
            output = output.model_dump()
        if isinstance(output, dict):
            rankings = output.get("rankings")
        elif isinstance(output, list):
            rankings = output

        if isinstance(rankings, list) and rankings:
            ids: list[str] = []
            for item in rankings:
                if hasattr(item, "model_dump"):
                    item = item.model_dump()
                if isinstance(item, dict):
                    ids.append(str(item.get("source_id", "")))
            if ids:
                return stable_hash("|".join(ids))

        try:
            return stable_hash(output)
        except Exception:
            return stable_hash(str(output))