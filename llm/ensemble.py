from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from core import constants
from core.exceptions import EnsembleError
from core.models import CoreModel, EnsembleVote, ModelReference
from core.schemas import (
    EnsembleRequestSchema,
    EnsembleResultSchema,
    EnsembleVoteSchema,
    LLMMessageSchema,
    LLMRequestSchema,
)
from llm.guardrails import validate_llm_request, validate_llm_response
from llm.parser import parse_json_response
from llm.provider import LLMProviderManager
from utils.hashing import stable_hash
from utils.logger import get_logger, get_trace_logger


class VoteFingerprint(CoreModel):
    strict: str = ""
    loose: str = ""
    reasoning: str = ""
    has_rankings: bool = False


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

    async def __aexit__(
        self,
        exc_type: Any,
        exc: Any,
        tb: Any,
    ) -> bool:
        if self._owns_manager and self._manager is not None:
            await self._manager.__aexit__(exc_type, exc, tb)
            self._manager = self._external_manager
        return False

    async def run(
        self,
        request: EnsembleRequestSchema,
    ) -> EnsembleResultSchema:
        if self._manager is None:
            raise EnsembleError(
                "EnsembleEngine has no active LLMProviderManager"
            )

        validated_request = EnsembleRequestSchema.model_validate(request)

        self._trace.emit(
            "ensemble_start",
            task_id=validated_request.task_id,
            model_count=len(validated_request.models),
            models=[
                {
                    "provider": m.provider.value,
                    "model_id": m.model_id,
                }
                for m in validated_request.models
            ],
        )

        if len(validated_request.models) == 1:
            single_vote = await self._run_model(
                validated_request, validated_request.models[0]
            )
            votes = [single_vote]
        else:
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

        min_votes = 2
        threshold = float(validated_request.threshold)

        (
            final_output,
            agreement_score,
            agreement_mode,
        ) = self._majority_output(
            successful_votes,
            threshold,
            min_votes,
        )

        self._trace.emit(
            "ensemble_result",
            task_id=validated_request.task_id,
            agreement_score=round(agreement_score, 4),
            agreement_mode=agreement_mode,
            consensus_reached=final_output is not None,
            threshold=threshold,
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
            reasoning=f"agreement_mode={agreement_mode}",
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
                max_tokens=int(
                    request.context.get(
                        "max_tokens",
                        constants.DEFAULT_MAX_TOKENS,
                    )
                ),
                response_format=request.context.get(
                    "response_format",
                    "json_object",
                ),
                timeout=request.context.get("timeout"),
            )
            llm_request = validate_llm_request(llm_request)
            response = await self._manager.complete(llm_request)
            response = validate_llm_response(response)
            output = self._parse_output(response.content, request)
            latency_ms = response.latency_ms or (
                time.perf_counter() - start
            ) * 1000

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

    def _build_messages(
        self,
        request: EnsembleRequestSchema,
    ) -> list[LLMMessageSchema]:
        system_prompt = str(
            request.context.get("system_prompt", "")
        ).strip()

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
            messages.append(
                LLMMessageSchema(role="system", content=system_prompt)
            )

        messages.append(LLMMessageSchema(role="user", content=user_content))

        return messages

    def _parse_output(
        self,
        content: str,
        request: EnsembleRequestSchema,
    ) -> Any:
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
    ) -> tuple[Any, float, str]:
        if not votes:
            return None, 0.0, "none"

        strict_groups: dict[str, dict[str, Any]] = {}
        loose_groups: dict[str, dict[str, Any]] = {}

        for vote in votes:
            fingerprint = self._vote_fingerprint(vote.output)

            strict_entry = strict_groups.get(fingerprint.strict)

            if strict_entry is None:
                strict_groups[fingerprint.strict] = {
                    "count": 1,
                    "output": vote.output,
                }
            else:
                strict_entry["count"] += 1

            loose_entry = loose_groups.get(fingerprint.loose)

            if loose_entry is None:
                loose_groups[fingerprint.loose] = {
                    "count": 1,
                    "output": vote.output,
                }
            else:
                loose_entry["count"] += 1

        total = len(votes)

        if total < min_votes:
            best = max(
                strict_groups.values(),
                key=lambda item: item["count"],
            )
            return None, best["count"] / total, "insufficient"

        best_strict = max(
            strict_groups.values(),
            key=lambda item: item["count"],
        )
        strict_agreement = best_strict["count"] / total

        if strict_agreement >= threshold:
            return best_strict["output"], strict_agreement, "strict"

        best_loose = max(
            loose_groups.values(),
            key=lambda item: item["count"],
        )
        loose_agreement = best_loose["count"] / total

        if loose_agreement >= threshold:
            return best_loose["output"], loose_agreement, "loose"

        return None, strict_agreement, "disagreement"

    def _vote_fingerprint(self, output: Any) -> VoteFingerprint:
        rankings: Any = None
        normalized = output

        if hasattr(normalized, "model_dump"):
            try:
                normalized = normalized.model_dump(mode="json")
            except Exception:
                pass

        if isinstance(normalized, dict):
            rankings = normalized.get("rankings")
        elif isinstance(normalized, list):
            rankings = normalized

        if isinstance(rankings, list) and rankings:
            strict_parts: list[str] = []
            loose_parts: list[str] = []
            reasoning_parts: list[str] = []

            for item in rankings:
                if hasattr(item, "model_dump"):
                    try:
                        item = item.model_dump(mode="json")
                    except Exception:
                        continue

                if not isinstance(item, dict):
                    continue

                source_id = str(item.get("source_id", "") or "").strip()
                score = item.get("score")
                reason = item.get("reason")

                try:
                    score_num = (
                        float(score) if score is not None else 0.0
                    )
                except (TypeError, ValueError):
                    score_num = 0.0

                strict_parts.append(
                    f"{source_id}:{round(score_num, 2):.2f}"
                )
                loose_parts.append(source_id)
                reasoning_parts.append(str(reason or ""))

            if strict_parts:
                strict = stable_hash("|".join(strict_parts))
                loose = stable_hash("|".join(loose_parts))
                reasoning = stable_hash("|".join(reasoning_parts))

                return VoteFingerprint(
                    strict=strict,
                    loose=loose,
                    reasoning=reasoning,
                    has_rankings=True,
                )

        try:
            fallback = stable_hash(normalized)
        except Exception:
            fallback = stable_hash(str(normalized))

        return VoteFingerprint(
            strict=fallback,
            loose=fallback,
            reasoning=fallback,
            has_rankings=False,
        )