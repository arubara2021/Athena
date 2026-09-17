from __future__ import annotations

import json
from typing import Any

from core import constants
from core.exceptions import ConsensusError
from llm.guardrails import validate_llm_request, validate_llm_response
from llm.parser import parse_json_object_response
from llm.prompts import build_consensus_judge_prompt, system_consensus_judge
from llm.provider import LLMProviderManager
from llm.router import get_judge_model_reference
from core.models import ConsensusResult, EnsembleVote, ProviderName
from core.schemas import (
    EnsembleRequestSchema,
    EnsembleResultSchema,
    LLMMessageSchema,
    LLMRequestSchema,
)
from utils.logger import get_logger, get_trace_logger

_FINAL_OUTPUT_KEYS = (
    "final_output",
    "finalOutput",
    "final_output_json",
    "output",
    "answer",
    "result",
    "best_output",
    "selected_output",
    "final_answer",
    "chosen_output",
    "best_answer",
    "selected_answer",
    "final_result",
)

_WRAPPER_KEYS = (
    "judge",
    "response",
    "data",
    "judgement",
    "judgment",
    "judge_response",
    "consensus",
    "consensus_result",
    "payload",
    "body",
)


def _provider_value(value: Any) -> str:
    return str(getattr(value, "value", value))


class ConsensusAggregator:
    def __init__(self, manager: LLMProviderManager | None = None) -> None:
        self._external_manager = manager
        self._manager = manager
        self._owns_manager = manager is None
        self._logger = get_logger("llm.aggregator")
        self._trace = get_trace_logger()

    async def __aenter__(self) -> ConsensusAggregator:
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

    def _validate_final_output(self, task_name: str, output: Any) -> bool:
        normalized = str(task_name or "").lower()

        if "rank" in normalized:
            if isinstance(output, list):
                return len(output) > 0
            if isinstance(output, dict):
                rankings = output.get("rankings")
                return isinstance(rankings, list) and len(rankings) > 0
            return False

        return output is not None

    async def aggregate(
        self,
        request: EnsembleRequestSchema,
        result: EnsembleResultSchema,
    ) -> ConsensusResult:
        successful_votes = [vote for vote in result.votes if vote.success]

        if not successful_votes:
            self._trace.emit("consensus_no_votes", task_id=request.task_id)
            raise ConsensusError(
                "No successful ensemble votes are available",
                details={"task_id": request.task_id},
            )

        internal_votes = self._to_internal_votes(result.votes)

        if (
            result.final_output is not None
            and result.agreement_score >= request.threshold
            and self._validate_final_output(request.name, result.final_output)
        ):
            self._trace.emit(
                "consensus_majority_reached",
                task_id=request.task_id,
                agreement_score=result.agreement_score,
                threshold=request.threshold,
            )
            return ConsensusResult(
                final_output=result.final_output,
                agreement_score=result.agreement_score,
                voting_mode=request.voting_mode,
                votes=internal_votes,
                reasoning="Majority consensus reached",
                judge_used=False,
            )

        judge_reference = request.judge_model
        if judge_reference is None:
            try:
                judge_reference = get_judge_model_reference()
            except Exception:
                judge_reference = None

        if judge_reference is not None and len(successful_votes) >= 2:
            try:
                self._trace.emit(
                    "judge_invoked",
                    task_id=request.task_id,
                    judge_model=judge_reference.model_id,
                    judge_provider=judge_reference.provider.value,
                    vote_count=len(successful_votes),
                )

                final_output, agreement_score, reasoning = await self._run_judge(
                    request, result, judge_reference
                )

                if self._validate_final_output(request.name, final_output):
                    self._trace.emit(
                        "judge_success",
                        task_id=request.task_id,
                        agreement_score=agreement_score,
                    )
                    return ConsensusResult(
                        final_output=final_output,
                        agreement_score=agreement_score,
                        voting_mode=request.voting_mode,
                        votes=internal_votes,
                        reasoning=reasoning,
                        judge_used=True,
                    )

                self._trace.emit(
                    "judge_output_invalid",
                    task_id=request.task_id,
                )
            except Exception as exc:
                self._logger.warning(f"Judge aggregation failed: {exc}")
                self._trace.emit(
                    "judge_failed",
                    task_id=request.task_id,
                    error=str(exc),
                    judge_model=judge_reference.model_id if judge_reference else None,
                )

        self._trace.emit(
            "consensus_fallback",
            task_id=request.task_id,
            reason="judge_failed_or_unavailable",
            fallback_method="weighted_vote_selection",
        )

        fallback_output = self._fallback_output(request, successful_votes)
        return ConsensusResult(
            final_output=fallback_output,
            agreement_score=result.agreement_score,
            voting_mode=request.voting_mode,
            votes=internal_votes,
            reasoning="Fallback selected from available successful votes",
            judge_used=False,
        )

    async def _run_judge(
        self,
        request: EnsembleRequestSchema,
        result: EnsembleResultSchema,
        judge_reference: Any,
    ) -> tuple[Any, float, str]:
        if self._manager is None:
            raise ConsensusError("ConsensusAggregator has no active LLMProviderManager")

        successful_votes = [vote.model_dump() for vote in result.votes if vote.success]

        prompt = build_consensus_judge_prompt(
            task_name=request.name,
            original_prompt=request.prompt,
            votes=successful_votes,
        )

        messages = [
            LLMMessageSchema(role="system", content=system_consensus_judge()),
            LLMMessageSchema(role="user", content=prompt),
        ]

        llm_request = LLMRequestSchema(
            provider=judge_reference.provider,
            model=judge_reference.model_id,
            messages=messages,
            temperature=constants.DEFAULT_TEMPERATURE,
            max_tokens=int(request.context.get("max_tokens", constants.DEFAULT_MAX_TOKENS)),
            response_format="json_object",
            timeout=request.context.get("timeout"),
        )

        llm_request = validate_llm_request(llm_request)
        response = await self._manager.complete(llm_request)
        response = validate_llm_response(response)

        payload = parse_json_object_response(response.content)

        if "final_output" in payload and payload["final_output"] is not None:
            final_output = payload["final_output"]
            raw_score = payload.get("agreement_score", result.agreement_score)
            reasoning = str(payload.get("reasoning", "")).strip()
        else:
            recovered = self._recover_final_output(payload)
            if recovered is None:
                self._trace.emit(
                    "judge_parse_failure",
                    task_id=request.task_id,
                    payload_keys=list(payload.keys()),
                    raw_response_preview=response.content[:500],
                )
                raise ConsensusError(
                    "Judge response did not contain final_output",
                    details={
                        "task_id": request.task_id,
                        "payload_keys": list(payload.keys()),
                    },
                )
            final_output, raw_score, reasoning = recovered
            self._trace.emit(
                "judge_output_recovered",
                task_id=request.task_id,
                recovery_method="fallback_key_or_unwrap",
                payload_keys=list(payload.keys()),
            )

        try:
            agreement_score = float(raw_score)
        except (TypeError, ValueError):
            agreement_score = result.agreement_score

        agreement_score = max(0.0, min(1.0, agreement_score))
        return final_output, agreement_score, reasoning

    def _recover_final_output(
        self,
        payload: dict[str, Any],
    ) -> tuple[Any, Any, str] | None:
        for key in _FINAL_OUTPUT_KEYS:
            if key in payload and payload[key] is not None:
                return (
                    payload[key],
                    payload.get("agreement_score"),
                    str(payload.get("reasoning", "")).strip(),
                )

        candidates: list[dict[str, Any]] = []
        if len(payload) == 1:
            single_value = next(iter(payload.values()))
            if isinstance(single_value, dict):
                candidates.append(single_value)

        for wrapper in _WRAPPER_KEYS:
            nested = payload.get(wrapper)
            if isinstance(nested, dict) and nested not in candidates:
                candidates.append(nested)

        for value in payload.values():
            if isinstance(value, dict) and value not in candidates:
                candidates.append(value)

        for candidate in candidates:
            for key in _FINAL_OUTPUT_KEYS:
                if key in candidate and candidate[key] is not None:
                    return (
                        candidate[key],
                        candidate.get("agreement_score", payload.get("agreement_score")),
                        str(
                            candidate.get("reasoning", payload.get("reasoning", ""))
                        ).strip(),
                    )
            if "rankings" in candidate:
                return (
                    candidate,
                    payload.get("agreement_score"),
                    str(payload.get("reasoning", "")).strip(),
                )

        return None

    def _fallback_output(
        self,
        request: EnsembleRequestSchema,
        successful_votes: list[Any],
    ) -> Any:
        weight_map: dict[tuple[str, str], tuple[float, int]] = {}
        for model_reference in request.models:
            key = (
                _provider_value(model_reference.provider),
                str(model_reference.model_id),
            )
            weight_map[key] = (
                float(model_reference.weight),
                int(model_reference.priority),
            )

        def score_vote(vote: Any) -> tuple[float, float, int]:
            key = (_provider_value(vote.provider), str(vote.model_id))
            weight, priority = weight_map.get(key, (1.0, 100))
            try:
                output_length = len(json.dumps(vote.output, ensure_ascii=False, default=str))
            except Exception:
                output_length = len(str(vote.output))
            return weight, -float(priority), output_length

        best_vote = max(successful_votes, key=score_vote)
        return best_vote.output

    def _to_internal_votes(self, votes: list[Any]) -> list[EnsembleVote]:
        internal_votes: list[EnsembleVote] = []
        for vote in votes:
            try:
                provider = ProviderName(_provider_value(vote.provider))
            except Exception:
                continue
            internal_votes.append(
                EnsembleVote(
                    provider=provider,
                    model_id=str(vote.model_id),
                    output=vote.output,
                    latency_ms=float(vote.latency_ms),
                    success=bool(vote.success),
                    error=vote.error,
                )
            )
        return internal_votes