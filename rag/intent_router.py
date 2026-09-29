from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.logger import get_logger
from utils.text import clean_text


_DEFAULT_CONFIG: dict[str, Any] = {
    "model_tier": "fast",
    "model_limit": 3,
    "timeout_seconds": 10.0,
    "max_tokens": 16,
    "temperature": 0.0,
    "retry_count": 1,
    "cache_size": 512,
    "fallback_route": "other",
    "fallback_reason": "llm_unavailable",
}

_CATALOG_LABEL = "catalog"
_TOPICAL_LABEL = "topical"

_CATALOG_ROUTE = "catalog"
_OTHER_ROUTE = "other"

_SYSTEM_PROMPT = (
    "You classify a user's question into exactly one of two categories.\n\n"
    "catalog: the user is asking about the local document corpus itself — "
    "what it contains, what topics or subjects it covers, what sources or "
    "documents it holds, how many items it has, what platforms, domains, "
    "years, runs, or titles it spans. The answer would come from metadata "
    "describing the corpus, not from the content of any document.\n\n"
    "topical: the user is asking about a subject to learn, explain, "
    "compare, research, or implement. The answer would come from the "
    "content of documents, not from a description of the corpus.\n\n"
    "Return exactly one word: catalog or topical.\n"
    "No punctuation. No explanation. No other words."
)


@dataclass(frozen=True)
class CatalogClassification:
    route: str
    score: float
    confidence: float
    threshold: float
    signals: dict[str, int]
    weights: dict[str, float]
    matched: dict[str, list[str]]
    reason: str
    is_catalog: bool


class IntentRouter:
    def __init__(self, config_path: Path | str | None = None) -> None:
        self._logger = get_logger("rag.intent_router")
        self._config_path = Path(config_path) if config_path else (
            Path(__file__).resolve().parent.parent / "configs" / "settings.yaml"
        )
        self._config: dict[str, Any] = dict(_DEFAULT_CONFIG)
        self._cache: OrderedDict[str, CatalogClassification] = OrderedDict()
        self._load_config()

    def _load_config(self) -> None:
        try:
            import yaml
        except Exception as exc:
            self._logger.warning(
                f"PyYAML unavailable for intent router: {exc}"
            )
            return

        try:
            if not self._config_path.exists():
                return

            with open(self._config_path, "r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
        except Exception as exc:
            self._logger.warning(f"Intent router config read failed: {exc}")
            return

        if not isinstance(raw, dict):
            return

        section = raw.get("rag_intent_router", {})

        if not isinstance(section, dict):
            return

        for key, default in _DEFAULT_CONFIG.items():
            if key not in section:
                continue

            value = section[key]

            if value is None:
                continue

            try:
                if isinstance(default, bool):
                    self._config[key] = bool(value)
                elif isinstance(default, int):
                    self._config[key] = int(value)
                elif isinstance(default, float):
                    self._config[key] = float(value)
                else:
                    self._config[key] = str(value)
            except (TypeError, ValueError):
                continue

        tier = str(self._config.get("model_tier", "fast")).strip().lower()

        if tier not in {"fast", "strong"}:
            tier = "fast"

        self._config["model_tier"] = tier
        self._config["model_limit"] = max(
            1, int(self._config.get("model_limit", 3))
        )
        self._config["timeout_seconds"] = max(
            1.0, float(self._config.get("timeout_seconds", 10.0))
        )
        self._config["max_tokens"] = max(
            4, int(self._config.get("max_tokens", 16))
        )
        self._config["temperature"] = max(
            0.0,
            min(
                1.0,
                float(self._config.get("temperature", 0.0)),
            ),
        )
        self._config["retry_count"] = max(
            0, int(self._config.get("retry_count", 1))
        )
        self._config["cache_size"] = max(
            0, int(self._config.get("cache_size", 512))
        )

        fallback_route = str(
            self._config.get("fallback_route", "other")
        ).strip().lower()

        if fallback_route not in {_CATALOG_ROUTE, _OTHER_ROUTE}:
            fallback_route = _OTHER_ROUTE

        self._config["fallback_route"] = fallback_route

        fallback_reason = str(
            self._config.get("fallback_reason", "llm_unavailable")
        ).strip()

        self._config["fallback_reason"] = fallback_reason or "llm_unavailable"

    def clear_cache(self) -> None:
        self._cache.clear()

    async def classify(self, question: str) -> CatalogClassification:
        cleaned = clean_text(question)

        if not cleaned:
            return self._empty_result("empty_question")

        key = self._cache_key(cleaned)
        cached = self._cache_lookup(key)

        if cached is not None:
            return cached

        result = await self._classify_via_llm(cleaned)

        if result is None:
            result = self._fallback_result()

        self._cache_store(key, result)
        return result

    async def route_with_confidence(
        self,
        question: str,
    ) -> CatalogClassification:
        return await self.classify(question)

    async def route(self, question: str) -> tuple[str, dict[str, Any]]:
        result = await self.classify(question)

        meta: dict[str, Any] = {
            "score": result.score,
            "confidence": result.confidence,
            "threshold": result.threshold,
            "signals": dict(result.signals),
            "weights": dict(result.weights),
            "matched": {
                key: list(value) for key, value in result.matched.items()
            },
            "reason": result.reason,
        }

        return result.route, meta

    def _cache_key(self, question: str) -> str:
        text = clean_text(question).lower()
        return text.strip(" .,;:!?\"'`")

    def _cache_lookup(self, key: str) -> CatalogClassification | None:
        if not key:
            return None

        if key not in self._cache:
            return None

        value = self._cache.pop(key)
        self._cache[key] = value
        return value

    def _cache_store(
        self,
        key: str,
        value: CatalogClassification,
    ) -> None:
        if not key:
            return

        max_size = int(self._config.get("cache_size", 0) or 0)

        if max_size <= 0:
            return

        if key in self._cache:
            self._cache.pop(key)

        self._cache[key] = value

        while len(self._cache) > max_size:
            self._cache.popitem(last=False)

    async def _classify_via_llm(
        self,
        question: str,
    ) -> CatalogClassification | None:
        try:
            from core.schemas import LLMMessageSchema, LLMRequestSchema
            from llm.guardrails import (
                validate_llm_request,
                validate_llm_response,
            )
            from llm.provider import LLMProviderManager
            from llm.router import (
                get_fast_model_references,
                get_strong_model_references,
            )
        except Exception as exc:
            self._logger.warning(
                f"Intent router LLM imports failed: {exc}"
            )
            return None

        tier = str(self._config.get("model_tier", "fast")).lower()
        limit = int(self._config.get("model_limit", 3))

        if tier == "strong":
            try:
                references = get_strong_model_references(limit=limit)
            except Exception:
                references = []
        else:
            try:
                references = get_fast_model_references(limit=limit)
            except Exception:
                references = []

        if not references:
            return None

        messages = [
            LLMMessageSchema(role="system", content=_SYSTEM_PROMPT),
            LLMMessageSchema(role="user", content=question),
        ]

        attempts = int(self._config.get("retry_count", 1)) + 1
        timeout = float(self._config.get("timeout_seconds", 10.0))
        max_tokens = int(self._config.get("max_tokens", 16))
        temperature = float(self._config.get("temperature", 0.0))

        for _ in range(max(1, attempts)):
            try:
                async with LLMProviderManager() as manager:
                    for reference in references:
                        label = await self._try_reference(
                            manager=manager,
                            reference=reference,
                            messages=messages,
                            timeout=timeout,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            validate_request=validate_llm_request,
                            validate_response=validate_llm_response,
                            request_schema=LLMRequestSchema,
                        )

                        if label is None:
                            continue

                        return self._build_result(label)
            except Exception as exc:
                self._logger.warning(
                    f"Intent router LLM attempt failed: {exc}"
                )
                continue

        return None

    async def _try_reference(
        self,
        manager: Any,
        reference: Any,
        messages: list[Any],
        timeout: float,
        max_tokens: int,
        temperature: float,
        validate_request: Any,
        validate_response: Any,
        request_schema: Any,
    ) -> str | None:
        try:
            request = request_schema(
                provider=reference.provider,
                model=reference.model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format="text",
            )
            request = validate_request(request)

            response = await asyncio.wait_for(
                manager.complete(request),
                timeout=timeout,
            )

            if response is None:
                return None

            response = validate_response(response)
            return self._parse_label(getattr(response, "content", ""))

        except Exception as exc:
            self._logger.warning(
                f"Intent router model "
                f"{getattr(reference, 'model_id', '?')} failed: {exc}"
            )
            return None

    def _parse_label(self, content: str) -> str | None:
        if not content:
            return None

        text = clean_text(content).lower()
        text = text.strip(" .,;:!?\"'`\n\r\t")

        if not text:
            return None

        first_token = text.split()[0] if text.split() else ""

        if first_token == _CATALOG_LABEL:
            return _CATALOG_LABEL

        if first_token == _TOPICAL_LABEL:
            return _TOPICAL_LABEL

        return None

    def _build_result(self, label: str) -> CatalogClassification:
        is_catalog = label == _CATALOG_LABEL

        return CatalogClassification(
            route=_CATALOG_ROUTE if is_catalog else _OTHER_ROUTE,
            score=0.0,
            confidence=1.0,
            threshold=0.0,
            signals={},
            weights={},
            matched={},
            reason=f"llm_classified_{label}",
            is_catalog=is_catalog,
        )

    def _fallback_result(self) -> CatalogClassification:
        route = str(self._config.get("fallback_route", _OTHER_ROUTE))
        reason = str(self._config.get("fallback_reason", "llm_unavailable"))

        return CatalogClassification(
            route=route,
            score=0.0,
            confidence=0.0,
            threshold=0.0,
            signals={},
            weights={},
            matched={},
            reason=reason,
            is_catalog=route == _CATALOG_ROUTE,
        )

    def _empty_result(self, reason: str) -> CatalogClassification:
        return CatalogClassification(
            route=_OTHER_ROUTE,
            score=0.0,
            confidence=0.0,
            threshold=0.0,
            signals={},
            weights={},
            matched={},
            reason=reason,
            is_catalog=False,
        )