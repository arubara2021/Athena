from __future__ import annotations

import asyncio
import random
import time
from typing import Any

import httpx

from core import constants
from core.config import get_settings
from core.exceptions import (
    ModelNotFoundError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from core.models import ProviderName
from core.schemas import LLMRequestSchema, LLMResponseSchema
from utils.logger import get_logger, get_trace_logger
from utils.rate_limiter import default_rate_limiter_registry


class LLMProviderManager:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._logger = get_logger("llm.provider")
        self._trace = get_trace_logger()
        self._client: httpx.AsyncClient | None = None
        self._rate_limiters: dict[str, Any] = {}
        self._token_hooks: list[Any] = []

    async def __aenter__(self) -> LLMProviderManager:
        self._client = httpx.AsyncClient(
            timeout=None,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=5),
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        await self.close()
        return False

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def register_token_hook(self, hook: Any) -> None:
        self._token_hooks.append(hook)

    def unregister_token_hook(self, hook: Any) -> None:
        try:
            self._token_hooks.remove(hook)
        except ValueError:
            pass

    def _get_rate_limiter(self, provider_value: str) -> Any:
        limiter = self._rate_limiters.get(provider_value)
        if limiter is not None:
            return limiter
        limiter = default_rate_limiter_registry.get(
            name=provider_value,
            rate_limit=self._settings.llm_rate_limit_per_minute,
            period_seconds=60.0,
        )
        self._rate_limiters[provider_value] = limiter
        return limiter

    async def complete(self, request: LLMRequestSchema) -> LLMResponseSchema | None:
        provider_value = request.provider.value
        base_url = constants.PROVIDER_BASE_URLS.get(provider_value)
        if base_url is None:
            raise ProviderError(
                "Unsupported provider",
                provider=provider_value,
                details={"provider": provider_value},
            )

        api_key = self._settings.get_provider_key_or_raise(request.provider)
        rate_limiter = self._get_rate_limiter(provider_value)
        max_attempts = max(1, self._settings.llm_max_retries + 1)

        self._trace.emit(
            "llm_call_start",
            provider=provider_value,
            model=request.model,
            max_attempts=max_attempts,
        )

        for attempt in range(1, max_attempts + 1):
            await rate_limiter.acquire()
            start = time.perf_counter()

            try:
                response = await self._make_request(
                    base_url=base_url,
                    api_key=api_key.get_secret_value(),
                    request=request,
                    provider_value=provider_value,
                )
            except (ProviderAuthError, ModelNotFoundError):
                raise
            except ProviderRateLimitError:
                if attempt < max_attempts:
                    await asyncio.sleep(self._calculate_backoff(attempt))
                    continue
                raise
            except ProviderTimeoutError:
                if attempt < max_attempts:
                    await asyncio.sleep(self._calculate_backoff(attempt))
                    continue
                raise
            except ProviderResponseError as exc:
                status = exc.details.get("status_code") if exc.details else None
                if status in (429, 500, 502, 503, 504) and attempt < max_attempts:
                    await asyncio.sleep(self._calculate_backoff(attempt))
                    continue
                raise
            except ProviderError:
                if attempt < max_attempts:
                    await asyncio.sleep(self._calculate_backoff(attempt))
                    continue
                raise

            latency_ms = (time.perf_counter() - start) * 1000

            if response.status_code == 200:
                try:
                    data = response.json()
                except Exception as exc:
                    raise ProviderResponseError(
                        "Invalid JSON response from provider",
                        provider=provider_value,
                        details={"error": str(exc)},
                    ) from exc

                content = self._extract_content(data, provider_value)

                if not content.strip():
                    if attempt < max_attempts:
                        await asyncio.sleep(self._calculate_backoff(attempt))
                        continue
                    raise ProviderResponseError(
                        "Provider returned empty content",
                        provider=provider_value,
                        status_code=200,
                        details={"model": request.model},
                    )

                tokens_used = self._extract_tokens(data, provider_value)
                if tokens_used <= 0:
                    tokens_used = self._estimate_tokens(content)

                input_tokens = self._extract_input_tokens(data, provider_value)
                output_tokens = self._extract_output_tokens(data, provider_value)

                self._trace.emit(
                    "llm_call_success",
                    provider=provider_value,
                    model=request.model,
                    status=response.status_code,
                    attempt=attempt,
                    latency_ms=round(latency_ms, 1),
                    content_length=len(content),
                    tokens_used=tokens_used,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

                self._report_token_usage(
                    provider=provider_value,
                    model=request.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=tokens_used,
                    latency_ms=latency_ms,
                )

                return LLMResponseSchema(
                    provider=request.provider,
                    model=request.model,
                    content=content,
                    latency_ms=latency_ms,
                    tokens_used=tokens_used,
                    raw=data,
                )

            if response.status_code in (401, 403):
                self._trace.emit(
                    "llm_call_failed",
                    provider=provider_value,
                    model=request.model,
                    status=response.status_code,
                    error="ProviderAuthError",
                    attempt=attempt,
                    latency_ms=round(latency_ms, 1),
                )
                raise ProviderAuthError(
                    "Provider authentication failed",
                    provider=provider_value,
                    status_code=response.status_code,
                    details={"body": response.text[:500]},
                )

            if response.status_code == 404:
                self._trace.emit(
                    "llm_call_failed",
                    provider=provider_value,
                    model=request.model,
                    status=response.status_code,
                    error="ModelNotFoundError",
                    attempt=attempt,
                    latency_ms=round(latency_ms, 1),
                )
                raise ModelNotFoundError(
                    "Requested model or endpoint was not found",
                    provider=provider_value,
                    status_code=response.status_code,
                    details={"body": response.text[:500]},
                )

            if response.status_code == 402:
                if attempt < max_attempts:
                    await asyncio.sleep(self._calculate_backoff(attempt))
                    continue
                raise ProviderRateLimitError(
                    "Provider quota exceeded",
                    provider=provider_value,
                    status_code=402,
                    details={"body": response.text[:500]},
                )

            if response.status_code == 429:
                if attempt < max_attempts:
                    await asyncio.sleep(self._calculate_backoff(attempt))
                    continue
                raise ProviderRateLimitError(
                    "Provider rate limit exceeded",
                    provider=provider_value,
                    status_code=response.status_code,
                    details={"body": response.text[:500]},
                )

            if response.status_code >= 500:
                if attempt < max_attempts:
                    await asyncio.sleep(self._calculate_backoff(attempt))
                    continue
                self._trace.emit(
                    "llm_call_failed",
                    provider=provider_value,
                    model=request.model,
                    status=response.status_code,
                    error="ProviderResponseError",
                    attempt=attempt,
                    latency_ms=round(latency_ms, 1),
                )
                raise ProviderResponseError(
                    "Provider returned an unexpected HTTP status",
                    provider=provider_value,
                    status_code=response.status_code,
                    details={"body": response.text[:500]},
                )

        self._trace.emit(
            "llm_call_failed",
            provider=provider_value,
            model=request.model,
            error="MaxRetriesExceeded",
            attempt=max_attempts,
        )
        return None

    async def _make_request(
        self,
        base_url: str,
        api_key: str,
        request: LLMRequestSchema,
        provider_value: str,
    ) -> httpx.Response:
        if self._client is None:
            raise ProviderError(
                "HTTP client not initialized",
                provider=provider_value,
            )

        timeout_value = request.timeout or self._settings.llm_timeout
        headers = {
            **constants.DEFAULT_HEADERS,
            "Content-Type": "application/json",
        }

        if provider_value == ProviderName.GOOGLE.value:
            model_id = request.model
            if model_id.startswith("models/"):
                model_id = model_id[len("models/"):]
            url = f"{base_url}/models/{model_id}:generateContent"
            headers["x-goog-api-key"] = api_key
            payload = self._build_google_payload(request)
        else:
            url = f"{base_url}/chat/completions"
            headers["Authorization"] = f"Bearer {api_key}"
            payload = self._build_openai_payload(request)

        try:
            return await self._client.post(
                url,
                json=payload,
                headers=headers,
                timeout=httpx.Timeout(
                    timeout_value,
                    connect=self._settings.llm_connect_timeout,
                ),
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                "LLM request timed out",
                provider=provider_value,
                details={"model": request.model},
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                "LLM network request failed",
                provider=provider_value,
                details={"model": request.model, "error": str(exc)},
            ) from exc

    def _build_openai_payload(self, request: LLMRequestSchema) -> dict[str, Any]:
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _build_google_payload(self, request: LLMRequestSchema) -> dict[str, Any]:
        contents = []
        system_text = ""
        for msg in request.messages:
            if msg.role == "system":
                system_text = msg.content
            else:
                contents.append({
                    "role": "user" if msg.role == "user" else "model",
                    "parts": [{"text": msg.content}],
                })

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
            },
        }
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}
        if request.response_format == "json_object":
            payload["generationConfig"]["responseMimeType"] = "application/json"
        return payload

    def _extract_content(self, data: Any, provider_value: str) -> str:
        if not isinstance(data, dict):
            return ""

        if provider_value == ProviderName.GOOGLE.value:
            candidates = data.get("candidates", [])
            if candidates and isinstance(candidates, list):
                parts = candidates[0].get("content", {}).get("parts", [])
                text_parts = [
                    p.get("text", "") for p in parts if isinstance(p, dict)
                ]
                return "".join(text_parts)
            return ""

        choices = data.get("choices", [])
        if choices and isinstance(choices, list):
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    item.get("text", "") for item in content if isinstance(item, dict)
                )
        return ""

    def _extract_tokens(self, data: Any, provider_value: str) -> int:
        if not isinstance(data, dict):
            return 0
        if provider_value == ProviderName.GOOGLE.value:
            usage = data.get("usageMetadata", {})
            if isinstance(usage, dict):
                return int(usage.get("totalTokenCount", 0))
            return 0
        usage = data.get("usage", {})
        if isinstance(usage, dict):
            return int(usage.get("total_tokens", 0))
        return 0

    def _extract_input_tokens(self, data: Any, provider_value: str) -> int:
        if not isinstance(data, dict):
            return 0
        if provider_value == ProviderName.GOOGLE.value:
            usage = data.get("usageMetadata", {})
            if isinstance(usage, dict):
                return int(usage.get("promptTokenCount", 0))
            return 0
        usage = data.get("usage", {})
        if isinstance(usage, dict):
            return int(usage.get("prompt_tokens", 0))
        return 0

    def _extract_output_tokens(self, data: Any, provider_value: str) -> int:
        if not isinstance(data, dict):
            return 0
        if provider_value == ProviderName.GOOGLE.value:
            usage = data.get("usageMetadata", {})
            if isinstance(usage, dict):
                return int(usage.get("candidatesTokenCount", 0))
            return 0
        usage = data.get("usage", {})
        if isinstance(usage, dict):
            return int(usage.get("completion_tokens", 0))
        return 0

    def _estimate_tokens(self, content: str) -> int:
        if not content:
            return 0
        return max(1, len(content) // 4)

    def _report_token_usage(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        latency_ms: float,
    ) -> None:
        for hook in self._token_hooks:
            try:
                hook(
                    provider=provider,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                )
            except Exception as exc:
                self._logger.warning(f"Token hook failed: {exc}")

    def _calculate_backoff(self, attempt: int) -> float:
        base = self._settings.llm_retry_backoff_seconds * (2 ** (attempt - 1))
        base = min(base, 30.0)
        return base + random.uniform(0.0, 0.1 * max(base, 0.1))