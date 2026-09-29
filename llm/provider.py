from __future__ import annotations

import asyncio
import random
import threading
import time
from typing import Any, ClassVar

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
from utils.async_helpers import loop_is_alive, safe_aclose
from utils.cache import default_cache_registry
from utils.hashing import stable_hash
from utils.logger import get_logger, get_trace_logger
from utils.rate_limiter import default_rate_limiter_registry
from pathlib import Path

_LLM_CACHE_NAMESPACE = "llm_response"
_LLM_CACHE_TTL_SECONDS = 86400
_LLM_CACHE_MAX_ENTRIES = 5000

def _provider_health_path() -> "Path":
    from pathlib import Path

    from core.config import get_data_directory

    return get_data_directory() / "provider_health.json"


def _key_fingerprint(api_key: str) -> str:
    return stable_hash({"key": api_key})[:16]


def _load_provider_health() -> dict[str, float]:
    try:
        import json

        path = _provider_health_path()

        if not path.exists():
            return {}

        raw = json.loads(path.read_text(encoding="utf-8"))

        if not isinstance(raw, dict):
            return {}

        now = time.time()
        loaded: dict[str, float] = {}

        for fingerprint, until in raw.items():
            try:
                value = float(until)
            except Exception:
                continue

            if value > now:
                loaded[str(fingerprint)] = value

        return loaded
    except Exception:
        return {}


def _save_provider_health(cooldowns: dict[str, float]) -> None:
    try:
        import json

        now = time.time()
        trimmed = {
            key: until
            for key, until in cooldowns.items()
            if until > now
        }

        path = _provider_health_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(trimmed, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass
    
def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _build_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=None,
        limits=httpx.Limits(
            max_connections=20,
            max_keepalive_connections=5,
        ),
        follow_redirects=True,
    )


class LLMProviderManager:
    _shared_lock: ClassVar[threading.Lock] = threading.Lock()
    _shared_instance: ClassVar["LLMProviderManager | None"] = None
    _shared_token_hooks: ClassVar[list[Any]] = []
    _shared_hooks_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        self._settings = get_settings()
        self._logger = get_logger("llm.provider")
        self._trace = get_trace_logger()
        self._client: httpx.AsyncClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None
        self._last_close_error: str | None = None
        self._rate_limiters: dict[str, Any] = {}
        self._token_hooks: list[Any] = []
        self._is_borrowed: bool = False
        self._key_cooldowns: dict[str, float] = _load_provider_health()
        self._cache = default_cache_registry.get(
            name=_LLM_CACHE_NAMESPACE,
            default_ttl=float(_LLM_CACHE_TTL_SECONDS),
            max_entries=int(_LLM_CACHE_MAX_ENTRIES),
        )

    @property
    def bound_loop(self) -> asyncio.AbstractEventLoop | None:
        return self._client_loop

    @property
    def last_close_error(self) -> str | None:
        return self._last_close_error

    @property
    def is_closed(self) -> bool:
        return self._client is None

    @classmethod
    def get_shared(cls) -> "LLMProviderManager":
        with cls._shared_lock:
            instance = cls._shared_instance
            running_loop = _running_loop()

            if instance is not None:
                client = instance._client
                bound = instance._client_loop

                if client is not None and not client.is_closed:
                    if bound is not None and not loop_is_alive(bound):
                        instance._client = None
                        instance._client_loop = None
                        cls._shared_instance = None
                        instance = None
                    else:
                        return instance
                else:
                    cls._shared_instance = None
                    instance = None

            instance = cls()

            if running_loop is not None:
                instance._client = _build_http_client()
                instance._client_loop = running_loop
            else:
                instance._client_loop = None

            cls._shared_instance = instance
            return instance

    @classmethod
    async def close_shared(cls) -> None:
        with cls._shared_lock:
            instance = cls._shared_instance
            cls._shared_instance = None

        if instance is not None:
            await instance.close()

    @classmethod
    def has_shared(cls) -> bool:
        instance = cls._shared_instance
        return (
            instance is not None
            and instance._client is not None
            and not instance._client.is_closed
        )

    async def __aenter__(self) -> "LLMProviderManager":
        running_loop = _running_loop()
        shared = type(self)._shared_instance

        if shared is not None and shared is not self:
            shared_client = shared._client
            shared_bound = shared._client_loop

            if shared_client is not None and not shared_client.is_closed:
                if shared_bound is running_loop:
                    self._client = shared_client
                    self._client_loop = shared_bound
                    self._is_borrowed = True
                    return self

                if shared_bound is not None and loop_is_alive(shared_bound):
                    self._logger.warning(
                        "Shared LLM manager is bound to a different live "
                        "event loop; creating a private client for this "
                        "context"
                    )

        if running_loop is None:
            self._client = None
            self._client_loop = None
            self._is_borrowed = False
            return self

        self._client = _build_http_client()
        self._client_loop = running_loop
        self._is_borrowed = False
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc: Any,
        tb: Any,
    ) -> bool:
        if type(self)._shared_instance is self:
            return False

        if self._is_borrowed:
            self._client = None
            self._client_loop = None
            self._is_borrowed = False
            return False

        await self.close()
        return False

    async def close(self) -> None:
        client = self._client
        owning_loop = self._client_loop
        self._client = None
        self._client_loop = None

        if client is None:
            self._last_close_error = None
            return

        ok = await safe_aclose(client, loop=owning_loop, logger=self._logger)
        self._last_close_error = None if ok else "close_failed"

    def register_token_hook(self, hook: Any) -> None:
        if hook is None:
            return

        cls = type(self)

        with cls._shared_hooks_lock:
            if hook not in cls._shared_token_hooks:
                cls._shared_token_hooks.append(hook)

    def unregister_token_hook(self, hook: Any) -> None:
        if hook is None:
            return

        cls = type(self)

        with cls._shared_hooks_lock:
            try:
                cls._shared_token_hooks.remove(hook)
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

    async def complete(
        self,
        request: LLMRequestSchema,
    ) -> LLMResponseSchema | None:
        provider_value = request.provider.value
        base_url = constants.PROVIDER_BASE_URLS.get(provider_value)

        if base_url is None:
            raise ProviderError(
                "Unsupported provider",
                provider=provider_value,
                details={"provider": provider_value},
            )

        cached_response = self._cache_get(request)

        if cached_response is not None:
            self._trace.emit(
                "llm_call_cache_hit",
                provider=provider_value,
                model=request.model,
            )
            return cached_response

        api_keys = tuple(
            key
            for key in self._settings.get_provider_keys_or_raise(
                request.provider
            )
            if key
        )

        if not api_keys:
            raise ProviderError(
                "No API keys configured for provider",
                provider=provider_value,
                details={"provider": provider_value},
            )

        rate_limiter = self._get_rate_limiter(provider_value)
        max_attempts = max(1, self._settings.llm_max_retries + 1)
        ordered_keys = self._order_api_keys(api_keys)

        last_rate_limit_error: ProviderRateLimitError | None = None

        for api_key in ordered_keys:
            result, rate_limited = await self._attempt_with_key(
                base_url=base_url,
                api_key=api_key,
                request=request,
                provider_value=provider_value,
                rate_limiter=rate_limiter,
                max_attempts=max_attempts,
            )

            if result is not None:
                self._clear_key_cooldown(api_key)
                self._cache_set(request, result)

                self._trace.emit(
                    "llm_call_cache_store",
                    provider=provider_value,
                    model=request.model,
                )

                return result

            if rate_limited:
                self._mark_key_cooldown(api_key)
                last_rate_limit_error = ProviderRateLimitError(
                    "Provider rate limit exceeded on all configured keys",
                    provider=provider_value,
                    details={"keys_tried": len(ordered_keys)},
                )
                continue

        if last_rate_limit_error is not None:
            raise last_rate_limit_error

        return None

    def _order_api_keys(self, api_keys: tuple[str, ...]) -> tuple[str, ...]:
        now = time.time()
        available: list[str] = []
        cooling: list[str] = []

        for key in api_keys:
            fingerprint = _key_fingerprint(key)
            until = self._key_cooldowns.get(fingerprint, 0.0)

            if until <= now:
                available.append(key)
            else:
                cooling.append(key)

        return tuple(available + cooling)

    def _mark_key_cooldown(self, api_key: str) -> None:
        try:
            cooldown = float(self._settings.llm_key_cooldown_seconds)
        except Exception:
            cooldown = 300.0

        cooldown = max(30.0, min(1800.0, cooldown))
        fingerprint = _key_fingerprint(api_key)
        self._key_cooldowns[fingerprint] = time.time() + cooldown
        _save_provider_health(self._key_cooldowns)

    def _clear_key_cooldown(self, api_key: str) -> None:
        fingerprint = _key_fingerprint(api_key)

        if fingerprint in self._key_cooldowns:
            self._key_cooldowns.pop(fingerprint, None)
            _save_provider_health(self._key_cooldowns)
    def _is_cache_eligible(self, request: LLMRequestSchema) -> bool:
        try:
            if float(request.temperature) > 0.0:
                return False
        except Exception:
            return False

        if not request.messages:
            return False

        return True

    def _cache_key_for(self, request: LLMRequestSchema) -> str:
        provider_value = request.provider.value
        model_id = str(request.model or "")
        temperature = float(request.temperature or 0.0)
        max_tokens = int(request.max_tokens or 0)
        response_format = str(request.response_format or "")

        messages = [
            (str(msg.role), str(msg.content))
            for msg in request.messages
        ]

        return stable_hash(
            {
                "provider": provider_value,
                "model": model_id,
                "temperature": round(temperature, 4),
                "max_tokens": max_tokens,
                "response_format": response_format,
                "messages": messages,
            }
        )

    def _cache_get(
        self, request: LLMRequestSchema
    ) -> LLMResponseSchema | None:
        if not self._is_cache_eligible(request):
            return None

        key = self._cache_key_for(request)

        try:
            cached = self._cache.get(key)
        except Exception:
            return None

        if cached is None:
            return None

        try:
            return LLMResponseSchema.model_validate(cached)
        except Exception:
            return None

    def _cache_set(
        self,
        request: LLMRequestSchema,
        response: LLMResponseSchema,
    ) -> None:
        if not self._is_cache_eligible(request):
            return

        key = self._cache_key_for(request)

        try:
            self._cache.set(
                key,
                response.model_dump(mode="json"),
                ttl_seconds=float(_LLM_CACHE_TTL_SECONDS),
            )
        except Exception:
            pass
    async def _attempt_with_key(
        self,
        base_url: str,
        api_key: str,
        request: LLMRequestSchema,
        provider_value: str,
        rate_limiter: Any,
        max_attempts: int,
    ) -> tuple[LLMResponseSchema | None, bool]:
        suffix = api_key[-4:] if len(api_key) >= 4 else "****"

        self._trace.emit(
            "llm_call_start",
            provider=provider_value,
            model=request.model,
            key_suffix=suffix,
            max_attempts=max_attempts,
        )

        for attempt in range(1, max_attempts + 1):
            await rate_limiter.acquire()
            start = time.perf_counter()

            try:
                response = await self._make_request(
                    base_url=base_url,
                    api_key=api_key,
                    request=request,
                    provider_value=provider_value,
                )
            except (ProviderAuthError, ModelNotFoundError):
                raise
            except ProviderRateLimitError:
                if attempt < max_attempts:
                    await asyncio.sleep(self._calculate_backoff(attempt))
                    continue
                return None, True
            except ProviderTimeoutError:
                if attempt < max_attempts:
                    await asyncio.sleep(self._calculate_backoff(attempt))
                    continue

                self._mark_key_cooldown(api_key)
                self._trace.emit(
                    "llm_key_timeout_cooldown",
                    provider=provider_value,
                    model=request.model,
                )
                raise
            except ProviderResponseError as exc:
                status = exc.details.get("status_code") if exc.details else None

                if status in (402, 429) and attempt < max_attempts:
                    await asyncio.sleep(self._calculate_backoff(attempt))
                    continue

                if status in (402, 429):
                    return None, True

                if status in (500, 502, 503, 504) and attempt < max_attempts:
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

                return (
                    LLMResponseSchema(
                        provider=request.provider,
                        model=request.model,
                        content=content,
                        latency_ms=latency_ms,
                        tokens_used=tokens_used,
                        raw=data,
                    ),
                    False,
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

            if response.status_code in (402, 429):
                if attempt < max_attempts:
                    await asyncio.sleep(self._calculate_backoff(attempt))
                    continue

                return None, True

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

        return None, False

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

        running_loop = _running_loop()

        if (
            self._client_loop is not None
            and running_loop is not None
            and self._client_loop is not running_loop
        ):
            raise ProviderError(
                "HTTP client is bound to a different event loop; "
                "recreate the LLM manager before issuing new requests",
                provider=provider_value,
                details={
                    "bound_loop_id": id(self._client_loop),
                    "running_loop_id": id(running_loop),
                },
            )

        timeout_value = request.timeout or self._settings.llm_timeout

        try:
            timeout_value = float(timeout_value)
        except Exception:
            timeout_value = float(self._settings.llm_timeout)

        timeout_value = max(5.0, min(120.0, timeout_value))

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

        httpx_timeout = httpx.Timeout(
            timeout_value,
            connect=self._settings.llm_connect_timeout,
        )

        try:
            return await asyncio.wait_for(
                self._client.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=httpx_timeout,
                ),
                timeout=timeout_value + 5.0,
            )
        except asyncio.TimeoutError as exc:
            self._trace.emit(
                "llm_call_hard_timeout",
                provider=provider_value,
                model=request.model,
                timeout_seconds=timeout_value,
            )
            raise ProviderTimeoutError(
                "LLM request exceeded hard time ceiling",
                provider=provider_value,
                details={
                    "model": request.model,
                    "timeout": timeout_value,
                },
            ) from exc
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

    def _build_openai_payload(
        self,
        request: LLMRequestSchema,
    ) -> dict[str, Any]:
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

    def _build_google_payload(
        self,
        request: LLMRequestSchema,
    ) -> dict[str, Any]:
        contents = []
        system_text = ""

        for msg in request.messages:
            if msg.role == "system":
                system_text = msg.content
            else:
                contents.append(
                    {
                        "role": "user" if msg.role == "user" else "model",
                        "parts": [{"text": msg.content}],
                    }
                )

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
            },
        }

        if system_text:
            payload["systemInstruction"] = {
                "parts": [{"text": system_text}]
            }

        if request.response_format == "json_object":
            payload["generationConfig"]["responseMimeType"] = (
                "application/json"
            )

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
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict)
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
        cls = type(self)

        with cls._shared_hooks_lock:
            hooks = list(cls._shared_token_hooks)

        for hook in hooks:
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