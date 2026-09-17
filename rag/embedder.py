from __future__ import annotations

import asyncio
import random
import time
from typing import Any

import httpx

from core import constants
from core.config import get_settings
from core.exceptions import ResearchAgentError
from core.models import ProviderName, RankedSource, Source
from utils.logger import get_logger
from utils.rate_limiter import default_rate_limiter_registry
from utils.text import clean_text, truncate_text


class EmbeddingError(ResearchAgentError):
    pass


class Embedder:
    def __init__(
        self,
        model_reference: Any = None,
        client: httpx.AsyncClient | None = None,
        batch_size: int = 8,
        max_text_length: int = 2000,
    ) -> None:
        self._external_model_reference = model_reference
        self._model_reference = model_reference
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=None,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=3),
            follow_redirects=True,
        )
        self._batch_size = max(1, batch_size)
        self._max_text_length = max(100, max_text_length)
        self._settings = get_settings()
        self._logger = get_logger("rag.embedder")
        self._rate_limiters: dict[str, Any] = {}

    async def __aenter__(self) -> Embedder:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        await self.close()
        return False

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def embed_text(self, text: str) -> list[float]:
        results = await self.embed_texts([text])
        if not results:
            return []
        return results[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        resolved_reference = self._resolve_model_reference()
        cleaned_texts = [self._prepare_text(text) for text in texts]
        cleaned_texts = [text for text in cleaned_texts if text]

        if not cleaned_texts:
            return []

        provider = self._enum_value(resolved_reference.provider)

        if provider == ProviderName.GOOGLE.value:
            return await self._embed_google(resolved_reference, cleaned_texts)

        return await self._embed_openai_compatible(resolved_reference, cleaned_texts)

    async def embed_source(self, source: Source) -> list[float]:
        text = self._source_text(source)
        return await self.embed_text(text)

    async def embed_ranked_sources(
        self,
        ranked_sources: list[Any],
    ) -> dict[str, list[float]]:
        valid_ranked = [r for r in ranked_sources if isinstance(r, RankedSource)]
        if not valid_ranked:
            return {}

        texts = [self._source_text(r.source) for r in valid_ranked]

        try:
            embeddings = await self.embed_texts(texts)
        except EmbeddingError as exc:
            details = getattr(exc, "details", {}) or {}
            self._logger.warning(
                f"Batch embedding failed: {exc.message} "
                f"[status={details.get('status_code')}, body={details.get('body', '')[:200]}]"
            )
            return {}
        except Exception as exc:
            self._logger.warning(f"Unexpected embedding error: {exc}")
            return {}

        result: dict[str, list[float]] = {}
        for ranked, embedding in zip(valid_ranked, embeddings):
            if embedding and len(embedding) > 0:
                result[ranked.source.source_id] = embedding

        embedded_count = len(result)
        total_count = len(valid_ranked)
        if embedded_count < total_count:
            self._logger.warning(
                f"Only embedded {embedded_count}/{total_count} sources "
                f"(some texts may have been empty after cleaning)"
            )
        else:
            self._logger.info(f"Successfully embedded {embedded_count} sources")

        return result

    def _resolve_model_reference(self) -> Any:
        if self._model_reference is not None:
            return self._model_reference
        from llm.router import get_embedding_model_reference
        self._model_reference = get_embedding_model_reference()
        return self._model_reference

    def _prepare_text(self, text: str) -> str:
        cleaned = clean_text(text)
        return truncate_text(cleaned, max_length=self._max_text_length, suffix="")

    def _source_text(self, source: Source) -> str:
        parts = [source.title]
        if source.abstract:
            parts.append(source.abstract)
        metadata = source.metadata if isinstance(source.metadata, dict) else {}
        description = metadata.get("description")
        if isinstance(description, str) and description:
            parts.append(description)
        return " ".join(parts)

    async def _embed_openai_compatible(
        self,
        model_reference: Any,
        texts: list[str],
    ) -> list[list[float]]:
        provider_value = self._enum_value(model_reference.provider)
        base_url = constants.PROVIDER_BASE_URLS[provider_value]
        url = f"{base_url}/embeddings"
        api_key = self._settings.get_provider_key_or_raise(model_reference.provider)
        headers = {
            **constants.DEFAULT_HEADERS,
            "Authorization": f"Bearer {api_key.get_secret_value()}",
        }

        all_embeddings: list[list[float]] = []

        for start in range(0, len(texts), self._batch_size):
            batch = texts[start: start + self._batch_size]
            payload = {
                "model": model_reference.model_id,
                "input": batch,
            }
            data = await self._post_with_retry(url, headers, payload, model_reference)
            items = data.get("data", []) if isinstance(data, dict) else []
            sorted_items = sorted(
                items,
                key=lambda item: int(item.get("index", 0)) if isinstance(item, dict) else 0,
            )
            for item in sorted_items:
                embedding = item.get("embedding", []) if isinstance(item, dict) else []
                if isinstance(embedding, list):
                    all_embeddings.append([float(value) for value in embedding])

        return all_embeddings

    async def _embed_google(
        self,
        model_reference: Any,
        texts: list[str],
    ) -> list[list[float]]:
        model_id = model_reference.model_id
        if model_id.startswith("models/"):
            model_id = model_id[len("models/"):]
        base_url = constants.PROVIDER_BASE_URLS[ProviderName.GOOGLE.value]
        url = f"{base_url}/models/{model_id}:embedContent"
        api_key = self._settings.get_provider_key_or_raise(model_reference.provider)
        headers = {
            **constants.DEFAULT_HEADERS,
            "x-goog-api-key": api_key.get_secret_value(),
        }

        all_embeddings: list[list[float]] = []

        for text in texts:
            payload = {
                "model": model_reference.model_id,
                "content": {
                    "parts": [{"text": text}],
                },
            }
            data = await self._post_with_retry(url, headers, payload, model_reference)
            embedding = data.get("embedding", {}) if isinstance(data, dict) else {}
            values = embedding.get("values", []) if isinstance(embedding, dict) else []
            if isinstance(values, list):
                all_embeddings.append([float(value) for value in values])

        return all_embeddings

    async def _post_with_retry(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        model_reference: Any,
    ) -> dict[str, Any]:
        rate_limiter = self._get_rate_limiter(model_reference)
        max_attempts = max(1, self._settings.llm_max_retries)

        for attempt in range(1, max_attempts + 1):
            await rate_limiter.acquire()
            try:
                response = await self._client.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=httpx.Timeout(
                        self._settings.llm_timeout,
                        connect=self._settings.llm_connect_timeout,
                    ),
                )
            except httpx.TimeoutException as exc:
                if attempt == max_attempts:
                    raise EmbeddingError(
                        "Embedding request timed out",
                        details={"url": url},
                    ) from exc
                await asyncio.sleep(self._calculate_backoff(attempt))
                continue
            except httpx.HTTPError as exc:
                if attempt == max_attempts:
                    raise EmbeddingError(
                        "Embedding network request failed",
                        details={"url": url, "error": exc.__class__.__name__},
                    ) from exc
                await asyncio.sleep(self._calculate_backoff(attempt))
                continue

            if response.status_code == 200:
                try:
                    return response.json()
                except Exception as exc:
                    raise EmbeddingError(
                        "Failed to parse embedding response",
                        details={"url": url},
                    ) from exc

            if response.status_code == 429 and attempt < max_attempts:
                await asyncio.sleep(self._calculate_backoff(attempt))
                continue

            if response.status_code >= 500 and attempt < max_attempts:
                await asyncio.sleep(self._calculate_backoff(attempt))
                continue

            raise EmbeddingError(
                "Embedding request failed",
                details={
                    "url": url,
                    "status_code": response.status_code,
                    "body": response.text[:300],
                },
            )

        raise EmbeddingError(
            "Embedding request failed after retries",
            details={"url": url},
        )

    def _get_rate_limiter(self, model_reference: Any) -> Any:
        provider_value = self._enum_value(model_reference.provider)
        rate_limiter = self._rate_limiters.get(provider_value)
        if rate_limiter is not None:
            return rate_limiter
        rate_limiter = default_rate_limiter_registry.get(
            name=provider_value,
            rate_limit=self._settings.llm_rate_limit_per_minute,
            period_seconds=60.0,
        )
        self._rate_limiters[provider_value] = rate_limiter
        return rate_limiter

    def _calculate_backoff(self, attempt: int) -> float:
        base = self._settings.llm_retry_backoff_seconds * (2 ** (attempt - 1))
        base = min(base, 30.0)
        return base + random.uniform(0.0, 0.1 * max(base, 0.1))

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value))