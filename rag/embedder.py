from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

import httpx

from core.config import get_settings
from core.models import ProviderName, RankedSource, Source
from utils.async_helpers import loop_is_alive, safe_aclose
from utils.logger import get_logger, get_trace_logger
from utils.text import clean_text, truncate_text

_DEFAULT_BATCH_SIZE = 32
_DEFAULT_MAX_CHARS = 6000
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_SECONDS = 1.5
_MISTRAL_EMBED_ENDPOINT = "https://api.mistral.ai/v1/embeddings"

_DEFAULT_CHUNK_TOKENS = 220
_MIN_CHUNK_TOKENS = 120
_MAX_CHUNK_TOKENS = 600
_DEFAULT_CHUNK_INDEX_MAX_PER_SOURCE = 30

_CHUNKED_INDEX_CONFIG_CACHE: dict[str, Any] | None = None

_EMBED_KIND_PARENT = "parent"
_EMBED_KIND_CHUNKED = "chunked"
_EMBED_KIND_FALLBACK = "fallback"


def _load_chunked_index_config() -> dict[str, Any]:
    global _CHUNKED_INDEX_CONFIG_CACHE

    if _CHUNKED_INDEX_CONFIG_CACHE is not None:
        return dict(_CHUNKED_INDEX_CONFIG_CACHE)

    config: dict[str, Any] = {
        "enabled": True,
        "chunk_tokens": _DEFAULT_CHUNK_TOKENS,
        "max_chunks_per_source": _DEFAULT_CHUNK_INDEX_MAX_PER_SOURCE,
        "fallback_to_parent": True,
        "fail_open": True,
    }

    try:
        import yaml

        config_path = (
            Path(__file__).resolve().parent.parent
            / "configs"
            / "settings.yaml"
        )

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}

            if isinstance(raw, dict):
                section = raw.get("rag", {})

                if isinstance(section, dict):
                    value = section.get("chunked_indexing_enabled")

                    if value is not None:
                        config["enabled"] = bool(value)

                    value = section.get("chunk_index_tokens")

                    if value is not None:
                        try:
                            config["chunk_tokens"] = int(value)
                        except (TypeError, ValueError):
                            pass

                    value = section.get("chunk_index_max_chunks_per_source")

                    if value is not None:
                        try:
                            config["max_chunks_per_source"] = int(value)
                        except (TypeError, ValueError):
                            pass

                    value = section.get("chunk_index_fallback_to_parent")

                    if value is not None:
                        config["fallback_to_parent"] = bool(value)

                    value = section.get("chunk_index_fail_open")

                    if value is not None:
                        config["fail_open"] = bool(value)
    except Exception:
        pass

    try:
        config["chunk_tokens"] = max(
            _MIN_CHUNK_TOKENS,
            min(_MAX_CHUNK_TOKENS, int(config["chunk_tokens"])),
        )
    except Exception:
        config["chunk_tokens"] = _DEFAULT_CHUNK_TOKENS

    try:
        config["max_chunks_per_source"] = max(
            0,
            int(config["max_chunks_per_source"]),
        )
    except Exception:
        config["max_chunks_per_source"] = _DEFAULT_CHUNK_INDEX_MAX_PER_SOURCE

    _CHUNKED_INDEX_CONFIG_CACHE = config
    return dict(config)


class Embedder:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_chars: int = _DEFAULT_MAX_CHARS,
        timeout: float = 60.0,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        self._settings = get_settings()
        self._model = model or self._settings.embedding_model
        self._api_key = api_key
        self._batch_size = max(
            1, min(128, int(batch_size or _DEFAULT_BATCH_SIZE))
        )
        self._max_chars = max(200, int(max_chars or _DEFAULT_MAX_CHARS))
        self._timeout = max(5.0, float(timeout or 60.0))
        self._max_retries = max(1, int(max_retries or _DEFAULT_MAX_RETRIES))
        self._client: httpx.AsyncClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None
        self._logger = get_logger("rag.embedder")
        self._trace = get_trace_logger()
        self._last_close_error: str | None = None
        self._last_embed_kind: str = _EMBED_KIND_PARENT
        self._chunked_config = _load_chunked_index_config()

    @property
    def last_close_error(self) -> str | None:
        return self._last_close_error

    @property
    def is_closed(self) -> bool:
        return self._client is None

    @property
    def bound_loop(self) -> asyncio.AbstractEventLoop | None:
        return self._client_loop

    @property
    def last_embed_kind(self) -> str:
        return self._last_embed_kind

    @property
    def chunked_index_enabled(self) -> bool:
        return bool(self._chunked_config.get("enabled", True))

    @property
    def chunked_config(self) -> dict[str, Any]:
        return dict(self._chunked_config)

    async def __aenter__(self) -> "Embedder":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        await self.aclose()
        return False

    async def aclose(self) -> None:
        client = self._client
        owning_loop = self._client_loop
        self._client = None
        self._client_loop = None

        if client is None:
            self._last_close_error = None
            return

        ok = await safe_aclose(client, loop=owning_loop, logger=self._logger)
        self._last_close_error = None if ok else "close_failed"

    async def close(self) -> None:
        await self.aclose()

    async def embed_text(self, text: str) -> list[float]:
        cleaned = self._prepare_text(text)

        if not cleaned:
            return []

        vectors = await self._embed_batch([cleaned])
        return vectors[0] if vectors else []

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        prepared = [self._prepare_text(text) for text in texts]
        pairs: list[tuple[int, str]] = [
            (index, value) for index, value in enumerate(prepared) if value
        ]
        results: list[list[float]] = [[] for _ in range(len(prepared))]

        if not pairs:
            return results

        batches = [
            pairs[index : index + self._batch_size]
            for index in range(0, len(pairs), self._batch_size)
        ]

        for batch in batches:
            batch_texts = [text for _, text in batch]
            vectors = await self._embed_batch(batch_texts)

            for position, (original_index, _) in enumerate(batch):
                if position < len(vectors):
                    results[original_index] = vectors[position]

        return results

    async def embed_ranked_sources(
        self,
        ranked_sources: list[RankedSource],
    ) -> list[list[float]]:
        if not ranked_sources:
            self._last_embed_kind = _EMBED_KIND_PARENT
            return []

        texts: list[str] = []

        for ranked in ranked_sources:
            source = getattr(ranked, "source", None)
            texts.append(self._source_text(source) if source is not None else "")

        result = await self.embed_texts(texts)
        self._last_embed_kind = _EMBED_KIND_PARENT
        return result

    async def embed_sources(
        self,
        sources: list[Source],
    ) -> list[list[float]]:
        if not sources:
            self._last_embed_kind = _EMBED_KIND_PARENT
            return []

        texts = [self._source_text(source) for source in sources]
        result = await self.embed_texts(texts)
        self._last_embed_kind = _EMBED_KIND_PARENT
        return result

    async def embed_ranked_sources_with_chunks(
        self,
        ranked_sources: list[RankedSource],
        chunk_tokens: int | None = None,
        text_builder: Callable[..., str] | None = None,
    ) -> tuple[list[list[float]], list[list[list[float]]], int]:
        resolved_tokens = self._resolve_chunk_tokens(chunk_tokens)

        if not ranked_sources:
            self._last_embed_kind = _EMBED_KIND_PARENT
            return [], [], resolved_tokens

        sources: list[Any] = []
        ranks: list[Any] = []
        scores: list[Any] = []
        confidences: list[Any] = []

        for ranked in ranked_sources:
            sources.append(getattr(ranked, "source", None))
            ranks.append(getattr(ranked, "rank", None))
            scores.append(getattr(ranked, "score", None))
            confidences.append(getattr(ranked, "confidence", None))

        return await self._embed_sources_with_chunks_impl(
            sources=sources,
            ranks=ranks,
            scores=scores,
            confidences=confidences,
            resolved_tokens=resolved_tokens,
            text_builder=text_builder,
        )

    async def embed_sources_with_chunks(
        self,
        sources: list[Source],
        chunk_tokens: int | None = None,
        text_builder: Callable[..., str] | None = None,
    ) -> tuple[list[list[float]], list[list[list[float]]], int]:
        resolved_tokens = self._resolve_chunk_tokens(chunk_tokens)

        if not sources:
            self._last_embed_kind = _EMBED_KIND_PARENT
            return [], [], resolved_tokens

        count = len(sources)

        return await self._embed_sources_with_chunks_impl(
            sources=list(sources),
            ranks=[None] * count,
            scores=[None] * count,
            confidences=[None] * count,
            resolved_tokens=resolved_tokens,
            text_builder=text_builder,
        )

    async def _embed_sources_with_chunks_impl(
        self,
        sources: list[Any],
        ranks: list[Any],
        scores: list[Any],
        confidences: list[Any],
        resolved_tokens: int,
        text_builder: Callable[..., str] | None,
    ) -> tuple[list[list[float]], list[list[list[float]]], int]:
        max_chunks = int(
            self._chunked_config.get("max_chunks_per_source", 0) or 0
        )

        parent_texts: list[str] = []
        chunk_texts_per_source: list[list[str]] = []

        for index, source in enumerate(sources):
            rank = ranks[index] if index < len(ranks) else None
            score = scores[index] if index < len(scores) else None
            confidence = confidences[index] if index < len(confidences) else None

            text = self._invoke_text_builder(
                text_builder,
                source,
                rank=rank,
                score=score,
                confidence=confidence,
            )

            if not text:
                parent_texts.append("")
                chunk_texts_per_source.append([])
                continue

            parent_texts.append(text)

            if not self._chunked_config.get("enabled", True):
                chunk_texts_per_source.append([])
                continue

            chunks = self._split_text_into_chunks(text, resolved_tokens)

            if max_chunks > 0 and len(chunks) > max_chunks:
                chunks = chunks[:max_chunks]

            chunk_texts_per_source.append(chunks)

        prepared_parents = [
            self._prepare_text(text) for text in parent_texts
        ]

        parent_embeddings = await self._safe_embed_batch(prepared_parents)

        if self._is_full_embed_failure(prepared_parents, parent_embeddings):
            self._last_embed_kind = _EMBED_KIND_FALLBACK

            self._trace.emit(
                "embedder_chunked_full_failure",
                input_sources=len(sources),
                resolved_tokens=resolved_tokens,
            )

            return [], [], resolved_tokens

        total_chunks = sum(len(chunks) for chunks in chunk_texts_per_source)

        if total_chunks <= 0:
            self._last_embed_kind = _EMBED_KIND_PARENT

            self._trace.emit(
                "embedder_chunked_no_chunks",
                input_sources=len(sources),
                resolved_tokens=resolved_tokens,
            )

            return (
                parent_embeddings,
                [[] for _ in chunk_texts_per_source],
                resolved_tokens,
            )

        flat_chunks: list[str] = []
        flat_map: list[tuple[int, int]] = []

        for source_index, chunks in enumerate(chunk_texts_per_source):
            for chunk_index, chunk in enumerate(chunks):
                if not chunk:
                    continue

                flat_chunks.append(chunk)
                flat_map.append((source_index, chunk_index))

        flat_embeddings = await self._safe_embed_batch(flat_chunks)

        grouped: list[list[list[float]]] = [
            [[] for _ in chunks] for chunks in chunk_texts_per_source
        ]

        for (source_index, chunk_index), embedding in zip(
            flat_map, flat_embeddings
        ):
            if not embedding:
                continue

            if source_index < 0 or source_index >= len(grouped):
                continue

            source_slot = grouped[source_index]

            if chunk_index < 0 or chunk_index >= len(source_slot):
                continue

            source_slot[chunk_index] = embedding

        successful_chunks = sum(
            1
            for per_source in grouped
            for embedding in per_source
            if embedding
        )

        if successful_chunks <= 0:
            self._last_embed_kind = _EMBED_KIND_PARENT

            self._trace.emit(
                "embedder_chunked_all_chunks_failed",
                input_sources=len(sources),
                total_chunks=total_chunks,
                resolved_tokens=resolved_tokens,
            )

            return (
                parent_embeddings,
                [[] for _ in chunk_texts_per_source],
                resolved_tokens,
            )

        self._last_embed_kind = _EMBED_KIND_CHUNKED

        self._trace.emit(
            "embedder_chunked_completed",
            input_sources=len(sources),
            total_chunks=total_chunks,
            successful_chunks=successful_chunks,
            resolved_tokens=resolved_tokens,
            max_chunks_per_source=max_chunks,
        )

        return parent_embeddings, grouped, resolved_tokens

    def _resolve_chunk_tokens(self, chunk_tokens: int | None) -> int:
        if chunk_tokens is not None:
            try:
                parsed = int(chunk_tokens)
            except (TypeError, ValueError):
                parsed = _DEFAULT_CHUNK_TOKENS

            return max(_MIN_CHUNK_TOKENS, min(_MAX_CHUNK_TOKENS, parsed))

        try:
            configured = int(
                self._chunked_config.get("chunk_tokens", _DEFAULT_CHUNK_TOKENS)
            )
        except Exception:
            configured = _DEFAULT_CHUNK_TOKENS

        return max(_MIN_CHUNK_TOKENS, min(_MAX_CHUNK_TOKENS, configured))

    def _split_text_into_chunks(
        self,
        text: str,
        chunk_tokens: int,
    ) -> list[str]:
        if not text:
            return []

        try:
            from rag.vector_store import VectorStore
        except Exception as exc:
            self._logger.warning(
                f"Failed to import VectorStore for chunking: {exc}"
            )
            return []

        try:
            return VectorStore.split_into_chunks(text, chunk_tokens)
        except Exception as exc:
            self._logger.warning(f"Chunk splitting failed: {exc}")
            return []

    def _invoke_text_builder(
        self,
        text_builder: Callable[..., str] | None,
        source: Any,
        rank: Any = None,
        score: Any = None,
        confidence: Any = None,
    ) -> str:
        if source is None:
            return ""

        if text_builder is None:
            return self._source_text(source)

        try:
            result = text_builder(
                source,
                rank=rank,
                score=score,
                confidence=confidence,
            )
            return clean_text(result)
        except TypeError:
            pass
        except Exception as exc:
            self._logger.warning(f"text_builder failed: {exc}")

        try:
            result = text_builder(source)
            return clean_text(result)
        except Exception as exc:
            self._logger.warning(f"text_builder (single-arg) failed: {exc}")

        return self._source_text(source)

    async def _safe_embed_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        try:
            result = await self._embed_batch(texts)
        except Exception as exc:
            self._logger.warning(f"Embedding batch raised: {exc}")
            return [[] for _ in texts]

        if not isinstance(result, list):
            return [[] for _ in texts]

        normalized: list[list[float]] = []

        for index in range(len(texts)):
            if index < len(result) and isinstance(result[index], list):
                normalized.append(result[index])
            else:
                normalized.append([])

        return normalized

    @staticmethod
    def _is_full_embed_failure(
        prepared_texts: list[str],
        embeddings: list[list[float]],
    ) -> bool:
        if not prepared_texts:
            return False

        has_any_text = any(text for text in prepared_texts)

        if not has_any_text:
            return False

        return all(not embedding for embedding in embeddings)

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        client = await self._ensure_client()

        if client is None:
            return [[] for _ in texts]

        api_key = self._resolve_api_key()

        if not api_key:
            self._logger.warning("Embedding API key unavailable")
            return [[] for _ in texts]

        payload = {"model": self._model, "input": texts}
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        for attempt in range(1, self._max_retries + 1):
            try:
                response = await client.post(
                    _MISTRAL_EMBED_ENDPOINT,
                    json=payload,
                    headers=headers,
                    timeout=httpx.Timeout(self._timeout),
                )
            except httpx.HTTPError as exc:
                self._logger.warning(
                    f"Embedding network error (attempt {attempt}): {exc}"
                )

                if attempt < self._max_retries:
                    await asyncio.sleep(_DEFAULT_BACKOFF_SECONDS * attempt)
                    continue

                return [[] for _ in texts]

            if response.status_code == 200:
                try:
                    data = response.json()
                except Exception as exc:
                    self._logger.warning(f"Embedding JSON parse failed: {exc}")
                    return [[] for _ in texts]

                return self._extract_vectors(data, expected=len(texts))

            if response.status_code in (429, 500, 502, 503, 504):
                if attempt < self._max_retries:
                    await asyncio.sleep(_DEFAULT_BACKOFF_SECONDS * attempt)
                    continue

            if response.status_code in (401, 403):
                self._logger.warning(
                    "Embedding authentication failed — check MISTRAL key"
                )
                return [[] for _ in texts]

            self._logger.warning(
                f"Embedding HTTP {response.status_code}: {response.text[:200]}"
            )
            return [[] for _ in texts]

        return [[] for _ in texts]

    async def _ensure_client(self) -> httpx.AsyncClient | None:
        running_loop = self._running_loop()
        client = self._client

        if client is not None and not client.is_closed:
            if self._client_loop is running_loop:
                return client

            old_client = client
            old_loop = self._client_loop
            self._client = None
            self._client_loop = None

            if old_loop is None or not loop_is_alive(old_loop):
                await safe_aclose(old_client, loop=old_loop, logger=self._logger)
            else:
                self._logger.warning(
                    "Embedder client was bound to a different running loop; "
                    "orphaning previous client and rebinding"
                )

                try:
                    asyncio.run_coroutine_threadsafe(
                        old_client.aclose(),
                        old_loop,
                    )
                except Exception:
                    pass
        elif client is not None and client.is_closed:
            self._client = None
            self._client_loop = None

        if running_loop is None:
            self._logger.warning(
                "Embedder invoked outside a running event loop; cannot bind client"
            )
            return None

        try:
            new_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                limits=httpx.Limits(
                    max_connections=10,
                    max_keepalive_connections=5,
                ),
                follow_redirects=True,
            )
        except Exception as exc:
            self._logger.warning(f"Failed to create embedding HTTP client: {exc}")
            return None

        self._client = new_client
        self._client_loop = running_loop
        return new_client

    def _extract_vectors(
        self,
        data: Any,
        expected: int,
    ) -> list[list[float]]:
        result: list[list[float]] = [[] for _ in range(expected)]

        if not isinstance(data, dict):
            return result

        items = data.get("data")

        if not isinstance(items, list):
            return result

        for item in items:
            if not isinstance(item, dict):
                continue

            index = item.get("index")
            embedding = item.get("embedding")

            if not isinstance(embedding, list):
                continue

            try:
                slot = int(index) if index is not None else None
            except (TypeError, ValueError):
                slot = None

            vector: list[float] = []

            for value in embedding:
                try:
                    vector.append(float(value))
                except (TypeError, ValueError):
                    continue

            if slot is None:
                for position in range(expected):
                    if not result[position]:
                        slot = position
                        break

                if slot is None:
                    continue

            if 0 <= slot < expected:
                result[slot] = vector

        return result

    def _resolve_api_key(self) -> str:
        if self._api_key:
            return self._api_key.strip()

        try:
            key = self._settings.get_provider_key(ProviderName.MISTRAL)
        except Exception:
            return ""

        if key is None:
            return ""

        if hasattr(key, "get_secret_value"):
            return str(key.get_secret_value() or "").strip()

        return str(key or "").strip()

    def _prepare_text(self, value: Any) -> str:
        text = clean_text(value)

        if not text:
            return ""

        return truncate_text(text, max_length=self._max_chars, suffix="")

    def _source_text(self, source: Any) -> str:
        if source is None:
            return ""

        parts = [
            clean_text(getattr(source, "title", "")),
            clean_text(getattr(source, "abstract", "")),
            clean_text(getattr(source, "summary", "")),
            clean_text(getattr(source, "url", "")),
            self._enum_value(getattr(source, "platform", "")),
            self._enum_value(getattr(source, "source_type", "")),
            self._enum_value(getattr(source, "difficulty", "")),
        ]

        year = getattr(source, "year", None)

        if year is not None:
            parts.append(str(year))

        citations = getattr(source, "citation_count", None)

        if citations is not None:
            parts.append(f"citations {citations}")

        metadata = getattr(source, "metadata", {}) or {}

        if isinstance(metadata, dict):
            description = metadata.get("description")

            if isinstance(description, str) and description:
                parts.append(clean_text(description))

        joined = " ".join(part for part in parts if part)

        if not joined:
            return ""

        return truncate_text(joined, max_length=self._max_chars, suffix="")

    @staticmethod
    def _running_loop() -> asyncio.AbstractEventLoop | None:
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value) or "")