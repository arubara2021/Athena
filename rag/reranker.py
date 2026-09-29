from __future__ import annotations

import hashlib
import math
import threading
from typing import Any

from rag.embedder import Embedder
from rag.vector_store import VectorSearchResult
from utils.logger import get_logger, get_trace_logger
from utils.text import clean_text

_DEFAULT_BLEND = 0.55
_MIN_BLEND = 0.35
_MAX_BLEND = 0.80
_STORED_COVERAGE_FLOOR = 0.80
_QUERY_CACHE_MAX = 128
_TOP_TRACE_COUNT = 3
_TIGHT_SPREAD_THRESHOLD = 0.02
_WIDE_SPREAD_THRESHOLD = 0.15
_TIGHT_BLEND_BOOST = 0.15
_WIDE_BLEND_REDUCTION = 0.10
_FINAL_BLEND_MIN = 0.30
_FINAL_BLEND_MAX = 0.80
_PRIMARY_WEIGHT_DEFAULT = 0.75
_PRIMARY_WEIGHT_MIN = 0.55
_PRIMARY_WEIGHT_MAX = 0.90


class Reranker:
    def __init__(
        self,
        embedder: Embedder | None = None,
        blend_weight: float = _DEFAULT_BLEND,
        store: Any | None = None,
        primary_weight: float = _PRIMARY_WEIGHT_DEFAULT,
    ) -> None:
        self._owns_embedder = embedder is None
        self._embedder = embedder or Embedder()
        self._blend_weight = max(
            _MIN_BLEND,
            min(_MAX_BLEND, float(blend_weight)),
        )
        self._primary_weight = max(
            _PRIMARY_WEIGHT_MIN,
            min(_PRIMARY_WEIGHT_MAX, float(primary_weight)),
        )
        self._store = store
        self._logger = get_logger("rag.reranker")
        self._trace = get_trace_logger()
        self._lock = threading.RLock()
        self._query_cache: dict[str, list[float]] = {}

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def owns_embedder(self) -> bool:
        return self._owns_embedder

    def set_store(self, store: Any | None) -> None:
        self._store = store

    async def aclose(self) -> None:
        if not self._owns_embedder:
            return

        try:
            await self._embedder.aclose()
        except Exception as exc:
            self._logger.warning(f"Reranker embedder close failed: {exc}")

    async def close(self) -> None:
        await self.aclose()

    async def rerank(
        self,
        candidates: list[VectorSearchResult],
        queries: list[str],
        top_k: int,
        store: Any | None = None,
        query_embeddings: list[list[float]] | None = None,
        primary_query: str | None = None,
        primary_query_embedding: list[float] | None = None,
    ) -> list[VectorSearchResult]:
        if not candidates:
            self._emit_trace(
                status="empty_candidates",
                input_count=0,
                output_count=0,
                top_before=[],
                top_after=[],
            )
            return []

        limit = max(1, int(top_k))
        resolved_store = store if store is not None else self._store

        (
            primary_text,
            primary_vec,
            secondary_texts,
            secondary_vecs,
        ) = await self._resolve_query_vectors(
            queries=queries,
            query_embeddings=query_embeddings,
            primary_query=primary_query,
            primary_query_embedding=primary_query_embedding,
        )

        if not primary_vec and not secondary_vecs:
            ordered = self._sort_by_original_score(candidates, limit)
            self._emit_trace(
                status="no_query_embeddings",
                input_count=len(candidates),
                output_count=len(ordered),
                top_before=self._trace_top(candidates),
                top_after=self._trace_top(ordered),
                primary_query=primary_text,
                secondary_count=len(secondary_texts),
            )
            return ordered

        candidate_vecs = await self._resolve_candidate_embeddings(
            candidates,
            resolved_store,
        )

        if not candidate_vecs or len(candidate_vecs) != len(candidates):
            ordered = self._sort_by_original_score(candidates, limit)
            self._emit_trace(
                status="no_candidate_embeddings",
                input_count=len(candidates),
                output_count=len(ordered),
                top_before=self._trace_top(candidates),
                top_after=self._trace_top(ordered),
                primary_query=primary_text,
                secondary_count=len(secondary_texts),
            )
            return ordered

        blend = self._adaptive_blend(candidates)

        scored = self._score_candidates(
            candidates=candidates,
            candidate_vecs=candidate_vecs,
            primary_vec=primary_vec,
            secondary_vecs=secondary_vecs,
            blend=blend,
        )

        scored.sort(key=lambda item: (-item[0], item[1].document.doc_id))
        final = [candidate for _, candidate in scored[:limit]]

        self._emit_trace(
            status="ok",
            input_count=len(candidates),
            output_count=len(final),
            top_before=self._trace_top(candidates),
            top_after=self._trace_top(final),
            primary_query=primary_text,
            secondary_count=len(secondary_texts),
            blend=round(blend, 4),
            primary_weight=round(self._primary_weight, 4),
        )

        return final

    def _score_candidates(
        self,
        candidates: list[VectorSearchResult],
        candidate_vecs: list[list[float]],
        primary_vec: list[float],
        secondary_vecs: list[list[float]],
        blend: float,
    ) -> list[tuple[float, VectorSearchResult]]:
        scored: list[tuple[float, VectorSearchResult]] = []

        for index, candidate in enumerate(candidates):
            vector = (
                candidate_vecs[index] if index < len(candidate_vecs) else []
            )

            if not vector:
                scored.append((float(candidate.score), candidate))
                continue

            similarity = self._combined_similarity(
                vector=vector,
                primary_vec=primary_vec,
                secondary_vecs=secondary_vecs,
            )

            blended = (
                blend * similarity
                + (1.0 - blend) * float(candidate.score)
            )
            blended = max(0.0, min(1.0, blended))
            scored.append((blended, candidate))

        return scored

    def _combined_similarity(
        self,
        vector: list[float],
        primary_vec: list[float],
        secondary_vecs: list[list[float]],
    ) -> float:
        primary_similarity = 0.0

        if primary_vec:
            primary_similarity = self._cosine(primary_vec, vector)

        if not secondary_vecs:
            return max(0.0, min(1.0, primary_similarity))

        secondary_scores: list[float] = []

        for secondary_vec in secondary_vecs:
            if not secondary_vec:
                continue

            secondary_scores.append(self._cosine(secondary_vec, vector))

        if not secondary_scores:
            return max(0.0, min(1.0, primary_similarity))

        secondary_mean = sum(secondary_scores) / len(secondary_scores)

        combined = (
            self._primary_weight * primary_similarity
            + (1.0 - self._primary_weight) * secondary_mean
        )

        return max(0.0, min(1.0, combined))

    async def _resolve_query_vectors(
        self,
        queries: list[str],
        query_embeddings: list[list[float]] | None,
        primary_query: str | None,
        primary_query_embedding: list[float] | None,
    ) -> tuple[str, list[float], list[str], list[list[float]]]:
        clean_queries: list[str] = []

        for query in queries or []:
            cleaned = clean_text(query)

            if cleaned and cleaned not in clean_queries:
                clean_queries.append(cleaned)

        primary_text = clean_text(primary_query) if primary_query else ""

        if not primary_text and clean_queries:
            primary_text = clean_queries[0]

        secondary_texts = [
            query for query in clean_queries if query != primary_text
        ]

        primary_vec: list[float] = []

        if primary_query_embedding:
            primary_vec = self._clean_vector(primary_query_embedding)

        secondary_vecs = self._align_secondary_embeddings(
            secondary_texts=secondary_texts,
            all_queries=clean_queries,
            primary_text=primary_text,
            provided=query_embeddings or [],
        )

        if not primary_vec and primary_text:
            embedded = await self._embed_queries([primary_text])

            if embedded and embedded[0]:
                primary_vec = embedded[0]

        missing_slots = [
            index
            for index, vec in enumerate(secondary_vecs)
            if not vec
        ]

        if missing_slots and secondary_texts:
            missing_texts = [secondary_texts[index] for index in missing_slots]
            embedded_secondary = await self._embed_queries(missing_texts)

            for offset, slot_index in enumerate(missing_slots):
                if offset < len(embedded_secondary):
                    vec = embedded_secondary[offset]

                    if vec:
                        secondary_vecs[slot_index] = vec

        return primary_text, primary_vec, secondary_texts, secondary_vecs

    def _align_secondary_embeddings(
        self,
        secondary_texts: list[str],
        all_queries: list[str],
        primary_text: str,
        provided: list[list[float]],
    ) -> list[list[float]]:
        count = len(secondary_texts)

        if count <= 0:
            return []

        if not provided:
            return [[] for _ in range(count)]

        cleaned_provided = [self._clean_vector(vec) for vec in provided]

        if len(cleaned_provided) == count:
            return cleaned_provided

        if all_queries and len(cleaned_provided) == len(all_queries):
            aligned: list[list[float]] = []

            for index, query in enumerate(all_queries):
                if query == primary_text:
                    continue

                if index < len(cleaned_provided):
                    aligned.append(cleaned_provided[index])
                else:
                    aligned.append([])

            while len(aligned) < count:
                aligned.append([])

            return aligned[:count]

        if all_queries and len(cleaned_provided) == len(all_queries) - 1:
            if clean_text(all_queries[0]) == primary_text:
                trimmed = cleaned_provided[:count]

                while len(trimmed) < count:
                    trimmed.append([])

                return trimmed

        if len(cleaned_provided) > count:
            return cleaned_provided[:count]

        padded = list(cleaned_provided)

        while len(padded) < count:
            padded.append([])

        return padded

    def _clean_vector(self, value: Any) -> list[float]:
        if not isinstance(value, list) or not value:
            return []

        cleaned: list[float] = []

        for item in value:
            try:
                cleaned.append(float(item))
            except (TypeError, ValueError):
                continue

        return cleaned

    async def _resolve_candidate_embeddings(
        self,
        candidates: list[VectorSearchResult],
        store: Any | None,
    ) -> list[list[float]]:
        if store is not None:
            stored = self._fetch_stored_embeddings(candidates, store)

            if stored is not None:
                return stored

        return await self._embed_candidates(candidates)

    def _fetch_stored_embeddings(
        self,
        candidates: list[VectorSearchResult],
        store: Any,
    ) -> list[list[float]] | None:
        fetch = getattr(store, "fetch_embeddings_by_ids", None)

        if not callable(fetch):
            return None

        doc_ids: set[str] = set()

        for candidate in candidates:
            doc_id = str(getattr(candidate.document, "doc_id", "") or "")

            if doc_id:
                doc_ids.add(doc_id)

        if not doc_ids:
            return None

        try:
            fetched = fetch(doc_ids)
        except Exception as exc:
            self._logger.warning(f"Stored embedding fetch failed: {exc}")
            return None

        if not isinstance(fetched, dict) or not fetched:
            return None

        coverage = len(fetched) / max(1, len(candidates))

        if coverage < _STORED_COVERAGE_FLOOR:
            return None

        result: list[list[float]] = []

        for candidate in candidates:
            doc_id = str(getattr(candidate.document, "doc_id", "") or "")
            vector = fetched.get(doc_id)

            if isinstance(vector, list) and vector:
                result.append(vector)
            else:
                result.append([])

        return result

    async def _embed_queries(self, queries: list[str]) -> list[list[float]]:
        if not queries:
            return []

        prepared: list[str] = []
        keys: list[str] = []

        for query in queries:
            cleaned = clean_text(query)
            prepared.append(cleaned)

            if cleaned:
                keys.append(self._cache_key(cleaned))
            else:
                keys.append("")

        if not any(prepared):
            return [[] for _ in prepared]

        vectors: list[list[float]] = [[] for _ in prepared]
        missing_texts: list[str] = []
        missing_slots: list[int] = []

        with self._lock:
            cached_snapshot = dict(self._query_cache)

        for index, key in enumerate(keys):
            if not key:
                continue

            hit = cached_snapshot.get(key)

            if hit:
                vectors[index] = hit
            else:
                missing_texts.append(prepared[index])
                missing_slots.append(index)

        if missing_texts:
            try:
                fresh = await self._embedder.embed_texts(missing_texts)
            except Exception as exc:
                self._logger.warning(f"Query embedding failed: {exc}")
                fresh = []

            with self._lock:
                for offset, slot_index in enumerate(missing_slots):
                    if offset >= len(fresh):
                        continue

                    vector = fresh[offset]

                    if not vector:
                        continue

                    vectors[slot_index] = vector
                    self._query_cache[keys[slot_index]] = vector

                if len(self._query_cache) > _QUERY_CACHE_MAX:
                    overflow = len(self._query_cache) - _QUERY_CACHE_MAX

                    for stale_key in list(self._query_cache.keys())[:overflow]:
                        self._query_cache.pop(stale_key, None)

        return vectors

    async def _embed_candidates(
        self,
        candidates: list[VectorSearchResult],
    ) -> list[list[float]]:
        texts: list[str] = []

        for candidate in candidates:
            document = candidate.document
            text = " ".join(
                part
                for part in (
                    clean_text(getattr(document, "title", "") or ""),
                    clean_text(getattr(document, "text", "") or ""),
                )
                if part
            )
            texts.append(text)

        if not any(texts):
            return []

        try:
            embedded = await self._embedder.embed_texts(texts)
        except Exception as exc:
            self._logger.warning(f"Candidate embedding failed: {exc}")
            return []

        if not isinstance(embedded, list):
            return []

        result: list[list[float]] = []

        for index in range(len(texts)):
            if index < len(embedded) and isinstance(embedded[index], list):
                result.append(embedded[index])
            else:
                result.append([])

        return result

    def _adaptive_blend(
        self,
        candidates: list[VectorSearchResult],
    ) -> float:
        scores: list[float] = []

        for candidate in candidates[:10]:
            try:
                scores.append(float(candidate.score))
            except (TypeError, ValueError):
                continue

        if len(scores) < 2:
            return self._clamp_blend(self._blend_weight)

        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        spread = math.sqrt(variance)

        if spread < _TIGHT_SPREAD_THRESHOLD:
            return self._clamp_blend(
                self._blend_weight + _TIGHT_BLEND_BOOST
            )

        if spread > _WIDE_SPREAD_THRESHOLD:
            return self._clamp_blend(
                self._blend_weight - _WIDE_BLEND_REDUCTION
            )

        return self._clamp_blend(self._blend_weight)

    def _clamp_blend(self, value: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = self._blend_weight

        return max(_FINAL_BLEND_MIN, min(_FINAL_BLEND_MAX, numeric))

    def _sort_by_original_score(
        self,
        candidates: list[VectorSearchResult],
        limit: int,
    ) -> list[VectorSearchResult]:
        ordered = sorted(
            candidates,
            key=lambda item: (-float(item.score), item.document.doc_id),
        )
        return ordered[:limit]

    def _trace_top(
        self,
        results: list[VectorSearchResult],
    ) -> list[dict[str, Any]]:
        top: list[dict[str, Any]] = []

        for result in results[:_TOP_TRACE_COUNT]:
            document = getattr(result, "document", None)
            doc_id = (
                str(getattr(document, "doc_id", "") or "")
                if document is not None
                else ""
            )
            title = (
                str(getattr(document, "title", "") or "")[:80]
                if document is not None
                else ""
            )

            try:
                score = round(float(result.score), 4)
            except (TypeError, ValueError):
                score = 0.0

            top.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "score": score,
                }
            )

        return top

    def _emit_trace(
        self,
        status: str,
        input_count: int,
        output_count: int,
        top_before: list[dict[str, Any]],
        top_after: list[dict[str, Any]],
        primary_query: str = "",
        secondary_count: int = 0,
        blend: float | None = None,
        primary_weight: float | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "status": status,
            "input_count": int(input_count),
            "output_count": int(output_count),
            "top_before": top_before,
            "top_after": top_after,
            "primary_query": primary_query[:120] if primary_query else "",
            "secondary_count": int(secondary_count),
        }

        if blend is not None:
            payload["blend"] = blend

        if primary_weight is not None:
            payload["primary_weight"] = primary_weight

        try:
            self._trace.emit("rerank_completed", **payload)
        except Exception as exc:
            self._logger.warning(f"Rerank trace emit failed: {exc}")

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(
            f"rerank|{text.lower()}".encode("utf-8")
        ).hexdigest()[:32]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0

        dot = 0.0
        left_norm = 0.0
        right_norm = 0.0

        for left_value, right_value in zip(left, right):
            try:
                lv = float(left_value)
                rv = float(right_value)
            except (TypeError, ValueError):
                continue

            dot += lv * rv
            left_norm += lv * lv
            right_norm += rv * rv

        if left_norm <= 0.0 or right_norm <= 0.0:
            return 0.0

        return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))