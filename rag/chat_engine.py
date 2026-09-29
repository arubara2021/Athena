from __future__ import annotations

import re
import traceback
from typing import Any, AsyncIterator

from pydantic import Field

from core.models import CoreModel
from rag.conversation import Conversation
from rag.embedder import Embedder
from rag.generator import Generator
from rag.query_expander import QueryExpander
from rag.reranker import Reranker
from rag.intent_router import IntentRouter
from rag.schemas import (
    ChatCitationSchema,
    FallbackStepSchema,
    TopicProfileSchema,
)
from rag.topic_analyzer import TopicAnalyzer
from rag.vector_store import VectorSearchResult, VectorStore
from utils.async_helpers import safe_aclose
from utils.logger import get_logger
from utils.text import clean_text

_SELF_REFERENCE_TOKENS = (
    "i have",
    "i've",
    "i did",
    "i asked",
    "we discussed",
    "we talked",
    "so far",
    "recently",
    "previously",
    "earlier",
    "before this",
    "my studies",
    "my study",
    "this session",
    "this chat",
    "our chat",
    "our session",
    "up to now",
    "until now",
)

_META_SESSION_VERBS = (
    "studying",
    "asked",
    "mentioned",
    "told you",
    "discussed",
    "talked about",
    "covered",
    "explored",
    "learned",
    "reviewed",
)

_CONVERSATION_STRONG_MARKERS = (
    "what have i",
    "what did i",
    "what was i",
    "what were we",
    "what have we",
    "what did we",
    "remind me what",
    "recap what",
    "summarize what i",
    "history of this",
    "history of our",
    "this conversation",
    "our conversation",
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_ANSWER_STYLE_SHORT = "short"
_ANSWER_STYLE_STANDARD = "standard"
_ANSWER_STYLE_DETAILED = "detailed"
_ANSWER_STYLE_EXHAUSTIVE = "exhaustive"

_VALID_ANSWER_STYLES = frozenset(
    {
        _ANSWER_STYLE_SHORT,
        _ANSWER_STYLE_STANDARD,
        _ANSWER_STYLE_DETAILED,
        _ANSWER_STYLE_EXHAUSTIVE,
    }
)

_RERANK_MIN_CANDIDATES = 2

_CONFIDENCE_WEIGHTS: dict[str, float] = {
    "retrieval": 0.35,
    "title_overlap": 0.20,
    "platform_coherence": 0.10,
    "coverage": 0.10,
    "length_match": 0.15,
    "citation_diversity": 0.10,
}

_LENGTH_MATCH_SHORT_LIMITS = (400, 800)
_LENGTH_MATCH_STANDARD_LIMITS = (400, 200)
_LENGTH_MATCH_DETAILED_LIMITS = (1800, 1200, 800)
_LENGTH_MATCH_EXHAUSTIVE_LIMITS = (2600, 1800, 1200)

_WEAK_TOPIC_CAP_MEDIUM = 0.35
_WEAK_TOPIC_CAP_LOW = 0.20
_WEAK_TOPIC_THRESHOLD_MEDIUM = 0.20
_WEAK_TOPIC_THRESHOLD_LOW = 0.10

_TOP_RESULTS_FOR_CONFIDENCE = 5

_MATCHED_CHUNK_MAX_CHARS = 1200
_REFUSAL_CONFIDENCE_CAP = 0.20

_REFUSAL_PHRASES = (
    "provided passages do not contain",
    "passages do not contain information",
    "passages do not contain enough",
    "provided context does not contain",
    "provided documents do not contain",
    "provided sources do not contain",
    "retrieved passages do not contain",
    "retrieved documents do not contain",
    "indexed sources do not contain",
    "does not contain information explaining",
    "do not contain information explaining",
    "does not contain enough information",
    "do not contain enough information",
    "not enough information in the provided",
    "not enough information in the retrieved",
    "insufficient information in the provided",
    "insufficient information in the retrieved",
    "cannot be answered from the provided",
    "cannot answer based on the provided",
    "unable to answer based on the provided",
    "unable to answer from the provided",
    "i do not have enough information",
    "i don't have enough information",
    "there is not enough information",
    "the sources do not provide",
    "no relevant information was found",
)


def _edit_threshold(token: str) -> int:
    if not token:
        return 0

    length = len(token)

    if token.isupper() and length <= 5:
        return 0

    if length <= 3:
        return 0

    if length <= 5:
        return 1

    if length <= 9:
        return 2

    return 3


def _edit_distance_within(a: str, b: str, max_distance: int) -> bool:
    if a == b:
        return True

    if max_distance <= 0:
        return False

    la = len(a)
    lb = len(b)

    if abs(la - lb) > max_distance:
        return False

    if la == 0 or lb == 0:
        return False

    if la > lb:
        a, b = b, a
        la, lb = lb, la

    previous = list(range(lb + 1))

    for i in range(1, la + 1):
        current = [i] + [0] * lb
        row_min = current[0]
        row_char = a[i - 1]

        for j in range(1, lb + 1):
            cost = 0 if row_char == b[j - 1] else 1

            insert_cost = previous[j] + 1
            delete_cost = current[j - 1] + 1
            substitute_cost = previous[j - 1] + cost

            best = insert_cost

            if delete_cost < best:
                best = delete_cost

            if substitute_cost < best:
                best = substitute_cost

            current[j] = best

            if best < row_min:
                row_min = best

        if row_min > max_distance:
            return False

        previous = current

    return previous[lb] <= max_distance


def _tokenize_title(text: str) -> set[str]:
    if not text:
        return set()

    return set(_TOKEN_RE.findall(str(text).lower()))


def _fuzzy_token_match(query_token: str, title_tokens: set[str]) -> bool:
    if not query_token or not title_tokens:
        return False

    if query_token in title_tokens:
        return True

    threshold = _edit_threshold(query_token)

    if threshold <= 0:
        return False

    query_len = len(query_token)

    for candidate in title_tokens:
        if not candidate:
            continue

        if abs(len(candidate) - query_len) > threshold:
            continue

        if _edit_distance_within(query_token, candidate, threshold):
            return True

    return False


class RAGCitation(CoreModel):
    source_id: str
    title: str
    url: str | None = None
    score: float = 0.0
    snippet: str = ""


class RAGAnswer(CoreModel):
    question: str
    answer: str
    citations: list[RAGCitation] = Field(default_factory=list)
    context_used: int = 0
    llm_used: bool = False
    fallback: bool = False
    strategy: str = ""
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    topic: TopicProfileSchema | None = None
    expanded_queries: list[str] = Field(default_factory=list)
    fallback_ladder: list[FallbackStepSchema] = Field(default_factory=list)
    session_id: str = ""
    answer_style: str = _ANSWER_STYLE_STANDARD


class RAGChatEngine:
    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedder: Embedder | None = None,
        top_k: int = 5,
        max_context_chars: int = 6000,
        session_id: str | None = None,
        conversation: Conversation | None = None,
        source_diversity: int = 0,
        enable_expansion: bool = True,
        enable_rerank: bool = True,
        enable_generation: bool = True,
    ) -> None:
        self._owns_vector_store = vector_store is None
        self._owns_embedder = embedder is None
        self._owns_conversation = conversation is None
        self._vector_store = vector_store or VectorStore()
        self._embedder = embedder or Embedder()
        self._chunk_gate_threshold = self._resolve_chunk_gate_threshold()
        self._top_k = max(1, int(top_k or 5))
        self._max_context_chars = max(1000, int(max_context_chars or 6000))
        self._source_diversity = max(0, int(source_diversity or 0))
        self._enable_expansion = bool(enable_expansion)
        self._enable_rerank = bool(enable_rerank)
        self._enable_generation = bool(enable_generation)

        self._logger = get_logger("rag.chat_engine")

        self._analyzer = TopicAnalyzer()
        self._intent_router = IntentRouter()
        self._expander = QueryExpander()
        self._reranker = Reranker(
            embedder=self._embedder,
            store=self._vector_store,
        )
        self._generator = Generator()
        self._conversation = (
            conversation
            if conversation is not None
            else Conversation(session_id=session_id)
        )

        self._warmed_up = False

    @property
    def vector_store(self) -> VectorStore:
        return self._vector_store

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def conversation(self) -> Conversation:
        return self._conversation

    @property
    def session_id(self) -> str:
        return self._conversation.session_id

    @property
    def owns_embedder(self) -> bool:
        return self._owns_embedder

    @property
    def owns_vector_store(self) -> bool:
        return self._owns_vector_store

    def _resolve_chunk_gate_threshold(self) -> int | None:
        try:
            value = getattr(
                self._vector_store,
                "chunk_query_token_threshold",
                None,
            )
        except Exception:
            return None

        if value is None:
            return None

        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None

        if parsed < 1:
            return None

        return parsed

    async def __aenter__(self) -> "RAGChatEngine":
        await self._warm_up()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        await self.aclose()
        return False

    async def _warm_up(self) -> None:
        if self._warmed_up:
            return

        try:
            self._vector_store.count()
        except Exception as exc:
            self._logger.warning(f"Vector store warm-up failed: {exc}")

        self._warmed_up = True

    async def aclose(self) -> None:
        issues: list[str] = []

        if self._owns_embedder and self._embedder is not None:
            ok = await safe_aclose(self._embedder, logger=self._logger)

            if not ok:
                issues.append("embedder_close_failed")

        if self._owns_vector_store and self._vector_store is not None:
            try:
                close = getattr(self._vector_store, "close", None)

                if callable(close):
                    close()
            except Exception as exc:
                self._logger.warning(f"Vector store close failed: {exc}")
                issues.append("vector_store_close_failed")

        if issues:
            self._logger.warning(
                f"RAGChatEngine close encountered issues: {', '.join(issues)}"
            )

    async def close(self) -> None:
        await self.aclose()

    async def ask(
        self,
        question: str,
        run_id: str | None = None,
        top_k: int | None = None,
    ) -> RAGAnswer:
        cleaned_question = clean_text(question)
        session_id = self._conversation.session_id

        diagnostics: dict[str, Any] = {
            "store_path": str(self._vector_store.path),
            "run_id": run_id,
            "session_id": session_id,
        }
        ladder: list[FallbackStepSchema] = []

        if not cleaned_question:
            diagnostics["reason"] = "empty_question"

            return RAGAnswer(
                question=question,
                answer="Please ask a non-empty question.",
                citations=[],
                context_used=0,
                llm_used=False,
                strategy="none",
                diagnostics=diagnostics,
                session_id=session_id,
                answer_style=_ANSWER_STYLE_STANDARD,
            )

        limit = max(1, int(top_k or self._top_k))

        classification = await self._intent_router.classify(cleaned_question)
        diagnostics["catalog_classification"] = {
            "route": classification.route,
            "score": classification.score,
            "confidence": classification.confidence,
            "threshold": classification.threshold,
            "signals": dict(classification.signals),
            "matched": {
                key: list(value)
                for key, value in classification.matched.items()
            },
            "reason": classification.reason,
        }

        if classification.is_catalog:
            self._logger.info(
                f"RAG catalog route: question={cleaned_question!r} "
                f"score={classification.score:.2f} "
                f"confidence={classification.confidence:.2f} "
                f"signals={classification.signals}"
            )

            return await self._delegate_catalog(
                question=cleaned_question,
                session_id=session_id,
                run_id=run_id,
                diagnostics=diagnostics,
                classification=classification,
                answer_style=_ANSWER_STYLE_STANDARD,
            )

        topic_profile = self._analyzer.analyze(
            cleaned_question,
            store=self._vector_store,
        )

        answer_style = self._resolve_answer_style(topic_profile)

        diagnostics["topic"] = {
            "primary": topic_profile.primary_topic,
            "intent": topic_profile.intent,
            "intent_confidence": topic_profile.intent_confidence,
            "answer_style": answer_style,
            "subtopics": list(topic_profile.subtopics)[:5],
            "related": list(topic_profile.related_concepts)[:5],
            "acronyms": list(topic_profile.acronyms)[:5],
            "corpus_available": topic_profile.corpus_available,
        }
        diagnostics["answer_style"] = answer_style

        route = self._classify_question(
            cleaned_question,
            topic_profile,
            self._conversation,
        )
        diagnostics["route"] = route

        if route == "conversation":
            return self._answer_from_conversation(
                cleaned_question,
                session_id,
                diagnostics,
                answer_style=answer_style,
            )

        try:
            store_total = self._vector_store.count()
        except Exception as exc:
            store_total = 0
            diagnostics["count_error"] = str(exc)

        diagnostics["store_total_documents"] = store_total

        if store_total <= 0:
            diagnostics["reason"] = "empty_vector_store"
            ladder.append(
                FallbackStepSchema(
                    stage="store_check",
                    reason="empty_store",
                    action="return_empty",
                    severity="warn",
                )
            )

            return RAGAnswer(
                question=cleaned_question,
                answer=(
                    "The local index is empty. Run search or agent with "
                    "--rag to build the index first."
                ),
                citations=[],
                context_used=0,
                llm_used=False,
                fallback=True,
                strategy="empty_store",
                diagnostics=diagnostics,
                fallback_ladder=ladder,
                session_id=session_id,
                answer_style=answer_style,
            )

        self._logger.info(
            f"RAG ask: question={cleaned_question!r} "
            f"run_id={run_id} store_total={store_total} route={route} "
            f"answer_style={answer_style}"
        )

        expanded_queries = [cleaned_question]

        if self._enable_expansion:
            try:
                expanded_queries = self._expander.expand(
                    question=cleaned_question,
                    profile=topic_profile,
                    conversation=self._conversation,
                    store=self._vector_store,
                )
            except Exception as exc:
                self._logger.warning(f"Query expansion failed: {exc}")
                ladder.append(
                    FallbackStepSchema(
                        stage="query_expansion",
                        reason=str(exc),
                        action="use_original_query",
                        severity="warn",
                    )
                )

        if not expanded_queries:
            expanded_queries = [cleaned_question]

        diagnostics["expanded_queries"] = list(expanded_queries)

        embed_inputs = [cleaned_question] + [
            q for q in expanded_queries[1:] if clean_text(q)
        ]

        embeddings: list[list[float]] = []

        try:
            embeddings = await self._embedder.embed_texts(embed_inputs)
        except Exception as exc:
            self._logger.warning(
                f"RAG embedding failed: {exc}\n{traceback.format_exc()}"
            )
            ladder.append(
                FallbackStepSchema(
                    stage="embedding",
                    reason=str(exc),
                    action="continue_without_embedding",
                    severity="warn",
                )
            )

        query_embedding: list[float] = []
        expanded_embeddings: list[list[float]] = []

        if embeddings:
            query_embedding = embeddings[0] if embeddings[0] else []
            expanded_embeddings = [
                vec for vec in embeddings[1:] if vec
            ]

        if not query_embedding:
            for fallback in expanded_embeddings:
                if fallback:
                    query_embedding = fallback
                    break

        diagnostics["embedding_dimensions"] = len(query_embedding)

        if store_total < 100:
            broad_limit = limit * 4
        elif store_total < 1000:
            broad_limit = limit * 6
        else:
            broad_limit = limit * 8

        broad_limit = max(broad_limit, limit * 3)
        diagnostics["broad_limit"] = broad_limit

        results: list[VectorSearchResult] = []
        strategy = ""

        try:
            results = self._vector_store.hybrid_search_v2(
                query_text=cleaned_question,
                query_embedding=query_embedding,
                top_k=broad_limit,
                run_id=run_id,
                chunk_token_threshold=self._chunk_gate_threshold,
            )
            strategy = "hybrid_v2"
        except Exception as exc:
            self._logger.warning(
                f"RAG hybrid_v2 failed: {exc}\n{traceback.format_exc()}"
            )
            ladder.append(
                FallbackStepSchema(
                    stage="retrieval.hybrid_v2",
                    reason=str(exc),
                    action="try_hybrid",
                    severity="warn",
                )
            )

        if not results:
            try:
                results = self._vector_store.hybrid_search(
                    query_text=cleaned_question,
                    query_embedding=query_embedding,
                    top_k=broad_limit,
                    run_id=run_id,
                )

                if results:
                    strategy = "hybrid"
                    ladder.append(
                        FallbackStepSchema(
                            stage="retrieval.hybrid",
                            reason="hybrid_v2_empty",
                            action="use_hybrid",
                            severity="info",
                        )
                    )
            except Exception as exc:
                self._logger.warning(f"RAG hybrid search failed: {exc}")

        if not results:
            try:
                results = self._vector_store.keyword_search(
                    query_text=cleaned_question,
                    top_k=broad_limit,
                    run_id=run_id,
                )

                if results:
                    strategy = "keyword"
                    ladder.append(
                        FallbackStepSchema(
                            stage="retrieval.keyword",
                            reason="hybrid_empty",
                            action="use_keyword",
                            severity="info",
                        )
                    )
            except Exception as exc:
                self._logger.warning(f"RAG keyword search failed: {exc}")

        if not results:
            try:
                results = self._vector_store.search(
                    query_embedding=query_embedding,
                    top_k=broad_limit,
                    run_id=run_id,
                )

                if results:
                    strategy = "embedding_only"
                    ladder.append(
                        FallbackStepSchema(
                            stage="retrieval.embedding",
                            reason="keyword_empty",
                            action="use_embedding",
                            severity="info",
                        )
                    )
            except Exception as exc:
                self._logger.warning(f"RAG embedding search failed: {exc}")

        if not results and run_id is not None:
            try:
                results = self._vector_store.hybrid_search_v2(
                    query_text=cleaned_question,
                    query_embedding=query_embedding,
                    top_k=broad_limit,
                    run_id=None,
                    chunk_token_threshold=self._chunk_gate_threshold,
                )

                if results:
                    strategy = "hybrid_all_runs"
                    diagnostics["fallback_to_all_runs"] = True
                    ladder.append(
                        FallbackStepSchema(
                            stage="retrieval.all_runs",
                            reason="run_id_filter_empty",
                            action="drop_run_id_filter",
                            severity="info",
                        )
                    )
            except Exception as exc:
                self._logger.warning(
                    f"RAG unfiltered hybrid search failed: {exc}"
                )

        threshold_diagnostics = self._read_threshold_diagnostics()

        diagnostics["broad_results"] = len(results)
        diagnostics["strategy"] = strategy
        diagnostics["threshold_used"] = threshold_diagnostics.get("threshold")
        diagnostics["threshold_source"] = threshold_diagnostics.get(
            "threshold_source", ""
        )
        diagnostics["threshold_floor"] = self._read_threshold_floor()
        diagnostics["retrieval_operation"] = threshold_diagnostics.get(
            "operation", ""
        )
        diagnostics["retrieval_path"] = self._classify_retrieval_path(
            strategy,
            threshold_diagnostics,
        )
        diagnostics["chunk_gate_threshold"] = self._chunk_gate_threshold
        diagnostics["chunk_gate_token_count"] = len(
            self._question_tokens(cleaned_question)
        )
        diagnostics["chunk_gate_effective"] = threshold_diagnostics.get(
            "chunk_gate_effective"
        )
        diagnostics["chunk_gate_source"] = threshold_diagnostics.get(
            "chunk_gate_source", ""
        )

        if not results:
            ladder.append(
                FallbackStepSchema(
                    stage="retrieval",
                    reason="no_results",
                    action="return_empty",
                    severity="warn",
                )
            )

            self._log_conversation_turn(
                cleaned_question,
                "",
                [],
                fallback=True,
                confidence=0.0,
                strategy="no_match",
            )

            return RAGAnswer(
                question=cleaned_question,
                answer=(
                    "No matching passages were found in the local index. "
                    "The store contains documents but none match this query."
                ),
                citations=[],
                context_used=0,
                llm_used=False,
                fallback=True,
                strategy="no_match",
                diagnostics=diagnostics,
                topic=topic_profile,
                expanded_queries=list(expanded_queries),
                fallback_ladder=ladder,
                session_id=session_id,
                answer_style=answer_style,
            )

        deduped = self._deduplicate_results(results)
        diagnostics["deduped_count"] = len(deduped)

        ranked: list[VectorSearchResult] = deduped
        rerank_applied = False

        if self._enable_rerank and len(deduped) >= _RERANK_MIN_CANDIDATES:
            try:
                rerank_queries = [
                    query
                    for query in expanded_queries[1:]
                    if clean_text(query)
                ]

                ranked = await self._reranker.rerank(
                    candidates=deduped,
                    queries=rerank_queries,
                    top_k=max(limit * 3, limit),
                    store=self._vector_store,
                    query_embeddings=expanded_embeddings,
                    primary_query=cleaned_question,
                    primary_query_embedding=query_embedding,
                )

                diagnostics["reranked_count"] = len(ranked)
                rerank_applied = True
            except Exception as exc:
                self._logger.warning(f"Rerank failed: {exc}")
                ladder.append(
                    FallbackStepSchema(
                        stage="rerank",
                        reason=str(exc),
                        action="keep_original_order",
                        severity="warn",
                    )
                )
                ranked = deduped

        diagnostics["rerank_applied"] = rerank_applied

        diversity_applied, diversity_dropped = self._apply_diversity(ranked)
        diagnostics["diversity_dropped_count"] = int(diversity_dropped)
        ranked = diversity_applied

        top_results = ranked[:limit]

        if not top_results:
            ladder.append(
                FallbackStepSchema(
                    stage="selection",
                    reason="empty_after_ranking",
                    action="return_empty",
                    severity="warn",
                )
            )

            return RAGAnswer(
                question=cleaned_question,
                answer="No usable passages remained after ranking.",
                citations=[],
                context_used=0,
                llm_used=False,
                fallback=True,
                strategy="no_match_after_rerank",
                diagnostics=diagnostics,
                topic=topic_profile,
                expanded_queries=list(expanded_queries),
                fallback_ladder=ladder,
                session_id=session_id,
                answer_style=answer_style,
            )

        diag_scores = [
            float(getattr(item, "score", 0.0) or 0.0)
            for item in top_results
        ]

        diagnostics["final_results"] = len(top_results)
        diagnostics["mean_score"] = (
            round(sum(diag_scores) / len(diag_scores), 4)
            if diag_scores
            else 0.0
        )
        diagnostics["max_score"] = (
            round(max(diag_scores), 4) if diag_scores else 0.0
        )
        diagnostics["min_score"] = (
            round(min(diag_scores), 4) if diag_scores else 0.0
        )
        diagnostics["top_titles"] = [
            str(getattr(item.document, "title", "") or "")[:120]
            for item in top_results[:3]
        ]

        passages = self._build_passages(top_results)

        generated_text = ""
        generated_cited: list[str] = []
        generated_confidence = 0.0
        generated_llm_used = False
        generated_answer_style = answer_style

        if self._enable_generation:
            try:
                generated = await self._generator.generate(
                    question=cleaned_question,
                    topic=topic_profile,
                    passages=passages,
                )
                generated_text = generated.text
                generated_cited = generated.cited_ids
                generated_confidence = generated.confidence
                generated_llm_used = generated.llm_used
                ladder.extend(generated.fallback_ladder)

                if getattr(generated, "answer_style", None):
                    candidate_style = str(
                        generated.answer_style
                    ).strip().lower()

                    if candidate_style in _VALID_ANSWER_STYLES:
                        generated_answer_style = candidate_style
            except Exception as exc:
                self._logger.warning(f"Generation failed: {exc}")
                ladder.append(
                    FallbackStepSchema(
                        stage="generation",
                        reason=str(exc),
                        action="use_extractive",
                        severity="warn",
                    )
                )

        if not generated_text:
            generated_text = self._build_extractive_answer(
                cleaned_question,
                top_results,
            )

        citations = self._build_citations(top_results)

        if generated_cited:
            cited_set = set(generated_cited)
            ordered = [
                citation
                for citation in citations
                if citation.source_id in cited_set
            ]
            remaining = [
                citation
                for citation in citations
                if citation.source_id not in cited_set
            ]
            citations = ordered + remaining

        title_overlap = self._compute_title_overlap(
            cleaned_question,
            top_results,
        )
        platform_coherence = self._compute_platform_coherence(top_results)

        diagnostics["title_overlap"] = round(title_overlap, 4)
        diagnostics["platform_coherence"] = round(platform_coherence, 4)

        answer_length = len(generated_text or "")

        confidence, confidence_breakdown = self._finalize_confidence(
            generated_confidence=generated_confidence,
            results=top_results,
            citations=citations,
            llm_used=generated_llm_used,
            title_overlap=title_overlap,
            platform_coherence=platform_coherence,
            answer_style=generated_answer_style,
            answer_length=answer_length,
        )

        diagnostics["confidence_breakdown"] = {
            key: round(float(value), 4)
            for key, value in confidence_breakdown.items()
        }

        refusal_detected = self._detect_refusal(generated_text)

        diagnostics["llm_refusal"] = bool(refusal_detected)

        if refusal_detected:
            confidence = min(confidence, _REFUSAL_CONFIDENCE_CAP)

            ladder.append(
                FallbackStepSchema(
                    stage="generation",
                    reason="llm_declared_insufficient_context",
                    action="cap_confidence",
                    severity="warn",
                )
            )

        weak_match = False

        if title_overlap < _WEAK_TOPIC_THRESHOLD_MEDIUM:
            confidence = min(confidence, _WEAK_TOPIC_CAP_MEDIUM)

        if title_overlap < _WEAK_TOPIC_THRESHOLD_LOW:
            confidence = min(confidence, _WEAK_TOPIC_CAP_LOW)
            weak_match = True

        diagnostics["confidence"] = round(confidence, 4)
        diagnostics["citation_count"] = len(citations)
        diagnostics["weak_topic_match"] = weak_match

        fallback_flag = bool(weak_match or refusal_detected)

        if weak_match:
            ladder.append(
                FallbackStepSchema(
                    stage="confidence",
                    reason="weak_topic_match",
                    action="cap_confidence",
                    severity="warn",
                )
            )

        self._log_conversation_turn(
            cleaned_question,
            generated_text,
            citations,
            fallback=fallback_flag,
            confidence=confidence,
            strategy=strategy or "unknown",
        )

        return RAGAnswer(
            question=cleaned_question,
            answer=generated_text,
            citations=citations,
            context_used=len(top_results),
            llm_used=generated_llm_used,
            fallback=fallback_flag,
            strategy=strategy or "unknown",
            diagnostics=diagnostics,
            confidence=round(confidence, 4),
            topic=topic_profile,
            expanded_queries=list(expanded_queries),
            fallback_ladder=ladder,
            session_id=session_id,
            answer_style=generated_answer_style,
        )

    async def ask_safe(
        self,
        question: str,
        run_id: str | None = None,
        top_k: int | None = None,
    ) -> RAGAnswer:
        try:
            return await self.ask(question, run_id=run_id, top_k=top_k)
        except Exception as exc:
            self._logger.warning(
                f"RAG ask_safe caught failure: {exc}\n{traceback.format_exc()}"
            )

            return RAGAnswer(
                question=str(question or ""),
                answer=(
                    "The engine failed to answer this question. "
                    "Check logs for the underlying error."
                ),
                citations=[],
                context_used=0,
                llm_used=False,
                fallback=True,
                strategy="engine_error",
                diagnostics={
                    "error": str(exc),
                    "store_path": str(self._vector_store.path),
                    "run_id": run_id,
                },
                fallback_ladder=[
                    FallbackStepSchema(
                        stage="engine",
                        reason=str(exc),
                        action="return_engine_error",
                        severity="error",
                    )
                ],
                session_id=self._conversation.session_id,
                answer_style=_ANSWER_STYLE_STANDARD,
            )

    async def ask_stream(
        self,
        question: str,
        run_id: str | None = None,
        top_k: int | None = None,
    ) -> AsyncIterator[str]:
        answer = await self.ask(question, run_id=run_id, top_k=top_k)

        async for chunk in self._stream_text(answer.answer):
            yield chunk

    async def _stream_text(self, text: str) -> AsyncIterator[str]:
        value = str(text or "")

        if not value:
            return

        step = 72
        index = 0

        while index < len(value):
            yield value[index : index + step]
            index += step

    def _classify_question(
        self,
        question: str,
        topic_profile: TopicProfileSchema,
        conversation: Conversation,
    ) -> str:
        text = clean_text(question).lower()

        if not text:
            return "corpus"

        for marker in _CONVERSATION_STRONG_MARKERS:
            if marker in text:
                return "conversation"

        has_topic = bool(clean_text(topic_profile.primary_topic))

        self_reference_hits = 0

        for token in _SELF_REFERENCE_TOKENS:
            if token in text:
                self_reference_hits += 1

        meta_verb_hits = 0

        for verb in _META_SESSION_VERBS:
            if verb in text:
                meta_verb_hits += 1

        pronoun_hits = 0

        for pronoun in (
            " i ", " i've ", " my ", " we ", " our ", " me ",
            " i,", " i.", " i?", " me,", " me.", " me?",
        ):
            if pronoun in f" {text} ":
                pronoun_hits += 1

        history_count = 0

        try:
            history_count = len(conversation.get_turns())
        except Exception:
            history_count = 0

        signals = self_reference_hits + meta_verb_hits + pronoun_hits

        if not has_topic and signals >= 2:
            return "conversation"

        if not has_topic and meta_verb_hits >= 1 and pronoun_hits >= 1:
            return "conversation"

        if self_reference_hits >= 2 and meta_verb_hits >= 1:
            return "conversation"

        if history_count > 0 and not has_topic and signals >= 3:
            return "conversation"

        return "corpus"

    def _answer_from_conversation(
        self,
        question: str,
        session_id: str,
        diagnostics: dict[str, Any],
        answer_style: str = _ANSWER_STYLE_STANDARD,
    ) -> RAGAnswer:
        diagnostics["strategy"] = "conversation_memory"
        diagnostics["fallback"] = False

        try:
            topics = self._conversation.recent_topic_summary(limit=12)
        except Exception as exc:
            diagnostics["conversation_error"] = str(exc)
            topics = []

        if not topics:
            answer_text = (
                "This session does not have any prior questions yet. "
                "Ask a question and I will remember it."
            )
        else:
            lines: list[str] = [
                f"In this session you have asked about "
                f"{len(topics)} distinct topic(s):",
                "",
            ]

            for index, entry in enumerate(topics, start=1):
                topic = str(entry.get("topic", "") or "").strip()
                count = int(entry.get("count", 1) or 1)
                last_seen = str(entry.get("last_seen", "") or "")

                short_time = ""

                if last_seen:
                    short_time = last_seen[:19].replace("T", " ")

                suffix = f"  (x{count})" if count > 1 else ""

                if short_time:
                    lines.append(f"{index}. {topic}{suffix}  ({short_time})")
                else:
                    lines.append(f"{index}. {topic}{suffix}")

            lines.append("")
            lines.append(
                "Ask a follow-up to continue any of these threads."
            )

            answer_text = "\n".join(lines)

        diagnostics["conversation_topics"] = len(topics)
        diagnostics["citation_count"] = 0

        return RAGAnswer(
            question=question,
            answer=answer_text,
            citations=[],
            context_used=len(topics),
            llm_used=False,
            fallback=False,
            strategy="conversation_memory",
            diagnostics=diagnostics,
            confidence=1.0,
            expanded_queries=[],
            fallback_ladder=[],
            session_id=session_id,
            answer_style=answer_style,
        )

    async def _delegate_catalog(
        self,
        question: str,
        session_id: str,
        run_id: str | None,
        diagnostics: dict[str, Any],
        classification: Any,
        answer_style: str = _ANSWER_STYLE_STANDARD,
    ) -> RAGAnswer:
        diagnostics["strategy"] = "catalog_snapshot"

        try:
            from rag.catalog_service import CatalogService
        except ImportError as exc:
            self._logger.warning(f"CatalogService import failed: {exc}")
            diagnostics["catalog_service_unavailable"] = str(exc)

            return RAGAnswer(
                question=question,
                answer=(
                    "Catalog mode is unavailable because the catalog "
                    "service module is not installed. Ask a topical "
                    "question, or install rag.catalog_service."
                ),
                citations=[],
                context_used=0,
                llm_used=False,
                fallback=True,
                strategy="catalog_unavailable",
                diagnostics=diagnostics,
                confidence=0.0,
                fallback_ladder=[
                    FallbackStepSchema(
                        stage="catalog_service",
                        reason="module_not_installed",
                        action="return_unavailable",
                        severity="warn",
                    )
                ],
                session_id=session_id,
                answer_style=answer_style,
            )

        service = CatalogService(
            vector_store=self._vector_store,
            generator=self._generator,
        )

        try:
            result = await service.answer(
                question=question,
                run_id=run_id,
                session_id=session_id,
                classification=classification,
            )
        except Exception as exc:
            self._logger.warning(
                f"Catalog service failed: {exc}\n{traceback.format_exc()}"
            )
            diagnostics["catalog_service_error"] = str(exc)

            return RAGAnswer(
                question=question,
                answer=(
                    "Catalog lookup failed. Check logs for the "
                    "underlying error."
                ),
                citations=[],
                context_used=0,
                llm_used=False,
                fallback=True,
                strategy="catalog_error",
                diagnostics=diagnostics,
                confidence=0.0,
                fallback_ladder=[
                    FallbackStepSchema(
                        stage="catalog_service",
                        reason=str(exc),
                        action="return_error",
                        severity="error",
                    )
                ],
                session_id=session_id,
                answer_style=answer_style,
            )

        text = str(getattr(result, "text", "") or "")
        llm_used = bool(getattr(result, "llm_used", False))
        context_used = int(getattr(result, "context_used", 0) or 0)
        confidence = float(getattr(result, "confidence", 0.0) or 0.0)
        breakdown = getattr(result, "confidence_breakdown", None) or {}
        snapshot = getattr(result, "snapshot", None)
        escalation = getattr(result, "escalation", None)
        ladder = list(getattr(result, "fallback_ladder", []) or [])

        refusal = self._detect_refusal(text)

        if refusal:
            confidence = min(confidence, _REFUSAL_CONFIDENCE_CAP)
            ladder.append(
                FallbackStepSchema(
                    stage="catalog_generation",
                    reason="refusal_detected",
                    action="cap_confidence",
                    severity="warn",
                )
            )

        diagnostics["catalog_refusal_detected"] = bool(refusal)
        diagnostics["catalog_confidence"] = round(confidence, 4)

        if isinstance(breakdown, dict) and breakdown:
            diagnostics["catalog_confidence_breakdown"] = {
                str(key): round(float(value), 4)
                for key, value in breakdown.items()
                if isinstance(value, (int, float))
            }

        if snapshot is not None:
            diagnostics["catalog_snapshot"] = snapshot

        if escalation:
            diagnostics["catalog_escalation"] = str(escalation)

        fallback_flag = bool(refusal or escalation)

        self._log_conversation_turn(
            question=question,
            answer=text,
            citations=[],
            fallback=fallback_flag,
            confidence=confidence,
            strategy="catalog_snapshot",
        )

        return RAGAnswer(
            question=question,
            answer=text,
            citations=[],
            context_used=context_used,
            llm_used=llm_used,
            fallback=fallback_flag,
            strategy="catalog_snapshot",
            diagnostics=diagnostics,
            confidence=round(confidence, 4),
            fallback_ladder=ladder,
            session_id=session_id,
            answer_style=answer_style,
        )

    def _compute_title_overlap(
        self,
        question: str,
        results: list[VectorSearchResult],
    ) -> float:
        if not results:
            return 0.0

        tokens = self._question_tokens(question)

        if not tokens:
            return 0.0

        top = results[:3]
        title_tokens: set[str] = set()

        for item in top:
            title = str(getattr(item.document, "title", "") or "")
            title_tokens.update(_tokenize_title(title))

        if not title_tokens:
            return 0.0

        matched = 0

        for token in tokens:
            if _fuzzy_token_match(token, title_tokens):
                matched += 1

        return matched / len(tokens)

    def _compute_platform_coherence(
        self,
        results: list[VectorSearchResult],
    ) -> float:
        if not results:
            return 0.0

        top = results[: min(5, len(results))]
        platforms: list[str] = []

        for item in top:
            platform = str(
                getattr(item.document, "platform", "") or ""
            ).strip().lower()

            if platform:
                platforms.append(platform)

        if not platforms:
            return 0.0

        most_common = max(set(platforms), key=platforms.count)

        return platforms.count(most_common) / len(platforms)

    def _question_tokens(self, question: str) -> list[str]:
        raw = _TOKEN_RE.findall(str(question or "").lower())

        return [
            token
            for token in raw
            if len(token) > 2
            and token
            not in {
                "the", "a", "an", "and", "or", "of", "to", "in", "on",
                "for", "with", "is", "are", "was", "were", "be", "been",
                "being", "what", "how", "why", "when", "where", "which",
                "who", "whom", "this", "that", "these", "those", "i",
                "you", "he", "she", "it", "we", "they", "me", "him",
                "her", "us", "them", "my", "your", "his", "its", "our",
                "their", "do", "does", "did", "doing", "have", "has",
                "had", "having", "can", "could", "should", "would",
                "may", "might", "must", "will", "shall", "not", "no",
                "yes", "about", "into", "over", "under", "between",
                "from", "as", "at", "by", "if", "then", "than", "too",
                "very", "just", "also", "tell", "show", "give", "know",
                "right", "explain", "describe", "detail", "detailed",
                "details", "please", "need", "want",
            }
        ]

    def _apply_diversity(
        self,
        results: list[VectorSearchResult],
    ) -> tuple[list[VectorSearchResult], int]:
        cap = self._source_diversity

        if cap <= 0 or not results:
            return list(results), 0

        counts: dict[str, int] = {}
        kept: list[VectorSearchResult] = []
        dropped = 0

        for result in results:
            platform = str(result.document.platform or "unknown")
            current = counts.get(platform, 0)

            if current >= cap:
                dropped += 1
                continue

            counts[platform] = current + 1
            kept.append(result)

        if dropped == 0:
            return list(results), 0

        return kept, dropped

    def _build_passages(
        self,
        results: list[VectorSearchResult],
    ) -> list[dict[str, Any]]:
        passages: list[dict[str, Any]] = []

        for result in results:
            document = result.document
            snippet = self._extract_matched_chunk(document)

            if not snippet:
                snippet = self._best_snippet(document.text)

            if not snippet:
                snippet = clean_text(document.text)

            passages.append(
                {
                    "source_id": document.source_id,
                    "title": document.title,
                    "url": document.url or "",
                    "text": snippet,
                    "platform": document.platform or "",
                    "source_type": document.source_type or "",
                    "score": float(result.score),
                }
            )

        return passages

    def _extract_matched_chunk(self, document: Any) -> str:
        if document is None:
            return ""

        metadata = getattr(document, "metadata", None)

        if not isinstance(metadata, dict):
            return ""

        raw = metadata.get("matched_chunk")

        if not isinstance(raw, str):
            return ""

        cleaned = clean_text(raw)

        if not cleaned:
            return ""

        if len(cleaned) > _MATCHED_CHUNK_MAX_CHARS:
            cleaned = cleaned[:_MATCHED_CHUNK_MAX_CHARS].rstrip() + "..."

        return cleaned

    def _deduplicate_results(
        self,
        results: list[VectorSearchResult],
    ) -> list[VectorSearchResult]:
        seen: dict[str, VectorSearchResult] = {}
        order: list[str] = []

        for result in results:
            key = result.document.source_id or result.document.doc_id
            existing = seen.get(key)

            if existing is None:
                seen[key] = result
                order.append(key)
            elif result.score > existing.score:
                seen[key] = result

        return [seen[key] for key in order]

    def _build_extractive_answer(
        self,
        question: str,
        results: list[VectorSearchResult],
    ) -> str:
        lines = [
            f"Question: {question}",
            "",
            "Locally retrieved passages:",
            "",
        ]

        for index, result in enumerate(results[: self._top_k], start=1):
            document = result.document
            snippet = self._best_snippet(document.text)
            meta = " | ".join(
                part
                for part in (
                    str(document.platform or ""),
                    str(document.source_type or ""),
                    f"score {result.score:.3f}",
                )
                if part
            )

            lines.append(f"{index}. {document.title}")

            if meta:
                lines.append(f"   {meta}")

            if document.url:
                lines.append(f"   URL: {document.url}")

            if snippet:
                lines.append(f"   {snippet}")

            lines.append("")

        lines.append(
            "This answer is extractive and drawn only from locally "
            "indexed research passages."
        )

        return "\n".join(lines).strip()

    def _build_citations(
        self,
        results: list[VectorSearchResult],
    ) -> list[RAGCitation]:
        citations: list[RAGCitation] = []

        for result in results[: self._top_k]:
            document = result.document
            snippet = self._best_snippet(document.text)

            citations.append(
                RAGCitation(
                    source_id=document.source_id,
                    title=document.title,
                    url=document.url,
                    score=round(float(result.score), 4),
                    snippet=snippet,
                )
            )

        return citations

    def _finalize_confidence(
        self,
        generated_confidence: float,
        results: list[VectorSearchResult],
        citations: list[RAGCitation],
        llm_used: bool,
        title_overlap: float = 0.0,
        platform_coherence: float = 0.0,
        answer_style: str = _ANSWER_STYLE_STANDARD,
        answer_length: int = 0,
    ) -> tuple[float, dict[str, float]]:
        breakdown: dict[str, float] = {
            "retrieval": 0.0,
            "title_overlap": 0.0,
            "platform_coherence": 0.0,
            "coverage": 0.0,
            "length_match": 0.0,
            "citation_diversity": 0.0,
        }

        if not results:
            return 0.0, breakdown

        top = results[: min(_TOP_RESULTS_FOR_CONFIDENCE, len(results))]
        scores: list[float] = []

        for result in top:
            try:
                scores.append(float(result.score))
            except (TypeError, ValueError):
                continue

        retrieval_score = sum(scores) / len(scores) if scores else 0.0
        coverage = len(citations) / max(1, len(results))
        length_match = self._compute_length_match(
            answer_style,
            answer_length,
        )
        citation_diversity = self._compute_citation_diversity(
            results,
            citations,
        )

        breakdown["retrieval"] = max(0.0, min(1.0, retrieval_score))
        breakdown["title_overlap"] = max(0.0, min(1.0, title_overlap))
        breakdown["platform_coherence"] = max(
            0.0, min(1.0, platform_coherence)
        )
        breakdown["coverage"] = max(0.0, min(1.0, coverage))
        breakdown["length_match"] = max(0.0, min(1.0, length_match))
        breakdown["citation_diversity"] = max(
            0.0, min(1.0, citation_diversity)
        )

        blended = 0.0

        for key, weight in _CONFIDENCE_WEIGHTS.items():
            blended += float(weight) * breakdown.get(key, 0.0)

        if not llm_used:
            blended *= 0.90

        if generated_confidence > 0.0:
            blended = max(blended, generated_confidence * 0.5)

        blended = max(0.0, min(1.0, blended))

        return blended, breakdown

    def _compute_length_match(
        self,
        answer_style: str,
        answer_length: int,
    ) -> float:
        if answer_length <= 0:
            return 0.0

        style = str(answer_style or "").strip().lower()

        if style == _ANSWER_STYLE_SHORT:
            short_max, short_soft = _LENGTH_MATCH_SHORT_LIMITS

            if answer_length <= short_max:
                return 1.0

            if answer_length <= short_soft:
                return 0.7

            return 0.4

        if style == _ANSWER_STYLE_DETAILED:
            full, mid, low = _LENGTH_MATCH_DETAILED_LIMITS

            if answer_length >= full:
                return 1.0

            if answer_length >= mid:
                return 0.75

            if answer_length >= low:
                return 0.5

            return 0.25

        if style == _ANSWER_STYLE_EXHAUSTIVE:
            full, mid, low = _LENGTH_MATCH_EXHAUSTIVE_LIMITS

            if answer_length >= full:
                return 1.0

            if answer_length >= mid:
                return 0.75

            if answer_length >= low:
                return 0.5

            return 0.25

        standard_full, standard_low = _LENGTH_MATCH_STANDARD_LIMITS

        if answer_length >= standard_full:
            return 1.0

        if answer_length >= standard_low:
            return 0.8

        return 0.5

    def _compute_citation_diversity(
        self,
        results: list[VectorSearchResult],
        citations: list[RAGCitation],
    ) -> float:
        if not results or not citations:
            return 0.0

        cited_ids = {
            str(citation.source_id)
            for citation in citations
            if citation.source_id
        }

        if not cited_ids:
            return 0.0

        platforms: set[str] = set()
        source_types: set[str] = set()

        for result in results:
            document = result.document

            if document.source_id not in cited_ids:
                continue

            platform = str(document.platform or "").strip().lower()

            if platform:
                platforms.add(platform)

            source_type = str(document.source_type or "").strip().lower()

            if source_type:
                source_types.add(source_type)

        platform_score = (
            min(1.0, len(platforms) / 3.0) if platforms else 0.0
        )
        type_score = (
            min(1.0, len(source_types) / 2.0) if source_types else 0.0
        )

        combined = 0.6 * platform_score + 0.4 * type_score

        return max(0.0, min(1.0, combined))

    def _resolve_answer_style(
        self,
        topic_profile: TopicProfileSchema | None,
    ) -> str:
        if topic_profile is None:
            return _ANSWER_STYLE_STANDARD

        try:
            raw = getattr(topic_profile, "answer_style", None)
        except Exception:
            return _ANSWER_STYLE_STANDARD

        if not isinstance(raw, str):
            return _ANSWER_STYLE_STANDARD

        text = raw.strip().lower()

        if text in _VALID_ANSWER_STYLES:
            return text

        return _ANSWER_STYLE_STANDARD

    def _read_threshold_diagnostics(self) -> dict[str, Any]:
        try:
            getter = getattr(
                self._vector_store, "last_search_diagnostics", None
            )
        except Exception:
            return {}

        if not isinstance(getter, dict):
            return {}

        return dict(getter)

    def _read_threshold_floor(self) -> float | None:
        try:
            config = getattr(self._vector_store, "min_score_config", None)
        except Exception:
            return None

        if not isinstance(config, dict):
            return None

        try:
            floor = float(config.get("minimum", 0.0))
        except (TypeError, ValueError):
            return None

        return floor

    def _classify_retrieval_path(
        self,
        strategy: str,
        threshold_diagnostics: dict[str, Any],
    ) -> str:
        normalized = str(strategy or "").strip().lower()

        operation = ""
        token_count = 0
        effective_gate = 0
        used_chunks = False

        if isinstance(threshold_diagnostics, dict):
            operation = str(
                threshold_diagnostics.get("operation", "") or ""
            ).strip().lower()

            try:
                token_count = int(
                    threshold_diagnostics.get("chunk_gate_token_count", 0)
                    or 0
                )
            except (TypeError, ValueError):
                token_count = 0

            try:
                effective_gate = int(
                    threshold_diagnostics.get("chunk_gate_effective", 0)
                    or 0
                )
            except (TypeError, ValueError):
                effective_gate = 0

            used_chunks = bool(
                threshold_diagnostics.get("chunk_gate_used_chunks", False)
            )

        if operation == "search_chunks":
            return "chunk"

        if operation == "hybrid_search":
            if not used_chunks and effective_gate > 0 and token_count < effective_gate:
                return "parent_gate_skipped"

            return "parent"

        if operation == "search":
            return "embedding"

        if operation == "keyword_search":
            return "keyword"

        if normalized == "hybrid_v2":
            return "chunk_or_parent"

        if normalized == "hybrid_all_runs":
            return "all_runs"

        if normalized == "empty_store":
            return "empty_store"

        if normalized == "conversation_memory":
            return "conversation"

        if normalized in {"no_match", "no_match_after_rerank"}:
            return "no_match"

        if normalized == "engine_error":
            return "engine_error"

        return normalized or "unknown"

    def _best_snippet(self, text: str) -> str:
        cleaned = clean_text(text).replace("\n", " ")

        if not cleaned:
            return ""

        if len(cleaned) <= 400:
            return cleaned

        return cleaned[:400].rstrip() + "..."

    def _detect_refusal(self, text: str) -> bool:
        if not text:
            return False

        cleaned = clean_text(text)

        if not cleaned:
            return False

        lowered = cleaned.lower()

        for phrase in _REFUSAL_PHRASES:
            if phrase in lowered:
                return True

        return False

    def _log_conversation_turn(
        self,
        question: str,
        answer: str,
        citations: list[RAGCitation],
        fallback: bool,
        confidence: float,
        strategy: str,
    ) -> None:
        try:
            self._conversation.add_turn(
                role="user",
                content=question,
            )
        except Exception as exc:
            self._logger.warning(f"Failed to persist user turn: {exc}")
            return

        if not answer:
            return

        try:
            self._conversation.add_turn(
                role="assistant",
                content=answer,
                metadata={
                    "citations": [
                        {
                            "source_id": citation.source_id,
                            "title": citation.title,
                            "url": citation.url,
                            "score": citation.score,
                        }
                        for citation in citations
                    ],
                    "fallback": bool(fallback),
                    "confidence": float(confidence),
                    "strategy": str(strategy),
                },
            )
        except Exception as exc:
            self._logger.warning(
                f"Failed to persist assistant turn: {exc}"
            )