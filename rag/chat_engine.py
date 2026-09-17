from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from core import constants
from core.models import CoreModel
from rag.embedder import Embedder
from rag.vector_store import VectorSearchResult, VectorStore
from utils.logger import get_logger
from utils.text import clean_text, truncate_text


class RAGCitation(CoreModel):
    source_id: str
    title: str
    url: str | None = None
    score: float = 0.0


class RAGAnswer(CoreModel):
    question: str
    answer: str
    citations: list[RAGCitation] = Field(default_factory=list)
    context_used: int = 0
    llm_used: bool = False


class RAGChatEngine:
    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedder: Embedder | None = None,
        top_k: int = 5,
        max_context_chars: int = 6000,
    ) -> None:
        self._vector_store = vector_store or VectorStore()
        self._embedder = embedder or Embedder()
        self._owns_embedder = embedder is None
        self._top_k = max(1, top_k)
        self._max_context_chars = max(1000, max_context_chars)
        self._logger = get_logger("rag.chat_engine")

    async def __aenter__(self) -> RAGChatEngine:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        await self.close()
        return False

    async def close(self) -> None:
        if self._owns_embedder:
            await self._embedder.close()

    async def ask(
        self,
        question: str,
        run_id: str | None = None,
        top_k: int | None = None,
    ) -> RAGAnswer:
        cleaned_question = clean_text(question)

        if not cleaned_question:
            return RAGAnswer(
                question=question,
                answer="Please ask a non-empty question.",
                citations=[],
                context_used=0,
                llm_used=False,
            )

        try:
            query_embedding = await self._embedder.embed_text(cleaned_question)
            results = self._vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k or self._top_k,
                run_id=run_id,
            )
        except Exception as exc:
            self._logger.warning(f"RAG retrieval failed: {exc}")
            results = []

        if not results:
            return RAGAnswer(
                question=cleaned_question,
                answer="I could not find enough saved research context to answer this question. Run a research search first, then ask again.",
                citations=[],
                context_used=0,
                llm_used=False,
            )

        llm_answer = await self._answer_with_llm(cleaned_question, results)

        if llm_answer is not None:
            return llm_answer

        return self._answer_with_fallback(cleaned_question, results)

    async def _answer_with_llm(
        self,
        question: str,
        results: list[VectorSearchResult],
    ) -> RAGAnswer | None:
        try:
            from llm.guardrails import validate_llm_request, validate_llm_response
            from llm.parser import parse_json_object_response
            from llm.provider import LLMProviderManager
            from llm.router import get_strong_model_references
            from core.schemas import LLMMessageSchema, LLMRequestSchema

            model_references = get_strong_model_references(limit=3)
            if not model_references:
                return None

            context = self._build_context(results)

            expected_output = {
                "answer": "string",
                "used_source_ids": ["string"],
            }

            system_prompt = (
                "You are a research assistant. "
                "Answer only using the provided saved research context. "
                "If the context is not enough, say that the saved sources do not contain enough information. "
                "Return only valid JSON."
            )

            user_prompt = (
                f"Question:\n{question}\n\n"
                f"Saved research context:\n{context}\n\n"
                f"Expected JSON output:\n{json.dumps(expected_output, ensure_ascii=False, indent=2)}\n"
                "Return only valid JSON."
            )

            messages = [
                LLMMessageSchema(role="system", content=system_prompt),
                LLMMessageSchema(role="user", content=user_prompt),
            ]

            async with LLMProviderManager() as manager:
                for model_reference in model_references:
                    try:
                        request = LLMRequestSchema(
                            provider=model_reference.provider,
                            model=model_reference.model_id,
                            messages=messages,
                            temperature=constants.DEFAULT_TEMPERATURE,
                            max_tokens=1200,
                            response_format="json_object",
                        )

                        request = validate_llm_request(request)
                        response = await manager.complete(request)

                        if response is None:
                            continue

                        response = validate_llm_response(response)
                        payload = parse_json_object_response(response.content)

                        answer = clean_text(payload.get("answer", ""))
                        if not answer:
                            continue

                        used_ids = payload.get("used_source_ids", [])
                        citations = self._build_citations(results, used_ids)

                        if not citations:
                            citations = self._build_citations(results, None)

                        return RAGAnswer(
                            question=question,
                            answer=answer,
                            citations=citations,
                            context_used=len(results),
                            llm_used=True,
                        )
                    except Exception:
                        continue

            return None
        except Exception as exc:
            self._logger.warning(f"RAG LLM answer failed, using fallback: {exc}")
            return None

    def _answer_with_fallback(
        self,
        question: str,
        results: list[VectorSearchResult],
    ) -> RAGAnswer:
        top_results = results[: min(3, len(results))]
        lines: list[str] = []

        lines.append("Based on the saved research, the most relevant sources are:")

        for index, result in enumerate(top_results, start=1):
            document = result.document
            snippet = truncate_text(document.text, max_length=450, suffix="...")
            lines.append(f"{index}. {document.title}: {snippet}")

        lines.append("For a stronger answer, enable LLM chat after embeddings are saved.")

        return RAGAnswer(
            question=question,
            answer="\n\n".join(lines),
            citations=self._build_citations(results, None),
            context_used=len(results),
            llm_used=False,
        )

    def _build_context(self, results: list[VectorSearchResult]) -> str:
        parts: list[str] = []
        used_chars = 0

        for index, result in enumerate(results, start=1):
            document = result.document
            block = (
                f"Source {index}\n"
                f"source_id: {document.source_id}\n"
                f"title: {document.title}\n"
                f"url: {document.url or ''}\n"
                f"score: {result.score}\n"
                f"text: {truncate_text(document.text, max_length=1500, suffix='...')}\n"
            )

            if used_chars + len(block) > self._max_context_chars:
                break

            parts.append(block)
            used_chars += len(block)

        return "\n".join(parts)

    def _build_citations(
        self,
        results: list[VectorSearchResult],
        used_source_ids: Any,
    ) -> list[RAGCitation]:
        allowed: set[str] | None = None

        if isinstance(used_source_ids, list):
            allowed = {str(item) for item in used_source_ids if item}

        citations: list[RAGCitation] = []

        for result in results:
            document = result.document

            if allowed is not None and document.source_id not in allowed:
                continue

            citations.append(
                RAGCitation(
                    source_id=document.source_id,
                    title=document.title,
                    url=document.url,
                    score=result.score,
                )
            )

            if len(citations) >= self._top_k:
                break

        return citations