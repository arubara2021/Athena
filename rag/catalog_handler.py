from __future__ import annotations

from typing import Any

from rag.generator import Generator
from rag.schemas import ChatCitationSchema, FallbackStepSchema
from utils.logger import get_logger
from utils.text import clean_text


class CatalogHandler:
    def __init__(
        self,
        vector_store: Any,
        generator: Generator | None = None,
        top_titles: int = 25,
    ) -> None:
        self._vector_store = vector_store
        self._generator = generator or Generator()
        self._top_titles = max(5, int(top_titles))
        self._logger = get_logger("rag.catalog_handler")

    async def answer(
        self,
        question: str,
        session_id: str = "",
        run_id: str | None = None,
    ) -> dict[str, Any]:
        snapshot = self._build_snapshot(run_id=run_id)

        if snapshot is None or int(snapshot.get("total_documents", 0)) <= 0:
            return {
                "answer": (
                    "The local index is empty. Run a search or agent "
                    "with --rag to populate it first."
                ),
                "citations": [],
                "confidence": 0.0,
                "fallback": True,
                "strategy": "catalog_empty",
                "context_used": 0,
                "llm_used": False,
                "fallback_ladder": [
                    FallbackStepSchema(
                        stage="catalog",
                        reason="empty_store",
                        action="return_empty",
                        severity="warn",
                    )
                ],
                "diagnostics": {"catalog_snapshot": snapshot or {}},
            }

        generated = await self._generator.generate_catalog_summary(
            question=clean_text(question),
            snapshot=snapshot,
        )

        ladder = list(generated.fallback_ladder or [])

        if not generated.llm_used:
            ladder.append(
                FallbackStepSchema(
                    stage="catalog_generation",
                    reason="no_model_available",
                    action="use_heuristic_summary",
                    severity="info",
                )
            )

        return {
            "answer": generated.text,
            "citations": self._build_citations(snapshot),
            "confidence": generated.confidence,
            "fallback": False,
            "strategy": "catalog",
            "context_used": int(snapshot.get("total_documents", 0)),
            "llm_used": generated.llm_used,
            "fallback_ladder": ladder,
            "diagnostics": {
                "catalog_snapshot": self._compact_snapshot(snapshot),
            },
        }

    def _build_snapshot(self, run_id: str | None) -> dict[str, Any] | None:
        method = getattr(self._vector_store, "catalog_snapshot", None)

        if not callable(method):
            return None

        try:
            return method(run_id=run_id, top_titles=self._top_titles)
        except Exception as exc:
            self._logger.warning(f"Catalog snapshot failed: {exc}")
            return None

    def _compact_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            "total_documents": int(snapshot.get("total_documents", 0) or 0),
            "by_platform": dict(snapshot.get("by_platform", {}) or {}),
            "by_source_type": dict(snapshot.get("by_source_type", {}) or {}),
            "top_topics": list(snapshot.get("top_topics", []) or [])[:10],
            "date_range": dict(snapshot.get("date_range", {}) or {}),
            "title_count": len(list(snapshot.get("titles", []) or [])),
        }

    def _build_citations(
        self,
        snapshot: dict[str, Any],
    ) -> list[ChatCitationSchema]:
        citations: list[ChatCitationSchema] = []

        for entry in list(snapshot.get("titles", []) or [])[:12]:
            if not isinstance(entry, dict):
                continue

            citations.append(
                ChatCitationSchema(
                    source_id=str(entry.get("source_id", "") or ""),
                    title=str(entry.get("title", "") or ""),
                    url=entry.get("url"),
                    score=0.0,
                    snippet="",
                )
            )

        return citations