from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import Field

from core.config import get_data_directory
from core.models import CoreModel, RankedSource, Source
from utils.logger import get_logger
from utils.text import clean_text, truncate_text


class VectorDocument(CoreModel):
    doc_id: str
    source_id: str
    run_id: str | None = None
    title: str
    text: str
    url: str | None = None
    platform: str | None = None
    source_type: str | None = None
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class VectorSearchResult(CoreModel):
    document: VectorDocument
    score: float


class MemoryVectorEntry(CoreModel):
    entry_id: str
    memory_type: str = "episodic"
    key: str
    content: str
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class VectorStore:
    def __init__(
        self,
        path: str | Path | None = None,
        min_score: float = 0.15,
    ) -> None:
        self._path = Path(path) if path is not None else get_data_directory() / "vector_store.json"
        self._min_score = max(0.0, min(1.0, min_score))
        self._logger = get_logger("rag.vector_store")
        self._documents: dict[str, VectorDocument] = {}
        self._memory_entries: dict[str, MemoryVectorEntry] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        self._documents = {}
        self._memory_entries = {}

        if not self._path.exists():
            self._loaded = True
            return

        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            items = raw.get("documents", []) if isinstance(raw, dict) else []
            if isinstance(items, list):
                for item in items:
                    try:
                        document = VectorDocument.model_validate(item)
                        if document.embedding:
                            self._documents[document.doc_id] = document
                    except Exception:
                        continue

            memory_items = raw.get("memory_entries", []) if isinstance(raw, dict) else []
            if isinstance(memory_items, list):
                for item in memory_items:
                    try:
                        entry = MemoryVectorEntry.model_validate(item)
                        if entry.embedding:
                            self._memory_entries[entry.entry_id] = entry
                    except Exception:
                        continue
        except Exception as exc:
            self._logger.warning(f"Failed to load vector store: {exc}")

        self._loaded = True

    def save(self) -> Path:
        self.load()
        self._path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "document_count": len(self._documents),
            "memory_count": len(self._memory_entries),
            "documents": [
                document.model_dump(mode="json")
                for document in self._documents.values()
            ],
            "memory_entries": [
                entry.model_dump(mode="json")
                for entry in self._memory_entries.values()
            ],
        }

        temp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temp_path, self._path)
        return self._path

    def add_document(self, document: VectorDocument, save: bool = True) -> None:
        self.load()
        if not document.embedding:
            return
        self._documents[document.doc_id] = document
        if save:
            self.save()

    def add_documents(self, documents: list[VectorDocument], save: bool = True) -> int:
        self.load()
        count = 0
        for document in documents:
            if not document.embedding:
                continue
            self._documents[document.doc_id] = document
            count += 1
        if save:
            self.save()
        return count

    def add_source(
        self,
        source: Source,
        embedding: list[float],
        run_id: str | None = None,
        save: bool = True,
    ) -> VectorDocument:
        self.load()
        document = self.document_from_source(
            source=source,
            embedding=embedding,
            run_id=run_id,
        )
        self.add_document(document, save=save)
        return document

    def add_ranked_sources(
        self,
        ranked_sources: list[Any],
        embeddings: dict[str, list[float]],
        run_id: str | None = None,
        save: bool = True,
    ) -> int:
        self.load()
        documents: list[VectorDocument] = []

        for ranked in ranked_sources:
            if not isinstance(ranked, RankedSource):
                continue
            source = ranked.source
            embedding = embeddings.get(source.source_id)
            if not embedding:
                continue
            document = self.document_from_source(
                source=source,
                embedding=embedding,
                run_id=run_id,
                metadata={
                    "rank": ranked.rank,
                    "score": ranked.score,
                    "confidence": ranked.confidence,
                    "reason": ranked.reason,
                },
            )
            documents.append(document)

        return self.add_documents(documents, save=save)

    def add_memory_entry(
        self,
        entry_id: str,
        key: str,
        content: str,
        embedding: list[float],
        memory_type: str = "episodic",
        metadata: dict[str, Any] | None = None,
        save: bool = True,
    ) -> MemoryVectorEntry:
        self.load()

        entry = MemoryVectorEntry(
            entry_id=entry_id,
            memory_type=memory_type,
            key=key,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
        )

        if embedding:
            self._memory_entries[entry_id] = entry

        if save:
            self.save()

        return entry

    def search_memory(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        memory_type: str | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        self.load()

        if not query_embedding:
            return []

        threshold = self._min_score if min_score is None else max(0.0, min(1.0, min_score))
        results: list[tuple[float, MemoryVectorEntry]] = []

        for entry in self._memory_entries.values():
            if memory_type and entry.memory_type != memory_type:
                continue
            if not entry.embedding:
                continue

            score = self._cosine_similarity(query_embedding, entry.embedding)
            if score < threshold:
                continue

            results.append((score, entry))

        results.sort(key=lambda x: x[0], reverse=True)
        top_results = results[:max(1, top_k)]

        return [
            {
                "entry_id": entry.entry_id,
                "memory_type": entry.memory_type,
                "key": entry.key,
                "content": entry.content,
                "metadata": entry.metadata,
                "relevance_score": round(score, 6),
            }
            for score, entry in top_results
        ]

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        run_id: str | None = None,
        min_score: float | None = None,
    ) -> list[VectorSearchResult]:
        self.load()

        if not query_embedding:
            return []

        threshold = self._min_score if min_score is None else max(0.0, min(1.0, min_score))
        results: list[VectorSearchResult] = []

        for document in self._documents.values():
            if run_id is not None and document.run_id != run_id:
                continue

            score = self._cosine_similarity(query_embedding, document.embedding)
            if score < threshold:
                continue

            results.append(
                VectorSearchResult(
                    document=document,
                    score=round(score, 6),
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[: max(1, top_k)]

    def list_documents(self, run_id: str | None = None) -> list[VectorDocument]:
        self.load()
        documents = list(self._documents.values())
        if run_id is not None:
            documents = [document for document in documents if document.run_id == run_id]
        return documents

    def list_memory_entries(self, memory_type: str | None = None) -> list[MemoryVectorEntry]:
        self.load()
        entries = list(self._memory_entries.values())
        if memory_type:
            entries = [e for e in entries if e.memory_type == memory_type]
        return entries

    def delete_run(self, run_id: str, save: bool = True) -> int:
        self.load()
        to_delete = [
            doc_id
            for doc_id, document in self._documents.items()
            if document.run_id == run_id
        ]
        for doc_id in to_delete:
            self._documents.pop(doc_id, None)
        if save:
            self.save()
        return len(to_delete)

    def delete_memory_entry(self, entry_id: str, save: bool = True) -> bool:
        self.load()
        if entry_id in self._memory_entries:
            del self._memory_entries[entry_id]
            if save:
                self.save()
            return True
        return False

    def clear(self, save: bool = True) -> None:
        self._documents = {}
        self._memory_entries = {}
        self._loaded = True
        if save:
            self.save()

    def document_from_source(
        self,
        source: Source,
        embedding: list[float],
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VectorDocument:
        document_metadata = dict(source.metadata or {})
        if metadata:
            document_metadata.update(metadata)

        return VectorDocument(
            doc_id=self._doc_id(source.source_id, run_id),
            source_id=source.source_id,
            run_id=run_id,
            title=source.title,
            text=self._source_text(source),
            url=source.url,
            platform=self._enum_value(source.platform),
            source_type=self._enum_value(source.source_type),
            embedding=[float(value) for value in embedding],
            metadata=document_metadata,
        )

    def _source_text(self, source: Source) -> str:
        parts: list[str] = []
        if source.title:
            parts.append(source.title)
        if source.abstract:
            parts.append(source.abstract)
        metadata = source.metadata if isinstance(source.metadata, dict) else {}
        for key in ("description", "summary", "readme", "topics", "tags", "categories"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                parts.append(value)
            elif isinstance(value, list):
                parts.extend(str(item) for item in value if item)
        return truncate_text(
            clean_text(" ".join(parts)),
            max_length=4000,
            suffix="",
        )

    def _doc_id(self, source_id: str, run_id: str | None) -> str:
        if run_id:
            return f"{run_id}:{source_id}"
        return source_id

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0

        length = min(len(left), len(right))
        if length <= 0:
            return 0.0

        a = left[:length]
        b = right[:length]

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        if norm_a <= 0 or norm_b <= 0:
            return 0.0

        return max(0.0, min(1.0, dot / (norm_a * norm_b)))

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value))