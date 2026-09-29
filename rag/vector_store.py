from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import struct
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import Field

from core.config import get_data_directory
from core.models import CoreModel, RankedSource, Source
from utils.logger import get_logger
from utils.text import clean_text

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "what", "how", "why",
    "when", "where", "which", "who", "whom", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "do", "does", "did",
    "doing", "have", "has", "had", "having", "can", "could", "should", "would",
    "may", "might", "must", "will", "shall", "not", "no", "yes", "about",
    "into", "over", "under", "between", "from", "as", "at", "by", "if",
    "then", "than", "too", "very", "just", "also",
    "tell", "tells", "telling", "told",
    "show", "shows", "showing", "showed", "shown",
    "give", "gives", "giving", "gave", "given",
    "know", "knows", "knowing", "knew", "known",
    "explain", "explains", "explaining", "explained",
    "describe", "describes", "describing", "described",
    "detail", "details", "detailed", "detailing",
    "please", "kindly",
    "need", "needs", "needed", "needing",
    "want", "wants", "wanted", "wanting",
    "like", "likes", "liked", "liking",
    "get", "gets", "getting", "got", "gotten",
    "make", "makes", "making", "made",
    "take", "takes", "taking", "took", "taken",
    "say", "says", "saying", "said",
    "let", "lets", "letting",
    "would", "should", "could",
    "specifically", "particularly", "actually",
}

_INTERROGATIVES = frozenset({
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "whats", "hows", "whys",
})

_IMPERATIVE_VERBS = frozenset({
    "tell", "tells", "telling", "told",
    "show", "shows", "showing", "showed", "shown",
    "give", "gives", "giving", "gave", "given",
    "know", "knows", "knowing", "knew", "known",
    "explain", "explains", "explaining", "explained",
    "describe", "describes", "describing", "described",
    "detail", "details", "detailed", "detailing",
    "make", "makes", "making", "made",
    "take", "takes", "taking", "took", "taken",
    "say", "says", "saying", "said",
    "let", "lets", "letting",
    "please", "kindly",
    "need", "needs", "needed", "needing",
    "want", "wants", "wanted", "wanting",
    "like", "likes", "liked", "liking",
    "get", "gets", "getting", "got", "gotten",
})

_FTS_EXCLUDED = _STOPWORDS | _INTERROGATIVES | _IMPERATIVE_VERBS

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_FTS_CACHE_MAX = 256
_EMBEDDING_CACHE_MAX = 512
_SCHEMA_CACHE_KEY = "schema_fingerprint"

_DOMAIN_ALIAS_CACHE: list[tuple[str, str]] | None = None


def _load_domain_aliases() -> list[tuple[str, str]]:
    global _DOMAIN_ALIAS_CACHE

    if _DOMAIN_ALIAS_CACHE is not None:
        return _DOMAIN_ALIAS_CACHE

    aliases: list[tuple[str, str]] = []

    try:
        import yaml

        config_path = (
            Path(__file__).resolve().parent.parent
            / "configs"
            / "domains.yaml"
        )

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}

            domains = raw.get("domains", {})

            if isinstance(domains, dict):
                for domain_name, config in domains.items():
                    if not isinstance(config, dict):
                        continue

                    canonical = str(domain_name).strip().lower()

                    if not canonical:
                        continue

                    aliases.append((canonical, canonical))

                    label = str(config.get("label", "")).strip().lower()

                    if label:
                        aliases.append((label, canonical))

                    for alias in config.get("aliases", []) or []:
                        text = str(alias).strip().lower()

                        if text:
                            aliases.append((text, canonical))
    except Exception:
        pass

    seen: set[str] = set()
    unique: list[tuple[str, str]] = []

    for alias, canonical in aliases:
        if alias and alias not in seen:
            seen.add(alias)
            unique.append((alias, canonical))

    unique.sort(key=lambda item: -len(item[0]))

    _DOMAIN_ALIAS_CACHE = unique
    return unique


_PRAGMA_STATEMENTS = (
    "PRAGMA journal_mode = WAL;",
    "PRAGMA synchronous = NORMAL;",
    "PRAGMA temp_store = MEMORY;",
    "PRAGMA mmap_size = 268435456;",
)

_DEFAULT_CHUNK_TOKENS = 220
_MIN_CHUNK_TOKENS = 120
_MAX_CHUNK_TOKENS = 600

_DEFAULT_CHUNK_GATE_TOKENS = 3
_DEFAULT_CHUNK_GATE_MIN_TOKENS = 2
_DEFAULT_CHUNK_GATE_MAX_TOKENS = 20
_DEFAULT_CHUNK_GATE_ADAPTIVE = True

_CHUNK_GATE_INTERROGATIVE_MARKERS = frozenset({
    "what", "which", "who", "whom", "whose", "when", "where",
    "why", "how", "compare", "comparison", "versus", "vs",
    "explain", "explanation", "difference", "differ",
    "describe", "define", "definition",
})

_DEFAULT_RAG_MIN_SCORE = 0.25
_DEFAULT_RAG_MIN_SCORE_SMALL = 0.30
_DEFAULT_RAG_MIN_SCORE_MINIMUM = 0.05
_DEFAULT_SMALL_CORPUS_CUTOFF = 100
_DEFAULT_MEDIUM_CORPUS_CUTOFF = 1000

_RAG_CONFIG_KEYS = (
    ("rag_min_score_default", "default"),
    ("rag_min_score_small_corpus", "small_corpus"),
    ("rag_min_score_minimum", "minimum"),
    ("rag_small_corpus_cutoff", "small_corpus_cutoff"),
    ("rag_medium_corpus_cutoff", "medium_corpus_cutoff"),
)

_RAG_THRESHOLD_CONFIG_CACHE: dict[str, float] | None = None
_CHUNK_GATE_CONFIG_CACHE: dict[str, Any] | None = None

_SCHEMA = """

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS vector_documents (
    doc_id       TEXT PRIMARY KEY,
    source_id    TEXT,
    run_id       TEXT,
    title        TEXT NOT NULL DEFAULT '',
    text         TEXT NOT NULL DEFAULT '',
    url          TEXT,
    platform     TEXT,
    source_type  TEXT,
    domain       TEXT,
    topic        TEXT,
    metadata     TEXT NOT NULL DEFAULT '{}',
    embedding    BLOB,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_vdoc_run_id
    ON vector_documents(run_id);
CREATE INDEX IF NOT EXISTS idx_vdoc_source_id
    ON vector_documents(source_id);
CREATE INDEX IF NOT EXISTS idx_vdoc_created_at
    ON vector_documents(created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS vector_documents_fts
USING fts5(
    doc_id UNINDEXED,
    title,
    text,
    metadata,
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS vector_document_chunks (
    chunk_id      TEXT PRIMARY KEY,
    parent_doc_id TEXT NOT NULL,
    chunk_index   INTEGER NOT NULL,
    text          TEXT NOT NULL DEFAULT '',
    embedding     BLOB,
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_vchunk_parent
    ON vector_document_chunks(parent_doc_id);
CREATE INDEX IF NOT EXISTS idx_vchunk_created
    ON vector_document_chunks(created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS vector_chunks_fts
USING fts5(
    chunk_id UNINDEXED,
    parent_doc_id UNINDEXED,
    text,
    tokenize='porter unicode61'
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL DEFAULT ''
);
"""

_SCHEMA_FINGERPRINT = hashlib.sha256(_SCHEMA.encode("utf-8")).hexdigest()[:32]


def _load_rag_threshold_config() -> dict[str, float]:
    global _RAG_THRESHOLD_CONFIG_CACHE

    if _RAG_THRESHOLD_CONFIG_CACHE is not None:
        return dict(_RAG_THRESHOLD_CONFIG_CACHE)

    config: dict[str, float] = {
        "default": float(_DEFAULT_RAG_MIN_SCORE),
        "small_corpus": float(_DEFAULT_RAG_MIN_SCORE_SMALL),
        "minimum": float(_DEFAULT_RAG_MIN_SCORE_MINIMUM),
        "small_corpus_cutoff": float(_DEFAULT_SMALL_CORPUS_CUTOFF),
        "medium_corpus_cutoff": float(_DEFAULT_MEDIUM_CORPUS_CUTOFF),
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
                section = raw.get("search", {})

                if isinstance(section, dict):
                    for yaml_key, config_key in _RAG_CONFIG_KEYS:
                        value = section.get(yaml_key)

                        if value is None:
                            continue

                        try:
                            config[config_key] = float(value)
                        except (TypeError, ValueError):
                            continue
    except Exception:
        pass

    default_value = max(0.0, min(1.0, float(config["default"])))
    small_value = max(0.0, min(1.0, float(config["small_corpus"])))
    minimum_value = max(0.0, min(1.0, float(config["minimum"])))

    if small_value < default_value:
        small_value = default_value

    if minimum_value > default_value:
        minimum_value = default_value

    small_cutoff = max(1, int(config["small_corpus_cutoff"]))
    medium_cutoff = max(small_cutoff, int(config["medium_corpus_cutoff"]))

    resolved = {
        "default": default_value,
        "small_corpus": small_value,
        "minimum": minimum_value,
        "small_corpus_cutoff": float(small_cutoff),
        "medium_corpus_cutoff": float(medium_cutoff),
    }

    _RAG_THRESHOLD_CONFIG_CACHE = resolved
    return dict(resolved)

def _load_chunk_gate_config() -> dict[str, Any]:
    global _CHUNK_GATE_CONFIG_CACHE

    if _CHUNK_GATE_CONFIG_CACHE is not None:
        return dict(_CHUNK_GATE_CONFIG_CACHE)

    resolved: dict[str, Any] = {
        "threshold": int(_DEFAULT_CHUNK_GATE_TOKENS),
        "min_tokens": int(_DEFAULT_CHUNK_GATE_MIN_TOKENS),
        "max_tokens": int(_DEFAULT_CHUNK_GATE_MAX_TOKENS),
        "adaptive": bool(_DEFAULT_CHUNK_GATE_ADAPTIVE),
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
                section = raw.get("search", {})

                if isinstance(section, dict):
                    value = section.get("rag_chunk_query_token_threshold")

                    if value is not None:
                        try:
                            resolved["threshold"] = int(value)
                        except (TypeError, ValueError):
                            pass

                    value = section.get("rag_chunk_gate_min_tokens")

                    if value is not None:
                        try:
                            resolved["min_tokens"] = int(value)
                        except (TypeError, ValueError):
                            pass

                    value = section.get("rag_chunk_gate_max_tokens")

                    if value is not None:
                        try:
                            resolved["max_tokens"] = int(value)
                        except (TypeError, ValueError):
                            pass

                    value = section.get("rag_chunk_gate_adaptive")

                    if value is not None:
                        resolved["adaptive"] = bool(value)
    except Exception:
        pass

    min_tokens = max(1, int(resolved["min_tokens"]))
    max_tokens = max(min_tokens, int(resolved["max_tokens"]))
    threshold = max(
        min_tokens,
        min(max_tokens, int(resolved["threshold"])),
    )

    resolved["min_tokens"] = min_tokens
    resolved["max_tokens"] = max_tokens
    resolved["threshold"] = threshold
    resolved["adaptive"] = bool(resolved["adaptive"])

    _CHUNK_GATE_CONFIG_CACHE = resolved
    return dict(resolved)


class VectorDocument(CoreModel):
    doc_id: str
    source_id: str
    run_id: str | None = None
    title: str
    text: str
    url: str | None = None
    platform: str | None = None
    source_type: str | None = None
    domain: str | None = None
    topic: str | None = None
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


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
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class VectorStore:
    def __init__(
        self,
        path: str | Path | None = None,
        min_score: float | None = None,
        max_documents: int = 10000,
    ) -> None:
        self._path = (
            Path(path)
            if path is not None
            else get_data_directory() / "vector_store.db"
        )
        self._legacy_json_path = self._path.with_suffix(".json")
        self._migrated_marker_path = Path(
            str(self._legacy_json_path) + ".migrated"
        )

        threshold_config = _load_rag_threshold_config()

        self._explicit_min_score = (
            max(0.0, min(1.0, float(min_score)))
            if min_score is not None
            else None
        )
        self._min_score_default = float(threshold_config["default"])
        self._min_score_small_corpus = float(threshold_config["small_corpus"])
        self._min_score_minimum = float(threshold_config["minimum"])
        self._small_corpus_cutoff = int(threshold_config["small_corpus_cutoff"])
        self._medium_corpus_cutoff = int(threshold_config["medium_corpus_cutoff"])

        self._chunk_gate_config = _load_chunk_gate_config()

        self._max_documents = max(1, int(max_documents or 10000))
        self._logger = get_logger("rag.vector_store")
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None
        self._loaded = False
        self._closed = False
        self._migration_attempted = False
        self._fts_cache: dict[str, str] = {}
        self._embedding_cache: dict[str, list[float]] = {}
        self._schema_verified = False
        self._last_search_diagnostics: dict[str, Any] = {}
        self._topic_extractor: Any = None
        self._domain_detector: Any = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def last_search_diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._last_search_diagnostics)

    @property
    def min_score_config(self) -> dict[str, float]:
        return {
            "explicit": (
                float(self._explicit_min_score)
                if self._explicit_min_score is not None
                else -1.0
            ),
            "default": self._min_score_default,
            "small_corpus": self._min_score_small_corpus,
            "minimum": self._min_score_minimum,
            "small_corpus_cutoff": float(self._small_corpus_cutoff),
            "medium_corpus_cutoff": float(self._medium_corpus_cutoff),
        }
        
    @property
    def chunk_query_token_threshold(self) -> int:
        return int(
            self._chunk_gate_config.get(
                "threshold",
                _DEFAULT_CHUNK_GATE_TOKENS,
            )
        )

    @property
    def chunk_gate_config(self) -> dict[str, Any]:
        return dict(self._chunk_gate_config)

    def exists(self) -> bool:
        try:
            return self._path.exists() and self._path.stat().st_size > 0
        except Exception:
            return False

    def is_empty(self) -> bool:
        try:
            return self.count() == 0
        except Exception:
            return True

    def close(self) -> None:
        with self._lock:
            connection = self._connection
            self._connection = None
            self._loaded = False

            if connection is None:
                self._closed = True
                return

            try:
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass

            try:
                connection.commit()
            except Exception:
                pass

            try:
                connection.close()
            except Exception as exc:
                self._logger.warning(f"VectorStore close raised: {exc}")

            self._closed = True

    def aclose(self) -> None:
        self.close()

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return

            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(
                str(self._path),
                timeout=30.0,
                check_same_thread=False,
            )
            self._connection.row_factory = sqlite3.Row

            self._apply_pragmas()

            stored = self._read_schema_fingerprint()

            if stored == _SCHEMA_FINGERPRINT and self._quick_schema_check():
                self._schema_verified = True
            else:
                self._connection.executescript(_SCHEMA)
                self._write_schema_fingerprint(_SCHEMA_FINGERPRINT)
                self._connection.commit()
                self._schema_verified = True

            self._ensure_catalog_columns()

            self._loaded = True
            self._closed = False

            existing = self._count_sync()
            self._logger.info(
                f"VectorStore loaded: path={self._path} "
                f"existing_docs={existing} "
                f"schema_cached={stored == _SCHEMA_FINGERPRINT}"
            )

            if not self._migration_attempted:
                self._migration_attempted = True
                self._migrate_legacy_json()

    def _apply_pragmas(self) -> None:
        if self._connection is None:
            return

        for statement in _PRAGMA_STATEMENTS:
            try:
                self._connection.execute(statement)
            except Exception:
                continue

    def _read_schema_fingerprint(self) -> str:
        if self._connection is None:
            return ""

        try:
            cursor = self._connection.execute(
                "SELECT value FROM meta WHERE key = ?",
                (_SCHEMA_CACHE_KEY,),
            )
            row = cursor.fetchone()
            return str(row["value"]) if row else ""
        except Exception:
            return ""

    def _write_schema_fingerprint(self, value: str) -> None:
        if self._connection is None:
            return

        try:
            self._connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                (_SCHEMA_CACHE_KEY, value),
            )
        except Exception as exc:
            self._logger.warning(f"Schema fingerprint write failed: {exc}")

    def _quick_schema_check(self) -> bool:
        if self._connection is None:
            return False

        required = (
            "vector_documents",
            "vector_document_chunks",
            "meta",
        )

        try:
            cursor = self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
            )
            names = {str(row["name"]) for row in cursor.fetchall()}
        except Exception:
            return False

        for table in required:
            if table not in names:
                return False

        return True
    def _ensure_catalog_columns(self) -> None:
        if self._connection is None:
            return

        self._ensure_column("vector_documents", "domain", "TEXT")
        self._ensure_column("vector_documents", "topic", "TEXT")

    def _ensure_column(
        self,
        table: str,
        column: str,
        declaration: str,
    ) -> None:
        if self._connection is None:
            return

        try:
            cursor = self._connection.execute(
                f"PRAGMA table_info({table})"
            )
            names = {str(row["name"]) for row in cursor.fetchall()}

            if column in names:
                return

            self._connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
            )
            self._connection.commit()
        except Exception as exc:
            self._logger.warning(
                f"Could not ensure column {table}.{column}: {exc}"
            )

    def _get_topic_extractor(self) -> Any:
        if self._topic_extractor is None:
            try:
                from rag.topic_analyzer import TopicAnalyzer

                self._topic_extractor = TopicAnalyzer()
            except Exception as exc:
                self._logger.warning(
                    f"TopicAnalyzer unavailable for indexing: {exc}"
                )
                self._topic_extractor = False

        return self._topic_extractor

    def _get_domain_detector(self) -> Any:
        if self._domain_detector is None:
            try:
                from search.query_intelligence import QueryIntelligence

                self._domain_detector = QueryIntelligence(use_llm=False)
            except Exception as exc:
                self._logger.warning(
                    f"QueryIntelligence unavailable for indexing: {exc}"
                )
                self._domain_detector = False

        return self._domain_detector

    def _classify_source_domain(
        self,
        source: Any,
    ) -> tuple[str | None, str | None]:
        title = clean_text(getattr(source, "title", "") or "")
        abstract = clean_text(getattr(source, "abstract", "") or "")
        combined = f"{title} {abstract}".strip()

        if not combined:
            return None, None

        lowered = combined.lower()

        domain: str | None = None
        topic: str | None = None

        matched_alias, matched_domain = self._match_registry_alias(lowered)

        if matched_alias:
            topic = matched_alias

        if matched_domain:
            domain = matched_domain

        if not domain:
            detector = self._get_domain_detector()

            if detector:
                try:
                    domains, _ = detector._detect_domains(
                        topic=combined,
                        goal="",
                        primary_concept=title or combined,
                        keywords=[],
                    )
                    if domains:
                        domain = str(domains[0]).strip().lower() or None
                except Exception:
                    pass

        if not domain:
            domain = "general"

        if not topic:
            topic = self._extract_short_topic(title)

        return domain, topic

    def _match_registry_alias(
        self,
        text: str,
    ) -> tuple[str | None, str | None]:
        if not text:
            return None, None

        aliases = _load_domain_aliases()

        for alias, canonical in aliases:
            if len(alias) < 3:
                continue

            try:
                pattern = (
                    r"(?:^|[^a-z0-9])"
                    + re.escape(alias)
                    + r"(?:[^a-z0-9]|$)"
                )

                if re.search(pattern, text):
                    return alias, canonical
            except Exception:
                continue

        return None, None

    def _extract_short_topic(self, title: str) -> str | None:
        if not title:
            return None

        try:
            from rag.topic_analyzer import _STOPWORDS as _TOPIC_STOPWORDS
        except Exception:
            return None

        head = re.split(r"[:(]", title, maxsplit=1)[0].strip()

        if not head:
            head = title.strip()

        head = re.split(r"\s+-\s+|\s+--\s+", head, maxsplit=1)[0].strip()

        if not head:
            return None

        words = re.findall(r"[^\W\d_]+|\d+", head, re.UNICODE)

        kept: list[str] = []

        for word in words:
            lowered = word.lower()

            if len(lowered) < 2:
                continue

            if lowered in _TOPIC_STOPWORDS:
                continue

            kept.append(lowered)

            if len(kept) >= 3:
                break

        if not kept:
            return None

        return " ".join(kept)

    def snapshot_documents(
        self,
        run_id: str | None = None,
        limit: int = 500,
    ) -> list[VectorDocument]:
        self._ensure_loaded()

        try:
            limit_value = max(1, int(limit))
        except (TypeError, ValueError):
            limit_value = 500

        with self._lock:
            if self._connection is None:
                return []

            try:
                if run_id is None:
                    cursor = self._connection.execute(
                        "SELECT * FROM vector_documents "
                        "ORDER BY created_at DESC LIMIT ?",
                        (limit_value,),
                    )
                else:
                    cursor = self._connection.execute(
                        "SELECT * FROM vector_documents "
                        "WHERE run_id = ? "
                        "ORDER BY created_at DESC LIMIT ?",
                        (run_id, limit_value),
                    )
                rows = cursor.fetchall()
            except Exception as exc:
                self._logger.warning(
                    f"snapshot_documents query failed: {exc}"
                )
                return []

        documents: list[VectorDocument] = []

        for row in rows:
            try:
                documents.append(
                    self._row_to_document(row, include_embedding=False)
                )
            except Exception as exc:
                self._logger.warning(
                    f"snapshot_documents row decode failed: {exc}"
                )
                continue

        return documents
    def save(self) -> None:
        if self._connection is not None:
            try:
                self._connection.commit()
            except Exception as exc:
                self._logger.warning(f"Vector store commit failed: {exc}")

    def migrate_from_json(self) -> int:
        self._ensure_loaded()
        return self._migrate_legacy_json(force=True)

    def build_document_text(
        self,
        source: Any,
        rank: Any = None,
        score: Any = None,
        confidence: Any = None,
    ) -> str:
        return self._document_text(
            source,
            rank=rank,
            score=score,
            confidence=confidence,
        )

    def resolve_chunk_tokens_public(
        self,
        chunk_tokens: int | None = None,
    ) -> int:
        return self._resolve_chunk_tokens(chunk_tokens)

    def chunk_embeddings_coverage(self, doc_id: str) -> float:
        self._ensure_loaded()

        normalized = str(doc_id or "").strip()

        if not normalized:
            return 0.0

        with self._lock:
            if self._connection is None:
                return 0.0

            try:
                cursor = self._connection.execute(
                    """
                    SELECT
                        COUNT(*) AS total,
                        SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END)
                            AS with_embedding
                    FROM vector_document_chunks
                    WHERE parent_doc_id = ?
                    """,
                    (normalized,),
                )
                row = cursor.fetchone()
            except Exception as exc:
                self._logger.warning(
                    f"chunk_embeddings_coverage query failed: {exc}"
                )
                return 0.0

        if row is None:
            return 0.0

        try:
            total = int(row["total"] or 0)
        except Exception:
            total = 0

        if total <= 0:
            return 0.0

        try:
            with_embedding = int(row["with_embedding"] or 0)
        except Exception:
            with_embedding = 0

        return max(0.0, min(1.0, with_embedding / total))

    def chunk_stats(self) -> dict[str, Any]:
        self._ensure_loaded()

        stats: dict[str, Any] = {
            "total_chunks": 0,
            "docs_with_chunks": 0,
            "avg_chunks_per_doc": 0.0,
            "docs_without_chunks": 0,
            "chunks_with_embedding": 0,
            "embedding_coverage": 0.0,
        }

        with self._lock:
            if self._connection is None:
                return stats

            try:
                cursor = self._connection.execute(
                    """
                    SELECT
                        COUNT(*) AS total_chunks,
                        SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END)
                            AS chunks_with_embedding
                    FROM vector_document_chunks
                    """
                )
                chunk_row = cursor.fetchone()
            except Exception as exc:
                self._logger.warning(f"chunk_stats chunk query failed: {exc}")
                return stats

            if chunk_row is not None:
                try:
                    stats["total_chunks"] = int(chunk_row["total_chunks"] or 0)
                except Exception:
                    stats["total_chunks"] = 0

                try:
                    stats["chunks_with_embedding"] = int(
                        chunk_row["chunks_with_embedding"] or 0
                    )
                except Exception:
                    stats["chunks_with_embedding"] = 0

            try:
                cursor = self._connection.execute(
                    """
                    SELECT COUNT(DISTINCT parent_doc_id) AS n
                    FROM vector_document_chunks
                    """
                )
                docs_row = cursor.fetchone()
            except Exception as exc:
                self._logger.warning(f"chunk_stats docs query failed: {exc}")
                docs_row = None

            if docs_row is not None:
                try:
                    stats["docs_with_chunks"] = int(docs_row["n"] or 0)
                except Exception:
                    stats["docs_with_chunks"] = 0

            try:
                cursor = self._connection.execute(
                    "SELECT COUNT(*) AS n FROM vector_documents"
                )
                total_docs_row = cursor.fetchone()
            except Exception:
                total_docs_row = None

            total_docs = 0

            if total_docs_row is not None:
                try:
                    total_docs = int(total_docs_row["n"] or 0)
                except Exception:
                    total_docs = 0

            stats["docs_without_chunks"] = max(
                0,
                total_docs - int(stats["docs_with_chunks"]),
            )

        docs_with_chunks = int(stats["docs_with_chunks"])

        if docs_with_chunks > 0:
            stats["avg_chunks_per_doc"] = round(
                stats["total_chunks"] / docs_with_chunks,
                2,
            )

        total_chunks = int(stats["total_chunks"])

        if total_chunks > 0:
            stats["embedding_coverage"] = round(
                stats["chunks_with_embedding"] / total_chunks,
                4,
            )

        return stats
    def catalog_snapshot(
        self,
        run_id: str | None = None,
        top_titles: int = 25,
    ) -> dict[str, Any]:
        self._ensure_loaded()

        snapshot: dict[str, Any] = {
            "total_documents": 0,
            "by_platform": {},
            "by_source_type": {},
            "by_domain": {},
            "by_topic": {},
            "by_run_id": {},
            "top_topics": [],
            "date_range": {"earliest": "", "latest": ""},
            "titles": [],
        }

        where_clause = "" if run_id is None else "WHERE run_id = ?"
        params: tuple[Any, ...] = () if run_id is None else (run_id,)

        with self._lock:
            if self._connection is None:
                return snapshot

            try:
                cursor = self._connection.execute(
                    f"SELECT COUNT(*) AS n FROM vector_documents {where_clause}",
                    params,
                )
                row = cursor.fetchone()
                snapshot["total_documents"] = int(row["n"] or 0) if row else 0
            except Exception as exc:
                self._logger.warning(f"Catalog count failed: {exc}")
                return snapshot

            try:
                cursor = self._connection.execute(
                    f"""
                    SELECT platform, COUNT(*) AS n
                    FROM vector_documents
                    {where_clause}
                    GROUP BY platform
                    """,
                    params,
                )
                snapshot["by_platform"] = {
                    str(r["platform"] or "unknown"): int(r["n"] or 0)
                    for r in cursor.fetchall()
                }
            except Exception:
                pass

            try:
                cursor = self._connection.execute(
                    f"""
                    SELECT source_type, COUNT(*) AS n
                    FROM vector_documents
                    {where_clause}
                    GROUP BY source_type
                    """,
                    params,
                )
                snapshot["by_source_type"] = {
                    str(r["source_type"] or "unknown"): int(r["n"] or 0)
                    for r in cursor.fetchall()
                }
            except Exception:
                pass

            try:
                cursor = self._connection.execute(
                    f"""
                    SELECT domain, COUNT(*) AS n
                    FROM vector_documents
                    {where_clause}
                    GROUP BY domain
                    """,
                    params,
                )
                snapshot["by_domain"] = {
                    str(r["domain"] or "unknown"): int(r["n"] or 0)
                    for r in cursor.fetchall()
                }
            except Exception:
                pass

            try:
                cursor = self._connection.execute(
                    f"""
                    SELECT topic, COUNT(*) AS n
                    FROM vector_documents
                    {where_clause}
                    GROUP BY topic
                    """,
                    params,
                )
                snapshot["by_topic"] = {
                    str(r["topic"] or "unknown"): int(r["n"] or 0)
                    for r in cursor.fetchall()
                }
            except Exception:
                pass

            try:
                cursor = self._connection.execute(
                    f"""
                    SELECT run_id, COUNT(*) AS n
                    FROM vector_documents
                    {where_clause}
                    GROUP BY run_id
                    """,
                    params,
                )
                snapshot["by_run_id"] = {
                    str(r["run_id"] or "unassigned"): int(r["n"] or 0)
                    for r in cursor.fetchall()
                }
            except Exception:
                pass

            try:
                cursor = self._connection.execute(
                    f"""
                    SELECT
                        MIN(created_at) AS earliest,
                        MAX(created_at) AS latest
                    FROM vector_documents
                    {where_clause}
                    """,
                    params,
                )
                row = cursor.fetchone()

                if row is not None:
                    snapshot["date_range"] = {
                        "earliest": str(row["earliest"] or ""),
                        "latest": str(row["latest"] or ""),
                    }
            except Exception:
                pass

            try:
                cursor = self._connection.execute(
                    f"""
                    SELECT title, source_id, url
                    FROM vector_documents
                    {where_clause}
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (*params, max(1, int(top_titles))),
                )
                snapshot["titles"] = [
                    {
                        "title": str(r["title"] or ""),
                        "source_id": str(r["source_id"] or ""),
                        "url": r["url"],
                    }
                    for r in cursor.fetchall()
                ]
            except Exception:
                pass

        snapshot["top_topics"] = self._compute_top_topics(
            by_topic=snapshot.get("by_topic", {}),
            limit=12,
        )

        return snapshot

    def _compute_top_topics(
        self,
        by_topic: dict[str, int],
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        if not isinstance(by_topic, dict) or not by_topic:
            return []

        try:
            resolved_limit = max(1, int(limit))
        except (TypeError, ValueError):
            resolved_limit = 12

        rows = sorted(
            by_topic.items(),
            key=lambda item: -int(item[1] or 0),
        )

        return [
            {"topic": str(topic), "frequency": int(count or 0)}
            for topic, count in rows[:resolved_limit]
        ]
    def add_document(self, document: VectorDocument) -> str:
        self._ensure_loaded()

        if not document.doc_id:
            document.doc_id = self._make_doc_id(document)

        with self._lock:
            self._insert_document(document)
            self._trim()

            if self._connection is not None:
                self._connection.commit()

        return document.doc_id

    def add_ranked_sources(
        self,
        ranked_sources: list[RankedSource],
        embeddings: list[list[float]],
        run_id: str | None = None,
    ) -> int:
        self._ensure_loaded()
        added = 0

        with self._lock:
            for index, ranked in enumerate(ranked_sources):
                source = getattr(ranked, "source", ranked)
                embedding = (
                    embeddings[index] if index < len(embeddings) else []
                )
                document = self._make_document(
                    source=source,
                    embedding=embedding,
                    run_id=run_id,
                    rank=getattr(ranked, "rank", index + 1),
                    score=getattr(ranked, "score", None),
                    confidence=getattr(ranked, "confidence", None),
                )
                self._insert_document(document)
                added += 1

            self._trim()

            if self._connection is not None:
                self._connection.commit()

        self._logger.info(
            f"VectorStore added {added} ranked sources (run_id={run_id})"
        )
        return added

    def add_sources(
        self,
        sources: list[Source],
        embeddings: list[list[float]],
        run_id: str | None = None,
    ) -> int:
        self._ensure_loaded()
        added = 0

        with self._lock:
            for index, source in enumerate(sources):
                embedding = (
                    embeddings[index] if index < len(embeddings) else []
                )
                document = self._make_document(
                    source=source,
                    embedding=embedding,
                    run_id=run_id,
                )
                self._insert_document(document)
                added += 1

            self._trim()

            if self._connection is not None:
                self._connection.commit()

        self._logger.info(
            f"VectorStore added {added} sources (run_id={run_id})"
        )
        return added

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        run_id: str | None = None,
        min_score: float | None = None,
    ) -> list[VectorSearchResult]:
        self._ensure_loaded()

        if not query_embedding:
            self._logger.info("VectorStore.search skipped: empty embedding")
            self._record_search_diagnostics(
                operation="search",
                status="skipped",
                reason="empty_embedding",
                threshold=None,
                threshold_source="not_applied",
            )
            return []

        rows = self._select_rows(run_id=run_id)

        if min_score is not None:
            threshold = max(0.0, min(1.0, float(min_score)))
            threshold_source = "caller"
        else:
            threshold, threshold_source = self._compute_adaptive_threshold(
                len(rows)
            )

        results: list[VectorSearchResult] = []

        for row in rows:
            try:
                document = self._row_to_document(row, include_embedding=True)
            except Exception as exc:
                self._logger.warning(f"Row decode failed during search: {exc}")
                continue

            score = self._cosine_similarity(
                query_embedding, document.embedding
            )

            if score < threshold:
                continue

            document.embedding = []
            results.append(VectorSearchResult(document=document, score=score))

        results.sort(key=lambda item: item.score, reverse=True)
        final = results[: max(1, int(top_k))]
        dropped = max(0, len(rows) - len(final))

        self._logger.info(
            f"VectorStore.search returned {len(final)} of {len(rows)} rows "
            f"(run_id={run_id}, threshold={threshold:.4f}, "
            f"source={threshold_source}, dropped={dropped})"
        )

        self._record_search_diagnostics(
            operation="search",
            status="ok",
            threshold=round(threshold, 4),
            threshold_source=threshold_source,
            input_count=len(rows),
            kept_count=len(final),
            dropped_count=dropped,
            run_id=run_id,
        )

        return final

    def keyword_search(
        self,
        query_text: str,
        top_k: int = 5,
        run_id: str | None = None,
        min_score: float | None = None,
    ) -> list[VectorSearchResult]:
        self._ensure_loaded()

        query = self._build_fts_query(query_text)

        if not query:
            self._logger.info(
                "VectorStore.keyword_search skipped: "
                "no meaningful tokens after stopword filtering"
            )
            self._record_search_diagnostics(
                operation="keyword_search",
                status="skipped",
                reason="no_meaningful_tokens",
                threshold=None,
                threshold_source="not_applied",
                run_id=run_id,
            )
            return []

        threshold = (
            0.0 if min_score is None else max(0.0, min(1.0, float(min_score)))
        )

        with self._lock:
            if self._connection is None:
                return []

            try:
                if run_id is None:
                    cursor = self._connection.execute(
                        """
                        SELECT d.*, bm25(vector_documents_fts) AS rank
                        FROM vector_documents_fts
                        JOIN vector_documents AS d
                            ON d.doc_id = vector_documents_fts.doc_id
                        WHERE vector_documents_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (query, max(1, int(top_k) * 3)),
                    )
                else:
                    cursor = self._connection.execute(
                        """
                        SELECT d.*, bm25(vector_documents_fts) AS rank
                        FROM vector_documents_fts
                        JOIN vector_documents AS d
                            ON d.doc_id = vector_documents_fts.doc_id
                        WHERE vector_documents_fts MATCH ?
                          AND d.run_id = ?
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (query, run_id, max(1, int(top_k) * 3)),
                    )
                rows = cursor.fetchall()
            except sqlite3.OperationalError as exc:
                self._logger.warning(
                    f"FTS5 search failed, falling back to lexical: {exc}"
                )
                return self._fallback_keyword_search(
                    query_text, top_k, run_id, threshold
                )

        results: list[VectorSearchResult] = []

        for row in rows:
            try:
                document = self._row_to_document(
                    row, include_embedding=False
                )
            except Exception as exc:
                self._logger.warning(
                    f"Row decode failed during keyword_search: {exc}"
                )
                continue

            raw_rank = row["rank"] if "rank" in row.keys() else 0.0
            score = self._normalize_bm25(raw_rank)

            if score < threshold:
                continue

            results.append(VectorSearchResult(document=document, score=score))

        results.sort(key=lambda item: item.score, reverse=True)
        final = results[: max(1, int(top_k))]

        self._logger.info(
            f"VectorStore.keyword_search returned {len(final)} of "
            f"{len(rows)} rows (query={query!r}, run_id={run_id}, "
            f"threshold={threshold:.4f})"
        )

        self._record_search_diagnostics(
            operation="keyword_search",
            status="ok",
            threshold=round(threshold, 4),
            threshold_source="caller_default" if min_score is not None else "bm25_default",
            input_count=len(rows),
            kept_count=len(final),
            dropped_count=max(0, len(rows) - len(final)),
            run_id=run_id,
            query=query,
        )

        return final

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float] | None = None,
        top_k: int = 5,
        run_id: str | None = None,
        min_score: float | None = None,
        embedding_weight: float = 0.55,
        keyword_weight: float = 0.45,
    ) -> list[VectorSearchResult]:
        self._ensure_loaded()
        limit = max(1, int(top_k))
        candidate_limit = max(limit * 4, 20)

        if min_score is not None:
            threshold = max(0.0, min(1.0, float(min_score)))
            threshold_source = "caller"
        else:
            corpus_size = self._count_sync()
            threshold, threshold_source = self._compute_adaptive_threshold(
                corpus_size
            )

        has_embedding = bool(query_embedding)
        keyword_hits: dict[str, float] = {}
        embedding_hits: dict[str, float] = {}

        for result in self.keyword_search(
            query_text,
            top_k=candidate_limit,
            run_id=run_id,
            min_score=0.0,
        ):
            keyword_hits[result.document.doc_id] = result.score

        if has_embedding:
            for result in self.search(
                query_embedding or [],
                top_k=candidate_limit,
                run_id=run_id,
                min_score=0.0,
            ):
                embedding_hits[result.document.doc_id] = result.score

        all_ids = set(keyword_hits) | set(embedding_hits)

        if not all_ids:
            self._logger.info(
                f"VectorStore.hybrid_search no candidates "
                f"(query={query_text!r}, run_id={run_id}, "
                f"has_embedding={has_embedding}, "
                f"threshold={threshold:.4f}, source={threshold_source})"
            )
            self._record_search_diagnostics(
                operation="hybrid_search",
                status="empty",
                threshold=round(threshold, 4),
                threshold_source=threshold_source,
                input_count=0,
                kept_count=0,
                dropped_count=0,
                keyword_hits=0,
                embedding_hits=0,
                has_embedding=has_embedding,
                run_id=run_id,
            )
            return []

        documents = self._fetch_documents_by_ids(all_ids)
        scored: list[VectorSearchResult] = []

        for doc_id in all_ids:
            document = documents.get(doc_id)

            if document is None:
                continue

            keyword_score = keyword_hits.get(doc_id, 0.0)
            embedding_score = embedding_hits.get(doc_id, 0.0)

            if has_embedding:
                combined = (
                    embedding_weight * embedding_score
                    + keyword_weight * keyword_score
                )
            else:
                combined = keyword_score

            if combined < threshold:
                continue

            scored.append(
                VectorSearchResult(document=document, score=combined)
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        final = scored[:limit]
        dropped = max(0, len(all_ids) - len(final))

        self._logger.info(
            f"VectorStore.hybrid_search returned {len(final)} of "
            f"{len(all_ids)} candidates "
            f"(keyword_hits={len(keyword_hits)}, "
            f"embedding_hits={len(embedding_hits)}, "
            f"has_embedding={has_embedding}, run_id={run_id}, "
            f"threshold={threshold:.4f}, source={threshold_source}, "
            f"dropped={dropped})"
        )

        self._record_search_diagnostics(
            operation="hybrid_search",
            status="ok",
            threshold=round(threshold, 4),
            threshold_source=threshold_source,
            input_count=len(all_ids),
            kept_count=len(final),
            dropped_count=dropped,
            keyword_hits=len(keyword_hits),
            embedding_hits=len(embedding_hits),
            has_embedding=has_embedding,
            run_id=run_id,
        )

        return final

    def hybrid_search_v2(
        self,
        query_text: str,
        query_embedding: list[float] | None = None,
        top_k: int = 5,
        run_id: str | None = None,
        min_score: float | None = None,
        chunk_token_threshold: int | None = None,
    ) -> list[VectorSearchResult]:
        self._ensure_loaded()

        tokens = self._tokenize(query_text)
        token_count = len(tokens)

        gate_config = self._chunk_gate_config
        min_gate = int(
            gate_config.get("min_tokens", _DEFAULT_CHUNK_GATE_MIN_TOKENS)
        )
        max_gate = int(
            gate_config.get("max_tokens", _DEFAULT_CHUNK_GATE_MAX_TOKENS)
        )
        base_gate = int(
            gate_config.get("threshold", _DEFAULT_CHUNK_GATE_TOKENS)
        )
        adaptive = bool(gate_config.get("adaptive", True))

        if chunk_token_threshold is not None:
            effective_gate = max(
                min_gate,
                min(max_gate, int(chunk_token_threshold)),
            )
            gate_source = "caller"
        else:
            effective_gate = max(min_gate, min(max_gate, base_gate))
            gate_source = "config"

            if adaptive and self._has_interrogative_marker(query_text):
                effective_gate = max(min_gate, effective_gate - 1)
                gate_source = "config_adaptive"

        use_chunks = token_count >= effective_gate

        gate_diagnostics: dict[str, Any] = {
            "chunk_gate_base": base_gate,
            "chunk_gate_effective": effective_gate,
            "chunk_gate_source": gate_source,
            "chunk_gate_token_count": token_count,
            "chunk_gate_adaptive": adaptive,
            "chunk_gate_min_tokens": min_gate,
            "chunk_gate_max_tokens": max_gate,
            "chunk_gate_used_chunks": use_chunks,
        }

        try:
            if use_chunks:
                chunk_results = self.search_chunks(
                    query_text=query_text,
                    query_embedding=query_embedding,
                    top_k=top_k,
                    run_id=run_id,
                    min_score=min_score,
                )

                if chunk_results:
                    self._logger.info(
                        f"VectorStore.hybrid_search_v2 chunk path: "
                        f"{len(chunk_results)} results "
                        f"(query_tokens={token_count}, "
                        f"gate={effective_gate}, source={gate_source})"
                    )
                    return chunk_results

            return self.hybrid_search(
                query_text=query_text,
                query_embedding=query_embedding,
                top_k=top_k,
                run_id=run_id,
                min_score=min_score,
            )
        finally:
            self._augment_last_diagnostics(gate_diagnostics)

    def add_document_with_chunks(
        self,
        document: VectorDocument,
        chunk_embeddings: list[list[float]] | None = None,
        chunk_tokens: int | None = None,
    ) -> dict[str, Any]:
        self._ensure_loaded()

        if not document.doc_id:
            document.doc_id = self._make_doc_id(document)

        resolved_chunk_tokens = self._resolve_chunk_tokens(chunk_tokens)
        chunks = self.split_into_chunks(document.text, resolved_chunk_tokens)

        stored_doc_id = ""
        stored_chunks = 0

        with self._lock:
            self._insert_document(document)
            stored_doc_id = document.doc_id

            if chunks:
                embeddings = list(chunk_embeddings or [])
                stored_chunks = self._insert_chunks(
                    document.doc_id,
                    chunks,
                    embeddings,
                )

            self._trim()

            if self._connection is not None:
                self._connection.commit()

        return {
            "doc_id": stored_doc_id,
            "chunks_stored": stored_chunks,
            "chunk_tokens": resolved_chunk_tokens,
        }

    def add_ranked_sources_with_chunks(
        self,
        ranked_sources: list[RankedSource],
        parent_embeddings: list[list[float]],
        chunk_embeddings: list[list[list[float]]] | None,
        run_id: str | None = None,
        chunk_tokens: int | None = None,
    ) -> int:
        self._ensure_loaded()
        added = 0

        with self._lock:
            for index, ranked in enumerate(ranked_sources):
                source = getattr(ranked, "source", ranked)
                parent_embedding = (
                    parent_embeddings[index]
                    if index < len(parent_embeddings)
                    else []
                )
                document = self._make_document(
                    source=source,
                    embedding=parent_embedding,
                    run_id=run_id,
                    rank=getattr(ranked, "rank", index + 1),
                    score=getattr(ranked, "score", None),
                    confidence=getattr(ranked, "confidence", None),
                )
                self._insert_document(document)

                if chunk_embeddings and index < len(chunk_embeddings):
                    resolved_chunk_tokens = self._resolve_chunk_tokens(
                        chunk_tokens
                    )
                    chunks = self.split_into_chunks(
                        document.text,
                        resolved_chunk_tokens,
                    )
                    self._insert_chunks(
                        document.doc_id,
                        chunks,
                        chunk_embeddings[index],
                    )

                added += 1

            self._trim()

            if self._connection is not None:
                self._connection.commit()

        self._logger.info(
            f"VectorStore added {added} ranked sources with chunks "
            f"(run_id={run_id})"
        )
        return added

    def search_chunks(
        self,
        query_text: str,
        query_embedding: list[float] | None = None,
        top_k: int = 5,
        run_id: str | None = None,
        min_score: float | None = None,
        embedding_weight: float = 0.55,
        keyword_weight: float = 0.45,
    ) -> list[VectorSearchResult]:
        self._ensure_loaded()

        limit = max(1, int(top_k))
        candidate_limit = max(limit * 4, 20)

        if min_score is not None:
            threshold = max(0.0, min(1.0, float(min_score)))
            threshold_source = "caller"
        else:
            corpus_size = self._count_sync()
            threshold, threshold_source = self._compute_adaptive_threshold(
                corpus_size
            )

        if not self._chunk_count_sync():
            self._record_search_diagnostics(
                operation="search_chunks",
                status="empty",
                reason="no_chunks",
                threshold=round(threshold, 4),
                threshold_source=threshold_source,
                input_count=0,
                kept_count=0,
                dropped_count=0,
                run_id=run_id,
            )
            return []

        keyword_scores = self._chunk_keyword_search(
            query_text,
            candidate_limit,
            run_id,
        )

        embedding_scores: dict[str, float] = {}

        if query_embedding and self._chunk_has_embeddings_sync():
            embedding_scores = self._chunk_embedding_search(
                query_embedding,
                candidate_limit,
                run_id,
            )

        has_embedding = bool(embedding_scores)
        candidate_chunks = set(keyword_scores) | set(embedding_scores)

        if not candidate_chunks:
            self._record_search_diagnostics(
                operation="search_chunks",
                status="empty",
                reason="no_candidate_chunks",
                threshold=round(threshold, 4),
                threshold_source=threshold_source,
                input_count=0,
                kept_count=0,
                dropped_count=0,
                has_embedding=has_embedding,
                run_id=run_id,
            )
            return []

        chunk_rows = self._fetch_chunks_by_ids(candidate_chunks)

        if not chunk_rows:
            self._record_search_diagnostics(
                operation="search_chunks",
                status="empty",
                reason="chunk_rows_empty",
                threshold=round(threshold, 4),
                threshold_source=threshold_source,
                input_count=len(candidate_chunks),
                kept_count=0,
                dropped_count=len(candidate_chunks),
                has_embedding=has_embedding,
                run_id=run_id,
            )
            return []

        parent_best: dict[str, float] = {}
        parent_chunk: dict[str, str] = {}

        for chunk_id, row in chunk_rows.items():
            parent_id = str(row.get("parent_doc_id", "") or "")

            if not parent_id:
                continue

            keyword_score = keyword_scores.get(chunk_id, 0.0)
            embedding_score = embedding_scores.get(chunk_id, 0.0)

            if has_embedding:
                combined = (
                    embedding_weight * embedding_score
                    + keyword_weight * keyword_score
                )
            else:
                combined = keyword_score

            if combined < threshold:
                continue

            existing = parent_best.get(parent_id)

            if existing is None or combined > existing:
                parent_best[parent_id] = combined
                parent_chunk[parent_id] = str(row.get("text", "") or "")

        if not parent_best:
            self._record_search_diagnostics(
                operation="search_chunks",
                status="empty",
                reason="no_parents_passed_threshold",
                threshold=round(threshold, 4),
                threshold_source=threshold_source,
                input_count=len(candidate_chunks),
                kept_count=0,
                dropped_count=len(candidate_chunks),
                has_embedding=has_embedding,
                run_id=run_id,
            )
            return []

        documents = self._fetch_documents_by_ids(set(parent_best.keys()))
        results: list[VectorSearchResult] = []

        for doc_id, score in parent_best.items():
            document = documents.get(doc_id)

            if document is None:
                continue

            snippet = parent_chunk.get(doc_id, "")

            if snippet:
                metadata = dict(document.metadata or {})
                metadata["matched_chunk"] = snippet[:500]
                try:
                    document = document.model_copy(
                        update={"metadata": metadata}
                    )
                except Exception:
                    pass

            results.append(
                VectorSearchResult(document=document, score=score)
            )

        results.sort(key=lambda item: item.score, reverse=True)
        final = results[:limit]
        dropped = max(0, len(parent_best) - len(final))

        self._logger.info(
            f"VectorStore.search_chunks returned {len(final)} parents "
            f"(chunk_hits={len(candidate_chunks)}, "
            f"has_embedding={has_embedding}, "
            f"threshold={threshold:.4f}, source={threshold_source}, "
            f"dropped={dropped})"
        )

        self._record_search_diagnostics(
            operation="search_chunks",
            status="ok",
            threshold=round(threshold, 4),
            threshold_source=threshold_source,
            input_count=len(candidate_chunks),
            kept_count=len(final),
            dropped_count=dropped,
            has_embedding=has_embedding,
            run_id=run_id,
        )

        return final

    def chunk_count(self, run_id: str | None = None) -> int:
        self._ensure_loaded()

        with self._lock:
            if self._connection is None:
                return 0

            try:
                if run_id is None:
                    cursor = self._connection.execute(
                        "SELECT COUNT(*) AS n FROM vector_document_chunks"
                    )
                else:
                    cursor = self._connection.execute(
                        """
                        SELECT COUNT(*) AS n
                        FROM vector_document_chunks AS c
                        JOIN vector_documents AS d
                            ON d.doc_id = c.parent_doc_id
                        WHERE d.run_id = ?
                        """,
                        (run_id,),
                    )
                row = cursor.fetchone()
                return int(row["n"]) if row else 0
            except Exception:
                return 0

    def all_documents(
        self, run_id: str | None = None
    ) -> list[VectorDocument]:
        self._ensure_loaded()
        rows = self._select_rows(run_id=run_id)

        documents: list[VectorDocument] = []

        for row in rows:
            try:
                documents.append(
                    self._row_to_document(row, include_embedding=False)
                )
            except Exception as exc:
                self._logger.warning(
                    f"Row decode failed in all_documents: {exc}"
                )
                continue

        return documents

    def count(self, run_id: str | None = None) -> int:
        self._ensure_loaded()

        with self._lock:
            if self._connection is None:
                return 0

            try:
                if run_id is None:
                    cursor = self._connection.execute(
                        "SELECT COUNT(*) AS n FROM vector_documents"
                    )
                else:
                    cursor = self._connection.execute(
                        "SELECT COUNT(*) AS n FROM vector_documents "
                        "WHERE run_id = ?",
                        (run_id,),
                    )
                row = cursor.fetchone()
                return int(row["n"]) if row else 0
            except Exception:
                return 0

    def clear(self) -> None:
        self._ensure_loaded()

        with self._lock:
            if self._connection is None:
                return

            try:
                self._connection.execute("DELETE FROM vector_documents")
                self._connection.execute("DELETE FROM vector_documents_fts")
                self._connection.execute("DELETE FROM vector_document_chunks")
                self._connection.execute("DELETE FROM vector_chunks_fts")
                self._connection.commit()
                self._connection.execute(
                    "PRAGMA wal_checkpoint(TRUNCATE)"
                )
            except Exception as exc:
                self._logger.warning(f"Vector store clear failed: {exc}")
                return

            self._invalidate_embedding_cache()

            with self._lock:
                self._fts_cache.clear()

            try:
                self._connection.execute("VACUUM")
            except Exception:
                pass

    @staticmethod
    def split_into_chunks(
        text: str,
        chunk_tokens: int = _DEFAULT_CHUNK_TOKENS,
    ) -> list[str]:
        cleaned = clean_text(text)

        if not cleaned:
            return []

        words = cleaned.split()

        if not words:
            return []

        size = max(_MIN_CHUNK_TOKENS, int(chunk_tokens))
        size = min(_MAX_CHUNK_TOKENS, size)

        if len(words) <= size:
            return [cleaned]

        overlap = max(10, size // 8)
        step = max(1, size - overlap)

        chunks: list[str] = []
        index = 0

        while index < len(words):
            window = words[index : index + size]

            if not window:
                break

            chunk = " ".join(window).strip()

            if chunk:
                chunks.append(chunk)

            if index + size >= len(words):
                break

            index += step

        return chunks

    def _resolve_chunk_tokens(self, chunk_tokens: int | None) -> int:
        if chunk_tokens is not None:
            try:
                parsed = int(chunk_tokens)
            except (TypeError, ValueError):
                parsed = _DEFAULT_CHUNK_TOKENS
            return max(_MIN_CHUNK_TOKENS, min(_MAX_CHUNK_TOKENS, parsed))

        avg = self._average_document_length()

        if avg <= 0:
            return _DEFAULT_CHUNK_TOKENS

        dynamic = int(avg * 0.35)
        dynamic = max(_MIN_CHUNK_TOKENS, min(_MAX_CHUNK_TOKENS, dynamic))
        return dynamic

    def _average_document_length(self) -> int:
        with self._lock:
            if self._connection is None:
                return 0

            try:
                cursor = self._connection.execute(
                    "SELECT text FROM vector_documents "
                    "ORDER BY created_at DESC LIMIT 200"
                )
                rows = cursor.fetchall()
            except Exception:
                return 0

        if not rows:
            return 0

        total = 0

        for row in rows:
            text = str(row["text"] or "")
            total += len(text.split())

        return int(total / len(rows))

    def _insert_chunks(
        self,
        parent_doc_id: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> int:
        if self._connection is None or not chunks:
            return 0

        try:
            self._connection.execute(
                "DELETE FROM vector_document_chunks WHERE parent_doc_id = ?",
                (parent_doc_id,),
            )
            self._connection.execute(
                "DELETE FROM vector_chunks_fts WHERE parent_doc_id = ?",
                (parent_doc_id,),
            )
        except Exception as exc:
            self._logger.warning(f"Chunk clear failed: {exc}")

        stored = 0
        now = datetime.now(timezone.utc).isoformat()

        for index, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue

            chunk_id = hashlib.sha256(
                f"{parent_doc_id}|{index}|{chunk_text[:64]}".encode("utf-8")
            ).hexdigest()[:32]

            embedding = (
                embeddings[index] if index < len(embeddings) else []
            )
            embedding_blob = self._pack_embedding(embedding)

            try:
                self._connection.execute(
                    """
                    INSERT OR REPLACE INTO vector_document_chunks (
                        chunk_id, parent_doc_id, chunk_index,
                        text, embedding, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        parent_doc_id,
                        index,
                        chunk_text,
                        embedding_blob,
                        now,
                    ),
                )
                self._connection.execute(
                    """
                    INSERT INTO vector_chunks_fts (
                        chunk_id, parent_doc_id, text
                    ) VALUES (?, ?, ?)
                    """,
                    (chunk_id, parent_doc_id, chunk_text),
                )
                stored += 1
            except Exception as exc:
                self._logger.warning(f"Chunk insert failed: {exc}")

        return stored

    def _chunk_count_sync(self) -> int:
        if self._connection is None:
            return 0

        try:
            cursor = self._connection.execute(
                "SELECT COUNT(*) AS n FROM vector_document_chunks"
            )
            row = cursor.fetchone()
            return int(row["n"]) if row else 0
        except Exception:
            return 0

    def _chunk_has_embeddings_sync(self) -> bool:
        if self._connection is None:
            return False

        try:
            cursor = self._connection.execute(
                "SELECT COUNT(*) AS n FROM vector_document_chunks "
                "WHERE embedding IS NOT NULL"
            )
            row = cursor.fetchone()
            return int(row["n"]) > 0 if row else False
        except Exception:
            return False

    def _chunk_keyword_search(
        self,
        query_text: str,
        top_k: int,
        run_id: str | None,
    ) -> dict[str, float]:
        query = self._build_fts_query(query_text)

        if not query or self._connection is None:
            return {}

        try:
            if run_id is None:
                cursor = self._connection.execute(
                    """
                    SELECT c.chunk_id AS chunk_id,
                           bm25(vector_chunks_fts) AS rank
                    FROM vector_chunks_fts
                    JOIN vector_document_chunks AS c
                        ON c.chunk_id = vector_chunks_fts.chunk_id
                    WHERE vector_chunks_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (query, max(1, int(top_k))),
                )
            else:
                cursor = self._connection.execute(
                    """
                    SELECT c.chunk_id AS chunk_id,
                           bm25(vector_chunks_fts) AS rank
                    FROM vector_chunks_fts
                    JOIN vector_document_chunks AS c
                        ON c.chunk_id = vector_chunks_fts.chunk_id
                    JOIN vector_documents AS d
                        ON d.doc_id = c.parent_doc_id
                    WHERE vector_chunks_fts MATCH ?
                      AND d.run_id = ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (query, run_id, max(1, int(top_k))),
                )
            rows = cursor.fetchall()
        except sqlite3.OperationalError as exc:
            self._logger.warning(f"Chunk FTS search failed: {exc}")
            return {}

        scores: dict[str, float] = {}

        for row in rows:
            chunk_id = str(row["chunk_id"] or "")

            if not chunk_id:
                continue

            raw_rank = row["rank"] if "rank" in row.keys() else 0.0
            scores[chunk_id] = self._normalize_bm25(raw_rank)

        return scores

    def _chunk_embedding_search(
        self,
        query_embedding: list[float],
        top_k: int,
        run_id: str | None,
    ) -> dict[str, float]:
        if self._connection is None or not query_embedding:
            return {}

        try:
            if run_id is None:
                cursor = self._connection.execute(
                    """
                    SELECT chunk_id, embedding
                    FROM vector_document_chunks
                    WHERE embedding IS NOT NULL
                    """
                )
            else:
                cursor = self._connection.execute(
                    """
                    SELECT c.chunk_id AS chunk_id, c.embedding AS embedding
                    FROM vector_document_chunks AS c
                    JOIN vector_documents AS d
                        ON d.doc_id = c.parent_doc_id
                    WHERE c.embedding IS NOT NULL
                      AND d.run_id = ?
                    """,
                    (run_id,),
                )
            rows = cursor.fetchall()
        except Exception as exc:
            self._logger.warning(f"Chunk embedding scan failed: {exc}")
            return {}

        scored: list[tuple[str, float]] = []

        for row in rows:
            chunk_id = str(row["chunk_id"] or "")

            if not chunk_id:
                continue

            vector = self._unpack_embedding(row["embedding"])

            if not vector:
                continue

            similarity = self._cosine_similarity(query_embedding, vector)

            if similarity > 0.0:
                scored.append((chunk_id, similarity))

        scored.sort(key=lambda item: item[1], reverse=True)

        return {
            chunk_id: score
            for chunk_id, score in scored[: max(1, int(top_k))]
        }

    def _fetch_chunks_by_ids(
        self,
        chunk_ids: set[str],
    ) -> dict[str, dict[str, Any]]:
        if not chunk_ids or self._connection is None:
            return {}

        placeholders = ",".join("?" for _ in chunk_ids)
        ids = list(chunk_ids)

        try:
            cursor = self._connection.execute(
                f"""
                SELECT chunk_id, parent_doc_id, chunk_index, text
                FROM vector_document_chunks
                WHERE chunk_id IN ({placeholders})
                """,
                ids,
            )
            rows = cursor.fetchall()
        except Exception:
            return {}

        result: dict[str, dict[str, Any]] = {}

        for row in rows:
            chunk_id = str(row["chunk_id"] or "")

            if not chunk_id:
                continue

            result[chunk_id] = {
                "parent_doc_id": str(row["parent_doc_id"] or ""),
                "chunk_index": int(row["chunk_index"] or 0),
                "text": str(row["text"] or ""),
            }

        return result

    def _count_sync(self) -> int:
        if self._connection is None:
            return 0

        try:
            cursor = self._connection.execute(
                "SELECT COUNT(*) AS n FROM vector_documents"
            )
            row = cursor.fetchone()
            return int(row["n"]) if row else 0
        except Exception:
            return 0

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def _compute_adaptive_threshold(
        self, corpus_size: int
    ) -> tuple[float, str]:
        if self._explicit_min_score is not None:
            return float(self._explicit_min_score), "explicit_override"

        if corpus_size <= 0:
            return self._min_score_default, "empty_corpus"

        if corpus_size <= self._small_corpus_cutoff:
            return self._min_score_small_corpus, "adaptive_small_corpus"

        if corpus_size <= self._medium_corpus_cutoff:
            return self._min_score_default, "adaptive_medium_corpus"

        return self._min_score_minimum, "adaptive_large_corpus"

    def _record_search_diagnostics(self, **kwargs: Any) -> None:
        payload = dict(kwargs)
        payload["recorded_at"] = datetime.now(timezone.utc).isoformat()

        with self._lock:
            self._last_search_diagnostics = payload
            
    def _augment_last_diagnostics(self, extra: dict[str, Any]) -> None:
        if not extra:
            return

        with self._lock:
            merged = dict(self._last_search_diagnostics)
            merged.update(extra)
            self._last_search_diagnostics = merged

    @staticmethod
    def _has_interrogative_marker(text: str) -> bool:
        if not text:
            return False

        lowered = str(text).lower()
        tokens = set(_TOKEN_PATTERN.findall(lowered))

        if not tokens:
            return False

        for marker in _CHUNK_GATE_INTERROGATIVE_MARKERS:
            if marker in tokens:
                return True

        return False

    def _invalidate_embedding_cache(self, doc_ids: Any = None) -> None:
        with self._lock:
            if doc_ids is None:
                self._embedding_cache.clear()
                return

            if isinstance(doc_ids, str):
                self._embedding_cache.pop(doc_ids, None)
                return

            for doc_id in doc_ids:
                self._embedding_cache.pop(str(doc_id), None)

    def _insert_document(self, document: VectorDocument) -> None:
        if self._connection is None:
            return

        embedding_blob = self._pack_embedding(document.embedding)
        metadata_text = json.dumps(
            document.metadata or {},
            ensure_ascii=False,
            default=str,
        )

        self._connection.execute(
            """
            INSERT OR REPLACE INTO vector_documents (
                doc_id, source_id, run_id, title, text, url,
                platform, source_type, domain, topic,
                metadata, embedding, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document.doc_id,
                document.source_id,
                document.run_id,
                document.title,
                document.text,
                document.url,
                document.platform,
                document.source_type,
                document.domain,
                document.topic,
                metadata_text,
                embedding_blob,
                document.created_at,
            ),
        )
        self._connection.execute(
            "DELETE FROM vector_documents_fts WHERE doc_id = ?",
            (document.doc_id,),
        )
        self._connection.execute(
            """
            INSERT INTO vector_documents_fts (doc_id, title, text, metadata)
            VALUES (?, ?, ?, ?)
            """,
            (
                document.doc_id,
                document.title or "",
                document.text or "",
                metadata_text,
            ),
        )
        self._invalidate_embedding_cache(document.doc_id)

    def _select_rows(self, run_id: str | None = None) -> list[sqlite3.Row]:
        with self._lock:
            if self._connection is None:
                return []

            try:
                if run_id is None:
                    cursor = self._connection.execute(
                        "SELECT * FROM vector_documents "
                        "ORDER BY created_at DESC"
                    )
                else:
                    cursor = self._connection.execute(
                        "SELECT * FROM vector_documents "
                        "WHERE run_id = ? ORDER BY created_at DESC",
                        (run_id,),
                    )
                return list(cursor.fetchall())
            except Exception as exc:
                self._logger.warning(f"Row selection failed: {exc}")
                return []

    def _fetch_documents_by_ids(
        self,
        doc_ids: set[str],
    ) -> dict[str, VectorDocument]:
        if not doc_ids:
            return {}

        placeholders = ",".join("?" for _ in doc_ids)
        ids = list(doc_ids)

        with self._lock:
            if self._connection is None:
                return {}

            try:
                cursor = self._connection.execute(
                    f"SELECT * FROM vector_documents "
                    f"WHERE doc_id IN ({placeholders})",
                    ids,
                )
                rows = cursor.fetchall()
            except Exception as exc:
                self._logger.warning(
                    f"Document fetch by ids failed: {exc}"
                )
                return {}

        result: dict[str, VectorDocument] = {}

        for row in rows:
            try:
                document = self._row_to_document(
                    row, include_embedding=False
                )
            except Exception as exc:
                self._logger.warning(
                    f"Row decode failed in fetch_documents_by_ids: {exc}"
                )
                continue

            result[document.doc_id] = document

        return result

    def _row_to_document(
        self,
        row: sqlite3.Row,
        include_embedding: bool = False,
    ) -> VectorDocument:
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except Exception:
            metadata = {}

        if not isinstance(metadata, dict):
            metadata = {}

        embedding: list[float] = []

        if include_embedding:
            embedding = self._unpack_embedding(row["embedding"])

        keys = row.keys()

        domain_value = (
            str(row["domain"] or "").strip().lower()
            if "domain" in keys and row["domain"] is not None
            else ""
        )
        topic_value = (
            str(row["topic"] or "").strip().lower()
            if "topic" in keys and row["topic"] is not None
            else ""
        )

        return VectorDocument(
            doc_id=str(row["doc_id"] or ""),
            source_id=str(row["source_id"] or ""),
            run_id=row["run_id"],
            title=str(row["title"] or ""),
            text=str(row["text"] or ""),
            url=row["url"],
            platform=row["platform"],
            source_type=row["source_type"],
            domain=domain_value or None,
            topic=topic_value or None,
            embedding=embedding,
            metadata=metadata,
            created_at=str(row["created_at"] or ""),
        )

    def fetch_embeddings_by_ids(
        self,
        doc_ids: set[str],
    ) -> dict[str, list[float]]:
        self._ensure_loaded()

        if not doc_ids:
            return {}

        result: dict[str, list[float]] = {}
        missing: list[str] = []

        with self._lock:
            for doc_id in doc_ids:
                cached = self._embedding_cache.get(doc_id)

                if cached is not None:
                    result[doc_id] = cached
                else:
                    missing.append(doc_id)

        if not missing:
            return result

        if self._connection is None:
            return result

        placeholders = ",".join("?" for _ in missing)

        try:
            with self._lock:
                cursor = self._connection.execute(
                    f"SELECT doc_id, embedding FROM vector_documents "
                    f"WHERE doc_id IN ({placeholders})",
                    missing,
                )
                rows = cursor.fetchall()
        except Exception as exc:
            self._logger.warning(f"fetch_embeddings_by_ids failed: {exc}")
            return result

        with self._lock:
            if len(self._embedding_cache) >= _EMBEDDING_CACHE_MAX:
                self._embedding_cache.clear()

            for row in rows:
                doc_id = str(row["doc_id"] or "")

                if not doc_id:
                    continue

                vector = self._unpack_embedding(row["embedding"])

                if vector:
                    result[doc_id] = vector
                    self._embedding_cache[doc_id] = vector

        return result

    def _trim(self) -> None:
        if self._connection is None:
            return

        try:
            cursor = self._connection.execute(
                "SELECT COUNT(*) AS n FROM vector_documents"
            )
            row = cursor.fetchone()
            count = int(row["n"]) if row else 0
        except Exception:
            return

        if count <= self._max_documents:
            return

        excess = count - self._max_documents

        try:
            cursor = self._connection.execute(
                """
                SELECT doc_id FROM vector_documents
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (excess,),
            )
            to_delete = [str(r["doc_id"]) for r in cursor.fetchall()]
        except Exception:
            return

        for doc_id in to_delete:
            try:
                self._connection.execute(
                    "DELETE FROM vector_documents WHERE doc_id = ?",
                    (doc_id,),
                )
                self._connection.execute(
                    "DELETE FROM vector_documents_fts WHERE doc_id = ?",
                    (doc_id,),
                )
                self._connection.execute(
                    "DELETE FROM vector_document_chunks "
                    "WHERE parent_doc_id = ?",
                    (doc_id,),
                )
                self._connection.execute(
                    "DELETE FROM vector_chunks_fts WHERE parent_doc_id = ?",
                    (doc_id,),
                )
            except Exception as exc:
                self._logger.warning(
                    f"Trim delete failed for {doc_id}: {exc}"
                )

            self._invalidate_embedding_cache(doc_id)

    def _make_document(
        self,
        source: Any,
        embedding: list[float],
        run_id: str | None,
        rank: Any = None,
        score: Any = None,
        confidence: Any = None,
    ) -> VectorDocument:
        title = clean_text(getattr(source, "title", "")) or "Untitled source"
        url = clean_text(getattr(source, "url", "")) or None
        source_id = (
            clean_text(getattr(source, "source_id", "")) or url or title
        )
        text = self._document_text(
            source, rank=rank, score=score, confidence=confidence
        )
        metadata = self._document_metadata(
            source, rank=rank, score=score, confidence=confidence
        )

        clean_embedding: list[float] = []

        for value in embedding or []:
            try:
                clean_embedding.append(float(value))
            except (TypeError, ValueError):
                continue

        explicit_domain = getattr(source, "domain", None)
        explicit_topic = getattr(source, "topic", None)

        if explicit_domain or explicit_topic:
            domain: str | None = (
                str(explicit_domain or "").strip().lower() or None
            )
            topic: str | None = (
                str(explicit_topic or "").strip().lower() or None
            )
        else:
            domain, topic = self._classify_source_domain(source)

        document = VectorDocument(
            doc_id="",
            source_id=source_id,
            run_id=run_id,
            title=title,
            text=text,
            url=url,
            platform=self._enum_value(getattr(source, "platform", "")) or None,
            source_type=(
                self._enum_value(getattr(source, "source_type", "")) or None
            ),
            domain=domain,
            topic=topic,
            embedding=clean_embedding,
            metadata=metadata,
        )
        document.doc_id = self._make_doc_id(document)
        return document

    def _document_text(
        self,
        source: Any,
        rank: Any = None,
        score: Any = None,
        confidence: Any = None,
    ) -> str:
        parts = [
            clean_text(getattr(source, "title", "")),
            clean_text(getattr(source, "abstract", "")),
            clean_text(getattr(source, "url", "")),
            self._enum_value(getattr(source, "platform", "")),
            self._enum_value(getattr(source, "source_type", "")),
            self._enum_value(getattr(source, "difficulty", "")),
        ]

        year = getattr(source, "year", None)
        citation_count = getattr(source, "citation_count", None)

        if year is not None:
            parts.append(str(year))

        if citation_count is not None:
            parts.append(f"citations {citation_count}")

        metadata = getattr(source, "metadata", {}) or {}

        if isinstance(metadata, dict):
            description = metadata.get("description")
            if isinstance(description, str) and description:
                parts.append(clean_text(description))

        if rank is not None:
            parts.append(f"rank {rank}")

        if score is not None:
            parts.append(f"score {score}")

        if confidence is not None:
            parts.append(f"confidence {confidence}")

        return " ".join(part for part in parts if part)

    def _document_metadata(
        self,
        source: Any,
        rank: Any = None,
        score: Any = None,
        confidence: Any = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        raw_metadata = getattr(source, "metadata", {}) or {}

        if isinstance(raw_metadata, dict):
            metadata.update(raw_metadata)

        metadata["platform"] = self._enum_value(
            getattr(source, "platform", "")
        )
        metadata["source_type"] = self._enum_value(
            getattr(source, "source_type", "")
        )
        metadata["difficulty"] = self._enum_value(
            getattr(source, "difficulty", "")
        )
        metadata["year"] = getattr(source, "year", None)
        metadata["citation_count"] = getattr(source, "citation_count", None)
        metadata["rank"] = rank
        metadata["score"] = score
        metadata["confidence"] = confidence

        return metadata

    def _make_doc_id(self, document: VectorDocument) -> str:
        raw = "|".join(
            [
                str(document.run_id or ""),
                str(document.source_id or ""),
                str(document.title or ""),
                str(document.url or ""),
            ]
        )
        return (
            hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
            or str(uuid4())
        )

    def _migrate_legacy_json(self, force: bool = False) -> int:
        legacy_path = self._legacy_json_path
        marker_path = self._migrated_marker_path

        if not legacy_path.exists():
            if marker_path.exists() and force:
                self._logger.info(
                    "Legacy JSON already migrated, marker present"
                )
            return 0

        if self._connection is None:
            return 0

        existing_count = self._count_sync()

        if existing_count > 0 and not force:
            self._logger.info(
                f"SQLite already has {existing_count} docs; "
                f"skipping JSON migration"
            )
            self._rename_legacy_json(legacy_path, marker_path)
            return 0

        try:
            raw = json.loads(
                legacy_path.read_text(encoding="utf-8", errors="replace")
            )
        except Exception as exc:
            self._logger.warning(f"Legacy JSON read failed: {exc}")
            return 0

        items: Any = []

        if isinstance(raw, dict):
            items = raw.get("documents", [])
        elif isinstance(raw, list):
            items = raw

        if not isinstance(items, list):
            self._logger.warning(
                "Legacy JSON has no document list; skipping migration"
            )
            return 0

        migrated = 0
        failed = 0

        for item in items:
            if not isinstance(item, dict):
                continue

            try:
                document = VectorDocument.model_validate(item)
            except Exception:
                failed += 1
                continue

            if not document.doc_id:
                document.doc_id = self._make_doc_id(document)

            try:
                self._insert_document(document)
                migrated += 1
            except Exception as exc:
                failed += 1
                self._logger.warning(
                    f"Migration insert failed for "
                    f"{document.source_id}: {exc}"
                )

        try:
            self._connection.commit()
        except Exception:
            pass

        self._logger.info(
            f"Legacy JSON migration: migrated={migrated} "
            f"failed={failed} total={len(items)}"
        )

        if migrated > 0 or force:
            self._rename_legacy_json(legacy_path, marker_path)

        return migrated

    def _rename_legacy_json(
        self,
        legacy_path: Path,
        marker_path: Path,
    ) -> None:
        try:
            if marker_path.exists():
                marker_path.unlink()

            legacy_path.replace(marker_path)
            self._logger.info(
                f"Renamed legacy JSON to {marker_path.name}"
            )
        except Exception as exc:
            self._logger.warning(
                f"Could not rename legacy JSON: {exc}"
            )

    def _fallback_keyword_search(
        self,
        query_text: str,
        top_k: int,
        run_id: str | None,
        threshold: float,
    ) -> list[VectorSearchResult]:
        tokens = self._tokenize(query_text)

        tokens = [
            token
            for token in tokens
            if token and token not in _FTS_EXCLUDED and len(token) > 1
        ]

        if not tokens:
            self._logger.info(
                "VectorStore._fallback_keyword_search skipped: "
                "no meaningful tokens after stopword filtering"
            )
            return []

        rows = self._select_rows(run_id=run_id)
        results: list[VectorSearchResult] = []

        for row in rows:
            try:
                document = self._row_to_document(
                    row, include_embedding=False
                )
            except Exception as exc:
                self._logger.warning(
                    f"Row decode failed in fallback keyword search: {exc}"
                )
                continue

            score = self._lexical_score(document, tokens)

            if score < threshold:
                continue

            results.append(
                VectorSearchResult(document=document, score=score)
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[: max(1, int(top_k))]

    def _lexical_score(
        self,
        document: VectorDocument,
        query_tokens: list[str],
    ) -> float:
        if not query_tokens:
            return 0.0

        unique_query = set(query_tokens)
        text_tokens = set(self._tokenize(document.text))
        title_tokens = set(self._tokenize(document.title))
        metadata_tokens = set(
            self._tokenize(
                " ".join(
                    [
                        str(document.platform or ""),
                        str(document.source_type or ""),
                        str(document.url or ""),
                        " ".join(
                            str(value)
                            for value in (document.metadata or {}).values()
                        ),
                    ]
                )
            )
        )

        text_overlap = len(unique_query & text_tokens)
        title_overlap = len(unique_query & title_tokens)
        metadata_overlap = len(unique_query & metadata_tokens)
        max_possible = max(1, len(unique_query)) * 4
        raw_score = text_overlap + (3 * title_overlap) + metadata_overlap

        return min(1.0, raw_score / max_possible)

    def _build_fts_query(self, text: str) -> str:
        normalized = " ".join(str(text or "").lower().split())

        if not normalized:
            return ""

        with self._lock:
            cached = self._fts_cache.get(normalized)

        if cached is not None:
            return cached

        tokens = self._tokenize(normalized)

        filtered: list[str] = []
        seen_tokens: set[str] = set()

        for token in tokens:
            cleaned = token.strip().strip('"').strip("'").strip()

            if not cleaned:
                continue

            if len(cleaned) <= 1:
                continue

            if cleaned in _FTS_EXCLUDED:
                continue

            if cleaned.isdigit():
                continue

            if cleaned in seen_tokens:
                continue

            seen_tokens.add(cleaned)
            filtered.append(cleaned)

            if len(filtered) >= 12:
                break

        if not filtered:
            with self._lock:
                if len(self._fts_cache) >= _FTS_CACHE_MAX:
                    self._fts_cache.clear()
                self._fts_cache[normalized] = ""
            return ""

        escaped = [f'"{t}"' for t in filtered]
        query = " OR ".join(escaped)

        with self._lock:
            if len(self._fts_cache) >= _FTS_CACHE_MAX:
                self._fts_cache.clear()
            self._fts_cache[normalized] = query

        return query

    @staticmethod
    def _normalize_bm25(raw_rank: Any) -> float:
        try:
            value = float(raw_rank)
        except (TypeError, ValueError):
            return 0.0

        value = abs(value)
        return value / (value + 5.0)

    @staticmethod
    def _pack_embedding(embedding: list[float]) -> bytes | None:
        if not embedding:
            return None

        try:
            return struct.pack(f"<{len(embedding)}f", *embedding)
        except (struct.error, TypeError):
            return None

    @staticmethod
    def _unpack_embedding(blob: Any) -> list[float]:
        if not blob:
            return []

        if isinstance(blob, memoryview):
            blob = bytes(blob)

        if not isinstance(blob, (bytes, bytearray)):
            return []

        size = len(blob)

        if size % 4 != 0:
            return []

        count = size // 4

        try:
            return list(struct.unpack(f"<{count}f", blob))
        except struct.error:
            return []

    @staticmethod
    def _cosine_similarity(
        left: list[float],
        right: list[float],
    ) -> float:
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

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        raw = _TOKEN_PATTERN.findall(str(text or "").lower())

        filtered = [
            token
            for token in raw
            if len(token) > 1 and token not in _STOPWORDS
        ]

        return filtered or [token for token in raw if token]

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value) or "")