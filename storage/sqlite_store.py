from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from core.exceptions import StorageError
from utils.async_helpers import run_sync_in_executor
from utils.logger import get_logger


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    request_id TEXT PRIMARY KEY,
    topic TEXT,
    goal TEXT,
    level TEXT,
    status TEXT,
    total_sources INTEGER,
    total_ranked INTEGER,
    total_steps INTEGER,
    latency_ms REAL,
    created_at TEXT,
    finished_at TEXT,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT,
    rank INTEGER,
    source_id TEXT,
    title TEXT,
    url TEXT,
    platform TEXT,
    source_type TEXT,
    difficulty TEXT,
    year INTEGER,
    citation_count INTEGER,
    score REAL,
    confidence REAL,
    reason TEXT,
    abstract TEXT,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS learning_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT,
    step INTEGER,
    title TEXT,
    objective TEXT,
    estimated_minutes INTEGER,
    resources TEXT
);

CREATE TABLE IF NOT EXISTS agent_runs (
    agent_id TEXT PRIMARY KEY,
    topic TEXT,
    goal TEXT,
    level TEXT,
    status TEXT,
    total_iterations INTEGER,
    total_tokens_used INTEGER,
    total_budget INTEGER,
    tokens_remaining INTEGER,
    total_sources INTEGER,
    total_ranked INTEGER,
    total_steps INTEGER,
    latency_ms REAL,
    created_at TEXT,
    finished_at TEXT,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS agent_memory (
    entry_id TEXT PRIMARY KEY,
    memory_type TEXT,
    key TEXT,
    content TEXT,
    metadata TEXT,
    embedding TEXT,
    relevance_score REAL,
    access_count INTEGER,
    created_at TEXT,
    last_accessed_at TEXT
);

CREATE TABLE IF NOT EXISTS agent_episodes (
    episode_id TEXT PRIMARY KEY,
    task TEXT,
    task_type TEXT,
    goal TEXT,
    level TEXT,
    status TEXT,
    quality_score REAL,
    total_iterations INTEGER,
    total_tokens_used INTEGER,
    actions_taken TEXT,
    errors TEXT,
    warnings TEXT,
    payload TEXT,
    created_at TEXT,
    finished_at TEXT,
    latency_ms REAL
);

CREATE INDEX IF NOT EXISTS idx_sources_request_id ON sources(request_id);
CREATE INDEX IF NOT EXISTS idx_steps_request_id ON learning_steps(request_id);
CREATE INDEX IF NOT EXISTS idx_agent_memory_key ON agent_memory(key);
CREATE INDEX IF NOT EXISTS idx_agent_memory_type ON agent_memory(memory_type);
CREATE INDEX IF NOT EXISTS idx_agent_episodes_task ON agent_episodes(task);
CREATE INDEX IF NOT EXISTS idx_agent_episodes_type ON agent_episodes(task_type);
"""


_COLUMN_DECLARATIONS: dict[str, list[tuple[str, str]]] = {
    "runs": [
        ("topic", "TEXT"),
        ("goal", "TEXT"),
        ("level", "TEXT"),
        ("status", "TEXT"),
        ("total_sources", "INTEGER"),
        ("total_ranked", "INTEGER"),
        ("total_steps", "INTEGER"),
        ("latency_ms", "REAL"),
        ("created_at", "TEXT"),
        ("finished_at", "TEXT"),
        ("payload", "TEXT"),
    ],
    "sources": [
        ("request_id", "TEXT"),
        ("rank", "INTEGER"),
        ("source_id", "TEXT"),
        ("title", "TEXT"),
        ("url", "TEXT"),
        ("platform", "TEXT"),
        ("source_type", "TEXT"),
        ("difficulty", "TEXT"),
        ("year", "INTEGER"),
        ("citation_count", "INTEGER"),
        ("score", "REAL"),
        ("confidence", "REAL"),
        ("reason", "TEXT"),
        ("abstract", "TEXT"),
        ("metadata", "TEXT"),
    ],
    "learning_steps": [
        ("request_id", "TEXT"),
        ("step", "INTEGER"),
        ("title", "TEXT"),
        ("objective", "TEXT"),
        ("estimated_minutes", "INTEGER"),
        ("resources", "TEXT"),
    ],
    "agent_runs": [
        ("topic", "TEXT"),
        ("goal", "TEXT"),
        ("level", "TEXT"),
        ("status", "TEXT"),
        ("total_iterations", "INTEGER"),
        ("total_tokens_used", "INTEGER"),
        ("total_budget", "INTEGER"),
        ("tokens_remaining", "INTEGER"),
        ("total_sources", "INTEGER"),
        ("total_ranked", "INTEGER"),
        ("total_steps", "INTEGER"),
        ("latency_ms", "REAL"),
        ("created_at", "TEXT"),
        ("finished_at", "TEXT"),
        ("payload", "TEXT"),
    ],
    "agent_memory": [
        ("memory_type", "TEXT"),
        ("key", "TEXT"),
        ("content", "TEXT"),
        ("metadata", "TEXT"),
        ("embedding", "TEXT"),
        ("relevance_score", "REAL"),
        ("access_count", "INTEGER"),
        ("created_at", "TEXT"),
        ("last_accessed_at", "TEXT"),
    ],
    "agent_episodes": [
        ("task", "TEXT"),
        ("task_type", "TEXT"),
        ("goal", "TEXT"),
        ("level", "TEXT"),
        ("status", "TEXT"),
        ("quality_score", "REAL"),
        ("total_iterations", "INTEGER"),
        ("total_tokens_used", "INTEGER"),
        ("actions_taken", "TEXT"),
        ("errors", "TEXT"),
        ("warnings", "TEXT"),
        ("payload", "TEXT"),
        ("created_at", "TEXT"),
        ("finished_at", "TEXT"),
        ("latency_ms", "REAL"),
    ],
}


class SQLiteStore:
    def __init__(self, database_path: Path | str) -> None:
        self._database_path = Path(database_path).expanduser()
        self._lock = threading.RLock()
        self._logger = get_logger("storage.sqlite_store")

    @property
    def database_path(self) -> Path:
        return self._database_path

    async def initialize(self) -> None:
        await run_sync_in_executor(self._ensure_schema_sync)

    def _ensure_schema_sync(self) -> None:
        with self._lock:
            try:
                self._database_path.parent.mkdir(parents=True, exist_ok=True)
                connection = sqlite3.connect(str(self._database_path))
                try:
                    connection.executescript(_SCHEMA)
                    for table, columns in _COLUMN_DECLARATIONS.items():
                        for column_name, declaration in columns:
                            self._ensure_column(
                                connection,
                                table,
                                column_name,
                                declaration,
                            )
                    connection.commit()
                finally:
                    connection.close()
            except Exception as exc:
                raise StorageError(
                    "Failed to initialize SQLite schema",
                    details={
                        "path": str(self._database_path),
                        "error": str(exc),
                    },
                ) from exc

    def _ensure_column(
        self,
        connection: sqlite3.Connection,
        table: str,
        column: str,
        declaration: str,
    ) -> None:
        try:
            rows = connection.execute(
                f"PRAGMA table_info({table})"
            ).fetchall()
            names = {str(row[1]) for row in rows}
            if column not in names:
                connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
                )
        except Exception as exc:
            self._logger.warning(
                f"Could not ensure column {table}.{column}: {exc}"
            )

    async def save_state(self, state: Any) -> None:
        await run_sync_in_executor(self._save_state_sync, state)

    def _save_state_sync(self, state: Any) -> None:
        with self._lock:
            self._ensure_schema_sync()
            connection = sqlite3.connect(str(self._database_path))
            try:
                cursor = connection.cursor()
                request_id = str(self._get(state, "request_id", "") or "")
                if not request_id:
                    raise StorageError("State is missing request_id")
                query = self._get(state, "query", None)
                topic = str(self._get(query, "topic", "") or "")
                goal = str(self._get(query, "goal", "") or "")
                level = str(self._get(query, "level", "") or "")
                status = self._enum_value(self._get(state, "status", ""))
                sources = self._as_list(self._get(state, "sources", []))
                ranked_sources = self._as_list(
                    self._get(state, "ranked_sources", [])
                )
                learning_path = self._get(state, "learning_path", None)
                total_sources = len(sources)
                total_ranked = len(ranked_sources)
                total_steps = self._count_learning_steps(learning_path)
                latency_ms = self._to_float(
                    self._get(state, "latency_ms", None), 0.0
                )
                created_at = self._iso(
                    self._get(state, "created_at", None)
                )
                finished_at = self._iso(
                    self._get(state, "finished_at", None)
                )
                payload = json.dumps(
                    self._model_to_dict(state),
                    ensure_ascii=False,
                    default=str,
                )
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO runs (
                        request_id, topic, goal, level, status,
                        total_sources, total_ranked, total_steps,
                        latency_ms, created_at, finished_at, payload
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request_id,
                        topic,
                        goal,
                        level,
                        status,
                        total_sources,
                        total_ranked,
                        total_steps,
                        latency_ms,
                        created_at,
                        finished_at,
                        payload,
                    ),
                )
                cursor.execute(
                    "DELETE FROM sources WHERE request_id = ?",
                    (request_id,),
                )
                cursor.execute(
                    "DELETE FROM learning_steps WHERE request_id = ?",
                    (request_id,),
                )
                if ranked_sources:
                    for index, ranked in enumerate(ranked_sources, start=1):
                        source = self._get(ranked, "source", ranked)
                        cursor.execute(
                            """
                            INSERT INTO sources (
                                request_id, rank, source_id, title, url,
                                platform, source_type, difficulty, year,
                                citation_count, score, confidence, reason,
                                abstract, metadata
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            self._source_row(
                                source,
                                request_id=request_id,
                                rank=self._to_int(
                                    self._get(ranked, "rank", index), index
                                ),
                                score=self._to_float_optional(
                                    self._get(ranked, "score", None)
                                ),
                                confidence=self._to_float_optional(
                                    self._get(ranked, "confidence", None)
                                ),
                                reason=str(
                                    self._get(ranked, "reason", "") or ""
                                )
                                or None,
                            ),
                        )
                else:
                    for index, source in enumerate(sources, start=1):
                        cursor.execute(
                            """
                            INSERT INTO sources (
                                request_id, rank, source_id, title, url,
                                platform, source_type, difficulty, year,
                                citation_count, score, confidence, reason,
                                abstract, metadata
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            self._source_row(
                                source,
                                request_id=request_id,
                                rank=index,
                                score=None,
                                confidence=None,
                                reason=None,
                            ),
                        )
                steps = (
                    self._as_list(self._get(learning_path, "steps", []))
                    if learning_path
                    else []
                )
                for index, step in enumerate(steps, start=1):
                    resources = self._as_list(
                        self._get(step, "resources", [])
                    )
                    cursor.execute(
                        """
                        INSERT INTO learning_steps (
                            request_id, step, title, objective,
                            estimated_minutes, resources
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            request_id,
                            self._to_int(
                                self._get(step, "step", index), index
                            ),
                            str(self._get(step, "title", "") or ""),
                            str(self._get(step, "objective", "") or ""),
                            self._to_int_optional(
                                self._get(step, "estimated_minutes", None)
                            ),
                            json.dumps(
                                self._serialize_resources(resources),
                                ensure_ascii=False,
                                default=str,
                            ),
                        ),
                    )
                connection.commit()
            except StorageError:
                raise
            except Exception as exc:
                connection.rollback()
                raise StorageError(
                    "Failed to save state to SQLite",
                    details={"error": str(exc)},
                ) from exc
            finally:
                connection.close()

    async def save_agent_run(self, result: Any) -> None:
        await run_sync_in_executor(self._save_agent_run_sync, result)

    def _save_agent_run_sync(self, result: Any) -> None:
        with self._lock:
            self._ensure_schema_sync()
            connection = sqlite3.connect(str(self._database_path))
            try:
                cursor = connection.cursor()
                agent_id = str(
                    self._get(result, "agent_id", "") or uuid4()
                )
                topic = str(self._get(result, "topic", "") or "")
                goal = str(self._get(result, "goal", "") or "")
                level = str(self._get(result, "level", "") or "")
                status = self._enum_value(self._get(result, "status", ""))
                findings = self._as_list(
                    self._get(result, "findings", None)
                    or self._get(result, "sources", [])
                )
                ranked_sources = self._as_list(
                    self._get(result, "ranked_sources", [])
                )
                learning_path = self._get(result, "learning_path", None)
                total_iterations = self._to_int(
                    self._get(result, "total_iterations", 0), 0
                )
                total_tokens_used = self._to_int(
                    self._get(result, "total_tokens_used", 0), 0
                )
                total_budget = self._to_int(
                    self._get(result, "total_budget", 0), 0
                )
                tokens_remaining = self._to_int(
                    self._get(result, "tokens_remaining", 0), 0
                )
                total_sources = len(findings)
                total_ranked = len(ranked_sources)
                total_steps = self._count_learning_steps(learning_path)
                latency_ms = self._to_float(
                    self._get(result, "latency_ms", None), 0.0
                )
                created_at = self._iso(
                    self._get(result, "started_at", None)
                )
                finished_at = self._iso(
                    self._get(result, "finished_at", None)
                )
                payload = json.dumps(
                    self._model_to_dict(result),
                    ensure_ascii=False,
                    default=str,
                )
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO runs (
                        request_id, topic, goal, level, status,
                        total_sources, total_ranked, total_steps,
                        latency_ms, created_at, finished_at, payload
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        agent_id,
                        topic,
                        goal,
                        level,
                        status,
                        total_sources,
                        total_ranked,
                        total_steps,
                        latency_ms,
                        created_at,
                        finished_at,
                        payload,
                    ),
                )
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO agent_runs (
                        agent_id, topic, goal, level, status,
                        total_iterations, total_tokens_used, total_budget,
                        tokens_remaining, total_sources, total_ranked,
                        total_steps, latency_ms, created_at, finished_at,
                        payload
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        agent_id,
                        topic,
                        goal,
                        level,
                        status,
                        total_iterations,
                        total_tokens_used,
                        total_budget,
                        tokens_remaining,
                        total_sources,
                        total_ranked,
                        total_steps,
                        latency_ms,
                        created_at,
                        finished_at,
                        payload,
                    ),
                )
                cursor.execute(
                    "DELETE FROM sources WHERE request_id = ?",
                    (agent_id,),
                )
                cursor.execute(
                    "DELETE FROM learning_steps WHERE request_id = ?",
                    (agent_id,),
                )
                if ranked_sources:
                    for index, ranked in enumerate(ranked_sources, start=1):
                        source = self._get(ranked, "source", ranked)
                        cursor.execute(
                            """
                            INSERT INTO sources (
                                request_id, rank, source_id, title, url,
                                platform, source_type, difficulty, year,
                                citation_count, score, confidence, reason,
                                abstract, metadata
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            self._source_row(
                                source,
                                request_id=agent_id,
                                rank=self._to_int(
                                    self._get(ranked, "rank", index), index
                                ),
                                score=self._to_float_optional(
                                    self._get(ranked, "score", None)
                                ),
                                confidence=self._to_float_optional(
                                    self._get(ranked, "confidence", None)
                                ),
                                reason=str(
                                    self._get(ranked, "reason", "") or ""
                                )
                                or None,
                            ),
                        )
                else:
                    for index, source in enumerate(findings, start=1):
                        cursor.execute(
                            """
                            INSERT INTO sources (
                                request_id, rank, source_id, title, url,
                                platform, source_type, difficulty, year,
                                citation_count, score, confidence, reason,
                                abstract, metadata
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            self._source_row(
                                source,
                                request_id=agent_id,
                                rank=index,
                                score=None,
                                confidence=None,
                                reason=None,
                            ),
                        )
                steps = (
                    self._as_list(self._get(learning_path, "steps", []))
                    if learning_path
                    else []
                )
                for index, step in enumerate(steps, start=1):
                    resources = self._as_list(
                        self._get(step, "resources", [])
                    )
                    cursor.execute(
                        """
                        INSERT INTO learning_steps (
                            request_id, step, title, objective,
                            estimated_minutes, resources
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            agent_id,
                            self._to_int(
                                self._get(step, "step", index), index
                            ),
                            str(self._get(step, "title", "") or ""),
                            str(self._get(step, "objective", "") or ""),
                            self._to_int_optional(
                                self._get(step, "estimated_minutes", None)
                            ),
                            json.dumps(
                                self._serialize_resources(resources),
                                ensure_ascii=False,
                                default=str,
                            ),
                        ),
                    )
                connection.commit()
            except StorageError:
                raise
            except Exception as exc:
                connection.rollback()
                raise StorageError(
                    "Failed to save agent run to SQLite",
                    details={"error": str(exc)},
                ) from exc
            finally:
                connection.close()

    async def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        return await run_sync_in_executor(self._list_runs_sync, limit)

    def _list_runs_sync(self, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            self._ensure_schema_sync()
            connection = sqlite3.connect(str(self._database_path))
            try:
                connection.row_factory = sqlite3.Row
                cursor = connection.cursor()
                cursor.execute(
                    "SELECT * FROM runs ORDER BY COALESCE(created_at, '') DESC LIMIT ?",
                    (max(1, int(limit)),),
                )
                return [self._row_to_dict(row) for row in cursor.fetchall()]
            except Exception as exc:
                raise StorageError(
                    "Failed to list runs",
                    details={"error": str(exc)},
                ) from exc
            finally:
                connection.close()

    async def get_run(self, request_id: str) -> Optional[dict[str, Any]]:
        return await run_sync_in_executor(self._get_run_sync, request_id)

    def _get_run_sync(self, request_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            self._ensure_schema_sync()
            connection = sqlite3.connect(str(self._database_path))
            try:
                connection.row_factory = sqlite3.Row
                cursor = connection.cursor()
                cursor.execute(
                    "SELECT * FROM runs WHERE request_id = ?",
                    (request_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    return None
                run = self._row_to_dict(row)
                cursor.execute(
                    "SELECT * FROM sources WHERE request_id = ? ORDER BY rank",
                    (request_id,),
                )
                run["sources"] = [
                    self._row_to_dict(source_row)
                    for source_row in cursor.fetchall()
                ]
                cursor.execute(
                    "SELECT * FROM learning_steps WHERE request_id = ? ORDER BY step",
                    (request_id,),
                )
                run["learning_steps"] = [
                    self._row_to_dict(step_row)
                    for step_row in cursor.fetchall()
                ]
                return run
            except Exception as exc:
                raise StorageError(
                    "Failed to get run",
                    details={
                        "request_id": request_id,
                        "error": str(exc),
                    },
                ) from exc
            finally:
                connection.close()

    async def list_agent_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        return await run_sync_in_executor(self._list_agent_runs_sync, limit)

    def _list_agent_runs_sync(self, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            self._ensure_schema_sync()
            connection = sqlite3.connect(str(self._database_path))
            try:
                connection.row_factory = sqlite3.Row
                cursor = connection.cursor()
                cursor.execute(
                    "SELECT * FROM agent_runs ORDER BY COALESCE(created_at, '') DESC LIMIT ?",
                    (max(1, int(limit)),),
                )
                return [self._row_to_dict(row) for row in cursor.fetchall()]
            except Exception as exc:
                raise StorageError(
                    "Failed to list agent runs",
                    details={"error": str(exc)},
                ) from exc
            finally:
                connection.close()

    async def get_agent_run(self, agent_id: str) -> Optional[dict[str, Any]]:
        return await run_sync_in_executor(self._get_agent_run_sync, agent_id)

    def _get_agent_run_sync(self, agent_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            self._ensure_schema_sync()
            connection = sqlite3.connect(str(self._database_path))
            try:
                connection.row_factory = sqlite3.Row
                cursor = connection.cursor()
                cursor.execute(
                    "SELECT * FROM agent_runs WHERE agent_id = ?",
                    (agent_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    return None
                run = self._row_to_dict(row)
                cursor.execute(
                    "SELECT * FROM sources WHERE request_id = ? ORDER BY rank",
                    (agent_id,),
                )
                run["sources"] = [
                    self._row_to_dict(source_row)
                    for source_row in cursor.fetchall()
                ]
                cursor.execute(
                    "SELECT * FROM learning_steps WHERE request_id = ? ORDER BY step",
                    (agent_id,),
                )
                run["learning_steps"] = [
                    self._row_to_dict(step_row)
                    for step_row in cursor.fetchall()
                ]
                return run
            except Exception as exc:
                raise StorageError(
                    "Failed to get agent run",
                    details={"agent_id": agent_id, "error": str(exc)},
                ) from exc
            finally:
                connection.close()

    async def save_memory_entry(self, entry: Any) -> None:
        await run_sync_in_executor(self._save_memory_entry_sync, entry)

    def _save_memory_entry_sync(self, entry: Any) -> None:
        with self._lock:
            self._ensure_schema_sync()
            connection = sqlite3.connect(str(self._database_path))
            try:
                cursor = connection.cursor()
                entry_id = str(
                    self._get(entry, "entry_id", "") or uuid4()
                )
                memory_type = str(
                    self._get(entry, "memory_type", "episodic") or "episodic"
                )
                key = str(self._get(entry, "key", "") or "")
                content = str(self._get(entry, "content", "") or "")
                metadata = json.dumps(
                    self._get(entry, "metadata", {}) or {},
                    ensure_ascii=False,
                    default=str,
                )
                embedding = json.dumps(
                    self._get(entry, "embedding", []) or [],
                    ensure_ascii=False,
                    default=str,
                )
                relevance_score = self._to_float(
                    self._get(entry, "relevance_score", 0.0), 0.0
                )
                access_count = self._to_int(
                    self._get(entry, "access_count", 0), 0
                )
                created_at = self._iso(
                    self._get(entry, "created_at", None)
                )
                last_accessed_at = self._iso(
                    self._get(entry, "last_accessed_at", None)
                )
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO agent_memory (
                        entry_id, memory_type, key, content, metadata,
                        embedding, relevance_score, access_count,
                        created_at, last_accessed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry_id,
                        memory_type,
                        key,
                        content,
                        metadata,
                        embedding,
                        relevance_score,
                        access_count,
                        created_at,
                        last_accessed_at,
                    ),
                )
                connection.commit()
            except Exception as exc:
                connection.rollback()
                raise StorageError(
                    "Failed to save memory entry",
                    details={"error": str(exc)},
                ) from exc
            finally:
                connection.close()

    async def recall_memory(
        self,
        key: str,
        memory_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        return await run_sync_in_executor(
            self._recall_memory_sync, key, memory_type
        )

    def _recall_memory_sync(
        self,
        key: str,
        memory_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            self._ensure_schema_sync()
            connection = sqlite3.connect(str(self._database_path))
            try:
                connection.row_factory = sqlite3.Row
                cursor = connection.cursor()
                if memory_type:
                    cursor.execute(
                        """
                        SELECT * FROM agent_memory
                        WHERE key LIKE ? AND memory_type = ?
                        ORDER BY created_at DESC
                        LIMIT 20
                        """,
                        (f"%{key}%", memory_type),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT * FROM agent_memory
                        WHERE key LIKE ?
                        ORDER BY created_at DESC
                        LIMIT 20
                        """,
                        (f"%{key}%",),
                    )
                rows = cursor.fetchall()
                results = [self._row_to_dict(row) for row in rows]
                now = self._now()
                for result in results:
                    entry_id = result.get("entry_id")
                    if entry_id:
                        cursor.execute(
                            """
                            UPDATE agent_memory
                            SET access_count = access_count + 1,
                                last_accessed_at = ?
                            WHERE entry_id = ?
                            """,
                            (now, entry_id),
                        )
                connection.commit()
                return results
            except Exception as exc:
                raise StorageError(
                    "Failed to recall memory",
                    details={"error": str(exc)},
                ) from exc
            finally:
                connection.close()

    async def save_episode(self, episode: Any) -> None:
        await run_sync_in_executor(self._save_episode_sync, episode)

    def _save_episode_sync(self, episode: Any) -> None:
        with self._lock:
            self._ensure_schema_sync()
            connection = sqlite3.connect(str(self._database_path))
            try:
                cursor = connection.cursor()
                episode_id = str(
                    self._get(episode, "episode_id", "") or uuid4()
                )
                task = str(self._get(episode, "task", "") or "")
                task_type = str(self._get(episode, "task_type", "") or "")
                goal = str(self._get(episode, "goal", "") or "")
                level = str(self._get(episode, "level", "") or "")
                status = self._enum_value(self._get(episode, "status", ""))
                quality_score = self._to_float(
                    self._get(episode, "quality_score", 0.0), 0.0
                )
                total_iterations = self._to_int(
                    self._get(episode, "total_iterations", 0), 0
                )
                total_tokens_used = self._to_int(
                    self._get(episode, "total_tokens_used", 0), 0
                )
                actions_taken = self._as_list(
                    self._get(episode, "actions_taken", [])
                )
                errors = self._as_list(self._get(episode, "errors", []))
                warnings = self._as_list(
                    self._get(episode, "warnings", [])
                )
                actions_payload = json.dumps(
                    [self._model_to_dict(action) for action in actions_taken],
                    ensure_ascii=False,
                    default=str,
                )
                errors_payload = json.dumps(
                    [str(item) for item in errors],
                    ensure_ascii=False,
                    default=str,
                )
                warnings_payload = json.dumps(
                    [str(item) for item in warnings],
                    ensure_ascii=False,
                    default=str,
                )
                payload = json.dumps(
                    self._model_to_dict(episode),
                    ensure_ascii=False,
                    default=str,
                )
                created_at = self._iso(
                    self._get(episode, "created_at", None)
                )
                finished_at = self._iso(
                    self._get(episode, "finished_at", None)
                )
                latency_ms = self._to_float(
                    self._get(episode, "latency_ms", None), 0.0
                )
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO agent_episodes (
                        episode_id, task, task_type, goal, level,
                        status, quality_score, total_iterations,
                        total_tokens_used, actions_taken, errors,
                        warnings, payload, created_at, finished_at,
                        latency_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        episode_id,
                        task,
                        task_type,
                        goal,
                        level,
                        status,
                        quality_score,
                        total_iterations,
                        total_tokens_used,
                        actions_payload,
                        errors_payload,
                        warnings_payload,
                        payload,
                        created_at,
                        finished_at,
                        latency_ms,
                    ),
                )
                connection.commit()
            except Exception as exc:
                connection.rollback()
                raise StorageError(
                    "Failed to save episode",
                    details={"error": str(exc)},
                ) from exc
            finally:
                connection.close()

    def _source_row(
        self,
        source: Any,
        request_id: str,
        rank: int,
        score: Optional[float],
        confidence: Optional[float],
        reason: Optional[str],
    ) -> tuple[Any, ...]:
        metadata = self._get(source, "metadata", {}) or {}
        return (
            request_id,
            rank,
            str(self._get(source, "source_id", "") or ""),
            str(self._get(source, "title", "") or ""),
            str(self._get(source, "url", "") or ""),
            self._enum_value(self._get(source, "platform", "")),
            self._enum_value(self._get(source, "source_type", "")),
            self._enum_value(self._get(source, "difficulty", "")) or None,
            self._to_int_optional(self._get(source, "year", None)),
            self._to_int_optional(
                self._get(source, "citation_count", None)
            ),
            score,
            confidence,
            reason,
            str(self._get(source, "abstract", "") or "") or None,
            json.dumps(metadata, ensure_ascii=False, default=str),
        )

    def _serialize_resources(
        self,
        resources: list[Any],
    ) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for resource in resources:
            source = self._get(resource, "source", resource)
            serialized.append(
                {
                    "source_id": str(
                        self._get(source, "source_id", "") or ""
                    ),
                    "title": str(self._get(source, "title", "") or ""),
                    "url": str(self._get(source, "url", "") or ""),
                    "platform": self._enum_value(
                        self._get(source, "platform", "")
                    ),
                    "source_type": self._enum_value(
                        self._get(source, "source_type", "")
                    ),
                    "difficulty": self._enum_value(
                        self._get(source, "difficulty", "")
                    )
                    or None,
                    "rank": self._to_int_optional(
                        self._get(resource, "rank", None)
                    ),
                    "score": self._to_float_optional(
                        self._get(resource, "score", None)
                    ),
                    "confidence": self._to_float_optional(
                        self._get(resource, "confidence", None)
                    ),
                }
            )
        return serialized

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        for key in (
            "payload",
            "metadata",
            "resources",
            "embedding",
            "actions_taken",
            "errors",
            "warnings",
        ):
            if key in data:
                data[key] = self._json_loads_safe(
                    data[key], self._default_for_key(key)
                )
        return data

    def _default_for_key(self, key: str) -> Any:
        if key in ("metadata", "payload"):
            return {}
        return []

    def _json_loads_safe(self, value: Any, default: Any) -> Any:
        if value is None:
            return default
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except Exception:
            return default

    def _model_to_dict(self, obj: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj
        model_dump = getattr(obj, "model_dump", None)
        if callable(model_dump):
            try:
                return model_dump(
                    mode="json",
                    exclude_none=False,
                    by_alias=False,
                )
            except Exception:
                pass
        dict_method = getattr(obj, "dict", None)
        if callable(dict_method):
            try:
                return dict_method()
            except Exception:
                pass
        return str(obj)

    def _get(self, obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _as_list(self, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    def _enum_value(self, value: Any) -> str:
        if value is None:
            return ""
        return str(getattr(value, "value", value))

    def _count_learning_steps(self, learning_path: Any) -> int:
        if learning_path is None:
            return 0
        steps = self._get(learning_path, "steps", []) or []
        return len(self._as_list(steps))

    def _to_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _to_int_optional(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    def _to_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _to_float_optional(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _iso(self, value: Any) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, str) and value.strip():
            return value
        return self._now()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()