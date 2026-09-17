from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

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
    source_id TEXT,
    title TEXT,
    url TEXT,
    platform TEXT,
    source_type TEXT,
    difficulty TEXT,
    rank INTEGER,
    score REAL,
    confidence REAL
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
    goal TEXT,
    level TEXT,
    topic TEXT,
    status TEXT,
    total_iterations INTEGER,
    total_tokens_used INTEGER,
    total_budget INTEGER,
    tokens_remaining INTEGER,
    findings_count INTEGER,
    ranked_count INTEGER,
    summaries_count INTEGER,
    learning_steps_count INTEGER,
    errors TEXT,
    warnings TEXT,
    saved_files TEXT,
    started_at TEXT,
    finished_at TEXT,
    latency_ms REAL,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS agent_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id TEXT UNIQUE,
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
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT UNIQUE,
    task TEXT,
    task_type TEXT,
    goal TEXT,
    level TEXT,
    strategy_used TEXT,
    actions_taken TEXT,
    tools_used TEXT,
    tokens_used INTEGER,
    tokens_by_category TEXT,
    total_iterations INTEGER,
    outcome TEXT,
    quality_score REAL,
    success INTEGER,
    errors TEXT,
    findings_count INTEGER,
    started_at TEXT,
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


class SQLiteStore:
    def __init__(self, database_path: Path | str) -> None:
        self._database_path = Path(database_path).expanduser()
        self._logger = get_logger("storage.sqlite_store")

    @property
    def database_path(self) -> Path:
        return self._database_path

    async def initialize(self) -> None:
        await run_sync_in_executor(self._ensure_schema_sync)

    def _ensure_schema_sync(self) -> None:
        try:
            self._database_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(str(self._database_path))
            try:
                connection.executescript(_SCHEMA)
                connection.commit()
            finally:
                connection.close()
        except Exception as exc:
            raise StorageError(
                "Failed to initialize SQLite schema",
                details={"path": str(self._database_path), "error": str(exc)},
            ) from exc

    async def save_state(self, state: Any) -> None:
        await run_sync_in_executor(self._save_state_sync, state)

    def _save_state_sync(self, state: Any) -> None:
        self._ensure_schema_sync()
        connection = sqlite3.connect(str(self._database_path))
        try:
            cursor = connection.cursor()
            request_id = str(getattr(state, "request_id", ""))
            if not request_id:
                raise StorageError("State is missing request_id")

            query = getattr(state, "query", None)
            topic = getattr(query, "topic", "") if query else ""
            goal = getattr(query, "goal", "") if query else ""
            level = getattr(query, "level", "") if query else ""
            status = self._enum_value(getattr(state, "status", ""))
            sources = getattr(state, "sources", []) or []
            ranked_sources = getattr(state, "ranked_sources", []) or []
            learning_path = getattr(state, "learning_path", None)
            steps = getattr(learning_path, "steps", []) if learning_path else []
            created_at = self._iso(getattr(state, "created_at", None))
            finished_at = self._iso(getattr(state, "finished_at", None))
            latency_ms = getattr(state, "latency_ms", None)

            payload = json.dumps(
                {
                    "errors": getattr(state, "errors", []) or [],
                    "expanded_queries": getattr(state, "expanded_queries", []) or [],
                },
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
                    len(sources),
                    len(ranked_sources),
                    len(steps),
                    latency_ms,
                    created_at,
                    finished_at,
                    payload,
                ),
            )

            cursor.execute("DELETE FROM sources WHERE request_id = ?", (request_id,))
            for ranked in ranked_sources:
                source = getattr(ranked, "source", None)
                cursor.execute(
                    """
                    INSERT INTO sources (
                        request_id, source_id, title, url, platform,
                        source_type, difficulty, rank, score, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request_id,
                        getattr(source, "source_id", "") if source else "",
                        getattr(source, "title", "") if source else "",
                        getattr(source, "url", "") if source else "",
                        self._enum_value(getattr(source, "platform", "")) if source else "",
                        self._enum_value(getattr(source, "source_type", "")) if source else "",
                        self._enum_value(getattr(source, "difficulty", "")) if source else "",
                        getattr(ranked, "rank", None),
                        getattr(ranked, "score", None),
                        getattr(ranked, "confidence", None),
                    ),
                )

            cursor.execute(
                "DELETE FROM learning_steps WHERE request_id = ?",
                (request_id,),
            )
            for step in steps:
                resources = getattr(step, "resources", []) or []
                resource_ids = []
                for resource in resources:
                    resource_source = getattr(resource, "source", None)
                    if resource_source is not None:
                        resource_ids.append(getattr(resource_source, "source_id", ""))
                cursor.execute(
                    """
                    INSERT INTO learning_steps (
                        request_id, step, title, objective,
                        estimated_minutes, resources
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request_id,
                        getattr(step, "step", None),
                        getattr(step, "title", ""),
                        getattr(step, "objective", ""),
                        getattr(step, "estimated_minutes", None),
                        json.dumps(resource_ids, ensure_ascii=False, default=str),
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
        self._ensure_schema_sync()
        connection = sqlite3.connect(str(self._database_path))
        try:
            cursor = connection.cursor()
            agent_id = str(getattr(result, "agent_id", ""))
            if not agent_id:
                raise StorageError("Agent result is missing agent_id")

            cursor.execute(
                """
                INSERT OR REPLACE INTO agent_runs (
                    agent_id, goal, level, topic, status,
                    total_iterations, total_tokens_used, total_budget,
                    tokens_remaining, findings_count, ranked_count,
                    summaries_count, learning_steps_count,
                    errors, warnings, saved_files,
                    started_at, finished_at, latency_ms, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    agent_id,
                    getattr(result, "goal", ""),
                    getattr(result, "level", ""),
                    getattr(result, "topic", ""),
                    self._enum_value(getattr(result, "status", "")),
                    getattr(result, "total_iterations", 0),
                    getattr(result, "total_tokens_used", 0),
                    getattr(result, "total_budget", 0),
                    getattr(result, "tokens_remaining", 0),
                    len(getattr(result, "findings", []) or []),
                    len(getattr(result, "ranked_sources", []) or []),
                    len(getattr(result, "source_summaries", []) or []),
                    len(getattr(result.learning_path, "steps", []) if getattr(result, "learning_path", None) else []),
                    json.dumps(getattr(result, "errors", []) or [], ensure_ascii=False),
                    json.dumps(getattr(result, "warnings", []) or [], ensure_ascii=False),
                    json.dumps(getattr(result, "saved_files", {}) or {}, ensure_ascii=False),
                    self._iso(getattr(result, "started_at", None)),
                    self._iso(getattr(result, "finished_at", None)),
                    getattr(result, "latency_ms", None),
                    json.dumps(
                        {"actions_taken": len(getattr(result, "actions_taken", []) or [])},
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

    async def save_memory_entry(self, entry: Any) -> None:
        await run_sync_in_executor(self._save_memory_entry_sync, entry)

    def _save_memory_entry_sync(self, entry: Any) -> None:
        self._ensure_schema_sync()
        connection = sqlite3.connect(str(self._database_path))
        try:
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO agent_memory (
                    entry_id, memory_type, key, content, metadata,
                    embedding, relevance_score, access_count,
                    created_at, last_accessed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    getattr(entry, "entry_id", ""),
                    getattr(entry, "memory_type", "episodic"),
                    getattr(entry, "key", ""),
                    getattr(entry, "content", ""),
                    json.dumps(getattr(entry, "metadata", {}) or {}, ensure_ascii=False),
                    json.dumps(getattr(entry, "embedding", []) or [], ensure_ascii=False),
                    getattr(entry, "relevance_score", 0.0),
                    getattr(entry, "access_count", 0),
                    self._iso(getattr(entry, "created_at", None)),
                    self._iso(getattr(entry, "last_accessed_at", None)),
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

    async def recall_memory(self, key: str, memory_type: str = "") -> list[dict[str, Any]]:
        return await run_sync_in_executor(self._recall_memory_sync, key, memory_type)

    def _recall_memory_sync(self, key: str, memory_type: str) -> list[dict[str, Any]]:
        self._ensure_schema_sync()
        connection = sqlite3.connect(str(self._database_path))
        try:
            connection.row_factory = sqlite3.Row
            cursor = connection.cursor()

            if memory_type:
                cursor.execute(
                    "SELECT * FROM agent_memory WHERE key LIKE ? AND memory_type = ? ORDER BY created_at DESC LIMIT 20",
                    (f"%{key}%", memory_type),
                )
            else:
                cursor.execute(
                    "SELECT * FROM agent_memory WHERE key LIKE ? ORDER BY created_at DESC LIMIT 20",
                    (f"%{key}%",),
                )

            rows = cursor.fetchall()
            results = []
            for row in rows:
                item = dict(row)
                try:
                    item["metadata"] = json.loads(item.get("metadata", "{}"))
                except Exception:
                    item["metadata"] = {}
                try:
                    item["embedding"] = json.loads(item.get("embedding", "[]"))
                except Exception:
                    item["embedding"] = []
                results.append(item)

            if results:
                entry_ids = [r["entry_id"] for r in results]
                placeholders = ",".join("?" * len(entry_ids))
                cursor.execute(
                    f"UPDATE agent_memory SET access_count = access_count + 1, last_accessed_at = datetime('now') WHERE entry_id IN ({placeholders})",
                    entry_ids,
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
        self._ensure_schema_sync()
        connection = sqlite3.connect(str(self._database_path))
        try:
            cursor = connection.cursor()
            episode_id = getattr(episode, "episode_id", "") or str(id(episode))

            actions = getattr(episode, "actions_taken", []) or []
            tools = getattr(episode, "tools_used", []) or []

            cursor.execute(
                """
                INSERT OR REPLACE INTO agent_episodes (
                    episode_id, task, task_type, goal, level,
                    strategy_used, actions_taken, tools_used,
                    tokens_used, tokens_by_category, total_iterations,
                    outcome, quality_score, success, errors,
                    findings_count, started_at, finished_at, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    getattr(episode, "task", ""),
                    getattr(episode, "task_type", ""),
                    getattr(episode, "goal", ""),
                    getattr(episode, "level", ""),
                    getattr(episode, "strategy_used", ""),
                    json.dumps(actions, ensure_ascii=False, default=str),
                    json.dumps(tools, ensure_ascii=False, default=str),
                    getattr(episode, "tokens_used", 0),
                    json.dumps(getattr(episode, "tokens_by_category", {}) or {}, ensure_ascii=False),
                    getattr(episode, "total_iterations", 0),
                    getattr(episode, "outcome", ""),
                    getattr(episode, "quality_score", 0.0),
                    1 if getattr(episode, "success", False) else 0,
                    json.dumps(getattr(episode, "errors", []) or [], ensure_ascii=False),
                    getattr(episode, "findings_count", 0),
                    getattr(episode, "started_at", ""),
                    getattr(episode, "finished_at", ""),
                    getattr(episode, "latency_ms", None),
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

    async def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        return await run_sync_in_executor(self._list_runs_sync, limit)

    def _list_runs_sync(self, limit: int) -> list[dict[str, Any]]:
        self._ensure_schema_sync()
        connection = sqlite3.connect(str(self._database_path))
        try:
            connection.row_factory = sqlite3.Row
            cursor = connection.cursor()
            cursor.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?",
                (max(1, int(limit)),),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as exc:
            raise StorageError(
                "Failed to list runs",
                details={"error": str(exc)},
            ) from exc
        finally:
            connection.close()

    async def list_agent_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        return await run_sync_in_executor(self._list_agent_runs_sync, limit)

    def _list_agent_runs_sync(self, limit: int) -> list[dict[str, Any]]:
        self._ensure_schema_sync()
        connection = sqlite3.connect(str(self._database_path))
        try:
            connection.row_factory = sqlite3.Row
            cursor = connection.cursor()
            cursor.execute(
                "SELECT * FROM agent_runs ORDER BY started_at DESC LIMIT ?",
                (max(1, int(limit)),),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as exc:
            raise StorageError(
                "Failed to list agent runs",
                details={"error": str(exc)},
            ) from exc
        finally:
            connection.close()

    async def get_run(self, request_id: str) -> dict[str, Any] | None:
        return await run_sync_in_executor(self._get_run_sync, request_id)

    def _get_run_sync(self, request_id: str) -> dict[str, Any] | None:
        self._ensure_schema_sync()
        connection = sqlite3.connect(str(self._database_path))
        try:
            connection.row_factory = sqlite3.Row
            cursor = connection.cursor()
            cursor.execute(
                "SELECT * FROM runs WHERE request_id = ?",
                (request_id,),
            )
            run_row = cursor.fetchone()
            if run_row is None:
                return None

            result = dict(run_row)

            cursor.execute(
                "SELECT * FROM sources WHERE request_id = ? ORDER BY rank",
                (request_id,),
            )
            result["sources"] = [dict(row) for row in cursor.fetchall()]

            cursor.execute(
                "SELECT * FROM learning_steps WHERE request_id = ? ORDER BY step",
                (request_id,),
            )
            result["learning_steps"] = [dict(row) for row in cursor.fetchall()]

            return result
        except Exception as exc:
            raise StorageError(
                "Failed to get run",
                details={"request_id": request_id, "error": str(exc)},
            ) from exc
        finally:
            connection.close()

    async def get_agent_run(self, agent_id: str) -> dict[str, Any] | None:
        return await run_sync_in_executor(self._get_agent_run_sync, agent_id)

    def _get_agent_run_sync(self, agent_id: str) -> dict[str, Any] | None:
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
            return dict(row)
        except Exception as exc:
            raise StorageError(
                "Failed to get agent run",
                details={"agent_id": agent_id, "error": str(exc)},
            ) from exc
        finally:
            connection.close()

    @staticmethod
    def _enum_value(value: Any) -> str:
        if value is None:
            return ""
        return str(getattr(value, "value", value))

    @staticmethod
    def _iso(value: Any) -> str | None:
        if value is None:
            return None
        try:
            return value.isoformat()
        except Exception:
            return str(value)