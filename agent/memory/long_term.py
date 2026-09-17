from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import Field

from core.config import get_data_directory
from core.exceptions import ResearchAgentError
from core.models import CoreModel
from utils.logger import get_logger


MEMORY_DB_FILENAME = "agent_memory.db"

TABLE_TOPICS = "memory_topics"
TABLE_STRATEGIES = "memory_strategies"
TABLE_PREFERENCES = "memory_preferences"
TABLE_KNOWLEDGE = "memory_knowledge"

_SCHEMA_TOPICS = f"""
CREATE TABLE IF NOT EXISTS {TABLE_TOPICS} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    goal TEXT DEFAULT '',
    level TEXT DEFAULT '',
    outcome TEXT DEFAULT '',
    quality_score REAL DEFAULT 0.0,
    sources_used INTEGER DEFAULT 0,
    tokens_used INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{{}}',
    created_at TEXT NOT NULL,
    last_accessed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_topics_topic ON {TABLE_TOPICS}(topic);
CREATE INDEX IF NOT EXISTS idx_topics_quality ON {TABLE_TOPICS}(quality_score);
"""

_SCHEMA_STRATEGIES = f"""
CREATE TABLE IF NOT EXISTS {TABLE_STRATEGIES} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    task_type TEXT DEFAULT '',
    success_rate REAL DEFAULT 0.0,
    total_uses INTEGER DEFAULT 0,
    total_successes INTEGER DEFAULT 0,
    avg_quality REAL DEFAULT 0.0,
    avg_tokens INTEGER DEFAULT 0,
    config TEXT DEFAULT '{{}}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_strategies_name ON {TABLE_STRATEGIES}(name);
CREATE INDEX IF NOT EXISTS idx_strategies_task ON {TABLE_STRATEGIES}(task_type);
"""

_SCHEMA_PREFERENCES = f"""
CREATE TABLE IF NOT EXISTS {TABLE_PREFERENCES} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    value TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_preferences_key ON {TABLE_PREFERENCES}(key);
"""

_SCHEMA_KNOWLEDGE = f"""
CREATE TABLE IF NOT EXISTS {TABLE_KNOWLEDGE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT DEFAULT '',
    category TEXT DEFAULT 'general',
    tags TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{{}}',
    access_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    last_accessed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_knowledge_key ON {TABLE_KNOWLEDGE}(key);
CREATE INDEX IF NOT EXISTS idx_knowledge_category ON {TABLE_KNOWLEDGE}(category);
"""


class MemoryError_(ResearchAgentError):
    pass


class LongTermMemory:
    def __init__(self, database_path: str | Path | None = None) -> None:
        self._lock = threading.Lock()
        self._logger = get_logger("agent.memory.long_term")

        if database_path is not None:
            self._db_path = Path(database_path).expanduser()
        else:
            self._db_path = get_data_directory() / MEMORY_DB_FILENAME

        self._initialize_schema()

    @property
    def database_path(self) -> Path:
        return self._db_path

    def _initialize_schema(self) -> None:
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(str(self._db_path))
            try:
                connection.executescript(_SCHEMA_TOPICS)
                connection.executescript(_SCHEMA_STRATEGIES)
                connection.executescript(_SCHEMA_PREFERENCES)
                connection.executescript(_SCHEMA_KNOWLEDGE)
                connection.commit()
            finally:
                connection.close()
        except Exception as exc:
            self._logger.warning(f"Failed to initialize memory schema: {exc}")

    def _get_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self._db_path))
        connection.row_factory = sqlite3.Row
        return connection

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def save_topic(
        self,
        topic: str,
        goal: str = "",
        level: str = "",
        outcome: str = "",
        quality_score: float = 0.0,
        sources_used: int = 0,
        tokens_used: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        with self._lock:
            try:
                connection = self._get_connection()
                try:
                    cursor = connection.cursor()
                    cursor.execute(
                        f"""
                        INSERT INTO {TABLE_TOPICS}
                        (topic, goal, level, outcome, quality_score,
                         sources_used, tokens_used, metadata, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            topic,
                            goal,
                            level,
                            outcome,
                            quality_score,
                            sources_used,
                            tokens_used,
                            json.dumps(metadata or {}, ensure_ascii=False),
                            self._now(),
                        ),
                    )
                    connection.commit()
                    return cursor.lastrowid or 0
                finally:
                    connection.close()
            except Exception as exc:
                self._logger.warning(f"Failed to save topic: {exc}")
                return 0

    def recall_topics(
        self,
        query: str = "",
        limit: int = 10,
        min_quality: float = 0.0,
    ) -> list[dict[str, Any]]:
        with self._lock:
            try:
                connection = self._get_connection()
                try:
                    cursor = connection.cursor()
                    if query:
                        cursor.execute(
                            f"""
                            SELECT * FROM {TABLE_TOPICS}
                            WHERE topic LIKE ? AND quality_score >= ?
                            ORDER BY quality_score DESC, created_at DESC
                            LIMIT ?
                            """,
                            (f"%{query}%", min_quality, limit),
                        )
                    else:
                        cursor.execute(
                            f"""
                            SELECT * FROM {TABLE_TOPICS}
                            WHERE quality_score >= ?
                            ORDER BY quality_score DESC, created_at DESC
                            LIMIT ?
                            """,
                            (min_quality, limit),
                        )
                    rows = cursor.fetchall()
                    results = []
                    for row in rows:
                        item = dict(row)
                        item["metadata"] = self._parse_json(item.get("metadata", "{}"))
                        results.append(item)
                    return results
                finally:
                    connection.close()
            except Exception as exc:
                self._logger.warning(f"Failed to recall topics: {exc}")
                return []

    def save_strategy(
        self,
        name: str,
        description: str = "",
        task_type: str = "",
        config: dict[str, Any] | None = None,
    ) -> int:
        with self._lock:
            try:
                connection = self._get_connection()
                try:
                    cursor = connection.cursor()
                    now = self._now()
                    cursor.execute(
                        f"""
                        INSERT INTO {TABLE_STRATEGIES}
                        (name, description, task_type, config, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            name,
                            description,
                            task_type,
                            json.dumps(config or {}, ensure_ascii=False),
                            now,
                            now,
                        ),
                    )
                    connection.commit()
                    return cursor.lastrowid or 0
                finally:
                    connection.close()
            except Exception as exc:
                self._logger.warning(f"Failed to save strategy: {exc}")
                return 0

    def record_strategy_result(
        self,
        strategy_id: int,
        success: bool,
        quality_score: float = 0.0,
        tokens_used: int = 0,
    ) -> None:
        with self._lock:
            try:
                connection = self._get_connection()
                try:
                    cursor = connection.cursor()
                    cursor.execute(
                        f"SELECT * FROM {TABLE_STRATEGIES} WHERE id = ?",
                        (strategy_id,),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        return

                    current_uses = row["total_uses"] + 1
                    current_successes = row["total_successes"] + (1 if success else 0)
                    success_rate = current_successes / current_uses if current_uses > 0 else 0.0

                    old_avg_quality = row["avg_quality"]
                    old_avg_tokens = row["avg_tokens"]
                    new_avg_quality = (
                        (old_avg_quality * (current_uses - 1) + quality_score) / current_uses
                    )
                    new_avg_tokens = int(
                        (old_avg_tokens * (current_uses - 1) + tokens_used) / current_uses
                    )

                    cursor.execute(
                        f"""
                        UPDATE {TABLE_STRATEGIES}
                        SET total_uses = ?, total_successes = ?, success_rate = ?,
                            avg_quality = ?, avg_tokens = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (
                            current_uses,
                            current_successes,
                            success_rate,
                            new_avg_quality,
                            new_avg_tokens,
                            self._now(),
                            strategy_id,
                        ),
                    )
                    connection.commit()
                finally:
                    connection.close()
            except Exception as exc:
                self._logger.warning(f"Failed to record strategy result: {exc}")

    def get_best_strategy(self, task_type: str = "", limit: int = 3) -> list[dict[str, Any]]:
        with self._lock:
            try:
                connection = self._get_connection()
                try:
                    cursor = connection.cursor()
                    if task_type:
                        cursor.execute(
                            f"""
                            SELECT * FROM {TABLE_STRATEGIES}
                            WHERE task_type = ? AND total_uses > 0
                            ORDER BY success_rate DESC, avg_quality DESC
                            LIMIT ?
                            """,
                            (task_type, limit),
                        )
                    else:
                        cursor.execute(
                            f"""
                            SELECT * FROM {TABLE_STRATEGIES}
                            WHERE total_uses > 0
                            ORDER BY success_rate DESC, avg_quality DESC
                            LIMIT ?
                            """,
                            (limit,),
                        )
                    rows = cursor.fetchall()
                    results = []
                    for row in rows:
                        item = dict(row)
                        item["config"] = self._parse_json(item.get("config", "{}"))
                        results.append(item)
                    return results
                finally:
                    connection.close()
            except Exception as exc:
                self._logger.warning(f"Failed to get strategies: {exc}")
                return []

    def save_preference(self, key: str, value: str, category: str = "general") -> None:
        with self._lock:
            try:
                connection = self._get_connection()
                try:
                    cursor = connection.cursor()
                    now = self._now()
                    cursor.execute(
                        f"""
                        INSERT OR REPLACE INTO {TABLE_PREFERENCES}
                        (key, value, category, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (key, value, category, now, now),
                    )
                    connection.commit()
                finally:
                    connection.close()
            except Exception as exc:
                self._logger.warning(f"Failed to save preference: {exc}")

    def get_preference(self, key: str, default: str = "") -> str:
        with self._lock:
            try:
                connection = self._get_connection()
                try:
                    cursor = connection.cursor()
                    cursor.execute(
                        f"SELECT value FROM {TABLE_PREFERENCES} WHERE key = ?",
                        (key,),
                    )
                    row = cursor.fetchone()
                    return row["value"] if row else default
                finally:
                    connection.close()
            except Exception as exc:
                self._logger.warning(f"Failed to get preference: {exc}")
                return default

    def get_all_preferences(self, category: str = "") -> dict[str, str]:
        with self._lock:
            try:
                connection = self._get_connection()
                try:
                    cursor = connection.cursor()
                    if category:
                        cursor.execute(
                            f"SELECT key, value FROM {TABLE_PREFERENCES} WHERE category = ?",
                            (category,),
                        )
                    else:
                        cursor.execute(f"SELECT key, value FROM {TABLE_PREFERENCES}")
                    rows = cursor.fetchall()
                    return {row["key"]: row["value"] for row in rows}
                finally:
                    connection.close()
            except Exception as exc:
                self._logger.warning(f"Failed to get preferences: {exc}")
                return {}

    def save_knowledge(
        self,
        key: str,
        content: str,
        source: str = "",
        category: str = "general",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        with self._lock:
            try:
                connection = self._get_connection()
                try:
                    cursor = connection.cursor()
                    cursor.execute(
                        f"""
                        INSERT INTO {TABLE_KNOWLEDGE}
                        (key, content, source, category, tags, metadata, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            key,
                            content,
                            source,
                            category,
                            json.dumps(tags or [], ensure_ascii=False),
                            json.dumps(metadata or {}, ensure_ascii=False),
                            self._now(),
                        ),
                    )
                    connection.commit()
                    return cursor.lastrowid or 0
                finally:
                    connection.close()
            except Exception as exc:
                self._logger.warning(f"Failed to save knowledge: {exc}")
                return 0

    def search_knowledge(
        self,
        query: str = "",
        category: str = "",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        with self._lock:
            try:
                connection = self._get_connection()
                try:
                    cursor = connection.cursor()
                    conditions = []
                    params: list[Any] = []

                    if query:
                        conditions.append("(key LIKE ? OR content LIKE ?)")
                        params.append(f"%{query}%")
                        params.append(f"%{query}%")

                    if category:
                        conditions.append("category = ?")
                        params.append(category)

                    where_clause = " AND ".join(conditions) if conditions else "1=1"
                    params.append(limit)

                    cursor.execute(
                        f"""
                        SELECT * FROM {TABLE_KNOWLEDGE}
                        WHERE {where_clause}
                        ORDER BY access_count DESC, created_at DESC
                        LIMIT ?
                        """,
                        params,
                    )
                    rows = cursor.fetchall()
                    results = []
                    for row in rows:
                        item = dict(row)
                        item["tags"] = self._parse_json(item.get("tags", "[]"))
                        item["metadata"] = self._parse_json(item.get("metadata", "{}"))
                        results.append(item)

                    if results:
                        ids = [item["id"] for item in results]
                        placeholders = ",".join("?" * len(ids))
                        cursor.execute(
                            f"""
                            UPDATE {TABLE_KNOWLEDGE}
                            SET access_count = access_count + 1, last_accessed_at = ?
                            WHERE id IN ({placeholders})
                            """,
                            [self._now()] + ids,
                        )
                        connection.commit()

                    return results
                finally:
                    connection.close()
            except Exception as exc:
                self._logger.warning(f"Failed to search knowledge: {exc}")
                return []

    def get_statistics(self) -> dict[str, int]:
        with self._lock:
            try:
                connection = self._get_connection()
                try:
                    cursor = connection.cursor()
                    stats: dict[str, int] = {}
                    for table in (TABLE_TOPICS, TABLE_STRATEGIES, TABLE_PREFERENCES, TABLE_KNOWLEDGE):
                        cursor.execute(f"SELECT COUNT(*) as cnt FROM {table}")
                        row = cursor.fetchone()
                        stats[table] = row["cnt"] if row else 0
                    return stats
                finally:
                    connection.close()
            except Exception as exc:
                self._logger.warning(f"Failed to get statistics: {exc}")
                return {}

    def _parse_json(self, value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except Exception:
            return {} if isinstance(value, str) and value.startswith("{") else []