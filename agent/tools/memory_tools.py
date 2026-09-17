from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent.tools.registry import ToolRegistry, ToolDefinition
from agent.tools.executor import ToolResult
from core.config import get_data_directory
from core.models import MemoryEntry
from utils.logger import get_logger
from utils.text import clean_text


_logger = get_logger("agent.tools.memory")

_MEMORY_TIMEOUT = 15.0
_MEMORY_TOKEN_COST = 100
_MEMORY_FILE = "agent_memory.json"


def _get_memory_path() -> Path:
    return get_data_directory() / _MEMORY_FILE


def _load_memory_store() -> list[dict[str, Any]]:
    path = _get_memory_path()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return raw
        return []
    except Exception:
        return []


def _save_memory_store(entries: list[dict[str, Any]]) -> None:
    path = _get_memory_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temp_path.replace(path)


async def save_to_memory(
    key: str,
    content: str,
    memory_type: str = "episodic",
    metadata: dict[str, Any] | None = None,
) -> ToolResult:
    try:
        cleaned_key = clean_text(key)
        cleaned_content = clean_text(content)

        if not cleaned_key:
            return ToolResult(
                tool_name="save_to_memory",
                success=False,
                data=None,
                error="Memory key cannot be empty",
            )

        if not cleaned_content:
            return ToolResult(
                tool_name="save_to_memory",
                success=False,
                data=None,
                error="Memory content cannot be empty",
            )

        entries = _load_memory_store()

        existing_index = None
        for i, entry in enumerate(entries):
            if entry.get("key") == cleaned_key:
                existing_index = i
                break

        entry_data = {
            "entry_id": str(uuid4()),
            "memory_type": memory_type,
            "key": cleaned_key,
            "content": cleaned_content,
            "metadata": metadata or {},
            "access_count": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_accessed_at": None,
        }

        if existing_index is not None:
            entry_data["entry_id"] = entries[existing_index].get("entry_id", str(uuid4()))
            entry_data["access_count"] = entries[existing_index].get("access_count", 0)
            entry_data["created_at"] = entries[existing_index].get("created_at", entry_data["created_at"])
            entries[existing_index] = entry_data
        else:
            entries.append(entry_data)

        _save_memory_store(entries)

        return ToolResult(
            tool_name="save_to_memory",
            success=True,
            data={
                "entry_id": entry_data["entry_id"],
                "key": cleaned_key,
                "action": "updated" if existing_index is not None else "created",
            },
            tokens_used=0,
        )
    except Exception as exc:
        _logger.warning(f"save_to_memory failed: {exc}")
        return ToolResult(
            tool_name="save_to_memory",
            success=False,
            data=None,
            error=str(exc),
        )


async def recall_memory(
    key: str,
) -> ToolResult:
    try:
        cleaned_key = clean_text(key)
        if not cleaned_key:
            return ToolResult(
                tool_name="recall_memory",
                success=False,
                data=None,
                error="Memory key cannot be empty",
            )

        entries = _load_memory_store()

        for i, entry in enumerate(entries):
            if entry.get("key") == cleaned_key:
                entry["access_count"] = entry.get("access_count", 0) + 1
                entry["last_accessed_at"] = datetime.now(timezone.utc).isoformat()
                entries[i] = entry
                _save_memory_store(entries)

                return ToolResult(
                    tool_name="recall_memory",
                    success=True,
                    data={
                        "entry_id": entry.get("entry_id"),
                        "key": entry.get("key"),
                        "content": entry.get("content"),
                        "memory_type": entry.get("memory_type"),
                        "metadata": entry.get("metadata", {}),
                        "access_count": entry.get("access_count"),
                        "created_at": entry.get("created_at"),
                    },
                    tokens_used=0,
                )

        return ToolResult(
            tool_name="recall_memory",
            success=False,
            data=None,
            error=f"No memory found for key: {cleaned_key}",
        )
    except Exception as exc:
        _logger.warning(f"recall_memory failed: {exc}")
        return ToolResult(
            tool_name="recall_memory",
            success=False,
            data=None,
            error=str(exc),
        )


async def search_memory(
    query: str,
    top_k: int = 5,
    memory_type: str = "",
) -> ToolResult:
    try:
        cleaned_query = clean_text(query).lower()
        if not cleaned_query:
            return ToolResult(
                tool_name="search_memory",
                success=False,
                data=None,
                error="Search query cannot be empty",
            )

        entries = _load_memory_store()

        if memory_type:
            entries = [e for e in entries if e.get("memory_type") == memory_type]

        query_tokens = set(cleaned_query.split())
        scored_entries = []

        for entry in entries:
            key_text = str(entry.get("key", "")).lower()
            content_text = str(entry.get("content", "")).lower()
            entry_tokens = set(key_text.split()) | set(content_text.split())

            overlap = len(query_tokens & entry_tokens)
            if overlap > 0:
                score = overlap / max(len(query_tokens), 1)
                scored_entries.append((score, entry))

        scored_entries.sort(key=lambda x: x[0], reverse=True)
        top_entries = scored_entries[:max(1, top_k)]

        results = []
        for score, entry in top_entries:
            results.append({
                "entry_id": entry.get("entry_id"),
                "key": entry.get("key"),
                "content_preview": str(entry.get("content", ""))[:500],
                "memory_type": entry.get("memory_type"),
                "relevance_score": round(score, 4),
            })

        return ToolResult(
            tool_name="search_memory",
            success=True,
            data={
                "query": cleaned_query,
                "total_matches": len(scored_entries),
                "results": results,
            },
            tokens_used=0,
        )
    except Exception as exc:
        _logger.warning(f"search_memory failed: {exc}")
        return ToolResult(
            tool_name="search_memory",
            success=False,
            data=None,
            error=str(exc),
        )


async def list_memory(
    memory_type: str = "",
    limit: int = 20,
) -> ToolResult:
    try:
        entries = _load_memory_store()

        if memory_type:
            entries = [e for e in entries if e.get("memory_type") == memory_type]

        entries = entries[:max(1, limit)]

        results = []
        for entry in entries:
            results.append({
                "entry_id": entry.get("entry_id"),
                "key": entry.get("key"),
                "memory_type": entry.get("memory_type"),
                "content_preview": str(entry.get("content", ""))[:200],
                "access_count": entry.get("access_count", 0),
                "created_at": entry.get("created_at"),
            })

        return ToolResult(
            tool_name="list_memory",
            success=True,
            data={
                "total_entries": len(results),
                "entries": results,
            },
            tokens_used=0,
        )
    except Exception as exc:
        _logger.warning(f"list_memory failed: {exc}")
        return ToolResult(
            tool_name="list_memory",
            success=False,
            data=None,
            error=str(exc),
        )


def register_memory_tools(registry: ToolRegistry) -> None:
    registry.register(
        ToolDefinition(
            name="save_to_memory",
            description="Save information to persistent agent memory",
            category="memory",
            token_cost_est=_MEMORY_TOKEN_COST,
            timeout_seconds=_MEMORY_TIMEOUT,
            requires_llm=False,
        ),
        save_to_memory,
    )

    registry.register(
        ToolDefinition(
            name="recall_memory",
            description="Recall specific information from agent memory by key",
            category="memory",
            token_cost_est=_MEMORY_TOKEN_COST,
            timeout_seconds=_MEMORY_TIMEOUT,
            requires_llm=False,
        ),
        recall_memory,
    )

    registry.register(
        ToolDefinition(
            name="search_memory",
            description="Search agent memory for relevant past information",
            category="memory",
            token_cost_est=500,
            timeout_seconds=_MEMORY_TIMEOUT,
            requires_llm=False,
        ),
        search_memory,
    )

    registry.register(
        ToolDefinition(
            name="list_memory",
            description="List all entries in agent memory",
            category="memory",
            token_cost_est=_MEMORY_TOKEN_COST,
            timeout_seconds=_MEMORY_TIMEOUT,
            requires_llm=False,
        ),
        list_memory,
    )