from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import Field

from core.config import get_data_directory
from core.models import CoreModel
from utils.logger import get_logger
from utils.text import clean_text

_MAX_TURNS = 200
_CONTEXT_TURNS = 6
_SAVE_EVERY = 1

_TOPIC_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "what", "how", "why",
    "when", "where", "which", "who", "whom", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "do", "does", "did",
    "doing", "have", "has", "had", "having", "can", "could", "should", "would",
    "may", "might", "must", "will", "shall", "not", "no", "yes", "about",
    "into", "over", "under", "between", "from", "as", "at", "by", "if",
    "then", "than", "too", "very", "just", "also", "tell", "show", "give",
    "please", "explain", "describe", "recently", "currently", "now", "so",
    "far", "before", "earlier", "previously", "studying", "asked",
    "mentioned", "discussed", "learned", "covered", "explored", "reviewed",
})


class ConversationTurn(CoreModel):
    role: str = "user"
    content: str = ""
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationState(CoreModel):
    session_id: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    turns: list[ConversationTurn] = Field(default_factory=list)


class Conversation:
    def __init__(
        self,
        session_id: str | None = None,
        base_dir: Path | str | None = None,
    ) -> None:
        self._logger = get_logger("rag.conversation")
        self._lock = threading.RLock()
        self._base_dir = (
            Path(base_dir)
            if base_dir is not None
            else get_data_directory() / "conversations"
        )
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._session_id = (
            str(session_id).strip() if session_id else self._new_session_id()
        )
        self._state = self._load_or_create()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def path(self) -> Path:
        return self._path_for(self._session_id)

    def add_turn(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationTurn:
        cleaned_role = str(role or "user").strip().lower()
        if cleaned_role not in {"user", "assistant", "system"}:
            cleaned_role = "user"

        cleaned_content = clean_text(content)
        if not cleaned_content:
            return ConversationTurn(role=cleaned_role, content="")

        turn = ConversationTurn(
            role=cleaned_role,
            content=cleaned_content,
            metadata=dict(metadata or {}),
        )

        with self._lock:
            self._state.turns.append(turn)
            if len(self._state.turns) > _MAX_TURNS:
                self._state.turns = self._state.turns[-_MAX_TURNS:]
            self._state.updated_at = datetime.now(timezone.utc).isoformat()
            self._save_locked()

        return turn

    def get_turns(self, limit: int | None = None) -> list[ConversationTurn]:
        with self._lock:
            turns = list(self._state.turns)

        if limit is None or limit <= 0:
            return turns

        return turns[-limit:]

    def get_last_user_topic(self) -> str:
        with self._lock:
            for turn in reversed(self._state.turns):
                if turn.role != "user":
                    continue
                text = clean_text(turn.content)
                if text:
                    return text
        return ""

    def get_last_user_metadata(self) -> dict[str, Any]:
        with self._lock:
            for turn in reversed(self._state.turns):
                if turn.role == "user":
                    return dict(turn.metadata or {})
        return {}

    def build_context_window(
        self,
        max_turns: int = _CONTEXT_TURNS,
    ) -> list[dict[str, str]]:
        with self._lock:
            turns = list(self._state.turns)

        if not turns:
            return []

        trimmed = turns[-max(1, int(max_turns)) :]

        return [
            {"role": turn.role, "content": turn.content}
            for turn in trimmed
            if turn.content
        ]

    def recent_user_terms(self, limit: int = 5) -> list[str]:
        with self._lock:
            turns = list(self._state.turns)

        terms: list[str] = []

        for turn in reversed(turns):
            if turn.role != "user":
                continue
            content = clean_text(turn.content)
            if content and content not in terms:
                terms.append(content)
            if len(terms) >= max(1, int(limit)):
                break

        return list(reversed(terms))

    def recent_topic_summary(
        self,
        limit: int = 10,
        similarity_threshold: float = 0.75,
    ) -> list[dict[str, Any]]:
        with self._lock:
            turns = list(self._state.turns)

        user_turns: list[tuple[str, str]] = []

        for turn in turns:
            if turn.role != "user":
                continue

            content = clean_text(turn.content)

            if not content:
                continue

            user_turns.append((content, str(turn.timestamp or "")))

        if not user_turns:
            return []

        clusters: list[dict[str, Any]] = []
        threshold = max(0.30, min(1.0, float(similarity_threshold)))

        for content, timestamp in user_turns:
            normalized = self._normalize_topic(content)
            tokens = self._topic_tokens(normalized)

            if not tokens:
                continue

            placed = False

            for cluster in clusters:
                similarity = self._jaccard(tokens, cluster["tokens"])

                if similarity >= threshold:
                    cluster["count"] += 1
                    cluster["last_seen"] = timestamp or cluster["last_seen"]
                    cluster["examples"].append(content)
                    placed = True
                    break

            if not placed:
                clusters.append(
                    {
                        "topic": content,
                        "normalized": normalized,
                        "tokens": tokens,
                        "first_seen": timestamp,
                        "last_seen": timestamp,
                        "count": 1,
                        "examples": [content],
                    }
                )

        clusters.sort(
            key=lambda c: (c["last_seen"] or "", c["count"]),
            reverse=True,
        )

        result: list[dict[str, Any]] = []

        for cluster in clusters[: max(1, int(limit))]:
            result.append(
                {
                    "topic": cluster["topic"],
                    "normalized": cluster["normalized"],
                    "first_seen": cluster["first_seen"],
                    "last_seen": cluster["last_seen"],
                    "count": cluster["count"],
                    "examples": cluster["examples"][:3],
                }
            )

        return result

    def clear(self) -> None:
        with self._lock:
            self._state = ConversationState(session_id=self._session_id)
            self._save_locked()

    def save(self) -> None:
        with self._lock:
            self._save_locked()

    def export(self, path: Path | str) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = self._state.model_dump(mode="json")
        self._atomic_write(target, payload)
        return target

    def _normalize_topic(self, text: str) -> str:
        cleaned = clean_text(text).lower()
        cleaned = re.sub(r"[^\w\s]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        words = [
            word
            for word in cleaned.split()
            if word not in _TOPIC_STOPWORDS and len(word) > 1
        ]

        return " ".join(words)

    def _topic_tokens(self, text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", str(text or "").lower()))

    def _jaccard(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0

        intersection = len(left & right)
        union = len(left | right)

        return intersection / union if union > 0 else 0.0

    def _load_or_create(self) -> ConversationState:
        path = self._path_for(self._session_id)

        if not path.exists():
            return ConversationState(session_id=self._session_id)

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._logger.warning(
                f"Failed to read conversation {self._session_id}: {exc}"
            )
            return ConversationState(session_id=self._session_id)

        try:
            state = ConversationState.model_validate(raw)
            if not state.session_id:
                state.session_id = self._session_id
            return state
        except Exception as exc:
            self._logger.warning(
                f"Failed to validate conversation {self._session_id}: {exc}"
            )
            return ConversationState(session_id=self._session_id)

    def _save_locked(self) -> None:
        payload = self._state.model_dump(mode="json")
        self._atomic_write(self._path_for(self._session_id), payload)

    def _atomic_write(self, path: Path, payload: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            os.replace(temp, path)
        except Exception as exc:
            self._logger.warning(
                f"Failed to persist conversation {self._session_id}: {exc}"
            )

    def _path_for(self, session_id: str) -> Path:
        safe = "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "_"
            for ch in str(session_id or "default")
        )
        return self._base_dir / f"{safe or 'default'}.json"

    @staticmethod
    def _new_session_id() -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{stamp}_{uuid4().hex[:8]}"


def list_sessions(base_dir: Path | str | None = None) -> list[dict[str, Any]]:
    directory = (
        Path(base_dir)
        if base_dir is not None
        else get_data_directory() / "conversations"
    )

    if not directory.exists():
        return []

    sessions: list[dict[str, Any]] = []

    for path in sorted(
        directory.glob("*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(raw, dict):
            continue

        sessions.append(
            {
                "session_id": str(raw.get("session_id") or path.stem),
                "created_at": str(raw.get("created_at") or ""),
                "updated_at": str(raw.get("updated_at") or ""),
                "turn_count": len(raw.get("turns") or []),
                "path": str(path),
            }
        )

    return sessions