from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from core.config import get_data_directory
from rag.conversation import Conversation
from rag.schemas import TopicProfileSchema
from utils.logger import get_logger
from utils.text import clean_text

_MAX_EXPANSIONS = 6
_ACRONYM_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]{1,7}\b")
_DEFINITION_TAIL_PATTERN = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_\-]{1,40})\s*\(([A-Z][A-Z0-9]{1,7})\)"
)
_DEFINITION_HEAD_PATTERN = re.compile(
    r"\b([A-Z][A-Z0-9]{1,7})\s*\(([A-Za-z][A-Za-z0-9_\- ]{1,60})\)"
)

_INTENT_TEMPLATES: dict[str, tuple[str, ...]] = {
    "definition": (
        "{topic} definition",
        "what is {topic}",
        "{topic} overview",
    ),
    "comparison": (
        "{topic} vs",
        "{topic} comparison",
        "difference between {topic}",
    ),
    "mechanism": (
        "how does {topic} work",
        "{topic} mechanism",
        "{topic} architecture",
    ),
    "causation": (
        "why {topic}",
        "causes of {topic}",
        "{topic} reasons",
    ),
    "enumeration": (
        "{topic} examples",
        "types of {topic}",
        "{topic} list",
    ),
    "timeline": (
        "{topic} history",
        "{topic} origins",
        "history of {topic}",
    ),
    "decision": (
        "{topic} pros and cons",
        "{topic} trade-offs",
        "should i use {topic}",
    ),
    "explanation": (
        "{topic} explained",
        "{topic} overview",
        "introduction to {topic}",
    ),
    "overview": (
        "{topic} overview",
        "{topic} introduction",
        "{topic} explained",
    ),
}


class QueryExpander:
    def __init__(
        self,
        max_expansions: int = _MAX_EXPANSIONS,
        cache_path: Path | str | None = None,
    ) -> None:
        self._max_expansions = max(1, int(max_expansions))
        self._logger = get_logger("rag.query_expander")
        self._lock = threading.RLock()
        self._cache_path = (
            Path(cache_path)
            if cache_path is not None
            else get_data_directory() / "acronym_cache.json"
        )
        self._cache = self._load_cache()

    def expand(
        self,
        question: str,
        profile: TopicProfileSchema | None = None,
        conversation: Conversation | None = None,
        store: Any | None = None,
    ) -> list[str]:
        cleaned = clean_text(question)

        if not cleaned:
            return []

        queries: list[str] = [cleaned]

        topic = profile.primary_topic if profile is not None else cleaned
        intent = profile.intent if profile is not None else "overview"
        acronyms = list(profile.acronyms) if profile is not None else []

        if not acronyms:
            acronyms = self._detect_acronyms(cleaned)

        for acronym in acronyms:
            long_form = self._resolve_acronym(acronym, store)
            if long_form:
                self._append(queries, long_form)

        if topic and topic.lower() != cleaned.lower():
            self._append(queries, topic)

        for template in _INTENT_TEMPLATES.get(intent, ()):
            if not topic:
                break
            self._append(queries, template.format(topic=topic))

        if conversation is not None:
            self._apply_pronoun_context(queries, cleaned, conversation)

        if profile is not None:
            for subtopic in profile.subtopics[:2]:
                if topic:
                    self._append(queries, f"{topic} {subtopic}")

        return queries[: self._max_expansions]

    def _apply_pronoun_context(
        self,
        queries: list[str],
        question: str,
        conversation: Conversation,
    ) -> None:
        tokens = re.findall(r"[a-zA-Z']+", question.lower())

        if not tokens:
            return

        has_pronoun = any(token in {
            "it", "its", "it's", "they", "them", "their", "theirs",
            "this", "that", "these", "those", "he", "she", "him",
            "her", "his", "hers",
        } for token in tokens[:3])

        if not has_pronoun:
            return

        last_topic = conversation.get_last_user_topic()

        if not last_topic or last_topic.lower() == question.lower():
            return

        self._append(queries, f"{last_topic} {question}")

    def _resolve_acronym(
        self,
        acronym: str,
        store: Any | None,
    ) -> str:
        normalized = str(acronym or "").strip().upper()

        if not normalized:
            return ""

        with self._lock:
            cached = self._cache.get(normalized)

        if cached:
            return str(cached)

        if store is None:
            return ""

        mined = self._mine_from_corpus(normalized, store)

        if not mined:
            return ""

        with self._lock:
            self._cache[normalized] = mined
            self._save_cache_locked()

        return mined

    def _mine_from_corpus(self, acronym: str, store: Any) -> str:
        method = getattr(store, "keyword_search", None)

        if not callable(method):
            return ""

        try:
            results = method(acronym, top_k=40)
        except Exception as exc:
            self._logger.warning(f"Acronym corpus scan failed: {exc}")
            return ""

        candidates: dict[str, int] = {}

        for result in results or []:
            document = getattr(result, "document", None)
            if document is None:
                continue

            for text in (
                str(getattr(document, "title", "") or ""),
                str(getattr(document, "text", "") or "")[:600],
            ):
                if not text:
                    continue

                for match in _DEFINITION_TAIL_PATTERN.finditer(text):
                    full = match.group(1)
                    code = match.group(2)
                    if code.upper() == acronym and full:
                        key = full.strip().lower()
                        candidates[key] = candidates.get(key, 0) + 1

                for match in _DEFINITION_HEAD_PATTERN.finditer(text):
                    code = match.group(1)
                    full = match.group(2)
                    if code.upper() == acronym and full:
                        key = full.strip().lower()
                        candidates[key] = candidates.get(key, 0) + 1

        if not candidates:
            return ""

        best = max(candidates.items(), key=lambda item: item[1])
        return best[0]

    def _detect_acronyms(self, question: str) -> list[str]:
        found: list[str] = []

        for match in _ACRONYM_PATTERN.findall(str(question or "")):
            token = match.strip()
            if token and token not in found:
                found.append(token)

        return found

    def _append(self, queries: list[str], value: str) -> None:
        cleaned = clean_text(value)

        if not cleaned:
            return

        key = cleaned.lower()

        for existing in queries:
            if existing.lower() == key:
                return

        queries.append(cleaned)

    def _load_cache(self) -> dict[str, str]:
        if not self._cache_path.exists():
            return {}

        try:
            raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

        if not isinstance(raw, dict):
            return {}

        cleaned: dict[str, str] = {}

        for key, value in raw.items():
            k = str(key or "").strip().upper()
            v = str(value or "").strip()
            if k and v:
                cleaned[k] = v

        return cleaned

    def _save_cache_locked(self) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
            temp.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temp, self._cache_path)
        except Exception as exc:
            self._logger.warning(f"Acronym cache save failed: {exc}")