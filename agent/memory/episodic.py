from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import Field

from core.config import get_logs_directory
from core.models import CoreModel
from utils.logger import get_logger


EPISODIC_FILENAME = "agent_episodes.jsonl"
MAX_EPISODES_IN_MEMORY = 500


class Episode(CoreModel):
    episode_id: str = Field(default_factory=lambda: str(uuid4()))
    task: str = ""
    task_type: str = ""
    goal: str = ""
    level: str = ""
    strategy_used: str = ""
    actions_taken: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    tokens_used: int = Field(default=0, ge=0)
    tokens_by_category: dict[str, int] = Field(default_factory=dict)
    total_iterations: int = Field(default=0, ge=0)
    outcome: str = ""
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    success: bool = False
    errors: list[str] = Field(default_factory=list)
    findings_count: int = Field(default=0, ge=0)
    started_at: str = ""
    finished_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    latency_ms: float = Field(default=0.0, ge=0.0)


class EpisodicMemory:
    def __init__(self, storage_path: str | Path | None = None) -> None:
        self._lock = threading.Lock()
        self._logger = get_logger("agent.memory.episodic")
        self._episodes: list[Episode] = []
        self._loaded = False

        if storage_path is not None:
            self._storage_path = Path(storage_path).expanduser()
        else:
            self._storage_path = get_logs_directory() / EPISODIC_FILENAME

    @property
    def storage_path(self) -> Path:
        return self._storage_path

    @property
    def episode_count(self) -> int:
        self._ensure_loaded()
        with self._lock:
            return len(self._episodes)

    def record_episode(
        self,
        task: str,
        task_type: str = "",
        goal: str = "",
        level: str = "",
        strategy_used: str = "",
        actions_taken: list[str] | None = None,
        tools_used: list[str] | None = None,
        tokens_used: int = 0,
        tokens_by_category: dict[str, int] | None = None,
        total_iterations: int = 0,
        outcome: str = "",
        quality_score: float = 0.0,
        success: bool = False,
        errors: list[str] | None = None,
        findings_count: int = 0,
        started_at: str = "",
        latency_ms: float = 0.0,
    ) -> Episode:
        self._ensure_loaded()

        episode = Episode(
            task=task,
            task_type=task_type,
            goal=goal,
            level=level,
            strategy_used=strategy_used,
            actions_taken=actions_taken or [],
            tools_used=tools_used or [],
            tokens_used=max(0, tokens_used),
            tokens_by_category=tokens_by_category or {},
            total_iterations=max(0, total_iterations),
            outcome=outcome,
            quality_score=max(0.0, min(1.0, quality_score)),
            success=success,
            errors=errors or [],
            findings_count=max(0, findings_count),
            started_at=started_at,
            latency_ms=max(0.0, latency_ms),
        )

        with self._lock:
            self._episodes.append(episode)
            if len(self._episodes) > MAX_EPISODES_IN_MEMORY:
                self._episodes = self._episodes[-MAX_EPISODES_IN_MEMORY:]

        self._persist_episode(episode)
        return episode

    def get_episodes(
        self,
        task_type: str = "",
        success_only: bool = False,
        limit: int = 20,
    ) -> list[Episode]:
        self._ensure_loaded()
        with self._lock:
            episodes = list(self._episodes)

        if task_type:
            episodes = [e for e in episodes if e.task_type == task_type]

        if success_only:
            episodes = [e for e in episodes if e.success]

        episodes.sort(
            key=lambda e: e.finished_at,
            reverse=True,
        )
        return episodes[:max(1, limit)]

    def get_similar_episodes(
        self,
        task: str,
        task_type: str = "",
        limit: int = 5,
    ) -> list[Episode]:
        self._ensure_loaded()
        with self._lock:
            episodes = list(self._episodes)

        if task_type:
            episodes = [e for e in episodes if e.task_type == task_type]

        task_lower = task.lower()
        task_tokens = set(task_lower.split())

        scored: list[tuple[float, Episode]] = []
        for episode in episodes:
            episode_tokens = set(episode.task.lower().split())
            if not episode_tokens:
                continue
            overlap = len(task_tokens & episode_tokens)
            score = overlap / max(len(task_tokens), 1)
            if score > 0.0:
                scored.append((score, episode))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [episode for _, episode in scored[:max(1, limit)]]

    def get_best_strategy_for_task(self, task_type: str = "", limit: int = 3) -> list[dict[str, Any]]:
        self._ensure_loaded()
        with self._lock:
            episodes = list(self._episodes)

        if task_type:
            episodes = [e for e in episodes if e.task_type == task_type]

        successful = [e for e in episodes if e.success]
        if not successful:
            return []

        strategy_scores: dict[str, dict[str, Any]] = {}
        for episode in successful:
            strategy = episode.strategy_used or "default"
            if strategy not in strategy_scores:
                strategy_scores[strategy] = {
                    "strategy": strategy,
                    "total_uses": 0,
                    "total_quality": 0.0,
                    "avg_tokens": 0,
                    "best_quality": 0.0,
                }
            entry = strategy_scores[strategy]
            entry["total_uses"] += 1
            entry["total_quality"] += episode.quality_score
            entry["avg_tokens"] += episode.tokens_used
            entry["best_quality"] = max(entry["best_quality"], episode.quality_score)

        results = []
        for strategy, data in strategy_scores.items():
            uses = data["total_uses"]
            results.append({
                "strategy": strategy,
                "total_uses": uses,
                "avg_quality": data["total_quality"] / uses if uses > 0 else 0.0,
                "avg_tokens": int(data["avg_tokens"] / uses) if uses > 0 else 0,
                "best_quality": data["best_quality"],
            })

        results.sort(key=lambda x: x["avg_quality"], reverse=True)
        return results[:max(1, limit)]

    def get_average_tokens_for_task_type(self, task_type: str = "") -> float:
        self._ensure_loaded()
        with self._lock:
            episodes = list(self._episodes)

        if task_type:
            episodes = [e for e in episodes if e.task_type == task_type]

        if not episodes:
            return 0.0

        total = sum(e.tokens_used for e in episodes)
        return total / len(episodes)

    def get_success_rate(self, task_type: str = "") -> float:
        self._ensure_loaded()
        with self._lock:
            episodes = list(self._episodes)

        if task_type:
            episodes = [e for e in episodes if e.task_type == task_type]

        if not episodes:
            return 0.0

        successes = sum(1 for e in episodes if e.success)
        return successes / len(episodes)

    def clear(self) -> None:
        with self._lock:
            self._episodes = []

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._episodes = []
            if self._storage_path.exists():
                try:
                    with open(self._storage_path, "r", encoding="utf-8") as handle:
                        for line in handle:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                                episode = Episode.model_validate(data)
                                self._episodes.append(episode)
                            except Exception:
                                continue
                except Exception as exc:
                    self._logger.warning(f"Failed to load episodes: {exc}")
            self._loaded = True

    def _persist_episode(self, episode: Episode) -> None:
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(episode.model_dump(mode="json"), ensure_ascii=False)
            with open(self._storage_path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception as exc:
            self._logger.warning(f"Failed to persist episode: {exc}")