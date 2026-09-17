from __future__ import annotations

from typing import Any

from agent.memory.long_term import LongTermMemory
from agent.memory.episodic import EpisodicMemory
from agent.memory.short_term import ShortTermMemory
from core.models import CoreModel
from utils.logger import get_logger
from utils.text import clean_text


DEFAULT_SIMILARITY_THRESHOLD = 0.3
DEFAULT_MAX_RESULTS = 5
DEFAULT_CONTEXT_WINDOW = 3000


class MemorySearchResult(CoreModel):
    source: str = ""
    relevance_score: float = 0.0
    content: str = ""
    metadata: dict[str, Any] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data: Any) -> None:
        if data.get("metadata") is None:
            data["metadata"] = {}
        super().__init__(**data)


class RetrievedContext(CoreModel):
    query: str = ""
    results: list[MemorySearchResult] = None
    total_found: int = 0
    search_method: str = ""

    def __init__(self, **data: Any) -> None:
        if data.get("results") is None:
            data["results"] = []
        super().__init__(**data)


class MemoryRetrieval:
    def __init__(
        self,
        long_term: LongTermMemory | None = None,
        episodic: EpisodicMemory | None = None,
        short_term: ShortTermMemory | None = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        max_results: int = DEFAULT_MAX_RESULTS,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
    ) -> None:
        self._long_term = long_term or LongTermMemory()
        self._episodic = episodic or EpisodicMemory()
        self._short_term = short_term or ShortTermMemory()
        self._similarity_threshold = max(0.0, min(1.0, similarity_threshold))
        self._max_results = max(1, max_results)
        self._context_window = max(500, context_window)
        self._logger = get_logger("agent.memory.retrieval")

    def search(
        self,
        query: str,
        category: str = "",
        include_episodic: bool = True,
        include_knowledge: bool = True,
        include_topics: bool = True,
        limit: int | None = None,
    ) -> RetrievedContext:
        cleaned_query = clean_text(query)
        if not cleaned_query:
            return RetrievedContext(query=query, search_method="empty_query")

        effective_limit = limit if limit is not None else self._max_results
        results: list[MemorySearchResult] = []

        if include_topics:
            topic_results = self._search_topics(cleaned_query, effective_limit)
            results.extend(topic_results)

        if include_knowledge:
            knowledge_results = self._search_knowledge(
                cleaned_query, category, effective_limit
            )
            results.extend(knowledge_results)

        if include_episodic:
            episodic_results = self._search_episodes(cleaned_query, effective_limit)
            results.extend(episodic_results)

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        results = results[:effective_limit]

        return RetrievedContext(
            query=cleaned_query,
            results=results,
            total_found=len(results),
            search_method="keyword",
        )

    def retrieve_context(
        self,
        goal: str,
        level: str = "",
        task_type: str = "",
    ) -> str:
        context_parts: list[str] = []

        if self._short_term.is_active:
            summary = self._short_term.to_summary()
            if summary:
                context_parts.append(
                    f"Current task context: {summary.get('goal', '')} "
                    f"({summary.get('total_actions', 0)} actions taken, "
                    f"{summary.get('total_tokens_used', 0)} tokens used)"
                )

        search_results = self.search(
            query=goal,
            include_episodic=True,
            include_knowledge=True,
            include_topics=True,
            limit=self._max_results,
        )

        for result in search_results.results:
            if result.relevance_score >= self._similarity_threshold:
                truncated = result.content[:self._context_window]
                context_parts.append(f"[{result.source}] {truncated}")

        if task_type:
            best_strategies = self._episodic.get_best_strategy_for_task(
                task_type=task_type,
                limit=2,
            )
            if best_strategies:
                strategy_text = "; ".join(
                    f"{s['strategy']} (quality: {s['avg_quality']:.2f})"
                    for s in best_strategies
                )
                context_parts.append(f"Best past strategies: {strategy_text}")

        if level:
            preferences = self._long_term.get_all_preferences(category="level")
            level_pref = preferences.get(level, "")
            if level_pref:
                context_parts.append(f"User preference for {level}: {level_pref}")

        if not context_parts:
            return ""

        combined = "\n".join(context_parts)
        return combined[:self._context_window * 2]

    def get_relevant_past_runs(
        self,
        task: str,
        task_type: str = "",
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        episodes = self._episodic.get_similar_episodes(
            task=task,
            task_type=task_type,
            limit=limit,
        )
        results = []
        for episode in episodes:
            results.append({
                "task": episode.task,
                "task_type": episode.task_type,
                "strategy_used": episode.strategy_used,
                "outcome": episode.outcome,
                "quality_score": episode.quality_score,
                "tokens_used": episode.tokens_used,
                "success": episode.success,
                "finished_at": episode.finished_at,
            })
        return results

    def build_task_context(
        self,
        goal: str,
        level: str = "",
        task_type: str = "",
        max_context_length: int | None = None,
    ) -> dict[str, Any]:
        effective_max = max_context_length or (self._context_window * 2)

        retrieved = self.retrieve_context(
            goal=goal,
            level=level,
            task_type=task_type,
        )

        past_runs = self.get_relevant_past_runs(
            task=goal,
            task_type=task_type,
            limit=3,
        )

        avg_tokens = self._episodic.get_average_tokens_for_task_type(
            task_type=task_type
        )
        success_rate = self._episodic.get_success_rate(task_type=task_type)

        context_text = retrieved[:effective_max]

        return {
            "context_text": context_text,
            "past_runs": past_runs,
            "avg_tokens_for_task_type": int(avg_tokens),
            "success_rate_for_task_type": success_rate,
            "has_relevant_history": bool(past_runs),
        }

    def _search_topics(self, query: str, limit: int) -> list[MemorySearchResult]:
        try:
            topics = self._long_term.recall_topics(query=query, limit=limit)
            results = []
            for topic in topics:
                relevance = self._compute_relevance(query, topic.get("topic", ""))
                if relevance >= self._similarity_threshold:
                    results.append(
                        MemorySearchResult(
                            source="topic_history",
                            relevance_score=relevance,
                            content=(
                                f"Topic: {topic.get('topic', '')} | "
                                f"Outcome: {topic.get('outcome', '')} | "
                                f"Quality: {topic.get('quality_score', 0.0):.2f}"
                            ),
                            metadata=topic,
                        )
                    )
            return results
        except Exception as exc:
            self._logger.warning(f"Topic search failed: {exc}")
            return []

    def _search_knowledge(
        self,
        query: str,
        category: str,
        limit: int,
    ) -> list[MemorySearchResult]:
        try:
            knowledge_items = self._long_term.search_knowledge(
                query=query,
                category=category,
                limit=limit,
            )
            results = []
            for item in knowledge_items:
                relevance = self._compute_relevance(
                    query,
                    f"{item.get('key', '')} {item.get('content', '')}",
                )
                if relevance >= self._similarity_threshold:
                    results.append(
                        MemorySearchResult(
                            source="knowledge_base",
                            relevance_score=relevance,
                            content=item.get("content", ""),
                            metadata=item,
                        )
                    )
            return results
        except Exception as exc:
            self._logger.warning(f"Knowledge search failed: {exc}")
            return []

    def _search_episodes(self, query: str, limit: int) -> list[MemorySearchResult]:
        try:
            episodes = self._episodic.get_similar_episodes(
                task=query,
                limit=limit,
            )
            results = []
            for episode in episodes:
                relevance = self._compute_relevance(query, episode.task)
                if relevance >= self._similarity_threshold:
                    results.append(
                        MemorySearchResult(
                            source="episodic_memory",
                            relevance_score=relevance,
                            content=(
                                f"Task: {episode.task} | "
                                f"Strategy: {episode.strategy_used} | "
                                f"Outcome: {episode.outcome} | "
                                f"Quality: {episode.quality_score:.2f} | "
                                f"Tokens: {episode.tokens_used}"
                            ),
                            metadata=episode.model_dump(mode="json"),
                        )
                    )
            return results
        except Exception as exc:
            self._logger.warning(f"Episodic search failed: {exc}")
            return []

    def _compute_relevance(self, query: str, target: str) -> float:
        query_tokens = set(clean_text(query).lower().split())
        target_tokens = set(clean_text(target).lower().split())

        if not query_tokens or not target_tokens:
            return 0.0

        overlap = len(query_tokens & target_tokens)
        return overlap / len(query_tokens)