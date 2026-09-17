from __future__ import annotations

import math
import difflib
from datetime import datetime, timezone
from typing import Any

from core import constants
from core.models import RankedSource, Source, SourcePlatform, SourceType
from core.ranking_config import get_ranking_config
from search.query_tokens import concept_tokens, tokenize
from utils.logger import get_logger
from utils.text import clean_text


class HeuristicScorer:
    def __init__(self) -> None:
        self._logger = get_logger("ranking.scorer")
        self._config = get_ranking_config()

    def score_sources(
        self,
        sources: list[Any],
        query: Any = None,
        goal: str | None = None,
        level: str | None = None,
    ) -> list[RankedSource]:
        query_context = self._extract_query_context(query)
        ranked: list[RankedSource] = []

        for source in sources:
            if not isinstance(source, Source):
                continue

            try:
                score, factors = self._evaluate(source, query_context, goal, level)
                confidence = self._confidence(source)
                reason = self._reason(factors)

                scored_source = source.model_copy(update={"score": score})

                ranked.append(
                    RankedSource(
                        source=scored_source,
                        rank=1,
                        score=score,
                        confidence=confidence,
                        reason=reason,
                    )
                )
            except Exception as exc:
                self._logger.warning(f"Skipping source during scoring: {exc}")
                continue

        ranked.sort(key=lambda item: item.score, reverse=True)

        for index, item in enumerate(ranked, start=1):
            item.rank = index

        return ranked

    def score_source(
        self,
        source: Source,
        query: Any = None,
        goal: str | None = None,
        level: str | None = None,
    ) -> float:
        query_context = self._extract_query_context(query)
        score, _ = self._evaluate(source, query_context, goal, level)
        return score

    def relevance_factor(self, source: Any, query: Any) -> float:
        if not isinstance(source, Source):
            return 0.0

        try:
            query_context = self._extract_query_context(query)
            return self._relevance_score(source, query_context)
        except Exception:
            return 0.0

    def compute_alignment(
        self,
        source: Source,
        goal: str | None,
        level: str | None,
    ) -> float:
        alignment = self._config.alignment_settings()

        goal_text = clean_text(goal).lower()
        level_text = clean_text(level).lower()

        beginner = "beginner" in level_text or "basics" in goal_text
        advanced = "advanced" in level_text or "advanced" in goal_text

        if not beginner and not advanced:
            return 0.0

        difficulty = (
            self._enum_value(source.difficulty)
            if source.difficulty is not None
            else ""
        )

        source_type = self._enum_value(source.source_type)
        adjustment = 0.0

        if beginner:
            adjustment += alignment.beginner_difficulty.get(difficulty, 0.0)
            adjustment += alignment.beginner_source_type.get(source_type, 0.0)

        if advanced:
            adjustment += alignment.advanced_difficulty.get(difficulty, 0.0)
            adjustment += alignment.advanced_source_type.get(source_type, 0.0)

        return max(-alignment.cap, min(alignment.cap, adjustment))

    def _extract_query_context(self, query: Any) -> dict[str, Any]:
        empty_context = {
            "topic": "",
            "primary_concept": "",
            "keywords": [],
            "short_search_queries": [],
        }

        if query is None:
            return empty_context

        if isinstance(query, str):
            topic = clean_text(query)

            return {
                "topic": topic,
                "primary_concept": topic,
                "keywords": [],
                "short_search_queries": [],
            }

        if isinstance(query, dict):
            topic = clean_text(
                query.get("corrected_topic", "")
                or query.get("topic", "")
            )

            primary_concept = clean_text(
                query.get("primary_concept", "")
                or topic
            )

            keywords = self._normalize_keywords(query.get("keywords", []))
            short_search_queries = self._normalize_keywords(
                query.get("short_search_queries", [])
            )

            if not keywords and topic:
                keywords = list(concept_tokens(topic))[:8]

            return {
                "topic": topic,
                "primary_concept": primary_concept,
                "keywords": keywords,
                "short_search_queries": short_search_queries,
            }

        topic = clean_text(
            getattr(query, "corrected_topic", "")
            or getattr(query, "topic", "")
        )

        primary_concept = clean_text(
            getattr(query, "primary_concept", "")
            or topic
        )

        keywords = self._normalize_keywords(getattr(query, "keywords", []))
        short_search_queries = self._normalize_keywords(
            getattr(query, "short_search_queries", [])
        )

        if not keywords and topic:
            keywords = list(concept_tokens(topic))[:8]

        return {
            "topic": topic,
            "primary_concept": primary_concept,
            "keywords": keywords,
            "short_search_queries": short_search_queries,
        }

    def _normalize_keywords(self, values: Any) -> list[str]:
        if values is None:
            return []

        if isinstance(values, str):
            values = [values]

        if not isinstance(values, (list, tuple, set)):
            return []

        result: list[str] = []

        for value in values:
            text = clean_text(value)

            if not text:
                continue

            if text not in result:
                result.append(text)

            if len(result) >= 16:
                break

        return result

    def _evaluate(
        self,
        source: Source,
        query_context: dict[str, Any],
        goal: str | None,
        level: str | None,
    ) -> tuple[float, dict[str, float]]:
        weights = self._config.scorer_weights()

        relevance = self._relevance_score(source, query_context)
        authority = self._platform_authority(source)
        type_quality = self._type_quality(source)
        impact = self._impact_score(source)
        recency = self._recency_score(source)
        content = self._content_score(source)
        support = self._support_score(source)
        authors = self._author_score(source)
        learner_alignment = self.compute_alignment(source, goal, level)

        factors = {
            "relevance": relevance,
            "authority": authority,
            "type_quality": type_quality,
            "impact": impact,
            "recency": recency,
            "content": content,
            "support": support,
            "authors": authors,
            "learner_alignment": learner_alignment,
        }

        score = (
            relevance * weights.relevance
            + authority * weights.authority
            + type_quality * weights.type_quality
            + impact * weights.impact
            + recency * weights.recency
            + content * weights.content
            + support * weights.support
            + authors * weights.authors
        )

        score += learner_alignment
        score = max(0.0, min(1.0, score))

        return round(score, constants.DEFAULT_SCORE_PRECISION), factors

    def _relevance_score(
        self,
        source: Source,
        query_context: dict[str, Any],
    ) -> float:
        if not isinstance(query_context, dict):
            query_context = {
                "topic": "",
                "primary_concept": "",
                "keywords": [],
                "short_search_queries": [],
            }

        primary_concept = clean_text(
            query_context.get("primary_concept", "")
            or query_context.get("topic", "")
        )

        topic = clean_text(query_context.get("topic", ""))
        keywords = query_context.get("keywords", []) or []

        primary_tokens = concept_tokens(primary_concept) if primary_concept else set()

        if not primary_tokens and topic:
            primary_tokens = concept_tokens(topic)

        keyword_tokens: set[str] = set()

        for keyword in keywords:
            keyword_tokens.update(concept_tokens(keyword))

        corrected_tokens = concept_tokens(topic) if topic else set()
        all_query_tokens = primary_tokens | keyword_tokens | corrected_tokens

        if not all_query_tokens:
            return 0.5

        metadata = source.metadata if isinstance(source.metadata, dict) else {}
        description = clean_text(metadata.get("description", ""))
        all_text = f"{source.title} {source.abstract or ''} {description}"

        source_tokens = set(tokenize(all_text.lower()))
        title_tokens = set(tokenize(source.title.lower()))

        if not source_tokens:
            return 0.1

        def matched_count(query_tokens: set[str], target_tokens: set[str]) -> int:
            if not query_tokens:
                return 0

            return sum(
                1
                for token in query_tokens
                if self._token_in_set(token, target_tokens)
            )

        primary_all = matched_count(primary_tokens, source_tokens)
        primary_title = matched_count(primary_tokens, title_tokens)

        keyword_all = matched_count(keyword_tokens, source_tokens)
        keyword_title = matched_count(keyword_tokens, title_tokens)

        corrected_all = matched_count(corrected_tokens, source_tokens)
        corrected_title = matched_count(corrected_tokens, title_tokens)

        primary_ratio = (
            primary_all / len(primary_tokens)
            if primary_tokens
            else 0.0
        )

        primary_title_ratio = (
            primary_title / len(primary_tokens)
            if primary_tokens
            else 0.0
        )

        keyword_ratio = (
            keyword_all / len(keyword_tokens)
            if keyword_tokens
            else 0.0
        )

        keyword_title_ratio = (
            keyword_title / len(keyword_tokens)
            if keyword_tokens
            else 0.0
        )

        corrected_ratio = (
            corrected_all / len(corrected_tokens)
            if corrected_tokens
            else 0.0
        )

        corrected_title_ratio = (
            corrected_title / len(corrected_tokens)
            if corrected_tokens
            else 0.0
        )

        title_signal = max(
            primary_title_ratio,
            keyword_title_ratio * 0.8,
            corrected_title_ratio * 0.6,
        )

        if primary_tokens:
            if primary_title_ratio >= 0.8:
                base = 0.88 + 0.12 * keyword_ratio
            elif primary_ratio >= 0.8:
                base = 0.72 + 0.18 * keyword_ratio
            elif primary_ratio >= 0.4:
                base = 0.48 + 0.22 * keyword_ratio + 0.10 * corrected_ratio
            elif keyword_ratio >= 0.5:
                base = 0.38 + 0.15 * corrected_ratio
            elif corrected_ratio >= 0.5:
                base = 0.28
            else:
                base = 0.05 + 0.15 * keyword_ratio + 0.10 * corrected_ratio
        else:
            best_ratio = max(keyword_ratio, corrected_ratio)

            if best_ratio >= 0.6:
                base = 0.60 + 0.20 * best_ratio
            elif best_ratio > 0:
                base = 0.20 + 0.35 * best_ratio
            else:
                base = 0.05

        score = base + 0.10 * title_signal

        return max(0.0, min(1.0, score))

    def _token_in_set(self, token: str, token_set: set[str]) -> bool:
        if token in token_set:
            return True

        if len(token) >= 3:
            for candidate in token_set:
                if candidate == token:
                    return True

                if candidate.startswith(token) or token.startswith(candidate):
                    return True

                if abs(len(candidate) - len(token)) <= 2:
                    ratio = difflib.SequenceMatcher(None, token, candidate).ratio()

                    if ratio >= 0.8:
                        return True

        return False

    def _platform_authority(self, source: Source) -> float:
        source_type = self._enum_value(source.source_type)
        platform = self._enum_value(source.platform)

        paper_authority = {
            SourcePlatform.SEMANTIC_SCHOLAR.value: 0.95,
            SourcePlatform.ARXIV.value: 0.95,
            SourcePlatform.OPENALEX.value: 0.85,
            SourcePlatform.WIKIPEDIA.value: 0.55,
            SourcePlatform.WEB.value: 0.45,
            SourcePlatform.GITHUB.value: 0.40,
            SourcePlatform.HUGGINGFACE.value: 0.45,
        }

        repository_authority = {
            SourcePlatform.GITHUB.value: 0.95,
            SourcePlatform.HUGGINGFACE.value: 0.80,
            SourcePlatform.WEB.value: 0.55,
            SourcePlatform.WIKIPEDIA.value: 0.35,
            SourcePlatform.SEMANTIC_SCHOLAR.value: 0.30,
            SourcePlatform.ARXIV.value: 0.30,
            SourcePlatform.OPENALEX.value: 0.30,
        }

        model_authority = {
            SourcePlatform.HUGGINGFACE.value: 0.95,
            SourcePlatform.GITHUB.value: 0.85,
            SourcePlatform.WEB.value: 0.55,
        }

        documentation_authority = {
            SourcePlatform.WIKIPEDIA.value: 0.90,
            SourcePlatform.WEB.value: 0.75,
            SourcePlatform.GITHUB.value: 0.70,
            SourcePlatform.HUGGINGFACE.value: 0.60,
        }

        if source_type == SourceType.PAPER.value:
            return paper_authority.get(platform, 0.35)

        if source_type == SourceType.REPOSITORY.value:
            return repository_authority.get(platform, 0.35)

        if source_type == SourceType.MODEL.value:
            return model_authority.get(platform, 0.40)

        if source_type == SourceType.DOCUMENTATION.value:
            return documentation_authority.get(platform, 0.40)

        if source_type in {SourceType.COURSE.value, SourceType.VIDEO.value}:
            return 0.65 if platform == SourcePlatform.WEB.value else 0.45

        return 0.40

    def _type_quality(self, source: Source) -> float:
        source_type = self._enum_value(source.source_type)

        quality = {
            SourceType.PAPER.value: 0.90,
            SourceType.REPOSITORY.value: 0.85,
            SourceType.DOCUMENTATION.value: 0.80,
            SourceType.COURSE.value: 0.80,
            SourceType.MODEL.value: 0.75,
            SourceType.DATASET.value: 0.70,
            SourceType.VIDEO.value: 0.70,
            SourceType.BLOG.value: 0.50,
            SourceType.OTHER.value: 0.40,
        }

        return quality.get(source_type, 0.40)

    def _impact_score(self, source: Source) -> float:
        metadata = source.metadata if isinstance(source.metadata, dict) else {}
        source_type = self._enum_value(source.source_type)

        if source_type in {SourceType.REPOSITORY.value, SourceType.MODEL.value}:
            stars = metadata.get("stars")
            downloads = metadata.get("downloads")

            stars_value = int(stars) if isinstance(stars, int) else 0
            downloads_value = int(downloads) if isinstance(downloads, int) else 0

            value = max(stars_value, downloads_value // 10)
            return self._log_scale(value, 5000)

        citation_count = source.citation_count or 0
        return self._log_scale(citation_count, 500)

    def _recency_score(self, source: Source) -> float:
        year = source.year

        if year is None and source.published_at is not None:
            year = source.published_at.year

        if year is None:
            return 0.30

        current_year = datetime.now(timezone.utc).year
        age = current_year - int(year)

        if age < 0:
            return 0.50

        if age <= 1:
            return 1.00

        if age <= 3:
            return 0.85

        if age <= 5:
            return 0.70

        if age <= 10:
            return 0.50

        return 0.30

    def _content_score(self, source: Source) -> float:
        abstract = clean_text(source.abstract or "")

        if len(abstract) >= 200:
            return 1.0

        if len(abstract) >= 100:
            return 0.7

        if abstract:
            return 0.4

        return 0.1

    def _support_score(self, source: Source) -> float:
        metadata = source.metadata if isinstance(source.metadata, dict) else {}
        score = 0.0

        if source.has_code is True:
            score += 0.6

        if metadata.get("has_pdf") is True:
            score += 0.4

        if metadata.get("language"):
            score += 0.1

        return max(0.0, min(1.0, score))

    def _author_score(self, source: Source) -> float:
        if not source.authors:
            return 0.0

        return min(len(source.authors) / 5, 1.0)

    def _confidence(self, source: Source) -> float:
        confidence = 0.25

        if source.title:
            confidence += 0.10

        if source.url:
            confidence += 0.10

        if source.abstract:
            confidence += 0.15

        metadata = source.metadata if isinstance(source.metadata, dict) else {}

        if source.citation_count is not None or metadata.get("stars") is not None:
            confidence += 0.15

        if source.year is not None or source.published_at is not None:
            confidence += 0.10

        if source.authors:
            confidence += 0.05

        if metadata:
            confidence += 0.05

        return max(0.0, min(1.0, confidence))

    def _reason(self, factors: dict[str, float]) -> str:
        parts: list[str] = []

        if factors.get("relevance", 0.0) > 0.65:
            parts.append("relevant")

        if factors.get("authority", 0.0) > 0.75:
            parts.append("authoritative platform")

        if factors.get("impact", 0.0) > 0.55:
            parts.append("strong impact")

        if factors.get("recency", 0.0) > 0.70:
            parts.append("recent")

        if factors.get("support", 0.0) > 0.50:
            parts.append("has code or pdf")

        if factors.get("content", 0.0) > 0.70:
            parts.append("good description")

        if not parts:
            parts.append("heuristic quality")

        return ", ".join(parts[:3])

    @staticmethod
    def _log_scale(value: Any, scale: float) -> float:
        try:
            numeric = float(value)
        except Exception:
            return 0.0

        if numeric <= 0:
            return 0.0

        return min(1.0, math.log1p(numeric) / math.log1p(scale))

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value))