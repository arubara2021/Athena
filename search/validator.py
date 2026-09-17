from __future__ import annotations

import difflib
import re
from typing import Any

from core.exceptions import ValidationError as SearchValidationError
from core.models import Source, SourcePlatform, SourceType
from core.schemas import SearchQuerySchema
from search.query_tokens import concept_tokens, get_generic_terms, strip_filler, tokenize
from utils.logger import get_logger
from utils.text import clean_text
from utils.url import is_valid_url


class SourceValidator:
    def __init__(
        self,
        min_title_length: int = 3,
        max_title_length: int = 500,
        max_abstract_length: int = 5000,
        strict_threshold: float = 0.18,
        relaxed_threshold: float = 0.03,
        fuzzy_match_ratio: float = 0.82,
        max_core_tokens: int = 12,
        max_phrases: int = 24,
    ) -> None:
        self._min_title_length = max(1, min_title_length)
        self._max_title_length = max(self._min_title_length, max_title_length)
        self._max_abstract_length = max(100, max_abstract_length)
        self._strict_threshold = max(0.0, min(1.0, strict_threshold))
        self._relaxed_threshold = max(0.0, min(1.0, relaxed_threshold))
        self._fuzzy_match_ratio = max(0.5, min(1.0, fuzzy_match_ratio))
        self._max_core_tokens = max(4, max_core_tokens)
        self._max_phrases = max(4, max_phrases)
        self._logger = get_logger("search.validator")

    def validate_query(self, query: Any) -> SearchQuerySchema:
        try:
            if isinstance(query, SearchQuerySchema):
                return query
            if isinstance(query, str):
                return SearchQuerySchema(topic=query)
            if isinstance(query, dict):
                return SearchQuerySchema.model_validate(query)
            raise SearchValidationError(
                "Unsupported search query type",
                details={"type": type(query).__name__},
            )
        except SearchValidationError:
            raise
        except Exception as exc:
            raise SearchValidationError(
                "Invalid search query",
                details={"error": str(exc)},
            ) from exc

    def validate_sources(
        self,
        sources: list[Any],
        query: Any = None,
        strict: bool = True,
    ) -> list[Source]:
        valid_sources: list[Source] = []

        topic = self._extract_topic(query)
        primary_concept = self._extract_primary_concept(query)
        keywords = self._extract_keywords(query)
        short_queries = self._extract_short_queries(query)

        core_tokens = self._build_core_tokens(topic, primary_concept, keywords)
        phrases = self._build_phrases(topic, primary_concept, keywords, short_queries)

        threshold = self._strict_threshold if strict else self._relaxed_threshold

        for source in sources:
            if not self.validate_source(source):
                continue

            if not core_tokens and not phrases:
                valid_sources.append(source)
                continue

            relevance = self._relevance_score(source, core_tokens, phrases)

            if relevance >= threshold:
                valid_sources.append(source)

        return valid_sources

    def validate_source(self, source: Any) -> bool:
        try:
            return self._validate_source(source)
        except Exception as exc:
            self._logger.warning(f"Source validation failed: {exc}")
            return False

    def _validate_source(self, source: Any) -> bool:
        if not isinstance(source, Source):
            return False

        title = clean_text(source.title)

        if len(title) < self._min_title_length:
            return False

        if len(title) > self._max_title_length:
            return False

        lower_title = title.lower()

        if "this paper has been withdrawn" in lower_title:
            return False

        if lower_title.strip() in {"withdrawn", "retracted"}:
            return False

        if not source.url or not is_valid_url(source.url):
            return False

        try:
            SourcePlatform(self._enum_value(source.platform))
            SourceType(self._enum_value(source.source_type))
        except Exception:
            return False

        if source.abstract is not None and len(source.abstract) > self._max_abstract_length:
            return False

        if source.year is not None and not (0 <= source.year <= 2100):
            return False

        if source.citation_count is not None and source.citation_count < 0:
            return False

        if source.score is not None and source.score < 0:
            return False

        if not isinstance(source.metadata, dict):
            return False

        return True

    def _extract_topic(self, query: Any) -> str:
        if isinstance(query, SearchQuerySchema):
            return clean_text(getattr(query, "corrected_topic", "") or query.topic)

        if isinstance(query, str):
            return clean_text(query)

        if isinstance(query, dict):
            return clean_text(query.get("corrected_topic", "") or query.get("topic", ""))

        return ""

    def _extract_primary_concept(self, query: Any) -> str:
        if isinstance(query, SearchQuerySchema):
            return clean_text(
                getattr(query, "primary_concept", "")
                or getattr(query, "corrected_topic", "")
                or query.topic
            )

        if isinstance(query, str):
            return clean_text(query)

        if isinstance(query, dict):
            return clean_text(
                query.get("primary_concept", "")
                or query.get("corrected_topic", "")
                or query.get("topic", "")
            )

        return ""

    def _extract_keywords(self, query: Any) -> list[str]:
        values: list[Any] = []

        if isinstance(query, SearchQuerySchema):
            values.extend(getattr(query, "keywords", []) or [])
        elif isinstance(query, dict):
            values.extend(query.get("keywords", []) or [])

        cleaned: list[str] = []

        for value in values:
            text = clean_text(value)
            if text and text not in cleaned:
                cleaned.append(text)

        return cleaned[:16]

    def _extract_short_queries(self, query: Any) -> list[str]:
        values: list[Any] = []

        if isinstance(query, SearchQuerySchema):
            values.extend(getattr(query, "short_search_queries", []) or [])
        elif isinstance(query, dict):
            values.extend(query.get("short_search_queries", []) or [])

        cleaned: list[str] = []

        for value in values:
            text = clean_text(value)
            if text and text not in cleaned:
                cleaned.append(text)

        return cleaned[:16]

    def _build_core_tokens(
        self,
        topic: str,
        primary_concept: str,
        keywords: list[str],
    ) -> set[str]:
        tokens: set[str] = set()

        if primary_concept:
            tokens.update(concept_tokens(primary_concept))

        if topic and topic != primary_concept:
            tokens.update(concept_tokens(topic))

        for keyword in keywords:
            tokens.update(concept_tokens(keyword))

        if not tokens and topic:
            tokens.update(tokenize(topic.lower()))

        if len(tokens) > self._max_core_tokens:
            return set(list(tokens)[: self._max_core_tokens])

        return tokens

    def _build_phrases(
        self,
        topic: str,
        primary_concept: str,
        keywords: list[str],
        short_queries: list[str],
    ) -> list[str]:
        values: list[str] = []

        if primary_concept:
            values.append(primary_concept)

        if topic and topic != primary_concept:
            values.append(topic)

        values.extend(keywords)
        values.extend(short_queries)

        phrases: list[str] = []
        seen: set[str] = set()

        for value in values:
            normalized = self._normalize_phrase(value)

            if not normalized:
                continue

            for phrase in self._phrase_variants(normalized):
                if phrase in seen:
                    continue

                seen.add(phrase)
                phrases.append(phrase)

                if len(phrases) >= self._max_phrases:
                    return phrases

        return phrases

    def _normalize_phrase(self, value: Any) -> str:
        text = strip_filler(clean_text(value)).lower()
        text = re.sub(r"[^\w\s+#.-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return ""

        words = text.split()
        generic_terms = get_generic_terms()

        meaningful = [
            word
            for word in words
            if len(word) >= 2 and word not in generic_terms
        ]

        if meaningful:
            return " ".join(meaningful)

        return " ".join([word for word in words if len(word) >= 2])

    def _phrase_variants(self, phrase: str) -> list[str]:
        words = phrase.split()

        if len(words) < 2:
            return []

        variants: list[str] = []

        variants.append(" ".join(words))

        for index in range(len(words) - 1):
            variants.append(" ".join(words[index:index + 2]))

        for index in range(len(words) - 2):
            variants.append(" ".join(words[index:index + 3]))

        if len(words) >= 2:
            variants.append(" ".join(words[:2]))
            variants.append(" ".join(words[-2:]))

        cleaned: list[str] = []
        seen: set[str] = set()

        for variant in variants:
            normalized = re.sub(r"\s+", " ", variant).strip()

            if len(normalized) < 4:
                continue

            if normalized in seen:
                continue

            seen.add(normalized)
            cleaned.append(normalized)

        return cleaned[:12]

    def _relevance_score(
        self,
        source: Source,
        core_tokens: set[str],
        phrases: list[str],
    ) -> float:
        if not core_tokens and not phrases:
            return 1.0

        title_text = source.title.lower()
        all_text = self._relevance_text(source).lower()

        title_tokens = set(tokenize(title_text))
        all_tokens = set(tokenize(all_text))

        phrase_title = False
        phrase_all = False

        for phrase in phrases:
            if phrase in title_text:
                phrase_title = True

            if phrase in all_text:
                phrase_all = True

        matched_title = self._count_matched(core_tokens, title_tokens)
        matched_all = self._count_matched(core_tokens, all_tokens)

        token_count = len(core_tokens) if core_tokens else 0

        title_ratio = matched_title / token_count if token_count else 0.0
        all_ratio = matched_all / token_count if token_count else 0.0

        score = 0.0

        if phrase_title:
            score = max(score, 0.95)
        elif phrase_all:
            score = max(score, 0.75)
        else:
            score = (title_ratio * 0.65) + (all_ratio * 0.35)

        if token_count >= 2 and not phrase_title and not phrase_all:
            if matched_all < 2:
                score = min(score, 0.08)
            elif matched_title < 1:
                score = min(score, 0.12)

        platform = self._enum_value(source.platform)
        source_type = self._enum_value(source.source_type)

        if platform == "wikipedia" or source_type == "documentation":
            if score >= 0.25:
                score += 0.05

        return max(0.0, min(1.0, score))

    def _count_matched(self, core_tokens: set[str], candidate_tokens: set[str]) -> int:
        if not core_tokens:
            return 0

        matched = 0

        for token in core_tokens:
            if self._token_exists(token, candidate_tokens):
                matched += 1

        return matched

    def _token_exists(self, token: str, candidate_tokens: set[str]) -> bool:
        if token in candidate_tokens:
            return True

        if len(token) < 4:
            return False

        for candidate in candidate_tokens:
            if candidate == token:
                return True

            if candidate.startswith(token) or token.startswith(candidate):
                return True

            if abs(len(candidate) - len(token)) <= 2:
                ratio = difflib.SequenceMatcher(None, token, candidate).ratio()

                if ratio >= self._fuzzy_match_ratio:
                    return True

        return False

    def _relevance_text(self, source: Source) -> str:
        metadata = source.metadata if isinstance(source.metadata, dict) else {}

        description = clean_text(metadata.get("description", ""))
        summary = clean_text(metadata.get("summary", ""))

        topics = metadata.get("topics", [])
        tags = metadata.get("tags", [])

        parts = [source.title]

        if source.abstract:
            parts.append(source.abstract)

        if description:
            parts.append(description)

        if summary:
            parts.append(summary)

        if isinstance(topics, list):
            parts.extend(str(item) for item in topics if item)

        if isinstance(tags, list):
            parts.extend(str(item) for item in tags if item)

        return " ".join(parts)

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value))