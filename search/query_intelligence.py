from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field

from core import constants
from core.models import CoreModel
from search.query_profile import _LEVEL_TOKENS as _PROFILE_LEVEL_TOKENS
from search.query_tokens import (
    _MEDIUM_ADVANCED_SIGNALS,
    _MEDIUM_BEGINNER_SIGNALS,
    _STRONG_ADVANCED_SIGNALS,
    _STRONG_BEGINNER_SIGNALS,
    _WEAK_ADVANCED_SIGNALS,
    _WEAK_BEGINNER_SIGNALS,
    get_generic_terms,
    get_intent_not_concept_words,
    strip_filler,
    tokenize,
    tokens_match,
)
from utils.logger import get_logger, get_trace_logger
from utils.text import clean_text, truncate_text


class QueryIntent(str, Enum):
    LEARN = "learn"
    RESEARCH = "research"
    IMPLEMENT = "implement"
    COMPARE = "compare"
    TROUBLESHOOT = "troubleshoot"
    EXPLORE = "explore"


class QueryAnalysis(CoreModel):
    original_topic: str
    cleaned_topic: str
    corrected_topic: str
    primary_concept: str = ""
    keywords: list[str] = Field(default_factory=list)
    intent: QueryIntent = QueryIntent.EXPLORE
    level: str = "mixed"
    learner_level: str = "mixed"
    level_priority: list[str] = Field(default_factory=list)
    intent_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    spell_corrections: dict[str, str] = Field(default_factory=dict)
    short_search_queries: list[str] = Field(default_factory=list)
    suggested_queries: list[str] = Field(default_factory=list)

    target_domains: list[str] = Field(default_factory=list)
    domain_confidences: dict[str, float] = Field(default_factory=dict)
    domain_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    target_formats: list[str] = Field(default_factory=list)

    level_profile: str = ""
    beginner_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    min_beginner_sources: int = Field(default=0, ge=0)
    exclude_research_platforms: bool = False

    preserve_tokens: list[str] = Field(default_factory=list)
    content_tokens: list[str] = Field(default_factory=list)

    llm_used: bool = False


_YEAR_PATTERN = re.compile(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2})\b")
_ARTICLE_PATTERN = re.compile(
    r"\b(?:Article|Section|Chapter|Clause|Paragraph|Title|Amendment)\s+\d+\b",
    re.IGNORECASE,
)
_TITLE_WORD_PATTERN = re.compile(r"\b[A-Z][a-z]{2,}\b")
_MULTIWORD_PROPER_PATTERN = re.compile(
    r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+\b"
)
_ACRONYM_PATTERN = re.compile(r"\b[A-Z]{2,6}(?:[/\-][A-Z]{2,6})*\b")

_COMMON_SENTENCE_STARTERS = frozenset({
    "the", "this", "that", "these", "those", "when", "where",
    "why", "how", "what", "who", "which", "whose", "whom",
    "an", "and", "but", "or", "if", "so", "yet", "for", "nor",
    "as", "at", "by", "in", "of", "on", "to", "up", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had",
    "i", "we", "you", "they", "it", "he", "she",
    "please", "kindly", "can", "could", "would", "should",
    "tell", "show", "give", "help", "explain", "find",
    "need", "want", "looking", "searching",
})

_COMMON_PROPER_NOUNS = frozenset({
    "usa", "us", "uk", "eu", "un", "nato", "unesco", "unicef",
    "who", "imf", "wto", "opec", "asean", "brics", "g20", "g7",
    "oecd", "unescap", "ilo", "fao", "iaea", "icj", "icc",
    "india", "china", "japan", "korea", "russia", "germany",
    "france", "italy", "spain", "brazil", "canada", "australia",
    "mexico", "egypt", "iran", "iraq", "israel", "saudi", "turkey",
    "pakistan", "bangladesh", "myanmar", "thailand", "vietnam",
    "indonesia", "philippines", "malaysia", "singapore",
    "africa", "nigeria", "kenya", "ethiopia", "ghana", "morocco",
    "algeria", "tunisia", "libya", "sudan", "syria", "lebanon",
    "jordan", "yemen", "oman", "qatar", "uae", "kuwait", "bahrain",
    "afghanistan", "ukraine", "poland", "romania", "greece",
    "portugal", "netherlands", "belgium", "switzerland", "austria",
    "sweden", "norway", "denmark", "finland", "iceland", "ireland",
    "scotland", "wales", "england", "britain", "nepal",
    "srilanka", "maldives", "bhutan", "mongolia", "kazakhstan",
    "uzbekistan", "turkmenistan", "kyrgyzstan", "tajikistan",
    "azerbaijan", "armenia", "georgia", "belarus", "moldova",
    "serbia", "croatia", "slovenia", "slovakia", "czechia",
    "hungary", "bulgaria", "albania", "macedonia", "bosnia",
    "kosovo", "montenegro", "estonia", "latvia", "lithuania",
    "cuba", "haiti", "jamaica", "argentina", "chile", "peru",
    "colombia", "venezuela", "ecuador", "bolivia", "paraguay",
    "uruguay", "panama", "costa", "guatemala", "honduras",
    "nicaragua", "elsalvador", "dominican", "puerto",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november",
    "december", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
    "google", "microsoft", "apple", "amazon", "meta", "openai",
    "deepmind", "anthropic", "tesla", "spacex", "nasa", "esa",
    "isro", "mit", "stanford", "harvard", "oxford", "cambridge",
    "ieee", "acm", "cdc", "nih",
    "mitochondria", "dna", "rna", "atp",
    "mars", "venus", "jupiter", "saturn", "mercury", "neptune",
    "uranus", "pluto", "earth", "sun", "moon",
    "rome", "paris", "london", "berlin", "moscow", "beijing",
    "tokyo", "delhi", "mumbai", "chennai", "kolkata", "bangalore",
    "hyderabad", "cairo", "istanbul", "baghdad", "tehran",
    "islamabad", "dhaka", "kathmandu", "colombo",
})

_ADVANCED_FRAMING_PHRASES = (
    "deeply understand",
    "deeply understanding",
    "comprehensive understanding",
    "thorough understanding",
    "rigorous understanding",
    "complete understanding",
    "in depth understanding",
    "in-depth understanding",
    "master ",
    "mastering ",
    "learn deeply about",
)

_STRIPPABLE_LEVEL_TOKENS: frozenset[str] = frozenset({
    "beginner", "beginners",
    "basic", "basics",
    "fundamental", "fundamentals",
    "intro", "introduction", "introductory",
    "intermediate",
    "advanced", "advance", "advances", "advancement", "advancements",
    "expert", "experts",
    "mixed",
    "deep", "deeply",
})

_STRIPPABLE_LEVEL_PHRASES: tuple[str, ...] = (
    "for absolute beginners",
    "for complete beginners",
    "for total beginners",
    "for beginners",
    "for newcomers",
    "for novices",
    "for dummies",
    "no prior experience",
    "no prior knowledge",
    "no experience needed",
    "no experience required",
    "complete beginner",
    "absolute beginner",
    "total beginner",
    "start from zero",
    "from zero to hero",
    "from zero",
    "zero to hero",
    "from scratch",
    "step by step",
    "the hard way",
    "made easy",
    "state of the art",
    "state-of-the-art",
    "cutting edge",
    "cutting-edge",
    "deep dive",
    "deep-dive",
    "in depth",
    "in-depth",
    "graduate level",
    "graduate-level",
    "research level",
    "research-level",
    "proof based",
    "proof-based",
    "mathematical foundations",
    "theoretical foundations",
    "principles of",
    "foundations of",
    "advanced treatment",
    "in-depth treatment",
    "to advanced",
    "up to advanced",
    "and beyond",
)


class QueryIntelligence:
    _DEFAULT_CONFIG = {
        "max_corrected_topic_words": 6,
        "max_primary_concept_words": 5,
        "max_keywords": 8,
        "max_short_queries": 6,
        "max_query_words": 7,
        "model_limit": 3,
        "temperature": constants.DEFAULT_TEMPERATURE,
        "max_tokens": constants.DEFAULT_MAX_TOKENS,
        "max_domains": 4,
        "min_domain_confidence": 0.25,
        "max_formats": 6,
    }

    _LEVEL_WORDS = frozenset({
        "beginner", "beginners", "basic", "basics",
        "fundamental", "fundamentals",
        "intro", "introduction", "introductory",
        "advanced", "advance", "advances", "advancement", "advancements",
        "expert", "intermediate", "mixed",
        "easy", "simple", "complete", "comprehensive", "quick", "fast",
        "slow", "crash", "from", "scratch", "step", "steps",
        "starter", "starting", "start", "for", "no", "prior",
        "to", "and", "beyond", "up", "deeply", "deep", "in",
    })

    _ADVERB_SUFFIXES = ("ly",)

    def __init__(self, use_llm: bool = True) -> None:
        self._use_llm = use_llm
        self._config = self._load_config()
        self._domain_rules = self._load_domain_rules()
        self._logger = get_logger("search.query_intelligence")
        self._trace = get_trace_logger()
        self._intent_not_concept_words = get_intent_not_concept_words()
        self._level_tokens = frozenset(
            set(self._LEVEL_WORDS) | set(_PROFILE_LEVEL_TOKENS)
        )
        self._level_phrase_patterns = tuple(
            (
                phrase,
                re.compile(
                    rf"(?<![A-Za-z0-9]){re.escape(phrase)}(?![A-Za-z0-9])",
                    re.IGNORECASE,
                ),
            )
            for phrase in _STRIPPABLE_LEVEL_PHRASES
        )

    async def analyze(
        self,
        topic: str,
        goal: str = "",
        level: str = "",
        use_llm: bool | None = None,
    ) -> QueryAnalysis:
        original = clean_text(topic)

        preserve_tokens = self._extract_preserve_tokens(original)
        content_tokens = self._extract_content_tokens(original)

        analysis = self._local_analysis(original, goal, level)
        analysis = self._strip_intent_words_from_concept(analysis)
        analysis = self._strip_llm_injected_level_words(analysis)

        if preserve_tokens:
            analysis = self._enforce_preserve_tokens(analysis, preserve_tokens)

        if content_tokens:
            analysis = self._enforce_content_tokens(analysis, content_tokens)

        if not original:
            return analysis

        use_llm_effective = self._use_llm if use_llm is None else bool(use_llm)

        if not use_llm_effective:
            self._emit_profile_trace(analysis, source="local")
            return analysis

        enhanced = await self._llm_enhance(analysis, goal, level)

        if enhanced is not None:
            enhanced = self._strip_intent_words_from_concept(enhanced)
            enhanced = self._strip_llm_injected_level_words(enhanced)

            if preserve_tokens:
                enhanced = self._enforce_preserve_tokens(
                    enhanced,
                    preserve_tokens,
                )

            if content_tokens:
                enhanced = self._enforce_content_tokens(
                    enhanced,
                    content_tokens,
                )

            self._emit_profile_trace(enhanced, source="llm")
            return enhanced

        self._emit_profile_trace(analysis, source="local_fallback")
        return analysis

    def _strip_llm_injected_level_words(
        self,
        analysis: QueryAnalysis,
    ) -> QueryAnalysis:
        if not analysis or not analysis.original_topic:
            return analysis

        user_tokens = self._original_topic_tokens(analysis.original_topic)

        if not user_tokens:
            return analysis

        def clean_field(value: str) -> str:
            text = clean_text(value)

            if not text:
                return text

            text = self._strip_level_phrases(text, user_tokens)

            tokens = text.split()
            kept: list[str] = []

            for token in tokens:
                stripped = token.strip(",.!?;:\"'()[]{}").lower()

                if not stripped:
                    kept.append(token)
                    continue

                if self._is_user_typed_level_word(stripped, user_tokens):
                    kept.append(token)
                    continue

                if stripped in _STRIPPABLE_LEVEL_TOKENS:
                    continue

                kept.append(token)

            return " ".join(kept).strip()

        def clean_keyword(value: str) -> str:
            text = clean_text(value)

            if not text:
                return ""

            text = self._strip_level_phrases(text, user_tokens)

            tokens = text.split()
            kept: list[str] = []

            for token in tokens:
                stripped = token.strip(",.!?;:\"'()[]{}").lower()

                if not stripped:
                    continue

                if self._is_user_typed_level_word(stripped, user_tokens):
                    kept.append(token)
                    continue

                if stripped in _STRIPPABLE_LEVEL_TOKENS:
                    continue

                kept.append(token)

            return " ".join(kept).strip()

        new_corrected = clean_field(analysis.corrected_topic)
        new_primary = clean_field(analysis.primary_concept)

        if not new_corrected.strip():
            new_corrected = analysis.corrected_topic

        if not new_primary.strip():
            new_primary = analysis.primary_concept

        new_keywords: list[str] = []

        for keyword in analysis.keywords or []:
            cleaned = clean_keyword(keyword)

            if cleaned and cleaned not in new_keywords:
                new_keywords.append(cleaned)

        new_short_queries: list[str] = []

        for query in analysis.short_search_queries or []:
            cleaned = clean_field(query)

            if cleaned and cleaned not in new_short_queries:
                new_short_queries.append(cleaned)

        changed = (
            new_corrected != analysis.corrected_topic
            or new_primary != analysis.primary_concept
            or new_keywords != list(analysis.keywords or [])
            or new_short_queries != list(analysis.short_search_queries or [])
        )

        if not changed:
            return analysis

        self._trace.emit(
            "query_llm_level_words_stripped",
            original_topic=analysis.original_topic,
            old_corrected=analysis.corrected_topic,
            new_corrected=new_corrected,
            old_primary=analysis.primary_concept,
            new_primary=new_primary,
            user_typed_tokens=sorted(user_tokens)[:20],
        )

        return analysis.model_copy(
            update={
                "corrected_topic": new_corrected,
                "primary_concept": new_primary,
                "keywords": new_keywords,
                "short_search_queries": new_short_queries,
                "suggested_queries": new_short_queries,
            }
        )

    def _original_topic_tokens(self, original_topic: str) -> list[str]:
        if not original_topic:
            return []

        lowered = original_topic.lower()
        raw_tokens = re.findall(r"[a-z0-9+#]+", lowered)

        result: list[str] = []

        for token in raw_tokens:
            if token and token not in result:
                result.append(token)

        return result

    def _is_user_typed_level_word(
        self,
        word: str,
        user_tokens: list[str],
    ) -> bool:
        if not word or not user_tokens:
            return False

        for token in user_tokens:
            if word == token:
                return True

            if tokens_match(word, token):
                return True

        return False

    def _strip_level_phrases(
        self,
        text: str,
        user_tokens: list[str],
    ) -> str:
        if not text:
            return text

        result = text

        for phrase, pattern in self._level_phrase_patterns:
            match = pattern.search(result)

            if match is None:
                continue

            phrase_tokens = [
                piece.strip().lower()
                for piece in phrase.split()
                if piece.strip()
            ]

            if phrase_tokens and all(
                self._is_user_typed_level_word(piece, user_tokens)
                for piece in phrase_tokens
            ):
                continue

            result = pattern.sub(" ", result)

        return re.sub(r"\s+", " ", result).strip()

    def _strip_intent_words_from_concept(
        self,
        analysis: QueryAnalysis,
    ) -> QueryAnalysis:
        if not self._intent_not_concept_words:
            return analysis

        user_content = set(
            self._extract_content_tokens(analysis.original_topic)
        )

        def clean_text_field(value: str) -> str:
            text = clean_text(value)

            if not text:
                return text

            tokens = text.split()
            kept: list[str] = []

            for token in tokens:
                lowered = token.strip(",.!?;:\"'()[]{}").lower()

                if not lowered:
                    kept.append(token)
                    continue

                if (
                    lowered in self._intent_not_concept_words
                    and lowered not in user_content
                ):
                    continue

                kept.append(token)

            return " ".join(kept).strip()

        def clean_keyword(value: str) -> str:
            text = clean_text(value)

            if not text:
                return ""

            tokens = text.split()
            kept: list[str] = []

            for token in tokens:
                lowered = token.strip(",.!?;:\"'()[]{}").lower()

                if not lowered:
                    continue

                if (
                    lowered in self._intent_not_concept_words
                    and lowered not in user_content
                ):
                    continue

                kept.append(token)

            if not kept:
                return ""

            return " ".join(kept).strip()

        new_primary = clean_text_field(analysis.primary_concept)
        new_corrected = clean_text_field(analysis.corrected_topic)

        if not new_primary.strip():
            new_primary = clean_text(analysis.primary_concept)

        if not new_corrected.strip():
            new_corrected = clean_text(analysis.corrected_topic)

        new_keywords: list[str] = []

        for keyword in analysis.keywords or []:
            cleaned = clean_keyword(keyword)

            if cleaned and cleaned not in new_keywords:
                new_keywords.append(cleaned)

        new_short_queries: list[str] = []

        for query in analysis.short_search_queries or []:
            cleaned = clean_text_field(query)

            if cleaned and cleaned not in new_short_queries:
                new_short_queries.append(cleaned)

        changed = (
            new_primary != analysis.primary_concept
            or new_corrected != analysis.corrected_topic
            or new_keywords != list(analysis.keywords or [])
            or new_short_queries != list(analysis.short_search_queries or [])
        )

        if not changed:
            return analysis

        self._trace.emit(
            "query_intent_words_stripped",
            original_topic=analysis.original_topic,
            old_primary=analysis.primary_concept,
            new_primary=new_primary,
            old_corrected=analysis.corrected_topic,
            new_corrected=new_corrected,
        )

        return analysis.model_copy(
            update={
                "primary_concept": new_primary,
                "corrected_topic": new_corrected,
                "keywords": new_keywords,
                "short_search_queries": new_short_queries,
                "suggested_queries": new_short_queries,
            }
        )

    def _extract_content_tokens(self, query: str) -> list[str]:
        if not query:
            return []

        lowered_query = query.lower()

        phrase_words: set[str] = set()

        for signal in (
            _STRONG_BEGINNER_SIGNALS
            + _MEDIUM_BEGINNER_SIGNALS
            + _WEAK_BEGINNER_SIGNALS
            + _STRONG_ADVANCED_SIGNALS
            + _MEDIUM_ADVANCED_SIGNALS
            + _WEAK_ADVANCED_SIGNALS
        ):
            text = str(signal or "").strip().lower()

            if not text or " " not in text:
                continue

            if text in lowered_query:
                for word in text.split():
                    cleaned_word = word.strip(",.!?;:\"'()[]{}")
                    if cleaned_word:
                        phrase_words.add(cleaned_word)

        generics = get_generic_terms()
        raw_tokens = re.findall(r"[A-Za-z0-9+#.]+", query)

        result: list[str] = []

        for raw_token in raw_tokens:
            text = raw_token.strip().strip(".,;:!?()[]{}'\"")
            if not text:
                continue

            lowered = text.lower()

            if len(lowered) < 2:
                continue

            if lowered in self._LEVEL_WORDS:
                continue

            if lowered in phrase_words:
                continue

            if lowered in generics:
                continue

            if self._is_adverb(lowered):
                continue

            is_acronym = (
                text.isupper()
                and 2 <= len(text) <= 6
                and any(ch.isalpha() for ch in text)
            )

            has_no_vowel = (
                len(lowered) >= 2
                and all(ch.isalpha() for ch in lowered)
                and not any(ch in "aeiouy" for ch in lowered)
            )

            if not is_acronym and not has_no_vowel and len(lowered) < 3:
                continue

            if is_acronym and lowered not in result:
                result.append(lowered)
                continue

            if has_no_vowel and lowered not in result:
                result.append(lowered)
                continue

            if lowered not in result:
                result.append(lowered)

        return result

    def _is_adverb(self, token: str) -> bool:
        if len(token) < 5:
            return False

        for suffix in self._ADVERB_SUFFIXES:
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                return True

        return False

    def _content_token_represented(
        self,
        token: str,
        concept_tokens: list[str],
    ) -> bool:
        if not token or not concept_tokens:
            return False

        for concept_token in concept_tokens:
            if tokens_match(token, concept_token):
                return True

        return False

    def _enforce_content_tokens(
        self,
        analysis: QueryAnalysis,
        content_tokens: list[str],
    ) -> QueryAnalysis:
        if not content_tokens:
            return analysis

        concept_tokens = self._collect_concept_tokens(analysis)

        missing: list[str] = []

        for token in content_tokens:
            if not self._content_token_represented(token, concept_tokens):
                missing.append(token)

        if not missing:
            existing = list(analysis.content_tokens or [])
            for token in content_tokens:
                if token not in existing:
                    existing.append(token)

            return analysis.model_copy(
                update={"content_tokens": existing}
            )

        corrected = clean_text(analysis.corrected_topic)
        primary = clean_text(analysis.primary_concept)
        keywords = list(analysis.keywords or [])

        prefix = " ".join(missing)

        if corrected:
            new_corrected = f"{prefix} {corrected}"
        else:
            new_corrected = prefix

        if primary:
            new_primary = f"{prefix} {primary}"
        else:
            new_primary = prefix

        new_keywords = list(keywords)

        for token in missing:
            if token.lower() not in [kw.lower() for kw in new_keywords]:
                new_keywords.append(token)

        new_primary = self._shorten_phrase(
            new_primary,
            int(self._config["max_primary_concept_words"]) + len(missing),
        )

        new_corrected = self._shorten_phrase(
            new_corrected,
            int(self._config["max_corrected_topic_words"]) + len(missing),
        )

        existing = list(analysis.content_tokens or [])

        for token in content_tokens:
            if token not in existing:
                existing.append(token)

        return analysis.model_copy(
            update={
                "corrected_topic": new_corrected,
                "primary_concept": new_primary,
                "keywords": new_keywords[
                    : int(self._config["max_keywords"]) + len(missing)
                ],
                "content_tokens": existing,
            }
        )

    def _collect_concept_tokens(self, analysis: QueryAnalysis) -> list[str]:
        result: list[str] = []

        for source_text in (analysis.primary_concept, analysis.corrected_topic):
            tokens = tokenize(source_text)

            for token in tokens:
                if token and token not in result:
                    result.append(token)

        for keyword in analysis.keywords or []:
            for token in tokenize(keyword):
                if token and token not in result:
                    result.append(token)

        return result

    def _extract_preserve_tokens(self, query: str) -> list[str]:
        if not query:
            return []

        found: list[str] = []
        seen: set[str] = set()

        for match in _YEAR_PATTERN.finditer(query):
            token = match.group(0)

            if token not in seen:
                seen.add(token)
                found.append(token)

        for match in _ARTICLE_PATTERN.finditer(query):
            token = match.group(0).strip()

            if token.lower() not in seen:
                seen.add(token.lower())
                found.append(token)

        for match in _ACRONYM_PATTERN.finditer(query):
            token = match.group(0).strip()

            if not token:
                continue

            lowered = token.lower()

            if lowered in _COMMON_SENTENCE_STARTERS:
                continue

            if lowered in seen:
                continue

            seen.add(lowered)
            found.append(token)

        for match in _MULTIWORD_PROPER_PATTERN.finditer(query):
            phrase = match.group(0).strip()
            tokens = phrase.split()

            if tokens and tokens[0].lower() in _COMMON_SENTENCE_STARTERS:
                tokens = tokens[1:]

            for token in tokens:
                lowered = token.lower()

                if lowered in _COMMON_SENTENCE_STARTERS:
                    continue

                if lowered in seen:
                    continue

                seen.add(lowered)
                found.append(token)

        for match in _TITLE_WORD_PATTERN.finditer(query):
            token = match.group(0)
            lowered = token.lower()

            start_index = match.start()

            if start_index == 0:
                continue

            previous = query[max(0, start_index - 2):start_index].strip()

            if previous.endswith((".", "!", "?")):
                continue

            if lowered in _COMMON_SENTENCE_STARTERS:
                continue

            if lowered in seen:
                continue

            seen.add(lowered)
            found.append(token)

        lower_query = query.lower()
        raw_words = re.findall(r"[a-z]+", lower_query)

        for word in raw_words:
            if word in _COMMON_PROPER_NOUNS and word not in seen:
                seen.add(word)
                found.append(word)

        return found

    def _enforce_preserve_tokens(
        self,
        analysis: QueryAnalysis,
        preserve_tokens: list[str],
    ) -> QueryAnalysis:
        if not preserve_tokens:
            return analysis

        existing_preserve = list(analysis.preserve_tokens or [])

        for token in preserve_tokens:
            if token and token not in existing_preserve:
                existing_preserve.append(token)

        corrected = clean_text(analysis.corrected_topic)
        primary = clean_text(analysis.primary_concept)
        keywords = list(analysis.keywords or [])

        combined_existing = " ".join(
            [corrected, primary, " ".join(keywords)]
        ).lower()

        missing: list[str] = []

        for token in preserve_tokens:
            text = str(token or "").strip()

            if not text:
                continue

            lowered = text.lower()

            if lowered in combined_existing:
                continue

            missing.append(text)

        if not missing:
            return analysis.model_copy(
                update={"preserve_tokens": existing_preserve}
            )

        prefix = " ".join(missing)

        if corrected:
            new_corrected = f"{prefix} {corrected}"
        else:
            new_corrected = prefix

        if primary:
            new_primary = f"{prefix} {primary}"
        else:
            new_primary = prefix

        new_keywords = list(keywords)

        for token in missing:
            if token.lower() not in [kw.lower() for kw in new_keywords]:
                new_keywords.append(token)

        new_primary = self._shorten_phrase(
            new_primary,
            int(self._config["max_primary_concept_words"]) + len(missing),
        )

        new_corrected = self._shorten_phrase(
            new_corrected,
            int(self._config["max_corrected_topic_words"]) + len(missing),
        )

        return analysis.model_copy(
            update={
                "corrected_topic": new_corrected,
                "primary_concept": new_primary,
                "keywords": new_keywords[
                    : int(self._config["max_keywords"]) + len(missing)
                ],
                "preserve_tokens": existing_preserve,
            }
        )

    def _load_config(self) -> dict[str, Any]:
        config = dict(self._DEFAULT_CONFIG)

        try:
            import yaml

            config_path = (
                Path(__file__).resolve().parent.parent
                / "configs"
                / "settings.yaml"
            )

            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as handle:
                    loaded = yaml.safe_load(handle) or {}

                section = loaded.get("query_intelligence", {})

                if isinstance(section, dict):
                    for key in config:
                        if section.get(key) is None:
                            continue

                        if isinstance(config[key], int):
                            config[key] = self._as_int(
                                section[key],
                                config[key],
                            )
                        elif isinstance(config[key], float):
                            config[key] = self._as_float(
                                section[key],
                                config[key],
                            )
                        else:
                            config[key] = section[key]
        except Exception:
            pass

        return config

    def _load_domain_rules(self) -> dict[str, Any]:
        fallback = self._fallback_domain_rules()

        try:
            import yaml

            config_path = (
                Path(__file__).resolve().parent.parent
                / "configs"
                / "domains.yaml"
            )

            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as handle:
                    loaded = yaml.safe_load(handle) or {}

                if (
                    isinstance(loaded, dict)
                    and isinstance(loaded.get("domains"), dict)
                ):
                    defaults = dict(fallback["defaults"])

                    if isinstance(loaded.get("defaults"), dict):
                        defaults.update(loaded["defaults"])

                    loaded["defaults"] = defaults
                    return loaded
        except Exception:
            pass

        return fallback

    def _fallback_domain_rules(self) -> dict[str, Any]:
        return {
            "defaults": {
                "max_domains": 4,
                "min_domain_confidence": 0.25,
                "max_formats": 6,
                "universal_platforms": [
                    "wikipedia",
                    "open_library",
                    "internet_archive",
                    "wikibooks",
                    "openstax",
                    "libretexts",
                    "wikiversity",
                    "mit_ocw",
                ],
                "beginner_formats": [
                    "course",
                    "book",
                    "documentation",
                    "video",
                    "research_paper",
                ],
                "intermediate_formats": [
                    "documentation",
                    "course",
                    "book",
                    "research_paper",
                    "repository",
                ],
                "advanced_formats": [
                    "research_paper",
                    "repository",
                    "model",
                    "dataset",
                    "documentation",
                ],
            },
            "domains": {
                "general": {
                    "label": "General",
                    "priority": 100,
                    "aliases": [
                        "general",
                        "general knowledge",
                        "overview",
                    ],
                },
                "education": {
                    "label": "Education",
                    "priority": 20,
                    "aliases": [
                        "education",
                        "learn",
                        "learning",
                        "course",
                        "tutorial",
                        "basics",
                        "beginner",
                        "fundamentals",
                        "textbook",
                        "curriculum",
                    ],
                },
            },
        }

    @staticmethod
    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return default

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _domain_defaults(self) -> dict[str, Any]:
        defaults = self._domain_rules.get("defaults", {})

        if not isinstance(defaults, dict):
            return {}

        return defaults

    def _domain_configs(self) -> dict[str, Any]:
        domains = self._domain_rules.get("domains", {})

        if not isinstance(domains, dict):
            return {}

        return domains

    def _local_analysis(
        self,
        topic: str,
        goal: str,
        level: str,
    ) -> QueryAnalysis:
        cleaned = clean_text(topic)
        stripped = strip_filler(cleaned) or cleaned

        primary_concept = self._extract_primary_concept(stripped)
        keywords = self._extract_keywords(primary_concept or stripped)

        corrected_topic = self._shorten_phrase(
            primary_concept or stripped,
            int(self._config["max_corrected_topic_words"]),
        )

        level_priority = self._detect_level_priority(topic, goal, level)
        learner_level = self._resolve_learner_level(
            level_priority,
            goal,
            level,
        )

        (
            profile,
            ratio,
            min_beginner,
            exclude_research,
        ) = self._detect_level_profile(
            topic=topic,
            goal=goal,
            level=level,
            level_priority=level_priority,
        )

        target_domains, domain_confidences = self._detect_domains(
            topic=cleaned,
            goal=goal,
            primary_concept=primary_concept,
            keywords=keywords,
        )

        target_formats = self._detect_formats(
            topic=cleaned,
            goal=goal,
            level_priority=level_priority,
            domains=target_domains,
        )

        domain_confidence = 0.0

        if domain_confidences:
            domain_confidence = max(domain_confidences.values())

        short_search_queries = self._build_short_queries(
            corrected_topic,
            primary_concept,
            keywords,
        )

        return QueryAnalysis(
            original_topic=topic,
            cleaned_topic=cleaned,
            corrected_topic=corrected_topic,
            primary_concept=primary_concept,
            keywords=keywords,
            intent=self._local_intent(topic),
            level=learner_level,
            learner_level=learner_level,
            level_priority=level_priority,
            intent_confidence=0.0,
            spell_corrections={},
            short_search_queries=short_search_queries,
            suggested_queries=short_search_queries,
            target_domains=target_domains,
            domain_confidences=domain_confidences,
            domain_confidence=domain_confidence,
            target_formats=target_formats,
            level_profile=profile,
            beginner_ratio=ratio,
            min_beginner_sources=min_beginner,
            exclude_research_platforms=exclude_research,
            preserve_tokens=[],
            content_tokens=[],
            llm_used=False,
        )

    def _weighted_level_scores(self, text: str) -> tuple[float, float]:
        if not text:
            return 0.0, 0.0

        beginner_score = 0.0
        advanced_score = 0.0

        for signal in _STRONG_BEGINNER_SIGNALS:
            if signal in text:
                beginner_score += 1.0

        for signal in _MEDIUM_BEGINNER_SIGNALS:
            if signal in text:
                beginner_score += 0.6

        for signal in _WEAK_BEGINNER_SIGNALS:
            if signal in text:
                beginner_score += 0.3

        for signal in _STRONG_ADVANCED_SIGNALS:
            if signal in text:
                advanced_score += 1.0

        for signal in _MEDIUM_ADVANCED_SIGNALS:
            if signal in text:
                advanced_score += 0.6

        for signal in _WEAK_ADVANCED_SIGNALS:
            if signal in text:
                advanced_score += 0.3

        for phrase in _ADVANCED_FRAMING_PHRASES:
            if phrase in text:
                advanced_score += 0.8

        return beginner_score, advanced_score

    def _detect_level_profile(
        self,
        topic: str,
        goal: str,
        level: str,
        level_priority: list[str],
    ) -> tuple[str, float, int, bool]:
        text = " ".join(
            [
                clean_text(topic).lower(),
                clean_text(goal).lower(),
                clean_text(level).lower(),
            ]
        ).strip()

        if not text:
            return "balanced", 0.4, 2, False

        beginner_score, advanced_score = self._weighted_level_scores(text)

        if advanced_score >= 1.0 and beginner_score < advanced_score:
            return "advanced", 0.1, 0, False

        if beginner_score >= 1.0 and advanced_score < beginner_score:
            return "beginner", 0.9, 5, True

        if advanced_score >= 1.0 and beginner_score >= 1.0:
            return "mixed", 0.5, 3, False

        if beginner_score > advanced_score * 1.2 and beginner_score > 0:
            return "beginner", 0.85, 4, True

        if advanced_score > beginner_score * 1.2 and advanced_score > 0:
            return "advanced", 0.1, 0, False

        if level_priority:
            first = str(level_priority[0]).strip().lower()

            if first == "beginner":
                return "beginner", 0.85, 4, True

            if first == "advanced":
                return "advanced", 0.15, 0, False

            if first == "intermediate":
                return "balanced", 0.45, 3, False

        return "balanced", 0.4, 2, False

    def _emit_profile_trace(
        self,
        analysis: QueryAnalysis,
        source: str,
    ) -> None:
        try:
            self._trace.emit(
                "query_level_profile",
                source=source,
                original_topic=analysis.original_topic,
                corrected_topic=analysis.corrected_topic,
                primary_concept=analysis.primary_concept,
                preserve_tokens=analysis.preserve_tokens,
                content_tokens=analysis.content_tokens,
                level_profile=analysis.level_profile,
                beginner_ratio=analysis.beginner_ratio,
                min_beginner_sources=analysis.min_beginner_sources,
                exclude_research_platforms=analysis.exclude_research_platforms,
                level_priority=analysis.level_priority,
                learner_level=analysis.learner_level,
            )
        except Exception:
            pass

    def _extract_primary_concept(self, value: Any) -> str:
        text = clean_text(value)

        if not text:
            return ""

        text = strip_filler(text).lower()
        text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        words = text.split()

        if not words:
            return ""

        return " ".join(
            words[: int(self._config["max_primary_concept_words"])]
        )

    def _extract_keywords(self, value: Any) -> list[str]:
        text = clean_text(value)

        if not text:
            return []

        tokens = tokenize(text.lower())
        generic_terms = get_generic_terms()

        keywords: list[str] = []

        for token in tokens:
            if len(token) < 2:
                continue

            if token in generic_terms:
                continue

            if token not in keywords:
                keywords.append(token)

            if len(keywords) >= int(self._config["max_keywords"]):
                break

        return keywords

    def _shorten_phrase(self, value: Any, max_words: int) -> str:
        text = clean_text(value).lower()

        if not text:
            return ""

        text = strip_filler(text)
        text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        words = text.split()

        if not words:
            return ""

        return " ".join(words[:max_words])

    def _detect_level_priority(
        self,
        topic: str,
        goal: str,
        level: str,
    ) -> list[str]:
        text = " ".join(
            [
                clean_text(topic).lower(),
                clean_text(goal).lower(),
                clean_text(level).lower(),
            ]
        )

        beginner_score, advanced_score = self._weighted_level_scores(text)

        if beginner_score >= 1.0 and advanced_score >= 1.0:
            return ["beginner", "intermediate", "advanced"]

        if advanced_score >= 1.0 and beginner_score < advanced_score:
            return ["advanced"]

        if beginner_score >= 1.0 and advanced_score < beginner_score:
            return ["beginner"]

        normalized_level = clean_text(level).lower()

        if normalized_level in {
            "beginner",
            "intermediate",
            "advanced",
            "mixed",
        }:
            return [normalized_level]

        if "to" in normalized_level:
            if "beginner" in normalized_level and (
                "advanced" in normalized_level
                or "advance" in normalized_level
            ):
                return ["beginner", "intermediate", "advanced"]

        return ["mixed"]

    def _resolve_learner_level(
        self,
        level_priority: list[str],
        goal: str,
        level: str,
    ) -> str:
        if level_priority:
            first = str(level_priority[0]).strip().lower()

            if first in {"beginner", "intermediate", "advanced"}:
                return first

        inferred = self._infer_level(goal, level)

        if inferred in {"beginner", "intermediate", "advanced"}:
            return inferred

        return "mixed"

    def _infer_level(self, goal: str, level: str) -> str:
        explicit = clean_text(level).lower()

        if explicit in {"beginner", "intermediate", "advanced"}:
            return explicit

        if explicit == "mixed":
            return "mixed"

        text = f"{clean_text(goal).lower()} {explicit}"

        beginner_score, advanced_score = self._weighted_level_scores(text)

        if beginner_score > 0 and advanced_score > 0:
            return "mixed"

        if beginner_score > 0:
            return "beginner"

        if advanced_score > 0:
            return "advanced"

        return "mixed"

    def _learning_intent(self, text: str) -> bool:
        learning_signals = (
            "learn", "learning", "study", "studies",
            "course", "courses", "tutorial", "tutorials",
            "basics", "basic", "beginner",
            "fundamentals", "fundamental",
            "teach", "teaching",
            "curriculum", "syllabus",
            "guide", "roadmap",
            "from scratch", "step by step",
        )

        return any(signal in text for signal in learning_signals)
    def _local_intent(self, topic: str) -> QueryIntent:
        text = str(topic or "").lower().strip()

        if not text:
            return QueryIntent.EXPLORE

        research_markers = (
            "literature review",
            "systematic review",
            "meta-analysis",
            "state of the art",
            "state-of-the-art",
            "survey of",
            "review of the",
            "review paper",
        )

        troubleshoot_markers = (
            "not working",
            "does not work",
            "broken",
            "debug",
            "error",
            "issue with",
            "fix my",
            "troubleshoot",
        )

        compare_markers = (
            "compare",
            "difference between",
            " versus ",
            " vs ",
            " vs. ",
            "better than",
        )

        implement_markers = (
            "implement",
            "build a",
            "build an",
            "write a",
            "write an",
            "code a",
            "develop a",
            "develop an",
            "create a",
            "create an",
        )

        learn_markers = (
            "learn",
            "study",
            "understand",
            "explain",
            "teach me",
            "tutorial",
            "lesson",
            "walk me through",
            "introduction to",
            "from basics",
            "from scratch",
            "step by step",
        )

        if any(marker in text for marker in research_markers):
            return QueryIntent.RESEARCH

        if any(marker in text for marker in troubleshoot_markers):
            return QueryIntent.TROUBLESHOOT

        if any(marker in text for marker in compare_markers):
            return QueryIntent.COMPARE

        if any(marker in text for marker in implement_markers):
            return QueryIntent.IMPLEMENT

        if any(marker in text for marker in learn_markers):
            return QueryIntent.LEARN

        return QueryIntent.EXPLORE

    def _normalize_domain_name(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        text = text.replace("-", "_").replace(" ", "_")
        text = "".join(ch for ch in text if ch.isalnum() or ch == "_")
        text = "_".join(part for part in text.split("_") if part)
        return text

    def _alias_in_text(
        self,
        alias: str,
        text: str,
        tokens: set[str],
    ) -> bool:
        alias = clean_text(alias).lower()

        if not alias:
            return False

        alias_tokens = set(re.findall(r"[a-z0-9+#]+", alias))

        if len(alias) <= 3:
            return alias in tokens

        if len(alias_tokens) == 1 and list(alias_tokens)[0] in tokens:
            return True

        try:
            pattern = (
                r"(?:^|[^a-z0-9])"
                + re.escape(alias)
                + r"(?:[^a-z0-9]|$)"
            )
            return re.search(pattern, text) is not None
        except Exception:
            return alias in text

    def _detect_domains(
        self,
        topic: str,
        goal: str,
        primary_concept: str,
        keywords: list[str],
    ) -> tuple[list[str], dict[str, float]]:
        text = " ".join(
            [
                clean_text(topic).lower(),
                clean_text(goal).lower(),
                clean_text(primary_concept).lower(),
                " ".join(keywords).lower(),
            ]
        )

        tokens = set(re.findall(r"[a-z0-9+#]+", text))
        domain_configs = self._domain_configs()
        defaults = self._domain_defaults()

        min_confidence = self._as_float(
            defaults.get("min_domain_confidence"),
            self._as_float(
                self._config.get("min_domain_confidence"),
                0.25,
            ),
        )

        max_domains = self._as_int(
            defaults.get("max_domains"),
            self._as_int(self._config.get("max_domains"), 4),
        )

        scores: dict[str, float] = {}

        for domain, config in domain_configs.items():
            if not isinstance(config, dict):
                continue

            domain_key = self._normalize_domain_name(domain)
            aliases: list[str] = [domain_key]

            label = str(config.get("label", "")).strip().lower()

            if label:
                aliases.append(self._normalize_domain_name(label))

            raw_aliases = config.get("aliases", [])

            if isinstance(raw_aliases, list):
                for alias in raw_aliases:
                    normalized_alias = self._normalize_domain_name(alias)

                    if normalized_alias:
                        aliases.append(normalized_alias)

            score = 0.0
            matched = False

            for alias in aliases:
                if not alias:
                    continue

                if self._alias_in_text(alias, text, tokens):
                    matched = True

                    if len(alias.split("_")) >= 2:
                        score += 0.48
                    elif len(alias) <= 3:
                        score += 0.30
                    else:
                        score += 0.38

                    if primary_concept and alias in primary_concept.lower():
                        score += 0.22

                    if any(alias in keyword.lower() for keyword in keywords):
                        score += 0.10

            if matched:
                scores[domain_key] = min(0.98, score)

        detected = {
            domain: score
            for domain, score in scores.items()
            if score >= min_confidence
        }

        learning_intent = self._learning_intent(text)

        subject_domains = [
            domain
            for domain in detected
            if domain not in {"education", "general"}
        ]

        if learning_intent and subject_domains and "education" not in detected:
            detected["education"] = 0.55

        if not detected:
            detected = {"general": 0.20}

        ordered: list[str] = []

        if subject_domains:
            ordered = sorted(
                subject_domains,
                key=lambda item: (
                    -detected.get(item, 0.0),
                    self._as_int(
                        domain_configs.get(item, {}).get("priority"),
                        100,
                    ),
                ),
            )

            if "education" in detected:
                ordered.append("education")
        else:
            ordered = sorted(
                detected.keys(),
                key=lambda item: (
                    -detected.get(item, 0.0),
                    self._as_int(
                        domain_configs.get(item, {}).get("priority"),
                        100,
                    ),
                ),
            )

        intent_only_domains = {"education", "general"}

        subject_ordered = [
            domain for domain in ordered if domain not in intent_only_domains
        ]

        if not subject_ordered:
            return [], {}

        subject_ordered = subject_ordered[:max_domains]

        confidences = {
            domain: round(float(detected.get(domain, 0.20)), 4)
            for domain in subject_ordered
        }

        return subject_ordered, confidences

    def _allowed_formats(self) -> list[str]:
        defaults = self._domain_defaults()
        allowed: list[str] = []

        for key in (
            "beginner_formats",
            "intermediate_formats",
            "advanced_formats",
        ):
            values = defaults.get(key, [])

            if isinstance(values, list):
                for value in values:
                    normalized = self._normalize_format_name(value)

                    if normalized and normalized not in allowed:
                        allowed.append(normalized)

        common = [
            "course",
            "book",
            "documentation",
            "video",
            "research_paper",
            "repository",
            "model",
            "dataset",
        ]

        for value in common:
            if value not in allowed:
                allowed.append(value)

        return allowed

    def _normalize_format_name(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        text = text.replace("-", "_").replace(" ", "_")

        aliases = {
            "docs": "documentation",
            "doc": "documentation",
            "wiki": "documentation",
            "paper": "research_paper",
            "papers": "research_paper",
            "article": "research_paper",
            "articles": "research_paper",
            "repo": "repository",
            "repos": "repository",
            "code": "repository",
            "textbook": "book",
            "textbooks": "book",
            "tutorial": "course",
            "tutorials": "course",
            "lectures": "video",
            "lecture": "video",
        }

        if text in aliases:
            return aliases[text]

        text = "".join(ch for ch in text if ch.isalnum() or ch == "_")
        text = "_".join(part for part in text.split("_") if part)

        return text

    def _detect_formats(
        self,
        topic: str,
        goal: str,
        level_priority: list[str],
        domains: list[str],
    ) -> list[str]:
        defaults = self._domain_defaults()

        beginner_formats = defaults.get("beginner_formats", [])
        intermediate_formats = defaults.get("intermediate_formats", [])
        advanced_formats = defaults.get("advanced_formats", [])

        if not isinstance(beginner_formats, list):
            beginner_formats = []

        if not isinstance(intermediate_formats, list):
            intermediate_formats = []

        if not isinstance(advanced_formats, list):
            advanced_formats = []

        formats: list[str] = []

        if "beginner" in level_priority and len(level_priority) == 1:
            formats.extend(beginner_formats)
        elif "advanced" in level_priority and len(level_priority) == 1:
            formats.extend(advanced_formats)
        elif "intermediate" in level_priority and len(level_priority) == 1:
            formats.extend(intermediate_formats)
        else:
            formats.extend(beginner_formats)
            formats.extend(intermediate_formats)
            formats.extend(advanced_formats)

        domain_configs = self._domain_configs()

        for domain in domains[:2]:
            config = domain_configs.get(domain, {})

            if not isinstance(config, dict):
                continue

            domain_formats = config.get("formats", [])

            if isinstance(domain_formats, list):
                for value in domain_formats:
                    if value not in formats:
                        formats.append(value)

        text = " ".join(
            [
                clean_text(topic).lower(),
                clean_text(goal).lower(),
            ]
        )

        format_signals = (
            ("course", "course"),
            ("courses", "course"),
            ("tutorial", "course"),
            ("tutorials", "course"),
            ("lecture", "video"),
            ("lectures", "video"),
            ("video", "video"),
            ("videos", "video"),
            ("youtube", "video"),
            ("textbook", "book"),
            ("textbooks", "book"),
            ("book", "book"),
            ("books", "book"),
            ("documentation", "documentation"),
            ("docs", "documentation"),
            ("guide", "documentation"),
            ("guides", "documentation"),
            ("wiki", "documentation"),
            ("wikipedia", "documentation"),
            ("paper", "research_paper"),
            ("papers", "research_paper"),
            ("research", "research_paper"),
            ("study", "research_paper"),
            ("studies", "research_paper"),
            ("journal", "research_paper"),
            ("article", "research_paper"),
            ("articles", "research_paper"),
            ("code", "repository"),
            ("implementation", "repository"),
            ("github", "repository"),
            ("repository", "repository"),
            ("model", "model"),
            ("models", "model"),
            ("dataset", "dataset"),
            ("datasets", "dataset"),
        )

        for signal, normalized in format_signals:
            if signal in text:
                normalized = self._normalize_format_name(normalized)

                if normalized and normalized not in formats:
                    formats.insert(0, normalized)

        cleaned: list[str] = []

        for value in formats:
            normalized = self._normalize_format_name(value)

            if normalized and normalized not in cleaned:
                cleaned.append(normalized)

        max_formats = self._as_int(
            defaults.get("max_formats"),
            self._as_int(self._config.get("max_formats"), 6),
        )

        return cleaned[:max_formats]

    def _build_short_queries(
        self,
        corrected_topic: str,
        primary_concept: str,
        keywords: list[str],
    ) -> list[str]:
        queries: list[str] = []

        candidates = [corrected_topic, primary_concept]

        if keywords:
            candidates.append(" ".join(keywords[:4]))

        if len(keywords) >= 2:
            candidates.append(" ".join(keywords[:2]))

        if len(keywords) >= 3:
            candidates.append(" ".join(keywords[1:4]))

        if primary_concept and keywords:
            candidates.append(f"{primary_concept} {keywords[0]}")

        for candidate in candidates:
            cleaned = self._clean_query(candidate)

            if cleaned and cleaned not in queries:
                queries.append(cleaned)

            if len(queries) >= int(self._config["max_short_queries"]):
                break

        return queries

    def _clean_query(self, value: Any) -> str:
        text = clean_text(value).lower()

        if not text:
            return ""

        text = strip_filler(text)
        text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        words = text.split()
        max_words = int(self._config["max_query_words"])

        if len(words) > max_words:
            words = words[:max_words]

        text = " ".join(words)

        return truncate_text(
            text,
            max_length=constants.MAX_SEARCH_QUERY_LENGTH,
            suffix="",
        )

    async def _llm_enhance(
        self,
        analysis: QueryAnalysis,
        goal: str,
        level: str,
    ) -> QueryAnalysis | None:
        try:
            from core.schemas import LLMMessageSchema, LLMRequestSchema
            from llm.guardrails import validate_llm_request
            from llm.parser import parse_json_object_response
            from llm.provider import LLMProviderManager
            from llm.router import get_fast_model_references

            model_references = get_fast_model_references(
                limit=int(self._config["model_limit"])
            )

            if not model_references:
                return None

            allowed_domains = list(self._domain_configs().keys())
            allowed_formats = self._allowed_formats()

            expected_output = {
                "corrected_topic": "short corrected topic",
                "primary_concept": "short primary concept",
                "keywords": ["keyword one", "keyword two"],
                "intent": (
                    "learn | research | implement | compare | "
                    "troubleshoot | explore"
                ),
                "level": "beginner | intermediate | advanced | mixed",
                "learner_level": (
                    "beginner | intermediate | advanced | mixed"
                ),
                "level_priority": [
                    "beginner",
                    "intermediate",
                    "advanced",
                ],
                "level_profile": "beginner | mixed | advanced | balanced",
                "beginner_ratio": 0.9,
                "min_beginner_sources": 5,
                "exclude_research_platforms": True,
                "target_domains": ["chemistry", "education"],
                "domain_confidences": {
                    "chemistry": 0.92,
                    "education": 0.78,
                },
                "domain_confidence": 0.92,
                "target_formats": [
                    "course",
                    "book",
                    "documentation",
                ],
                "short_search_queries": [
                    "short query one",
                    "short query two",
                ],
                "preserve_tokens": ["iraq", "1861", "tcp", "udp"],
            }

            system_prompt = (
                "You are a query understanding engine. "
                "Convert any user request into short, clean, searchable "
                "concepts. "
                "Detect subject domains, learner level, level priority, "
                "and preferred source formats. "
                "Correct spelling mistakes. "
                "Preserve all named entities, proper nouns, geographic "
                "names, dates, technical acronyms, and organisation names. "
                "Do not invent facts. "
                "Return only valid JSON."
            )

            user_prompt = (
                f"Original topic: {analysis.original_topic}"
                f"Goal: {goal}\n"
                f"Level: {level}\n"
                f"Allowed domains: "
                f"{json.dumps(allowed_domains, ensure_ascii=False)}\n"
                f"Allowed formats: "
                f"{json.dumps(allowed_formats, ensure_ascii=False)}\n"
                "Rules:\n"
                f"- corrected_topic must be 2 to "
                f"{self._config['max_corrected_topic_words']} "
                "short keywords.\n"
                f"- primary_concept must be 2 to "
                f"{self._config['max_primary_concept_words']} "
                "short keywords.\n"
                f"- keywords must contain 2 to "
                f"{self._config['max_keywords']} "
                "short search keywords.\n"
                f"- short_search_queries must contain 2 to "
                f"{self._config['max_short_queries']} short queries.\n"
                f"- Every query must be 2 to "
                f"{self._config['max_query_words']} words.\n"
                "- target_domains must only use allowed domains.\n"
                "- domain_confidences must map each detected domain to "
                "a number between 0 and 1.\n"
                "- domain_confidence must be the highest domain "
                "confidence.\n"
                "- target_formats must only use allowed formats.\n"
                "- level_priority must put the most important learner "
                "level first.\n"
                "- If the user says 'basics to advance' or 'beginner to "
                "advanced', level_priority must be beginner, "
                "intermediate, advanced, and level_profile must be "
                "'mixed'.\n"
                "- If the user asks to learn basics only, learner_level "
                "must be beginner and level_profile must be 'beginner'.\n"
                "- level_profile must be one of: beginner, mixed, "
                "advanced, balanced.\n"
                "- If the user wants only beginner material, "
                "level_profile is beginner, beginner_ratio is at least "
                "0.8, min_beginner_sources is 5, "
                "exclude_research_platforms is true.\n"
                "- If the user wants both beginner and advanced "
                "material, level_profile is mixed, beginner_ratio is "
                "0.5, min_beginner_sources is 3, "
                "exclude_research_platforms is false.\n"
                "- If the user wants only advanced material, "
                "level_profile is advanced, beginner_ratio is at most "
                "0.2, min_beginner_sources is 0, "
                "exclude_research_platforms is false.\n"
                "- If the user does not express a preference, "
                "level_profile is balanced, beginner_ratio is 0.4, "
                "min_beginner_sources is 2, "
                "exclude_research_platforms is false.\n"
                "- beginner_ratio must be a number between 0 and 1.\n"
                "- min_beginner_sources must be an integer between 0 "
                "and 20.\n"
                "Named entity rules:\n"
                "- Do not drop proper nouns, geographic names, "
                "organisation names, dates, or named entities from the "
                "user's query.\n"
                "- If the user mentions a country, city, year, law, "
                "treaty, or organisation, keep it in primary_concept "
                "and corrected_topic.\n"
                "- preserve_tokens must list every named entity, proper "
                "noun, geographic name, year, and organisation name "
                "from the user's query.\n"
                "Technical acronym rules:\n"
                "- Preserve every technical acronym, protocol name, and "
                "multi-letter identifier exactly as the user wrote it.\n"
                "- Examples: TCP, UDP, HTTP, HTTPS, SQL, JSON, XML, "
                "HTML, API, OS, ML, AI, LLM, GPU, CPU, RAM, DNS, "
                "TCP/IP, I/O, OSI.\n"
                "- Do not generalise an acronym to its parent category. "
                "TCP and UDP are not 'network protocol'. They are "
                "'TCP' and 'UDP'.\n"
                "- If the user writes 'difference between TCP and UDP', "
                "keep both acronyms in primary_concept.\n"
                "- Acronyms must appear in preserve_tokens.\n"
                "Concept purity rules:\n"
                "- Do not add intent words to the concept. Words like "
                "education, learning, study, research, teaching, "
                "training, tutorial, course, lesson, curriculum, "
                "school, student, teacher, academic describe the "
                "user's intent or the type of material, not the "
                "subject. Do not put them in primary_concept or "
                "corrected_topic.\n"
                "- The subject is only what the user wants to learn "
                "about.\n"
                "- Example: user query 'learn python' must produce "
                "primary_concept 'python programming', not 'python "
                "education'.\n"
                "- Example: user query 'study chemistry for beginners' "
                "must produce primary_concept 'chemistry', not "
                "'chemistry education'.\n"
                "- Example: user query 'learn machine learning' must "
                "produce primary_concept 'machine learning', not "
                "'machine learning education'.\n"
                "- Example: user query 'learn python from zero to hero' "
                "must produce primary_concept 'python programming', "
                "level_profile 'beginner'.\n"
                "Level rules:\n"
                "- Do not classify as beginner unless the user's query "
                "contains at least one of: beginner, basic, basics, "
                "fundamental, introduction, intro, from scratch, for "
                "beginners, no prior, step by step, easy, simple, from "
                "zero, zero to hero, complete beginner, absolute "
                "beginner, no experience.\n"
                "- If the user's query contains the words deeply, "
                "thorough, comprehensive, rigorous, mathematical, or "
                "theoretical, or phrases like 'mathematical foundations' "
                "or 'principles of', prefer advanced or mixed, never "
                "beginner.\n"
                "- If the user's query contains 'from zero', 'zero to "
                "hero', 'complete beginner', or 'no experience', "
                "level_profile must be beginner.\n"
                "- If the query contains both beginner and advanced "
                "signals, set level_profile to mixed.\n"
                "- Do not return full sentences.\n"
                "- Do not use phrases like I want to learn.\n"
                f"Expected JSON output:\n"
                f"{json.dumps(expected_output, ensure_ascii=False, indent=2)}\n"
                "Return only valid JSON."
            )

            messages = [
                LLMMessageSchema(role="system", content=system_prompt),
                LLMMessageSchema(role="user", content=user_prompt),
            ]

            async with LLMProviderManager() as manager:
                for model_reference in model_references:
                    try:
                        request = LLMRequestSchema(
                            provider=model_reference.provider,
                            model=model_reference.model_id,
                            messages=messages,
                            temperature=float(self._config["temperature"]),
                            max_tokens=int(self._config["max_tokens"]),
                            response_format="json_object",
                        )

                        request = validate_llm_request(request)
                        response = await manager.complete(request)

                        if response is None:
                            continue

                        payload = parse_json_object_response(
                            response.content
                        )

                        return self._apply_llm_payload(analysis, payload)
                    except Exception:
                        continue

            return None
        except Exception as exc:
            self._logger.warning(
                f"LLM query enhancement failed: {exc}"
            )
            return None

    def _normalize_llm_domains(self, value: Any) -> list[str]:
        if value is None:
            return []

        if isinstance(value, str):
            value = [item.strip() for item in value.split(",")]

        if not isinstance(value, (list, tuple, set)):
            value = [value]

        domain_configs = self._domain_configs()
        result: list[str] = []

        for item in value:
            normalized = self._normalize_domain_name(item)

            if not normalized:
                continue

            if (
                normalized in domain_configs
                or normalized in {"general", "education"}
            ):
                if normalized not in result:
                    result.append(normalized)

        return result

    def _normalize_llm_formats(self, value: Any) -> list[str]:
        if value is None:
            return []

        if isinstance(value, str):
            value = [item.strip() for item in value.split(",")]

        if not isinstance(value, (list, tuple, set)):
            value = [value]

        allowed = set(self._allowed_formats())
        result: list[str] = []

        for item in value:
            normalized = self._normalize_format_name(item)

            if normalized and normalized in allowed and normalized not in result:
                result.append(normalized)

        return result

    def _normalize_llm_level_priority(
        self,
        value: Any,
        fallback: list[str],
    ) -> list[str]:
        if value is None:
            return fallback

        if isinstance(value, str):
            text = value.lower()

            if "beginner" in text and (
                "advanced" in text or "advance" in text
            ):
                return ["beginner", "intermediate", "advanced"]

            for level in ("beginner", "intermediate", "advanced", "mixed"):
                if level in text:
                    return [level]

            return fallback

        if not isinstance(value, (list, tuple, set)):
            return fallback

        result: list[str] = []

        for item in value:
            level = str(item or "").strip().lower()

            if level in {
                "beginner",
                "intermediate",
                "advanced",
                "mixed",
            }:
                if level not in result:
                    result.append(level)

        return result or fallback

    def _normalize_level_value(self, value: Any) -> str:
        text = str(value or "").strip().lower()

        if text in {"beginner", "intermediate", "advanced", "mixed"}:
            return text

        return ""

    def _normalize_llm_level_profile(
        self,
        value: Any,
        fallback: str,
    ) -> str:
        text = str(value or "").strip().lower()

        if text in constants.VALID_LEVEL_PROFILES:
            return text

        return fallback

    def _normalize_llm_ratio(self, value: Any, fallback: float) -> float:
        try:
            numeric = float(value)
        except Exception:
            return fallback

        return max(0.0, min(1.0, numeric))

    def _normalize_llm_int(
        self,
        value: Any,
        fallback: int,
        lower: int = 0,
        upper: int = 20,
    ) -> int:
        try:
            numeric = int(value)
        except Exception:
            return fallback

        return max(lower, min(upper, numeric))

    def _normalize_llm_bool(self, value: Any, fallback: bool) -> bool:
        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return bool(value)

        if isinstance(value, str):
            text = value.strip().lower()

            if text in {"true", "yes", "on", "1"}:
                return True

            if text in {"false", "no", "off", "0"}:
                return False

        return fallback

    def _normalize_llm_preserve_tokens(self, value: Any) -> list[str]:
        if value is None:
            return []

        if isinstance(value, str):
            value = [value]

        if not isinstance(value, (list, tuple, set)):
            return []

        result: list[str] = []

        for item in value:
            text = str(item or "").strip()

            if text and text not in result:
                result.append(text)

        return result

    def _normalize_llm_domain_confidences(
        self,
        payload: dict[str, Any],
        domains: list[str],
        local_confidences: dict[str, float],
    ) -> dict[str, float]:
        raw = payload.get("domain_confidences")
        result: dict[str, float] = {}

        if isinstance(raw, dict):
            for key, item in raw.items():
                normalized_key = self._normalize_domain_name(key)

                if not normalized_key:
                    continue

                try:
                    score = float(item)
                except Exception:
                    continue

                result[normalized_key] = max(0.0, min(1.0, score))

        for domain in domains:
            if domain not in result:
                result[domain] = max(
                    0.0,
                    min(
                        1.0,
                        float(local_confidences.get(domain, 0.60)),
                    ),
                )

        return result

    def _apply_llm_payload(
        self,
        analysis: QueryAnalysis,
        payload: dict[str, Any],
    ) -> QueryAnalysis:
        corrected_topic = self._sanitize_phrase(
            payload.get("corrected_topic"),
            analysis.corrected_topic,
            int(self._config["max_corrected_topic_words"]),
        )

        primary_concept = self._sanitize_phrase(
            payload.get("primary_concept"),
            corrected_topic,
            int(self._config["max_primary_concept_words"]),
        )

        keywords = self._sanitize_keywords(
            payload.get("keywords"),
            primary_concept or corrected_topic,
        )

        level_priority = self._normalize_llm_level_priority(
            payload.get("level_priority"),
            analysis.level_priority,
        )

        learner_level = self._normalize_level_value(
            payload.get("learner_level")
        )

        if not learner_level:
            if (
                level_priority
                and level_priority[0]
                in {"beginner", "intermediate", "advanced"}
            ):
                learner_level = level_priority[0]
            else:
                learner_level = analysis.learner_level

        detected_level = self._sanitize_level(
            payload.get("level"),
            learner_level,
        )

        intent = self._resolve_intent(
            payload.get("intent"),
            analysis.original_topic,
        )
        intent_confidence = self._parse_confidence(
            payload.get("intent_confidence")
        )
        spell_corrections = self._parse_spell_corrections(
            payload.get("spell_corrections")
        )

        target_domains = self._normalize_llm_domains(
            payload.get("target_domains")
        )

        if not target_domains:
            target_domains = analysis.target_domains

        domain_confidences = self._normalize_llm_domain_confidences(
            payload,
            target_domains,
            analysis.domain_confidences,
        )

        domain_confidence = self._parse_confidence(
            payload.get("domain_confidence")
        )

        if domain_confidence <= 0.0 and domain_confidences:
            domain_confidence = max(domain_confidences.values())

        target_formats = self._normalize_llm_formats(
            payload.get("target_formats")
        )

        if not target_formats:
            target_formats = analysis.target_formats

        short_search_queries = self._sanitize_short_queries(
            payload.get("short_search_queries"),
            corrected_topic,
            primary_concept,
            keywords,
        )

        level_profile = self._normalize_llm_level_profile(
            payload.get("level_profile"),
            analysis.level_profile,
        )

        beginner_ratio = self._normalize_llm_ratio(
            payload.get("beginner_ratio"),
            analysis.beginner_ratio,
        )

        min_beginner_sources = self._normalize_llm_int(
            payload.get("min_beginner_sources"),
            analysis.min_beginner_sources,
            lower=0,
            upper=20,
        )

        exclude_research_platforms = self._normalize_llm_bool(
            payload.get("exclude_research_platforms"),
            analysis.exclude_research_platforms,
        )

        preserve_tokens = self._normalize_llm_preserve_tokens(
            payload.get("preserve_tokens")
        )

        detected_level_lower = str(detected_level or "").strip().lower()
        original_text = (analysis.original_topic or "").lower()

        has_beginner_signal = False

        for signal in _STRONG_BEGINNER_SIGNALS + _MEDIUM_BEGINNER_SIGNALS:
            if signal in original_text:
                has_beginner_signal = True
                break

        has_advanced_signal = False

        for signal in _STRONG_ADVANCED_SIGNALS + _MEDIUM_ADVANCED_SIGNALS:
            if signal in original_text:
                has_advanced_signal = True
                break

        if (
            level_profile == "beginner"
            and not has_beginner_signal
            and not has_advanced_signal
        ):
            if detected_level_lower != "beginner":
                level_profile = analysis.level_profile or "balanced"
                beginner_ratio = min(beginner_ratio, 0.5)

        if level_profile == "beginner":
            beginner_ratio = max(beginner_ratio, 0.75)
            min_beginner_sources = max(min_beginner_sources, 4)
            exclude_research_platforms = True
        elif level_profile == "advanced":
            beginner_ratio = min(beginner_ratio, 0.25)
            exclude_research_platforms = False
        elif level_profile == "mixed":
            beginner_ratio = max(0.35, min(0.65, beginner_ratio))
            exclude_research_platforms = False
        elif level_profile == "balanced":
            beginner_ratio = max(0.3, min(0.55, beginner_ratio))
            exclude_research_platforms = False

        existing_content = list(analysis.content_tokens or [])

        return QueryAnalysis(
            original_topic=analysis.original_topic,
            cleaned_topic=analysis.cleaned_topic,
            corrected_topic=corrected_topic,
            primary_concept=primary_concept,
            keywords=keywords,
            intent=intent,
            level=detected_level,
            learner_level=learner_level,
            level_priority=level_priority,
            intent_confidence=intent_confidence,
            spell_corrections=spell_corrections,
            short_search_queries=short_search_queries,
            suggested_queries=short_search_queries,
            target_domains=target_domains,
            domain_confidences=domain_confidences,
            domain_confidence=domain_confidence,
            target_formats=target_formats,
            level_profile=level_profile,
            beginner_ratio=beginner_ratio,
            min_beginner_sources=min_beginner_sources,
            exclude_research_platforms=exclude_research_platforms,
            preserve_tokens=preserve_tokens,
            content_tokens=existing_content,
            llm_used=True,
        )

    def _sanitize_phrase(
        self,
        value: Any,
        fallback: str,
        max_words: int,
    ) -> str:
        text = clean_text(value)

        if not text:
            text = fallback

        if not text:
            return ""

        text = strip_filler(text).lower()
        text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        words = text.split()

        if not words and fallback:
            words = fallback.lower().split()

        return " ".join(words[:max_words])

    def _sanitize_keywords(self, value: Any, fallback: str) -> list[str]:
        keywords: list[str] = []

        if isinstance(value, list):
            items = value
        elif isinstance(value, str):
            items = [value]
        else:
            items = []

        for item in items:
            text = clean_text(item)

            if not text:
                continue

            text = strip_filler(text).lower()
            text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

            words = text.split()[:3]
            phrase = " ".join(words)

            if phrase and phrase not in keywords:
                keywords.append(phrase)

            if len(keywords) >= int(self._config["max_keywords"]):
                break

        if not keywords:
            keywords = self._extract_keywords(fallback)

        return keywords[: int(self._config["max_keywords"])]

    def _sanitize_short_queries(
        self,
        value: Any,
        corrected_topic: str,
        primary_concept: str,
        keywords: list[str],
    ) -> list[str]:
        raw_queries: list[Any] = []

        if isinstance(value, list):
            raw_queries = value
        elif isinstance(value, str):
            raw_queries = [value]

        queries: list[str] = []

        for item in raw_queries:
            cleaned = self._clean_query(item)

            if cleaned and cleaned not in queries:
                queries.append(cleaned)

            if len(queries) >= int(self._config["max_short_queries"]):
                break

        if corrected_topic and corrected_topic not in queries:
            queries.insert(0, corrected_topic)

        if len(queries) < 2:
            for fallback_query in self._build_short_queries(
                corrected_topic,
                primary_concept,
                keywords,
            ):
                if fallback_query not in queries:
                    queries.append(fallback_query)

                if len(queries) >= int(self._config["max_short_queries"]):
                    break

        return queries[: int(self._config["max_short_queries"])]

    def _parse_intent(self, value: Any) -> QueryIntent:
        try:
            return QueryIntent(str(value).strip().lower())
        except Exception:
            return QueryIntent.EXPLORE
        
    def _resolve_intent(
        self,
        llm_value: Any,
        topic: str,
    ) -> QueryIntent:
        llm_intent = self._parse_intent(llm_value)
        local_intent = self._local_intent(topic)
        text = str(topic or "").lower()

        strong_learn = (
            "learn",
            "study",
            "understand",
            "explain",
            "teach me",
            "walk me through",
        )

        strong_research = (
            "literature review",
            "systematic review",
            "meta-analysis",
            "state of the art",
            "state-of-the-art",
            "survey of",
        )

        has_strong_learn = any(marker in text for marker in strong_learn)
        has_strong_research = any(marker in text for marker in strong_research)

        if has_strong_research:
            return QueryIntent.RESEARCH

        if has_strong_learn:
            return QueryIntent.LEARN

        if llm_intent != QueryIntent.EXPLORE:
            return llm_intent

        return local_intent

    def _parse_confidence(self, value: Any) -> float:
        try:
            numeric = float(value)
        except Exception:
            return 0.0

        return min(1.0, max(0.0, numeric))

    def _parse_spell_corrections(self, value: Any) -> dict[str, str]:
        corrections: dict[str, str] = {}

        if not isinstance(value, dict):
            return corrections

        for raw_key, raw_value in value.items():
            original_word = clean_text(raw_key)
            corrected_word = clean_text(raw_value)

            if original_word and corrected_word:
                corrections[original_word] = corrected_word

        return corrections

    def _sanitize_level(self, value: Any, fallback: str) -> str:
        fallback_text = clean_text(fallback).lower()

        if fallback_text in {"beginner", "intermediate", "advanced"}:
            return fallback_text

        text = clean_text(value).lower()

        has_beginner = "beginner" in text
        has_intermediate = "intermediate" in text
        has_advanced = "advanced" in text or "advance" in text
        has_mixed = "mixed" in text

        if has_mixed:
            return "mixed"

        if has_beginner and has_advanced:
            return "mixed"

        if has_beginner:
            return "beginner"

        if has_advanced:
            return "advanced"

        if has_intermediate:
            return "intermediate"

        return "mixed"