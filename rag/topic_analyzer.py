from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

from rag.schemas import TopicProfileSchema
from utils.logger import get_logger
from utils.text import clean_text

_MAX_TOPIC_WORDS = 8
_MAX_SUBTOPICS = 8
_MAX_RELATED = 8
_CORPUS_SAMPLE = 200

_ACRONYM_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]{1,7}\b")
_QUOTED_PATTERN = re.compile(r'"([^"]{2,80})"|\'([^\']{2,80})\'')
_WORD_PATTERN = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)

_QUESTION_LEADERS = (
    "what is", "what are", "who is", "who are",
    "how does", "how do", "how can", "how is", "how are",
    "how to", "why does", "why do", "why is", "why are",
    "when did", "when was", "when is",
    "where is", "where are",
    "explain", "describe", "define", "definition of",
    "meaning of", "tell me about", "tell me", "talk about",
    "give me", "list", "examples of", "example of",
    "should i", "can you", "could you", "please",
)

_TRAILING_NOISE = (
    "?", "!", ".", ",", ":", ";", " please", " thanks", " thank you",
)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "what", "how", "why",
    "when", "where", "which", "who", "whom", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "do", "does", "did",
    "doing", "have", "has", "had", "having", "can", "could", "should", "would",
    "may", "might", "must", "will", "shall", "not", "no", "yes", "about",
    "into", "over", "under", "between", "from", "as", "at", "by", "if",
    "then", "than", "too", "very", "just", "also", "explain", "describe",
    "define", "tell", "talk", "give", "show", "list", "mean", "means",
    "meaning", "definition", "example", "examples", "overview", "summary",
    "difference", "differences", "vs", "versus", "compare", "compared",
}

_PRONOUNS = {
    "it", "its", "it's", "they", "them", "their", "theirs",
    "this", "that", "these", "those",
    "he", "she", "him", "her", "his", "hers",
}

_ANSWER_STYLE_SHORT = "short"
_ANSWER_STYLE_STANDARD = "standard"
_ANSWER_STYLE_DETAILED = "detailed"
_ANSWER_STYLE_EXHAUSTIVE = "exhaustive"

_VALID_ANSWER_STYLES = frozenset(
    {
        _ANSWER_STYLE_SHORT,
        _ANSWER_STYLE_STANDARD,
        _ANSWER_STYLE_DETAILED,
        _ANSWER_STYLE_EXHAUSTIVE,
    }
)

_ANSWER_STYLE_PRIORITY: dict[str, int] = {
    _ANSWER_STYLE_STANDARD: 0,
    _ANSWER_STYLE_SHORT: 1,
    _ANSWER_STYLE_DETAILED: 2,
    _ANSWER_STYLE_EXHAUSTIVE: 3,
}

_SHORT_MARKERS: tuple[str, ...] = (
    "briefly",
    "in one line",
    "in one sentence",
    "one line answer",
    "one sentence answer",
    "short answer",
    "short explanation",
    "quick answer",
    "quick summary",
    "tldr",
    "tl;dr",
    "in a nutshell",
    "just the gist",
    "keep it short",
    "keep it brief",
    "concise answer",
    "summarize quickly",
)

_DETAILED_MARKERS: tuple[str, ...] = (
    "very detailed",
    "in detail",
    "in depth",
    "in-depth",
    "explain fully",
    "explain in detail",
    "explain thoroughly",
    "step by step",
    "walk me through",
    "walk through",
    "walk-through",
    "comprehensive explanation",
    "comprehensive answer",
    "detailed explanation",
    "detailed answer",
    "thorough explanation",
    "thorough answer",
    "thoroughly explain",
    "full explanation",
    "full answer",
    "deep dive",
    "deep-dive",
    "elaborate on",
)

_EXHAUSTIVE_MARKERS: tuple[str, ...] = (
    "exhaustive",
    "exhaustively",
    "everything about",
    "everything on",
    "complete guide",
    "complete explanation",
    "in every detail",
    "nothing left out",
    "all aspects",
    "cover everything",
    "as much detail as possible",
    "maximum detail",
    "most detailed",
    "the most detail",
)

_INTENT_DETAILED_EXPLANATION = "detailed_explanation"

_DETAILED_INTENT_REGEX = re.compile(
    r"\b("
    + "|".join(re.escape(marker) for marker in _DETAILED_MARKERS)
    + r")\b",
    re.IGNORECASE,
)

_INTENT_PRIORITY: tuple[str, ...] = (
    "comparison",
    "mechanism",
    "causation",
    "decision",
    "timeline",
    "enumeration",
    _INTENT_DETAILED_EXPLANATION,
    "definition",
    "explanation",
)

_INTENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "definition",
        re.compile(
            r"^\s*(what\s+is|what\s+are|what's|define|definition\s+of|"
            r"meaning\s+of|what\s+does\s+.+\s+mean)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "comparison",
        re.compile(
            r"\b(vs\.?|versus|compared\s+to|compared\s+with|"
            r"difference\s+between|differences\s+between|"
            r"better\s+than|pros\s+and\s+cons|advantages?\s+and\s+disadvantages?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "mechanism",
        re.compile(
            r"^\s*(how\s+(does|do|can|is|are|would|should)|how\s+to|"
            r"in\s+what\s+way)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "causation",
        re.compile(
            r"^\s*(why\s+(does|do|is|are|did|would|should)|what\s+causes?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "enumeration",
        re.compile(
            r"^\s*(list|give\s+me|name|enumerate|examples?\s+of|"
            r"what\s+are\s+some|which\s+are)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "timeline",
        re.compile(
            r"\b(history|historical|origin[s]?|evolution|timeline|"
            r"when\s+was|when\s+did|first\s+developed|invented)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "decision",
        re.compile(
            r"^\s*(should\s+i|is\s+it\s+(better|worth|good|safe)|"
            r"which\s+is\s+better|do\s+i\s+need)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "explanation",
        re.compile(
            r"^\s*(explain|describe|walk\s+me\s+through|tell\s+me\s+about)\b",
            re.IGNORECASE,
        ),
    ),
    (
        _INTENT_DETAILED_EXPLANATION,
        _DETAILED_INTENT_REGEX,
    ),
)


class TopicAnalyzer:
    def __init__(
        self,
        max_topic_words: int = _MAX_TOPIC_WORDS,
        max_subtopics: int = _MAX_SUBTOPICS,
        max_related: int = _MAX_RELATED,
        corpus_sample: int = _CORPUS_SAMPLE,
    ) -> None:
        self._max_topic_words = max(2, int(max_topic_words))
        self._max_subtopics = max(1, int(max_subtopics))
        self._max_related = max(1, int(max_related))
        self._corpus_sample = max(10, int(corpus_sample))
        self._logger = get_logger("rag.topic_analyzer")

    def analyze(
        self,
        question: str,
        store: Any | None = None,
    ) -> TopicProfileSchema:
        cleaned = clean_text(question)

        if not cleaned:
            return TopicProfileSchema(
                question=question or "",
                intent="overview",
                intent_confidence=0.0,
                corpus_available=False,
                answer_style=_ANSWER_STYLE_STANDARD,
            )

        intent, intent_confidence = self.detect_intent(cleaned)
        answer_style = self.detect_answer_style(cleaned)
        language = self._detect_language(cleaned)
        acronyms = self._extract_acronyms(cleaned)
        quoted = self._extract_quoted(cleaned)
        primary_topic = self.extract_primary_topic(cleaned)
        topic_tokens = self._topic_tokens(primary_topic or cleaned)

        profile = TopicProfileSchema(
            question=cleaned,
            intent=intent,
            intent_confidence=intent_confidence,
            primary_topic=primary_topic,
            topic_tokens=topic_tokens,
            acronyms=acronyms,
            quoted_phrases=quoted,
            language=language,
            corpus_available=False,
            answer_style=answer_style,
        )

        if store is None or not primary_topic:
            return profile

        corpus_titles = self._fetch_corpus_titles(store, primary_topic)

        if not corpus_titles:
            return profile

        profile.corpus_available = True
        profile.corpus_documents_seen = len(corpus_titles)
        profile.subtopics = self._extract_subtopics(
            corpus_titles,
            topic_tokens,
            self._max_subtopics,
        )
        profile.related_concepts = self._extract_related(
            corpus_titles,
            topic_tokens,
            self._max_related,
        )

        return profile

    def detect_intent(self, question: str) -> tuple[str, float]:
        text = str(question or "").strip()

        if not text:
            return "overview", 0.0

        matches: list[str] = []

        for intent, pattern in _INTENT_PATTERNS:
            try:
                if pattern.search(text):
                    matches.append(intent)
            except Exception:
                continue

        if not matches:
            return "overview", 0.3

        if len(matches) == 1:
            return matches[0], 0.85

        for candidate in _INTENT_PRIORITY:
            if candidate in matches:
                return candidate, 0.65

        return matches[0], 0.5

    def detect_answer_style(self, question: str) -> str:
        text = clean_text(question).lower()

        if not text:
            return _ANSWER_STYLE_STANDARD

        normalized = re.sub(r"\s+", " ", text).strip()

        short_score = self._count_marker_hits(normalized, _SHORT_MARKERS)
        detailed_score = self._count_marker_hits(normalized, _DETAILED_MARKERS)
        exhaustive_score = self._count_marker_hits(normalized, _EXHAUSTIVE_MARKERS)

        best_style = _ANSWER_STYLE_STANDARD
        best_score = 0
        best_priority = -1

        for style, score in (
            (_ANSWER_STYLE_SHORT, short_score),
            (_ANSWER_STYLE_DETAILED, detailed_score),
            (_ANSWER_STYLE_EXHAUSTIVE, exhaustive_score),
        ):
            if score <= 0:
                continue

            priority = _ANSWER_STYLE_PRIORITY.get(style, 0)

            if score > best_score or (
                score == best_score and priority > best_priority
            ):
                best_style = style
                best_score = score
                best_priority = priority

        return best_style

    def extract_primary_topic(self, question: str) -> str:
        text = clean_text(question)

        if not text:
            return ""

        quoted = self._extract_quoted(text)

        if quoted:
            return quoted[0]

        lowered = text.lower()

        for leader in sorted(_QUESTION_LEADERS, key=len, reverse=True):
            if lowered.startswith(leader + " "):
                text = text[len(leader):].strip()
                break
            if lowered.startswith(leader):
                text = text[len(leader):].strip()
                break

        for noise in _TRAILING_NOISE:
            if text.lower().endswith(noise):
                text = text[: -len(noise)].rstrip()

        text = text.strip(" \t\n\r\"'`")

        if not text:
            return ""

        words = self._split_words(text)
        kept: list[str] = []

        for word in words:
            lowered_word = word.lower()
            if lowered_word in _STOPWORDS:
                continue
            kept.append(word)
            if len(kept) >= self._max_topic_words:
                break

        if kept:
            return " ".join(kept)

        return " ".join(words[: self._max_topic_words])

    def is_pronoun_question(self, question: str) -> bool:
        text = clean_text(question).lower()

        if not text:
            return False

        tokens = self._split_words(text)

        if not tokens:
            return False

        head = tokens[:2]

        for token in head:
            if token in _PRONOUNS:
                return True

        return False

    def topic_strength(self, question: str) -> float:
        topic = self.extract_primary_topic(question)

        if not topic:
            return 0.0

        tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", topic.lower())
            if len(token) >= 3
        ]

        if not tokens:
            return 0.0

        from search.query_tokens import get_generic_terms

        try:
            generics = get_generic_terms()
        except Exception:
            generics = frozenset()

        meaningful = [
            token
            for token in tokens
            if token not in generics
        ]

        count_component = min(1.0, len(meaningful) / 3.0)
        length_component = (
            min(1.0, sum(len(token) for token in meaningful) / 20.0)
            if meaningful
            else 0.0
        )

        return round(0.7 * count_component + 0.3 * length_component, 4)

    def _count_marker_hits(self, text: str, markers: tuple[str, ...]) -> int:
        if not text or not markers:
            return 0

        count = 0

        for marker in markers:
            if not marker:
                continue

            if marker in text:
                count += 1

        return count

    def _extract_acronyms(self, question: str) -> list[str]:
        found: list[str] = []

        for match in _ACRONYM_PATTERN.findall(str(question or "")):
            token = match.strip()
            if token and token not in found:
                found.append(token)

        return found

    def _extract_quoted(self, question: str) -> list[str]:
        phrases: list[str] = []

        for double, single in _QUOTED_PATTERN.findall(str(question or "")):
            text = (double or single or "").strip()
            if text and text not in phrases:
                phrases.append(text)

        return phrases

    def _topic_tokens(self, text: str) -> list[str]:
        tokens: list[str] = []

        for word in self._split_words(str(text or "")):
            lowered = word.lower()
            if lowered in _STOPWORDS:
                continue
            if len(lowered) < 2:
                continue
            if lowered not in tokens:
                tokens.append(lowered)

        return tokens

    def _split_words(self, text: str) -> list[str]:
        return [match.group(0) for match in _WORD_PATTERN.finditer(str(text or ""))]

    def _detect_language(self, text: str) -> str:
        if not text:
            return ""

        non_ascii = sum(1 for ch in text if ord(ch) > 127)
        ratio = non_ascii / max(1, len(text))

        if ratio < 0.05:
            return "en"

        first = text[0]

        if 0x4E00 <= ord(first) <= 0x9FFF:
            return "zh"

        if 0x0400 <= ord(first) <= 0x04FF:
            return "ru"

        if 0x0600 <= ord(first) <= 0x06FF:
            return "ar"

        if 0x0900 <= ord(first) <= 0x097F:
            return "hi"

        try:
            unicodedata.name(first)
        except Exception:
            return "unknown"

        return "unknown"

    def _fetch_corpus_titles(
        self,
        store: Any,
        topic: str,
    ) -> list[str]:
        if not topic:
            return []

        method = getattr(store, "keyword_search", None)

        if not callable(method):
            return []

        try:
            results = method(topic, top_k=self._corpus_sample)
        except Exception as exc:
            self._logger.warning(f"Corpus keyword search failed: {exc}")
            return []

        titles: list[str] = []

        for result in results or []:
            document = getattr(result, "document", None)

            if document is None:
                continue

            title = clean_text(getattr(document, "title", "") or "")

            if title:
                titles.append(title)

        return titles

    def _extract_subtopics(
        self,
        titles: list[str],
        topic_tokens: list[str],
        limit: int,
    ) -> list[str]:
        if not titles:
            return []

        topic_set = set(topic_tokens)
        documents_count = len(titles)
        doc_freq: dict[str, int] = {}
        total_count: dict[str, int] = {}

        for title in titles:
            tokens = [
                word.lower()
                for word in self._split_words(title)
                if len(word) >= 3 and word.lower() not in _STOPWORDS
            ]

            unique = set(tokens)

            for token in unique:
                doc_freq[token] = doc_freq.get(token, 0) + 1

            for token in tokens:
                total_count[token] = total_count.get(token, 0) + 1

        scored: list[tuple[float, str]] = []

        for token, freq in doc_freq.items():
            if token in topic_set:
                continue
            if freq <= 0 or freq >= documents_count:
                continue
            if token.isdigit():
                continue

            tf = total_count.get(token, 0) / max(1, documents_count)
            idf = math.log((documents_count + 1) / (freq + 1)) + 1.0
            score = tf * idf

            scored.append((score, token))

        scored.sort(key=lambda item: (-item[0], item[1]))

        return [token for _, token in scored[:limit]]

    def _extract_related(
        self,
        titles: list[str],
        topic_tokens: list[str],
        limit: int,
    ) -> list[str]:
        if not titles or not topic_tokens:
            return []

        topic_set = set(topic_tokens)
        co_occurrence: dict[str, int] = {}

        for title in titles:
            tokens = {
                word.lower()
                for word in self._split_words(title)
                if len(word) >= 3 and word.lower() not in _STOPWORDS
            }

            if not (tokens & topic_set):
                continue

            for token in tokens:
                if token in topic_set:
                    continue
                co_occurrence[token] = co_occurrence.get(token, 0) + 1

        ranked = sorted(
            co_occurrence.items(),
            key=lambda item: (-item[1], item[0]),
        )

        return [token for token, count in ranked[:limit] if count >= 2]