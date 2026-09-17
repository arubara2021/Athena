from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from utils.text import clean_text

_TOKEN_PATTERN = re.compile(r"[a-z0-9+#.]+")
_RAW_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9+#.]+")

FILLER_PATTERNS = (
    r"\b(?:i|i'm|im|we|we're)\s+(?:need|want|would like|require|am looking for|am searching for)\b",
    r"\b(?:please|kindly|can you|could you|would you|help|me|us)\b",
    r"\b(?:need|needs|want|wants|require|requires|show|give|tell|explain|provide|find|search|search for|look for|looking for)\b",
    r"\b(?:detailed|details|detail|comprehensive|complete|in[- ]depth|deep)\b",
    r"\b(?:fully|completely|totally|best|top|latest|greatest|ultimate|perfect|ideal)\b",
    r"\b(?:about|on|of|for|regarding|related to|related)\b",
    r"\b(?:what|which|who|whom|whose|when|where|why|how)\b",
    r"\b(?:is|are|was|were|do|does|did|to|from|with|without|in|at|by|as|be|been|being)\b",
    r"\b(?:and|or|but|nor|so|yet|also|than|not|into|onto|upon|per|via)\b",
    r"\b(?:a|an|the)\b",
)

_DEFAULT_GENERIC_TERMS = frozenset({
    "about", "above", "across", "after", "again", "all", "also", "and",
    "any", "anything", "are", "around", "article", "articles", "available",
    "back", "base", "based", "basic", "basics", "because", "been", "before",
    "begin", "beginner", "beginners", "being", "best", "better", "between",
    "blog", "book", "books", "both", "brief", "but", "can", "case", "cases",
    "clear", "come", "comes", "complete", "comprehensive", "concept",
    "concepts", "could", "course", "courses", "cover", "covered", "covers",
    "deep", "detail", "detailed", "details", "different", "documentation",
    "does", "doing", "done", "download", "each", "easy", "either", "end",
    "enough", "entire", "even", "ever", "every", "example", "examples",
    "explain", "explained", "explains", "explanation", "few", "find",
    "first", "follow", "following", "free", "from", "full", "fully",
    "general", "get", "gets", "give", "given", "good", "great", "guide",
    "guides", "had", "has", "have", "help", "here", "high", "how",
    "however", "idea", "ideas", "include", "includes", "including", "info",
    "information", "inside", "instead", "into", "intro", "introduction",
    "its", "just", "keep", "know", "knowledge", "large", "last", "latest",
    "learn", "learned", "learning", "learns", "level", "levels", "like",
    "list", "little", "look", "looking", "lot", "low", "made", "main",
    "make", "makes", "making", "many", "may", "mean", "meaning", "means",
    "might", "more", "most", "much", "must", "name", "named", "need",
    "needed", "needs", "new", "next", "nor", "not", "now", "number", "off",
    "one", "online", "only", "onto", "open", "other", "others", "out",
    "outline", "over", "overview", "own", "part", "parts", "people",
    "per", "place", "please", "possible", "practical", "practice",
    "provide", "provides", "put", "quick", "quickly", "really", "recent",
    "related", "right", "run", "same", "search", "see", "set", "several",
    "shall", "should", "show", "shows", "simple", "simply", "since",
    "small", "so", "some", "something", "soon", "source", "sources",
    "start", "started", "starting", "step", "steps", "still", "stuff",
    "such", "take", "teach", "teaches", "tell", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "thing", "things",
    "think", "this", "those", "through", "time", "times", "tips",
    "together", "too", "top", "topic", "topics", "toward", "trick",
    "tricks", "try", "tutorial", "tutorials", "two", "type", "types",
    "understand", "understanding", "upon", "use", "used", "useful", "uses",
    "using", "various", "very", "via", "video", "videos", "want", "wanted",
    "wants", "was", "watch", "way", "ways", "well", "were", "what", "when",
    "where", "which", "while", "who", "why", "will", "with", "within",
    "without", "word", "words", "work", "works", "working", "world",
    "would", "write", "year", "years", "yet", "you", "your", "youtube",
})

_GENERIC_CACHE: frozenset[str] | None = None


def get_generic_terms() -> frozenset[str]:
    global _GENERIC_CACHE

    if _GENERIC_CACHE is not None:
        return _GENERIC_CACHE

    terms = set(_DEFAULT_GENERIC_TERMS)

    try:
        import yaml

        config_path = (
            Path(__file__).resolve().parent.parent
            / "configs"
            / "ranking.yaml"
        )

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}

            if isinstance(loaded, dict):
                section = loaded.get("tokenization", {})

                if isinstance(section, dict):
                    custom = section.get("generic_terms")

                    if isinstance(custom, list):
                        cleaned = {
                            str(item).strip().lower()
                            for item in custom
                            if str(item).strip()
                        }

                        if cleaned:
                            terms = cleaned
    except Exception:
        pass

    _GENERIC_CACHE = frozenset(terms)
    return _GENERIC_CACHE


def _protected_tokens(value: Any) -> set[str]:
    text = clean_text(value)

    if not text:
        return set()

    raw_tokens = _RAW_TOKEN_PATTERN.findall(text)
    generic_terms = get_generic_terms()
    protected: set[str] = set()

    for token in raw_tokens:
        lowered = token.lower()

        if len(lowered) < 2 or len(lowered) > 8:
            continue

        if token.isupper():
            protected.add(lowered)
            continue

        if any(character.isdigit() for character in token):
            protected.add(lowered)
            continue

        if "+" in token or "#" in token or "." in token:
            protected.add(lowered)
            continue

        if len(lowered) <= 5 and lowered not in generic_terms:
            protected.add(lowered)

    return protected


def strip_filler(value: Any) -> str:
    lowered = clean_text(value).lower()

    for pattern in FILLER_PATTERNS:
        lowered = re.sub(pattern, " ", lowered)

    return re.sub(r"\s+", " ", lowered).strip()


def clean_topic(value: Any) -> str:
    original = clean_text(value)
    stripped = strip_filler(original)

    if stripped:
        return stripped

    return original.lower()


def normalize_token(token: str) -> str:
    token = token.lower().strip(".")

    if len(token) <= 6:
        return token

    if token.endswith("ing") and len(token) > 5:
        token = token[:-3]

    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("es") and len(token) > 3:
        token = token[:-2]
    elif token.endswith("s") and len(token) > 2:
        token = token[:-1]

    return token


def tokenize(value: Any) -> list[str]:
    cleaned = clean_text(value).lower()
    raw_tokens = _TOKEN_PATTERN.findall(cleaned)

    normalized: list[str] = []

    for token in raw_tokens:
        reduced = normalize_token(token)

        if reduced:
            normalized.append(reduced)

    return normalized


def concept_tokens(
    value: Any,
    generic_terms: frozenset[str] | None = None,
) -> set[str]:
    tokens = tokenize(value)
    protected = _protected_tokens(value)
    generics = generic_terms if generic_terms is not None else get_generic_terms()

    core: set[str] = set()

    for token in tokens:
        if token in protected:
            core.add(token)
            continue

        if len(token) > 2 and token not in generics:
            core.add(token)

    if core:
        return core

    return {token for token in tokens if len(token) >= 2}


def contains_any_token(value: Any, tokens: Iterable[str]) -> bool:
    lowered = clean_text(value).lower()

    if not lowered:
        return False

    found = set(tokenize(lowered))

    for token in tokens:
        token = token.lower()

        if not token:
            continue

        if token in found:
            return True

        if len(token) >= 4:
            for candidate in found:
                if candidate.startswith(token) or token.startswith(candidate):
                    return True

        if re.search(
            rf"(?:^|[^a-z0-9]){re.escape(token)}(?:[^a-z0-9]|$)",
            lowered,
        ):
            return True

    return False