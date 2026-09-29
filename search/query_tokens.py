from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable
from utils.text import clean_text

_TOKEN_PATTERN = re.compile(r"[a-z0-9+#.]+")
_RAW_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9+#.]+")
_PHRASE_SPLIT_RE = re.compile(r"\s+")

_MIN_PREFIX_LENGTH = 5
_MIN_ONE_SIDED_PREFIX_LENGTH = 4
_MAX_SHORT_TOKEN_LENGTH_FOR_PREFIX = 6
_MIN_SUBSTRING_LENGTH = 5
_MIN_FUZZY_LENGTH = 5
_FUZZY_RATIO_THRESHOLD = 0.80
_FUZZY_LENGTH_TOLERANCE = 2

FILLER_PATTERNS = (
    r"\b(?:i|i'm|im|we|we're)\s+(?:need|want|would like|require|am looking for|am searching for)\b",
    r"\b(?:please|kindly|can you|could you|would you|help|me|us)\b",
    r"\b(?:need|needs|want|wants|require|requires|show|give|tell|explain|provide|find|search|search for|look for|looking for)\b",
    r"\b(?:detailed|details|detail|comprehensive|complete|in[- ]depth|deep)\b",
    r"\b(?:fully|completely|totally|best|top|latest|greatest|ultimate|perfect|ideal)\b",
    r"\b(?:the\s+)?(?:life|world|story|tale|idea|notion|concept|essence)\s+of\b",
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

_BEGINNER_INDICATORS = frozenset({
    "beginner", "beginners", "basics", "basic",
    "fundamentals", "fundamental",
    "introduction", "intro", "introductory",
    "tutorial", "tutorials", "course", "courses",
    "textbook", "textbooks", "guide", "guides",
    "step by step", "from scratch", "no prior",
    "starter", "easy", "simple",
    "learn", "learning", "study", "studying",
    "overview", "explained", "primer", "quickstart",
    "for beginners", "getting started",
})

_ADVANCED_INDICATORS = frozenset({
    "advanced", "advance", "advances", "advancement", "advancements",
    "expert", "experts",
    "research",
    "state of the art", "state-of-the-art",
    "cutting edge", "cutting-edge",
    "in depth", "in-depth",
    "deep dive", "deep-dive",
    "graduate",
    "latest",
    "novel",
    "survey", "surveys",
    "benchmark", "benchmarks",
    "architecture", "architectures",
    "implementation", "implementations",
    "optimization", "optimisation",
    "frontier", "specialized", "specialised",
    "high level", "high-level",
    "to advanced", "and beyond", "up to advanced",
    "deeply", "deeply understand",
    "thorough understanding", "comprehensive understanding",
    "rigorous", "rigour", "rigor",
    "formal", "formal treatment",
    "mathematical foundations", "theoretical foundations",
    "principles of", "theory of", "foundations of",
    "graduate-level", "research-level",
    "proofs", "proof-based", "proof", "axiomatic",
    "in-depth treatment", "exhaustive", "advanced treatment",
    "mathematical", "theoretical",
})

_NEGATIVE_OFF_TOPIC_MARKERS = frozenset({
    "metal-organic framework", "mof", "metal organic framework",
    "covalent organic framework", "cof", "photocatalytic",
    "transdermal", "drug delivery", "nanoparticle", "nanomaterial",
    "forensic", "nervous system", "physiology",
})

_INTENT_NOT_CONCEPT_WORDS = frozenset({
    "education",
    "educational",
    "educate",
    "educating",
    "learning",
    "learn",
    "learners",
    "learner",
    "study",
    "studies",
    "studying",
    "teaching",
    "teach",
    "teacher",
    "teachers",
    "training",
    "train",
    "tutorial",
    "tutorials",
    "course",
    "courses",
    "lesson",
    "lessons",
    "curriculum",
    "curricula",
    "pedagogy",
    "pedagogical",
    "student",
    "students",
    "school",
    "schools",
    "university",
    "universities",
    "college",
    "colleges",
    "class",
    "classes",
    "classroom",
    "classrooms",
    "academic",
    "academia",
    "research",
    "researcher",
    "researchers",
    "study guide",
    "studying",
})

_GENERIC_CACHE: frozenset[str] | None = None

_STEM_SUFFIXES = (
    "ness",
    "ment",
    "tion",
    "sion",
    "ing",
    "ity",
    "ies",
    "es",
    "s",
)


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
                            terms.update(cleaned)
    except Exception:
        pass
    _GENERIC_CACHE = frozenset(terms)
    return _GENERIC_CACHE


def get_intent_not_concept_words() -> frozenset[str]:
    return _INTENT_NOT_CONCEPT_WORDS


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


def stem_token(token: str) -> str:
    text = str(token or "").lower().strip(".")

    if not text or len(text) < 4:
        return text

    for suffix in _STEM_SUFFIXES:
        if text.endswith(suffix) and len(text) - len(suffix) >= 3:
            return text[: -len(suffix)]

    return text


def shared_prefix_length(a: str, b: str) -> int:
    if not a or not b:
        return 0

    limit = min(len(a), len(b))
    shared = 0

    for index in range(limit):
        if a[index] != b[index]:
            break
        shared += 1

    return shared


def tokens_match(token: str, candidate: str) -> bool:
    if not token or not candidate:
        return False

    token_text = str(token).strip().lower()
    candidate_text = str(candidate).strip().lower()

    if not token_text or not candidate_text:
        return False

    if token_text == candidate_text:
        return True

    stem_a = stem_token(token_text)
    stem_b = stem_token(candidate_text)

    if not stem_a or not stem_b:
        return False

    if stem_a == stem_b:
        return True

    shared = shared_prefix_length(stem_a, stem_b)

    if shared >= _MIN_PREFIX_LENGTH:
        return True

    if (
        shared >= _MIN_ONE_SIDED_PREFIX_LENGTH
        and min(len(stem_a), len(stem_b)) <= _MAX_SHORT_TOKEN_LENGTH_FOR_PREFIX
    ):
        return True

    shorter = stem_a if len(stem_a) <= len(stem_b) else stem_b
    longer = stem_b if shorter is stem_a else stem_a

    if len(shorter) >= _MIN_SUBSTRING_LENGTH and shorter in longer:
        return True

    if (
        len(stem_a) >= _MIN_FUZZY_LENGTH
        and len(stem_b) >= _MIN_FUZZY_LENGTH
    ):
        if abs(len(stem_a) - len(stem_b)) <= _FUZZY_LENGTH_TOLERANCE:
            if sorted(stem_a) == sorted(stem_b):
                return True

        ratio = SequenceMatcher(None, stem_a, stem_b).ratio()

        if ratio >= _FUZZY_RATIO_THRESHOLD:
            return True

    return False


def any_token_matches(token: str, candidates: Iterable[str]) -> bool:
    if not token or not candidates:
        return False

    for candidate in candidates:
        if tokens_match(token, candidate):
            return True

    return False


def count_token_matches(tokens: Iterable[str], candidates: Iterable[str]) -> int:
    if not tokens or not candidates:
        return 0

    candidate_list = list(candidates)

    if not candidate_list:
        return 0

    matched = 0

    for token in tokens:
        if any_token_matches(token, candidate_list):
            matched += 1

    return matched


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


def extract_concept_phrases(value: Any, max_phrases: int = 12) -> list[str]:
    text = strip_filler(clean_text(value)).lower()
    text = re.sub(r"[^\w\s+#.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    words = text.split()
    generic_terms = get_generic_terms()
    phrases: list[str] = []
    seen: set[str] = set()
    full = " ".join(words)
    if full and len(full) >= 4:
        phrases.append(full)
        seen.add(full)
    for length in range(min(len(words), 4), 1, -1):
        for start in range(len(words) - length + 1):
            chunk = " ".join(words[start:start + length])
            meaningful = [w for w in words[start:start + length] if w not in generic_terms]
            if len(meaningful) < 2:
                continue
            if chunk in seen:
                continue
            seen.add(chunk)
            phrases.append(chunk)
    if len(phrases) > max_phrases:
        phrases = phrases[:max_phrases]
    return phrases


def build_required_tokens(
    topic: str,
    primary_concept: str = "",
    keywords: list[str] | None = None,
    max_tokens: int = 12,
) -> tuple[set[str], list[str]]:
    concept = primary_concept or topic
    phrases = extract_concept_phrases(concept)
    core = concept_tokens(topic)
    if keywords:
        for kw in keywords:
            core.update(concept_tokens(kw))
    if len(core) > max_tokens:
        core = set(list(core)[:max_tokens])
    return core, phrases


def build_negative_tokens(topic: str) -> set[str]:
    lowered = clean_text(topic).lower()
    negatives: set[str] = set()
    for marker in _NEGATIVE_OFF_TOPIC_MARKERS:
        if marker not in lowered:
            negatives.add(marker)
    return negatives


def get_level_tokens(level: str) -> dict[str, set[str]]:
    normalized = str(level or "").strip().lower()
    if "beginner" in normalized:
        return {
            "required": set(_BEGINNER_INDICATORS),
            "penalty": set(_ADVANCED_INDICATORS),
        }
    if "advanced" in normalized:
        return {
            "required": set(_ADVANCED_INDICATORS),
            "penalty": set(_BEGINNER_INDICATORS),
        }
    return {
        "required": set(),
        "penalty": set(),
    }


def is_beginner_query(topic: str, goal: str = "", level: str = "") -> bool:
    combined = f"{topic} {goal} {level}".lower()
    beginner_count = sum(1 for term in _BEGINNER_INDICATORS if term in combined)
    advanced_count = sum(1 for term in _ADVANCED_INDICATORS if term in combined)
    if beginner_count > advanced_count:
        return True
    if "beginner" in level.lower():
        return True
    if "basics" in combined and "advanced" not in combined and "advance" not in combined:
        return True
    return False


def is_advanced_query(topic: str, goal: str = "", level: str = "") -> bool:
    combined = f"{topic} {goal} {level}".lower()
    advanced_count = sum(1 for term in _ADVANCED_INDICATORS if term in combined)
    beginner_count = sum(1 for term in _BEGINNER_INDICATORS if term in combined)
    if advanced_count > beginner_count:
        return True
    if "advanced" in level.lower():
        return True
    return False


def detect_off_topic(source_text: str, required_phrases: list[str], negative_tokens: set[str]) -> bool:
    lowered = source_text.lower()
    if required_phrases:
        phrase_found = any(phrase in lowered for phrase in required_phrases)
        if not phrase_found:
            core_words = set()
            for phrase in required_phrases:
                core_words.update(phrase.split())
            generic_terms = get_generic_terms()
            meaningful = {w for w in core_words if w not in generic_terms and len(w) > 2}
            if meaningful:
                matched = sum(1 for w in meaningful if w in lowered)
                if matched < max(1, len(meaningful) // 2):
                    return True
    if negative_tokens:
        neg_hits = sum(1 for neg in negative_tokens if neg in lowered)
        total_words = len(lowered.split())
        if total_words > 0 and neg_hits >= 2:
            return True
    return False


_STRONG_BEGINNER_SIGNALS = (
    "for beginners",
    "for absolute beginners",
    "for complete beginners",
    "for total beginners",
    "from scratch",
    "no prior",
    "no experience",
    "no experience needed",
    "no prior experience",
    "beginner",
    "beginners",
    "introduction to",
    "intro to",
    "step by step",
    "from zero",
    "from zero to hero",
    "zero to hero",
    "complete beginner",
    "absolute beginner",
    "total beginner",
    "start from zero",
)

_MEDIUM_BEGINNER_SIGNALS = (
    "basics",
    "basic",
    "fundamentals",
    "fundamental",
    "introduction",
    "intro",
    "introductory",
    "getting started",
    "tutorial",
    "tutorials",
    "primer",
    "quickstart",
    "quick start",
    "for dummies",
    "made easy",
    "from the ground up",
    "the hard way",
    "hand-holding",
    "kid friendly",
    "for newcomers",
    "for novices",
)

_WEAK_BEGINNER_SIGNALS = (
    "course",
    "courses",
    "textbook",
    "textbooks",
    "guide",
    "guides",
    "learn",
    "learning",
    "study",
    "studying",
    "overview",
    "explained",
    "simple",
    "easy",
    "self study",
)

_STRONG_ADVANCED_SIGNALS = (
    "mathematical foundations",
    "theoretical foundations",
    "principles of",
    "theory of",
    "foundations of",
    "graduate-level",
    "research-level",
    "axiomatic",
    "proof-based",
    "advanced treatment",
)

_MEDIUM_ADVANCED_SIGNALS = (
    "advanced",
    "deep dive",
    "deep-dive",
    "in depth",
    "in-depth",
    "rigorous",
    "rigour",
    "rigor",
    "formal treatment",
    "formal",
    "comprehensive understanding",
    "thorough understanding",
    "exhaustive",
    "in-depth treatment",
    "state of the art",
    "state-of-the-art",
    "cutting edge",
    "cutting-edge",
)

_WEAK_ADVANCED_SIGNALS = (
    "research",
    "graduate",
    "latest",
    "novel",
    "survey",
    "surveys",
    "benchmark",
    "benchmarks",
    "architecture",
    "architectures",
    "implementation",
    "implementations",
    "optimization",
    "frontier",
    "specialized",
    "specialised",
    "expert",
    "experts",
    "deeply understand",
    "deeply",
    "mathematical",
    "theoretical",
    "proofs",
    "proof",
    "theorem",
)

BEGINNER_SIGNAL_WEIGHTS: dict[str, float] = {
    "strong": 1.0,
    "medium": 0.6,
    "weak": 0.3,
}

ADVANCED_SIGNAL_WEIGHTS: dict[str, float] = {
    "strong": 1.0,
    "medium": 0.6,
    "weak": 0.3,
}