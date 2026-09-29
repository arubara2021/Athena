from __future__ import annotations

import re
from typing import Any

from pydantic import Field

from core.models import CoreModel
from search.query_tokens import (
    get_generic_terms,
    strip_filler,
    tokenize,
    tokens_match,
)
from utils.text import clean_text

_RAW_WORD_PATTERN = re.compile(r"[A-Za-z0-9+#.]+")
_SHORT_LOWERCASE_PATTERN = re.compile(r"^[a-z]{2,4}$")

_VOWELS = frozenset({"a", "e", "i", "o", "u", "y"})

_LEVEL_TOKENS = frozenset({
    "beginner",
    "beginners",
    "basic",
    "basics",
    "fundamental",
    "fundamentals",
    "intro",
    "introduction",
    "introductory",
    "advanced",
    "advance",
    "advances",
    "advancement",
    "advancements",
    "expert",
    "intermediate",
    "mixed",
    "easy",
    "simple",
    "complete",
    "comprehensive",
    "quick",
    "fast",
    "slow",
    "crash",
    "from",
    "scratch",
    "step",
    "steps",
    "starter",
    "starting",
    "start",
    "for",
    "no",
    "prior",
    "to",
    "and",
    "beyond",
    "up",
})

_COMPOUND_FORMING_TERMS = frozenset({
    "learning",
    "learnings",
    "network",
    "networks",
    "networking",
    "model",
    "models",
    "modeling",
    "modelling",
    "system",
    "systems",
    "theory",
    "theories",
    "method",
    "methods",
    "methodology",
    "methodologies",
    "analysis",
    "analyses",
    "design",
    "designs",
    "designing",
    "engineering",
    "engineer",
    "engineers",
    "science",
    "sciences",
    "scientific",
    "study",
    "studies",
    "studying",
    "process",
    "processes",
    "processing",
    "data",
    "structure",
    "structures",
    "structural",
    "control",
    "controls",
    "controlling",
    "mining",
    "vision",
    "language",
    "languages",
    "search",
    "searching",
    "retrieval",
    "retrieving",
    "recommendation",
    "recommendations",
    "recommender",
    "recognition",
    "detection",
    "detecting",
    "prediction",
    "predictions",
    "predicting",
    "classification",
    "classifications",
    "classifying",
    "regression",
    "clustering",
    "cluster",
    "clusters",
    "optimization",
    "optimisation",
    "training",
    "inference",
    "generation",
    "generating",
    "translation",
    "translating",
    "constitution",
    "constitutional",
    "framework",
    "frameworks",
    "law",
    "laws",
    "legal",
    "policy",
    "policies",
    "governance",
    "government",
    "governments",
    "parliament",
    "parliaments",
    "court",
    "courts",
    "rights",
    "right",
    "treaty",
    "treaties",
    "agreement",
    "agreements",
    "protocol",
    "protocols",
})

_INTENT_SOURCE_TYPES: dict[str, tuple[str, ...]] = {
    "learn": ("course", "book", "documentation", "video"),
    "research": ("research_paper", "dataset", "repository"),
    "implement": ("repository", "model", "documentation"),
    "compare": ("research_paper", "documentation"),
    "troubleshoot": ("documentation", "repository"),
    "explore": ("documentation", "research_paper", "book"),
}

_LEVEL_SOURCE_TYPES: dict[str, tuple[str, ...]] = {
    "beginner": ("course", "book", "documentation", "video"),
    "intermediate": ("course", "book", "documentation", "repository"),
    "advanced": ("research_paper", "repository", "documentation"),
    "mixed": ("documentation", "course", "research_paper"),
    "balanced": ("documentation", "course", "research_paper"),
}

_DEFAULT_SOURCE_TYPES = ("documentation", "course", "research_paper")

_MAX_LLM_ADDED_TOKENS = 1

_ABBREVIATION_MIN_LENGTH = 2
_ABBREVIATION_MAX_LENGTH = 5


class QueryProfile(CoreModel):
    original_query: str = ""
    concept_phrase: str = ""
    concept_tokens: list[str] = Field(default_factory=list)
    concept_fingerprint: list[str] = Field(default_factory=list)
    level_profile: str = ""
    level_words: list[str] = Field(default_factory=list)
    expected_source_types: list[str] = Field(default_factory=list)
    target_domains: list[str] = Field(default_factory=list)
    query_length: int = 0
    specificity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    primary_intent: str = ""
    has_specific_concept: bool = False

    concept_origin: str = "unknown"
    qualifier_tokens: list[str] = Field(default_factory=list)

    user_typed_tokens: list[str] = Field(default_factory=list)
    user_typed_specificity: float = Field(default=0.0, ge=0.0, le=1.0)
    expanded_tokens: list[str] = Field(default_factory=list)
    has_abbreviation: bool = False
    abbreviation_tokens: list[str] = Field(default_factory=list)


def build_query_profile(
    query: str,
    corrected_topic: str = "",
    primary_concept: str = "",
    level_profile: str = "",
    intent: str = "",
    target_domains: list[str] | None = None,
    target_formats: list[str] | None = None,
    preserve_tokens: list[str] | None = None,
    original_query: str = "",
) -> QueryProfile:
    raw_user_query = clean_text(original_query) or clean_text(query)

    if primary_concept:
        topic_source = clean_text(primary_concept)
    elif corrected_topic:
        topic_source = clean_text(corrected_topic)
    else:
        topic_source = raw_user_query

    normalized_preserve = _normalize_preserve_tokens(preserve_tokens)

    user_typed_tokens = _compute_user_typed_tokens(raw_user_query)
    has_abbreviation = _has_abbreviation_token(raw_user_query, user_typed_tokens)
    abbreviation_tokens = _collect_abbreviation_tokens(
        raw_user_query,
        user_typed_tokens,
    )

    if not topic_source:
        user_specificity = _compute_specificity(
            _fingerprint(user_typed_tokens)
        )

        return QueryProfile(
            original_query=raw_user_query,
            level_profile=_normalize_level(level_profile),
            primary_intent=_normalize_intent(intent),
            expected_source_types=_expected_source_types(intent, level_profile),
            target_domains=_clean_string_list(target_domains),
            qualifier_tokens=normalized_preserve,
            concept_origin="unknown" if not normalized_preserve else "qualified",
            user_typed_tokens=user_typed_tokens,
            user_typed_specificity=user_specificity,
            has_abbreviation=has_abbreviation,
            abbreviation_tokens=abbreviation_tokens,
        )

    stripped = strip_filler(topic_source)

    phrase_after_level, level_words = strip_level_words(stripped)
    phrase_after_level = _strip_generic_only_phrase(phrase_after_level)
    concept_phrase = _clean_phrase(phrase_after_level)
    concept_tokens_raw = _concept_tokens(concept_phrase)

    concept_tokens = _cap_concept_tokens(
        concept_tokens_raw,
        user_typed_tokens,
        has_abbreviation=has_abbreviation,
        max_extra=_MAX_LLM_ADDED_TOKENS,
    )

    _merge_preserve_into_concept(concept_tokens, normalized_preserve)

    concept_fingerprint = _fingerprint(concept_tokens)
    expanded_tokens = [
        token for token in concept_tokens if token not in user_typed_tokens
    ]

    if user_typed_tokens and not has_abbreviation:
        user_fingerprint = _fingerprint(user_typed_tokens)
        user_specificity = _compute_specificity(user_fingerprint)
        specificity = user_specificity
        has_specific = bool(user_fingerprint) and user_specificity >= 0.30
    elif user_typed_tokens and has_abbreviation:
        user_fingerprint = _fingerprint(user_typed_tokens)
        user_specificity = _compute_specificity(user_fingerprint)
        specificity = _compute_specificity(concept_fingerprint)
        has_specific = bool(concept_fingerprint) and specificity >= 0.30
    else:
        user_specificity = 0.0
        specificity = _compute_specificity(concept_fingerprint)
        has_specific = bool(concept_fingerprint) and specificity >= 0.30

    qualifier_tokens = _detect_qualifier_tokens(
        concept_tokens=concept_tokens,
        raw_query=raw_user_query,
        preserve_tokens=normalized_preserve,
        corrected_topic=corrected_topic,
        primary_concept=primary_concept,
    )

    concept_origin = _detect_concept_origin(
        concept_tokens=concept_tokens,
        raw_query=raw_user_query,
        qualifier_tokens=qualifier_tokens,
        user_typed_tokens=user_typed_tokens,
        has_abbreviation=has_abbreviation,
    )

    return QueryProfile(
        original_query=raw_user_query,
        concept_phrase=concept_phrase,
        concept_tokens=concept_tokens,
        concept_fingerprint=concept_fingerprint,
        level_profile=_normalize_level(level_profile),
        level_words=level_words,
        expected_source_types=_expected_source_types(intent, level_profile),
        target_domains=_clean_string_list(target_domains),
        query_length=len(concept_tokens),
        specificity_score=specificity,
        primary_intent=_normalize_intent(intent),
        has_specific_concept=has_specific,
        concept_origin=concept_origin,
        qualifier_tokens=qualifier_tokens,
        user_typed_tokens=user_typed_tokens,
        user_typed_specificity=user_specificity,
        expanded_tokens=expanded_tokens,
        has_abbreviation=has_abbreviation,
        abbreviation_tokens=abbreviation_tokens,
    )


def _normalize_preserve_tokens(value: Any) -> list[str]:
    if not value:
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


def _compute_user_typed_tokens(original_query: str) -> list[str]:
    if not original_query:
        return []

    stripped = strip_filler(original_query)
    after_level, _ = strip_level_words(stripped)
    cleaned = _clean_phrase(after_level)

    if not cleaned:
        return []

    generics = get_generic_terms()
    raw_tokens = [token.lower() for token in _RAW_WORD_PATTERN.findall(cleaned)]

    result: list[str] = []

    for token in raw_tokens:
        if len(token) < 2:
            continue
        if token in _LEVEL_TOKENS:
            continue
        if token in generics:
            continue
        if token not in result:
            result.append(token)

    return result


def _is_abbreviation_token(
    token: str,
    raw_query: str,
) -> bool:
    if not token:
        return False

    text = str(token).strip()

    if len(text) < _ABBREVIATION_MIN_LENGTH or len(text) > _ABBREVIATION_MAX_LENGTH:
        return False

    if text.isdigit():
        return False

    lowered = text.lower()

    if not raw_query:
        return False

    raw_words = _RAW_WORD_PATTERN.findall(raw_query)

    for raw_word in raw_words:
        if raw_word.lower() != lowered:
            continue

        if raw_word.isupper() and len(raw_word) >= 2:
            return True

    if len(lowered) >= 2 and all(ch.isalpha() for ch in lowered):
        has_vowel = any(ch in _VOWELS for ch in lowered)

        if not has_vowel:
            return True

        if _SHORT_LOWERCASE_PATTERN.match(lowered):
            vowel_count = sum(1 for ch in lowered if ch in _VOWELS)

            if vowel_count <= 1:
                return True

    return False


def _has_abbreviation_token(
    raw_query: str,
    user_tokens: list[str],
) -> bool:
    if not user_tokens:
        return False

    for token in user_tokens:
        if _is_abbreviation_token(token, raw_query):
            return True

    return False


def _cap_concept_tokens(
    concept_tokens: list[str],
    user_tokens: list[str],
    has_abbreviation: bool = False,
    max_extra: int = _MAX_LLM_ADDED_TOKENS,
) -> list[str]:
    if not concept_tokens:
        return []

    if not user_tokens:
        return concept_tokens

    if has_abbreviation:
        return concept_tokens

    if len(concept_tokens) <= len(user_tokens) + max_extra:
        return concept_tokens

    user_origin: list[str] = []
    llm_added: list[str] = []

    for concept_token in concept_tokens:
        matched_to_user = False

        for user_token in user_tokens:
            if tokens_match(concept_token, user_token):
                matched_to_user = True
                break

        if matched_to_user:
            user_origin.append(concept_token)
        else:
            llm_added.append(concept_token)

    capped_added = llm_added[: max(1, max_extra)]
    keep_set = set(user_origin) | set(capped_added)

    ordered: list[str] = []

    for token in concept_tokens:
        if token in keep_set and token not in ordered:
            ordered.append(token)

    if not ordered:
        return concept_tokens

    return ordered


def _merge_preserve_into_concept(
    concept_tokens: list[str],
    preserve_tokens: list[str],
) -> None:
    if not concept_tokens or not preserve_tokens:
        return

    for preserve in preserve_tokens:
        text = str(preserve or "").strip().lower()

        if not text:
            continue

        already_present = False

        for existing in concept_tokens:
            if tokens_match(text, existing):
                already_present = True
                break

        if not already_present:
            concept_tokens.append(text)


def strip_level_words(text: str) -> tuple[str, list[str]]:
    if not text:
        return "", []

    tokens = text.split()
    kept: list[str] = []
    stripped: list[str] = []

    for token in tokens:
        cleaned = token.strip(",.!?;:\"'()[]{}").lower()

        if cleaned in _LEVEL_TOKENS:
            if cleaned not in stripped:
                stripped.append(cleaned)
            continue

        kept.append(token)

    return " ".join(kept).strip(), stripped


def _strip_generic_only_phrase(text: str) -> str:
    if not text:
        return ""

    tokens = tokenize(text)
    generics = get_generic_terms()

    meaningful = [
        token
        for token in tokens
        if token not in generics and len(token) >= 3
    ]

    if meaningful:
        return text

    return ""


def _clean_phrase(text: str) -> str:
    if not text:
        return ""

    cleaned = text.lower()
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    cleaned = " ".join(cleaned.split())

    return cleaned.strip()


def _concept_tokens(phrase: str) -> list[str]:
    if not phrase:
        return []

    generics = get_generic_terms()
    raw_tokens = [token.lower() for token in _RAW_WORD_PATTERN.findall(phrase)]

    if not raw_tokens:
        return []

    meaningful_mask: list[bool] = []

    for token in raw_tokens:
        if len(token) < 2:
            meaningful_mask.append(False)
            continue

        if token in generics:
            meaningful_mask.append(False)
            continue

        meaningful_mask.append(True)

    result: list[str] = []

    for index, token in enumerate(raw_tokens):
        if len(token) < 2:
            continue

        keep = meaningful_mask[index]

        if not keep and token in _COMPOUND_FORMING_TERMS:
            left_meaningful = index > 0 and meaningful_mask[index - 1]
            right_meaningful = (
                index < len(raw_tokens) - 1 and meaningful_mask[index + 1]
            )

            if left_meaningful or right_meaningful:
                keep = True

        if keep and token not in result:
            result.append(token)

    return result


def _fingerprint(tokens: list[str]) -> list[str]:
    if not tokens:
        return []

    ordered = sorted(tokens, key=lambda item: (-len(item), item))

    return ordered


def _compute_specificity(fingerprint: list[str]) -> float:
    if not fingerprint:
        return 0.0

    count = len(fingerprint)
    average_length = sum(len(item) for item in fingerprint) / count

    count_component = min(0.55, count * 0.15)
    length_component = min(0.25, max(0.0, (average_length - 4.0) * 0.08))
    multiword_component = 0.15 if count >= 2 else 0.0

    return min(1.0, count_component + length_component + multiword_component)


def _detect_qualifier_tokens(
    concept_tokens: list[str],
    raw_query: str,
    preserve_tokens: list[str],
    corrected_topic: str,
    primary_concept: str,
) -> list[str]:
    if not concept_tokens or not preserve_tokens:
        return []

    result: list[str] = []

    preserve_lower = [
        str(item or "").strip().lower()
        for item in preserve_tokens
        if str(item or "").strip()
    ]

    if not preserve_lower:
        return result

    for token in concept_tokens:
        for preserve in preserve_lower:
            if tokens_match(token, preserve):
                if token not in result:
                    result.append(token)
                break

    return result


def _detect_concept_origin(
    concept_tokens: list[str],
    raw_query: str,
    qualifier_tokens: list[str],
    user_typed_tokens: list[str],
    has_abbreviation: bool = False,
) -> str:
    if not concept_tokens:
        return "unknown"

    if qualifier_tokens:
        return "qualified"

    if has_abbreviation:
        return "qualified"

    if user_typed_tokens and len(concept_tokens) > len(user_typed_tokens):
        return "qualified"

    lower_query = clean_text(raw_query).lower()

    if not lower_query:
        return "unknown"

    full_phrase = " ".join(concept_tokens)

    if full_phrase and full_phrase in lower_query:
        return "fixed"

    all_present = all(token in lower_query for token in concept_tokens)

    if all_present:
        return "fixed"

    return "qualified"


def _expected_source_types(intent: str, level: str) -> list[str]:
    intent_key = _normalize_intent(intent)
    level_key = _normalize_level(level)

    result: list[str] = []

    for item in _INTENT_SOURCE_TYPES.get(intent_key, ()):
        if item not in result:
            result.append(item)

    for item in _LEVEL_SOURCE_TYPES.get(level_key, ()):
        if item not in result:
            result.append(item)

    if not result:
        result = list(_DEFAULT_SOURCE_TYPES)

    return result


def _normalize_intent(intent: str) -> str:
    text = str(intent or "").strip().lower()

    if text in _INTENT_SOURCE_TYPES:
        return text

    return ""


def _normalize_level(level: str) -> str:
    text = str(level or "").strip().lower()

    if text in {"beginner", "intermediate", "advanced", "mixed", "balanced"}:
        return text

    return ""


def _clean_string_list(values: Any) -> list[str]:
    if not values:
        return []

    if isinstance(values, str):
        values = [values]

    if not isinstance(values, (list, tuple, set)):
        return []

    result: list[str] = []

    for value in values:
        text = str(value or "").strip().lower()

        if text and text not in result:
            result.append(text)

    return result


def _collect_abbreviation_tokens(
    raw_query: str,
    user_tokens: list[str],
) -> list[str]:
    if not user_tokens:
        return []

    result: list[str] = []

    for token in user_tokens:
        if _is_abbreviation_token(token, raw_query):
            if token not in result:
                result.append(token)

    return result