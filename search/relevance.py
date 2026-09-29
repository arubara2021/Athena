from __future__ import annotations

import math
import re
from typing import Any

from core.models import Source
from search.query_profile import QueryProfile
from search.query_tokens import (
    any_token_matches,
    count_token_matches,
    get_generic_terms,
    tokenize,
    tokens_match,
)
from utils.text import clean_text

_SENTENCE_SPLIT = re.compile(r"[.!?]+\s+")
_NORMALIZE_PATTERN = re.compile(r"[-_]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_WORD_SPLIT_PATTERN = re.compile(r"[^a-z0-9+#.]+")

_WEIGHTS: dict[str, float] = {
    "phrase_position": 0.30,
    "fingerprint_overlap": 0.25,
    "token_density": 0.15,
    "token_proximity": 0.15,
    "content_concentration": 0.15,
}

_GATE_SHORT_QUERY_MAX_TOKENS = 4
_GATE_LONG_QUERY_SCALE = 0.4


def _normalize_for_match(text: Any) -> str:
    if not text:
        return ""

    lowered = str(text).lower()
    lowered = _NORMALIZE_PATTERN.sub(" ", lowered)
    lowered = _WHITESPACE_PATTERN.sub(" ", lowered)

    return lowered.strip()


def _split_words(text: str) -> list[str]:
    if not text:
        return []

    normalized = _normalize_for_match(text)

    if not normalized:
        return []

    return [word for word in _WORD_SPLIT_PATTERN.split(normalized) if word]


def phrase_position_score(
    source: Source,
    phrase: str,
    tokens: list[str] | None = None,
) -> float:
    normalized_phrase = _normalize_for_match(phrase)

    if not normalized_phrase:
        return 0.0

    title = _normalize_for_match(getattr(source, "title", ""))
    abstract = _normalize_for_match(getattr(source, "abstract", "") or "")
    summary = _normalize_for_match(getattr(source, "summary", "") or "")

    for text, base_score in ((title, 1.0), (abstract, 0.75), (summary, 0.50)):
        if not text:
            continue

        position = text.find(normalized_phrase)

        if position >= 0:
            relative = position / max(1, len(text))
            return base_score * (1.0 - relative * 0.3)

    if tokens:
        normalized_tokens = [
            _normalize_for_match(token)
            for token in tokens
            if token
        ]

        normalized_tokens = [token for token in normalized_tokens if token]

        if normalized_tokens:
            for text, base_score in ((title, 0.85), (abstract, 0.60), (summary, 0.40)):
                if not text:
                    continue

                hits = sum(1 for token in normalized_tokens if token in text)

                if hits > 0:
                    coverage = hits / len(normalized_tokens)
                    return base_score * coverage

    return 0.0


def token_density(text: str, tokens: list[str]) -> float:
    if not text or not tokens:
        return 0.0

    lowered = _normalize_for_match(text)

    if not lowered:
        return 0.0

    word_count = max(1, len(lowered.split()))

    total = 0

    for token in tokens:
        normalized_token = _normalize_for_match(token)

        if not normalized_token:
            continue

        total += lowered.count(normalized_token)

    density = (total / word_count) * 100.0

    return min(1.0, density / (density + 3.0))


def token_proximity(text: str, tokens: list[str]) -> float:
    if not text or not tokens:
        return 0.0

    if len(tokens) < 2:
        normalized = _normalize_for_match(text)
        if not normalized:
            return 0.0
        word = _normalize_for_match(tokens[0])
        return 1.0 if word and word in normalized else 0.0

    lowered = _normalize_for_match(text)
    words = lowered.split()

    if not words:
        return 0.0

    positions: dict[str, list[int]] = {}

    for index, word in enumerate(words):
        for token in tokens:
            normalized_token = _normalize_for_match(token)

            if not normalized_token:
                continue

            if tokens_match(normalized_token, word):
                positions.setdefault(normalized_token, []).append(index)

    if len(positions) < 2:
        return 0.0

    all_positions: list[tuple[int, str]] = []

    for token, pos_list in positions.items():
        for pos in pos_list:
            all_positions.append((pos, token))

    all_positions.sort()

    token_set = set(positions.keys())
    best_window = len(words)

    for start_index in range(len(all_positions)):
        seen: set[str] = set()

        for end_index in range(start_index, len(all_positions)):
            seen.add(all_positions[end_index][1])

            if seen >= token_set:
                window = all_positions[end_index][0] - all_positions[start_index][0] + 1

                if window < best_window:
                    best_window = window

                break

    if best_window >= len(words):
        return 0.0

    expected_gap = max(1.0, len(words) / max(1, len(tokens)))
    ratio = expected_gap / max(1.0, best_window)

    return min(1.0, ratio / 2.5)


def title_topic_signature(
    title: str,
    generic_terms: frozenset[str] | None = None,
) -> list[str]:
    if not title:
        return []

    generics = generic_terms if generic_terms is not None else get_generic_terms()
    tokens = tokenize(title)

    signature: list[str] = []

    for token in tokens:
        if len(token) < 3:
            continue
        if token in generics:
            continue
        if token not in signature:
            signature.append(token)

    signature.sort(key=lambda item: (-len(item), item))

    return signature


def topic_overlap(
    source_signature: list[str],
    query_signature: list[str],
) -> float:
    if not query_signature:
        return 1.0

    if not source_signature:
        return 0.0

    matched = count_token_matches(query_signature, source_signature)

    return min(1.0, matched / len(query_signature))


def content_concentration(
    abstract: str,
    concept_tokens: list[str],
) -> float:
    if not abstract or not concept_tokens:
        return 0.0

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(abstract) if s.strip()]

    if len(sentences) < 2:
        return 0.5

    first_portion = " ".join(sentences[:2]).lower()
    whole = abstract.lower()

    first_count = sum(first_portion.count(token) for token in concept_tokens)
    whole_count = sum(whole.count(token) for token in concept_tokens)

    if whole_count == 0:
        return 0.0

    first_ratio = first_count / whole_count
    length_ratio = len(first_portion) / max(1, len(whole))

    if length_ratio <= 0:
        return 0.0

    concentration = first_ratio / length_ratio

    return min(1.0, concentration / 2.0)


def source_relevance_vector(
    source: Source,
    profile: QueryProfile,
) -> tuple[dict[str, float], float]:
    concept_phrase = profile.concept_phrase
    concept_tokens = profile.concept_tokens
    fingerprint = profile.concept_fingerprint

    if not concept_tokens:
        neutral = {
            "phrase_position": 1.0,
            "fingerprint_overlap": 1.0,
            "token_density": 1.0,
            "token_proximity": 1.0,
            "content_concentration": 1.0,
        }
        return neutral, 1.0

    title = clean_text(getattr(source, "title", ""))
    abstract = clean_text(getattr(source, "abstract", "") or "")
    summary = clean_text(getattr(source, "summary", "") or "")
    combined = f"{title} {abstract} {summary}".strip()

    position_score = phrase_position_score(source, concept_phrase, concept_tokens)

    if position_score == 0.0 and fingerprint:
        position_score = phrase_position_score(source, " ".join(fingerprint), fingerprint)

    density_score = token_density(combined, concept_tokens)

    proximity_source = abstract or combined
    proximity_score = token_proximity(proximity_source, concept_tokens)

    source_signature = title_topic_signature(title)
    overlap_score = topic_overlap(source_signature, fingerprint)

    concentration_score = content_concentration(abstract, concept_tokens)

    vector = {
        "phrase_position": position_score,
        "fingerprint_overlap": overlap_score,
        "token_density": density_score,
        "token_proximity": proximity_score,
        "content_concentration": concentration_score,
    }

    score = 0.0

    for key, weight in _WEIGHTS.items():
        score += vector.get(key, 0.0) * weight

    return vector, max(0.0, min(1.0, score))


def passes_concept_presence_gate(
    source: Source,
    profile: QueryProfile,
) -> bool:
    concept_tokens = profile.concept_tokens

    if not concept_tokens:
        return True

    title_raw = getattr(source, "title", "") or ""
    abstract_raw = getattr(source, "abstract", "") or ""

    if not title_raw and not abstract_raw:
        return False

    title = _normalize_for_match(title_raw)
    abstract_head = _normalize_for_match(abstract_raw[:400])

    if not title and not abstract_head:
        return False

    title_words = _split_words(title)
    abstract_words = _split_words(abstract_head)

    for token in concept_tokens:
        normalized_token = _normalize_for_match(token)

        if not normalized_token:
            continue

        if normalized_token in title:
            return True

        if normalized_token in abstract_head:
            return True

        if any_token_matches(normalized_token, title_words):
            return True

        if any_token_matches(normalized_token, abstract_words):
            return True

    return False


def _required_gate_overlap(
    size: int,
    has_qualifier: bool,
    user_specificity: float,
) -> int:
    if size <= 0:
        return 0

    if size <= _GATE_SHORT_QUERY_MAX_TOKENS:
        return 1

    return max(1, int(math.ceil(size * _GATE_LONG_QUERY_SCALE)))


def passes_title_fingerprint_gate(
    source: Source,
    profile: QueryProfile,
) -> bool:
    if not profile.has_specific_concept:
        return True

    user_tokens = list(profile.user_typed_tokens or [])
    concept_tokens = list(profile.concept_fingerprint or [])

    if not user_tokens and not concept_tokens:
        return True

    title = clean_text(getattr(source, "title", ""))

    if not title:
        return True

    source_type = str(
        getattr(getattr(source, "source_type", ""), "value", "")
        or getattr(source, "source_type", "")
    ).strip().lower()

    is_code_like = source_type in {"repository", "model", "dataset"}

    if is_code_like:
        abstract = clean_text(getattr(source, "abstract", "") or "")
        combined = f"{title} {abstract[:400]}".strip()
        source_signature = title_topic_signature(combined)
    else:
        source_signature = title_topic_signature(title)

    if not source_signature:
        return False

    abbreviation_tokens = list(profile.abbreviation_tokens or [])

    if abbreviation_tokens:
        abbreviation_matched = count_token_matches(
            abbreviation_tokens,
            source_signature,
        )

        if abbreviation_matched <= 0:
            return False

    has_qualifier = bool(profile.qualifier_tokens) or bool(profile.has_abbreviation)

    if user_tokens:
        user_matched = count_token_matches(user_tokens, source_signature)

        user_required = _required_gate_overlap(
            len(user_tokens),
            has_qualifier,
            profile.user_typed_specificity,
        )

        if user_matched >= user_required:
            return True

    if concept_tokens:
        concept_matched = count_token_matches(concept_tokens, source_signature)

        concept_required = _required_gate_overlap(
            len(concept_tokens),
            has_qualifier,
            profile.specificity_score,
        )

        if concept_matched >= concept_required:
            return True

    return False