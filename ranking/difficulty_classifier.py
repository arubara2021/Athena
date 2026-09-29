from __future__ import annotations

import math
import re
from typing import Any

from pydantic import Field

from core.models import CoreModel, Difficulty, Source
from core.ranking_config import get_ranking_config
from utils.logger import get_logger, get_trace_logger
from utils.text import clean_text

try:
    from search.query_tokens import get_generic_terms, tokenize
except Exception:
    _FALLBACK_GENERIC_TERMS: frozenset[str] = frozenset({
        "the", "and", "for", "with", "from", "this", "that", "these",
        "those", "into", "onto", "upon", "about", "above", "below",
        "each", "every", "some", "any", "all", "both", "many", "much",
        "more", "most", "other", "another", "such", "only", "also",
        "very", "just", "then", "than", "when", "where", "which",
        "while", "what", "why", "how", "who", "whom", "whose",
        "will", "would", "shall", "should", "may", "might", "must",
        "can", "could", "have", "has", "had", "been", "being",
        "does", "did", "done", "doing", "was", "were", "are", "is",
        "am", "be",
    })

    def get_generic_terms() -> frozenset[str]:
        return _FALLBACK_GENERIC_TERMS

    def tokenize(value: Any) -> list[str]:
        return re.findall(r"[a-z0-9]+", str(value or "").lower())


_DEFAULT_CONFIG: dict[str, Any] = {
    "treatment_weight": 0.55,
    "topic_weight": 0.35,
    "metadata_weight": 0.10,
    "prior_cap": 0.15,
    "beginner_threshold": 0.35,
    "advanced_threshold": 0.65,
    "treatment_neutral": 0.50,
    "topic_neutral": 0.40,
    "metadata_neutral": 0.50,
    "confidence_base": 0.30,
    "confidence_signal_weight": 0.05,
    "confidence_signal_cap": 6,
    "confidence_text_weight": 0.10,
    "confidence_text_full_length": 500,
    "confidence_agreement_weight": 0.20,
    "confidence_explicit_boost": 0.15,
    "confidence_min": 0.10,
    "confidence_max": 0.95,
}

_DEFAULT_TITLE_PROMOTION: dict[str, Any] = {
    "enabled": True,
    "confidence_floor": 0.75,
    "starts_advanced": [
        "advanced",
        "graduate",
        "postgraduate",
    ],
    "advanced_qualifiers": [
        "mechanism",
        "mechanisms",
        "reaction",
        "reactions",
        "synthesis",
        "syntheses",
        "structure",
        "structures",
        "theory",
        "analysis",
        "applications",
    ],
    "beginner_phrases": [
        "introduction to",
        "intro to",
        "for beginners",
        "fundamentals of",
        "basics of",
        "getting started",
        "from scratch",
        "step by step",
        "beginner's guide",
        "made easy",
        "basics",
        "basic",
        "fundamentals",
        "introductory",
    ],
}

_DEFAULT_DISTRIBUTION_GUARD: dict[str, Any] = {
    "enabled": True,
    "trigger_advanced_ratio": 0.50,
    "demote_advanced_max_treatment": 0.45,
    "demote_intermediate_max_treatment": 0.30,
    "demote_intermediate_max_topic": 0.50,
    "require_beginner_in_level": True,
}

_BEGINNER_MARKERS: tuple[str, ...] = (
    "introduction to",
    "introduction",
    "introductory",
    "beginners",
    "beginner",
    "basics",
    "basic",
    "fundamentals",
    "getting started",
    "from scratch",
    "step by step",
    "for beginners",
    "no prior",
    "made easy",
    "for dummies",
    "tutorial",
    "primer",
    "quickstart",
    "quick start",
    "textbook",
    "self-study",
)

_ADVANCED_MARKERS: tuple[str, ...] = (
    "advanced",
    "graduate",
    "postgraduate",
    "specialized",
    "specialised",
    "state of the art",
    "state-of-the-art",
    "cutting edge",
    "cutting-edge",
    "monograph",
    "theorem",
    "lemma",
    "corollary",
    "proposition",
    "rigorous",
    "formal treatment",
    "assumes familiarity",
    "assumes background",
    "for researchers",
    "for experts",
    "research monograph",
    "survey of",
    "comprehensive treatment",
)

_ACADEMIC_PROSE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwe\s+(?:prove|show|demonstrate|establish|derive|present|propose|introduce|construct|analyse|analyze)\b"),
    re.compile(r"\b(?:suppose|assume)\s+that\b"),
    re.compile(r"\blet\s+[a-z]\s+be\b"),
    re.compile(r"\bin\s+this\s+(?:paper|section|chapter|study|article|work)\b"),
    re.compile(r"\bour\s+(?:results|approach|method|framework|model|analysis|contribution)\b"),
    re.compile(r"\bas\s+shown\s+in\b"),
    re.compile(r"\bit\s+(?:follows|can\s+be\s+shown)\b"),
    re.compile(r"\b(?:furthermore|moreover|consequently|nonetheless)\b"),
    re.compile(r"\b(?:theorem|lemma|proposition|corollary)\s+\d+\b"),
    re.compile(r"\bq\.?e\.?d\.?\b"),
)

_BEGINNER_PROSE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\byou\s+(?:will|can|might|should|need)\b"),
    re.compile(r"\bwe'?ll\b"),
    re.compile(r"\blet'?s\b"),
    re.compile(r"\btry\s+(?:this|the\s+following)\b"),
    re.compile(r"\bin\s+this\s+(?:tutorial|guide|lesson)\b"),
    re.compile(r"\bwe\s+will\s+learn\b"),
    re.compile(r"\byour\s+(?:first|own)\b"),
    re.compile(r"\bgetting\s+started\b"),
)

_FORMAL_NOTATION_PATTERN = re.compile(
    r"[∈∀∃⊂⊆⊇∪∩∧∨¬⇒⇔→←↔∑∏∫∂∇ℏαβγδεζηθικλμνξπρστυφχψω]"
)

_LATEX_PATTERN = re.compile(
    r"\\(?:frac|sum|int|partial|alpha|beta|gamma|delta|sigma|omega|theta|lambda|infty|cdot|times|le|ge|ne|approx|equiv|to|rightarrow|leftarrow|mathbb|mathcal|mathrm|cdot|ldots|cdots|nabla|langle|rangle|sqrt)"
)

_SUBSCRIPT_PATTERN = re.compile(r"\b[A-Za-z][_^]\{?[A-Za-z0-9]+\}?")

_REFERENCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\[\d+(?:[\s,\-]+\d+)*\]"),
    re.compile(r"\(\s*[A-Z][a-z]+\s+(?:et\s+al\.?\s+)?\d{4}\s*\)"),
    re.compile(r"arXiv:\s*\d+\.\d+"),
    re.compile(r"\bdoi:\s*10\.\S+", re.IGNORECASE),
    re.compile(r"https?://doi\.org/10\.\S+"),
    re.compile(r"\b(?:Figure|Fig\.|Table|Eq\.|Equation|Section|Chapter|Appendix|Theorem)\s+\d+"),
)

_TECHNICAL_SUFFIXES: tuple[str, ...] = (
    "tion", "sion", "ment", "ance", "ence", "ity", "ism",
    "ology", "ology", "graphy", "nomy", "metry", "esis",
    "itic", "ical", "emia", "pathy", "plasty", "phagy",
    "morphism", "hedron", "tope", "thesis", "stasis",
    "genesis", "kinesis", "lysis", "ptosis",
)

_TECHNICAL_PREFIXES: tuple[str, ...] = (
    "hydro", "electro", "thermo", "photo", "bio", "neuro",
    "psycho", "physio", "chemo", "astro", "geo", "morpho",
    "phono", "chromato", "spectro", "cyto", "histo",
    "hemo", "haemo", "cardio", "pulmo", "nephro", "hepato",
    "onco", "pharma", "immuno", "micro", "macro", "nano",
    "pico", "femto", "iso", "poly", "mono", "di", "tri",
    "tetra", "penta", "hexa", "hepta", "octa",
)

_NAMED_ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+\b")

_SENTENCE_SPLIT = re.compile(r"[.!?]+")
_WORD_SPLIT = re.compile(r"\s+")
_ALPHA_TOKEN = re.compile(r"[A-Za-z]+")

_PROPER_NOUN_TITLE_PATTERN = re.compile(r"\b[A-Z][a-z]{3,}\b")

_SELF_PUBLISHED_MARKERS: tuple[str, ...] = (
    "blogspot", "medium.com", "substack", "wordpress",
    "self-published", "self published", "kindle direct",
    "createspace",
)

_ACADEMIC_PUBLISHER_MARKERS: tuple[str, ...] = (
    "university press", "university of", "academic press",
    "springer", "elsevier", "wiley", "cambridge university",
    "oxford university", "mit press", "princeton university",
    "crc press", "sage publications", "taylor & francis",
    "taylor and francis", "kluwer", "mcgraw-hill", "mcgraw hill",
    "pearson education", "cengage learning", "palgrave",
    "routledge", "world scientific", "ieee", "acm press",
    "de gruyter", "john wiley",
)

_PLATFORM_PRIOR: dict[str, float] = {
    "arxiv": 0.12,
    "semantic_scholar": 0.10,
    "openalex": 0.08,
    "pubmed": 0.12,
    "europe_pmc": 0.10,
    "crossref": 0.10,
    "doaj": 0.05,
    "core": 0.10,
    "zenodo": 0.02,
    "wikipedia": -0.05,
    "wikibooks": -0.05,
    "wikiversity": -0.05,
    "openstax": -0.05,
    "mit_ocw": -0.03,
    "libretexts": -0.05,
    "open_library": -0.03,
    "internet_archive": -0.03,
    "github": 0.00,
    "huggingface": 0.05,
    "web": 0.00,
    "tavily": 0.00,
    "exa": 0.00,
    "serper": 0.00,
    "serpapi": 0.00,
    "jina": 0.00,
}

_DIFFICULTY_RANK: dict[Difficulty, int] = {
    Difficulty.UNKNOWN: -1,
    Difficulty.BEGINNER: 0,
    Difficulty.INTERMEDIATE: 1,
    Difficulty.ADVANCED: 2,
}


class DifficultyClassification(CoreModel):
    difficulty: Difficulty = Difficulty.INTERMEDIATE
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    treatment_score: float = Field(default=0.5, ge=0.0, le=1.0)
    topic_score: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata_score: float = Field(default=0.5, ge=0.0, le=1.0)
    prior_offset: float = 0.0
    signals_fired: int = Field(default=0, ge=0)
    reason: str | None = None
    title_promoted: bool = False
    title_rule: str = ""
    title_token: str = ""


class DifficultyClassifier:
    def __init__(self, config: Any | None = None) -> None:
        self._logger = get_logger("ranking.difficulty_classifier")
        self._trace = get_trace_logger()
        self._ranking_config = self._safe_ranking_config()
        self._config = self._load_two_dimension_config(config)
        self._generic_terms = get_generic_terms()
        self._title_promotion_config = self._load_title_promotion_config()
        self._distribution_guard_config = self._load_distribution_guard_config()

        self._title_promotion_enabled = bool(
            self._title_promotion_config.get("enabled", True)
        )
        self._title_promotion_confidence_floor = float(
            self._title_promotion_config.get("confidence_floor", 0.75)
        )
        self._title_starts_advanced = tuple(
            self._title_promotion_config.get("starts_advanced", [])
        )
        self._title_advanced_qualifiers = tuple(
            self._title_promotion_config.get("advanced_qualifiers", [])
        )
        self._title_beginner_phrases = tuple(
            self._title_promotion_config.get("beginner_phrases", [])
        )

        self._distribution_guard_enabled = bool(
            self._distribution_guard_config.get("enabled", True)
        )
        self._distribution_guard_trigger_ratio = float(
            self._distribution_guard_config.get(
                "trigger_advanced_ratio", 0.50
            )
        )
        self._distribution_guard_demote_advanced_max_treatment = float(
            self._distribution_guard_config.get(
                "demote_advanced_max_treatment", 0.45
            )
        )
        self._distribution_guard_demote_intermediate_max_treatment = float(
            self._distribution_guard_config.get(
                "demote_intermediate_max_treatment", 0.30
            )
        )
        self._distribution_guard_demote_intermediate_max_topic = float(
            self._distribution_guard_config.get(
                "demote_intermediate_max_topic", 0.50
            )
        )
        self._distribution_guard_require_beginner_in_level = bool(
            self._distribution_guard_config.get(
                "require_beginner_in_level", True
            )
        )

    def classify_sources(
        self,
        sources: list[Any],
        level: str = "",
    ) -> list[Source]:
        updated_sources: list[Source] = []

        for source in sources:
            if not isinstance(source, Source):
                continue

            try:
                classification = self.classify_source(source, level=level)

                new_metadata = dict(getattr(source, "metadata", {}) or {})
                new_metadata["classifier_confidence"] = round(
                    classification.confidence, 4
                )
                new_metadata["classifier_treatment"] = round(
                    classification.treatment_score, 4
                )
                new_metadata["classifier_topic"] = round(
                    classification.topic_score, 4
                )
                new_metadata["classifier_metadata"] = round(
                    classification.metadata_score, 4
                )
                new_metadata["classifier_prior"] = round(
                    classification.prior_offset, 4
                )
                new_metadata["classifier_signals"] = (
                    classification.signals_fired
                )

                if classification.title_promoted:
                    new_metadata["classifier_title_promoted"] = True
                    new_metadata["classifier_title_rule"] = (
                        classification.title_rule
                    )
                    new_metadata["classifier_title_token"] = (
                        classification.title_token
                    )

                try:
                    updated_source = source.model_copy(
                        update={
                            "difficulty": classification.difficulty,
                            "metadata": new_metadata,
                        }
                    )
                except Exception:
                    try:
                        source.difficulty = classification.difficulty
                        source.metadata = new_metadata
                        updated_source = source
                    except Exception:
                        updated_source = source

                updated_sources.append(updated_source)
            except Exception as exc:
                self._logger.warning(
                    f"Difficulty classification failed: {exc}"
                )
                updated_sources.append(source)

        updated_sources = self._apply_distribution_guard(
            updated_sources,
            level,
        )

        return updated_sources

    def classify_source(
        self,
        source: Source,
        level: str = "",
    ) -> DifficultyClassification:
        treatment, treatment_signals, explicit_marker = self._score_treatment(
            source
        )
        topic, topic_signals = self._score_topic(source)
        metadata, metadata_signals = self._score_metadata(source)
        prior = self._score_prior(source)

        weights = self._config

        combined = (
            weights["treatment_weight"] * treatment
            + weights["topic_weight"] * topic
            + weights["metadata_weight"] * metadata
        )

        final_score = combined + prior
        final_score = max(0.0, min(1.0, final_score))

        final_score = self._apply_level_bias(final_score, level)

        difficulty = self._score_to_difficulty(final_score)

        difficulty = self._apply_sanity_check(
            difficulty=difficulty,
            treatment=treatment,
            topic=topic,
            level=level,
            source=source,
        )

        (
            difficulty,
            title_promoted,
            title_rule,
            title_token,
        ) = self._apply_title_promotion(source, difficulty)

        signals_fired = treatment_signals + topic_signals + metadata_signals

        confidence = self._compute_confidence(
            treatment=treatment,
            topic=topic,
            signals_fired=signals_fired,
            explicit_marker=explicit_marker,
            source=source,
        )

        if title_promoted:
            confidence = max(
                confidence,
                self._title_promotion_confidence_floor,
            )

        reason_parts = [
            f"treatment={treatment:.2f}",
            f"topic={topic:.2f}",
            f"metadata={metadata:.2f}",
            f"prior={prior:+.2f}",
            f"final={final_score:.2f}",
            f"confidence={confidence:.2f}",
        ]

        if title_promoted:
            reason_parts.append(
                f"title_promoted={title_rule}:{title_token}"
            )

        reason = " ".join(reason_parts)

        return DifficultyClassification(
            difficulty=difficulty,
            confidence=confidence,
            treatment_score=treatment,
            topic_score=topic,
            metadata_score=metadata,
            prior_offset=prior,
            signals_fired=signals_fired,
            reason=reason,
            title_promoted=title_promoted,
            title_rule=title_rule,
            title_token=title_token,
        )

    def _apply_title_promotion(
        self,
        source: Source,
        difficulty: Difficulty,
    ) -> tuple[Difficulty, bool, str, str]:
        if not self._title_promotion_enabled:
            return difficulty, False, "", ""

        title = clean_text(getattr(source, "title", "")).lower().strip()

        if not title:
            return difficulty, False, "", ""

        current_rank = _DIFFICULTY_RANK.get(difficulty, -1)

        matched_token = self._match_phrase(
            title,
            self._title_starts_advanced,
            starts_with=True,
        )

        if matched_token is not None:
            if current_rank < _DIFFICULTY_RANK[Difficulty.ADVANCED]:
                self._emit_title_promoted(
                    source,
                    difficulty,
                    Difficulty.ADVANCED,
                    "starts_advanced",
                    matched_token,
                )
                return (
                    Difficulty.ADVANCED,
                    True,
                    "starts_advanced",
                    matched_token,
                )
            return difficulty, False, "", ""

        advanced_qualifier = self._match_phrase(
            title, self._title_advanced_qualifiers
        )
        advanced_word = self._match_advanced_word(title)

        if advanced_qualifier is not None and advanced_word is not None:
            if current_rank < _DIFFICULTY_RANK[Difficulty.ADVANCED]:
                self._emit_title_promoted(
                    source,
                    difficulty,
                    Difficulty.ADVANCED,
                    "advanced_marker_with_qualifier",
                    f"{advanced_word}+{advanced_qualifier}",
                )
                return (
                    Difficulty.ADVANCED,
                    True,
                    "advanced_marker_with_qualifier",
                    f"{advanced_word}+{advanced_qualifier}",
                )
            return difficulty, False, "", ""

        beginner_phrase = self._match_phrase(
            title, self._title_beginner_phrases
        )

        if beginner_phrase is not None and advanced_word is None:
            if current_rank > _DIFFICULTY_RANK[Difficulty.BEGINNER]:
                self._emit_title_promoted(
                    source,
                    difficulty,
                    Difficulty.BEGINNER,
                    "beginner_phrase",
                    beginner_phrase,
                )
                return (
                    Difficulty.BEGINNER,
                    True,
                    "beginner_phrase",
                    beginner_phrase,
                )
            return difficulty, False, "", ""

        if advanced_word is not None:
            if current_rank < _DIFFICULTY_RANK[Difficulty.INTERMEDIATE]:
                self._emit_title_promoted(
                    source,
                    difficulty,
                    Difficulty.INTERMEDIATE,
                    "advanced_marker",
                    advanced_word,
                )
                return (
                    Difficulty.INTERMEDIATE,
                    True,
                    "advanced_marker",
                    advanced_word,
                )

        return difficulty, False, "", ""

    def _apply_distribution_guard(
        self,
        sources: list[Source],
        level: str,
    ) -> list[Source]:
        if not self._distribution_guard_enabled or not sources:
            return sources

        level_lower = str(level or "").strip().lower()

        if (
            self._distribution_guard_require_beginner_in_level
            and "beginner" not in level_lower
        ):
            return sources

        total = len(sources)

        if total == 0:
            return sources

        advanced_count = sum(
            1
            for s in sources
            if getattr(s, "difficulty", None) == Difficulty.ADVANCED
        )

        advanced_ratio = advanced_count / total

        if advanced_ratio <= self._distribution_guard_trigger_ratio:
            return sources

        demoted_to_intermediate = 0
        demoted_to_beginner = 0
        updated: list[Source] = []

        for source in sources:
            current = getattr(source, "difficulty", None)
            metadata = dict(getattr(source, "metadata", {}) or {})

            try:
                treatment = float(
                    metadata.get("classifier_treatment", 0.5) or 0.5
                )
            except Exception:
                treatment = 0.5

            try:
                topic = float(
                    metadata.get("classifier_topic", 0.5) or 0.5
                )
            except Exception:
                topic = 0.5

            new_difficulty = current

            if current == Difficulty.ADVANCED:
                if (
                    treatment
                    < self._distribution_guard_demote_advanced_max_treatment
                ):
                    new_difficulty = Difficulty.INTERMEDIATE
                    demoted_to_intermediate += 1
            elif current == Difficulty.INTERMEDIATE:
                if (
                    treatment
                    < self._distribution_guard_demote_intermediate_max_treatment
                    and topic
                    < self._distribution_guard_demote_intermediate_max_topic
                ):
                    new_difficulty = Difficulty.BEGINNER
                    demoted_to_beginner += 1

            if new_difficulty != current:
                metadata["classifier_distribution_demoted"] = True
                metadata["classifier_demoted_from"] = (
                    current.value
                    if isinstance(current, Difficulty)
                    else str(current)
                )
                try:
                    updated_source = source.model_copy(
                        update={
                            "difficulty": new_difficulty,
                            "metadata": metadata,
                        }
                    )
                except Exception:
                    updated_source = source
                updated.append(updated_source)
            else:
                updated.append(source)

        if demoted_to_intermediate or demoted_to_beginner:
            self._trace.emit(
                "classifier_distribution_rebalanced",
                advanced_ratio_before=round(advanced_ratio, 4),
                trigger_ratio=self._distribution_guard_trigger_ratio,
                demoted_to_intermediate=demoted_to_intermediate,
                demoted_to_beginner=demoted_to_beginner,
                total_sources=total,
            )

        return updated

    def _emit_title_promoted(
        self,
        source: Source,
        old_difficulty: Difficulty,
        new_difficulty: Difficulty,
        rule: str,
        token: str,
    ) -> None:
        try:
            self._trace.emit(
                "classifier_title_promoted",
                source_id=str(getattr(source, "source_id", "") or ""),
                title=clean_text(getattr(source, "title", ""))[:120],
                old_difficulty=(
                    old_difficulty.value
                    if isinstance(old_difficulty, Difficulty)
                    else str(old_difficulty)
                ),
                new_difficulty=(
                    new_difficulty.value
                    if isinstance(new_difficulty, Difficulty)
                    else str(new_difficulty)
                ),
                rule=rule,
                matched_token=token,
            )
        except Exception:
            pass

    def _match_advanced_word(self, title: str) -> str | None:
        for token in ("advanced", "advance", "advances", "advancement", "advancements"):
            pattern = r"\b" + re.escape(token) + r"\b"
            if re.search(pattern, title):
                return token
        return None

    def _match_phrase(
        self,
        title: str,
        tokens: tuple[str, ...],
        starts_with: bool = False,
    ) -> str | None:
        if not title or not tokens:
            return None

        for token in tokens:
            normalized = str(token or "").strip().lower()
            if not normalized:
                continue

            if starts_with:
                pattern = r"^" + re.escape(normalized) + r"\b"
            else:
                pattern = r"\b" + re.escape(normalized) + r"\b"

            if re.search(pattern, title):
                return normalized

        return None

    def _apply_sanity_check(
        self,
        difficulty: Difficulty,
        treatment: float,
        topic: float,
        level: str,
        source: Source,
    ) -> Difficulty:
        if difficulty != Difficulty.BEGINNER:
            return difficulty

        if treatment >= 0.55 or topic >= 0.60:
            return Difficulty.INTERMEDIATE

        if len(self._count_academic_patterns(source)) >= 2:
            return Difficulty.INTERMEDIATE

        level_text = clean_text(level).lower()

        if "mixed" in level_text or "advanced" in level_text:
            if treatment >= 0.50:
                return Difficulty.INTERMEDIATE

        return difficulty

    def _count_academic_patterns(self, source: Source) -> list[str]:
        text = self._source_text(source).lower()

        if not text:
            return []

        matched: list[str] = []

        for pattern in _ACADEMIC_PROSE_PATTERNS:
            found = pattern.search(text)

            if found:
                matched.append(found.group(0))

        return matched

    def _load_title_promotion_config(self) -> dict[str, Any]:
        resolved: dict[str, Any] = dict(_DEFAULT_TITLE_PROMOTION)

        section = self._extract_section("title_promotion")

        if not isinstance(section, dict):
            return resolved

        for key in resolved:
            if key in section and section[key] is not None:
                resolved[key] = section[key]

        return resolved

    def _load_distribution_guard_config(self) -> dict[str, Any]:
        resolved: dict[str, Any] = dict(_DEFAULT_DISTRIBUTION_GUARD)

        section = self._extract_section("distribution_guard")

        if not isinstance(section, dict):
            return resolved

        for key in resolved:
            if key in section and section[key] is not None:
                resolved[key] = section[key]

        return resolved

    def _extract_section(self, name: str) -> dict[str, Any] | None:
        config = self._ranking_config

        if config is None:
            return None

        try:
            data = getattr(config, "_data", None)
        except Exception:
            data = None

        if not isinstance(data, dict):
            return None

        classifier_section = data.get("classifier")
        if not isinstance(classifier_section, dict):
            return None

        section = classifier_section.get(name)
        if isinstance(section, dict):
            return section

        return None

    def _load_two_dimension_config(
        self,
        config: Any | None,
    ) -> dict[str, Any]:
        resolved = dict(_DEFAULT_CONFIG)

        source_configs: list[Any] = []

        if config is not None:
            source_configs.append(config)

        if self._ranking_config is not None:
            source_configs.append(self._ranking_config)

        for source_config in source_configs:
            section = self._extract_two_dimension_section(source_config)

            if not isinstance(section, dict):
                continue

            for key in resolved:
                if key in section and section[key] is not None:
                    resolved[key] = section[key]

        return resolved

    def _extract_two_dimension_section(
        self,
        config: Any,
    ) -> dict[str, Any] | None:
        if config is None:
            return None

        if isinstance(config, dict):
            candidates = (
                config.get("two_dimension_classifier"),
                config.get("classifier", {}).get("two_dimension")
                if isinstance(config.get("classifier"), dict)
                else None,
            )
        else:
            candidates = (
                getattr(config, "two_dimension_classifier", None),
                getattr(
                    getattr(config, "classifier", None),
                    "two_dimension",
                    None,
                )
                if getattr(config, "classifier", None) is not None
                else None,
            )

        for candidate in candidates:
            if isinstance(candidate, dict):
                return candidate

        return None

    def _safe_ranking_config(self) -> Any:
        try:
            return get_ranking_config()
        except Exception:
            return None

    def _score_treatment(
        self,
        source: Source,
    ) -> tuple[float, int, bool]:
        text = self._source_text(source)
        lowered = text.lower()

        if not lowered:
            return self._config["treatment_neutral"], 0, False

        score = self._config["treatment_neutral"]
        signals = 0
        explicit_marker = False

        beginner_hits = 0

        for marker in _BEGINNER_MARKERS:
            if marker in lowered:
                beginner_hits += 1

        beginner_offset = -0.05 * beginner_hits
        beginner_offset = max(-0.35, beginner_offset)

        if beginner_hits > 0:
            score += beginner_offset
            signals += 1
            explicit_marker = True

        advanced_hits = 0

        for marker in _ADVANCED_MARKERS:
            if marker in lowered:
                advanced_hits += 1

        advanced_offset = 0.05 * advanced_hits
        advanced_offset = min(0.35, advanced_offset)

        if advanced_hits > 0:
            score += advanced_offset
            signals += 1
            explicit_marker = True

        academic_hits = 0

        for pattern in _ACADEMIC_PROSE_PATTERNS:
            if pattern.search(lowered):
                academic_hits += 1

        if academic_hits > 0:
            score += min(0.20, 0.04 * academic_hits)
            signals += 1

        beginner_prose_hits = 0

        for pattern in _BEGINNER_PROSE_PATTERNS:
            if pattern.search(lowered):
                beginner_prose_hits += 1

        if beginner_prose_hits > 0:
            score -= min(0.15, 0.03 * beginner_prose_hits)
            signals += 1

        second_person_density = self._second_person_density(text)

        if second_person_density > 0.15:
            score -= 0.10
            signals += 1
        elif second_person_density > 0.05:
            score -= 0.05
            signals += 1
        elif second_person_density < 0.03 and len(text) > 200:
            score += 0.05
            signals += 1

        notation_density = self._formal_notation_density(text)

        if notation_density > 0.20:
            score += 0.20
            signals += 1
        elif notation_density > 0.05:
            score += 0.10
            signals += 1

        reference_density = self._reference_density(text)

        if reference_density > 0.10:
            score += 0.10
            signals += 1
        elif reference_density > 0.03:
            score += 0.05
            signals += 1

        mean_sentence_length = self._mean_sentence_length(text)

        if mean_sentence_length > 0:
            if mean_sentence_length >= 32:
                score += 0.15
                signals += 1
            elif mean_sentence_length >= 24:
                score += 0.10
                signals += 1
            elif mean_sentence_length <= 12:
                score -= 0.08
                signals += 1
            elif mean_sentence_length <= 16:
                score -= 0.03
                signals += 1

        score = max(0.0, min(1.0, score))

        return score, signals, explicit_marker

    def _score_topic(self, source: Source) -> tuple[float, int]:
        text = self._source_text(source)
        title = clean_text(getattr(source, "title", ""))

        if not text and not title:
            return self._config["topic_neutral"], 0

        score = self._config["topic_neutral"]
        signals = 0

        technical_density = self._technical_term_density(text)

        if technical_density > 0.15:
            score += 0.25
            signals += 1
        elif technical_density > 0.08:
            score += 0.15
            signals += 1
        elif technical_density < 0.02:
            score -= 0.05
            signals += 1

        compound_density = self._compound_noun_density(text)

        if compound_density > 5.0:
            score += 0.25
            signals += 1
        elif compound_density > 3.5:
            score += 0.15
            signals += 1
        elif 0 < compound_density < 2.0:
            score -= 0.05
            signals += 1

        named_density = self._named_entity_density(title)

        if named_density > 2:
            score += 0.10
            signals += 1
        elif named_density == 0 and len(title) > 20:
            score -= 0.05
            signals += 1

        title_notation_density = self._formal_notation_density(title)

        if title_notation_density > 0.05:
            score += 0.10
            signals += 1

        score = max(0.0, min(1.0, score))

        return score, signals

    def _score_metadata(self, source: Source) -> tuple[float, int]:
        metadata = getattr(source, "metadata", None)

        if not isinstance(metadata, dict):
            metadata = {}

        score = self._config["metadata_neutral"]
        signals = 0

        publisher_parts: list[str] = []

        for key in (
            "publisher",
            "journal",
            "venue",
            "container_title",
            "publication",
            "source",
        ):
            value = metadata.get(key)

            if isinstance(value, str) and value.strip():
                publisher_parts.append(value.strip().lower())

        if publisher_parts:
            combined = " ".join(publisher_parts)

            if any(
                fragment in combined
                for fragment in _ACADEMIC_PUBLISHER_MARKERS
            ):
                score += 0.20
                signals += 1

            if any(
                fragment in combined
                for fragment in _SELF_PUBLISHED_MARKERS
            ):
                score -= 0.20
                signals += 1

        citation_count = getattr(source, "citation_count", None)

        if isinstance(citation_count, int):
            if citation_count >= 500:
                score += 0.20
                signals += 1
            elif citation_count >= 100:
                score += 0.10
                signals += 1
            elif citation_count >= 10:
                score += 0.05
                signals += 1

        doi = metadata.get("doi")

        if isinstance(doi, str) and doi.strip():
            score += 0.05
            signals += 1

        score = max(0.0, min(1.0, score))

        return score, signals

    def _score_prior(self, source: Source) -> float:
        platform = self._enum_value(
            getattr(source, "platform", "")
        ).lower()
        source_type = self._enum_value(
            getattr(source, "source_type", "")
        ).lower()

        platform_offset = _PLATFORM_PRIOR.get(platform, 0.0)

        source_type_offset = 0.0

        if source_type in {"research_paper", "proceedings_article"}:
            source_type_offset = 0.05
        elif source_type in {
            "book",
            "textbook",
            "documentation",
            "course",
        }:
            source_type_offset = -0.03
        elif source_type == "video":
            source_type_offset = -0.05
        elif source_type == "repository":
            source_type_offset = 0.00
        elif source_type in {"model", "dataset"}:
            source_type_offset = 0.03

        total = platform_offset + source_type_offset

        cap = float(self._config.get("prior_cap", 0.15))
        total = max(-cap, min(cap, total))

        return total

    def _apply_level_bias(self, score: float, level: str) -> float:
        text = clean_text(level).lower()

        if not text:
            return score

        has_beginner = "beginner" in text
        has_advanced = "advanced" in text or "advance" in text

        if has_beginner and not has_advanced:
            return max(0.0, score - 0.05)

        if has_advanced and not has_beginner:
            return min(1.0, score + 0.05)

        return score

    def _score_to_difficulty(self, score: float) -> Difficulty:
        beginner_threshold = float(
            self._config.get("beginner_threshold", 0.35)
        )
        advanced_threshold = float(
            self._config.get("advanced_threshold", 0.65)
        )

        if score < beginner_threshold:
            return Difficulty.BEGINNER

        if score < advanced_threshold:
            return Difficulty.INTERMEDIATE

        return Difficulty.ADVANCED

    def _compute_confidence(
        self,
        treatment: float,
        topic: float,
        signals_fired: int,
        explicit_marker: bool,
        source: Source,
    ) -> float:
        config = self._config

        base = float(config.get("confidence_base", 0.30))
        signal_weight = float(
            config.get("confidence_signal_weight", 0.05)
        )
        signal_cap = int(config.get("confidence_signal_cap", 6))
        text_weight = float(
            config.get("confidence_text_weight", 0.10)
        )
        text_full = float(
            config.get("confidence_text_full_length", 500)
        )
        agreement_weight = float(
            config.get("confidence_agreement_weight", 0.20)
        )
        explicit_boost = float(
            config.get("confidence_explicit_boost", 0.15)
        )
        floor = float(config.get("confidence_min", 0.10))
        ceiling = float(config.get("confidence_max", 0.95))

        signal_component = signal_weight * min(signals_fired, signal_cap)

        text = self._source_text(source)
        text_length = len(text)

        if text_full > 0:
            text_component = text_weight * min(
                1.0, text_length / text_full
            )
        else:
            text_component = 0.0

        agreement = 1.0 - abs(treatment - topic)
        agreement_component = agreement_weight * agreement

        marker_component = explicit_boost if explicit_marker else 0.0

        no_text_penalty = -0.15 if text_length < 80 else 0.0

        confidence = (
            base
            + signal_component
            + text_component
            + agreement_component
            + marker_component
            + no_text_penalty
        )

        return max(floor, min(ceiling, confidence))

    def _source_text(self, source: Source) -> str:
        parts = [
            clean_text(getattr(source, "title", "")),
            clean_text(getattr(source, "abstract", "")),
            clean_text(getattr(source, "summary", "")),
        ]

        metadata = getattr(source, "metadata", {}) or {}

        if isinstance(metadata, dict):
            for key in (
                "description",
                "summary",
                "topics",
                "tags",
                "subjects",
                "categories",
                "pipeline_tag",
            ):
                value = metadata.get(key)

                if isinstance(value, str):
                    parts.append(clean_text(value))
                elif isinstance(value, (list, tuple, set)):
                    for item in value:
                        parts.append(clean_text(item))

        return " ".join(part for part in parts if part).strip()

    def _tokens(self, text: str) -> list[str]:
        try:
            raw_tokens = tokenize(text)
        except Exception:
            raw_tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())

        return [str(token).lower() for token in raw_tokens if token]

    def _second_person_density(self, text: str) -> float:
        if not text:
            return 0.0

        lowered = text.lower()
        total_chars = max(1, len(lowered))

        count = 0
        for pronoun in (
            " you ",
            " your ",
            " yours ",
            "you're",
            "you'll",
            "you've",
        ):
            count += lowered.count(pronoun)

        if lowered.startswith("you "):
            count += 1

        density = count / (total_chars / 1000.0)

        return max(0.0, density)

    def _formal_notation_density(self, text: str) -> float:
        if not text:
            return 0.0

        total_chars = max(1, len(text))

        hits = 0
        hits += len(_FORMAL_NOTATION_PATTERN.findall(text))
        hits += len(_LATEX_PATTERN.findall(text))
        hits += len(_SUBSCRIPT_PATTERN.findall(text))

        density = hits / (total_chars / 1000.0)

        return max(0.0, density)

    def _reference_density(self, text: str) -> float:
        if not text:
            return 0.0

        total_chars = max(1, len(text))

        hits = 0

        for pattern in _REFERENCE_PATTERNS:
            hits += len(pattern.findall(text))

        density = hits / (total_chars / 1000.0)

        return max(0.0, density)

    def _mean_sentence_length(self, text: str) -> float:
        if not text:
            return 0.0

        sentences = [
            s.strip()
            for s in _SENTENCE_SPLIT.split(text)
            if s.strip()
        ]

        if not sentences:
            return 0.0

        total_words = sum(
            len(_WORD_SPLIT.split(s)) for s in sentences
        )

        if total_words == 0:
            return 0.0

        return total_words / len(sentences)

    def _technical_term_density(self, text: str) -> float:
        if not text:
            return 0.0

        tokens = self._tokens(text)

        if not tokens:
            return 0.0

        generics = self._generic_terms
        technical = 0

        for token in tokens:
            if len(token) < 8:
                continue

            if token in generics:
                continue

            if not all(ch.isalpha() for ch in token):
                continue

            is_technical = False

            for prefix in _TECHNICAL_PREFIXES:
                if token.startswith(prefix):
                    is_technical = True
                    break

            if not is_technical:
                for suffix in _TECHNICAL_SUFFIXES:
                    if token.endswith(suffix):
                        is_technical = True
                        break

            if is_technical:
                technical += 1

        return technical / len(tokens)

    def _compound_noun_density(self, text: str) -> float:
        if not text:
            return 0.0

        sentences = [
            s.strip()
            for s in _SENTENCE_SPLIT.split(text)
            if s.strip()
        ]

        if not sentences:
            return 0.0

        generics = self._generic_terms
        total_runs: list[int] = []

        for sentence in sentences:
            words = [
                w.lower() for w in _ALPHA_TOKEN.findall(sentence)
            ]

            if not words:
                continue

            current_run = 0

            for word in words:
                is_content = (
                    len(word) >= 3
                    and word not in generics
                )

                if is_content:
                    current_run += 1
                else:
                    if current_run > 0:
                        total_runs.append(current_run)
                    current_run = 0

            if current_run > 0:
                total_runs.append(current_run)

        if not total_runs:
            return 0.0

        return sum(total_runs) / len(total_runs)

    def _named_entity_density(self, title: str) -> int:
        if not title:
            return 0

        matches = _NAMED_ENTITY_PATTERN.findall(title)

        if matches:
            return len(matches)

        proper_count = 0
        words = title.split()

        for index, word in enumerate(words):
            if index == 0:
                continue

            if _PROPER_NOUN_TITLE_PATTERN.match(word):
                proper_count += 1

        return proper_count

    def _enum_value(self, value: Any) -> str:
        return str(getattr(value, "value", value) or "")


def is_advanced_with_confidence(
    source: Source,
    min_confidence: float = 0.50,
) -> bool:
    difficulty = getattr(source, "difficulty", None)

    if difficulty is None:
        return False

    is_advanced = False

    if isinstance(difficulty, Difficulty):
        is_advanced = difficulty == Difficulty.ADVANCED
    else:
        is_advanced = str(difficulty).strip().lower() == "advanced"

    if not is_advanced:
        return False

    metadata = getattr(source, "metadata", None)

    if not isinstance(metadata, dict):
        return True

    confidence = metadata.get("classifier_confidence")

    if confidence is None:
        return True

    try:
        value = float(confidence)
    except Exception:
        return True

    return value >= min_confidence