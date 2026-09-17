from __future__ import annotations

import re
from typing import Any

from pydantic import Field

from core.models import CoreModel, Difficulty, Source, SourcePlatform, SourceType
from core.ranking_config import get_ranking_config
from utils.logger import get_logger
from utils.text import clean_text


class DifficultyClassification(CoreModel):
    difficulty: Difficulty = Difficulty.INTERMEDIATE
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str | None = None


class DifficultyClassifier:
    BEGINNER_MARKERS = {
        "introduction", "tutorial", "guide", "explained", "basics",
        "fundamentals", "step by step", "how to", "for beginners",
        "getting started", "overview", "primer", "learn", "understanding",
        "what is", "simple", "easy", "crash course", "101", "starter",
        "no prior", "from scratch", "beginner friendly", "quickstart",
        "walkthrough", "handbook", "explained simply", "made simple",
        "demystified", "plain language", "introductory",
        "beginner friendly",
"step by step",
"from scratch",
"no prior experience",
"made simple",
"explained simply",
"introductory",
"starter",
"quickstart",
"handbook",
"101",
"friendly guide",
    }

    ACADEMIC_MARKERS = {
        "we propose", "we introduce", "we present", "we demonstrate",
        "we show that", "we prove", "we derive", "we establish",
        "we evaluate", "we compare", "we train", "we develop",
        "we design", "we formulate", "we analyze", "we investigate",
        "our approach", "our method", "our framework", "our model",
        "our results", "our contribution", "our work",
        "in this paper", "in this work", "in this study",
        "novel", "state of the art", "state-of-the-art",
        "theorem", "lemma", "corollary", "proposition",
        "we conjecture", "we hypothesize", "we argue",
        "formally", "rigorous", "we extend", "we generalize",
        "we optimize", "we benchmark", "we ablate",
        "proposed method", "proposed approach", "proposed framework",
        "experimental results", "empirical evaluation",
        "convergence guarantee", "theoretical analysis",
        "we outperform", "we achieve", "we observe",
        "we conduct", "we perform", "we implement",
    }

    INTERMEDIATE_MARKERS = {
        "implementation", "practical", "applied", "hands-on",
        "example", "examples", "case study", "walkthrough",
        "project", "build", "building", "developing",
        "techniques", "methods", "patterns", "best practices",
        "comparison", "evaluation", "study", "analysis",
        "framework", "architecture", "design", "approach",
    }

    INTRODUCTORY_PHRASES = (
        "what is", "what are", "introduction to", "intro to",
        "a guide", "getting started", "basics", "fundamentals",
        "explained", "made simple", "for beginners", "easy",
        "tutorial", "overview of", "understanding", "learn",
        "how to", "primer", "101", "crash course", "friendly",
        "demystified", "no prior", "from scratch", "simple",
        "step by step", "beginner",
        "no prior experience",
"made simple",
"explained simply",
"friendly guide",
"starter guide",
"step by step",
    )

    def __init__(self) -> None:
        self._logger = get_logger("ranking.difficulty_classifier")
        self._config = get_ranking_config()

    def classify_source(self, source: Any) -> DifficultyClassification:
        try:
            if not isinstance(source, Source):
                return DifficultyClassification(
                    difficulty=Difficulty.INTERMEDIATE,
                    confidence=0.2,
                    reason="unsupported source type",
                )

            cw = self._config.classifier_weights()

            title_text = clean_text(source.title).lower()
            abstract_text = clean_text(source.abstract or "").lower()
            full_text = f"{title_text} {abstract_text}".strip()
            source_type = self._enum_value(source.source_type)
            platform = self._enum_value(source.platform)

            beginner_score = 0.0
            intermediate_score = 0.0
            advanced_score = 0.0

            type_bonus = cw.source_type_bonus.get(source_type, {})
            beginner_score += type_bonus.get("beginner", 0.0)
            intermediate_score += type_bonus.get("intermediate", 0.0)

            plat_bonus = cw.platform_bonus.get(platform, {})
            beginner_score += plat_bonus.get("beginner", 0.0)
            intermediate_score += plat_bonus.get("intermediate", 0.0)

            beginner_score += self._marker_score(title_text, self.BEGINNER_MARKERS) * cw.keyword_beginner
            intermediate_score += self._marker_score(title_text, self.INTERMEDIATE_MARKERS) * cw.keyword_intermediate
            advanced_score += self._marker_score(full_text, self.ACADEMIC_MARKERS) * cw.keyword_advanced

            intro_points = self._introductory_phrasing(title_text, abstract_text, cw.intro_max_points)
            beginner_score += intro_points * cw.intro_beginner_weight
            intermediate_score += intro_points * cw.intro_intermediate_weight

            complexity = self._content_complexity(full_text, cw)
            content_weight = (
                cw.content_weight_paper
                if source_type == SourceType.PAPER.value
                else cw.content_weight_other
            )

            if complexity <= cw.beginner_max_complexity:
                beginner_score += content_weight
            elif complexity <= cw.intermediate_max_complexity:
                intermediate_score += content_weight
            else:
                advanced_score += content_weight

            scores = {
                Difficulty.BEGINNER: beginner_score,
                Difficulty.INTERMEDIATE: intermediate_score,
                Difficulty.ADVANCED: advanced_score,
            }
            total_score = sum(scores.values())

            if total_score <= 0:
                return DifficultyClassification(
                    difficulty=Difficulty.INTERMEDIATE,
                    confidence=0.35,
                    reason="no strong difficulty signals",
                )

            difficulty = max(scores, key=lambda item: scores[item])

            ordered = sorted(scores.values(), reverse=True)
            top = ordered[0]
            second = ordered[1] if len(ordered) > 1 else 0.0
            margin = top - second
            confidence = min(
                cw.confidence_max,
                cw.confidence_base + (margin / (top + 1e-6)) * cw.confidence_margin_factor,
            )

            reason = self._reason(scores, difficulty, complexity)
            return DifficultyClassification(
                difficulty=difficulty,
                confidence=round(confidence, 3),
                reason=reason,
            )
        except Exception as exc:
            self._logger.warning(f"Difficulty classification failed: {exc}")
            return DifficultyClassification(
                difficulty=Difficulty.INTERMEDIATE,
                confidence=0.2,
                reason="classification error",
            )

    def classify_sources(self, sources: list[Any]) -> list[Source]:
        updated_sources: list[Source] = []
        for source in sources:
            if not isinstance(source, Source):
                continue
            try:
                if source.difficulty is not None:
                    updated_sources.append(source)
                    continue
                classification = self.classify_source(source)
                updated_sources.append(
                    source.model_copy(update={"difficulty": classification.difficulty})
                )
            except Exception:
                updated_sources.append(source)
        return updated_sources

    def _introductory_phrasing(
        self,
        title: str,
        abstract: str,
        max_points: int,
    ) -> float:
        combined = f"{title} {abstract}"
        hits = sum(1 for phrase in self.INTRODUCTORY_PHRASES if phrase in combined)
        return float(min(hits, max_points))

    def _content_complexity(self, text: str, cw: Any) -> float:
        if not text:
            return 0.5
        fog = self._gunning_fog_normalized(text, cw)
        academic = self._academic_marker_density(text, cw)
        combined = fog * cw.fog_weight + academic * cw.jargon_weight
        return max(0.0, min(1.0, combined))

    def _gunning_fog_normalized(self, text: str, cw: Any) -> float:
        sentences = [
            sentence
            for sentence in re.split(r"[.!?]+", text)
            if len(sentence.split()) >= 3
        ]
        if not sentences:
            return 0.5

        all_words: list[str] = []
        for sentence in sentences:
            all_words.extend(re.findall(r"[a-zA-Z']+", sentence))

        if not all_words:
            return 0.5

        total_words = len(all_words)
        complex_words = sum(
            1 for word in all_words
            if self._count_syllables(word) >= cw.complex_word_syllables
        )

        words_per_sentence = total_words / len(sentences)
        complex_ratio = complex_words / total_words
        fog_index = 0.4 * (words_per_sentence + 100 * complex_ratio)

        fog_range = cw.fog_max - cw.fog_min
        if fog_range <= 0:
            return 0.5
        normalized = (fog_index - cw.fog_min) / fog_range
        return max(0.0, min(1.0, normalized))

    def _academic_marker_density(self, text: str, cw: Any) -> float:
        words = re.findall(r"[a-z]+", text.lower())
        if not words:
            return 0.0
        hits = sum(1 for word in words if word in self.ACADEMIC_MARKERS)
        phrase_hits = sum(1 for phrase in self.ACADEMIC_MARKERS if " " in phrase and phrase in text)
        total_hits = hits + phrase_hits * 3
        cap = cw.jargon_density_cap
        if cap <= 0:
            return 0.0
        return min(1.0, (total_hits / max(1, len(words))) / cap)

    def _count_syllables(self, word: str) -> int:
        cleaned = re.sub(r"[^a-z]", "", word.lower().strip())
        if not cleaned:
            return 0
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        for char in cleaned:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if cleaned.endswith("e") and count > 1:
            count -= 1
        return max(1, count)

    def _marker_score(self, text: str, markers: set[str]) -> float:
        score = 0.0
        for marker in markers:
            if " " in marker:
                if marker in text:
                    score += 1.0
            else:
                if re.search(rf"\b{re.escape(marker)}\b", text):
                    score += 1.0
        return score

    def _reason(
        self,
        scores: dict[Difficulty, float],
        difficulty: Difficulty,
        complexity: float,
    ) -> str:
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_signals = [f"{level.value}:{value:.1f}" for level, value in ordered[:2]]
        return (
            f"selected {difficulty.value}, complexity {complexity:.2f}, "
            f"signals {', '.join(top_signals)}"
        )

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value))