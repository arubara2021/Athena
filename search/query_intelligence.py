from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field

from core import constants
from core.models import CoreModel
from search.query_tokens import get_generic_terms, strip_filler, tokenize
from utils.logger import get_logger
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
    intent_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    spell_corrections: dict[str, str] = Field(default_factory=dict)
    short_search_queries: list[str] = Field(default_factory=list)
    suggested_queries: list[str] = Field(default_factory=list)
    llm_used: bool = False


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
    }

    def __init__(self, use_llm: bool = True) -> None:
        self._use_llm = use_llm
        self._config = self._load_config()
        self._logger = get_logger("search.query_intelligence")

    async def analyze(
        self,
        topic: str,
        goal: str = "",
        level: str = "",
    ) -> QueryAnalysis:
        original = clean_text(topic)

        analysis = self._local_analysis(original, goal, level)

        if not original:
            return analysis

        if not self._use_llm:
            return analysis

        enhanced = await self._llm_enhance(analysis, goal, level)

        if enhanced is not None:
            return enhanced

        return analysis

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
                            config[key] = self._as_int(section[key], config[key])
                        elif isinstance(config[key], float):
                            config[key] = self._as_float(section[key], config[key])
                        else:
                            config[key] = section[key]
        except Exception:
            pass

        return config

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

        detected_level = self._infer_level(goal, level)
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
            intent=QueryIntent.EXPLORE,
            level=detected_level,
            intent_confidence=0.0,
            spell_corrections={},
            short_search_queries=short_search_queries,
            suggested_queries=short_search_queries,
            llm_used=False,
        )

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

        return " ".join(words[: int(self._config["max_primary_concept_words"])])

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

    def _infer_level(self, goal: str, level: str) -> str:
        explicit = clean_text(level).lower()
        if explicit in {"beginner", "intermediate", "advanced"}:
            return explicit
        if explicit == "mixed":
            return "mixed"
        text = f"{clean_text(goal).lower()} {explicit}"
        has_beginner = "beginner" in text or "basics" in text
        has_advanced = "advanced" in text
        if has_beginner and has_advanced:
            return "mixed"
        if has_beginner:
            return "beginner"
        if has_advanced:
            return "advanced"
        if "intermediate" in text:
            return "intermediate"
        return "mixed"

    async def _llm_enhance(
        self,
        analysis: QueryAnalysis,
        goal: str,
        level: str,
    ) -> QueryAnalysis | None:
        try:
            from llm.guardrails import validate_llm_request
            from llm.parser import parse_json_object_response
            from llm.provider import LLMProviderManager
            from llm.router import get_fast_model_references
            from core.schemas import LLMMessageSchema, LLMRequestSchema

            model_references = get_fast_model_references(
                limit=int(self._config["model_limit"])
            )

            if not model_references:
                return None

            expected_output = {
                "corrected_topic": "short keywords",
                "primary_concept": "short primary concept",
                "keywords": ["keyword one", "keyword two"],
                "intent": "learn | research | implement | compare | troubleshoot | explore",
                "level": "beginner | intermediate | advanced | mixed",
                "intent_confidence": 0.0,
                "spell_corrections": {"original_word": "corrected_word"},
                "short_search_queries": ["short query one", "short query two"],
            }

            system_prompt = (
                "You are a query understanding engine. "
                "Convert any user request into short, clean, searchable concepts. "
                "Correct spelling mistakes. "
                "Do not invent facts. "
                "Return only valid JSON."
            )

            user_prompt = (
                f"Original topic: {analysis.original_topic}\n"
                f"Goal: {goal}\n"
                f"Level: {level}\n"
                "Rules:\n"
                f"- corrected_topic must be 2 to {self._config['max_corrected_topic_words']} short keywords.\n"
                f"- primary_concept must be 2 to {self._config['max_primary_concept_words']} short keywords.\n"
                f"- keywords must contain 2 to {self._config['max_keywords']} short search keywords.\n"
                f"- short_search_queries must contain 2 to {self._config['max_short_queries']} short queries.\n"
                f"- Every query must be 2 to {self._config['max_query_words']} words.\n"
                "- Correct spelling mistakes.\n"
                "- Preserve acronyms and technical terms.\n"
                "- Do not return full sentences.\n"
                "- Do not use phrases like I want to learn.\n"
                "- If level is unclear, use mixed.\n"
                f"Expected JSON output:\n{json.dumps(expected_output, ensure_ascii=False, indent=2)}\n"
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

                        payload = parse_json_object_response(response.content)
                        return self._apply_llm_payload(analysis, payload)
                    except Exception:
                        continue

            return None
        except Exception as exc:
            self._logger.warning(f"LLM query enhancement failed: {exc}")
            return None

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

        detected_level = self._sanitize_level(
            payload.get("level"),
            analysis.level,
        )

        intent = self._parse_intent(payload.get("intent"))
        intent_confidence = self._parse_confidence(payload.get("intent_confidence"))
        spell_corrections = self._parse_spell_corrections(
            payload.get("spell_corrections")
        )

        short_search_queries = self._sanitize_short_queries(
            payload.get("short_search_queries"),
            corrected_topic,
            primary_concept,
            keywords,
        )

        return QueryAnalysis(
            original_topic=analysis.original_topic,
            cleaned_topic=analysis.cleaned_topic,
            corrected_topic=corrected_topic,
            primary_concept=primary_concept,
            keywords=keywords,
            intent=intent,
            level=detected_level,
            intent_confidence=intent_confidence,
            spell_corrections=spell_corrections,
            short_search_queries=short_search_queries,
            suggested_queries=short_search_queries,
            llm_used=True,
        )

    def _sanitize_phrase(self, value: Any, fallback: str, max_words: int) -> str:
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
        has_advanced = "advanced" in text
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