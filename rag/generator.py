from __future__ import annotations

import asyncio
import re
from typing import Any, AsyncIterator

from core.schemas import LLMMessageSchema, LLMRequestSchema
from llm.guardrails import validate_llm_request, validate_llm_response
from llm.provider import LLMProviderManager
from llm.router import (
    get_fast_model_references,
    get_strong_model_references,
)
from rag.schemas import (
    FallbackStepSchema,
    TopicProfileSchema,
)
from utils.logger import get_logger
from utils.text import clean_text, truncate_text

_CITATION_PATTERN = re.compile(r"\[(\d{1,3})\]")
_MAX_PASSAGE_CHARS = 900
_DEFAULT_MAX_OUTPUT_TOKENS = 1600
_RACE_TIMEOUT = 90.0

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

_ANSWER_STYLE_PRIORITY = {
    _ANSWER_STYLE_STANDARD: 0,
    _ANSWER_STYLE_SHORT: 1,
    _ANSWER_STYLE_DETAILED: 2,
    _ANSWER_STYLE_EXHAUSTIVE: 3,
}

_SHORT_MARKERS = (
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

_DETAILED_MARKERS = (
    "very detailed",
    "in detail",
    "in depth",
    "in-depth",
    "explain fully",
    "explain in detail",
    "explain thoroughly",
    "step by step",
    "comprehensive explanation",
    "comprehensive answer",
    "walk me through",
    "walk through",
    "elaborate",
    "elaborate on",
    "detailed explanation",
    "detailed answer",
    "detailed about",
    "detail about",
    "detailed overview",
    "long answer",
    "long-form",
    "thorough explanation",
    "thorough answer",
    "thoroughly explain",
    "full explanation",
    "full answer",
    "deep dive",
    "deep-dive",
)

_EXHAUSTIVE_MARKERS = (
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

_SHORT_MAX_TOKENS = 500
_DETAILED_MAX_TOKENS = 2800
_EXHAUSTIVE_MAX_TOKENS = 3200

_SHORT_TEMPERATURE = 0.05
_DETAILED_TEMPERATURE = 0.10
_EXHAUSTIVE_TEMPERATURE = 0.08

_CATALOG_MAX_OUTPUT_TOKENS = 1200
_CATALOG_TEMPERATURE = 0.10
_CATALOG_MODEL_LIMIT = 2
_CATALOG_MAX_PLATFORM_SLICES = 12
_CATALOG_MAX_SOURCE_TYPE_SLICES = 12
_CATALOG_MAX_DOMAIN_SLICES = 12
_CATALOG_MAX_TOPIC_SLICES = 15
_CATALOG_MAX_YEAR_SLICES = 10
_CATALOG_MAX_TITLE_SLICES = 20

_INTENT_DETAILED_EXPLANATION = "detailed_explanation"

_DEEP_INTENTS = {
    "comparison",
    "mechanism",
    "timeline",
    "explanation",
    _INTENT_DETAILED_EXPLANATION,
}

_SHALLOW_INTENTS = {
    "definition",
    "overview",
    "enumeration",
    "causation",
    "decision",
}

_SYSTEM_BASE = (
    "You are a research assistant. "
    "Answer the user's question using ONLY the numbered passages provided. "
    "Every factual claim must end with a citation marker like [1] or [2]. "
    "If the passages do not contain enough information to answer, say so "
    "explicitly and do not invent content. "
    "Do not invent sources, links, statistics, names, or dates. "
    "Answer in the same language as the user's question. "
    "Do not use markdown headers, code fences, or JSON. "
    "Do not mention concepts, subtopics, related terms, related concepts, "
    "neighbouring fields, or any topic the user did not explicitly ask "
    "about. "
    "Do not append any closing line or section that was not requested, "
    "including but not limited to: Related terms, Related concepts, "
    "Further reading, See also, Suggested next steps, Additional "
    "resources, References, Sources, or Further exploration. "
    "Do not close with a summary sentence unless a summary was explicitly "
    "requested. "
    "Answer only what was asked, at the level of detail requested. "
    "The structure sections below describe the maximum scope of your "
    "answer, not a target to fill. If the passages do not support a "
    "section, omit it. If the passages only support a short answer, give "
    "a short answer even when a longer structure is described."
)

_SYSTEM_SHORT_SUFFIX = (
    "Answer in one short paragraph. "
    "No bullet lists. "
    "No headers. "
    "No related terms. "
    "No closing summary line. "
    "Maximum three sentences."
)

_SYSTEM_DETAILED_SECTIONS = (
    "Structure the answer with these sections in order, using plain prose "
    "and short bullet lists where they help: "
    "(1) Overview: one short paragraph that frames the topic. "
    "(2) Key aspects: a short list of the main points present in the passages. "
    "(3) How it works: explain the mechanism or reasoning if the passages "
    "support it. "
    "(4) Examples: give concrete examples only if the passages contain them. "
    "(5) Caveats and limitations: only those actually stated in the passages. "
    "Every sentence with a claim must carry a citation marker. "
    "Do not add sections that the passages do not support. "
    "Do not add a Related terms line, a Sources line, a Further reading "
    "line, or any closing summary unless the user explicitly asked for it. "
    "Do not pad the answer with filler."
)

_SYSTEM_EXHAUSTIVE_SUFFIX = (
    "Cover every distinct point present in the passages. "
    "Do not omit relevant detail. "
    "Do not introduce information that is not in the passages. "
    "If two passages disagree, state the disagreement and cite both. "
    "Do not add closing lines for related terms, sources, further "
    "reading, or next steps."
)

_CATALOG_SYSTEM_PROMPT = (
    "You describe a local document corpus to the user. "
    "You receive structured catalog data: counts and breakdowns by "
    "platform, source type, domain, topic, year, and run, plus a sample "
    "of recent titles. "
    "Answer the user's question using ONLY the catalog data shown. "
    "Do not retrieve passages. Do not invent sources. Do not speculate "
    "about documents you have not seen. "
    "When the answer draws on a specific dimension of the catalog, name "
    "that dimension explicitly (platform, source type, domain, topic, "
    "year, run, or title). "
    "If the catalog does not contain the dimension the user is asking "
    "about, say so plainly and describe what the catalog does contain. "
    "Be concise. Use plain prose. Do not use markdown headers or code "
    "fences. Do not close with a summary line unless the user asked for "
    "one."
)

_INTENT_STRUCTURE: dict[str, str] = {
    "definition": (
        "Structure the answer as: (1) a short definition paragraph, "
        "(2) a bulleted list of key aspects. "
        "Stop after the key aspects list."
    ),
    "comparison": (
        "Structure the answer as: (1) a short comparison summary, "
        "(2) a set of side-by-side points showing differences, "
        "(3) one line stating when each side is preferable."
    ),
    "mechanism": (
        "Structure the answer as: (1) prerequisites, (2) a stepwise walkthrough "
        "of how it works, (3) common pitfalls."
    ),
    "causation": (
        "Structure the answer as: (1) a ranked list of causes with a short "
        "explanation for each, (2) counter-causes if any are mentioned."
    ),
    "enumeration": (
        "Structure the answer as: a grouped list of items with a one-line "
        "explanation for each."
    ),
    "timeline": (
        "Structure the answer as: a chronological list of milestones with "
        "dates only when the passages state them."
    ),
    "decision": (
        "Structure the answer as: (1) pros, (2) cons, (3) trade-offs, "
        "(4) a final recommendation grounded in the passages."
    ),
    "explanation": (
        "Structure the answer as: (1) an introductory paragraph, "
        "(2) supporting detail, (3) one or two examples."
    ),
    "overview": (
        "Structure the answer as: (1) a short overview paragraph, "
        "(2) key points as bullets. "
        "Stop after the key points."
    ),
    _INTENT_DETAILED_EXPLANATION: _SYSTEM_DETAILED_SECTIONS,
}

_INTENT_TEMPERATURE: dict[str, float] = {
    "definition": 0.05,
    "comparison": 0.10,
    "mechanism": 0.10,
    "causation": 0.10,
    "enumeration": 0.15,
    "timeline": 0.05,
    "decision": 0.10,
    "explanation": 0.15,
    "overview": 0.15,
    _INTENT_DETAILED_EXPLANATION: 0.10,
}

_INTENT_MAX_TOKENS: dict[str, int] = {
    "definition": 900,
    "comparison": 1400,
    "mechanism": 1200,
    "causation": 1000,
    "enumeration": 900,
    "timeline": 1000,
    "decision": 1200,
    "explanation": 1300,
    "overview": 1000,
    _INTENT_DETAILED_EXPLANATION: _DETAILED_MAX_TOKENS,
}

_CATALOG_DIMENSIONS: tuple[str, ...] = (
    "platform",
    "source_type",
    "domain",
    "topic",
    "year",
    "run",
    "title",
)

_CATALOG_DIMENSION_MARKERS: dict[str, tuple[str, ...]] = {
    "platform": ("platform", "platforms", "site", "sites"),
    "source_type": (
        "source type",
        "source types",
        "type of source",
        "types of sources",
        "format",
        "formats",
    ),
    "domain": (
        "domain",
        "domains",
        "field",
        "fields",
        "subject area",
        "subject areas",
    ),
    "topic": ("topic", "topics", "subject", "subjects"),
    "year": ("year", "years", "date", "dates", "publication"),
    "run": ("run", "runs", "session", "sessions", "indexing run"),
    "title": ("title", "titles"),
}

_CATALOG_CONFIDENCE_WEIGHTS: dict[str, float] = {
    "llm_used": 0.30,
    "classifier_confidence": 0.25,
    "snapshot_richness": 0.15,
    "answer_substance": 0.20,
    "dimension_coverage": 0.10,
}


class GeneratedAnswer:
    def __init__(
        self,
        text: str = "",
        cited_ids: list[str] | None = None,
        confidence: float = 0.0,
        llm_used: bool = False,
        fallback_ladder: list[FallbackStepSchema] | None = None,
        error: str | None = None,
        answer_style: str = _ANSWER_STYLE_STANDARD,
        confidence_breakdown: dict[str, float] | None = None,
        cited_dimensions: list[str] | None = None,
    ) -> None:
        self.text = text
        self.cited_ids = list(cited_ids or [])
        self.confidence = float(confidence or 0.0)
        self.llm_used = bool(llm_used)
        self.fallback_ladder = list(fallback_ladder or [])
        self.error = error
        self.answer_style = answer_style
        self.confidence_breakdown = dict(confidence_breakdown or {})
        self.cited_dimensions = list(cited_dimensions or [])


class Generator:
    def __init__(
        self,
        max_output_tokens: int = _DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = 0.10,
        model_limit: int = 2,
        race_timeout: float = _RACE_TIMEOUT,
        catalog_max_output_tokens: int = _CATALOG_MAX_OUTPUT_TOKENS,
        catalog_temperature: float = _CATALOG_TEMPERATURE,
        catalog_model_limit: int = _CATALOG_MODEL_LIMIT,
    ) -> None:
        self._max_output_tokens = max(256, int(max_output_tokens))
        self._temperature = max(0.0, min(1.0, float(temperature)))
        self._model_limit = max(1, int(model_limit))
        self._race_timeout = max(10.0, float(race_timeout))
        self._catalog_max_output_tokens = max(256, int(catalog_max_output_tokens))
        self._catalog_temperature = max(0.0, min(1.0, float(catalog_temperature)))
        self._catalog_model_limit = max(1, int(catalog_model_limit))
        self._logger = get_logger("rag.generator")

    async def generate(
        self,
        question: str,
        topic: TopicProfileSchema | None,
        passages: list[dict[str, Any]],
    ) -> GeneratedAnswer:
        cleaned_question = clean_text(question)

        if not cleaned_question:
            return GeneratedAnswer(
                text="",
                confidence=0.0,
                llm_used=False,
                answer_style=_ANSWER_STYLE_STANDARD,
            )

        if not passages:
            return GeneratedAnswer(
                text=(
                    "The local index does not contain passages for this "
                    "question."
                ),
                confidence=0.0,
                llm_used=False,
                answer_style=_ANSWER_STYLE_STANDARD,
            )

        answer_style = self._resolve_answer_style(cleaned_question, topic)
        intent = self._resolve_intent(topic, answer_style)

        system_prompt = self._build_system_prompt(intent, answer_style)
        user_prompt = self._build_user_prompt(
            cleaned_question,
            topic,
            passages,
            answer_style,
        )

        messages = [
            LLMMessageSchema(role="system", content=system_prompt),
            LLMMessageSchema(role="user", content=user_prompt),
        ]

        references = self._select_models(intent, answer_style)
        ladder: list[FallbackStepSchema] = []

        if not references:
            ladder.append(
                FallbackStepSchema(
                    stage="llm_selection",
                    reason="no_models_available",
                    action="return_extractive",
                    severity="warn",
                )
            )
            return GeneratedAnswer(
                text=self._extractive_fallback(cleaned_question, passages),
                cited_ids=[
                    str(item.get("source_id", "")) for item in passages
                ],
                confidence=0.25,
                llm_used=False,
                fallback_ladder=ladder,
                answer_style=answer_style,
            )

        max_tokens = self._resolve_max_tokens(intent, answer_style)
        temperature = self._resolve_temperature(intent, answer_style)

        raced = await self._race_models(
            references=references,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if raced is None:
            ladder.append(
                FallbackStepSchema(
                    stage="llm_generation",
                    reason="all_models_failed_or_timeout",
                    action="return_extractive",
                    severity="warn",
                )
            )
            return GeneratedAnswer(
                text=self._extractive_fallback(cleaned_question, passages),
                cited_ids=[
                    str(item.get("source_id", "")) for item in passages
                ],
                confidence=0.25,
                llm_used=False,
                fallback_ladder=ladder,
                answer_style=answer_style,
            )

        text, _ = raced
        text = self._strip_unrequested_closers(text)
        cited = self._extract_citations(text, passages)
        confidence = self._compute_confidence(text, cited, passages)

        if answer_style in {_ANSWER_STYLE_DETAILED, _ANSWER_STYLE_EXHAUSTIVE}:
            length_factor = min(1.0, len(text) / 1200.0)
            confidence = max(
                0.0, min(1.0, confidence * (0.6 + 0.4 * length_factor))
            )

        return GeneratedAnswer(
            text=text,
            cited_ids=cited,
            confidence=confidence,
            llm_used=True,
            fallback_ladder=ladder,
            answer_style=answer_style,
        )

    async def generate_catalog(
        self,
        question: str,
        snapshot: dict[str, Any],
        classifier_result: Any = None,
    ) -> GeneratedAnswer:
        cleaned_question = clean_text(question)

        if not cleaned_question:
            return GeneratedAnswer(
                text="Please ask a non-empty question.",
                confidence=0.0,
                llm_used=False,
                answer_style=_ANSWER_STYLE_STANDARD,
            )

        total = 0

        if isinstance(snapshot, dict):
            try:
                total = int(snapshot.get("total_documents", 0) or 0)
            except (TypeError, ValueError):
                total = 0

        if total <= 0:
            return GeneratedAnswer(
                text=(
                    "The local corpus is empty. Run a search or agent "
                    "session with the --rag flag to build the corpus "
                    "before asking catalog questions."
                ),
                confidence=0.0,
                llm_used=False,
                answer_style=_ANSWER_STYLE_STANDARD,
            )

        classifier_score = 0.0
        classifier_confidence = 0.0

        if classifier_result is not None:
            try:
                classifier_score = self._clamp(
                    float(getattr(classifier_result, "score", 0.0) or 0.0)
                )
            except (TypeError, ValueError):
                classifier_score = 0.0

            try:
                classifier_confidence = self._clamp(
                    float(
                        getattr(
                            classifier_result,
                            "confidence",
                            0.0,
                        )
                        or 0.0
                    )
                )
            except (TypeError, ValueError):
                classifier_confidence = 0.0

        user_prompt = self._build_catalog_prompt(cleaned_question, snapshot)

        messages = [
            LLMMessageSchema(role="system", content=_CATALOG_SYSTEM_PROMPT),
            LLMMessageSchema(role="user", content=user_prompt),
        ]

        ladder: list[FallbackStepSchema] = []

        try:
            references = get_fast_model_references(
                limit=self._catalog_model_limit
            )
        except Exception as exc:
            references = []
            ladder.append(
                FallbackStepSchema(
                    stage="catalog_model_selection",
                    reason=str(exc),
                    action="use_deterministic_fallback",
                    severity="warn",
                )
            )

        if not references:
            ladder.append(
                FallbackStepSchema(
                    stage="catalog_model_selection",
                    reason="no_models_available",
                    action="use_deterministic_fallback",
                    severity="warn",
                )
            )
            return self._build_catalog_deterministic_answer(
                snapshot=snapshot,
                classifier_score=classifier_score,
                classifier_confidence=classifier_confidence,
                ladder=ladder,
            )

        raced = await self._race_models(
            references=references,
            messages=messages,
            temperature=self._catalog_temperature,
            max_tokens=self._catalog_max_output_tokens,
            model_limit=self._catalog_model_limit,
        )

        if raced is None:
            ladder.append(
                FallbackStepSchema(
                    stage="catalog_generation",
                    reason="all_models_failed_or_timeout",
                    action="use_deterministic_fallback",
                    severity="warn",
                )
            )
            return self._build_catalog_deterministic_answer(
                snapshot=snapshot,
                classifier_score=classifier_score,
                classifier_confidence=classifier_confidence,
                ladder=ladder,
            )

        text, _ = raced
        text = self._strip_unrequested_closers(text)

        if not text.strip():
            ladder.append(
                FallbackStepSchema(
                    stage="catalog_generation",
                    reason="empty_model_output",
                    action="use_deterministic_fallback",
                    severity="warn",
                )
            )
            return self._build_catalog_deterministic_answer(
                snapshot=snapshot,
                classifier_score=classifier_score,
                classifier_confidence=classifier_confidence,
                ladder=ladder,
            )

        cited_dimensions = self._detect_cited_dimensions(text, snapshot)
        confidence, breakdown = self._compute_catalog_confidence(
            text=text,
            snapshot=snapshot,
            cited_dimensions=cited_dimensions,
            classifier_score=classifier_score,
            classifier_confidence=classifier_confidence,
            llm_used=True,
        )

        return GeneratedAnswer(
            text=text,
            llm_used=True,
            confidence=confidence,
            fallback_ladder=ladder,
            answer_style=_ANSWER_STYLE_STANDARD,
            confidence_breakdown=breakdown,
            cited_dimensions=cited_dimensions,
        )

    def _build_catalog_deterministic_answer(
        self,
        snapshot: dict[str, Any],
        classifier_score: float,
        classifier_confidence: float,
        ladder: list[FallbackStepSchema],
    ) -> GeneratedAnswer:
        text, cited_dimensions = self._build_catalog_fallback_text(snapshot)

        confidence, breakdown = self._compute_catalog_confidence(
            text=text,
            snapshot=snapshot,
            cited_dimensions=cited_dimensions,
            classifier_score=classifier_score,
            classifier_confidence=classifier_confidence,
            llm_used=False,
        )

        return GeneratedAnswer(
            text=text,
            llm_used=False,
            confidence=confidence,
            fallback_ladder=ladder,
            answer_style=_ANSWER_STYLE_STANDARD,
            confidence_breakdown=breakdown,
            cited_dimensions=cited_dimensions,
        )

    def _build_catalog_prompt(
        self,
        question: str,
        snapshot: dict[str, Any],
    ) -> str:
        lines: list[str] = [f"Question: {question}", ""]

        total = 0

        if isinstance(snapshot, dict):
            try:
                total = int(snapshot.get("total_documents", 0) or 0)
            except (TypeError, ValueError):
                total = 0

        lines.append(f"Total documents: {total}")

        by_platform = snapshot.get("by_platform") if isinstance(snapshot, dict) else None

        if isinstance(by_platform, dict) and by_platform:
            lines.append("")
            lines.append("By platform:")
            for platform, count in sorted(
                by_platform.items(),
                key=lambda item: -int(item[1] or 0),
            )[:_CATALOG_MAX_PLATFORM_SLICES]:
                lines.append(f"- {platform}: {int(count or 0)}")

        by_source_type = (
            snapshot.get("by_source_type") if isinstance(snapshot, dict) else None
        )

        if isinstance(by_source_type, dict) and by_source_type:
            lines.append("")
            lines.append("By source type:")
            for source_type, count in sorted(
                by_source_type.items(),
                key=lambda item: -int(item[1] or 0),
            )[:_CATALOG_MAX_SOURCE_TYPE_SLICES]:
                lines.append(f"- {source_type}: {int(count or 0)}")

        by_domain = snapshot.get("by_domain") if isinstance(snapshot, dict) else None

        if isinstance(by_domain, dict) and by_domain:
            lines.append("")
            lines.append("By domain:")
            for domain, count in sorted(
                by_domain.items(),
                key=lambda item: -int(item[1] or 0),
            )[:_CATALOG_MAX_DOMAIN_SLICES]:
                lines.append(f"- {domain}: {int(count or 0)}")

        by_topic = snapshot.get("by_topic") if isinstance(snapshot, dict) else None

        if isinstance(by_topic, dict) and by_topic:
            lines.append("")
            lines.append("Top topics:")
            for topic, count in sorted(
                by_topic.items(),
                key=lambda item: -int(item[1] or 0),
            )[:_CATALOG_MAX_TOPIC_SLICES]:
                lines.append(f"- {topic}: {int(count or 0)}")

        by_year = snapshot.get("by_year") if isinstance(snapshot, dict) else None

        if isinstance(by_year, dict) and by_year:
            lines.append("")
            lines.append("By year:")
            for year, count in sorted(
                by_year.items(),
                key=lambda item: -int(item[1] or 0),
            )[:_CATALOG_MAX_YEAR_SLICES]:
                lines.append(f"- {year}: {int(count or 0)}")

        by_run = snapshot.get("by_run_id") if isinstance(snapshot, dict) else None

        if isinstance(by_run, dict) and by_run:
            lines.append("")
            lines.append(f"Indexed across {len(by_run)} run(s).")

        titles = snapshot.get("titles") if isinstance(snapshot, dict) else None

        if isinstance(titles, list) and titles:
            lines.append("")
            lines.append("Recent titles:")
            for entry in titles[:_CATALOG_MAX_TITLE_SLICES]:
                if isinstance(entry, dict):
                    title = str(entry.get("title", "") or "")
                else:
                    title = str(entry or "")

                if title:
                    lines.append(f"- {title}")

        lines.append("")
        lines.append(
            "Answer the question using only the catalog above. "
            "If the user asks for a list, present a short list. "
            "If the user asks for a summary, give two or three sentences. "
            "If the catalog does not contain the dimension the user is "
            "asking about, say so plainly and describe what the catalog "
            "does contain. "
            "Do not invent any source that is not shown in the catalog."
        )

        return "\n".join(lines)

    def _build_catalog_fallback_text(
        self,
        snapshot: dict[str, Any],
    ) -> tuple[str, list[str]]:
        total = 0

        if isinstance(snapshot, dict):
            try:
                total = int(snapshot.get("total_documents", 0) or 0)
            except (TypeError, ValueError):
                total = 0

        parts: list[str] = [f"The local corpus contains {total} documents."]
        cited: list[str] = []

        by_platform = snapshot.get("by_platform") if isinstance(snapshot, dict) else None

        if isinstance(by_platform, dict) and by_platform:
            rows = sorted(
                by_platform.items(),
                key=lambda item: -int(item[1] or 0),
            )[:_CATALOG_MAX_PLATFORM_SLICES]
            parts.append(
                "By platform: "
                + ", ".join(f"{key} ({int(value or 0)})" for key, value in rows)
                + "."
            )
            cited.append("platform")

        by_source_type = (
            snapshot.get("by_source_type") if isinstance(snapshot, dict) else None
        )

        if isinstance(by_source_type, dict) and by_source_type:
            rows = sorted(
                by_source_type.items(),
                key=lambda item: -int(item[1] or 0),
            )[:_CATALOG_MAX_SOURCE_TYPE_SLICES]
            parts.append(
                "By source type: "
                + ", ".join(f"{key} ({int(value or 0)})" for key, value in rows)
                + "."
            )
            cited.append("source_type")

        by_domain = snapshot.get("by_domain") if isinstance(snapshot, dict) else None

        if isinstance(by_domain, dict) and by_domain:
            rows = sorted(
                by_domain.items(),
                key=lambda item: -int(item[1] or 0),
            )[:_CATALOG_MAX_DOMAIN_SLICES]
            parts.append(
                "By domain: "
                + ", ".join(f"{key} ({int(value or 0)})" for key, value in rows)
                + "."
            )
            cited.append("domain")

        by_topic = snapshot.get("by_topic") if isinstance(snapshot, dict) else None

        if isinstance(by_topic, dict) and by_topic:
            rows = sorted(
                by_topic.items(),
                key=lambda item: -int(item[1] or 0),
            )[:_CATALOG_MAX_TOPIC_SLICES]
            parts.append(
                "Top topics: "
                + ", ".join(f"{key} ({int(value or 0)})" for key, value in rows)
                + "."
            )
            cited.append("topic")

        by_year = snapshot.get("by_year") if isinstance(snapshot, dict) else None

        if isinstance(by_year, dict) and by_year:
            rows = sorted(
                by_year.items(),
                key=lambda item: -int(item[1] or 0),
            )[:_CATALOG_MAX_YEAR_SLICES]
            parts.append(
                "Years covered: "
                + ", ".join(f"{key} ({int(value or 0)})" for key, value in rows)
                + "."
            )
            cited.append("year")

        by_run = snapshot.get("by_run_id") if isinstance(snapshot, dict) else None

        if isinstance(by_run, dict) and by_run:
            parts.append(f"Indexed across {len(by_run)} run(s).")
            cited.append("run")

        titles = snapshot.get("titles") if isinstance(snapshot, dict) else None

        if isinstance(titles, list) and titles:
            sample: list[str] = []

            for entry in titles[:_CATALOG_MAX_TITLE_SLICES]:
                if isinstance(entry, dict):
                    title = str(entry.get("title", "") or "")
                else:
                    title = str(entry or "")

                if title:
                    sample.append(title)

            if sample:
                parts.append("Recent titles: " + "; ".join(sample) + ".")
                cited.append("title")

        return " ".join(parts), cited

    def _compute_catalog_confidence(
        self,
        text: str,
        snapshot: dict[str, Any],
        cited_dimensions: list[str],
        classifier_score: float,
        classifier_confidence: float,
        llm_used: bool,
    ) -> tuple[float, dict[str, float]]:
        breakdown: dict[str, float] = {
            "llm_used": 1.0 if llm_used else 0.55,
            "classifier_confidence": self._clamp(classifier_confidence),
            "snapshot_richness": self._compute_snapshot_richness(snapshot),
            "answer_substance": self._compute_answer_substance(text),
            "dimension_coverage": self._clamp(
                len(cited_dimensions) / max(1, len(_CATALOG_DIMENSIONS))
            ),
        }

        blended = 0.0

        for key, weight in _CATALOG_CONFIDENCE_WEIGHTS.items():
            blended += float(weight) * breakdown.get(key, 0.0)

        if not str(text or "").strip():
            blended = 0.0
        elif not llm_used:
            blended *= 0.85

        blended = max(0.0, min(1.0, blended))

        return blended, breakdown

    def _compute_snapshot_richness(self, snapshot: dict[str, Any]) -> float:
        if not isinstance(snapshot, dict):
            return 0.0

        present = 0

        if snapshot.get("by_platform"):
            present += 1

        if snapshot.get("by_source_type"):
            present += 1

        if snapshot.get("by_domain"):
            present += 1

        if snapshot.get("by_topic"):
            present += 1

        if snapshot.get("by_year"):
            present += 1

        if snapshot.get("by_run_id"):
            present += 1

        if snapshot.get("titles"):
            present += 1

        return present / max(1, len(_CATALOG_DIMENSIONS))

    def _compute_answer_substance(self, text: str) -> float:
        cleaned = str(text or "").strip()

        if not cleaned:
            return 0.0

        length = len(cleaned)
        has_number = any(ch.isdigit() for ch in cleaned)

        if length >= 200 and has_number:
            return 1.0

        if length >= 120 and has_number:
            return 0.85

        if length >= 80:
            return 0.7

        if length >= 40:
            return 0.5

        if length > 0:
            return 0.3

        return 0.0

    def _detect_cited_dimensions(
        self,
        text: str,
        snapshot: dict[str, Any],
    ) -> list[str]:
        if not text:
            return []

        lowered = text.lower()
        cited: list[str] = []

        for dimension in _CATALOG_DIMENSIONS:
            markers = _CATALOG_DIMENSION_MARKERS.get(dimension, ())

            for marker in markers:
                if marker in lowered:
                    if dimension not in cited:
                        cited.append(dimension)
                    break

        return cited

    @staticmethod
    def _clamp(value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0

        return max(0.0, min(1.0, numeric))

    async def stream_generate(
        self,
        question: str,
        topic: TopicProfileSchema | None,
        passages: list[dict[str, Any]],
    ) -> AsyncIterator[str]:
        cleaned_question = clean_text(question)

        if not cleaned_question or not passages:
            fallback = self._extractive_fallback(cleaned_question, passages)

            if fallback:
                yield fallback

            return

        answer_style = self._resolve_answer_style(cleaned_question, topic)
        intent = self._resolve_intent(topic, answer_style)

        system_prompt = self._build_system_prompt(intent, answer_style)
        user_prompt = self._build_user_prompt(
            cleaned_question,
            topic,
            passages,
            answer_style,
        )

        messages = [
            LLMMessageSchema(role="system", content=system_prompt),
            LLMMessageSchema(role="user", content=user_prompt),
        ]

        references = self._select_models(intent, answer_style)

        if not references:
            fallback = self._extractive_fallback(cleaned_question, passages)

            if fallback:
                yield fallback

            return

        reference = references[0]
        max_tokens = self._resolve_max_tokens(intent, answer_style)
        temperature = self._resolve_temperature(intent, answer_style)

        yielded_any = False

        try:
            async with LLMProviderManager() as manager:
                stream_method = getattr(manager, "stream", None)

                if not callable(stream_method):
                    raise RuntimeError("Provider streaming unavailable")

                request = LLMRequestSchema(
                    provider=reference.provider,
                    model=reference.model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format="text",
                )
                request = validate_llm_request(request)

                async for chunk in stream_method(request):
                    text = str(chunk or "")

                    if not text:
                        continue

                    yielded_any = True
                    yield text
        except Exception as exc:
            self._logger.warning(f"Streaming generation failed: {exc}")

        if not yielded_any:
            fallback = self._extractive_fallback(cleaned_question, passages)

            if fallback:
                yield fallback

    async def stream(
        self,
        answer: GeneratedAnswer,
    ) -> AsyncIterator[str]:
        text = str(answer.text or "")

        if not text:
            return

        step = 72
        index = 0

        while index < len(text):
            yield text[index : index + step]
            index += step

    async def _race_models(
        self,
        references: list[Any],
        messages: list[LLMMessageSchema],
        temperature: float,
        max_tokens: int,
        model_limit: int | None = None,
    ) -> tuple[str, Any] | None:
        if not references:
            return None

        effective_limit = (
            int(model_limit)
            if model_limit is not None
            else int(self._model_limit)
        )
        effective_limit = max(1, effective_limit)

        top = references[: max(2, min(effective_limit, len(references)))]

        if len(top) == 1:
            return await self._call_one(
                top[0],
                messages,
                temperature,
                max_tokens,
            )

        async with LLMProviderManager() as manager:
            tasks = {
                asyncio.create_task(
                    self._call_one_with_manager(
                        manager,
                        reference,
                        messages,
                        temperature,
                        max_tokens,
                    )
                ): reference
                for reference in top
            }

            pending = set(tasks.keys())

            try:
                while pending:
                    done, pending = await asyncio.wait(
                        pending,
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=self._race_timeout,
                    )

                    if not done:
                        return None

                    for task in done:
                        try:
                            result = task.result()
                        except Exception:
                            continue

                        if result is not None:
                            return result

                return None
            finally:
                for task in pending:
                    if not task.done():
                        task.cancel()

                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)

    async def _call_one_with_manager(
        self,
        manager: LLMProviderManager,
        reference: Any,
        messages: list[LLMMessageSchema],
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, Any] | None:
        try:
            request = LLMRequestSchema(
                provider=reference.provider,
                model=reference.model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format="text",
            )
            request = validate_llm_request(request)
            response = await manager.complete(request)

            if response is None:
                return None

            response = validate_llm_response(response)
            text = clean_text(response.content)

            if not text:
                return None

            return text, reference
        except Exception as exc:
            self._logger.warning(
                f"Model {getattr(reference, 'model_id', '?')} failed: {exc}"
            )
            return None

    async def _call_one(
        self,
        reference: Any,
        messages: list[LLMMessageSchema],
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, Any] | None:
        try:
            async with LLMProviderManager() as manager:
                return await self._call_one_with_manager(
                    manager,
                    reference,
                    messages,
                    temperature,
                    max_tokens,
                )
        except Exception as exc:
            self._logger.warning(f"Single-model call failed: {exc}")
            return None

    def _resolve_answer_style(
        self,
        question: str,
        topic: TopicProfileSchema | None,
    ) -> str:
        text_style = self._detect_style_from_text(question)

        if text_style != _ANSWER_STYLE_STANDARD:
            return text_style

        analyzer_style = ""

        if topic is not None:
            raw = getattr(topic, "answer_style", None)

            if isinstance(raw, str):
                candidate = raw.strip().lower()

                if candidate in _VALID_ANSWER_STYLES:
                    analyzer_style = candidate

        if analyzer_style and analyzer_style != _ANSWER_STYLE_STANDARD:
            return analyzer_style

        return _ANSWER_STYLE_STANDARD

    def _detect_style_from_text(self, question: str) -> str:
        lowered = clean_text(question).lower()

        if not lowered:
            return _ANSWER_STYLE_STANDARD

        scores: dict[str, int] = {
            _ANSWER_STYLE_SHORT: 0,
            _ANSWER_STYLE_DETAILED: 0,
            _ANSWER_STYLE_EXHAUSTIVE: 0,
        }

        for marker in _SHORT_MARKERS:
            if marker in lowered:
                scores[_ANSWER_STYLE_SHORT] += 1

        for marker in _DETAILED_MARKERS:
            if marker in lowered:
                scores[_ANSWER_STYLE_DETAILED] += 1

        for marker in _EXHAUSTIVE_MARKERS:
            if marker in lowered:
                scores[_ANSWER_STYLE_EXHAUSTIVE] += 1

        best_style = _ANSWER_STYLE_STANDARD
        best_score = 0
        best_priority = -1

        for style, score in scores.items():
            if score <= 0:
                continue

            priority = _ANSWER_STYLE_PRIORITY.get(style, 0)

            if score > best_score or (
                score == best_score and priority > best_priority
            ):
                best_style = style
                best_score = score
                best_priority = priority

        if best_score <= 0:
            return _ANSWER_STYLE_STANDARD

        return best_style

    def _resolve_intent(
        self,
        topic: TopicProfileSchema | None,
        answer_style: str,
    ) -> str:
        raw_intent = ""

        if topic is not None:
            raw = getattr(topic, "intent", None)

            if isinstance(raw, str):
                raw_intent = raw.strip().lower()

        if not raw_intent:
            raw_intent = "overview"

        if answer_style in {_ANSWER_STYLE_DETAILED, _ANSWER_STYLE_EXHAUSTIVE}:
            if raw_intent in {"overview", "explanation", "", "definition"}:
                return _INTENT_DETAILED_EXPLANATION

        return raw_intent

    def _select_models(
        self,
        intent: str,
        answer_style: str,
    ) -> list[Any]:
        prefer_strong = (
            intent in _DEEP_INTENTS
            or answer_style
            in {_ANSWER_STYLE_DETAILED, _ANSWER_STYLE_EXHAUSTIVE}
        )

        if prefer_strong:
            try:
                strong = get_strong_model_references(limit=self._model_limit)
            except Exception:
                strong = []

            try:
                fast = get_fast_model_references(limit=1)
            except Exception:
                fast = []

            merged = strong + fast
        elif intent in _SHALLOW_INTENTS:
            try:
                fast = get_fast_model_references(limit=self._model_limit)
            except Exception:
                fast = []

            try:
                strong = get_strong_model_references(limit=1)
            except Exception:
                strong = []

            merged = fast + strong
        else:
            try:
                fast = get_fast_model_references(limit=self._model_limit)
            except Exception:
                fast = []

            merged = fast

        seen: set[tuple[str, str]] = set()
        result: list[Any] = []

        for reference in merged:
            key = (
                str(getattr(reference.provider, "value", reference.provider)),
                str(reference.model_id),
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(reference)

        return result

    def _build_system_prompt(
        self,
        intent: str,
        answer_style: str,
    ) -> str:
        if answer_style == _ANSWER_STYLE_SHORT:
            return f"{_SYSTEM_BASE} {_SYSTEM_SHORT_SUFFIX}"

        if answer_style == _ANSWER_STYLE_DETAILED:
            return f"{_SYSTEM_BASE} {_SYSTEM_DETAILED_SECTIONS}"

        if answer_style == _ANSWER_STYLE_EXHAUSTIVE:
            return (
                f"{_SYSTEM_BASE} {_SYSTEM_DETAILED_SECTIONS} "
                f"{_SYSTEM_EXHAUSTIVE_SUFFIX}"
            )

        structure = (
            _INTENT_STRUCTURE.get(intent)
            or _INTENT_STRUCTURE["overview"]
        )

        return f"{_SYSTEM_BASE} {structure}"

    def _build_user_prompt(
        self,
        question: str,
        topic: TopicProfileSchema | None,
        passages: list[dict[str, Any]],
        answer_style: str,
    ) -> str:
        lines: list[str] = [f"Question: {question}"]

        if topic is not None:
            primary_topic = str(
                getattr(topic, "primary_topic", "") or ""
            ).strip()

            if primary_topic:
                lines.append(f"Primary topic: {primary_topic}")

            intent = str(
                getattr(topic, "intent", "") or ""
            ).strip().lower()

            if intent:
                lines.append(f"Detected intent: {intent}")

        if answer_style in {
            _ANSWER_STYLE_SHORT,
            _ANSWER_STYLE_DETAILED,
            _ANSWER_STYLE_EXHAUSTIVE,
        }:
            lines.append(f"Requested answer style: {answer_style}")

        lines.append("")
        lines.append("Numbered passages:")
        lines.append("")

        for index, passage in enumerate(passages, start=1):
            title = clean_text(passage.get("title", "")) or "Untitled"
            text = truncate_text(
                clean_text(passage.get("text", "") or ""),
                max_length=_MAX_PASSAGE_CHARS,
                suffix="",
            )
            lines.append(f"[{index}] {title}")

            if text:
                lines.append(text)

            lines.append("")

        lines.append(
            "Answer the question using only the passages above. "
            "Cite each claim with the passage number in square brackets. "
            "Do not introduce any concept that is not in the passages. "
            "Do not add a related terms line, related concepts line, "
            "further reading line, sources line, references section, "
            "see also line, or closing summary line. "
            "Stop writing as soon as the asked question is answered."
        )

        return "\n".join(lines)

    def _resolve_max_tokens(
        self,
        intent: str,
        answer_style: str,
    ) -> int:
        if answer_style == _ANSWER_STYLE_SHORT:
            return _SHORT_MAX_TOKENS

        if answer_style == _ANSWER_STYLE_DETAILED:
            return _DETAILED_MAX_TOKENS

        if answer_style == _ANSWER_STYLE_EXHAUSTIVE:
            return _EXHAUSTIVE_MAX_TOKENS

        return _INTENT_MAX_TOKENS.get(intent, self._max_output_tokens)

    def _resolve_temperature(
        self,
        intent: str,
        answer_style: str,
    ) -> float:
        if answer_style == _ANSWER_STYLE_SHORT:
            return _SHORT_TEMPERATURE

        if answer_style == _ANSWER_STYLE_DETAILED:
            return _DETAILED_TEMPERATURE

        if answer_style == _ANSWER_STYLE_EXHAUSTIVE:
            return _EXHAUSTIVE_TEMPERATURE

        return _INTENT_TEMPERATURE.get(intent, self._temperature)

    def _strip_unrequested_closers(self, text: str) -> str:
        if not text:
            return text

        banned_patterns = (
            r"\n+\s*\*?\*?\s*related\s+terms\s*:.*$",
            r"\n+\s*\*?\*?\s*related\s+concepts\s*:.*$",
            r"\n+\s*\*?\*?\s*further\s+reading\s*:.*$",
            r"\n+\s*\*?\*?\s*see\s+also\s*:.*$",
            r"\n+\s*\*?\*?\s*suggested\s+next\s+steps?\s*:.*$",
            r"\n+\s*\*?\*?\s*additional\s+resources?\s*:.*$",
            r"\n+\s*\*?\*?\s*further\s+exploration\s*:.*$",
            r"\n+\s*\*?\*?\s*references?\s*:.*$",
            r"\n+\s*\*?\*?\s*sources?\s*:\s*\[.*$",
        )

        result = text

        for pattern in banned_patterns:
            try:
                result = re.sub(
                    pattern,
                    "",
                    result,
                    flags=re.IGNORECASE | re.DOTALL,
                )
            except Exception:
                continue

        result = result.rstrip()

        return result

    def _extract_citations(
        self,
        text: str,
        passages: list[dict[str, Any]],
    ) -> list[str]:
        cited_ids: list[str] = []
        seen: set[str] = set()

        for match in _CITATION_PATTERN.finditer(text):
            try:
                index = int(match.group(1))
            except (TypeError, ValueError):
                continue

            if index < 1 or index > len(passages):
                continue

            source_id = str(
                passages[index - 1].get("source_id", "") or ""
            )

            if source_id and source_id not in seen:
                seen.add(source_id)
                cited_ids.append(source_id)

        return cited_ids

    def _compute_confidence(
        self,
        text: str,
        cited_ids: list[str],
        passages: list[dict[str, Any]],
    ) -> float:
        if not text or not passages:
            return 0.0

        total = len(passages)
        cited_count = len(cited_ids)
        coverage = cited_count / max(1, total)

        scores: list[float] = []

        for passage in passages:
            source_id = str(passage.get("source_id", "") or "")

            if source_id not in cited_ids:
                continue

            try:
                scores.append(float(passage.get("score", 0.0) or 0.0))
            except (TypeError, ValueError):
                continue

        mean_score = sum(scores) / len(scores) if scores else 0.0
        length_factor = min(1.0, len(text) / 600.0)

        confidence = (
            0.35 * coverage
            + 0.45 * mean_score
            + 0.20 * length_factor
        )

        return max(0.0, min(1.0, confidence))

    def _extractive_fallback(
        self,
        question: str,
        passages: list[dict[str, Any]],
    ) -> str:
        lines: list[str] = [
            f"Question: {question}",
            "",
            "Locally retrieved passages:",
            "",
        ]

        for index, passage in enumerate(passages[:5], start=1):
            title = clean_text(passage.get("title", "")) or "Untitled"
            text = truncate_text(
                clean_text(passage.get("text", "") or ""),
                max_length=360,
                suffix="...",
            )
            url = str(passage.get("url", "") or "")

            lines.append(f"{index}. {title}")

            if url:
                lines.append(f"   URL: {url}")

            if text:
                lines.append(f"   {text}")

            lines.append("")

        lines.append(
            "This fallback is extractive because the language model was "
            "unavailable."
        )

        return "\n".join(lines).strip()