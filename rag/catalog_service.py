from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.schemas import FallbackStepSchema
from utils.logger import get_logger
from utils.text import clean_text


_DEFAULT_LIMITS: dict[str, int] = {
    "max_documents": 2000,
    "max_titles": 20,
    "max_platform_slices": 12,
    "max_source_type_slices": 12,
    "max_domain_slices": 12,
    "max_topic_slices": 15,
    "max_year_slices": 10,
    "max_run_slices": 10,
}

_DEFAULT_ESCALATION: dict[str, int] = {
    "min_documents_for_topics": 5,
    "min_documents_for_domains": 3,
    "min_documents_for_platforms": 2,
}

_DIMENSION_TO_SNAPSHOT_KEY: dict[str, str] = {
    "domain": "by_domain",
    "topic": "by_topic",
    "platform": "by_platform",
    "source_type": "by_source_type",
    "year": "by_year",
    "run": "by_run_id",
}

_DIMENSION_MARKERS: dict[str, tuple[str, ...]] = {
    "domain": (
        "domain", "domains", "field", "fields",
        "subject area", "subject areas",
        "discipline", "disciplines",
    ),
    "topic": ("topic", "topics", "subject", "subjects"),
    "platform": ("platform", "platforms", "source site", "source sites"),
    "source_type": (
        "source type", "source types",
        "type of source", "types of sources",
        "format", "formats",
    ),
    "year": (
        "year", "years",
        "publication year", "publication years",
        "date", "dates",
    ),
    "run": ("run", "runs", "session", "sessions", "indexing run"),
}

_DIMENSION_MIN_KEYS: dict[str, str] = {
    "topic": "min_documents_for_topics",
    "domain": "min_documents_for_domains",
    "platform": "min_documents_for_platforms",
}

_NON_ANSWERABLE_VALUES: dict[str, frozenset[str]] = {
    "domain": frozenset({"unknown", "general"}),
    "topic": frozenset({
        "unknown",
        "general",
        "introduction",
        "overview",
    }),
    "platform": frozenset({"unknown"}),
    "source_type": frozenset({"unknown"}),
    "year": frozenset({"unknown"}),
    "run": frozenset({"unknown"}),
}


@dataclass(frozen=True)
class CatalogAnswer:
    text: str
    confidence: float
    confidence_breakdown: dict[str, float]
    snapshot: dict[str, Any]
    signals: dict[str, Any]
    escalation: str | None
    fallback_ladder: list[FallbackStepSchema] = field(default_factory=list)
    llm_used: bool = False
    context_used: int = 0
    cited_dimensions: list[str] = field(default_factory=list)


class CatalogService:
    def __init__(
        self,
        vector_store: Any,
        generator: Any,
        config_path: Path | str | None = None,
        config_override: dict[str, Any] | None = None,
    ) -> None:
        self._store = vector_store
        self._generator = generator
        self._config_path = Path(config_path) if config_path else (
            Path(__file__).resolve().parent.parent / "configs" / "settings.yaml"
        )
        self._logger = get_logger("rag.catalog_service")

        self._limits: dict[str, int] = dict(_DEFAULT_LIMITS)
        self._escalation: dict[str, int] = dict(_DEFAULT_ESCALATION)

        self._load_config()
        self._apply_override(config_override)

    def _load_config(self) -> None:
        try:
            import yaml
        except Exception as exc:
            self._logger.warning(
                f"PyYAML unavailable for catalog service: {exc}"
            )
            return

        try:
            if not self._config_path.exists():
                return

            with open(self._config_path, "r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
        except Exception as exc:
            self._logger.warning(f"Catalog service config read failed: {exc}")
            return

        if not isinstance(raw, dict):
            return

        section = raw.get("rag_catalog", {})

        if not isinstance(section, dict):
            return

        snapshot_section = section.get("snapshot", {})

        if isinstance(snapshot_section, dict):
            for key in self._limits:
                value = snapshot_section.get(key)

                if value is None:
                    continue

                try:
                    self._limits[key] = max(1, int(value))
                except (TypeError, ValueError):
                    continue

        escalation_section = section.get("escalation", {})

        if isinstance(escalation_section, dict):
            for key in self._escalation:
                value = escalation_section.get(key)

                if value is None:
                    continue

                try:
                    self._escalation[key] = max(0, int(value))
                except (TypeError, ValueError):
                    continue

    def _apply_override(self, override: dict[str, Any] | None) -> None:
        if not isinstance(override, dict):
            return

        for key, value in override.items():
            if key in self._limits:
                try:
                    self._limits[key] = max(1, int(value))
                except (TypeError, ValueError):
                    continue

            if key in self._escalation:
                try:
                    self._escalation[key] = max(0, int(value))
                except (TypeError, ValueError):
                    continue

    async def answer(
        self,
        question: str,
        run_id: str | None,
        session_id: str,
        classification: Any,
    ) -> CatalogAnswer:
        cleaned = clean_text(question)
        ladder: list[FallbackStepSchema] = []

        if not cleaned:
            return CatalogAnswer(
                text="Please ask a non-empty question.",
                confidence=0.0,
                confidence_breakdown={},
                snapshot={},
                signals={},
                escalation="empty_question",
                fallback_ladder=[
                    FallbackStepSchema(
                        stage="catalog_input",
                        reason="empty_question",
                        action="return_empty",
                        severity="warn",
                    )
                ],
                llm_used=False,
                context_used=0,
                cited_dimensions=[],
            )

        raw_snapshot = self._build_snapshot(run_id=run_id, ladder=ladder)

        if raw_snapshot is None:
            return CatalogAnswer(
                text=(
                    "The catalog snapshot could not be read from the "
                    "vector store. Check logs for the underlying error."
                ),
                confidence=0.0,
                confidence_breakdown={},
                snapshot={},
                signals=self._signals_from(classification),
                escalation="snapshot_failed",
                fallback_ladder=ladder,
                llm_used=False,
                context_used=0,
                cited_dimensions=[],
            )

        snapshot = self._trim_snapshot(raw_snapshot)

        total = 0

        try:
            total = int(snapshot.get("total_documents", 0) or 0)
        except (TypeError, ValueError):
            total = 0

        if total <= 0:
            ladder.append(
                FallbackStepSchema(
                    stage="catalog_snapshot",
                    reason="empty_corpus",
                    action="return_escalation",
                    severity="warn",
                )
            )
            return CatalogAnswer(
                text=(
                    "The local corpus is empty. Run a search or agent "
                    "session with the --rag flag to build the corpus "
                    "before asking catalog questions."
                ),
                confidence=0.0,
                confidence_breakdown={},
                snapshot=snapshot,
                signals=self._signals_from(classification),
                escalation="empty_corpus",
                fallback_ladder=ladder,
                llm_used=False,
                context_used=0,
                cited_dimensions=[],
            )

        requested_dimensions = self._detect_requested_dimensions(cleaned)

        escalation = self._check_escalation(
            snapshot=snapshot,
            requested_dimensions=requested_dimensions,
        )

        if escalation is not None:
            ladder.append(
                FallbackStepSchema(
                    stage="catalog_escalation",
                    reason=escalation,
                    action="return_escalation",
                    severity="warn",
                )
            )
            message = self._build_escalation_message(
                snapshot=snapshot,
                requested_dimensions=requested_dimensions,
                escalation=escalation,
            )
            confidence, breakdown = self._compute_escalation_confidence(
                snapshot=snapshot,
                requested_dimensions=requested_dimensions,
                classification=classification,
            )

            return CatalogAnswer(
                text=message,
                confidence=confidence,
                confidence_breakdown=breakdown,
                snapshot=snapshot,
                signals=self._signals_from(classification),
                escalation=escalation,
                fallback_ladder=ladder,
                llm_used=False,
                context_used=total,
                cited_dimensions=[],
            )

        generated = await self._run_generator(
            question=cleaned,
            snapshot=snapshot,
            classification=classification,
            ladder=ladder,
        )

        return CatalogAnswer(
            text=generated.text,
            confidence=generated.confidence,
            confidence_breakdown=dict(generated.confidence_breakdown or {}),
            snapshot=snapshot,
            signals=self._signals_from(classification),
            escalation=None,
            fallback_ladder=ladder + list(generated.fallback_ladder or []),
            llm_used=bool(generated.llm_used),
            context_used=total,
            cited_dimensions=list(generated.cited_dimensions or []),
        )

    def _build_snapshot(
        self,
        run_id: str | None,
        ladder: list[FallbackStepSchema],
    ) -> dict[str, Any] | None:
        method = getattr(self._store, "catalog_snapshot", None)

        if not callable(method):
            ladder.append(
                FallbackStepSchema(
                    stage="catalog_snapshot",
                    reason="store_missing_catalog_snapshot",
                    action="return_failure",
                    severity="error",
                )
            )
            return None

        try:
            return method(
                run_id=run_id,
                top_titles=self._limits["max_titles"],
            )
        except Exception as exc:
            self._logger.warning(f"catalog_snapshot failed: {exc}")
            ladder.append(
                FallbackStepSchema(
                    stage="catalog_snapshot",
                    reason=str(exc),
                    action="return_failure",
                    severity="error",
                )
            )
            return None

    def _trim_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(snapshot, dict):
            return {}

        trimmed = dict(snapshot)

        for key, limit_key in (
            ("by_platform", "max_platform_slices"),
            ("by_source_type", "max_source_type_slices"),
            ("by_domain", "max_domain_slices"),
            ("by_topic", "max_topic_slices"),
            ("by_year", "max_year_slices"),
            ("by_run_id", "max_run_slices"),
        ):
            value = snapshot.get(key)

            if not isinstance(value, dict):
                continue

            limit = self._limits.get(limit_key, 0)

            if limit <= 0:
                continue

            sorted_items = sorted(
                value.items(),
                key=lambda item: -int(item[1] or 0),
            )

            trimmed[key] = dict(sorted_items[:limit])

        titles = snapshot.get("titles")

        if isinstance(titles, list):
            limit = self._limits.get("max_titles", 0)

            if limit > 0:
                trimmed["titles"] = titles[:limit]

        return trimmed

    def _detect_requested_dimensions(self, question: str) -> list[str]:
        lowered = question.lower()
        requested: list[str] = []

        for dimension, markers in _DIMENSION_MARKERS.items():
            for marker in markers:
                if marker in lowered:
                    if dimension not in requested:
                        requested.append(dimension)
                    break

        return requested

    def _check_escalation(
        self,
        snapshot: dict[str, Any],
        requested_dimensions: list[str],
    ) -> str | None:
        if not requested_dimensions:
            return None

        try:
            total_corpus = int(snapshot.get("total_documents", 0) or 0)
        except (TypeError, ValueError):
            total_corpus = 0

        for dimension in requested_dimensions:
            key = _DIMENSION_TO_SNAPSHOT_KEY.get(dimension)

            if key is None:
                continue

            value = snapshot.get(key)

            if not isinstance(value, dict) or not value:
                return f"dimension_missing:{dimension}"

            non_answerable = _NON_ANSWERABLE_VALUES.get(
                dimension,
                frozenset({"unknown"}),
            )

            real_entries = {
                str(k): int(v or 0)
                for k, v in value.items()
                if str(k).strip().lower() not in non_answerable
            }

            if not real_entries:
                return f"dimension_unknown:{dimension}"

            min_key = _DIMENSION_MIN_KEYS.get(dimension)

            if min_key is None:
                continue

            min_required = self._escalation.get(min_key, 0)

            if min_required <= 0:
                continue

            if total_corpus < min_required:
                continue

            total_in_dimension = sum(real_entries.values())

            if total_in_dimension < min_required:
                return f"dimension_sparse:{dimension}"

        return None

    def _build_escalation_message(
        self,
        snapshot: dict[str, Any],
        requested_dimensions: list[str],
        escalation: str,
    ) -> str:
        total = 0

        try:
            total = int(snapshot.get("total_documents", 0) or 0)
        except (TypeError, ValueError):
            total = 0

        available: list[str] = []

        for label, key in (
            ("platform", "by_platform"),
            ("source type", "by_source_type"),
            ("domain", "by_domain"),
            ("topic", "by_topic"),
            ("year", "by_year"),
            ("run", "by_run_id"),
        ):
            value = snapshot.get(key)

            if isinstance(value, dict) and value:
                real = [
                    k for k in value.keys()
                    if str(k).strip().lower() != "unknown"
                ]

                if real:
                    available.append(label)

        topic = ""

        if ":" in escalation:
            _, topic = escalation.split(":", 1)

        head = f"The local corpus contains {total} documents."

        if escalation.startswith("dimension_missing"):
            body = (
                f"The corpus does not currently have information about "
                f"{topic}. The documents were indexed before {topic} "
                f"was captured. Run a new search or agent session with "
                f"--rag to rebuild the index with {topic} tags."
            )
        elif escalation.startswith("dimension_unknown"):
            body = (
                f"The corpus has documents, but none of them carry "
                f"a meaningful {topic} tag yet. Run a new search or agent "
                f"session with --rag so the index-time classifier can "
                f"tag them."
            )
        elif escalation.startswith("dimension_sparse"):
            body = (
                f"Only a small number of documents carry a {topic} tag, "
                f"which is not enough to answer this question reliably. "
                f"Run a new search or agent session with --rag to add "
                f"more documents."
            )
        else:
            body = "The catalog data is not sufficient to answer this question."

        if available:
            footer = (
                "The corpus does contain information about "
                + ", ".join(available)
                + ". Ask a question about one of those dimensions instead."
            )
        else:
            footer = (
                "The corpus exists but does not carry any tagged "
                "dimensions yet."
            )

        return f"{head} {body} {footer}"

    def _compute_escalation_confidence(
        self,
        snapshot: dict[str, Any],
        requested_dimensions: list[str],
        classification: Any,
    ) -> tuple[float, dict[str, float]]:
        richness = 0.0

        if isinstance(snapshot, dict):
            present = 0

            for key in (
                "by_platform",
                "by_source_type",
                "by_domain",
                "by_topic",
                "by_year",
                "by_run_id",
                "titles",
            ):
                value = snapshot.get(key)

                if isinstance(value, dict) and value:
                    present += 1
                elif isinstance(value, list) and value:
                    present += 1

            richness = present / 7.0

        classifier_confidence = 0.0

        if classification is not None:
            try:
                classifier_confidence = float(
                    getattr(classification, "confidence", 0.0) or 0.0
                )
            except (TypeError, ValueError):
                classifier_confidence = 0.0

        breakdown = {
            "snapshot_richness": round(richness, 4),
            "classifier_confidence": round(classifier_confidence, 4),
            "requested_dimension_count": float(len(requested_dimensions)),
        }

        confidence = 0.35 * richness + 0.25 * classifier_confidence
        confidence = max(0.0, min(0.5, confidence))

        return round(confidence, 4), breakdown

    async def _run_generator(
        self,
        question: str,
        snapshot: dict[str, Any],
        classification: Any,
        ladder: list[FallbackStepSchema],
    ) -> Any:
        method = getattr(self._generator, "generate_catalog", None)

        if not callable(method):
            ladder.append(
                FallbackStepSchema(
                    stage="catalog_generator",
                    reason="generator_missing_generate_catalog",
                    action="return_static",
                    severity="error",
                )
            )

            return _StaticCatalogResult(
                text=(
                    "The catalog generator is not available. "
                    "The snapshot was read successfully but no answer "
                    "could be produced."
                ),
                confidence=0.0,
                llm_used=False,
                confidence_breakdown={},
                cited_dimensions=[],
                fallback_ladder=[],
            )

        try:
            result = await method(
                question=question,
                snapshot=snapshot,
                classifier_result=classification,
            )
        except Exception as exc:
            self._logger.warning(f"generate_catalog raised: {exc}")
            ladder.append(
                FallbackStepSchema(
                    stage="catalog_generator",
                    reason=str(exc),
                    action="return_static",
                    severity="error",
                )
            )

            return _StaticCatalogResult(
                text=(
                    "The catalog generator failed while producing "
                    "the answer. Check logs for the underlying error."
                ),
                confidence=0.0,
                llm_used=False,
                confidence_breakdown={},
                cited_dimensions=[],
                fallback_ladder=[],
            )

        return result

    @staticmethod
    def _signals_from(classification: Any) -> dict[str, Any]:
        if classification is None:
            return {}

        signals = getattr(classification, "signals", None)

        if isinstance(signals, dict):
            return dict(signals)

        return {}


class _StaticCatalogResult:
    def __init__(
        self,
        text: str,
        confidence: float,
        llm_used: bool,
        confidence_breakdown: dict[str, float],
        cited_dimensions: list[str],
        fallback_ladder: list[FallbackStepSchema],
    ) -> None:
        self.text = text
        self.confidence = confidence
        self.llm_used = llm_used
        self.confidence_breakdown = confidence_breakdown
        self.cited_dimensions = cited_dimensions
        self.fallback_ladder = fallback_ladder