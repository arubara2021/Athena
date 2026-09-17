from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.logger import get_logger

_logger = get_logger("core.ranking_config")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "ranking.yaml"


# ════════════════════════════════════════════════════════════════════════════
# In-code defaults. The app behaves identically to these when ranking.yaml is
# missing, malformed, or only partially filled.
# ════════════════════════════════════════════════════════════════════════════
_DEFAULTS: dict[str, Any] = {
    "scorer": {
        "weights": {
            "relevance": 0.42,
            "authority": 0.12,
            "type_quality": 0.10,
            "impact": 0.10,
            "recency": 0.08,
            "content": 0.08,
            "support": 0.05,
            "authors": 0.05,
        }
    },
    "classifier": {
        "keyword_weights": {"beginner": 0.9, "intermediate": 0.6, "advanced": 0.7},
        "content": {
            "weight_paper": 2.6,
            "weight_other": 1.1,
            "beginner_max_complexity": 0.35,
            "intermediate_max_complexity": 0.62,
            "fog_min": 6.0,
            "fog_max": 18.0,
            "fog_weight": 0.6,
            "jargon_weight": 0.4,
            "jargon_density_cap": 0.05,
            "complex_word_syllables": 3,
        },
        "intro_phrasing": {
            "beginner_weight": 1.1,
            "intermediate_weight": 0.3,
            "max_points": 3,
        },
        "confidence": {"base": 0.40, "margin_factor": 0.40, "max": 0.95},
        "source_type_bonus": {
            "documentation": {"beginner": 1.2, "intermediate": 0.4},
            "course": {"beginner": 1.6, "intermediate": 0.3},
            "video": {"beginner": 1.0, "intermediate": 0.4},
            "blog": {"beginner": 0.8, "intermediate": 0.4},
            "repository": {"beginner": 0.0, "intermediate": 1.4},
            "model": {"beginner": 0.0, "intermediate": 1.0},
            "dataset": {"beginner": 0.0, "intermediate": 0.8},
            "research_paper": {"beginner": 0.0, "intermediate": 0.2},
        },
        "platform_bonus": {
            "wikipedia": {"beginner": 0.5, "intermediate": 0.4},
            "github": {"beginner": 0.0, "intermediate": 0.8},
            "huggingface": {"beginner": 0.0, "intermediate": 0.5},
        },
    },
    "alignment": {
        "cap": 0.30,
        "beginner_request": {
            "difficulty": {"beginner": 0.20, "intermediate": 0.06, "advanced": -0.24},
            "source_type": {
                "documentation": 0.10,
                "course": 0.10,
                "video": 0.10,
                "blog": 0.05,
                "research_paper": -0.06,
            },
        },
        "advanced_request": {
            "difficulty": {"advanced": 0.12, "intermediate": 0.03, "beginner": -0.10},
            "source_type": {"research_paper": 0.05},
        },
    },
    "consensus": {
        "max_models": 3,
        "max_llm_sources": 20,
        "judge_max_attempts": 2,
        "min_votes_for_judge": 2,
    },
}


# ════════════════════════════════════════════════════════════════════════════
# Typed models returned by the accessors
# ════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ScorerWeights:
    relevance: float
    authority: float
    type_quality: float
    impact: float
    recency: float
    content: float
    support: float
    authors: float


@dataclass(frozen=True)
class ClassifierWeights:
    keyword_beginner: float
    keyword_intermediate: float
    keyword_advanced: float
    content_weight_paper: float
    content_weight_other: float
    beginner_max_complexity: float
    intermediate_max_complexity: float
    fog_min: float
    fog_max: float
    fog_weight: float
    jargon_weight: float
    jargon_density_cap: float
    complex_word_syllables: int
    intro_beginner_weight: float
    intro_intermediate_weight: float
    intro_max_points: int
    confidence_base: float
    confidence_margin_factor: float
    confidence_max: float
    source_type_bonus: dict[str, dict[str, float]]
    platform_bonus: dict[str, dict[str, float]]


@dataclass(frozen=True)
class AlignmentSettings:
    cap: float
    beginner_difficulty: dict[str, float]
    beginner_source_type: dict[str, float]
    advanced_difficulty: dict[str, float]
    advanced_source_type: dict[str, float]


@dataclass(frozen=True)
class ConsensusSettings:
    max_models: int
    max_llm_sources: int
    judge_max_attempts: int
    min_votes_for_judge: int


# ════════════════════════════════════════════════════════════════════════════
# Safe extraction helpers — never raise, always coerce or fall back
# ════════════════════════════════════════════════════════════════════════════
def _get_nested(data: Any, *keys: str, default: Any = None) -> Any:
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_dict(value: Any, default: dict | None = None) -> dict:
    return value if isinstance(value, dict) else (default if default is not None else {})


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _merge_float_map(default: dict, override: Any) -> dict[str, float]:
    merged: dict[str, float] = {k: float(v) for k, v in default.items()}
    if isinstance(override, dict):
        for key, value in override.items():
            merged[key] = _as_float(value, merged.get(key, 0.0))
    return merged


def _merge_bonus_map(default: dict, override: Any) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = {}
    override_map = override if isinstance(override, dict) else {}
    for key, default_inner in default.items():
        override_inner = override_map.get(key, {})
        merged[key] = _merge_float_map(default_inner, override_inner)
    for key, value in override_map.items():
        if key not in merged and isinstance(value, dict):
            merged[key] = {k: _as_float(v, 0.0) for k, v in value.items()}
    return merged


# ════════════════════════════════════════════════════════════════════════════
# RankingConfig
# ════════════════════════════════════════════════════════════════════════════
class RankingConfig:
    def __init__(self, config_path: str | Path | None = None) -> None:
        self._config_path = (
            Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH
        )
        self._data = self._load()
        self._scorer = self._build_scorer()
        self._classifier = self._build_classifier()
        self._alignment = self._build_alignment()
        self._consensus = self._build_consensus()

    @property
    def config_path(self) -> Path:
        return self._config_path

    # ── Typed accessors ─────────────────────────────────────────────────────
    def scorer_weights(self) -> ScorerWeights:
        return self._scorer

    def classifier_weights(self) -> ClassifierWeights:
        return self._classifier

    def alignment_settings(self) -> AlignmentSettings:
        return self._alignment

    def consensus_settings(self) -> ConsensusSettings:
        return self._consensus

    # ── Loading ─────────────────────────────────────────────────────────────
    def _load(self) -> dict:
        defaults = copy.deepcopy(_DEFAULTS)
        raw = self._read_yaml()
        if not raw:
            return defaults
        try:
            merged = _deep_merge(defaults, raw)
            _logger.info(f"Loaded ranking config: {self._config_path}")
            return merged
        except Exception as exc:
            _logger.warning(f"Failed to merge ranking config, using defaults: {exc}")
            return defaults

    def _read_yaml(self) -> dict:
        if not self._config_path.exists():
            _logger.info(
                f"No ranking config at {self._config_path}, using built-in defaults"
            )
            return {}
        try:
            import yaml
        except ImportError:
            _logger.warning(
                "PyYAML not installed - using built-in ranking defaults. "
                "Run: pip install pyyaml"
            )
            return {}
        try:
            with open(self._config_path, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
            if not isinstance(loaded, dict):
                _logger.warning("ranking.yaml is not a mapping - using defaults")
                return {}
            return loaded
        except Exception as exc:
            _logger.warning(
                f"Failed to parse {self._config_path}: {exc} - using defaults"
            )
            return {}

    # ── Builders ────────────────────────────────────────────────────────────
    def _build_scorer(self) -> ScorerWeights:
        section = _as_dict(_get_nested(self._data, "scorer", "weights", default={}))
        d = _DEFAULTS["scorer"]["weights"]
        return ScorerWeights(
            relevance=_as_float(section.get("relevance"), d["relevance"]),
            authority=_as_float(section.get("authority"), d["authority"]),
            type_quality=_as_float(section.get("type_quality"), d["type_quality"]),
            impact=_as_float(section.get("impact"), d["impact"]),
            recency=_as_float(section.get("recency"), d["recency"]),
            content=_as_float(section.get("content"), d["content"]),
            support=_as_float(section.get("support"), d["support"]),
            authors=_as_float(section.get("authors"), d["authors"]),
        )

    def _build_classifier(self) -> ClassifierWeights:
        d = _DEFAULTS["classifier"]
        kw = _as_dict(_get_nested(self._data, "classifier", "keyword_weights", default={}))
        ct = _as_dict(_get_nested(self._data, "classifier", "content", default={}))
        intro = _as_dict(_get_nested(self._data, "classifier", "intro_phrasing", default={}))
        conf = _as_dict(_get_nested(self._data, "classifier", "confidence", default={}))
        stb = _get_nested(self._data, "classifier", "source_type_bonus", default={})
        pb = _get_nested(self._data, "classifier", "platform_bonus", default={})
        return ClassifierWeights(
            keyword_beginner=_as_float(kw.get("beginner"), d["keyword_weights"]["beginner"]),
            keyword_intermediate=_as_float(kw.get("intermediate"), d["keyword_weights"]["intermediate"]),
            keyword_advanced=_as_float(kw.get("advanced"), d["keyword_weights"]["advanced"]),
            content_weight_paper=_as_float(ct.get("weight_paper"), d["content"]["weight_paper"]),
            content_weight_other=_as_float(ct.get("weight_other"), d["content"]["weight_other"]),
            beginner_max_complexity=_as_float(ct.get("beginner_max_complexity"), d["content"]["beginner_max_complexity"]),
            intermediate_max_complexity=_as_float(ct.get("intermediate_max_complexity"), d["content"]["intermediate_max_complexity"]),
            fog_min=_as_float(ct.get("fog_min"), d["content"]["fog_min"]),
            fog_max=_as_float(ct.get("fog_max"), d["content"]["fog_max"]),
            fog_weight=_as_float(ct.get("fog_weight"), d["content"]["fog_weight"]),
            jargon_weight=_as_float(ct.get("jargon_weight"), d["content"]["jargon_weight"]),
            jargon_density_cap=_as_float(ct.get("jargon_density_cap"), d["content"]["jargon_density_cap"]),
            complex_word_syllables=_as_int(ct.get("complex_word_syllables"), d["content"]["complex_word_syllables"]),
            intro_beginner_weight=_as_float(intro.get("beginner_weight"), d["intro_phrasing"]["beginner_weight"]),
            intro_intermediate_weight=_as_float(intro.get("intermediate_weight"), d["intro_phrasing"]["intermediate_weight"]),
            intro_max_points=_as_int(intro.get("max_points"), d["intro_phrasing"]["max_points"]),
            confidence_base=_as_float(conf.get("base"), d["confidence"]["base"]),
            confidence_margin_factor=_as_float(conf.get("margin_factor"), d["confidence"]["margin_factor"]),
            confidence_max=_as_float(conf.get("max"), d["confidence"]["max"]),
            source_type_bonus=_merge_bonus_map(d["source_type_bonus"], stb),
            platform_bonus=_merge_bonus_map(d["platform_bonus"], pb),
        )

    def _build_alignment(self) -> AlignmentSettings:
        d = _DEFAULTS["alignment"]
        cap = _as_float(_get_nested(self._data, "alignment", "cap"), d["cap"])
        beg_diff = _get_nested(self._data, "alignment", "beginner_request", "difficulty", default={})
        beg_st = _get_nested(self._data, "alignment", "beginner_request", "source_type", default={})
        adv_diff = _get_nested(self._data, "alignment", "advanced_request", "difficulty", default={})
        adv_st = _get_nested(self._data, "alignment", "advanced_request", "source_type", default={})
        return AlignmentSettings(
            cap=cap,
            beginner_difficulty=_merge_float_map(d["beginner_request"]["difficulty"], beg_diff),
            beginner_source_type=_merge_float_map(d["beginner_request"]["source_type"], beg_st),
            advanced_difficulty=_merge_float_map(d["advanced_request"]["difficulty"], adv_diff),
            advanced_source_type=_merge_float_map(d["advanced_request"]["source_type"], adv_st),
        )

    def _build_consensus(self) -> ConsensusSettings:
        d = _DEFAULTS["consensus"]
        section = _as_dict(_get_nested(self._data, "consensus", default={}))
        return ConsensusSettings(
            max_models=_as_int(section.get("max_models"), d["max_models"]),
            max_llm_sources=_as_int(section.get("max_llm_sources"), d["max_llm_sources"]),
            judge_max_attempts=_as_int(section.get("judge_max_attempts"), d["judge_max_attempts"]),
            min_votes_for_judge=_as_int(section.get("min_votes_for_judge"), d["min_votes_for_judge"]),
        )


# ════════════════════════════════════════════════════════════════════════════
# Module-level singleton
# ════════════════════════════════════════════════════════════════════════════
_instance: RankingConfig | None = None


def get_ranking_config() -> RankingConfig:
    global _instance
    if _instance is None:
        _instance = RankingConfig()
    return _instance


def reload_ranking_config() -> RankingConfig:
    global _instance
    _instance = RankingConfig()
    return _instance