from __future__ import annotations

import asyncio
import importlib
import inspect
import time
from pathlib import Path
from typing import Any, Iterable

import yaml

from agent.tools.executor import ToolResult
from agent.tools.registry import ToolDefinition, ToolRegistry
from core import constants
from core.config import get_project_root
from core.models import SourcePlatform, SourceType
from core.schemas import SearchQuerySchema
from search.orchestrator import SearchOrchestrator
from search.source_fetcher import load_domain_routing_config
from utils.logger import get_logger
from utils.text import clean_text

_logger = get_logger("agent.tools.search")

_SEARCH_TOOL_TIMEOUT = float(constants.SEARCH_TOOL_TIMEOUT)
_ORCHESTRATOR_TIMEOUT_FLOOR = float(constants.ORCHESTRATOR_TIMEOUT)
_DIRECT_CLIENT_TIMEOUT = float(constants.DIRECT_CLIENT_TIMEOUT)
_SEARCH_TOKEN_COST = int(constants.AGENT_SEARCH_BUDGET)

_MODE_CONFIG_CACHE: dict[str, dict[str, Any]] | None = None


def _load_all_mode_configs() -> dict[str, dict[str, Any]]:
    global _MODE_CONFIG_CACHE

    if _MODE_CONFIG_CACHE is not None:
        return _MODE_CONFIG_CACHE

    configs: dict[str, dict[str, Any]] = {
        mode: dict(values)
        for mode, values in constants.MODE_DEFAULTS.items()
    }

    try:
        path = get_project_root() / "configs" / "settings.yaml"

        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}

            modes = raw.get("modes", {}) if isinstance(raw, dict) else {}

            if isinstance(modes, dict):
                for mode, values in modes.items():
                    if mode == "default_mode":
                        continue

                    if isinstance(values, dict):
                        merged = configs.setdefault(str(mode), {})
                        merged.update(values)
    except Exception:
        pass

    _MODE_CONFIG_CACHE = configs
    return configs


def _resolve_mode(mode: str) -> str:
    text = str(mode or "").strip().lower()

    if text in constants.VALID_MODES:
        return text

    return constants.DEFAULT_MODE


def _mode_config(mode: str) -> dict[str, Any]:
    configs = _load_all_mode_configs()
    resolved = _resolve_mode(mode)

    if resolved not in configs:
        resolved = constants.DEFAULT_MODE

    fallback = constants.MODE_DEFAULTS[constants.DEFAULT_MODE]
    return dict(configs.get(resolved, fallback))


def _mode_timeout(mode_config: dict[str, Any]) -> float:
    try:
        return max(
            5.0,
            min(
                60.0,
                float(
                    mode_config.get(
                        "search_timeout_seconds",
                        constants.SEARCH_REQUEST_TIMEOUT,
                    )
                ),
            ),
        )
    except Exception:
        return float(constants.SEARCH_REQUEST_TIMEOUT)


def _mode_concurrency(mode_config: dict[str, Any]) -> int:
    try:
        return max(
            1,
            min(
                16,
                int(
                    mode_config.get(
                        "concurrency",
                        constants.SEARCH_CONCURRENCY,
                    )
                ),
            ),
        )
    except Exception:
        return int(constants.SEARCH_CONCURRENCY)


def _mode_max_variants(mode_config: dict[str, Any]) -> int:
    try:
        return max(
            1,
            min(
                12,
                int(
                    mode_config.get(
                        "max_query_variants",
                        constants.SEARCH_MAX_QUERY_VARIANTS,
                    )
                ),
            ),
        )
    except Exception:
        return int(constants.SEARCH_MAX_QUERY_VARIANTS)


def _mode_min_sources(mode_config: dict[str, Any]) -> int:
    try:
        return max(
            1,
            int(
                mode_config.get(
                    "min_sources",
                    constants.MIN_SOURCES_BEFORE_FALLBACK,
                )
            ),
        )
    except Exception:
        return int(constants.MIN_SOURCES_BEFORE_FALLBACK)


_RESEARCH_PLATFORMS = [
    "arxiv",
    "openalex",
    "pubmed",
    "europe_pmc",
    "doaj",
    "crossref",
]

_BOOK_PLATFORMS = [
    "open_library",
    "internet_archive",
    "wikibooks",
    "openstax",
]

_COURSE_PLATFORMS = [
    "mit_ocw",
    "libretexts",
    "wikiversity",
]

_CODE_PLATFORMS = [
    "github",
    "huggingface",
]

_EXPLANATION_PLATFORMS = [
    "wikipedia",
]

_ALL_NO_KEY_PLATFORMS = (
    _RESEARCH_PLATFORMS
    + _BOOK_PLATFORMS
    + _COURSE_PLATFORMS
    + _CODE_PLATFORMS
    + _EXPLANATION_PLATFORMS
)

_TOOL_DEFAULT_PLATFORMS = {
    "search_academic": _RESEARCH_PLATFORMS,
    "search_books": _BOOK_PLATFORMS,
    "search_courses": _COURSE_PLATFORMS,
    "search_code_models": _CODE_PLATFORMS,
    "search_explanation": _EXPLANATION_PLATFORMS,
    "search_github": ["github"],
    "search_wikipedia": ["wikipedia"],
    "search_web": _ALL_NO_KEY_PLATFORMS,
}

_PLATFORM_ALIASES = {
    "research": _RESEARCH_PLATFORMS,
    "academic": _RESEARCH_PLATFORMS,
    "papers": _RESEARCH_PLATFORMS,
    "research_papers": _RESEARCH_PLATFORMS,
    "books": _BOOK_PLATFORMS,
    "textbooks": _BOOK_PLATFORMS,
    "book": _BOOK_PLATFORMS,
    "courses": _COURSE_PLATFORMS,
    "course": _COURSE_PLATFORMS,
    "code": _CODE_PLATFORMS,
    "models": _CODE_PLATFORMS,
    "code_models": _CODE_PLATFORMS,
    "explanation": _EXPLANATION_PLATFORMS,
    "basic": _EXPLANATION_PLATFORMS,
    "wikipedia": ["wikipedia"],
    "github": ["github"],
    "huggingface": ["huggingface"],
    "web": _ALL_NO_KEY_PLATFORMS,
    "all": _ALL_NO_KEY_PLATFORMS,
    "docs": (
        _BOOK_PLATFORMS
        + _COURSE_PLATFORMS
        + _EXPLANATION_PLATFORMS
    ),
}

_KNOWN_PLATFORMS = set(_ALL_NO_KEY_PLATFORMS) | {
    "semantic_scholar",
    "core",
    "web",
    "tavily",
    "serper",
    "serpapi",
    "exa",
    "jina",
}

_DEFAULT_SOURCE_TYPES = {
    "arxiv": "research_paper",
    "semantic_scholar": "research_paper",
    "openalex": "research_paper",
    "core": "research_paper",
    "pubmed": "research_paper",
    "europe_pmc": "research_paper",
    "doaj": "research_paper",
    "crossref": "research_paper",
    "github": "repository",
    "huggingface": "model",
    "wikipedia": "documentation",
    "open_library": "book",
    "internet_archive": "book",
    "wikibooks": "book",
    "openstax": "book",
    "mit_ocw": "course",
    "libretexts": "course",
    "wikiversity": "course",
    "web": "other",
    "tavily": "other",
    "serper": "other",
    "serpapi": "other",
    "exa": "other",
    "jina": "documentation",
}

_DEFAULT_TOOL_SOURCE_TYPES = {
    "search_books": ["book", "documentation"],
    "search_courses": ["course"],
    "search_code_models": ["repository", "model"],
    "search_explanation": ["documentation"],
}

_CLIENT_SPECS = {
    "arxiv": ("clients.arxiv_client", "ArxivClient"),
    "semantic_scholar": (
        "clients.semantic_scholar_client",
        "SemanticScholarClient",
    ),
    "openalex": ("clients.openalex_client", "OpenAlexClient"),
    "core": ("clients.core_client", "CoreClient"),
    "github": ("clients.github_client", "GithubClient"),
    "wikipedia": ("clients.wikipedia_client", "WikipediaClient"),
    "huggingface": ("clients.huggingface_client", "HuggingFaceClient"),
    "web": ("clients.web_search_client", "WebSearchClient"),
    "tavily": ("clients.tavily_client", "TavilyClient"),
    "serper": ("clients.serper_client", "SerperClient"),
    "serpapi": ("clients.serpapi_client", "SerpapiClient"),
    "exa": ("clients.exa_client", "ExaClient"),
    "jina": ("clients.jina_client", "JinaClient"),
    "pubmed": ("clients.pubmed_client", "PubmedClient"),
    "europe_pmc": ("clients.europe_pmc_client", "EuropePmcClient"),
    "doaj": ("clients.doaj_client", "DoajClient"),
    "crossref": ("clients.crossref_client", "CrossrefClient"),
    "open_library": ("clients.open_library_client", "OpenLibraryClient"),
    "internet_archive": (
        "clients.internet_archive_client",
        "InternetArchiveClient",
    ),
    "wikibooks": ("clients.wikibooks_client", "WikibooksClient"),
    "openstax": ("clients.openstax_client", "OpenStaxClient"),
    "mit_ocw": ("clients.mit_ocw_client", "MitOcwClient"),
    "libretexts": ("clients.libretexts_client", "LibreTextsClient"),
    "wikiversity": (
        "clients.wikiversity_client",
        "WikiversityClient",
    ),
}

_CLIENT_METHODS = (
    "search",
    "search_query",
    "query",
    "fetch",
    "get_sources",
    "get_results",
    "retrieve",
)

_SOURCES_CONFIG_CACHE: dict[str, Any] | None = None


def _load_sources_config() -> dict[str, Any]:
    global _SOURCES_CONFIG_CACHE

    if _SOURCES_CONFIG_CACHE is not None:
        return _SOURCES_CONFIG_CACHE

    config: dict[str, Any] = {}

    try:
        config_path = (
            Path(__file__).resolve().parent.parent.parent
            / "configs"
            / "sources.yaml"
        )

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}

            if isinstance(raw, dict):
                config = raw
    except Exception:
        config = {}

    _SOURCES_CONFIG_CACHE = config
    return config


def _platform_enabled_in_config(platform: str) -> bool:
    config = _load_sources_config()
    platforms = config.get("platforms", {})

    if not isinstance(platforms, dict):
        return True

    platform_config = platforms.get(platform, {})

    if not isinstance(platform_config, dict):
        return True

    return bool(platform_config.get("enabled", True))


def _filter_usable_platforms(names: list[str]) -> list[str]:
    cleaned: list[str] = []

    for name in names:
        platform = str(name or "").strip().lower()

        if not platform:
            continue

        if not _platform_enabled_in_config(platform):
            continue

        if platform not in cleaned:
            cleaned.append(platform)

    try:
        from utils.platform_health import PlatformHealthTracker

        tracker = PlatformHealthTracker.get_instance()
        return [
            platform for platform in cleaned if tracker.is_usable(platform)
        ]
    except Exception:
        return cleaned


def _domain_routing_enabled() -> bool:
    try:
        return bool(load_domain_routing_config().get("enabled", True))
    except Exception:
        return True


def _domain_registry() -> Any:
    try:
        from core.domain_registry import get_domain_registry

        return get_domain_registry()
    except Exception:
        return None


def _infer_domains_from_text(text: str) -> list[str]:
    registry = _domain_registry()

    if registry is None:
        return []

    lowered = clean_text(text).lower()

    if not lowered:
        return []

    aliases = getattr(registry, "_aliases", {})

    if not isinstance(aliases, dict):
        aliases = {}

    domains: list[str] = []
    tokens = lowered.split()

    for alias, domain in aliases.items():
        alias_text = str(alias or "").strip().lower()
        domain_text = str(domain or "").strip().lower()

        if not alias_text or not domain_text:
            continue

        if " " in alias_text or len(alias_text) >= 3:
            if alias_text in lowered and domain_text not in domains:
                domains.append(domain_text)
        elif len(alias_text) == 2:
            if alias_text in tokens and domain_text not in domains:
                domains.append(domain_text)

    education_hints = (
        "learn",
        "learning",
        "basics",
        "beginner",
        "fundamentals",
        "course",
        "courses",
        "tutorial",
        "tutorials",
        "textbook",
        "textbooks",
        "study",
        "syllabus",
        "curriculum",
    )

    if any(hint in lowered for hint in education_hints):
        if "education" not in domains:
            domains.append("education")

    if any(domain != "general" for domain in domains):
        domains = [domain for domain in domains if domain != "general"]

    if not domains:
        return []

    def priority(domain: str) -> int:
        try:
            definition = registry.get_domain(domain)
        except Exception:
            definition = None

        if definition is None:
            return 999

        return int(getattr(definition, "priority", 999))

    domains.sort(key=priority)

    try:
        max_domains = int(
            load_domain_routing_config().get(
                "max_domains_per_query",
                3,
            )
        )
    except Exception:
        max_domains = 3

    return domains[: max(1, max_domains)]


def _domain_direct_platforms(query: str, tool_name: str) -> list[str]:
    default_platforms = list(_TOOL_DEFAULT_PLATFORMS.get(tool_name, []))
    registry = _domain_registry()

    if registry is None:
        return default_platforms

    domains = _infer_domains_from_text(query)

    if not domains:
        return default_platforms

    try:
        domain_platforms = registry.get_platforms_for_domains(
            domains,
            include_fallback=False,
            include_universal=True,
        )
    except Exception:
        return default_platforms

    if not domain_platforms:
        return default_platforms

    default_set = set(default_platforms)

    preferred = [
        platform
        for platform in domain_platforms
        if platform in default_set
    ]

    if preferred:
        return preferred[:10]

    return list(domain_platforms)[:10]


def _ok(tool_name: str, data: dict[str, Any]) -> ToolResult:
    return ToolResult(
        tool_name=tool_name,
        success=True,
        data=data,
        error=None,
        tokens_used=0,
    )


def _err(tool_name: str, error: Any) -> ToolResult:
    return ToolResult(
        tool_name=tool_name,
        success=False,
        data=None,
        error=str(error or "Search failed"),
        tokens_used=0,
    )


def _bounded_int(
    value: Any,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)

    return max(int(minimum), min(int(maximum), parsed))


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        parts = [piece.strip() for piece in value.split(",")]
    elif isinstance(value, Iterable) and not isinstance(
        value,
        (bytes, bytearray),
    ):
        parts = [str(piece).strip() for piece in value]
    else:
        parts = [str(value).strip()]

    result: list[str] = []
    seen: set[str] = set()

    for part in parts:
        text = clean_text(part).lower()

        if text and text not in seen:
            seen.add(text)
            result.append(text)

    return result


def _platform_names(
    value: Any,
    tool_name: str,
    query: str = "",
) -> list[str]:
    raw_names = _string_list(value)

    if not raw_names:
        specific_tools = {
            "search_github": ["github"],
            "search_wikipedia": ["wikipedia"],
        }

        if tool_name in specific_tools:
            raw_names = list(specific_tools[tool_name])
        elif _domain_routing_enabled():
            return []
        else:
            raw_names = list(
                _TOOL_DEFAULT_PLATFORMS.get(tool_name, [])
            )

    expanded: list[str] = []
    seen: set[str] = set()

    for name in raw_names:
        pieces = _PLATFORM_ALIASES.get(name, [name])

        for piece in pieces:
            text = clean_text(piece).lower()

            if text and text not in seen:
                seen.add(text)
                expanded.append(text)

    return [name for name in expanded if name in _KNOWN_PLATFORMS]


def _enum_for(value: Any, enum_class: Any) -> Any | None:
    if value is None:
        return None

    if isinstance(value, enum_class):
        return value

    text = clean_text(value).lower()

    if not text:
        return None

    for item in enum_class:
        if clean_text(getattr(item, "value", "")).lower() == text:
            return item

        if clean_text(getattr(item, "name", "")).lower() == text:
            return item

    return None


def _platform_objects(names: list[str]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()

    for name in names:
        item = _enum_for(name, SourcePlatform)

        if item is None:
            continue

        key = clean_text(getattr(item, "value", item)).lower()

        if key and key not in seen:
            seen.add(key)
            result.append(item)

    return result


def _source_type_objects(names: list[str]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()

    for name in names:
        item = _enum_for(name, SourceType)

        if item is None:
            continue

        key = clean_text(getattr(item, "value", item)).lower()

        if key and key not in seen:
            seen.add(key)
            result.append(item)

    return result


def _build_schema(
    query: str,
    goal: str,
    level: str,
    max_results: int,
    platform_names: list[str],
    source_type_names: list[str],
) -> SearchQuerySchema:
    base_payload: dict[str, Any] = {
        "topic": query,
        "goal": goal,
        "level": level,
        "max_results": max_results,
    }

    platform_objs = _platform_objects(platform_names)
    source_type_objs = _source_type_objects(source_type_names)

    candidates: list[dict[str, Any]] = []

    if platform_objs or source_type_objs:
        payload = dict(base_payload)

        if platform_objs:
            payload["platforms"] = platform_objs

        if source_type_objs:
            payload["source_types"] = source_type_objs

        candidates.append(payload)

    if platform_names or source_type_names:
        payload = dict(base_payload)

        if platform_names:
            payload["platforms"] = platform_names

        if source_type_names:
            payload["source_types"] = source_type_names

        candidates.append(payload)

    candidates.append(base_payload)

    for payload in candidates:
        try:
            return SearchQuerySchema.model_validate(payload)
        except Exception:
            continue

    return SearchQuerySchema(
        topic=query,
        goal=goal,
        level=level,
        max_results=max_results,
    )


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, set):
        return list(value)

    return [value]


def _clean_strings(value: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for item in _as_list(value):
        text = clean_text(item)

        if text and text not in seen:
            seen.add(text)
            result.append(text)

    return result


def _model_to_dict(obj: Any) -> dict[str, Any] | None:
    if obj is None:
        return None

    if isinstance(obj, dict):
        return dict(obj)

    model_dump = getattr(obj, "model_dump", None)

    if callable(model_dump):
        try:
            data = model_dump(
                mode="json",
                exclude_none=False,
                by_alias=False,
            )

            if isinstance(data, dict):
                return data
        except Exception:
            pass

    dict_method = getattr(obj, "dict", None)

    if callable(dict_method):
        try:
            data = dict_method()

            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return None


def _first_value(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]

    return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except Exception:
        return None


def _default_source_type(platform: str) -> str:
    return _DEFAULT_SOURCE_TYPES.get(
        clean_text(platform).lower(),
        "other",
    )


def _normalize_source_dict(
    raw: Any,
    platform_name: str = "",
) -> dict[str, Any] | None:
    data = _model_to_dict(raw)

    if data is None:
        if isinstance(raw, str):
            data = {"title": raw, "url": ""}
        else:
            nested = _get(raw, "source", None)
            data = _model_to_dict(nested)

    if data is None:
        return None

    title = clean_text(
        _first_value(
            data,
            ("title", "name", "heading", "document_title"),
        )
    )

    url = clean_text(
        _first_value(
            data,
            (
                "url",
                "link",
                "source_url",
                "uri",
                "href",
                "landing_page_url",
                "pdf_url",
            ),
        )
    )

    if not url:
        return None

    if not title:
        title = url

    abstract = clean_text(
        _first_value(
            data,
            (
                "abstract",
                "snippet",
                "content",
                "description",
                "summary",
                "text",
                "preview",
                "body",
            ),
        )
    )

    platform = clean_text(
        _first_value(
            data,
            ("platform", "source_platform", "provider"),
        )
    ).lower()

    if not platform:
        platform = clean_text(platform_name).lower() or "web"

    source_type = clean_text(
        _first_value(data, ("source_type", "type", "kind"))
    ).lower()

    if not source_type:
        source_type = _default_source_type(platform)

    source_id = clean_text(
        _first_value(
            data,
            ("source_id", "id", "doc_id", "paper_id", "node_id"),
        )
    )

    if not source_id:
        source_id = url

    year = _int_or_none(
        _first_value(
            data,
            ("year", "published_year", "publication_year"),
        )
    )
    citation_count = _int_or_none(
        _first_value(
            data,
            ("citation_count", "citations", "reference_count"),
        )
    )

    metadata = data.get("metadata")

    if not isinstance(metadata, dict):
        metadata = {}

    difficulty = (
        clean_text(_first_value(data, ("difficulty", "level"))).lower()
        or None
    )

    return {
        "source_id": source_id,
        "title": title,
        "url": url,
        "platform": platform,
        "source_type": source_type,
        "abstract": abstract,
        "year": year,
        "citation_count": citation_count,
        "difficulty": difficulty,
        "metadata": metadata,
    }


def _extract_sources_any(
    value: Any,
    platform_name: str = "",
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []

    if value is None:
        return sources

    if isinstance(value, (list, tuple, set)):
        for item in value:
            sources.extend(_extract_sources_any(item, platform_name))

        return sources

    if isinstance(value, dict):
        for key in (
            "sources",
            "results",
            "items",
            "data",
            "documents",
            "papers",
            "repositories",
            "books",
            "courses",
        ):
            nested = value.get(key)

            if nested is not None:
                sources.extend(
                    _extract_sources_any(nested, platform_name)
                )

        normalized = _normalize_source_dict(value, platform_name)

        if normalized is not None:
            sources.append(normalized)

        return sources

    nested_source = _get(value, "source", None)

    if nested_source is not None and nested_source is not value:
        sources.extend(
            _extract_sources_any(nested_source, platform_name)
        )

    for key in (
        "sources",
        "results",
        "items",
        "data",
        "documents",
        "papers",
        "repositories",
        "books",
        "courses",
    ):
        nested = _get(value, key, None)

        if nested is not None:
            sources.extend(
                _extract_sources_any(nested, platform_name)
            )

    normalized = _normalize_source_dict(value, platform_name)

    if normalized is not None:
        sources.append(normalized)

    return sources


def _dedupe_sources(
    sources: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []

    for source in sources:
        key = clean_text(
            source.get("source_id")
            or source.get("url")
            or source.get("title")
            or ""
        ).lower()

        if not key or key in seen:
            continue

        seen.add(key)
        result.append(source)

        if len(result) >= max(1, int(limit)):
            break

    return result


def _metadata_from_raw(raw: Any) -> dict[str, Any]:
    data = _model_to_dict(raw) or {}

    return {
        "expanded_queries": _clean_strings(
            _first_value(data, ("expanded_queries", "queries"))
        ),
        "corrected_topic": clean_text(
            _first_value(data, ("corrected_topic", "topic"))
        ),
        "primary_concept": clean_text(
            _first_value(data, ("primary_concept", "concept"))
        ),
        "keywords": _clean_strings(
            _first_value(data, ("keywords", "terms"))
        ),
        "intent": clean_text(
            _first_value(data, ("intent", "query_intent"))
        ),
        "detected_level": clean_text(
            _first_value(data, ("detected_level", "level"))
        ),
        "errors": _clean_strings(_first_value(data, ("errors",))),
        "warnings": _clean_strings(_first_value(data, ("warnings",))),
        "platform_failures": _clean_strings(
            _first_value(
                data,
                ("platform_failures", "failed_platforms"),
            )
        ),
    }


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value

    return value


async def _close_client(client: Any) -> None:
    for name in ("close", "aclose", "shutdown"):
        method = getattr(client, name, None)

        if callable(method):
            try:
                await _maybe_await(method())
            except Exception:
                pass

    return


def _load_client_class(module_name: str, class_name: str) -> Any:
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name, None)

    if isinstance(cls, type):
        return cls

    for attr in vars(module).values():
        if (
            isinstance(attr, type)
            and attr.__name__.endswith("Client")
            and attr.__name__ not in {"BaseClient", "Client"}
        ):
            return attr

    raise RuntimeError(f"Client class not found in {module_name}")


async def _direct_platform_search(
    query: str,
    max_results: int,
    platform_name: str,
) -> list[dict[str, Any]]:
    spec = _CLIENT_SPECS.get(platform_name)

    if spec is None:
        return []

    module_name, class_name = spec

    try:
        client_class = _load_client_class(module_name, class_name)
        client = client_class()
    except Exception as exc:
        _logger.debug(
            f"Direct client init failed for {platform_name}: {exc}"
        )
        return []

    try:
        for method_name in _CLIENT_METHODS:
            method = getattr(client, method_name, None)

            if not callable(method):
                continue

            attempts = (
                lambda: method(query, max_results=max_results),
                lambda: method(query, limit=max_results),
                lambda: method(query, top_k=max_results),
                lambda: method(query),
                lambda: method(q=query, max_results=max_results),
                lambda: method(
                    search_query=query,
                    max_results=max_results,
                ),
                lambda: method(topic=query, max_results=max_results),
            )

            for attempt in attempts:
                try:
                    raw = await asyncio.wait_for(
                        _maybe_await(attempt()),
                        timeout=_DIRECT_CLIENT_TIMEOUT,
                    )

                    sources = _extract_sources_any(raw, platform_name)

                    if sources:
                        return _dedupe_sources(sources, max_results)
                except TypeError:
                    continue
                except asyncio.TimeoutError:
                    break
                except Exception as exc:
                    _logger.debug(
                        f"Direct client method failed for "
                        f"{platform_name}.{method_name}: {exc}"
                    )
                    break
    finally:
        await _close_client(client)

    return []


async def _direct_search(
    query: str,
    max_results: int,
    platform_names: list[str],
) -> list[dict[str, Any]]:
    platform_names = _filter_usable_platforms(platform_names)

    if not platform_names:
        return []

    tasks = [
        _direct_platform_search(query, max_results, platform_name)
        for platform_name in platform_names[:10]
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    sources: list[dict[str, Any]] = []

    for result in results:
        if isinstance(result, list):
            sources.extend(result)

    return _dedupe_sources(sources, max_results)


async def _search_with_orchestrator(
    schema: SearchQuerySchema,
    platform_names: list[str],
    mode_config: dict[str, Any] | None = None,
) -> Any:
    platforms = _platform_objects(platform_names) or None

    resolved = mode_config or {}

    request_timeout = _mode_timeout(resolved)
    concurrency = _mode_concurrency(resolved)
    max_variants = _mode_max_variants(resolved)

    async with SearchOrchestrator(
        platforms=platforms,
        concurrency=concurrency,
        max_query_variants=max_variants,
        request_timeout=request_timeout,
    ) as orchestrator:
        resolved_mode = _resolve_mode(
            str(resolved.get("mode", "") or "")
        )

        search_method = getattr(orchestrator, "search", None)

        if callable(search_method):
            try:
                return await asyncio.wait_for(
                    search_method(schema, mode=resolved_mode),
                    timeout=request_timeout
                    + _ORCHESTRATOR_TIMEOUT_FLOOR,
                )
            except TypeError:
                return await asyncio.wait_for(
                    search_method(schema),
                    timeout=request_timeout
                    + _ORCHESTRATOR_TIMEOUT_FLOOR,
                )

        run_method = getattr(orchestrator, "run", None)

        if callable(run_method):
            try:
                return await asyncio.wait_for(
                    run_method(schema, mode=resolved_mode),
                    timeout=request_timeout
                    + _ORCHESTRATOR_TIMEOUT_FLOOR,
                )
            except TypeError:
                return await asyncio.wait_for(
                    run_method(schema),
                    timeout=request_timeout
                    + _ORCHESTRATOR_TIMEOUT_FLOOR,
                )

        raise RuntimeError(
            "SearchOrchestrator has no search or run method"
        )


async def _run_search(
    tool_name: str,
    query: str,
    max_results: int,
    level: str,
    goal: str,
    platforms: Any,
    source_types: Any,
    mode: str = "",
) -> ToolResult:
    started = time.perf_counter()
    cleaned_query = clean_text(query)

    if not cleaned_query:
        return _err(tool_name, "Empty search query")

    resolved_mode = _resolve_mode(mode)
    mode_config = _mode_config(resolved_mode)
    mode_min_sources = _mode_min_sources(mode_config)

    limit = _bounded_int(max_results, 8, 1, 50)

    platform_names = _platform_names(
        platforms,
        tool_name,
        cleaned_query,
    )
    source_type_names = _string_list(source_types)

    if not source_type_names:
        source_type_names = list(
            _DEFAULT_TOOL_SOURCE_TYPES.get(tool_name, [])
        )

    schema = _build_schema(
        query=cleaned_query,
        goal=clean_text(goal),
        level=clean_text(level),
        max_results=limit,
        platform_names=platform_names,
        source_type_names=source_type_names,
    )

    sources: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    fallback_used = False

    try:
        mode_config_with_mode = dict(mode_config)
        mode_config_with_mode["mode"] = resolved_mode

        raw = await _search_with_orchestrator(
            schema, platform_names, mode_config_with_mode
        )

        sources = _extract_sources_any(raw)
        metadata = _metadata_from_raw(raw)
    except Exception as exc:
        metadata["orchestrator_error"] = str(exc)
        _logger.warning(
            f"Orchestrator search failed for {tool_name}: {exc}"
        )

    if len(sources) < mode_min_sources:
        fallback_platform_names = (
            platform_names
            or _domain_direct_platforms(cleaned_query, tool_name)
        )

        fallback_platform_names = _filter_usable_platforms(
            fallback_platform_names
        )

        fallback_sources = await _direct_search(
            cleaned_query,
            limit,
            fallback_platform_names,
        )

        if fallback_sources:
            merged = _dedupe_sources(
                [*sources, *fallback_sources], limit
            )
            sources = merged
            fallback_used = True
            metadata["fallback_used"] = True

    if not sources and metadata.get("orchestrator_error"):
        return _err(tool_name, metadata["orchestrator_error"])

    data = {
        "sources": sources[:limit],
        "total_sources": len(sources),
        "expanded_queries": metadata.get("expanded_queries", []),
        "corrected_topic": metadata.get(
            "corrected_topic",
            cleaned_query,
        ),
        "primary_concept": metadata.get("primary_concept", ""),
        "keywords": metadata.get("keywords", []),
        "intent": metadata.get("intent", ""),
        "detected_level": metadata.get(
            "detected_level",
            clean_text(level),
        ),
        "errors": metadata.get("errors", []),
        "warnings": metadata.get("warnings", []),
        "platform_failures": metadata.get("platform_failures", []),
        "fallback_used": fallback_used,
        "mode": resolved_mode,
        "latency_ms": round(
            (time.perf_counter() - started) * 1000,
            1,
        ),
    }

    return _ok(tool_name, data)


async def search_academic(
    query: str = "",
    max_results: int = 8,
    level: str = "",
    goal: str = "",
    platforms: Any = None,
    source_types: Any = None,
    mode: str = "",
    **_: Any,
) -> ToolResult:
    return await _run_search(
        tool_name="search_academic",
        query=query,
        max_results=max_results,
        level=level,
        goal=goal,
        platforms=platforms,
        source_types=source_types,
        mode=mode,
    )


async def search_books(
    query: str = "",
    max_results: int = 8,
    level: str = "",
    goal: str = "",
    platforms: Any = None,
    source_types: Any = None,
    mode: str = "",
    **_: Any,
) -> ToolResult:
    return await _run_search(
        tool_name="search_books",
        query=query,
        max_results=max_results,
        level=level,
        goal=goal,
        platforms=platforms,
        source_types=source_types,
        mode=mode,
    )


async def search_courses(
    query: str = "",
    max_results: int = 8,
    level: str = "",
    goal: str = "",
    platforms: Any = None,
    source_types: Any = None,
    mode: str = "",
    **_: Any,
) -> ToolResult:
    return await _run_search(
        tool_name="search_courses",
        query=query,
        max_results=max_results,
        level=level,
        goal=goal,
        platforms=platforms,
        source_types=source_types,
        mode=mode,
    )


async def search_code_models(
    query: str = "",
    max_results: int = 8,
    level: str = "",
    goal: str = "",
    platforms: Any = None,
    source_types: Any = None,
    mode: str = "",
    **_: Any,
) -> ToolResult:
    return await _run_search(
        tool_name="search_code_models",
        query=query,
        max_results=max_results,
        level=level,
        goal=goal,
        platforms=platforms,
        source_types=source_types,
        mode=mode,
    )


async def search_explanation(
    query: str = "",
    max_results: int = 8,
    level: str = "",
    goal: str = "",
    platforms: Any = None,
    source_types: Any = None,
    mode: str = "",
    **_: Any,
) -> ToolResult:
    return await _run_search(
        tool_name="search_explanation",
        query=query,
        max_results=max_results,
        level=level,
        goal=goal,
        platforms=platforms,
        source_types=source_types,
        mode=mode,
    )


async def search_web(
    query: str = "",
    max_results: int = 8,
    level: str = "",
    goal: str = "",
    platforms: Any = None,
    source_types: Any = None,
    mode: str = "",
    **_: Any,
) -> ToolResult:
    return await _run_search(
        tool_name="search_web",
        query=query,
        max_results=max_results,
        level=level,
        goal=goal,
        platforms=platforms,
        source_types=source_types,
        mode=mode,
    )


async def search_github(
    query: str = "",
    max_results: int = 8,
    level: str = "",
    goal: str = "",
    platforms: Any = None,
    source_types: Any = None,
    mode: str = "",
    **_: Any,
) -> ToolResult:
    return await _run_search(
        tool_name="search_github",
        query=query,
        max_results=max_results,
        level=level,
        goal=goal,
        platforms=platforms or ["github"],
        source_types=source_types,
        mode=mode,
    )


async def search_wikipedia(
    query: str = "",
    max_results: int = 8,
    level: str = "",
    goal: str = "",
    platforms: Any = None,
    source_types: Any = None,
    mode: str = "",
    **_: Any,
) -> ToolResult:
    return await _run_search(
        tool_name="search_wikipedia",
        query=query,
        max_results=max_results,
        level=level,
        goal=goal,
        platforms=platforms or ["wikipedia"],
        source_types=source_types,
        mode=mode,
    )


def register_search_tools(registry: ToolRegistry) -> None:
    common_schema = {
        "query": "string",
        "max_results": "integer",
        "level": "string",
        "goal": "string",
        "platforms": "array",
        "source_types": "array",
        "mode": "string",
    }

    registry.register(
        ToolDefinition(
            name="search_academic",
            description=(
                "Search research papers from arXiv, OpenAlex, PubMed, "
                "Europe PMC, DOAJ, and Crossref"
            ),
            category="search",
            token_cost_est=_SEARCH_TOKEN_COST,
            timeout_seconds=_SEARCH_TOOL_TIMEOUT,
            requires_llm=False,
            parameters_schema=common_schema,
        ),
        search_academic,
    )

    registry.register(
        ToolDefinition(
            name="search_books",
            description=(
                "Search books and textbooks from Open Library, "
                "Internet Archive, Wikibooks, and OpenStax"
            ),
            category="search",
            token_cost_est=_SEARCH_TOKEN_COST,
            timeout_seconds=_SEARCH_TOOL_TIMEOUT,
            requires_llm=False,
            parameters_schema=common_schema,
        ),
        search_books,
    )

    registry.register(
        ToolDefinition(
            name="search_courses",
            description=(
                "Search courses from MIT OCW, LibreTexts, and "
                "Wikiversity"
            ),
            category="search",
            token_cost_est=_SEARCH_TOKEN_COST,
            timeout_seconds=_SEARCH_TOOL_TIMEOUT,
            requires_llm=False,
            parameters_schema=common_schema,
        ),
        search_courses,
    )

    registry.register(
        ToolDefinition(
            name="search_code_models",
            description=(
                "Search code and models from GitHub and Hugging Face"
            ),
            category="search",
            token_cost_est=_SEARCH_TOKEN_COST,
            timeout_seconds=_SEARCH_TOOL_TIMEOUT,
            requires_llm=False,
            parameters_schema=common_schema,
        ),
        search_code_models,
    )

    registry.register(
        ToolDefinition(
            name="search_explanation",
            description="Search basic explanations from Wikipedia",
            category="search",
            token_cost_est=_SEARCH_TOKEN_COST,
            timeout_seconds=_SEARCH_TOOL_TIMEOUT,
            requires_llm=False,
            parameters_schema=common_schema,
        ),
        search_explanation,
    )

    registry.register(
        ToolDefinition(
            name="search_web",
            description=(
                "Search all enabled no-key research, book, course, "
                "code, and explanation sources"
            ),
            category="search",
            token_cost_est=_SEARCH_TOKEN_COST,
            timeout_seconds=_SEARCH_TOOL_TIMEOUT,
            requires_llm=False,
            parameters_schema=common_schema,
        ),
        search_web,
    )

    registry.register(
        ToolDefinition(
            name="search_github",
            description=(
                "Search GitHub for code repositories related to the "
                "topic"
            ),
            category="search",
            token_cost_est=_SEARCH_TOKEN_COST,
            timeout_seconds=_SEARCH_TOOL_TIMEOUT,
            requires_llm=False,
            parameters_schema=common_schema,
        ),
        search_github,
    )

    registry.register(
        ToolDefinition(
            name="search_wikipedia",
            description=(
                "Search Wikipedia for documentation and introductory "
                "content"
            ),
            category="search",
            token_cost_est=_SEARCH_TOKEN_COST,
            timeout_seconds=_SEARCH_TOOL_TIMEOUT,
            requires_llm=False,
            parameters_schema=common_schema,
        ),
        search_wikipedia,
    )