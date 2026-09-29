from __future__ import annotations

import asyncio
import importlib
import importlib.util
import math
from pathlib import Path
from typing import Any

import yaml

from core import constants
from core.config import get_project_root
from core.models import Source
from utils.async_helpers import run_with_timeout
from utils.logger import get_logger, get_trace_logger
from utils.text import clean_text

_SOURCES_CONFIG_CACHE: dict[str, Any] | None = None
_DOMAIN_ROUTING_CONFIG_CACHE: dict[str, Any] | None = None
_MODE_CONFIG_CACHE: dict[str, dict[str, Any]] | None = None

_DOMAIN_ROUTING_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "min_sources_before_domain_expansion": 8,
    "max_domains_per_query": 3,
    "min_confidence": 0.35,
    "use_universal_when_low_confidence": True,
    "use_universal_when_low_sources": True,
    "respect_explicit_platforms": True,
    "allow_domain_fallback": True,
    "max_domain_platforms": 16,
}

_LEVEL_PROFILE_BEGINNER = "beginner"
_LEVEL_PROFILE_MIXED = "mixed"
_LEVEL_PROFILE_ADVANCED = "advanced"
_LEVEL_PROFILE_BALANCED = "balanced"

_VALID_LEVEL_PROFILES = {
    _LEVEL_PROFILE_BEGINNER,
    _LEVEL_PROFILE_MIXED,
    _LEVEL_PROFILE_ADVANCED,
    _LEVEL_PROFILE_BALANCED,
}

_MIN_QUERIES_PER_PLATFORM = 2

_FETCH_STATUS_OK = "ok"
_FETCH_STATUS_EMPTY = "empty"
_FETCH_STATUS_RATE_LIMITED = "rate_limited"
_FETCH_STATUS_FAILED = "failed"
_FETCH_STATUS_SKIPPED = "skipped"


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


def load_domain_routing_config() -> dict[str, Any]:
    global _DOMAIN_ROUTING_CONFIG_CACHE

    if _DOMAIN_ROUTING_CONFIG_CACHE is not None:
        return dict(_DOMAIN_ROUTING_CONFIG_CACHE)

    config = dict(_DOMAIN_ROUTING_DEFAULTS)

    try:
        from core.config import get_settings

        settings = get_settings()
        section = getattr(settings, "domain_routing", None)

        if isinstance(section, dict):
            for key in config:
                if section.get(key) is not None:
                    config[key] = section[key]
        else:
            for key in config:
                value = getattr(
                    settings, f"domain_routing_{key}", None
                )

                if value is not None:
                    config[key] = value
    except Exception:
        pass

    try:
        config_path = (
            Path(__file__).resolve().parent.parent
            / "configs"
            / "settings.yaml"
        )

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}

            if isinstance(loaded, dict):
                section = loaded.get("domain_routing", {})

                if isinstance(section, dict):
                    for key in config:
                        if section.get(key) is not None:
                            config[key] = section[key]
    except Exception:
        pass

    _DOMAIN_ROUTING_CONFIG_CACHE = config
    return dict(config)


def _load_sources_yaml() -> dict[str, Any]:
    global _SOURCES_CONFIG_CACHE

    if _SOURCES_CONFIG_CACHE is not None:
        return _SOURCES_CONFIG_CACHE

    loaded: dict[str, Any] = {}

    try:
        config_path = (
            Path(__file__).resolve().parent.parent
            / "configs"
            / "sources.yaml"
        )

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}

            if isinstance(raw, dict):
                loaded = raw
    except Exception:
        loaded = {}

    _SOURCES_CONFIG_CACHE = loaded
    return loaded


def _get_sources_config() -> dict[str, Any]:
    return _load_sources_yaml()


def _new_outcome() -> dict[str, Any]:
    return {
        "attempted": False,
        "succeeded": False,
        "empty": False,
        "failed": False,
        "rate_limited": False,
        "error_count": 0,
        "source_count": 0,
        "query_count": 0,
        "errors": [],
        "status": _FETCH_STATUS_SKIPPED,
        "reason_if_skipped": None,
    }


class SourceFetcher:
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
        "huggingface": (
            "clients.huggingface_client",
            "HuggingFaceClient",
        ),
        "web": ("clients.web_search_client", "WebSearchClient"),
        "tavily": ("clients.tavily_client", "TavilyClient"),
        "serper": ("clients.serper_client", "SerperClient"),
        "serpapi": ("clients.serpapi_client", "SerpapiClient"),
        "exa": ("clients.exa_client", "ExaClient"),
        "jina": ("clients.jina_client", "JinaClient"),
        "crossref": ("clients.crossref_client", "CrossrefClient"),
        "europe_pmc": (
            "clients.europe_pmc_client",
            "EuropePmcClient",
        ),
        "pubmed": ("clients.pubmed_client", "PubMedClient"),
        "doaj": ("clients.doaj_client", "DoajClient"),
        "zenodo": ("clients.zenodo_client", "ZenodoClient"),
        "open_library": (
            "clients.open_library_client",
            "OpenLibraryClient",
        ),
        "internet_archive": (
            "clients.internet_archive_client",
            "InternetArchiveClient",
        ),
        "wikibooks": ("clients.wikibooks_client", "WikibooksClient"),
        "wikiversity": (
            "clients.wikiversity_client",
            "WikiversityClient",
        ),
        "openstax": ("clients.openstax_client", "OpenStaxClient"),
        "mit_ocw": ("clients.mit_ocw_client", "MitOcwClient"),
        "libretexts": (
            "clients.libretexts_client",
            "LibreTextsClient",
        ),
    }

    _PLATFORM_HINTS: dict[str, tuple[str, ...]] = {
        "arxiv": ("arxiv",),
        "semantic_scholar": ("semantic scholar",),
        "openalex": ("openalex",),
        "core": ("core.ac", "open access paper"),
        "github": ("github", "source code", "codebase"),
        "wikipedia": ("wikipedia",),
        "huggingface": ("huggingface", "hugging face"),
        "web": (),
        "tavily": (),
        "exa": (),
        "serper": (),
        "serpapi": (),
        "jina": (),
        "crossref": (
            "crossref",
            "doi",
            "journal",
            "paper",
            "citation",
        ),
        "europe_pmc": (
            "europe pmc",
            "pmc",
            "biomedical",
            "medicine",
            "life sciences",
        ),
        "pubmed": ("pubmed", "medline", "medical", "clinical"),
        "doaj": ("doaj", "open access journal", "journal"),
        "zenodo": (
            "zenodo",
            "dataset",
            "research data",
            "software",
        ),
        "open_library": (
            "open library",
            "book",
            "books",
            "textbook",
            "isbn",
        ),
        "internet_archive": (
            "internet archive",
            "archive.org",
            "book",
            "text",
        ),
        "wikibooks": ("wikibooks", "textbook", "tutorial"),
        "wikiversity": ("wikiversity", "course", "learning"),
        "openstax": ("openstax", "textbook", "course"),
        "mit_ocw": ("mit opencourseware", "mit ocw", "course"),
        "libretexts": ("libretexts", "textbook", "course"),
    }

    _DEFAULT_CATEGORY_PLATFORMS: dict[str, list[str]] = {
        "research": [
            "arxiv",
            "openalex",
            "crossref",
            "europe_pmc",
            "pubmed",
            "doaj",
            "zenodo",
            "semantic_scholar",
            "core",
        ],
        "books": [
            "open_library",
            "internet_archive",
            "wikibooks",
            "openstax",
        ],
        "courses": [
            "wikiversity",
            "openstax",
            "mit_ocw",
            "libretexts",
        ],
        "code": [
            "github",
            "huggingface",
            "zenodo",
        ],
        "explanation": [
            "wikipedia",
        ],
    }

    _SOURCE_TYPE_CATEGORY_MAP: dict[str, str] = {
        "research_paper": "research",
        "paper": "research",
        "book": "books",
        "documentation": "books",
        "course": "courses",
        "repository": "code",
        "model": "code",
        "dataset": "code",
        "other": "explanation",
    }

    _DEFAULT_CATEGORY_FALLBACK_ORDER = [
        "research",
        "books",
        "courses",
        "code",
        "explanation",
    ]

    _WEB_SEARCH_PLATFORMS = {
        "web",
        "tavily",
        "exa",
        "serper",
        "serpapi",
        "jina",
    }

    _BOOK_PLATFORMS = {
        "open_library",
        "internet_archive",
        "wikibooks",
        "openstax",
    }

    _COURSE_PLATFORMS = {
        "wikiversity",
        "openstax",
        "mit_ocw",
        "libretexts",
    }

    _EXPLANATION_PLATFORMS = {
        "wikipedia",
    }

    _ACADEMIC_PLATFORMS = {
        "arxiv",
        "semantic_scholar",
        "openalex",
        "core",
        "crossref",
        "europe_pmc",
        "pubmed",
        "doaj",
        "zenodo",
    }

    _RESEARCH_ONLY_PLATFORMS = {
        "arxiv",
        "semantic_scholar",
        "openalex",
        "crossref",
        "europe_pmc",
        "pubmed",
        "doaj",
    }

    _BEGINNER_PLATFORMS = {
        "wikipedia",
        "wikibooks",
        "wikiversity",
        "openstax",
        "mit_ocw",
        "libretexts",
        "open_library",
        "internet_archive",
    }

    _CODE_PLATFORMS = {
        "github",
        "huggingface",
    }

    _REFERENCE_PLATFORMS = {
        "zenodo",
    }

    _CODE_HINTS = (
        "github",
        "source code",
        "codebase",
        "implementation",
        "repository",
    )

    _NON_ACADEMIC_HINTS = (
        "video",
        "course",
        "documentation",
        "getting started",
        "blog",
        "website",
        "wikipedia",
        "huggingface",
        "github",
    )

    _BOOK_HINTS = (
        "book",
        "books",
        "textbook",
        "isbn",
        "open library",
        "internet archive",
        "wikibooks",
        "openstax",
    )

    _COURSE_HINTS = (
        "course",
        "courses",
        "tutorial",
        "learning",
        "wikiversity",
        "openstax",
        "mit ocw",
        "libretexts",
    )

    _EXPLANATION_HINTS = (
        "wikipedia",
        "what is",
        "explained",
        "introduction",
        "basics",
        "beginner",
        "overview",
    )

    _DEFAULT_KEY_REQUIRED_PLATFORMS = {
        "tavily",
        "exa",
        "core",
        "serper",
        "serpapi",
        "jina",
        "semantic_scholar",
    }

    _MODULE_AVAILABILITY_CACHE: dict[str, bool] | None = None

    def __init__(
        self,
        platforms: list[Any] | None = None,
        concurrency: int | None = None,
        max_query_variants: int | None = None,
        request_timeout: float | None = None,
        validate_registry: bool = True,
    ) -> None:
        self._clients: dict[str, Any] = {}
        self._skipped_clients: dict[str, str] = {}
        self._broken_clients: dict[str, str] = {}
        self._health_tracker: Any = None
        self._outcomes: dict[str, dict[str, Any]] = {}
        self._last_active_platforms: list[str] = []

        self._logger = get_logger("search.source_fetcher")
        self._trace = get_trace_logger()

        resolved_concurrency = (
            concurrency
            if concurrency is not None
            else self._config_value(
                "concurrency",
                getattr(constants, "SEARCH_CONCURRENCY", 6),
            )
        )

        resolved_variants = (
            max_query_variants
            if max_query_variants is not None
            else self._config_value(
                "max_query_variants",
                getattr(
                    constants, "SEARCH_MAX_QUERY_VARIANTS", 6
                ),
            )
        )

        resolved_timeout = (
            request_timeout
            if request_timeout is not None
            else self._config_value(
                "request_timeout_seconds",
                getattr(constants, "SEARCH_REQUEST_TIMEOUT", 15.0),
            )
        )

        self._concurrency = max(1, min(8, int(resolved_concurrency)))
        self._max_query_variants = max(
            1, min(8, int(resolved_variants))
        )
        self._request_timeout = max(
            10.0, min(60.0, float(resolved_timeout))
        )

        self._category_fallback_enabled = bool(
            self._config_value(
                "category_fallback_enabled",
                getattr(
                    constants, "CATEGORY_FALLBACK_ENABLED", True
                ),
            )
        )

        self._min_sources_before_fallback = max(
            1,
            int(
                self._config_value(
                    "min_sources_before_fallback",
                    getattr(
                        constants,
                        "MIN_SOURCES_BEFORE_CATEGORY_FALLBACK",
                        10,
                    ),
                )
            ),
        )

        self._category_fallback_order = (
            self._category_fallback_order_from_config()
        )

        self.last_platform_failures: list[str] = []
        self.platform_failures: list[str] = []
        self.platform_warnings: list[str] = []
        self.failed_platforms: list[str] = []
        self.last_platform_errors: dict[str, list[str]] = {}
        self.last_fetch_report: dict[str, Any] = {}

        if validate_registry:
            self._validate_registry()

        self._default_platforms = (
            self._resolve_platforms(platforms)
            if platforms is not None
            else self._default_pool()
        )

    async def __aenter__(self) -> SourceFetcher:
        return self

    async def __aexit__(
        self, exc_type: Any, exc: Any, tb: Any
    ) -> bool:
        await self.close()
        return False

    async def close(self) -> None:
        for client in list(self._clients.values()):
            try:
                await client.close()
            except Exception:
                pass

        self._clients.clear()

    @classmethod
    def _validate_registry(cls) -> None:
        missing: list[tuple[str, str]] = []

        for platform, (
            module_name,
            class_name,
        ) in cls._CLIENT_SPECS.items():
            try:
                spec = importlib.util.find_spec(module_name)
            except Exception:
                spec = None

            if spec is None:
                missing.append((platform, module_name))
                continue

            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue

            client_class = getattr(module, class_name, None)

            if client_class is None:
                candidates = [
                    attr
                    for name, attr in vars(module).items()
                    if isinstance(attr, type)
                    and name.endswith("Client")
                    and callable(getattr(attr, "search", None))
                ]

                if not candidates:
                    missing.append(
                        (platform, f"{module_name}.{class_name}")
                    )

        if missing:
            details = ", ".join(
                f"{platform}->{target}"
                for platform, target in missing
            )
            raise RuntimeError(
                f"Platform registry validation failed: {details}"
            )

    def _get_health_tracker(self) -> Any:
        if self._health_tracker is None:
            try:
                from utils.platform_health import (
                    PlatformHealthTracker,
                )

                self._health_tracker = (
                    PlatformHealthTracker.get_instance()
                )
            except Exception:
                self._health_tracker = False

        if self._health_tracker is False:
            return None

        return self._health_tracker

    def _get_domain_registry(self) -> Any:
        try:
            from core.domain_registry import get_domain_registry

            return get_domain_registry()
        except Exception:
            return None

    def _resolve_domains(self, domains: Any) -> list[str]:
        registry = self._get_domain_registry()

        if registry is not None:
            try:
                return registry.resolve_domains(domains)
            except Exception:
                pass

        if domains is None:
            return []

        if isinstance(domains, str):
            domains = [
                item.strip() for item in domains.split(",")
            ]

        if not isinstance(domains, (list, tuple, set)):
            domains = [domains]

        resolved: list[str] = []

        for domain in domains:
            text = str(domain or "").strip().lower()
            text = text.replace("-", "_").replace(" ", "_")

            if text and text not in resolved:
                resolved.append(text)

        return resolved

    def _domain_platforms(
        self,
        domains: list[str],
        include_fallback: bool = False,
        include_universal: bool = False,
        min_platforms: int | None = None,
        max_platforms: int | None = None,
    ) -> list[str]:
        registry = self._get_domain_registry()

        if registry is None:
            return []

        try:
            return registry.get_platforms_for_domains(
                domains,
                include_fallback=include_fallback,
                include_universal=include_universal,
                min_platforms=min_platforms,
                max_platforms=max_platforms,
            )
        except Exception:
            return []

    def _universal_platforms(self) -> list[str]:
        registry = self._get_domain_registry()

        if registry is None:
            return []

        try:
            return registry.get_universal_platforms()
        except Exception:
            return []

    def category_platforms(self, category: str) -> list[str]:
        category = str(category or "").strip().lower()

        config_categories = self._config_value(
            "category_platforms", None
        )

        if isinstance(config_categories, dict):
            configured = config_categories.get(category)

            if isinstance(configured, list):
                return self._clean_platform_list(configured)

        return self._clean_platform_list(
            self._DEFAULT_CATEGORY_PLATFORMS.get(category, [])
        )

    def category_fallback_platforms(
        self,
        active_platforms: list[Any] | None = None,
        source_types: list[Any] | None = None,
        target_count: int | None = None,
    ) -> list[str]:
        active_keys = self._resolve_platforms(active_platforms)
        current_categories = self.categories_for_platforms(
            active_keys
        )
        requested_categories = self._categories_for_source_types(
            source_types
        )
        order = list(self._category_fallback_order)

        desired_categories: list[str] = []

        if requested_categories:
            for category in order:
                if (
                    category in requested_categories
                    and category not in desired_categories
                ):
                    desired_categories.append(category)

        for category in order:
            if (
                category not in current_categories
                and category not in desired_categories
            ):
                desired_categories.append(category)

        for category in order:
            if category not in desired_categories:
                desired_categories.append(category)

        fallback: list[str] = []
        desired_count = max(
            3,
            int(
                target_count
                or self._min_sources_before_fallback
            ),
        )

        for category in desired_categories:
            for platform in self.category_platforms(category):
                if platform in active_keys:
                    continue

                if platform in fallback:
                    continue

                is_usable, _ = self._platform_usable(platform)

                if not is_usable:
                    continue

                fallback.append(platform)

                if len(fallback) >= desired_count:
                    break

        return fallback

    def categories_for_platforms(
        self, platforms: list[Any]
    ) -> list[str]:
        platform_keys = self._resolve_platforms(platforms)
        categories: list[str] = []
        reverse_map: dict[str, list[str]] = {}

        for category in self._category_fallback_order:
            for platform in self.category_platforms(category):
                reverse_map.setdefault(platform, []).append(category)

        for platform in platform_keys:
            for category in reverse_map.get(platform, []):
                if category not in categories:
                    categories.append(category)

        return categories

    async def fetch(
        self,
        queries: list[str],
        platforms: list[Any] | None = None,
        max_results: int = getattr(
            constants, "DEFAULT_MAX_RESULTS", 20
        ),
        domains: list[Any] | None = None,
        domain_mode: bool = False,
        domain_fallback_enabled: bool | None = None,
        min_domain_sources: int | None = None,
        level_profile: str = "",
        beginner_ratio: float = 0.0,
        min_beginner_sources: int = 0,
        exclude_research_platforms: bool = False,
        mode: str = "",
    ) -> list[Source]:
        self._reset_run_state()

        mode_config = _mode_config(mode)
        resolved_mode = _resolve_mode(mode)

        try:
            mode_timeout = float(
                mode_config.get(
                    "search_timeout_seconds",
                    self._request_timeout,
                )
            )
        except Exception:
            mode_timeout = self._request_timeout

        try:
            mode_concurrency = int(
                mode_config.get(
                    "concurrency", self._concurrency
                )
            )
        except Exception:
            mode_concurrency = self._concurrency

        try:
            mode_variants = int(
                mode_config.get(
                    "max_query_variants",
                    self._max_query_variants,
                )
            )
        except Exception:
            mode_variants = self._max_query_variants

        try:
            mode_min_sources = int(
                mode_config.get(
                    "min_sources",
                    self._min_sources_before_fallback,
                )
            )
        except Exception:
            mode_min_sources = self._min_sources_before_fallback

        self._request_timeout = max(
            5.0, min(60.0, mode_timeout)
        )
        self._concurrency = max(
            1, min(16, mode_concurrency)
        )
        self._max_query_variants = max(
            1, min(12, mode_variants)
        )
        self._min_sources_before_fallback = max(
            1, mode_min_sources
        )

        self._trace.emit(
            "fetcher_mode_applied",
            mode=resolved_mode,
            request_timeout=self._request_timeout,
            concurrency=self._concurrency,
            max_query_variants=self._max_query_variants,
            min_sources_before_fallback=(
                self._min_sources_before_fallback
            ),
        )

        profile = self._normalize_level_profile(level_profile)
        beginner_ratio = max(
            0.0, min(1.0, float(beginner_ratio or 0.0))
        )
        min_beginner_sources = max(
            0, int(min_beginner_sources or 0)
        )

        cleaned_queries: list[str] = []
        seen_queries: set[str] = set()
        global_cap = max(8, self._max_query_variants * 2)

        for query in queries:
            cleaned = clean_text(query)

            if not cleaned:
                continue

            key = " ".join(cleaned.lower().split())

            if key in seen_queries:
                continue

            seen_queries.add(key)
            cleaned_queries.append(cleaned)

            if len(cleaned_queries) >= global_cap:
                break

        if not cleaned_queries:
            self._finalize_fetch_report([], [], [])
            return []

        domain_config = load_domain_routing_config()
        resolved_domains = (
            self._resolve_domains(domains) if domains else []
        )

        if platforms is not None:
            selected_platforms = self._resolve_platforms(platforms)
        elif domain_mode and resolved_domains:
            selected_platforms = self._domain_platforms(
                resolved_domains,
                include_fallback=False,
                include_universal=False,
            )
        else:
            selected_platforms = list(self._default_platforms)

        if not selected_platforms and domain_mode and resolved_domains:
            selected_platforms = self._domain_platforms(
                resolved_domains,
                include_fallback=True,
                include_universal=True,
            )

        if not selected_platforms:
            selected_platforms = self._default_pool()

        selected_platforms = self._filter_platforms_by_level(
            selected_platforms,
            profile=profile,
            exclude_research=exclude_research_platforms,
        )

        active_platforms = [
            platform
            for platform in selected_platforms
            if self._platform_usable(platform)[0]
        ]

        if not active_platforms:
            fallback_pool = self._default_pool()

            active_platforms = [
                platform
                for platform in fallback_pool
                if self._platform_usable(platform)[0]
            ]

        if not active_platforms:
            self._finalize_fetch_report([], [], [])
            return []

        max_results = max(1, int(max_results))
        self._last_active_platforms = list(active_platforms)

        for platform in active_platforms:
            self._ensure_outcome(platform)

        for platform in self._broken_clients:
            self._add_platform_warning(
                f"platform_client_unusable:{platform}:"
                f"{self._broken_clients[platform]}"
            )

        sources, failed_platform_values = (
            await self._fetch_from_platforms(
                cleaned_queries,
                active_platforms,
                max_results,
                profile=profile,
                beginner_ratio=beginner_ratio,
            )
        )

        active_keys = set(active_platforms)

        if domain_mode and resolved_domains:
            min_domain_sources = self._as_int(
                min_domain_sources,
                self._min_sources_before_fallback,
            )

            allow_domain_fallback = self._as_bool(
                domain_fallback_enabled,
                self._as_bool(
                    domain_config.get("allow_domain_fallback"),
                    True,
                ),
            )

            use_universal = self._as_bool(
                domain_config.get(
                    "use_universal_when_low_sources"
                ),
                True,
            )

            if (
                allow_domain_fallback
                and len(sources) < min_domain_sources
            ):
                expanded_platforms = self._domain_platforms(
                    resolved_domains,
                    include_fallback=True,
                    include_universal=use_universal,
                    min_platforms=min_domain_sources,
                )

                expanded_platforms = (
                    self._filter_platforms_by_level(
                        expanded_platforms,
                        profile=profile,
                        exclude_research=(
                            exclude_research_platforms
                        ),
                    )
                )

                new_platforms = [
                    platform
                    for platform in expanded_platforms
                    if platform not in active_keys
                    and self._platform_usable(platform)[0]
                ]

                if new_platforms:
                    (
                        additional_sources,
                        additional_failed,
                    ) = await self._fetch_from_platforms(
                        cleaned_queries,
                        new_platforms,
                        max_results,
                        profile=profile,
                        beginner_ratio=beginner_ratio,
                    )

                    sources = self._merge_sources(
                        sources, additional_sources
                    )
                    failed_platform_values.extend(
                        additional_failed
                    )
                    active_keys.update(new_platforms)

                if (
                    use_universal
                    and len(sources) < min_domain_sources
                ):
                    universal_platforms = (
                        self._universal_platforms()
                    )

                    universal_platforms = (
                        self._filter_platforms_by_level(
                            universal_platforms,
                            profile=profile,
                            exclude_research=(
                                exclude_research_platforms
                            ),
                        )
                    )

                    new_platforms = [
                        platform
                        for platform in universal_platforms
                        if platform not in active_keys
                        and self._platform_usable(platform)[0]
                    ]

                    if new_platforms:
                        (
                            additional_sources,
                            additional_failed,
                        ) = await self._fetch_from_platforms(
                            cleaned_queries,
                            new_platforms,
                            max_results,
                            profile=profile,
                            beginner_ratio=beginner_ratio,
                        )

                        sources = self._merge_sources(
                            sources, additional_sources
                        )
                        failed_platform_values.extend(
                            additional_failed
                        )
                        active_keys.update(new_platforms)
        elif (
            self._category_fallback_enabled
            and len(sources) < self._min_sources_before_fallback
        ):
            fallback_platforms = self.category_fallback_platforms(
                active_platforms=list(active_keys),
                source_types=None,
                target_count=self._min_sources_before_fallback,
            )

            fallback_platforms = self._filter_platforms_by_level(
                fallback_platforms,
                profile=profile,
                exclude_research=exclude_research_platforms,
            )

            new_platforms = [
                platform
                for platform in fallback_platforms
                if platform not in active_keys
                and self._platform_usable(
                    platform, check_outcomes=True
                )[0]
            ]

            if new_platforms:
                (
                    additional_sources,
                    additional_failed,
                ) = await self._fetch_from_platforms(
                    cleaned_queries,
                    new_platforms,
                    max_results,
                    profile=profile,
                    beginner_ratio=beginner_ratio,
                )

                sources = self._merge_sources(
                    sources, additional_sources
                )
                failed_platform_values.extend(additional_failed)

        self.last_platform_failures = list(
            self.platform_failures
        )
        self.failed_platforms = self._unique(
            failed_platform_values
        )

        self._finalize_fetch_report(
            sources,
            [
                *active_platforms,
                *[
                    p
                    for p in self._outcomes
                    if p not in active_platforms
                ],
            ],
            failed_platform_values,
        )

        return sources

    async def _fetch_from_platforms(
        self,
        queries: list[str],
        platforms: list[str],
        max_results: int,
        profile: str = "",
        beginner_ratio: float = 0.0,
    ) -> tuple[list[Source], list[str]]:
        if not platforms:
            return [], []

        request_limit = self._calculate_request_limit(
            max_results, len(platforms)
        )
        routes = self._route_queries(
            queries,
            platforms,
            profile=profile,
            beginner_ratio=beginner_ratio,
        )
        semaphore = asyncio.Semaphore(self._concurrency)
        tasks = []
        failed_platform_values: list[str] = []

        for platform in platforms:
            client = self._get_client(platform)

            if client is None:
                skip_warning = self._skipped_clients.get(platform)

                if skip_warning:
                    self._add_platform_warning(skip_warning)

                failed_platform_values.append(platform)
                self._mark_skipped(
                    platform, skip_warning or "client_unavailable"
                )
                continue

            platform_queries = routes.get(platform) or []

            if not platform_queries:
                platform_queries = [queries[0]]

            self._ensure_outcome(platform)
            self._outcomes[platform]["query_count"] = len(
                platform_queries
            )
            self._outcomes[platform]["attempted"] = True

            self._trace.emit(
                "fetcher_platform_routed",
                platform=platform,
                queries_routed=len(platform_queries),
                level_profile=profile,
                queries=platform_queries[:6],
            )

            for query in platform_queries:
                tasks.append(
                    self._fetch_one(
                        client=client,
                        platform=platform,
                        query=query,
                        limit=request_limit,
                        semaphore=semaphore,
                    )
                )

        if not tasks:
            return [], failed_platform_values

        results = await asyncio.gather(*tasks)

        sources: list[Source] = []
        attempts: dict[str, int] = {}
        failures: dict[str, list[str]] = {}
        found_counts: dict[str, int] = {}

        for platform, batch, error in results:
            attempts[platform] = attempts.get(platform, 0) + 1

            if error is not None:
                failures.setdefault(platform, []).append(error)

                outcome = self._ensure_outcome(platform)
                outcome["error_count"] += 1
                outcome["errors"].append(
                    self._short_error(error)
                )

                if (
                    "429" in error
                    or "rate limit" in error.lower()
                ):
                    outcome["rate_limited"] = True

                continue

            found_counts[platform] = found_counts.get(
                platform, 0
            ) + len(batch)
            sources.extend(batch)

        platform_failures: list[str] = []

        for platform in platforms:
            attempt_count = attempts.get(platform, 0)
            failure_count = len(failures.get(platform, []))
            found_count = found_counts.get(platform, 0)

            outcome = self._ensure_outcome(platform)
            outcome["source_count"] = found_count
            outcome["error_count"] = failure_count

            if attempt_count == 0:
                outcome["status"] = _FETCH_STATUS_SKIPPED
                outcome["reason_if_skipped"] = (
                    self._skipped_clients.get(platform)
                )
                continue

            if found_count > 0:
                outcome["succeeded"] = True
                outcome["status"] = _FETCH_STATUS_OK
            elif failure_count > 0 and outcome["rate_limited"]:
                outcome["status"] = _FETCH_STATUS_RATE_LIMITED
            elif failure_count > 0 and found_count == 0:
                outcome["failed"] = True
                outcome["status"] = _FETCH_STATUS_FAILED
            else:
                outcome["empty"] = True
                outcome["status"] = _FETCH_STATUS_EMPTY

            self._trace.emit(
                "fetcher_platform_summary",
                platform=platform,
                attempts=attempt_count,
                failures=failure_count,
                sources_found=found_count,
                status=outcome["status"],
                level_profile=profile,
            )

            if failure_count > 0:
                error_samples = failures.get(platform, [])[:3]

                self.last_platform_errors[platform] = [
                    self._short_error(item)
                    for item in error_samples
                ]

                if found_count == 0:
                    platform_failures.append(
                        f"platform_unavailable:{platform}"
                    )
                    failed_platform_values.append(platform)
                else:
                    platform_failures.append(
                        f"platform_degraded:{platform}:"
                        f"failures={failure_count}/"
                        f"{attempt_count}"
                    )
                    failed_platform_values.append(platform)

                for error_sample in error_samples:
                    platform_failures.append(
                        f"platform_error:{platform}:"
                        f"{self._short_error(error_sample)}"
                    )

        for failure in platform_failures:
            self._add_platform_warning(failure)

        return sources, failed_platform_values

    async def _fetch_one(
        self,
        client: Any,
        platform: str,
        query: str,
        limit: int,
        semaphore: asyncio.Semaphore,
    ) -> tuple[str, list[Source], str | None]:
        async with semaphore:
            try:
                sources = await run_with_timeout(
                    client.search(query, max_results=limit),
                    timeout=self._request_timeout,
                )
            except Exception as exc:
                error = str(exc)

                self._logger.warning(
                    f"{platform} fetch failed for query "
                    f"'{query}': {error}"
                )

                self._trace.emit(
                    "fetcher_platform_error",
                    platform=platform,
                    query=query,
                    error=error,
                )

                if (
                    "BaseHTTPClient" in error
                    or "no attribute 'search'" in error
                    or "client unusable" in error
                ):
                    self._mark_broken(platform, error)

                if (
                    "429" in error
                    or "rate limit" in error.lower()
                ):
                    tracker = self._get_health_tracker()

                    if tracker is not None:
                        try:
                            tracker.mark_degraded(
                                platform,
                                "rate_limit_429",
                                duration_seconds=60.0,
                            )
                        except Exception:
                            pass

                return platform, [], error

        tagged_sources: list[Source] = []

        for source in sources:
            try:
                source.metadata["matched_query"] = query
                source.metadata["fetched_platform"] = platform
                tagged_sources.append(source)
            except Exception:
                tagged_sources.append(source)

        self._trace.emit(
            "fetcher_platform_result",
            platform=platform,
            query=query,
            sources_found=len(tagged_sources),
        )

        return platform, tagged_sources, None

    def _mark_broken(self, platform: str, reason: str) -> None:
        platform = str(platform or "").strip().lower()

        if not platform:
            return

        self._broken_clients[platform] = reason

        warning = f"platform_client_unusable:{platform}:{reason}"

        if warning not in self.platform_warnings:
            self.platform_warnings.append(warning)

        if warning not in self.platform_failures:
            self.platform_failures.append(warning)

        self._logger.error(f"{platform} client unusable: {reason}")

    def _mark_skipped(
        self, platform: str, reason: str | None
    ) -> None:
        outcome = self._ensure_outcome(platform)
        outcome["status"] = _FETCH_STATUS_SKIPPED
        outcome["reason_if_skipped"] = reason or "unknown"

    def _get_client(self, platform: str) -> Any:
        platform = str(platform or "").strip().lower()

        if platform in self._broken_clients:
            self._skipped_clients[platform] = (
                f"platform_client_unusable:{platform}:"
                f"{self._broken_clients[platform]}"
            )
            return None

        client = self._clients.get(platform)

        if client is not None:
            return client

        is_usable, reason = self._platform_usable(platform)

        if not is_usable:
            self._skipped_clients[platform] = reason
            return None

        spec = self._CLIENT_SPECS.get(platform)

        if spec is None:
            self._skipped_clients[platform] = (
                f"platform_spec_missing:{platform}"
            )
            return None

        module_name, class_name = spec

        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            reason = (
                f"client_module_import_failed:"
                f"{self._short_error(exc)}"
            )
            self._mark_broken(platform, reason)
            self._skipped_clients[platform] = (
                f"platform_client_unusable:{platform}:{reason}"
            )
            return None

        client_class = self._resolve_client_class(
            module, class_name, platform
        )

        if client_class is None:
            reason = "client_class_missing_or_no_search_method"
            self._mark_broken(platform, reason)
            self._skipped_clients[platform] = (
                f"platform_client_unusable:{platform}:{reason}"
            )
            return None

        try:
            client = client_class()
        except Exception as exc:
            reason = (
                f"client_init_failed:{self._short_error(exc)}"
            )
            self._mark_broken(platform, reason)
            self._skipped_clients[platform] = (
                f"platform_client_unusable:{platform}:{reason}"
            )
            return None

        if not callable(getattr(client, "search", None)):
            reason = "client_has_no_search_method"
            self._mark_broken(platform, reason)
            self._skipped_clients[platform] = (
                f"platform_client_unusable:{platform}:{reason}"
            )
            return None

        self._clients[platform] = client
        return client

    def _resolve_client_class(
        self,
        module: Any,
        class_name: str,
        platform: str,
    ) -> Any:
        try:
            from clients.base_client import BaseHTTPClient
        except Exception:
            BaseHTTPClient = None

        exact = getattr(module, class_name, None)

        if self._client_class_usable(exact, BaseHTTPClient):
            return exact

        candidates: list[tuple[str, Any]] = []
        module_name = getattr(module, "__name__", "")

        for name, attr in vars(module).items():
            if not isinstance(attr, type):
                continue

            if (
                BaseHTTPClient is not None
                and attr is BaseHTTPClient
            ):
                continue

            if name == "BaseHTTPClient":
                continue

            if not name.endswith("Client"):
                continue

            if not callable(getattr(attr, "search", None)):
                continue

            if getattr(attr, "__module__", "") != module_name:
                continue

            candidates.append((name, attr))

        normalized_target = class_name.lower()

        for name, attr in candidates:
            if name.lower() == normalized_target:
                return attr

        platform_word = platform.replace("_", "").lower()

        for name, attr in candidates:
            if platform_word in name.lower():
                return attr

        if candidates:
            return candidates[0][1]

        return None

    def _client_class_usable(
        self,
        candidate: Any,
        base_client_class: Any,
    ) -> bool:
        if not isinstance(candidate, type):
            return False

        if (
            base_client_class is not None
            and candidate is base_client_class
        ):
            return False

        if getattr(candidate, "__name__", "") == "BaseHTTPClient":
            return False

        return callable(getattr(candidate, "search", None))

    def _platform_usable(
        self,
        platform: str,
        check_outcomes: bool = False,
    ) -> tuple[bool, str]:
        platform = str(platform or "").strip().lower()

        if not platform:
            return False, "platform_empty"

        if platform in self._broken_clients:
            return (
                False,
                f"platform_broken:"
                f"{self._broken_clients[platform]}",
            )

        tracker = self._get_health_tracker()

        if tracker is not None:
            try:
                if not tracker.is_usable(platform):
                    return False, f"platform_degraded:{platform}"
            except Exception:
                pass

        if platform not in self._CLIENT_SPECS:
            return False, f"platform_not_registered:{platform}"

        if not self._is_platform_enabled(platform):
            return False, f"platform_disabled:{platform}"

        has_key, key_reason = self._has_platform_key(platform)

        if not has_key:
            return False, key_reason

        if check_outcomes:
            outcome = self._outcomes.get(platform)

            if outcome is not None:
                if (
                    outcome.get("failed")
                    and outcome.get("source_count", 0) == 0
                ):
                    return (
                        False,
                        f"platform_failed_this_run:{platform}",
                    )

        return True, ""

    def _is_platform_enabled(self, platform: str) -> bool:
        config = self._platform_config(platform)

        if not isinstance(config, dict):
            return True

        return bool(config.get("enabled", True))

    def _has_platform_key(
        self, platform: str
    ) -> tuple[bool, str]:
        config = self._platform_config(platform)

        requires_key = bool(
            config.get(
                "requires_api_key",
                platform in self._DEFAULT_KEY_REQUIRED_PLATFORMS,
            )
        )

        if not requires_key:
            return True, ""

        setting_name = str(
            config.get("api_key_setting")
            or f"{platform}_api_key"
        )

        try:
            from core.config import get_settings

            settings = get_settings()
            secret = getattr(settings, setting_name, None)

            if secret is None:
                return (
                    False,
                    f"platform_key_missing:{platform}:"
                    f"{setting_name}",
                )

            if hasattr(secret, "get_secret_value"):
                if not secret.get_secret_value().strip():
                    return (
                        False,
                        f"platform_key_empty:{platform}:"
                        f"{setting_name}",
                    )
                return True, ""

            if not str(secret).strip():
                return (
                    False,
                    f"platform_key_empty:{platform}:"
                    f"{setting_name}",
                )

            return True, ""
        except Exception as exc:
            return (
                False,
                f"platform_key_lookup_failed:{platform}:"
                f"{self._short_error(exc)}",
            )

    def _platform_config(self, platform: str) -> dict[str, Any]:
        config = _get_sources_config()
        platforms = config.get("platforms", {})

        if not isinstance(platforms, dict):
            return {}

        platform_config = platforms.get(platform, {})

        if not isinstance(platform_config, dict):
            return {}

        return platform_config

    def _config_value(self, name: str, default: Any) -> Any:
        try:
            from core.config import get_settings

            settings = get_settings()
            section = getattr(settings, "search", None)

            if isinstance(section, dict):
                value = section.get(name)

                if value is not None:
                    return value
            elif section is not None:
                value = getattr(section, name, None)

                if value is not None:
                    return value

            for attr in (f"search_{name}", name):
                value = getattr(settings, attr, None)

                if value is not None:
                    return value
        except Exception:
            pass

        config = _get_sources_config()

        fetch_section = config.get("fetch", {})

        if (
            isinstance(fetch_section, dict)
            and fetch_section.get(name) is not None
        ):
            return fetch_section[name]

        search_section = config.get("search", {})

        if (
            isinstance(search_section, dict)
            and search_section.get(name) is not None
        ):
            return search_section[name]

        return default

    def _category_fallback_order_from_config(self) -> list[str]:
        configured = self._config_value(
            "category_fallback_order", None
        )

        if isinstance(configured, str):
            configured = [
                item.strip()
                for item in configured.split(",")
            ]

        if isinstance(configured, list):
            cleaned = self._clean_string_list(configured)

            if cleaned:
                return cleaned

        return list(self._DEFAULT_CATEGORY_FALLBACK_ORDER)

    def _categories_for_source_types(
        self, source_types: list[Any] | None
    ) -> list[str]:
        if not source_types:
            return []

        categories: list[str] = []

        config_map = self._config_value(
            "source_type_category_map", None
        )

        if not isinstance(config_map, dict):
            config_map = {}

        for source_type in source_types:
            value = str(
                getattr(source_type, "value", source_type) or ""
            ).strip().lower()

            if not value:
                continue

            category = str(
                config_map.get(value, "")
            ).strip().lower()

            if not category:
                category = self._SOURCE_TYPE_CATEGORY_MAP.get(
                    value, ""
                )

            if category and category not in categories:
                categories.append(category)

        return categories

    def _default_pool(self) -> list[str]:
        configured = self._config_value("default_platforms", None)

        if configured is not None:
            configured_platforms = self._clean_platform_list(
                configured
            )

            usable_configured = [
                platform
                for platform in configured_platforms
                if self._platform_usable(platform)[0]
            ]

            if usable_configured:
                return usable_configured

        pool: list[str] = []

        for category in self._category_fallback_order:
            for platform in self.category_platforms(category):
                if platform in pool:
                    continue

                if not self._platform_usable(platform)[0]:
                    continue

                pool.append(platform)

        return pool

    def _resolve_platforms(self, platforms: Any) -> list[str]:
        if platforms is None:
            return []

        if isinstance(platforms, str):
            platforms = [platforms]

        if not isinstance(platforms, (list, tuple, set)):
            return []

        resolved: list[str] = []

        for platform in platforms:
            value = str(
                getattr(platform, "value", platform) or ""
            ).strip().lower()

            if not value:
                continue

            if value in resolved:
                continue

            resolved.append(value)

        return resolved

    def _clean_platform_list(self, platforms: Any) -> list[str]:
        if platforms is None:
            return []

        if isinstance(platforms, str):
            platforms = [platforms]

        if not isinstance(platforms, (list, tuple, set)):
            return []

        cleaned: list[str] = []

        for platform in platforms:
            value = str(platform or "").strip().lower()

            if value and value not in cleaned:
                cleaned.append(value)

        return cleaned

    def _clean_string_list(self, values: Any) -> list[str]:
        if values is None:
            return []

        if isinstance(values, str):
            values = [values]

        if not isinstance(values, (list, tuple, set)):
            return []

        cleaned: list[str] = []

        for value in values:
            text = str(value or "").strip()

            if text and text not in cleaned:
                cleaned.append(text)

        return cleaned

    def _calculate_request_limit(
        self, max_results: int, platform_count: int
    ) -> int:
        if platform_count <= 0:
            return int(
                self._config_value("min_request_limit", 3)
            )

        per_platform = math.ceil(max_results / platform_count)

        return max(
            int(self._config_value("min_request_limit", 3)),
            min(
                int(self._config_value("max_request_limit", 10)),
                int(per_platform),
            ),
        )

    def _normalize_level_profile(self, value: Any) -> str:
        text = str(value or "").strip().lower()

        if text in _VALID_LEVEL_PROFILES:
            return text

        return ""

    def _filter_platforms_by_level(
        self,
        platforms: list[str],
        profile: str,
        exclude_research: bool,
    ) -> list[str]:
        if not platforms:
            return platforms

        if profile not in {
            _LEVEL_PROFILE_BEGINNER,
            _LEVEL_PROFILE_MIXED,
            _LEVEL_PROFILE_ADVANCED,
        }:
            return platforms

        filtered: list[str] = []

        if profile == _LEVEL_PROFILE_BEGINNER:
            if exclude_research:
                preferred: list[str] = []
                others: list[str] = []

                for platform in platforms:
                    if (
                        platform in self._BEGINNER_PLATFORMS
                        or platform in self._CODE_PLATFORMS
                        or platform in self._REFERENCE_PLATFORMS
                    ):
                        preferred.append(platform)
                    else:
                        others.append(platform)

                return [*preferred, *others]

            return platforms

        if profile == _LEVEL_PROFILE_ADVANCED:
            research_first: list[str] = []
            beginner_later: list[str] = []
            others: list[str] = []

            for platform in platforms:
                if (
                    platform in self._RESEARCH_ONLY_PLATFORMS
                    or platform == "zenodo"
                ):
                    research_first.append(platform)
                elif platform in self._BEGINNER_PLATFORMS:
                    beginner_later.append(platform)
                else:
                    others.append(platform)

            return [*research_first, *others, *beginner_later]

        return platforms

    def _route_queries(
        self,
        queries: list[str],
        platforms: list[str],
        profile: str = "",
        beginner_ratio: float = 0.0,
    ) -> dict[str, list[str]]:
        def hinted_platforms(query: str) -> set[str]:
            lowered = query.lower()
            owners: set[str] = set()

            for platform, hints in self._PLATFORM_HINTS.items():
                if any(hint in lowered for hint in hints):
                    owners.add(platform)

            return owners

        neutral: list[str] = []
        beginner_tagged: list[str] = []
        advanced_tagged: list[str] = []

        tagged: dict[str, list[str]] = {
            platform: [] for platform in self._PLATFORM_HINTS
        }

        for query in queries:
            owners = hinted_platforms(query)
            is_beginner_shaped = self._is_beginner_query(query)
            is_advanced_shaped = self._is_advanced_query(query)

            if not owners:
                if is_beginner_shaped:
                    beginner_tagged.append(query)
                elif is_advanced_shaped:
                    advanced_tagged.append(query)
                else:
                    neutral.append(query)
                continue

            for platform in owners:
                tagged[platform].append(query)

        routes: dict[str, list[str]] = {}

        for platform in platforms:
            if profile == _LEVEL_PROFILE_BEGINNER:
                selected = self._select_beginner_queries(
                    platform=platform,
                    tagged=tagged,
                    neutral=neutral,
                    beginner_tagged=beginner_tagged,
                    queries=queries,
                )
            elif profile == _LEVEL_PROFILE_ADVANCED:
                selected = self._select_advanced_queries(
                    platform=platform,
                    tagged=tagged,
                    neutral=neutral,
                    advanced_tagged=advanced_tagged,
                    queries=queries,
                )
            elif profile == _LEVEL_PROFILE_MIXED:
                selected = self._select_mixed_queries(
                    platform=platform,
                    tagged=tagged,
                    neutral=neutral,
                    beginner_tagged=beginner_tagged,
                    advanced_tagged=advanced_tagged,
                    queries=queries,
                )
            else:
                selected = self._select_default_queries(
                    platform=platform,
                    tagged=tagged,
                    neutral=neutral,
                    queries=queries,
                )

            capped: list[str] = []
            seen: set[str] = set()

            for query in selected:
                key = " ".join(query.lower().split())

                if key in seen:
                    continue

                seen.add(key)
                capped.append(query)

                if len(capped) >= self._max_query_variants:
                    break

            if not capped:
                capped = [queries[0]]

            if (
                len(capped) < _MIN_QUERIES_PER_PLATFORM
                and len(queries) >= _MIN_QUERIES_PER_PLATFORM
            ):
                for query in queries:
                    key = " ".join(query.lower().split())

                    if key in seen:
                        continue

                    seen.add(key)
                    capped.append(query)

                    if len(capped) >= _MIN_QUERIES_PER_PLATFORM:
                        break

            routes[platform] = capped

        return routes

    def _select_beginner_queries(
        self,
        platform: str,
        tagged: dict[str, list[str]],
        neutral: list[str],
        beginner_tagged: list[str],
        queries: list[str],
    ) -> list[str]:
        if platform in self._RESEARCH_ONLY_PLATFORMS:
            if beginner_tagged:
                return list(beginner_tagged)
            return []

        if platform in self._BEGINNER_PLATFORMS:
            base = list(tagged.get(platform, [])) + list(
                beginner_tagged
            )

            if not base:
                base = list(neutral)

            if not base:
                base = list(queries)

            return base

        if platform in self._CODE_PLATFORMS:
            base = list(tagged.get(platform, [])) + [
                query
                for query in neutral
                if self._contains_hints(query, self._CODE_HINTS)
            ]

            if not base:
                base = list(neutral[:3])

            if not base:
                base = list(queries)

            return base

        if platform in self._REFERENCE_PLATFORMS:
            base = list(tagged.get(platform, [])) + list(
                beginner_tagged
            )

            if not base:
                base = list(neutral[:3])

            if not base:
                base = list(queries)

            return base

        base = list(tagged.get(platform, [])) + list(
            beginner_tagged
        )

        if not base:
            base = list(neutral[:3])

        if not base:
            base = list(queries)

        return base

    def _select_advanced_queries(
        self,
        platform: str,
        tagged: dict[str, list[str]],
        neutral: list[str],
        advanced_tagged: list[str],
        queries: list[str],
    ) -> list[str]:
        if platform in self._RESEARCH_ONLY_PLATFORMS:
            base = (
                list(tagged.get(platform, []))
                + list(advanced_tagged)
                + self._without_hints(
                    neutral,
                    self._NON_ACADEMIC_HINTS + self._CODE_HINTS,
                )
            )

            if not base:
                base = list(neutral)

            if not base:
                base = list(queries)

            return base

        if platform in self._BEGINNER_PLATFORMS:
            base = list(tagged.get(platform, [])) + list(
                neutral[:3]
            )

            if not base:
                base = list(queries)

            return base

        base = list(tagged.get(platform, [])) + list(neutral[:3])

        if not base:
            base = list(queries)

        return base

    def _select_mixed_queries(
        self,
        platform: str,
        tagged: dict[str, list[str]],
        neutral: list[str],
        beginner_tagged: list[str],
        advanced_tagged: list[str],
        queries: list[str],
    ) -> list[str]:
        if platform in self._RESEARCH_ONLY_PLATFORMS:
            base = (
                list(tagged.get(platform, []))
                + list(advanced_tagged)
                + self._without_hints(
                    neutral,
                    self._NON_ACADEMIC_HINTS + self._CODE_HINTS,
                )
            )

            if not base:
                base = list(neutral)

            if not base:
                base = list(queries)

            return base

        if platform in self._BEGINNER_PLATFORMS:
            base = list(tagged.get(platform, [])) + list(
                beginner_tagged
            )

            if not base:
                base = list(neutral)

            if not base:
                base = list(queries)

            return base

        base = list(tagged.get(platform, [])) + list(neutral)

        if not base:
            base = list(queries)

        return base

    def _select_default_queries(
        self,
        platform: str,
        tagged: dict[str, list[str]],
        neutral: list[str],
        queries: list[str],
    ) -> list[str]:
        if platform in self._WEB_SEARCH_PLATFORMS:
            return list(queries)

        if platform == "github":
            return (
                tagged.get(platform, [])
                + [
                    query
                    for query in neutral
                    if self._contains_hints(
                        query, self._CODE_HINTS
                    )
                ][:2]
                + neutral[:2]
            )

        if platform in self._BOOK_PLATFORMS:
            return (
                tagged.get(platform, [])
                + self._without_hints(
                    neutral, self._CODE_HINTS
                )[:4]
            )

        if platform in self._COURSE_PLATFORMS:
            return (
                tagged.get(platform, [])
                + self._without_hints(
                    neutral, self._CODE_HINTS
                )[:4]
            )

        if platform in self._EXPLANATION_PLATFORMS:
            return (
                tagged.get(platform, [])
                + self._without_hints(
                    neutral, self._CODE_HINTS
                )[:4]
            )

        if platform in self._ACADEMIC_PLATFORMS:
            return (
                tagged.get(platform, [])
                + self._without_hints(
                    neutral,
                    self._NON_ACADEMIC_HINTS + self._CODE_HINTS,
                )
            )

        return list(neutral)

    def _is_beginner_query(self, query: str) -> bool:
        lowered = str(query or "").lower()

        beginner_markers = (
            "beginner",
            "basics",
            "basic",
            "fundamentals",
            "introduction",
            "intro",
            "tutorial",
            "course",
            "for beginners",
            "from scratch",
            "step by step",
            "explained",
        )

        return any(
            marker in lowered for marker in beginner_markers
        )

    def _is_advanced_query(self, query: str) -> bool:
        lowered = str(query or "").lower()

        advanced_markers = (
            "advanced",
            "research paper",
            "state of the art",
            "state-of-the-art",
            "in depth",
            "deep dive",
            "architecture",
            "implementation",
            "novel",
            "survey",
        )

        return any(
            marker in lowered for marker in advanced_markers
        )

    def _contains_hints(
        self, query: str, hints: tuple[str, ...]
    ) -> bool:
        lowered = str(query or "").lower()
        return any(hint in lowered for hint in hints)

    def _without_hints(
        self, items: list[str], hints: tuple[str, ...]
    ) -> list[str]:
        return [
            item
            for item in items
            if not any(hint in item.lower() for hint in hints)
        ]

    def _merge_sources(
        self,
        primary: list[Source],
        secondary: list[Source],
    ) -> list[Source]:
        seen: set[str] = set()
        merged: list[Source] = []

        for source in [*primary, *secondary]:
            key = str(
                getattr(source, "source_id", "")
                or getattr(source, "url", "")
                or getattr(source, "title", "")
                or ""
            ).lower()

            if not key or key in seen:
                continue

            seen.add(key)
            merged.append(source)

        return merged

    def _reset_run_state(self) -> None:
        self.last_platform_failures = []
        self.platform_failures = []
        self.platform_warnings = []
        self.failed_platforms = []
        self.last_platform_errors = {}
        self._outcomes = {}
        self._skipped_clients = {}
        self._last_active_platforms = []

    def _ensure_outcome(self, platform: str) -> dict[str, Any]:
        outcome = self._outcomes.get(platform)

        if outcome is None:
            outcome = _new_outcome()
            self._outcomes[platform] = outcome

        return outcome

    def _finalize_fetch_report(
        self,
        sources: list[Source],
        platforms: list[str],
        failed_platform_values: list[str],
    ) -> None:
        seen: set[str] = set()
        per_platform: list[dict[str, Any]] = []

        for platform in platforms:
            if platform in seen:
                continue

            seen.add(platform)
            outcome = self._ensure_outcome(platform)

            per_platform.append(
                {
                    "platform": platform,
                    "attempted": bool(
                        outcome.get("attempted", False)
                    ),
                    "query_count": int(
                        outcome.get("query_count", 0)
                    ),
                    "source_count": int(
                        outcome.get("source_count", 0)
                    ),
                    "status": str(
                        outcome.get(
                            "status", _FETCH_STATUS_SKIPPED
                        )
                    ),
                    "error": (outcome.get("errors") or [None])[0],
                    "reason_if_skipped": outcome.get(
                        "reason_if_skipped"
                    ),
                    "rate_limited": bool(
                        outcome.get("rate_limited", False)
                    ),
                    "error_count": int(
                        outcome.get("error_count", 0)
                    ),
                }
            )

        attempted = sum(
            1 for item in per_platform if item["attempted"]
        )
        with_data = sum(
            1
            for item in per_platform
            if item["source_count"] > 0
        )
        empty = sum(
            1
            for item in per_platform
            if item["attempted"]
            and item["source_count"] == 0
            and item["status"] == _FETCH_STATUS_EMPTY
        )
        failed = sum(
            1
            for item in per_platform
            if item["attempted"]
            and item["status"]
            in {
                _FETCH_STATUS_FAILED,
                _FETCH_STATUS_RATE_LIMITED,
            }
        )

        self.last_fetch_report = {
            "per_platform": per_platform,
            "totals": {
                "total_sources": len(sources),
                "total_platforms_attempted": attempted,
                "total_platforms_with_data": with_data,
                "total_platforms_empty": empty,
                "total_platforms_failed": failed,
            },
        }

    def get_last_fetch_report(self) -> dict[str, Any]:
        return dict(self.last_fetch_report)

    def _add_platform_warning(self, value: Any) -> None:
        text = str(value)

        if not text:
            return

        if text not in self.platform_warnings:
            self.platform_warnings.append(text)

        if text not in self.platform_failures:
            self.platform_failures.append(text)

    @staticmethod
    def _unique(values: list[Any]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []

        for value in values:
            text = str(value)

            if text and text not in seen:
                seen.add(text)
                result.append(text)

        return result

    def _short_error(self, error: Any) -> str:
        text = clean_text(error)
        text = text.replace("\n", " ")

        if len(text) > 180:
            text = text[:180] + "..."

        return text

    @staticmethod
    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    @staticmethod
    def _as_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return bool(value)

        if isinstance(value, str):
            text = value.strip().lower()

            if text in {"true", "yes", "on", "1"}:
                return True

            if text in {"false", "no", "off", "0"}:
                return False

        return bool(default)