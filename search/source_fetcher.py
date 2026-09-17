from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any

from core import constants
from clients.arxiv_client import ArxivClient
from clients.core_client import CoreClient
from clients.exa_client import ExaClient
from clients.github_client import GithubClient
from clients.huggingface_client import HuggingFaceClient
from clients.jina_client import JinaClient
from clients.openalex_client import OpenAlexClient
from clients.semantic_scholar_client import SemanticScholarClient
from clients.serper_client import SerperClient
from clients.serpapi_client import SerpApiClient
from clients.tavily_client import TavilyClient
from clients.web_search_client import WebSearchClient
from clients.wikipedia_client import WikipediaClient
from core.models import Source, SourcePlatform
from utils.async_helpers import run_with_timeout
from utils.logger import get_logger, get_trace_logger
from utils.text import clean_text

_SOURCES_CONFIG_CACHE: dict[str, Any] | None = None


def _load_sources_yaml() -> dict[str, Any]:
    try:
        import yaml

        config_path = Path(__file__).resolve().parent.parent / "configs" / "sources.yaml"

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}

            if isinstance(raw, dict):
                return raw
    except Exception:
        pass

    return {}


def _get_sources_config() -> dict[str, Any]:
    global _SOURCES_CONFIG_CACHE

    if _SOURCES_CONFIG_CACHE is not None:
        return _SOURCES_CONFIG_CACHE

    _SOURCES_CONFIG_CACHE = _load_sources_yaml()

    return _SOURCES_CONFIG_CACHE


def _is_platform_enabled(platform_name: str) -> bool:
    config = _get_sources_config()

    platforms = config.get("platforms", {})

    if not isinstance(platforms, dict):
        return True

    platform_config = platforms.get(platform_name, {})

    if not isinstance(platform_config, dict):
        return True

    return bool(platform_config.get("enabled", True))


class SourceFetcher:
    _CLIENT_FACTORIES = {
        SourcePlatform.ARXIV: ArxivClient,
        SourcePlatform.SEMANTIC_SCHOLAR: SemanticScholarClient,
        SourcePlatform.OPENALEX: OpenAlexClient,
        SourcePlatform.GITHUB: GithubClient,
        SourcePlatform.WIKIPEDIA: WikipediaClient,
        SourcePlatform.HUGGINGFACE: HuggingFaceClient,
        SourcePlatform.WEB: WebSearchClient,
        SourcePlatform.TAVILY: TavilyClient,
        SourcePlatform.EXA: ExaClient,
        SourcePlatform.CORE: CoreClient,
        SourcePlatform.SERPER: SerperClient,
        SourcePlatform.SERPAPI: SerpApiClient,
        SourcePlatform.JINA: JinaClient,
    }

    _PLATFORM_HINTS: dict[SourcePlatform, tuple[str, ...]] = {
        SourcePlatform.ARXIV: ("arxiv",),
        SourcePlatform.SEMANTIC_SCHOLAR: ("semantic scholar",),
        SourcePlatform.OPENALEX: ("openalex",),
        SourcePlatform.GITHUB: ("github", "source code", "codebase"),
        SourcePlatform.WIKIPEDIA: ("wikipedia",),
        SourcePlatform.HUGGINGFACE: ("huggingface", "hugging face"),
        SourcePlatform.WEB: (),
        SourcePlatform.TAVILY: (),
        SourcePlatform.EXA: (),
        SourcePlatform.CORE: ("core.ac", "open access paper"),
        SourcePlatform.SERPER: (),
        SourcePlatform.SERPAPI: (),
        SourcePlatform.JINA: (),
    }

    _ACADEMIC_PLATFORMS = (
        SourcePlatform.ARXIV,
        SourcePlatform.SEMANTIC_SCHOLAR,
        SourcePlatform.OPENALEX,
        SourcePlatform.CORE,
    )

    _WEB_SEARCH_PLATFORMS = (
        SourcePlatform.WEB,
        SourcePlatform.TAVILY,
        SourcePlatform.EXA,
        SourcePlatform.SERPER,
        SourcePlatform.SERPAPI,
        SourcePlatform.JINA,
    )

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

    def __init__(
        self,
        platforms: list[Any] | None = None,
        concurrency: int | None = None,
        max_query_variants: int | None = None,
        request_timeout: float | None = None,
    ) -> None:
        self._default_platforms = self._resolve_platforms(
            platforms if platforms is not None else list(self._CLIENT_FACTORIES.keys())
        )

        resolved_concurrency = (
            concurrency
            if concurrency is not None
            else self._config_value("concurrency", 6)
        )

        resolved_variants = (
            max_query_variants
            if max_query_variants is not None
            else self._config_value("max_query_variants", 6)
        )

        resolved_timeout = (
            request_timeout
            if request_timeout is not None
            else self._config_value(
                "request_timeout_seconds",
                constants.DEFAULT_TIMEOUT_SECONDS,
            )
        )

        self._concurrency = max(1, min(8, int(resolved_concurrency)))
        self._max_query_variants = max(1, min(8, int(resolved_variants)))
        self._request_timeout = max(10.0, min(60.0, float(resolved_timeout)))

        self._clients: dict[SourcePlatform, Any] = {}
        self._skipped_clients: dict[SourcePlatform, str] = {}
        self._disabled_platforms: set[str] = set()

        self.last_platform_failures: list[str] = []
        self.platform_failures: list[str] = []
        self.platform_warnings: list[str] = []
        self.failed_platforms: list[str] = []
        self.last_platform_errors: dict[str, list[str]] = {}

        self._logger = get_logger("search.source_fetcher")
        self._trace = get_trace_logger()

        self._scan_disabled_platforms()

    def _scan_disabled_platforms(self) -> None:
        for platform_enum in self._CLIENT_FACTORIES.keys():
            platform_name = platform_enum.value

            if not _is_platform_enabled(platform_name):
                self._disabled_platforms.add(platform_name)

                self._trace.emit(
                    "fetcher_platform_disabled",
                    platform=platform_name,
                    reason="disabled_in_sources_yaml",
                )

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

        try:
            import yaml

            settings_path = Path(__file__).resolve().parent.parent / "configs" / "settings.yaml"

            if settings_path.exists():
                with open(settings_path, "r", encoding="utf-8") as handle:
                    loaded = yaml.safe_load(handle) or {}

                section = loaded.get("search", {}) if isinstance(loaded, dict) else {}

                if isinstance(section, dict) and section.get(name) is not None:
                    return section[name]

            sources_path = Path(__file__).resolve().parent.parent / "configs" / "sources.yaml"

            if sources_path.exists():
                with open(sources_path, "r", encoding="utf-8") as handle:
                    loaded = yaml.safe_load(handle) or {}

                section = loaded.get("fetch", {}) if isinstance(loaded, dict) else {}

                if isinstance(section, dict) and section.get(name) is not None:
                    return section[name]
        except Exception:
            pass

        return default

    async def __aenter__(self) -> SourceFetcher:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
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
    def _resolve_platforms(cls, platforms: list[Any]) -> list[SourcePlatform]:
        resolved: list[SourcePlatform] = []

        for platform in platforms:
            platform_value = str(getattr(platform, "value", platform)).strip().lower()

            try:
                platform_enum = SourcePlatform(platform_value)
            except Exception:
                continue

            if platform_enum in cls._CLIENT_FACTORIES and platform_enum not in resolved:
                resolved.append(platform_enum)

        return resolved

    def _has_semantic_scholar_key(self) -> bool:
        try:
            from core.config import get_settings

            settings = get_settings()
            secret = settings.semantic_scholar_api_key

            return bool(secret.get_secret_value().strip())
        except Exception:
            return False

    def _has_platform_key(self, platform: SourcePlatform) -> bool:
        try:
            from core.config import get_settings

            settings = get_settings()

            key_map = {
                SourcePlatform.TAVILY: "tavily_api_key",
                SourcePlatform.EXA: "exa_api_key",
                SourcePlatform.CORE: "core_api_key",
                SourcePlatform.SERPER: "serper_api_key",
                SourcePlatform.SERPAPI: "serpapi_api_key",
                SourcePlatform.JINA: "jina_api_key",
            }

            attr_name = key_map.get(platform)

            if attr_name is None:
                return True

            secret = getattr(settings, attr_name, None)

            if secret is None:
                return False

            return bool(secret.get_secret_value().strip())
        except Exception:
            return False

    def _get_client(self, platform: SourcePlatform) -> Any:
        platform_name = platform.value

        if platform_name in self._disabled_platforms:
            self._skipped_clients[platform] = f"platform_skipped_disabled:{platform_name}"
            return None

        client = self._clients.get(platform)

        if client is not None:
            return client

        if platform == SourcePlatform.SEMANTIC_SCHOLAR and not self._has_semantic_scholar_key():
            self._skipped_clients[platform] = "platform_skipped_no_api_key:semantic_scholar"
            return None

        if platform in (
            SourcePlatform.TAVILY,
            SourcePlatform.EXA,
            SourcePlatform.CORE,
            SourcePlatform.SERPER,
            SourcePlatform.SERPAPI,
            SourcePlatform.JINA,
        ) and not self._has_platform_key(platform):
            self._skipped_clients[platform] = f"platform_skipped_no_api_key:{platform_name}"
            return None

        factory = self._CLIENT_FACTORIES.get(platform)

        if factory is None:
            return None

        try:
            client = factory()
        except Exception as exc:
            self._skipped_clients[platform] = (
                f"platform_client_init_failed:{platform_name}:{self._short_error(exc)}"
            )
            return None

        self._clients[platform] = client

        return client

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

    def _route_queries(
        self,
        queries: list[str],
        platforms: list[SourcePlatform],
    ) -> dict[SourcePlatform, list[str]]:
        def hinted_platforms(query: str) -> set[SourcePlatform]:
            lowered = query.lower()
            owners: set[SourcePlatform] = set()

            for platform, hints in self._PLATFORM_HINTS.items():
                if any(hint in lowered for hint in hints):
                    owners.add(platform)

            return owners

        neutral: list[str] = []

        tagged: dict[SourcePlatform, list[str]] = {
            platform: [] for platform in self._PLATFORM_HINTS
        }

        for query in queries:
            owners = hinted_platforms(query)

            if not owners:
                neutral.append(query)
                continue

            for platform in owners:
                tagged[platform].append(query)

        def without_hints(items: list[str], hints: tuple[str, ...]) -> list[str]:
            return [
                item
                for item in items
                if not any(hint in item.lower() for hint in hints)
            ]

        routes: dict[SourcePlatform, list[str]] = {}

        for platform in platforms:
            if platform in self._WEB_SEARCH_PLATFORMS:
                selected = list(queries)
            elif platform == SourcePlatform.GITHUB:
                selected = tagged[SourcePlatform.GITHUB] + neutral[:4]
            elif platform == SourcePlatform.WIKIPEDIA:
                selected = tagged[SourcePlatform.WIKIPEDIA] + without_hints(
                    neutral,
                    self._CODE_HINTS,
                )
            elif platform == SourcePlatform.HUGGINGFACE:
                selected = tagged[SourcePlatform.HUGGINGFACE] + neutral[:2]
            elif platform in self._ACADEMIC_PLATFORMS:
                academic_neutral = without_hints(
                    neutral,
                    self._NON_ACADEMIC_HINTS + self._CODE_HINTS,
                )
                selected = tagged.get(platform, []) + academic_neutral
            else:
                selected = list(neutral)

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

            routes[platform] = capped

        return routes

    async def fetch(
        self,
        queries: list[str],
        platforms: list[Any] | None = None,
        max_results: int = constants.DEFAULT_MAX_RESULTS,
    ) -> list[Source]:
        self.last_platform_failures = []
        self.platform_failures = []
        self.platform_warnings = []
        self.failed_platforms = []
        self.last_platform_errors = {}

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
            return []

        selected_platforms = (
            self._resolve_platforms(platforms)
            if platforms
            else self._default_platforms
        )

        if not selected_platforms:
            return []

        active_platforms: list[SourcePlatform] = []

        for platform in selected_platforms:
            if platform.value in self._disabled_platforms:
                continue

            active_platforms.append(platform)

        if not active_platforms:
            self.failed_platforms = []
            self.last_platform_failures = []
            return []

        max_results = max(1, int(max_results))

        request_limit = self._calculate_request_limit(max_results, len(active_platforms))
        routes = self._route_queries(cleaned_queries, active_platforms)

        self._trace.emit(
            "fetcher_queries_selected",
            total_available=len(queries),
            max_query_variants=self._max_query_variants,
            disabled_platforms=list(self._disabled_platforms),
            active_platforms=[p.value for p in active_platforms],
            routes={
                platform.value: route_queries
                for platform, route_queries in routes.items()
            },
        )

        semaphore = asyncio.Semaphore(self._concurrency)
        tasks = []
        failed_platform_values: list[str] = []

        for platform in active_platforms:
            client = self._get_client(platform)

            if client is None:
                skip_warning = self._skipped_clients.get(platform)

                if skip_warning:
                    self._add_platform_warning(skip_warning)

                failed_platform_values.append(platform.value)
                continue

            platform_queries = routes.get(platform) or [cleaned_queries[0]]

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
            self.failed_platforms = self._unique(failed_platform_values)
            self.last_platform_failures = list(self.platform_failures)
            return []

        results = await asyncio.gather(*tasks)

        sources: list[Source] = []
        attempts: dict[SourcePlatform, int] = {}
        failures: dict[SourcePlatform, list[str]] = {}
        found_counts: dict[SourcePlatform, int] = {}

        for platform, batch, error in results:
            attempts[platform] = attempts.get(platform, 0) + 1

            if error is not None:
                failures.setdefault(platform, []).append(error)
                continue

            found_counts[platform] = found_counts.get(platform, 0) + len(batch)
            sources.extend(batch)

        platform_failures: list[str] = []

        for platform in active_platforms:
            attempt_count = attempts.get(platform, 0)
            failure_count = len(failures.get(platform, []))
            found_count = found_counts.get(platform, 0)

            self._trace.emit(
                "fetcher_platform_summary",
                platform=platform.value,
                attempts=attempt_count,
                failures=failure_count,
                sources_found=found_count,
            )

            if attempt_count == 0:
                continue

            if failure_count > 0:
                error_samples = failures.get(platform, [])[:3]

                self.last_platform_errors[platform.value] = [
                    self._short_error(item) for item in error_samples
                ]

                if found_count == 0:
                    platform_failures.append(f"platform_unavailable:{platform.value}")
                    failed_platform_values.append(platform.value)
                else:
                    platform_failures.append(
                        f"platform_degraded:{platform.value}:failures={failure_count}/{attempt_count}"
                    )
                    failed_platform_values.append(platform.value)

                for error_sample in error_samples:
                    platform_failures.append(
                        f"platform_error:{platform.value}:{self._short_error(error_sample)}"
                    )

        for failure in platform_failures:
            self._add_platform_warning(failure)

        self.last_platform_failures = list(self.platform_failures)
        self.failed_platforms = self._unique(failed_platform_values)

        return sources

    def _calculate_request_limit(self, max_results: int, platform_count: int) -> int:
        if platform_count <= 0:
            return 5

        per_platform = math.ceil(max_results / platform_count)

        return max(3, min(10, per_platform))

    async def _fetch_one(
        self,
        client: Any,
        platform: SourcePlatform,
        query: str,
        limit: int,
        semaphore: asyncio.Semaphore,
    ) -> tuple[SourcePlatform, list[Source], str | None]:
        async with semaphore:
            try:
                sources = await run_with_timeout(
                    client.search(query, max_results=limit),
                    timeout=self._request_timeout,
                )
            except Exception as exc:
                error = str(exc)

                self._logger.warning(
                    f"{platform.value} fetch failed for query '{query}': {error}"
                )

                self._trace.emit(
                    "fetcher_platform_error",
                    platform=platform.value,
                    query=query,
                    error=error,
                )

                return platform, [], error

            tagged_sources: list[Source] = []

            for source in sources:
                try:
                    source.metadata["matched_query"] = query
                    source.metadata["fetched_platform"] = platform.value
                    tagged_sources.append(source)
                except Exception:
                    tagged_sources.append(source)

            self._trace.emit(
                "fetcher_platform_result",
                platform=platform.value,
                query=query,
                sources_found=len(tagged_sources),
            )

            return platform, tagged_sources, None