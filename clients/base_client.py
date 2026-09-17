from __future__ import annotations

import asyncio
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx

from core import constants
from core.exceptions import SourceFetchError
from utils.cache import default_cache_registry
from utils.hashing import stable_hash
from utils.logger import get_logger
from utils.rate_limiter import default_rate_limiter_registry

_SOURCES_CONFIG_CACHE: dict[str, Any] | None = None


def _load_sources_yaml() -> dict[str, Any]:
    global _SOURCES_CONFIG_CACHE

    if _SOURCES_CONFIG_CACHE is not None:
        return _SOURCES_CONFIG_CACHE

    loaded: dict[str, Any] = {}

    try:
        import yaml

        config_path = Path(__file__).resolve().parent.parent / "configs" / "sources.yaml"

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}

            if isinstance(raw, dict):
                loaded = raw
    except Exception:
        loaded = {}

    _SOURCES_CONFIG_CACHE = loaded
    return loaded


class BaseHTTPClient:
    platform_name: str = "unknown"
    retryable_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504)
    permanent_status_codes: tuple[int, ...] = (400, 401, 403, 404, 422)

    def __init__(
        self,
        base_url: str = "",
        rate_limit: int = 10,
        period_seconds: float = 60.0,
        cache_enabled: bool = True,
        cache_ttl_seconds: int = constants.CACHE_DEFAULT_TTL_SECONDS,
        timeout: float = constants.DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = constants.DEFAULT_MAX_RETRIES,
        retry_backoff_seconds: float = constants.DEFAULT_RETRY_BACKOFF_SECONDS,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_headers = dict(default_headers or {})
        self._cache_enabled = cache_enabled
        self._cache_ttl_seconds = cache_ttl_seconds

        resolved_rate_limit = int(rate_limit)
        resolved_period = float(period_seconds)
        resolved_timeout = float(timeout)
        resolved_max_retries = int(max_retries)
        resolved_backoff = float(retry_backoff_seconds)

        try:
            from core.config import get_settings

            settings = get_settings()

            if resolved_timeout == constants.DEFAULT_TIMEOUT_SECONDS:
                resolved_timeout = float(settings.search_request_timeout_seconds)

            if resolved_max_retries == constants.DEFAULT_MAX_RETRIES:
                resolved_max_retries = int(settings.source_max_retries)

            if resolved_backoff == constants.DEFAULT_RETRY_BACKOFF_SECONDS:
                resolved_backoff = float(settings.source_retry_backoff_seconds)

            settings_rate_limit = self._settings_platform_rate_limit(settings)

            if settings_rate_limit is not None:
                resolved_rate_limit = int(settings_rate_limit)
        except Exception:
            pass

        sources_config = _load_sources_yaml()
        platform_config: dict[str, Any] = {}
        fetch_config: dict[str, Any] = {}

        if isinstance(sources_config, dict):
            platforms = sources_config.get("platforms")

            if isinstance(platforms, dict):
                raw_platform_config = platforms.get(self.platform_name)

                if isinstance(raw_platform_config, dict):
                    platform_config = raw_platform_config

            raw_fetch_config = sources_config.get("fetch")

            if isinstance(raw_fetch_config, dict):
                fetch_config = raw_fetch_config

        self.platform_enabled = bool(platform_config.get("enabled", True))

        if platform_config.get("rate_limit_per_minute") is not None:
            resolved_rate_limit = int(platform_config["rate_limit_per_minute"])

        if platform_config.get("timeout_seconds") is not None:
            resolved_timeout = float(platform_config["timeout_seconds"])

        if platform_config.get("max_retries") is not None:
            resolved_max_retries = int(platform_config["max_retries"])

        if platform_config.get("retry_backoff_seconds") is not None:
            resolved_backoff = float(platform_config["retry_backoff_seconds"])

        if fetch_config.get("request_timeout_seconds") is not None:
            resolved_timeout = min(
                resolved_timeout,
                float(fetch_config["request_timeout_seconds"]),
            )

        if fetch_config.get("max_retries") is not None:
            resolved_max_retries = min(
                resolved_max_retries,
                int(fetch_config["max_retries"]),
            )

        if fetch_config.get("retry_backoff_seconds") is not None:
            resolved_backoff = max(
                resolved_backoff,
                float(fetch_config["retry_backoff_seconds"]),
            )

        self._timeout = max(5.0, min(60.0, resolved_timeout))
        self._max_retries = max(1, min(4, resolved_max_retries))
        self._retry_backoff_seconds = max(0.1, min(10.0, resolved_backoff))

        self._client: httpx.AsyncClient | None = None
        self._logger = get_logger(f"clients.{self.platform_name}")

        self._rate_limiter = default_rate_limiter_registry.get(
            name=f"client_{self.platform_name}",
            rate_limit=max(1, resolved_rate_limit),
            period_seconds=resolved_period,
        )

        if cache_enabled:
            self._cache = default_cache_registry.get(
                name=f"client_{self.platform_name}",
                max_entries=constants.CACHE_MAX_ENTRIES,
            )
        else:
            self._cache = None

    def _settings_platform_rate_limit(self, settings: Any) -> int | None:
        mapping = {
            SourcePlatformValue.ARXIV: "arxiv_rate_limit_per_minute",
            SourcePlatformValue.SEMANTIC_SCHOLAR: "semantic_scholar_rate_limit_per_minute",
            SourcePlatformValue.OPENALEX: "openalex_rate_limit_per_minute",
            SourcePlatformValue.GITHUB: "github_rate_limit_per_minute",
            SourcePlatformValue.WIKIPEDIA: "wikipedia_rate_limit_per_minute",
            SourcePlatformValue.HUGGINGFACE: "huggingface_rate_limit_per_minute",
            SourcePlatformValue.WEB: "web_search_rate_limit_per_minute",
        }

        attr = mapping.get(self.platform_name)

        if attr is None:
            return None

        value = getattr(settings, attr, None)

        if value is None:
            return None

        try:
            return int(value)
        except Exception:
            return None

    async def __aenter__(self) -> BaseHTTPClient:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        await self.close()
        return False

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

        self._client = None

    def _build_default_headers(self) -> dict[str, str]:
        headers = dict(constants.DEFAULT_HEADERS)
        headers.update(self._default_headers)
        return headers

    def _cache_get(self, key: str) -> Any:
        if self._cache is None:
            return None

        try:
            return self._cache.get(key)
        except Exception:
            return None

    def _cache_set(self, key: str, value: Any) -> None:
        if self._cache is None:
            return

        try:
            self._cache.set(key, value, ttl_seconds=self._cache_ttl_seconds)
        except Exception:
            pass

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    self._timeout,
                    connect=constants.DEFAULT_CONNECT_TIMEOUT_SECONDS,
                ),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=5,
                ),
            )

        return self._client

    async def _request(
        self,
        url_or_path: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        url = self._build_url(url_or_path)

        if not self.platform_enabled:
            raise SourceFetchError(
                f"{self.platform_name} is disabled by configuration",
                details={"url": url},
            )

        merged_headers = self._build_default_headers()

        if headers:
            merged_headers.update(headers)

        for attempt in range(1, self._max_retries + 1):
            await self._rate_limiter.acquire()
            client = await self._get_client()

            try:
                response = await client.get(url, params=params, headers=merged_headers)
            except httpx.TimeoutException as exc:
                if attempt == self._max_retries:
                    raise SourceFetchError(
                        f"{self.platform_name} request timed out",
                        details={"url": url, "error": str(exc)},
                    ) from exc

                await asyncio.sleep(self._calculate_backoff(attempt, None))
                continue
            except httpx.HTTPError as exc:
                if attempt == self._max_retries:
                    raise SourceFetchError(
                        f"{self.platform_name} network error",
                        details={"url": url, "error": str(exc)},
                    ) from exc

                await asyncio.sleep(self._calculate_backoff(attempt, None))
                continue

            if response.status_code == 200:
                return response

            if (
                response.status_code in self.retryable_status_codes
                and attempt < self._max_retries
            ):
                delay = self._retry_after_seconds(response) or self._calculate_backoff(
                    attempt,
                    response.status_code,
                )
                await asyncio.sleep(delay)
                continue

            if response.status_code in self.permanent_status_codes or attempt == self._max_retries:
                raise SourceFetchError(
                    f"{self.platform_name} HTTP {response.status_code} client error",
                    details={
                        "url": url,
                        "status_code": response.status_code,
                        "body": response.text[:300],
                    },
                )

            await asyncio.sleep(self._calculate_backoff(attempt, response.status_code))

        raise SourceFetchError(
            f"{self.platform_name} request failed after retries",
            details={"url": url},
        )

    async def _get_json(
        self,
        url_or_path: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        response = await self._request(url_or_path, params=params, headers=headers)

        try:
            return response.json()
        except Exception as exc:
            raise SourceFetchError(
                f"{self.platform_name} invalid JSON response",
                details={"url": self._build_url(url_or_path), "error": str(exc)},
            ) from exc

    async def _get_text(
        self,
        url_or_path: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        response = await self._request(url_or_path, params=params, headers=headers)
        return response.text

    def _build_url(self, url_or_path: str) -> str:
        if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
            return url_or_path

        if not self._base_url:
            return url_or_path

        return urljoin(self._base_url + "/", url_or_path.lstrip("/"))

    def _clean_url(self, url: Any) -> str:
        raw = str(url or "").strip()

        if not raw:
            return ""

        if raw.startswith("//"):
            return "https:" + raw

        if not raw.startswith("http://") and not raw.startswith("https://"):
            return "https://" + raw

        return raw

    def _make_source_id(self, platform: str, identifier: str) -> str:
        return stable_hash({"platform": platform, "identifier": identifier})

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None:
            return None

        if isinstance(value, datetime):
            return value

        raw = str(value).strip()

        if not raw:
            return None

        for fmt in (
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%Y",
        ):
            try:
                parsed = datetime.strptime(raw, fmt)

                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)

                return parsed
            except Exception:
                continue

        return None

    def _parse_year(self, value: Any) -> int | None:
        if value is None:
            return None

        try:
            year = int(value)
        except Exception:
            return None

        if 0 <= year <= 2100:
            return year

        return None

    def _retry_after_seconds(self, response: httpx.Response) -> float | None:
        header = response.headers.get("Retry-After") or response.headers.get("retry-after")

        if header is None:
            return None

        try:
            return max(float(header), 0.1)
        except Exception:
            return None

    def _calculate_backoff(self, attempt: int, status_code: int | None = None) -> float:
        base = self._retry_backoff_seconds * (2 ** (attempt - 1))

        if status_code in (429, 503):
            base = max(base, 2.0)
            base = min(base, 20.0)
        else:
            base = min(base, 10.0)

        return base + random.uniform(0.0, 0.1 * max(base, 0.1))


class SourcePlatformValue:
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    OPENALEX = "openalex"
    GITHUB = "github"
    WIKIPEDIA = "wikipedia"
    HUGGINGFACE = "huggingface"
    WEB = "web"