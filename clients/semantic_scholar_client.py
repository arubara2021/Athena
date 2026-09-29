from __future__ import annotations

import os
from typing import Any

from clients.base_client import BaseHTTPClient
from core import constants
from core.config import get_settings
from core.exceptions import SourceFetchError
from core.models import Source, SourcePlatform, SourceType
from utils.cache import build_cache_key
from utils.logger import get_logger, get_trace_logger
from utils.platform_health import PlatformHealthTracker
from utils.rate_limiter import default_rate_limiter_registry
from utils.text import clean_text, truncate_text


class SemanticScholarClient(BaseHTTPClient):
    platform_name = SourcePlatform.SEMANTIC_SCHOLAR.value
    retryable_status_codes = (429, 500, 502, 503, 504)
    permanent_status_codes = (400, 401, 403, 404, 422)

    _RATE_LIMIT_COOLDOWN_SECONDS = 60.0
    _RATE_LIMIT_PENALTY_SECONDS = 45.0
    _AUTH_FAILURE_COOLDOWN_SECONDS = 300.0

    def __init__(self) -> None:
        self._api_key = self._read_api_key()
        self._allow_without_key = self._read_allow_without_key()
        rate_limit = self._read_rate_limit()

        super().__init__(
            base_url=constants.SEMANTIC_SCHOLAR_BASE_URL,
            rate_limit=rate_limit,
            period_seconds=60.0,
            cache_enabled=True,
            timeout=30.0,
            max_retries=2,
            retry_backoff_seconds=2.0,
        )

        self.last_error: str | None = None
        self._skip_logged = False
        self._health_tracker = PlatformHealthTracker.get_instance()
        self._limiter = default_rate_limiter_registry.get(
            name=f"client_{self.platform_name}"
        )
        self._trace = get_trace_logger()
        self._logger = get_logger(f"clients.{self.platform_name}")

    @staticmethod
    def _read_api_key() -> str:
        try:
            settings = get_settings()
            secret = settings.semantic_scholar_api_key

            if secret is None:
                return ""

            return str(secret.get_secret_value()).strip()
        except Exception:
            return ""

    @staticmethod
    def _read_allow_without_key() -> bool:
        try:
            settings = get_settings()
            return bool(
                getattr(settings, "allow_semantic_scholar_without_key", False)
            )
        except Exception:
            return False

    def _read_rate_limit(self) -> int:
        try:
            settings = get_settings()

            if self._api_key:
                return int(settings.semantic_scholar_rate_limit_with_key_per_minute)

            return int(settings.semantic_scholar_rate_limit_per_minute)
        except Exception:
            return 3

    def _build_default_headers(self) -> dict[str, str]:
        headers = super()._build_default_headers() or {}
        contact = os.getenv("RESEARCH_AGENT_CONTACT", "").strip()

        if contact:
            headers["User-Agent"] = (
                f"{constants.APP_NAME}/{constants.APP_VERSION} "
                f"(personal research agent; educational use; contact: {contact})"
            )
        else:
            headers["User-Agent"] = (
                f"{constants.APP_NAME}/{constants.APP_VERSION} "
                f"(personal research agent; educational use)"
            )

        headers["Accept"] = "application/json"

        if self._api_key:
            headers["x-api-key"] = self._api_key

        return headers

    async def search(self, query: str, max_results: int = 10) -> list[Source]:
        if not self.platform_enabled:
            self.last_error = "semantic_scholar_disabled_config"
            return []

        if not self._health_tracker.is_usable(self.platform_name):
            self.last_error = "semantic_scholar_degraded"
            self._trace.emit(
                "semantic_scholar_skipped",
                reason="platform_degraded",
                cooldown_seconds=self._health_tracker.get_cooldown_remaining(
                    self.platform_name
                ),
            )
            return []

        if not self._api_key and not self._allow_without_key:
            if not self._skip_logged:
                self._logger.warning(
                    "Semantic Scholar skipped because no API key is configured"
                )
                self._skip_logged = True

            self.last_error = "semantic_scholar_no_api_key"
            return []

        cleaned_query = clean_text(query)

        if not cleaned_query:
            return []

        cache_key = build_cache_key(
            self.platform_name,
            "search_v4",
            cleaned_query,
            max_results,
        )

        cached = self._cache_get(cache_key)

        if cached is not None:
            return cached

        params = {
            "query": cleaned_query,
            "limit": max_results,
            "fields": (
                "title,abstract,year,authors,citationCount,"
                "externalIds,url,openAccessPdf,publicationDate,venue"
            ),
        }

        try:
            payload = await self._get_json("/paper/search", params=params)
        except SourceFetchError as exc:
            handled = self._handle_source_fetch_error(
                exc=exc,
                cleaned_query=cleaned_query,
            )

            if handled:
                return []

            self.last_error = str(exc)
            raise
        except Exception as exc:
            self.last_error = str(exc)
            raise SourceFetchError(
                "Semantic Scholar search failed",
                details={"query": cleaned_query, "error": str(exc)},
            ) from exc

        if isinstance(payload, dict) and payload.get("error"):
            error_message = str(payload.get("error"))
            self.last_error = error_message
            raise SourceFetchError(
                f"Semantic Scholar API error: {error_message}",
                details={"query": cleaned_query},
            )

        self.last_error = None
        items = payload.get("data", []) if isinstance(payload, dict) else []
        sources: list[Source] = []

        for item in items[:max_results]:
            source = self._map_item(item)

            if source is not None:
                sources.append(source)

        if sources:
            self._cache_set(cache_key, sources)

        self._trace.emit(
            "semantic_scholar_result",
            query=cleaned_query,
            sources_found=len(sources),
        )

        return sources

    def _handle_source_fetch_error(
        self,
        exc: SourceFetchError,
        cleaned_query: str,
    ) -> bool:
        status_code = None

        try:
            status_code = exc.details.get("status_code") if exc.details else None
        except Exception:
            status_code = None

        error_text = str(exc).lower()

        if status_code == 403 or "403" in error_text or "forbidden" in error_text:
            self._mark_auth_failure(cleaned_query, status_code)
            return True

        if status_code == 401 or "401" in error_text or "unauthorized" in error_text:
            self._mark_auth_failure(cleaned_query, status_code)
            return True

        rate_limited = (
            status_code == 429
            or "429" in error_text
            or "rate limit" in error_text
        )

        if rate_limited:
            self._mark_rate_limited(cleaned_query, status_code)
            return True

        return False

    def _mark_auth_failure(self, query: str, status_code: int | None) -> None:
        self._health_tracker.mark_degraded(
            self.platform_name,
            "auth_failure",
            duration_seconds=self._AUTH_FAILURE_COOLDOWN_SECONDS,
        )

        self._trace.emit(
            "semantic_scholar_auth_failure",
            query=query,
            status_code=status_code,
            cooldown_seconds=self._AUTH_FAILURE_COOLDOWN_SECONDS,
            hint=(
                "Check that SEMANTIC_SCHOLAR_API_KEY in .env is valid "
                "or that allow_semantic_scholar_without_key is set."
            ),
        )

        self._logger.warning(
            f"Semantic Scholar returned "
            f"{status_code if status_code is not None else 'auth error'}. "
            f"Marked degraded for {self._AUTH_FAILURE_COOLDOWN_SECONDS:.0f}s. "
            f"Verify the API key configuration."
        )

        self.last_error = "semantic_scholar_auth_failure"

    def _mark_rate_limited(self, query: str, status_code: int | None) -> None:
        self._health_tracker.mark_degraded(
            self.platform_name,
            "rate_limit_429",
            duration_seconds=self._RATE_LIMIT_COOLDOWN_SECONDS,
        )

        try:
            import asyncio

            limiter = self._limiter

            if limiter is not None:
                async def _apply_penalty() -> None:
                    try:
                        await limiter.report_rate_limit(
                            penalty_seconds=self._RATE_LIMIT_PENALTY_SECONDS
                        )
                    except Exception:
                        pass

                try:
                    loop = asyncio.get_running_loop()

                    if loop is not None and loop.is_running():
                        loop.create_task(_apply_penalty())
                except Exception:
                    pass
        except Exception:
            pass

        self._trace.emit(
            "semantic_scholar_rate_limited",
            query=query,
            status_code=status_code,
            cooldown_seconds=self._RATE_LIMIT_COOLDOWN_SECONDS,
            penalty_seconds=self._RATE_LIMIT_PENALTY_SECONDS,
        )

        self.last_error = "semantic_scholar_rate_limited"

    def _map_item(self, item: dict) -> Source | None:
        try:
            title = clean_text(item.get("title", ""))

            if not title:
                return None

            paper_url = item.get("url") or ""
            open_access = item.get("openAccessPdf") or {}
            pdf_url = open_access.get("url") if isinstance(open_access, dict) else None
            final_url = self._clean_url(pdf_url or paper_url)

            if not final_url:
                return None

            authors = [
                clean_text(author.get("name", ""))
                for author in item.get("authors", []) or []
                if author.get("name")
            ]

            external_ids = item.get("externalIds") or {}
            citation_count = item.get("citationCount")
            publication_date = self._parse_datetime(item.get("publicationDate"))
            year = item.get("year") or (
                publication_date.year if publication_date else None
            )

            return Source(
                source_id=self._make_source_id(
                    self.platform_name,
                    str(item.get("paperId") or final_url),
                ),
                title=title,
                url=final_url,
                platform=SourcePlatform.SEMANTIC_SCHOLAR,
                source_type=SourceType.PAPER,
                abstract=truncate_text(
                    clean_text(item.get("abstract", "")),
                    max_length=2000,
                    suffix="",
                )
                or None,
                authors=authors,
                published_at=publication_date,
                year=self._parse_year(year),
                citation_count=int(citation_count)
                if isinstance(citation_count, int)
                else None,
                has_code=None,
                difficulty=None,
                score=None,
                metadata={
                    "paper_id": item.get("paperId"),
                    "external_ids": external_ids,
                    "has_pdf": bool(pdf_url),
                    "venue": clean_text(item.get("venue", "")),
                },
            )
        except Exception as exc:
            self._logger.warning(f"Skipping Semantic Scholar item: {exc}")
            return None