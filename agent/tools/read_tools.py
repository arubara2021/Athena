from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from agent.tools.executor import ToolResult
from agent.tools.registry import ToolDefinition, ToolRegistry
from utils.logger import get_logger

_logger = get_logger("agent.tools.read")

_READ_TIMEOUT = 45.0
_READ_TOKEN_COST = 2000


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _clip(value: Any, limit: int) -> str:
    text = _clean(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _jina_available() -> bool:
    try:
        from core.config import get_settings

        settings = get_settings()
        key_obj = getattr(settings, "jina_api_key", None)

        if key_obj is None:
            return False

        if hasattr(key_obj, "get_secret_value"):
            key = str(key_obj.get_secret_value() or "").strip()
        else:
            key = str(key_obj or "").strip()

        if not key:
            return False

        try:
            import yaml

            config_path = Path(__file__).resolve().parents[2] / "configs" / "sources.yaml"

            if config_path.exists():
                raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                platforms = raw.get("platforms", {}) if isinstance(raw, dict) else {}
                jina_config = platforms.get("jina", {}) if isinstance(platforms, dict) else {}

                if isinstance(jina_config, dict) and not bool(jina_config.get("enabled", True)):
                    return False
        except Exception:
            pass

        return True

    except Exception:
        return False


async def read_url(url: str = "") -> ToolResult:
    started = time.perf_counter()
    cleaned_url = _clean(url)

    if not cleaned_url:
        return ToolResult(
            tool_name="read_url",
            success=False,
            data=None,
            error="Empty URL provided",
            tokens_used=0,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    try:
        from clients.jina_client import JinaClient

        async with JinaClient() as client:
            source = await client.read_url(cleaned_url)

        if source is None:
            return ToolResult(
                tool_name="read_url",
                success=False,
                data=None,
                error="Failed to read URL content",
                tokens_used=0,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        title = _clean(getattr(source, "title", ""))
        abstract = _clean(getattr(source, "abstract", ""))
        source_url = _clean(getattr(source, "url", cleaned_url))
        metadata = getattr(source, "metadata", {}) or {}
        full_content = ""

        if isinstance(metadata, dict):
            full_content = _clean(metadata.get("full_content", ""))

        preview = abstract or full_content

        return ToolResult(
            tool_name="read_url",
            success=True,
            data={
                "title": title,
                "url": source_url,
                "content_preview": _clip(preview, 3000),
                "full_content": _clip(full_content, 10000),
                "content_length": len(full_content or preview),
            },
            tokens_used=0,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    except Exception as exc:
        _logger.warning(f"read_url failed: {exc}")
        return ToolResult(
            tool_name="read_url",
            success=False,
            data=None,
            error=str(exc),
            tokens_used=0,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


async def read_paper_abstract(
    title: str = "",
    abstract: str = "",
    url: str = "",
    **_: Any,
) -> ToolResult:
    started = time.perf_counter()
    cleaned_title = _clean(title)
    cleaned_abstract = _clean(abstract)
    cleaned_url = _clean(url)

    if not cleaned_title and not cleaned_abstract:
        return ToolResult(
            tool_name="read_paper_abstract",
            success=True,
            data={
                "title": "",
                "abstract": "",
                "url": cleaned_url,
                "combined_text": "",
                "skipped": True,
                "reason": "No title or abstract provided",
            },
            tokens_used=0,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    if not cleaned_title:
        cleaned_title = "Untitled source"

    combined = f"Title: {cleaned_title}\nAbstract: {cleaned_abstract}"

    return ToolResult(
        tool_name="read_paper_abstract",
        success=True,
        data={
            "title": cleaned_title,
            "abstract": cleaned_abstract,
            "url": cleaned_url,
            "combined_text": _clip(combined, 5000),
        },
        tokens_used=0,
        latency_ms=(time.perf_counter() - started) * 1000,
    )


async def extract_source_summary(
    title: str = "",
    abstract: str = "",
    source_type: str = "",
    platform: str = "",
    **_: Any,
) -> ToolResult:
    started = time.perf_counter()
    cleaned_title = _clean(title)
    cleaned_abstract = _clean(abstract)
    cleaned_source_type = _clean(source_type)
    cleaned_platform = _clean(platform)

    if not cleaned_title and not cleaned_abstract:
        return ToolResult(
            tool_name="extract_source_summary",
            success=True,
            data={
                "title": "",
                "summary": "",
                "source_type": cleaned_source_type,
                "platform": cleaned_platform,
                "has_abstract": False,
                "skipped": True,
                "reason": "No title or abstract provided",
            },
            tokens_used=0,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    if not cleaned_title:
        cleaned_title = "Untitled source"

    summary_text = cleaned_abstract or f"No abstract available for: {cleaned_title}"

    return ToolResult(
        tool_name="extract_source_summary",
        success=True,
        data={
            "title": cleaned_title,
            "summary": _clip(summary_text, 1000),
            "source_type": cleaned_source_type or "other",
            "platform": cleaned_platform or "web",
            "has_abstract": bool(cleaned_abstract),
        },
        tokens_used=0,
        latency_ms=(time.perf_counter() - started) * 1000,
    )


def register_read_tools(registry: ToolRegistry) -> None:
    if _jina_available():
        registry.register(
            ToolDefinition(
                name="read_url",
                description="Read and extract content from a URL using Jina Reader",
                category="read",
                token_cost_est=_READ_TOKEN_COST,
                timeout_seconds=_READ_TIMEOUT,
                requires_llm=False,
                parameters_schema={
                    "url": "string",
                },
            ),
            read_url,
        )

    registry.register(
        ToolDefinition(
            name="read_paper_abstract",
            description="Read and format a paper title and abstract for analysis",
            category="read",
            token_cost_est=500,
            timeout_seconds=10.0,
            requires_llm=False,
            parameters_schema={
                "title": "string",
                "abstract": "string",
                "url": "string",
            },
        ),
        read_paper_abstract,
    )

    registry.register(
        ToolDefinition(
            name="extract_source_summary",
            description="Extract a concise summary from a source title and abstract",
            category="read",
            token_cost_est=500,
            timeout_seconds=10.0,
            requires_llm=False,
            parameters_schema={
                "title": "string",
                "abstract": "string",
                "source_type": "string",
                "platform": "string",
            },
        ),
        extract_source_summary,
    )