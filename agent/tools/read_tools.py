from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.tools.registry import ToolRegistry, ToolDefinition
from agent.tools.executor import ToolResult
from clients.jina_client import JinaClient
from core.models import Source
from utils.logger import get_logger
from utils.text import clean_text, truncate_text

_logger = get_logger("agent.tools.read")

_READ_TIMEOUT = 45.0
_READ_TOKEN_COST = 2000


async def read_url(url: str) -> ToolResult:
    try:
        cleaned_url = clean_text(url)

        if not cleaned_url:
            return ToolResult(
                tool_name="read_url",
                success=False,
                data=None,
                error="Empty URL provided",
            )

        async with JinaClient() as client:
            source = await client.read_url(cleaned_url)

        if source is None:
            return ToolResult(
                tool_name="read_url",
                success=False,
                data=None,
                error="Failed to read URL content",
            )

        content = source.abstract or ""
        metadata = source.metadata if isinstance(source.metadata, dict) else {}
        full_content = metadata.get("full_content", "")

        return ToolResult(
            tool_name="read_url",
            success=True,
            data={
                "title": source.title,
                "url": source.url,
                "content_preview": truncate_text(content, max_length=3000, suffix=""),
                "full_content": truncate_text(full_content, max_length=10000, suffix=""),
                "content_length": len(full_content),
            },
            tokens_used=0,
        )

    except Exception as exc:
        _logger.warning(f"read_url failed: {exc}")
        return ToolResult(
            tool_name="read_url",
            success=False,
            data=None,
            error=str(exc),
        )


async def read_paper_abstract(
    title: str,
    abstract: str,
    url: str = "",
) -> ToolResult:
    try:
        cleaned_title = clean_text(title)
        cleaned_abstract = clean_text(abstract)

        if not cleaned_title and not cleaned_abstract:
            return ToolResult(
                tool_name="read_paper_abstract",
                success=False,
                data=None,
                error="No title or abstract provided",
            )

        content = f"Title: {cleaned_title}\nAbstract: {cleaned_abstract}"

        return ToolResult(
            tool_name="read_paper_abstract",
            success=True,
            data={
                "title": cleaned_title,
                "abstract": cleaned_abstract,
                "url": url,
                "combined_text": truncate_text(content, max_length=5000, suffix=""),
            },
            tokens_used=0,
        )

    except Exception as exc:
        _logger.warning(f"read_paper_abstract failed: {exc}")
        return ToolResult(
            tool_name="read_paper_abstract",
            success=False,
            data=None,
            error=str(exc),
        )


async def extract_source_summary(
    title: str,
    abstract: str,
    source_type: str = "",
    platform: str = "",
) -> ToolResult:
    try:
        cleaned_title = clean_text(title)
        cleaned_abstract = clean_text(abstract)

        if not cleaned_title:
            return ToolResult(
                tool_name="extract_source_summary",
                success=False,
                data=None,
                error="No title provided",
            )

        summary_text = cleaned_abstract if cleaned_abstract else f"No abstract available for: {cleaned_title}"

        return ToolResult(
            tool_name="extract_source_summary",
            success=True,
            data={
                "title": cleaned_title,
                "summary": truncate_text(summary_text, max_length=1000, suffix=""),
                "source_type": source_type,
                "platform": platform,
                "has_abstract": bool(cleaned_abstract),
            },
            tokens_used=0,
        )

    except Exception as exc:
        _logger.warning(f"extract_source_summary failed: {exc}")
        return ToolResult(
            tool_name="extract_source_summary",
            success=False,
            data=None,
            error=str(exc),
        )


def _jina_available() -> bool:
    try:
        from core.config import get_settings

        settings = get_settings()
        key = settings.jina_api_key.get_secret_value().strip()

        if not key:
            return False

        import yaml

        config_path = Path(__file__).resolve().parents[2] / "configs" / "sources.yaml"

        if config_path.exists():
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            platforms = raw.get("platforms", {}) if isinstance(raw, dict) else {}
            jina_config = platforms.get("jina", {}) if isinstance(platforms, dict) else {}

            if isinstance(jina_config, dict) and not bool(jina_config.get("enabled", True)):
                return False

        return True

    except Exception:
        return False


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
            ),
            read_url,
        )

    registry.register(
        ToolDefinition(
            name="read_paper_abstract",
            description="Read and format a paper's title and abstract for analysis",
            category="read",
            token_cost_est=500,
            timeout_seconds=10.0,
            requires_llm=False,
        ),
        read_paper_abstract,
    )

    registry.register(
        ToolDefinition(
            name="extract_source_summary",
            description="Extract a concise summary from a source's title and abstract",
            category="read",
            token_cost_est=500,
            timeout_seconds=10.0,
            requires_llm=False,
        ),
        extract_source_summary,
    )