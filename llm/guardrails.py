from __future__ import annotations

import re
from typing import Any

from core import constants
from core.exceptions import LLMRequestError, LLMResponseParseError
from core.schemas import LLMMessageSchema, LLMRequestSchema, LLMResponseSchema
from utils.text import clean_text, truncate_text

MAX_MESSAGE_CHARS = 120000

_SECRET_PATTERNS = (
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9_\-\.=+/]{16,}"), r"\1***"),
    (
        re.compile(
            r"(?i)((api[_-]?key|token|secret|password|authorization)\s*[:=]\s*)['\"]?[A-Za-z0-9_\-\.=+/]{8,}['\"]?"
        ),
        r"\1***",
    ),
    (re.compile(r"(?i)(nvapi-|gsk_|sk-|AIza)[A-Za-z0-9_\-]{8,}"), "***"),
)


def redact_secrets(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def contains_secrets(value: Any) -> bool:
    text = str(value or "")
    if not text:
        return False
    for pattern, _ in _SECRET_PATTERNS:
        if pattern.search(text):
            return True
    return False


def sanitize_prompt(value: Any, max_length: int = constants.MAX_LLM_PROMPT_LENGTH) -> str:
    text = clean_text(value)
    text = redact_secrets(text)
    return truncate_text(text, max_length=max_length, suffix="")


def sanitize_llm_output(value: Any) -> str:
    text = clean_text(value)
    return redact_secrets(text)


def validate_temperature(value: Any) -> float:
    try:
        temperature = float(value)
    except Exception as exc:
        raise LLMRequestError("temperature must be a float") from exc
    if temperature < constants.MIN_TEMPERATURE:
        return constants.MIN_TEMPERATURE
    if temperature > constants.MAX_TEMPERATURE:
        return constants.MAX_TEMPERATURE
    return temperature


def validate_max_tokens(value: Any) -> int:
    try:
        max_tokens = int(value)
    except Exception as exc:
        raise LLMRequestError("max_tokens must be an integer") from exc
    if max_tokens < constants.MIN_MAX_TOKENS:
        return constants.MIN_MAX_TOKENS
    if max_tokens > constants.MAX_MAX_TOKENS:
        return constants.MAX_MAX_TOKENS
    return max_tokens


def validate_messages(messages: list[LLMMessageSchema]) -> list[LLMMessageSchema]:
    if not messages:
        raise LLMRequestError("LLM messages cannot be empty")

    sanitized: list[LLMMessageSchema] = []
    for message in messages:
        content = clean_text(message.content)
        content = redact_secrets(content)

        if not content:
            raise LLMRequestError("LLM message content cannot be empty")

        if len(content) > MAX_MESSAGE_CHARS:
            content = content[:MAX_MESSAGE_CHARS]

        sanitized.append(
            LLMMessageSchema(
                role=message.role,
                content=content,
            )
        )
    return sanitized


def validate_llm_request(request: LLMRequestSchema) -> LLMRequestSchema:
    if not str(request.model or "").strip():
        raise LLMRequestError("LLM model cannot be empty")

    request.messages = validate_messages(request.messages)
    request.temperature = validate_temperature(request.temperature)
    request.max_tokens = validate_max_tokens(request.max_tokens)
    return request


def validate_llm_response(response: LLMResponseSchema) -> LLMResponseSchema:
    response.content = sanitize_llm_output(response.content)
    if not response.content.strip():
        raise LLMResponseParseError(
            "LLM response content is empty after sanitization",
            details={"provider": str(response.provider), "model": response.model},
        )
    return response