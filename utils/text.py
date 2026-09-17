from __future__ import annotations

import re
import unicodedata
from typing import Any

from core import constants
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")
_HTML_TAGS = re.compile(r"<[^>]*>")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def safe_str(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    return str(value)


def normalize_unicode(value: Any) -> str:
    return unicodedata.normalize("NFKC", safe_str(value))


def remove_control_chars(value: Any) -> str:
    return _CONTROL_CHARS.sub("", safe_str(value))


def normalize_whitespace(value: Any) -> str:
    return _WHITESPACE.sub(" ", safe_str(value)).strip()


def remove_html_tags(value: Any) -> str:
    return _HTML_TAGS.sub(" ", safe_str(value))


def clean_text(value: Any, normalize: bool = True, strip: bool = True) -> str:
    text = safe_str(value)

    if normalize:
        text = normalize_unicode(text)

    text = remove_control_chars(text)
    text = remove_html_tags(text)
    text = normalize_whitespace(text)

    if strip:
        text = text.strip()

    return text


def truncate_text(value: Any, max_length: int, suffix: str = "...") -> str:
    if max_length < 0:
        raise ValueError("max_length must be non-negative")

    text = safe_str(value)

    if len(text) <= max_length:
        return text

    if max_length <= len(suffix):
        return suffix[:max_length]

    return text[: max_length - len(suffix)] + suffix


def snippet(value: Any, max_length: int = constants.TEXT_SNIPPET_LENGTH) -> str:
    return truncate_text(clean_text(value), max_length=max_length)


def slugify(value: Any, max_length: int = 100) -> str:
    if max_length < 0:
        raise ValueError("max_length must be non-negative")

    text = normalize_unicode(value).lower().strip()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = _NON_ALNUM.sub("-", text).strip("-")

    return text[:max_length].rstrip("-")


def normalize_title(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9+\s]", " ", text)
    return _WHITESPACE.sub(" ", text).strip()


def normalize_newlines(value: Any) -> str:
    return safe_str(value).replace("\r\n", "\n").replace("\r", "\n")


def count_words(value: Any) -> int:
    text = clean_text(value)

    if not text:
        return 0

    return len(text.split())