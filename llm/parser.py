from __future__ import annotations

import json
import re
from typing import Any

from core.exceptions import LLMResponseParseError

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def strip_code_fences(text: Any) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    matches = _FENCE_RE.findall(raw)
    if matches:
        return "\n".join(match.strip() for match in matches)
    return raw


def _repair_json_text(text: str) -> str:
    repaired = _CONTROL_RE.sub(" ", text)
    repaired = _TRAILING_COMMA_RE.sub(r"\1", repaired)
    return repaired


def _try_loads(candidate: str) -> Any:
    try:
        return json.loads(candidate)
    except Exception:
        repaired = _repair_json_text(candidate)
        if repaired != candidate:
            try:
                return json.loads(repaired)
            except Exception:
                return None
        return None


def _find_balanced_from(
    text: str,
    start_index: int,
    start_char: str,
    end_char: str,
) -> str | None:
    depth = 0
    in_string = False
    escape = False
    for index in range(start_index, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == start_char:
            depth += 1
        elif char == end_char:
            depth -= 1
            if depth == 0:
                return text[start_index:index + 1]
    return None


def _extract_first_json(text: str) -> str | None:
    object_start = text.find("{")
    array_start = text.find("[")
    candidates = [index for index in (object_start, array_start) if index != -1]
    if not candidates:
        return None
    start = min(candidates)
    if text[start] == "{":
        return _find_balanced_from(text, start, "{", "}")
    return _find_balanced_from(text, start, "[", "]")


def _extract_all_valid_json(text: str) -> list[str]:
    results: list[str] = []
    for start_char, end_char in (("{", "}"), ("[", "]")):
        search_start = 0
        while search_start < len(text):
            idx = text.find(start_char, search_start)
            if idx == -1:
                break
            found = _find_balanced_from(text, idx, start_char, end_char)
            if found and _try_loads(found) is not None:
                results.append(found)
            search_start = idx + 1
    return results


def extract_json_candidate(text: Any) -> str:
    cleaned = strip_code_fences(text).strip()
    if not cleaned:
        return ""

    if _try_loads(cleaned) is not None:
        return cleaned

    fence_blocks = _FENCE_RE.findall(cleaned)
    for block in fence_blocks:
        candidate = _extract_first_json(block.strip()) or block.strip()
        if _try_loads(candidate) is not None:
            return candidate

    candidate = _extract_first_json(cleaned)
    if candidate and _try_loads(candidate) is not None:
        return candidate

    all_valid = _extract_all_valid_json(cleaned)
    if all_valid:
        return max(all_valid, key=len)

    return candidate or cleaned


def parse_json_response(text: Any) -> Any:
    candidate = extract_json_candidate(text)
    parsed = _try_loads(candidate)
    if parsed is None:
        raise LLMResponseParseError(
            "Unable to parse JSON from LLM response",
            details={"preview": str(candidate)[:500]},
        )
    return parsed


def parse_json_object_response(text: Any) -> dict[str, Any]:
    payload = parse_json_response(text)

    if isinstance(payload, dict):
        return payload

    if isinstance(payload, list):
        dicts = [item for item in payload if isinstance(item, dict)]
        if len(dicts) == 1:
            return dicts[0]
        if dicts:
            return max(dicts, key=lambda d: len(json.dumps(d, default=str)))
        raise LLMResponseParseError(
            "LLM JSON array response contained no objects",
            details={"length": len(payload)},
        )

    if isinstance(payload, str):
        inner = _try_loads(payload)
        if isinstance(inner, dict):
            return inner
        if isinstance(inner, list):
            dicts = [item for item in inner if isinstance(item, dict)]
            if dicts:
                return dicts[0]

    raise LLMResponseParseError(
        "LLM JSON response must be an object",
        details={"type": type(payload).__name__},
    )


def parse_json_array_response(text: Any) -> list[Any]:
    payload = parse_json_response(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                return value
        return [payload]
    raise LLMResponseParseError(
        "LLM JSON response must be an array",
        details={"type": type(payload).__name__},
    )


def safe_parse_llm_json(text: Any, default: Any = None) -> Any:
    try:
        return parse_json_response(text)
    except Exception:
        return default


def parse_json_or_text(text: Any) -> Any:
    try:
        return parse_json_response(text)
    except Exception:
        return str(text or "")


def _extract_complete_objects_from_array(
    text: str,
    key: str,
) -> list[Any]:
    pattern = re.compile(
        rf'"{re.escape(key)}"\s*:\s*\[',
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        return []

    pos = match.end()
    objects: list[Any] = []

    while pos < len(text):
        while pos < len(text) and text[pos] in " \t\n\r,":
            pos += 1

        if pos >= len(text):
            break
        if text[pos] == "]":
            break
        if text[pos] != "{":
            break

        substring = _find_balanced_from(text, pos, "{", "}")
        if substring is None:
            break

        parsed_obj = _try_loads(substring)
        if parsed_obj is not None:
            objects.append(parsed_obj)

        pos += len(substring)

    return objects


def _count_unbalanced(text: str) -> int:
    opens = text.count("{") + text.count("[")
    closes = text.count("}") + text.count("]")
    return max(0, opens - closes)


def parse_truncated_json_object(
    text: Any,
) -> dict[str, Any] | None:
    cleaned = strip_code_fences(text).strip()
    if not cleaned:
        return None

    parsed = _try_loads(cleaned)
    if isinstance(parsed, dict) and parsed:
        return parsed

    for key in ("steps", "rankings", "sources"):
        items = _extract_complete_objects_from_array(cleaned, key)
        if items:
            return {key: items}

    stripped = cleaned.rstrip()
    if stripped.endswith((",", ":", "[", "{")):
        trimmed = stripped.rstrip(",:").rstrip()

        for suffix in ("}", "]}", "}}", "]} }"):
            candidate = trimmed + suffix
            parsed = _try_loads(candidate)
            if isinstance(parsed, dict) and parsed:
                return parsed

    unbalanced = _count_unbalanced(cleaned)
    if unbalanced > 0:
        for suffix in ("}" * unbalanced, "]" * unbalanced):
            candidate = cleaned + suffix
            parsed = _try_loads(candidate)
            if isinstance(parsed, dict) and parsed:
                return parsed

        for suffix in ("}]}", '"]} }', "}}"):
            candidate = cleaned + suffix
            parsed = _try_loads(candidate)
            if isinstance(parsed, dict) and parsed:
                return parsed

    return None