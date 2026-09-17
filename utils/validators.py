from __future__ import annotations

import json
from collections.abc import Callable
from enum import Enum
from typing import Any

from utils.url import is_valid_url


def validate_non_empty_string(
    value: Any,
    field_name: str,
    min_length: int = 1,
    max_length: int | None = None,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")

    cleaned = " ".join(value.split()).strip()

    if len(cleaned) < min_length:
        raise ValueError(f"{field_name} must be at least {min_length} characters")

    if max_length is not None and len(cleaned) > max_length:
        raise ValueError(f"{field_name} must be at most {max_length} characters")

    return cleaned


def validate_optional_string(
    value: Any,
    field_name: str,
    min_length: int = 1,
    max_length: int | None = None,
) -> str | None:
    if value is None:
        return None

    return validate_non_empty_string(
        value,
        field_name=field_name,
        min_length=min_length,
        max_length=max_length,
    )


def validate_int_range(
    value: Any,
    field_name: str,
    min_value: int | None = None,
    max_value: int | None = None,
    optional: bool = False,
) -> int | None:
    if optional and value is None:
        return None

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")

    if isinstance(value, int):
        numeric_value = value
    elif isinstance(value, str) and value.strip().lstrip("-").isdigit():
        numeric_value = int(value.strip())
    else:
        raise ValueError(f"{field_name} must be an integer")

    if min_value is not None and numeric_value < min_value:
        raise ValueError(f"{field_name} must be greater than or equal to {min_value}")

    if max_value is not None and numeric_value > max_value:
        raise ValueError(f"{field_name} must be less than or equal to {max_value}")

    return numeric_value


def validate_float_range(
    value: Any,
    field_name: str,
    min_value: float | None = None,
    max_value: float | None = None,
    optional: bool = False,
) -> float | None:
    if optional and value is None:
        return None

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a float")

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a float") from exc

    if min_value is not None and numeric_value < min_value:
        raise ValueError(f"{field_name} must be greater than or equal to {min_value}")

    if max_value is not None and numeric_value > max_value:
        raise ValueError(f"{field_name} must be less than or equal to {max_value}")

    return numeric_value


def validate_url(value: Any, field_name: str = "url") -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")

    cleaned = value.strip()

    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")

    if not is_valid_url(cleaned):
        raise ValueError(f"{field_name} must be a valid HTTP or HTTPS URL")

    return cleaned


def validate_optional_url(value: Any, field_name: str = "url") -> str | None:
    if value is None:
        return None

    return validate_url(value, field_name=field_name)


def validate_year(value: Any, optional: bool = True) -> int | None:
    return validate_int_range(
        value,
        field_name="year",
        min_value=0,
        max_value=2100,
        optional=optional,
    )


def validate_enum_value(
    value: Any,
    enum_class: type[Enum],
    field_name: str,
    optional: bool = False,
) -> Any:
    if optional and value is None:
        return None

    if isinstance(value, enum_class):
        return value

    try:
        return enum_class(str(value))
    except Exception as exc:
        valid_values = ", ".join(str(item.value) for item in enum_class)
        raise ValueError(
            f"{field_name} must be one of: {valid_values}"
        ) from exc


def validate_list(
    value: Any,
    field_name: str,
    item_validator: Callable[[Any], Any] | None = None,
) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")

    if item_validator is None:
        return list(value)

    return [item_validator(item) for item in value]


def validate_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a dictionary")

    return dict(value)


def validate_boolean(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        cleaned = value.strip().lower()

        if cleaned in {"true", "1", "yes", "y", "on"}:
            return True

        if cleaned in {"false", "0", "no", "n", "off"}:
            return False

    if isinstance(value, int) and value in {0, 1}:
        return bool(value)

    raise ValueError(f"{field_name} must be a boolean")


def validate_optional(value: Any, validator: Callable[[Any], Any]) -> Any:
    if value is None:
        return None

    return validator(value)


def validate_json_payload(value: Any) -> dict[str, Any] | list[Any]:
    if isinstance(value, (dict, list)):
        return value

    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")

    if not isinstance(value, str):
        raise ValueError("JSON payload must be a string, bytes, dict, or list")

    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON payload") from exc

    if not isinstance(payload, (dict, list)):
        raise ValueError("JSON payload must be an object or array")

    return payload