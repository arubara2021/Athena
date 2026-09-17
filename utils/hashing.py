from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "model_dump"):
        return value.model_dump()

    if hasattr(value, "__dict__"):
        return value.__dict__

    return str(value)


def stable_json_payload(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=_json_default,
        )
    except Exception as exc:
        raise ValueError("Value could not be serialized for stable hashing") from exc


def stable_hash(value: Any, algorithm: str = "sha256") -> str:
    try:
        hasher = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from exc

    if isinstance(value, bytes):
        hasher.update(value)
    elif isinstance(value, str):
        hasher.update(value.encode("utf-8"))
    else:
        hasher.update(stable_json_payload(value).encode("utf-8"))

    return hasher.hexdigest()


def hash_text(text: str, algorithm: str = "sha256") -> str:
    return stable_hash(str(text), algorithm=algorithm)


def hash_dict(payload: dict[str, Any], algorithm: str = "sha256") -> str:
    return stable_hash(payload, algorithm=algorithm)


def hash_file(
    path: Path | str,
    algorithm: str = "sha256",
    chunk_size: int = 65536,
) -> str:
    file_path = Path(path).expanduser()

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        hasher = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from exc

    with open(file_path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)

    return hasher.hexdigest()