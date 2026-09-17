from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from core import constants
from core.exceptions import CacheError
from utils.hashing import stable_hash

_MISSING = object()


def build_cache_key(*parts: Any) -> str:
    return stable_hash(parts)


class MemoryCache:
    def __init__(
        self,
        default_ttl: float = constants.CACHE_DEFAULT_TTL_SECONDS,
        max_entries: int = constants.CACHE_MAX_ENTRIES,
    ) -> None:
        if default_ttl <= 0:
            raise ValueError("default_ttl must be positive")
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")

        self.default_ttl = default_ttl
        self.max_entries = max_entries
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.RLock()

    @staticmethod
    def _now() -> float:
        return time.time()

    def _is_expired(self, expires_at: float) -> bool:
        return expires_at <= self._now()

    def _purge_expired(self) -> None:
        for key in list(self._data.keys()):
            expires_at, _ = self._data[key]
            if self._is_expired(expires_at):
                del self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            self._purge_expired()
            item = self._data.get(key)

            if item is None:
                return default

            self._data.move_to_end(key)
            return item[1]

    def set(
        self,
        key: str,
        value: Any,
        ttl: float | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        if ttl_seconds is not None:
            ttl = ttl_seconds

        resolved_ttl = self.default_ttl if ttl is None else ttl

        if resolved_ttl <= 0:
            raise ValueError("ttl must be positive")

        expires_at = self._now() + resolved_ttl

        with self._lock:
            self._purge_expired()

            if key in self._data:
                self._data.move_to_end(key)

            self._data[key] = (expires_at, value)

            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def contains(self, key: str) -> bool:
        with self._lock:
            self._purge_expired()
            return key in self._data

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def size(self) -> int:
        with self._lock:
            self._purge_expired()
            return len(self._data)


class DiskCache:
    def __init__(
        self,
        directory: Path | str,
        default_ttl: float = constants.CACHE_DEFAULT_TTL_SECONDS,
        namespace: str = constants.CACHE_NAMESPACE,
    ) -> None:
        if default_ttl <= 0:
            raise ValueError("default_ttl must be positive")

        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self.namespace = namespace
        self._lock = threading.RLock()

    def _key_hash(self, key: str) -> str:
        return stable_hash((self.namespace, key))

    def _path_for_key(self, key: str) -> Path:
        return self.directory / f"{self._key_hash(key)}.json"

    def get(self, key: str, default: Any = None) -> Any:
        path = self._path_for_key(key)

        if not path.exists():
            return default

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            expires_at = float(payload.get("expires_at", 0))

            if expires_at <= time.time():
                self.delete(key)
                return default

            return payload.get("value", default)
        except Exception:
            self.delete(key)
            return default

    def set(
        self,
        key: str,
        value: Any,
        ttl: float | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        if ttl_seconds is not None:
            ttl = ttl_seconds

        resolved_ttl = self.default_ttl if ttl is None else ttl

        if resolved_ttl <= 0:
            raise ValueError("ttl must be positive")

        payload = {
            "expires_at": time.time() + resolved_ttl,
            "value": value,
        }

        path = self._path_for_key(key)
        tmp_path = None

        with self._lock:
            try:
                fd, tmp_path = tempfile.mkstemp(dir=self.directory, suffix=".tmp")

                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, default=str)

                os.replace(tmp_path, path)
            except Exception as exc:
                if tmp_path is not None:
                    try:
                        Path(tmp_path).unlink(missing_ok=True)
                    except Exception:
                        pass

                raise CacheError(
                    "Failed to write cache entry",
                    details={"key": key, "error": str(exc)},
                ) from exc

    def delete(self, key: str) -> None:
        path = self._path_for_key(key)

        with self._lock:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    def contains(self, key: str) -> bool:
        return self.get(key, default=_MISSING) is not _MISSING

    def clear(self) -> None:
        with self._lock:
            for file in self.directory.glob("*.json"):
                try:
                    file.unlink(missing_ok=True)
                except Exception:
                    pass

            for file in self.directory.glob("*.tmp"):
                try:
                    file.unlink(missing_ok=True)
                except Exception:
                    pass


class CacheRegistry:
    def __init__(self) -> None:
        self._caches: dict[str, MemoryCache] = {}
        self._lock = threading.RLock()

    def get(
        self,
        name: str,
        default_ttl: float = constants.CACHE_DEFAULT_TTL_SECONDS,
        max_entries: int = constants.CACHE_MAX_ENTRIES,
    ) -> MemoryCache:
        with self._lock:
            cache = self._caches.get(name)

            if cache is None:
                cache = MemoryCache(
                    default_ttl=default_ttl,
                    max_entries=max_entries,
                )
                self._caches[name] = cache

            return cache

    def clear(self, name: str | None = None) -> None:
        with self._lock:
            if name is None:
                for cache in self._caches.values():
                    cache.clear()
                return

            cache = self._caches.get(name)
            if cache is not None:
                cache.clear()

    def clear_all(self) -> None:
        self.clear(None)


default_cache_registry = CacheRegistry()