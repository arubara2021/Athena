from __future__ import annotations

import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from core.exceptions import StorageError
from utils.logger import get_logger


class FileManager:
    def __init__(self, base_directory: Path | str) -> None:
        self._base_directory = Path(base_directory).expanduser()
        self._logger = get_logger("storage.file_manager")
        self._ensure_directory(self._base_directory)

    @property
    def base_directory(self) -> Path:
        return self._base_directory

    def _ensure_directory(self, directory: Path) -> None:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise StorageError(
                "Failed to create directory",
                details={"directory": str(directory), "error": str(exc)},
            ) from exc

    def sanitize_filename(self, name: str, max_length: int = 120) -> str:
        cleaned = re.sub(r"[^\w\s\-\.]", "", str(name)).strip()
        cleaned = re.sub(r"\s+", "_", cleaned)
        cleaned = cleaned.strip("._")

        if not cleaned:
            cleaned = "unnamed"

        return cleaned[:max_length]

    def build_path(self, *parts: str, subdirectory: str | None = None) -> Path:
        base = self._base_directory

        if subdirectory:
            base = base / self.sanitize_filename(subdirectory)
            self._ensure_directory(base)

        return base.joinpath(*[self.sanitize_filename(part) for part in parts])

    def timestamped_filename(self, prefix: str, extension: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_prefix = self.sanitize_filename(prefix)
        safe_extension = extension.lstrip(".")
        return f"{safe_prefix}_{timestamp}.{safe_extension}"

    def write_text(self, path: Path, content: str, encoding: str = "utf-8") -> Path:
        try:
            self._ensure_directory(path.parent)
            fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")

            try:
                with os.fdopen(fd, "w", encoding=encoding) as handle:
                    handle.write(content)
                os.replace(tmp_path, path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                raise

            return path

        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(
                "Failed to write file",
                details={"path": str(path), "error": str(exc)},
            ) from exc

    def write_bytes(self, path: Path, content: bytes) -> Path:
        try:
            self._ensure_directory(path.parent)
            fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")

            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(content)
                os.replace(tmp_path, path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                raise

            return path

        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(
                "Failed to write bytes",
                details={"path": str(path), "error": str(exc)},
            ) from exc

    def read_text(self, path: Path, encoding: str = "utf-8") -> str:
        try:
            return path.read_text(encoding=encoding)
        except Exception as exc:
            raise StorageError(
                "Failed to read file",
                details={"path": str(path), "error": str(exc)},
            ) from exc

    def exists(self, path: Path) -> bool:
        return path.exists()

    def list_files(self, subdirectory: str | None = None, pattern: str = "*") -> list[Path]:
        base = self._base_directory

        if subdirectory:
            base = base / self.sanitize_filename(subdirectory)

        if not base.exists():
            return []

        try:
            return sorted(base.glob(pattern))
        except Exception:
            return []