from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.exceptions import StorageError
from storage.file_manager import FileManager
from utils.logger import get_logger


class JSONWriter:
    def __init__(self, file_manager: FileManager, indent: int = 2) -> None:
        self._file_manager = file_manager
        self._indent = indent
        self._logger = get_logger("storage.json_writer")

    def serialize(self, data: Any) -> str:
        try:
            payload = self._to_serializable(data)
            return json.dumps(
                payload,
                ensure_ascii=False,
                indent=self._indent,
                default=str,
            )
        except Exception as exc:
            raise StorageError(
                "Failed to serialize data to JSON",
                details={"error": str(exc)},
            ) from exc

    def _to_serializable(self, data: Any) -> Any:
        if hasattr(data, "model_dump"):
            return data.model_dump(mode="json")

        if isinstance(data, list):
            return [self._to_serializable(item) for item in data]

        if isinstance(data, dict):
            return {key: self._to_serializable(value) for key, value in data.items()}

        return data

    def write(
        self,
        data: Any,
        filename: str,
        subdirectory: str | None = None,
    ) -> Path:
        content = self.serialize(data)
        path = self._file_manager.build_path(filename, subdirectory=subdirectory)
        return self._file_manager.write_text(path, content)

    def write_with_generated_name(
        self,
        data: Any,
        prefix: str,
        subdirectory: str | None = None,
    ) -> Path:
        filename = self._file_manager.timestamped_filename(prefix, "json")
        return self.write(data, filename, subdirectory=subdirectory)