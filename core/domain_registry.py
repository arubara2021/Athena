from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.logger import get_logger

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
_DEFAULT_DOMAINS_PATH = _CONFIG_DIR / "domains.yaml"
_DEFAULT_SOURCES_PATH = _CONFIG_DIR / "sources.yaml"
_DOMAIN_PATTERN = re.compile(r"[^a-z0-9_]+")


@dataclass(frozen=True)
class DomainRoutingSettings:
    enabled: bool
    min_confidence: float
    max_domains: int
    max_platforms_per_domain: int
    min_platforms_before_universal: int
    universal_on_low_confidence: bool
    use_format_filtering: bool


@dataclass(frozen=True)
class DomainDefinition:
    name: str
    label: str
    priority: int
    platforms: tuple[str, ...]
    fallback_domains: tuple[str, ...]
    formats: tuple[str, ...]
    aliases: tuple[str, ...]


class DomainRegistry:
    def __init__(
        self,
        domains_path: Path | str | None = None,
        sources_path: Path | str | None = None,
    ) -> None:
        self._domains_path = Path(domains_path or _DEFAULT_DOMAINS_PATH)
        self._sources_path = Path(sources_path or _DEFAULT_SOURCES_PATH)
        self._logger = get_logger("core.domain_registry")
        self._settings = self._default_settings()
        self._domains: dict[str, DomainDefinition] = {}
        self._aliases: dict[str, str] = {}
        self._universal_platforms: tuple[str, ...] = ()
        self._enabled_platforms: dict[str, bool] = {}
        self._load()

    @staticmethod
    def _default_settings() -> DomainRoutingSettings:
        return DomainRoutingSettings(
            enabled=True,
            min_confidence=0.35,
            max_domains=3,
            max_platforms_per_domain=12,
            min_platforms_before_universal=4,
            universal_on_low_confidence=True,
            use_format_filtering=True,
        )

    def _load(self) -> None:
        raw = self._read_yaml(self._domains_path)
        sources = self._read_yaml(self._sources_path)

        self._load_enabled_platforms(sources)

        if not isinstance(raw, dict):
            self._logger.warning(
                "Domain registry configuration missing; domain routing will use safe defaults"
            )
            return

        self._load_settings(raw.get("routing"))
        self._universal_platforms = tuple(
            self._clean_strings(raw.get("universal_platforms"))
        )

        domains_payload = raw.get("domains")
        if isinstance(domains_payload, dict):
            for name, payload in domains_payload.items():
                self._load_domain(str(name), payload)

        if not self._domains:
            self._logger.warning("Domain registry contains no usable domains")

    def _read_yaml(self, path: Path) -> dict[str, Any]:
        try:
            import yaml
        except Exception as exc:
            self._logger.warning(f"PyYAML unavailable while loading domain registry: {exc}")
            return {}

        try:
            if not path.exists():
                return {}

            with open(path, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}

            if isinstance(loaded, dict):
                return loaded
        except Exception as exc:
            self._logger.warning(f"Failed to load domain registry YAML from {path}: {exc}")

        return {}

    def _load_settings(self, payload: Any) -> None:
        defaults = self._default_settings()

        if not isinstance(payload, dict):
            self._settings = defaults
            return

        self._settings = DomainRoutingSettings(
            enabled=self._as_bool(payload.get("enabled"), defaults.enabled),
            min_confidence=self._as_float(
                payload.get("min_confidence"),
                defaults.min_confidence,
            ),
            max_domains=self._as_int(payload.get("max_domains"), defaults.max_domains),
            max_platforms_per_domain=self._as_int(
                payload.get("max_platforms_per_domain"),
                defaults.max_platforms_per_domain,
            ),
            min_platforms_before_universal=self._as_int(
                payload.get("min_platforms_before_universal"),
                defaults.min_platforms_before_universal,
            ),
            universal_on_low_confidence=self._as_bool(
                payload.get("universal_on_low_confidence"),
                defaults.universal_on_low_confidence,
            ),
            use_format_filtering=self._as_bool(
                payload.get("use_format_filtering"),
                defaults.use_format_filtering,
            ),
        )

    def _load_enabled_platforms(self, sources: dict[str, Any]) -> None:
        platforms = sources.get("platforms") if isinstance(sources, dict) else None

        if not isinstance(platforms, dict):
            self._enabled_platforms = {}
            return

        enabled_map: dict[str, bool] = {}

        for name, config in platforms.items():
            platform = str(name or "").strip().lower()
            if not platform:
                continue

            enabled = True
            if isinstance(config, dict):
                enabled = self._as_bool(config.get("enabled"), True)

            enabled_map[platform] = enabled

        self._enabled_platforms = enabled_map

    def _load_domain(self, name: str, payload: Any) -> None:
        domain_name = self.normalize_domain(name)

        if not domain_name or not isinstance(payload, dict):
            return

        platforms = tuple(self._clean_strings(payload.get("platforms")))
        if not platforms:
            return

        fallback_domains = tuple(
            self.normalize_domain(item)
            for item in self._clean_strings(payload.get("fallback_domains"))
            if self.normalize_domain(item)
        )

        formats = tuple(self._clean_strings(payload.get("formats")))

        aliases = tuple(
            self.normalize_domain(item)
            for item in self._clean_strings(payload.get("aliases"))
            if self.normalize_domain(item)
        )

        label = str(payload.get("label") or name).strip()
        priority = self._as_int(payload.get("priority"), 100)

        definition = DomainDefinition(
            name=domain_name,
            label=label,
            priority=priority,
            platforms=platforms,
            fallback_domains=fallback_domains,
            formats=formats,
            aliases=aliases,
        )

        self._domains[domain_name] = definition
        self._aliases[domain_name] = domain_name

        if label:
            self._aliases[self.normalize_domain(label)] = domain_name

        for alias in aliases:
            self._aliases[alias] = domain_name

    def normalize_domain(self, value: Any) -> str:
        text = str(value or "").strip().lower()

        if not text:
            return ""

        text = text.replace("-", "_").replace(" ", "_")
        text = _DOMAIN_PATTERN.sub("_", text)
        text = "_".join(part for part in text.split("_") if part)

        return text

    def _clean_strings(self, value: Any) -> list[str]:
        if value is None:
            return []

        if isinstance(value, str):
            value = [item.strip() for item in value.split(",")]

        if not isinstance(value, (list, tuple, set)):
            return []

        cleaned: list[str] = []

        for item in value:
            text = str(item or "").strip().lower()
            if text and text not in cleaned:
                cleaned.append(text)

        return cleaned

    @staticmethod
    def _as_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return bool(value)

        if isinstance(value, str):
            text = value.strip().lower()

            if text in {"true", "yes", "on", "1"}:
                return True

            if text in {"false", "no", "off", "0"}:
                return False

        return default

    @staticmethod
    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return default

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    @property
    def settings(self) -> DomainRoutingSettings:
        return self._settings

    @property
    def domain_names(self) -> list[str]:
        return sorted(self._domains.keys())

    def has_domain(self, value: Any) -> bool:
        return self.normalize_domain(value) in self._aliases

    def resolve_domains(self, values: Any) -> list[str]:
        if values is None:
            return []

        if isinstance(values, str):
            values = [item.strip() for item in values.split(",")]

        if not isinstance(values, (list, tuple, set)):
            values = [values]

        resolved: list[str] = []

        for value in values:
            domain = self.normalize_domain(value)
            if not domain:
                continue

            canonical = self._aliases.get(domain)
            if canonical and canonical not in resolved:
                resolved.append(canonical)

        return resolved

    def get_domain(self, value: Any) -> DomainDefinition | None:
        domain = self.normalize_domain(value)

        if not domain:
            return None

        canonical = self._aliases.get(domain)
        if canonical is None:
            return None

        return self._domains.get(canonical)

    def is_platform_enabled(self, platform: Any) -> bool:
        name = str(platform or "").strip().lower()

        if not name:
            return False

        return self._enabled_platforms.get(name, True)

    def get_universal_platforms(self, include_disabled: bool = False) -> list[str]:
        platforms = list(self._universal_platforms)

        if include_disabled:
            return platforms

        return [platform for platform in platforms if self.is_platform_enabled(platform)]

    def get_platforms_for_domain(
        self,
        domain: Any,
        include_disabled: bool = False,
    ) -> list[str]:
        definition = self.get_domain(domain)

        if definition is None:
            return []

        platforms = list(definition.platforms)

        if include_disabled:
            return platforms

        return [platform for platform in platforms if self.is_platform_enabled(platform)]

    def get_fallback_domains(self, domains: Any) -> list[str]:
        resolved = self.resolve_domains(domains)
        fallbacks: list[str] = []

        for domain in resolved:
            definition = self._domains.get(domain)
            if definition is None:
                continue

            for fallback in definition.fallback_domains:
                canonical = self._aliases.get(fallback, fallback)

                if canonical in self._domains and canonical not in resolved and canonical not in fallbacks:
                    fallbacks.append(canonical)

        return fallbacks

    def get_fallback_platforms(
        self,
        domains: Any,
        include_disabled: bool = False,
    ) -> list[str]:
        fallback_domains = self.get_fallback_domains(domains)

        return self.get_platforms_for_domains(
            fallback_domains,
            include_fallback=False,
            include_universal=False,
            include_disabled=include_disabled,
        )

    def get_formats_for_domains(self, domains: Any) -> list[str]:
        resolved = self.resolve_domains(domains)
        formats: list[str] = []

        for domain in resolved:
            definition = self._domains.get(domain)
            if definition is None:
                continue

            for source_format in definition.formats:
                if source_format not in formats:
                    formats.append(source_format)

        return formats

    def get_platforms_for_domains(
        self,
        domains: Any,
        include_fallback: bool = True,
        include_universal: bool = False,
        min_platforms: int | None = None,
        max_platforms: int | None = None,
        include_disabled: bool = False,
    ) -> list[str]:
        selected: list[str] = []
        seen: set[str] = set()

        def add_platforms(platforms: list[str]) -> None:
            for platform in platforms:
                if platform in seen:
                    continue

                if not include_disabled and not self.is_platform_enabled(platform):
                    continue

                seen.add(platform)
                selected.append(platform)

        resolved = self.resolve_domains(domains)

        if not resolved:
            if include_universal:
                add_platforms(self.get_universal_platforms(include_disabled=True))

            return self._limit_platforms(selected, max_platforms)

        queue = list(resolved)
        processed: set[str] = set()

        while queue:
            domain = queue.pop(0)

            if domain in processed:
                continue

            processed.add(domain)
            add_platforms(self.get_platforms_for_domain(domain, include_disabled=True))

            if include_fallback:
                definition = self._domains.get(domain)

                if definition is not None:
                    for fallback in definition.fallback_domains:
                        canonical = self._aliases.get(fallback, fallback)

                        if canonical in self._domains and canonical not in processed:
                            queue.append(canonical)

        resolved_min = self._as_int(min_platforms, 0) if min_platforms is not None else 0

        if include_universal and len(selected) < max(1, resolved_min):
            add_platforms(self.get_universal_platforms(include_disabled=True))

        return self._limit_platforms(selected, max_platforms)

    def _limit_platforms(self, platforms: list[str], max_platforms: Any) -> list[str]:
        limit = self._as_int(max_platforms, 0) if max_platforms is not None else 0

        if limit <= 0:
            return platforms

        return platforms[:limit]

    def to_summary(self) -> dict[str, Any]:
        return {
            "domains": self.domain_names,
            "settings": {
                "enabled": self._settings.enabled,
                "min_confidence": self._settings.min_confidence,
                "max_domains": self._settings.max_domains,
                "max_platforms_per_domain": self._settings.max_platforms_per_domain,
                "min_platforms_before_universal": self._settings.min_platforms_before_universal,
                "universal_on_low_confidence": self._settings.universal_on_low_confidence,
                "use_format_filtering": self._settings.use_format_filtering,
            },
            "universal_platforms": list(self._universal_platforms),
            "enabled_platforms": {
                platform: enabled
                for platform, enabled in sorted(self._enabled_platforms.items())
            },
        }


_registry: DomainRegistry | None = None
_registry_lock = threading.Lock()


def get_domain_registry() -> DomainRegistry:
    global _registry

    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = DomainRegistry()

    return _registry


def reload_domain_registry() -> DomainRegistry:
    global _registry

    with _registry_lock:
        _registry = DomainRegistry()

    return _registry