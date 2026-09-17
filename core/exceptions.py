from __future__ import annotations

from typing import Any


class ResearchAgentError(Exception):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class ConfigError(ResearchAgentError):
    pass


class SettingsValidationError(ConfigError):
    pass


class MissingAPIKeyError(ConfigError):
    def __init__(
        self,
        message: str,
        provider: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        payload = details or {}
        payload["provider"] = provider
        super().__init__(message, details=payload)
        self.provider = provider


class ProviderError(ResearchAgentError):
    def __init__(
        self,
        message: str,
        provider: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        payload = details or {}
        payload["provider"] = provider
        payload["status_code"] = status_code
        super().__init__(message, details=payload)
        self.provider = provider
        self.status_code = status_code


class ProviderAuthError(ProviderError):
    pass


class ProviderRateLimitError(ProviderError):
    def __init__(
        self,
        message: str,
        provider: str | None = None,
        status_code: int | None = None,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        payload = details or {}
        payload["retry_after"] = retry_after
        super().__init__(message, provider=provider, status_code=status_code, details=payload)
        self.retry_after = retry_after


class ProviderTimeoutError(ProviderError):
    pass


class ProviderResponseError(ProviderError):
    pass


class ModelNotFoundError(ProviderError):
    pass


class LLMRequestError(ResearchAgentError):
    pass


class LLMResponseParseError(ResearchAgentError):
    pass


class EnsembleError(ResearchAgentError):
    pass


class ConsensusError(EnsembleError):
    pass


class SearchError(ResearchAgentError):
    pass


class SourceFetchError(SearchError):
    pass


class QueryExpansionError(SearchError):
    pass


class NormalizationError(SearchError):
    pass


class DeduplicationError(SearchError):
    pass


class ValidationError(SearchError):
    pass


class RankingError(ResearchAgentError):
    pass


class LearningPathError(RankingError):
    pass


class StorageError(ResearchAgentError):
    pass


class CacheError(ResearchAgentError):
    pass


class LoggingError(ResearchAgentError):
    pass