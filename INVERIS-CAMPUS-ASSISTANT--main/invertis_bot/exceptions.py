"""Domain-specific exceptions for clear, recoverable error handling."""


class InvertisBotError(Exception):
    """Base exception for expected application failures."""


class ScrapingError(InvertisBotError):
    """Raised when public website content cannot be collected safely."""


class IndexNotFoundError(InvertisBotError):
    """Raised when a persistent vector index has not been built yet."""


class LLMError(InvertisBotError):
    """Raised when a model request fails or returns unusable output."""


class DocumentLoadError(InvertisBotError):
    """Raised when a local document cannot be parsed."""
