"""Domain-level exceptions translated to HTTP responses."""

from __future__ import annotations


class DomainError(Exception):
    """Base class for predictable application errors."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class NotFoundError(DomainError):
    """Entity missing."""

    def __init__(self, message: str = "Resource not found.") -> None:
        super().__init__(message, status_code=404)


class ValidationDomainError(DomainError):
    """Input violated business rules."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=422)
