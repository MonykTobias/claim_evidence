"""Typed errors a caller can map without reading messages.

Driver exceptions and model response bodies stay inside the package. A caller
gets a category and a safe sentence, because a connection string or a prompt
echoed into an HTTP error is a leak, not a diagnostic.
"""

from __future__ import annotations


class ClaimEvidenceError(Exception):
    """Base class for every error this package raises on purpose."""


class NotFoundError(ClaimEvidenceError):
    """A requested document, audit, or evidence unit does not exist."""


class ValidationError(ClaimEvidenceError):
    """The caller's arguments are wrong; no state was changed."""


class DependencyUnavailableError(ClaimEvidenceError):
    """PostgreSQL, pgvector, or Ollama could not be reached or used."""


class IndexNotReadyError(ClaimEvidenceError):
    """The schema is missing, or no ready document version exists to query."""


__all__ = [
    "ClaimEvidenceError",
    "DependencyUnavailableError",
    "IndexNotReadyError",
    "NotFoundError",
    "ValidationError",
]
