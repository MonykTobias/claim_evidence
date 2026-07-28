"""Optional progress reporting for ingestion and auditing.

One callback, one emit path, no queue and no threads. A frontend passes a
callable and gets ordered phase events; passing nothing behaves exactly as
before.

A broken UI must never corrupt an index build, so a callback that raises is
disabled for the rest of the operation and the work continues.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .errors import (
    ClaimEvidenceError,
    DependencyUnavailableError,
    IndexNotReadyError,
    NotFoundError,
    ValidationError,
)
from .models import ProgressEvent

ProgressCallback = Callable[[ProgressEvent], None]

# Stable codes a frontend can branch on, with the minimum retry classification.
ERROR_CODES: dict[str, bool] = {
    "validation_error": False,
    "dependency_unavailable": True,
    "index_not_ready": True,
    "not_found": False,
    "internal_error": True,
}
INTERNAL_ERROR_MESSAGE = "An internal error interrupted the operation."
DEPENDENCY_ERROR_MESSAGE = "A required service was unavailable."


def classify_error(exc: BaseException) -> tuple[str, bool, str]:
    """Map an exception to ``(error_code, retryable, safe_message)``.

    Only the package's own typed errors are quoted back to the caller; their
    messages are written for public consumption. Anything else -- a driver
    error, a model response body, an unexpected bug -- is reported by category,
    because those messages carry hosts, credentials, prompts, and source text.
    """
    from .ollama import OllamaError

    if isinstance(exc, OllamaError):
        # Deliberately not str(exc): it embeds the model's response body to
        # make server-side debugging possible.
        return "dependency_unavailable", True, DEPENDENCY_ERROR_MESSAGE
    if isinstance(exc, ValidationError):
        return "validation_error", False, str(exc)
    if isinstance(exc, NotFoundError):
        return "not_found", False, str(exc)
    if isinstance(exc, IndexNotReadyError):
        return "index_not_ready", True, str(exc)
    if isinstance(exc, DependencyUnavailableError):
        return "dependency_unavailable", True, str(exc)
    if isinstance(exc, ClaimEvidenceError):
        return "internal_error", True, str(exc)
    return "internal_error", True, INTERNAL_ERROR_MESSAGE


class ProgressReporter:
    """Funnels every event through one place so the rules hold everywhere.

    Construct one per operation. ``callback=None`` makes every method a no-op,
    which is what keeps the un-instrumented path free.
    """

    def __init__(
        self,
        callback: ProgressCallback | None,
        operation: str,
        *,
        document_id: int | None = None,
        audit_id: int | None = None,
    ) -> None:
        self._callback = callback
        self.operation = operation
        self.document_id = document_id
        self.audit_id = audit_id
        self.started = time.monotonic()
        self.broken = False
        self.phase = ""

    @property
    def active(self) -> bool:
        return self._callback is not None and not self.broken

    @property
    def elapsed_seconds(self) -> float:
        return round(time.monotonic() - self.started, 3)

    def emit(
        self,
        phase: str,
        status: str,
        message: str,
        *,
        completed: int | None = None,
        total: int | None = None,
        current_item: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """The single emit path. Never raises."""
        self.phase = phase
        if not self.active:
            return
        event = ProgressEvent(
            operation=self.operation,
            phase=phase,
            status=status,
            message=message,
            completed=completed,
            total=total,
            document_id=self.document_id,
            audit_id=self.audit_id,
            current_item=current_item,
            details=details or {},
        )
        try:
            self._callback(event)  # type: ignore[misc]
        except Exception:
            # A UI that crashes is a UI problem. Stop talking to it and let the
            # index build finish.
            self.broken = True

    # --- convenience wrappers over the one emit path ------------------------

    def start(self, phase: str, message: str, *, total: int | None = None) -> None:
        self.emit(phase, "start", message, completed=0 if total else None, total=total)

    def step(
        self,
        phase: str,
        message: str,
        *,
        completed: int,
        total: int,
        current_item: str | None = None,
    ) -> None:
        self.emit(
            phase,
            "progress",
            message,
            completed=completed,
            total=total,
            current_item=current_item,
        )

    def done(
        self,
        phase: str,
        message: str,
        *,
        completed: int | None = None,
        total: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.emit(
            phase,
            "completed",
            message,
            completed=completed,
            total=total,
            details=details,
        )

    def warn(self, phase: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        self.emit(phase, "warning", message, details=details)

    def fail(self, exc: BaseException, *, phase: str | None = None) -> None:
        """Emit one safe terminal event. The caller re-raises afterwards."""
        failed_phase = phase or self.phase or "unknown"
        code, retryable, message = classify_error(exc)
        self.emit(
            failed_phase,
            "failed",
            message,
            details={
                "failed_phase": failed_phase,
                "error_code": code,
                "retryable": retryable,
                "elapsed_seconds": self.elapsed_seconds,
            },
        )


__all__ = [
    "DEPENDENCY_ERROR_MESSAGE",
    "ERROR_CODES",
    "INTERNAL_ERROR_MESSAGE",
    "ProgressCallback",
    "ProgressReporter",
    "classify_error",
]
