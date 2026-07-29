"""Environment-driven settings.

Every knob is a plain environment variable so the package can run against a
managed PostgreSQL and a remote Ollama without a config file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .errors import ValidationError

DEFAULT_DATABASE_URL = (
    "postgresql://claim_evidence:claim_evidence@localhost:5433/claim_evidence"
)
DEFAULT_DATABASE_CONNECT_TIMEOUT = 10.0
DEFAULT_BUILD_STALE_MINUTES = 60.0
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "qwen3-embedding:4b"
DEFAULT_EMBED_DIMENSIONS = 1024
DEFAULT_CHAT_MODEL = "hf.co/unsloth/Qwen3-VL-4B-Instruct-GGUF:UD-Q8_K_XL"
DEFAULT_VISION_MODEL = DEFAULT_CHAT_MODEL


@dataclass(frozen=True)
class Settings:
    database_url: str = DEFAULT_DATABASE_URL
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    embed_model: str = DEFAULT_EMBED_MODEL
    embed_dimensions: int = DEFAULT_EMBED_DIMENSIONS
    chat_model: str = DEFAULT_CHAT_MODEL
    vision_model: str = DEFAULT_VISION_MODEL
    embed_batch_size: int = 32
    request_timeout: float = 600.0
    # Bounded on purpose: every frontend call opens its own connection, so an
    # unreachable host must fail in seconds rather than on the OS network
    # timeout with the browser spinning.
    database_connect_timeout: float = DEFAULT_DATABASE_CONNECT_TIMEOUT
    # How long a version may sit in 'building' with no recorded progress before
    # health calls it interrupted. Conservative on purpose: a 494-page ingest
    # with narrative facts is legitimately slow, and calling live work dead is
    # worse than reporting a dead build late.
    build_stale_minutes: float = DEFAULT_BUILD_STALE_MINUTES

    def __post_init__(self) -> None:
        # Validated for every construction path, not just from_env(), and the
        # message names the variable rather than echoing the database URL.
        for variable, value in (
            ("CLAIM_EVIDENCE_DATABASE_CONNECT_TIMEOUT", self.database_connect_timeout),
            ("CLAIM_EVIDENCE_BUILD_STALE_MINUTES", self.build_stale_minutes),
        ):
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValidationError(f"{variable} must be a positive number")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_url=os.environ.get(
                "CLAIM_EVIDENCE_DATABASE_URL", DEFAULT_DATABASE_URL
            ),
            database_connect_timeout=_positive_float(
                "CLAIM_EVIDENCE_DATABASE_CONNECT_TIMEOUT",
                DEFAULT_DATABASE_CONNECT_TIMEOUT,
            ),
            build_stale_minutes=_positive_float(
                "CLAIM_EVIDENCE_BUILD_STALE_MINUTES", DEFAULT_BUILD_STALE_MINUTES
            ),
            ollama_base_url=os.environ.get(
                "OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL
            ).rstrip("/"),
            embed_model=os.environ.get(
                "CLAIM_EVIDENCE_EMBED_MODEL", DEFAULT_EMBED_MODEL
            ),
            embed_dimensions=int(
                os.environ.get(
                    "CLAIM_EVIDENCE_EMBED_DIMENSIONS", DEFAULT_EMBED_DIMENSIONS
                )
            ),
            chat_model=os.environ.get("CLAIM_EVIDENCE_CHAT_MODEL", DEFAULT_CHAT_MODEL),
            vision_model=os.environ.get(
                "CLAIM_EVIDENCE_VISION_MODEL", DEFAULT_VISION_MODEL
            ),
            embed_batch_size=int(
                os.environ.get("CLAIM_EVIDENCE_EMBED_BATCH_SIZE", "32")
            ),
            request_timeout=float(
                os.environ.get("CLAIM_EVIDENCE_REQUEST_TIMEOUT", "600")
            ),
        )

    @property
    def index_fingerprint_parts(self) -> tuple[str, str]:
        """Model identity that invalidates an existing index when it changes."""
        return (self.embed_model, str(self.embed_dimensions))


def _positive_float(variable: str, default: float) -> float:
    raw = os.environ.get(variable)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        # Never echo the value: an operator can paste a URL into the wrong
        # variable, and the error goes to a browser.
        raise ValidationError(f"{variable} must be a positive number") from None


__all__ = [
    "DEFAULT_BUILD_STALE_MINUTES",
    "DEFAULT_DATABASE_CONNECT_TIMEOUT",
    "DEFAULT_DATABASE_URL",
    "Settings",
]
