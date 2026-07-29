"""Environment-driven settings.

Every knob is a plain environment variable so the package can run against a
managed PostgreSQL and a remote Ollama without a config file.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .errors import ValidationError

DEFAULT_DATABASE_URL = (
    "postgresql://claim_evidence:claim_evidence@localhost:5433/claim_evidence"
)
DEFAULT_DATABASE_CONNECT_TIMEOUT = 10.0
DEFAULT_BUILD_STALE_MINUTES = 60.0
# Every structured call this package makes is bounded: one evidence passage for
# fact extraction, at most 15 passages of 800 characters for adjudication. The
# model's 64k default buys nothing for prompts that size and costs KV-cache
# memory that would otherwise hold model layers on the GPU.
DEFAULT_NUM_CTX = 16384
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "qwen3-embedding:4b"
DEFAULT_EMBED_DIMENSIONS = 1024
DEFAULT_CHAT_MODEL = "hf.co/unsloth/Qwen3-VL-4B-Instruct-GGUF:UD-Q8_K_XL"
DEFAULT_VISION_MODEL = DEFAULT_CHAT_MODEL

# One shared file that says "the local application is running". The frontend
# creates it at startup and removes it on exit; the destructive reset refuses
# while it exists. A fixed temp-directory path rather than something derived
# from a checkout, so the CLI and the app agree without being configured to.
DEFAULT_APP_MARKER = str(Path(tempfile.gettempdir()) / "claim_evidence_app.running")


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
    # Context window for chat and vision requests. Embeddings are unaffected.
    num_ctx: int = DEFAULT_NUM_CTX
    # Empty unless CE_ENVIRONMENT is set. Destructive operations require the
    # exact value "development", so an unset environment is never a development
    # one by accident.
    environment: str = ""
    app_marker: str = DEFAULT_APP_MARKER

    def __post_init__(self) -> None:
        # Validated for every construction path, not just from_env(), and the
        # message names the variable rather than echoing the database URL.
        for variable, value in (
            ("CLAIM_EVIDENCE_DATABASE_CONNECT_TIMEOUT", self.database_connect_timeout),
            ("CLAIM_EVIDENCE_BUILD_STALE_MINUTES", self.build_stale_minutes),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValidationError(f"{variable} must be a positive number")
        if isinstance(self.num_ctx, bool) or not isinstance(self.num_ctx, int) or self.num_ctx <= 0:
            raise ValidationError("CLAIM_EVIDENCE_NUM_CTX must be a positive integer")

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
            num_ctx=_positive_int("CLAIM_EVIDENCE_NUM_CTX", DEFAULT_NUM_CTX),
            environment=os.environ.get("CE_ENVIRONMENT", "").strip(),
            app_marker=os.environ.get("CLAIM_EVIDENCE_APP_MARKER", DEFAULT_APP_MARKER),
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


def _positive_int(variable: str, default: int) -> int:
    raw = os.environ.get(variable)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValidationError(f"{variable} must be a positive integer") from None


__all__ = [
    "DEFAULT_BUILD_STALE_MINUTES",
    "DEFAULT_DATABASE_CONNECT_TIMEOUT",
    "DEFAULT_DATABASE_URL",
    "DEFAULT_NUM_CTX",
    "Settings",
]
