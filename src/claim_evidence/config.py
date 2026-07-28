"""Environment-driven settings.

Every knob is a plain environment variable so the package can run against a
managed PostgreSQL and a remote Ollama without a config file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_DATABASE_URL = (
    "postgresql://claim_evidence:claim_evidence@localhost:5433/claim_evidence"
)
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

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_url=os.environ.get(
                "CLAIM_EVIDENCE_DATABASE_URL", DEFAULT_DATABASE_URL
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


__all__ = ["Settings", "DEFAULT_DATABASE_URL"]
