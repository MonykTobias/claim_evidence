"""Ollama HTTP client for embeddings, structured chat, and vision checks.

Deliberately not a provider abstraction: one vendor, one transport. Every chat
call sends a JSON schema as ``format`` and parses the reply through a Pydantic
model, so a malformed or invented response is an exception rather than a
verdict. One retry, then a hard error -- a guessed verdict is worse than none.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Iterable, Sequence, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from .config import Settings

T = TypeVar("T", bound=BaseModel)


class OllamaError(RuntimeError):
    """The model was unreachable, or twice failed to produce valid output."""


# Ollama compiles the response schema into a GBNF grammar, which supports only
# a subset of JSON Schema. Pydantic's Decimal schema carries a regex with a
# lookahead that the converter rejects outright ("failed to parse grammar"), so
# constraint keywords are dropped before the schema is sent. Nothing is lost:
# the schema only steers generation, and Pydantic still validates the reply.
_UNSUPPORTED_KEYWORDS = frozenset({"pattern", "format", "contentEncoding"})


def gbnf_safe_schema(schema: type[BaseModel]) -> dict[str, Any]:
    def prune(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: prune(v) for k, v in node.items() if k not in _UNSUPPORTED_KEYWORDS}
        if isinstance(node, list):
            return [prune(v) for v in node]
        return node

    return prune(schema.model_json_schema())


class OllamaClient:
    def __init__(self, settings: Settings, session: requests.Session | None = None) -> None:
        self.settings = settings
        self.session = session or requests.Session()

    # --- transport ----------------------------------------------------------

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.settings.ollama_base_url}{path}"
        try:
            response = self.session.post(
                url, json=payload, timeout=self.settings.request_timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            # Ollama explains a rejected request in the body; without it a 400
            # is indistinguishable from a network failure.
            detail = getattr(getattr(exc, "response", None), "text", "") or ""
            raise OllamaError(f"{url} failed: {exc}{f' -- {detail[:400]}' if detail else ''}") from exc
        except json.JSONDecodeError as exc:
            raise OllamaError(f"{url} returned non-JSON: {exc}") from exc

    # --- embeddings ---------------------------------------------------------

    def embed(self, inputs: Sequence[str]) -> list[list[float]]:
        """Embed a batch. Returns one vector per input, in order."""
        if not inputs:
            return []
        data = self._post(
            "/api/embed",
            {"model": self.settings.embed_model, "input": list(inputs)},
        )
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(inputs):
            raise OllamaError(
                f"/api/embed returned {len(vectors or [])} vectors for {len(inputs)} inputs"
            )
        return [self._fit_dimension(vector) for vector in vectors]

    def _fit_dimension(self, vector: list[float]) -> list[float]:
        """Match the configured dimension, truncating a longer vector.

        Ollama's `/api/embed` has no output-dimension parameter, so a
        Matryoshka-trained model such as qwen3-embedding returns its full width
        (2560) even when a shorter representation is wanted. Keeping the prefix
        is exactly how MRL embeddings are shortened, and `normalize_embedding`
        re-normalizes afterwards.

        This is only valid for MRL-trained models. A model that is not trained
        that way must be configured with its native dimension; truncating it
        would quietly degrade recall instead of failing.
        """
        expected = self.settings.embed_dimensions
        if len(vector) == expected:
            return vector
        if len(vector) > expected:
            return vector[:expected]
        raise OllamaError(
            f"{self.settings.embed_model} returned dimension {len(vector)}, "
            f"which is shorter than the configured {expected}; "
            f"set CLAIM_EVIDENCE_EMBED_DIMENSIONS to match the model"
        )

    def embed_batched(self, inputs: Sequence[str]) -> Iterable[list[list[float]]]:
        size = max(1, self.settings.embed_batch_size)
        for start in range(0, len(inputs), size):
            yield self.embed(inputs[start : start + size])

    # --- structured chat ----------------------------------------------------

    def structured(
        self,
        schema: type[T],
        system: str,
        user: str,
        *,
        images: Sequence[bytes] = (),
        model: str | None = None,
    ) -> T:
        message: dict[str, Any] = {"role": "user", "content": user}
        if images:
            message["images"] = [base64.b64encode(img).decode("ascii") for img in images]
        payload = {
            "model": model or self.settings.chat_model,
            "messages": [{"role": "system", "content": system}, message],
            "format": gbnf_safe_schema(schema),
            "stream": False,
            "options": {"temperature": 0},
        }

        last_error: Exception | None = None
        for attempt in range(2):
            data = self._post("/api/chat", payload)
            content = (data.get("message") or {}).get("content") or ""
            try:
                return schema.model_validate_json(content)
            except (ValidationError, ValueError) as exc:
                last_error = exc
                # One retry with the failure echoed back; a second miss is an
                # explicit error, never a guessed answer.
                payload["messages"] = [
                    *payload["messages"][:2],
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "That response did not match the required schema: "
                            f"{exc}. Reply with valid JSON only."
                        ),
                    },
                ]
        raise OllamaError(
            f"{payload['model']} failed schema {schema.__name__} twice: {last_error}"
        )

    def vision(
        self, schema: type[T], system: str, user: str, image: bytes
    ) -> T:
        return self.structured(
            schema, system, user, images=[image], model=self.settings.vision_model
        )


__all__ = ["OllamaClient", "OllamaError", "gbnf_safe_schema"]
