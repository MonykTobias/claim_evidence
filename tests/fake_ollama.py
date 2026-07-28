"""A requests.Session stand-in so tests need no GPU and no live model."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable


class FakeResponse:
    def __init__(self, payload: Any, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"status {self.status_code}")

    def json(self) -> Any:
        return self._payload


class FakeSession:
    """Routes ``/api/embed`` to a deterministic hash embedding and ``/api/chat``
    to a queue of scripted replies."""

    def __init__(
        self,
        dimensions: int = 8,
        chat_replies: list[str] | None = None,
        embed_hook: Callable[[list[str]], Any] | None = None,
    ) -> None:
        self.dimensions = dimensions
        self.chat_replies = list(chat_replies or [])
        self.embed_hook = embed_hook
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def post(self, url: str, json: dict[str, Any], timeout: float) -> FakeResponse:  # noqa: A002
        payload = json
        self.requests.append((url, payload))
        if url.endswith("/api/embed"):
            inputs = payload["input"]
            if self.embed_hook is not None:
                override = self.embed_hook(inputs)
                if override is not None:
                    return FakeResponse({"embeddings": override})
            return FakeResponse({"embeddings": [self.vector(t) for t in inputs]})
        if url.endswith("/api/chat"):
            if not self.chat_replies:
                raise AssertionError(f"unexpected chat call: {payload['messages'][-1]}")
            return FakeResponse({"message": {"content": self.chat_replies.pop(0)}})
        raise AssertionError(f"unexpected url {url}")

    def vector(self, text: str) -> list[float]:
        """Stable pseudo-embedding: same text always yields the same vector."""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [
            digest[i % len(digest)] / 255.0 - 0.5 for i in range(self.dimensions)
        ]
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]


def reply(payload: Any) -> str:
    return json.dumps(payload)
