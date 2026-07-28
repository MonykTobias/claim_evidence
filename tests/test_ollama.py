"""Ollama client behaviour, driven entirely by a fake HTTP session."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_ollama import FakeSession, reply  # noqa: E402

from claim_evidence.config import Settings  # noqa: E402
from claim_evidence.models import Adjudication, VisualVerification  # noqa: E402
from claim_evidence.ollama import OllamaClient, OllamaError  # noqa: E402

SETTINGS = Settings(embed_dimensions=8, embed_batch_size=2)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


def test_embed_batches_and_checks_dimension() -> None:
    session = FakeSession(dimensions=8)
    client = OllamaClient(SETTINGS, session)
    batches = list(client.embed_batched(["a", "b", "c"]))
    check([len(b) for b in batches] == [2, 1], "inputs split into batches of two")
    check(len(session.requests) == 2, "one HTTP call per batch")
    check(client.embed(["a"])[0] == client.embed(["a"])[0], "embeddings deterministic")


def test_short_vector_is_an_error() -> None:
    session = FakeSession(dimensions=4)
    try:
        OllamaClient(SETTINGS, session).embed(["a"])
    except OllamaError as exc:
        check("shorter than the configured 8" in str(exc), f"short vector rejected ({exc})")
        return
    raise AssertionError("an undersized embedding was accepted")


def test_longer_vector_is_truncated_to_the_configured_width() -> None:
    # qwen3-embedding returns its full 2560 width; /api/embed has no output
    # dimension parameter, so the MRL prefix is taken.
    session = FakeSession(dimensions=32)
    vector = OllamaClient(SETTINGS, session).embed(["a"])[0]
    check(len(vector) == 8, "longer vector truncated to the configured dimension")
    check(vector == session.vector("a")[:8], "truncation keeps the leading prefix")


def test_short_embedding_batch_is_an_error() -> None:
    session = FakeSession(dimensions=8, embed_hook=lambda inputs: [[0.0] * 8])
    try:
        OllamaClient(SETTINGS, session).embed(["a", "b"])
    except OllamaError as exc:
        check("1 vectors for 2 inputs" in str(exc), f"count mismatch reported ({exc})")
        return
    raise AssertionError("truncated embedding batch was accepted")


def test_structured_output_is_validated() -> None:
    session = FakeSession(
        chat_replies=[reply({"verdict": "supported", "rationale": "matches"})]
    )
    result = OllamaClient(SETTINGS, session).structured(Adjudication, "sys", "user")
    check(result.verdict == "supported", "structured reply parsed")
    schema = session.requests[0][1]["format"]
    check(schema["title"] == "Adjudication", "json schema sent as format")
    check(session.requests[0][1]["options"]["temperature"] == 0, "temperature is zero")


def test_invalid_json_gets_exactly_one_retry() -> None:
    session = FakeSession(
        chat_replies=["not json at all", reply({"verdict": "insufficient", "rationale": "x"})]
    )
    result = OllamaClient(SETTINGS, session).structured(Adjudication, "sys", "user")
    check(result.verdict == "insufficient", "retry result used")
    check(len(session.requests) == 2, "exactly one retry")
    check(
        "did not match the required schema" in session.requests[1][1]["messages"][-1]["content"],
        "retry echoes the validation failure back",
    )


def test_second_failure_raises_instead_of_guessing() -> None:
    session = FakeSession(chat_replies=["nope", '{"verdict": "banana"}'])
    try:
        OllamaClient(SETTINGS, session).structured(Adjudication, "sys", "user")
    except OllamaError as exc:
        check("twice" in str(exc), f"hard error after two failures ({exc})")
        check(len(session.requests) == 2, "no third attempt")
        return
    raise AssertionError("invalid model output became a result")


def test_vision_sends_the_image_and_vision_model() -> None:
    session = FakeSession(
        chat_replies=[reply({"supports_claim": True, "visible_text": "40.2%"})]
    )
    settings = Settings(embed_dimensions=8, vision_model="vision-model")
    result = OllamaClient(settings, session).vision(
        VisualVerification, "sys", "does the crop show 40.2%?", b"\x89PNG"
    )
    check(result.supports_claim, "vision verdict parsed")
    payload = session.requests[0][1]
    check(payload["model"] == "vision-model", "vision model used")
    check(len(payload["messages"][-1]["images"]) == 1, "crop attached as base64")


def main() -> int:
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
