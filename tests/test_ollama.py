"""Ollama client behaviour, driven entirely by a fake HTTP session."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path


from fake_ollama import FakeSession, reply

from claim_evidence.config import DEFAULT_NUM_CTX, Settings
from claim_evidence.errors import ValidationError
from claim_evidence.models import (
    Adjudication,
    Fact,
    FactExtraction,
    VisualVerification,
)
from claim_evidence.ollama import OllamaClient, OllamaError, gbnf_safe_schema

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


def test_schema_is_stripped_of_grammar_hostile_keywords() -> None:
    # Ollama compiles the schema to a GBNF grammar. Pydantic's Decimal schema
    # carries a lookahead regex that the converter rejects with HTTP 400
    # "failed to parse grammar", which took out the whole fact extractor.
    schema = gbnf_safe_schema(FactExtraction)
    text = json.dumps(schema)
    check("pattern" not in text, "pattern constraints removed")
    check('"$defs"' in text, "nested model definitions retained")
    check(
        "quote" in schema["$defs"]["Fact"]["required"],
        "quote stays required so the model cannot omit it",
    )


def test_displayed_values_coerce_to_decimal() -> None:
    # Models answer with the figure as printed, which is not JSON number syntax.
    for raw, expected in (
        ("-40.2%", Decimal("-40.2")),
        ("(40.2) %", Decimal("-40.2")),
        ("1,044", Decimal("1044")),
        ("not a number", None),
    ):
        fact = Fact(subject="Danone", metric="m", value_decimal=raw, quote="q")
        check(fact.value_decimal == expected, f"{raw!r} parsed as {expected}")


def test_vision_sends_the_image_and_vision_model() -> None:
    session = FakeSession(
        chat_replies=[reply({"result": "support", "visible_text": "40.2%", "reason_code": "value_and_metric_visible"})]
    )
    settings = Settings(embed_dimensions=8, vision_model="vision-model")
    result = OllamaClient(settings, session).vision(
        VisualVerification, "sys", "does the crop show 40.2%?", b"\x89PNG"
    )
    check(result.supports_claim, "vision verdict parsed")
    payload = session.requests[0][1]
    check(payload["model"] == "vision-model", "vision model used")
    check(len(payload["messages"][-1]["images"]) == 1, "crop attached as base64")


def test_structured_requests_bound_the_context_window() -> None:
    session = FakeSession(chat_replies=[reply({"facts": []})])
    OllamaClient(SETTINGS, session).structured(FactExtraction, "sys", "passage")
    options = session.requests[0][1]["options"]
    check(options["num_ctx"] == DEFAULT_NUM_CTX == 16384, "the default context is 16k")
    check(options["temperature"] == 0, "and determinism is unchanged")


def test_configured_context_reaches_the_request() -> None:
    session = FakeSession(chat_replies=[reply({"facts": []})])
    settings = Settings(embed_dimensions=8, num_ctx=4096)
    OllamaClient(settings, session).structured(FactExtraction, "sys", "passage")
    check(session.requests[0][1]["options"]["num_ctx"] == 4096, "an override is honoured")


def test_vision_uses_the_same_configured_context() -> None:
    session = FakeSession(
        chat_replies=[reply({"result": "illegible", "reason_code": "figures_not_legible", "reason": "unreadable"})]
    )
    settings = Settings(embed_dimensions=8, vision_model="vision-model", num_ctx=8192)
    OllamaClient(settings, session).vision(
        VisualVerification, "sys", "does the crop show 40.2%?", b"\x89PNG"
    )
    check(session.requests[0][1]["options"]["num_ctx"] == 8192, "vision shares the setting")


def test_embed_requests_carry_no_context_option() -> None:
    """/api/embed has no chat context; sending one would be noise at best."""
    session = FakeSession(dimensions=8)
    OllamaClient(SETTINGS, session).embed(["a"])
    url, payload = session.requests[0]
    check(url.endswith("/api/embed"), "the embed endpoint was called")
    check("options" not in payload, "no options block is sent")
    check("num_ctx" not in json.dumps(payload), "and no context anywhere in the payload")


def test_retry_keeps_the_bounded_context() -> None:
    session = FakeSession(chat_replies=["not json", reply({"facts": []})])
    OllamaClient(SETTINGS, session).structured(FactExtraction, "sys", "passage")
    check(len(session.requests) == 2, "the invalid reply was retried once")
    check(
        all(r[1]["options"]["num_ctx"] == DEFAULT_NUM_CTX for r in session.requests),
        "the retry does not silently widen the context",
    )


def test_invalid_context_configuration_is_rejected() -> None:
    import os

    for value in (0, -1, True):
        try:
            Settings(embed_dimensions=8, num_ctx=value)
        except ValidationError as exc:
            check("CLAIM_EVIDENCE_NUM_CTX" in str(exc), f"{value!r} rejected by name")
            continue
        raise AssertionError(f"num_ctx={value!r} was accepted")

    original = os.environ.get("CLAIM_EVIDENCE_NUM_CTX")
    for raw in ("abc", "true", "16k", "1.5"):
        os.environ["CLAIM_EVIDENCE_NUM_CTX"] = raw
        try:
            Settings.from_env()
        except ValidationError:
            check(True, f"{raw!r} from the environment is rejected")
        else:
            raise AssertionError(f"CLAIM_EVIDENCE_NUM_CTX={raw!r} was accepted")
    os.environ["CLAIM_EVIDENCE_NUM_CTX"] = "32768"
    check(Settings.from_env().num_ctx == 32768, "a valid override is parsed")
    if original is None:
        del os.environ["CLAIM_EVIDENCE_NUM_CTX"]
    else:
        os.environ["CLAIM_EVIDENCE_NUM_CTX"] = original


def main() -> int:
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
