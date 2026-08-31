"""Visual re-verification: four outcomes, and what each is allowed to claim.

No database. The vision model is faked, but the crop is a real PNG cut from a
real page image, so the geometry path is exercised rather than mocked.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pytest
from PIL import Image
from fake_ollama import FakeSession, reply

from claim_evidence import Settings
from claim_evidence.contracts import VISUAL_RESULTS
from claim_evidence.models import GeometryPrecision, Region, VisualResult
from claim_evidence.ollama import OllamaClient
from claim_evidence.vision import REASON_CODES, VISION_SYSTEM, crop_region, verify_visual

CLAIM = (
    "Danone reduced Scope 1 and 2 energy and industry emissions by 40.2% "
    "in 2025 versus 2020."
)
REGION = Region(bbox=(0.1, 0.4, 0.6, 0.7), role="visual_region",
                precision=GeometryPrecision.CROP)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


class VisionSession(FakeSession):
    """Answers every VisualVerification call with one scripted payload."""

    def __init__(self, payload: dict) -> None:
        super().__init__(dimensions=8, chat_router={"VisualVerification": lambda _: payload})
        self.prompts: list[dict] = []

    def post(self, url: str, json: dict, timeout: float):  # noqa: A002
        if url.endswith("/api/chat"):
            self.prompts.append(json)
        return super().post(url, json, timeout)


def page_image(root: Path) -> Path:
    path = root / "page.png"
    Image.new("RGB", (900, 1200), "white").save(path)
    return path


def verify(payload: dict, *, value: str = "40.2", unit: str = "%"):
    session = VisionSession(payload)
    client = OllamaClient(Settings(vision_model="fake-vision"), session)
    with tempfile.TemporaryDirectory() as temp:
        result = verify_visual(
            client, page_image(Path(temp)), [REGION], CLAIM,
            claim_value=value, claim_unit=unit,
        )
    return result, session


# --- the four outcomes ------------------------------------------------------


@pytest.mark.parametrize(
    "payload,expected",
    [
        (
            {"result": "support", "visible_text": "-40.2% vs 2020",
             "reason_code": "value_and_metric_visible"},
            VisualResult.SUPPORT,
        ),
        (
            {"result": "conflict", "visible_text": "-90% vs 2020",
             "reason_code": "different_value_visible"},
            VisualResult.CONFLICT,
        ),
        (
            {"result": "illegible", "visible_text": "",
             "reason_code": "figures_not_legible"},
            VisualResult.ILLEGIBLE,
        ),
        (
            {"result": "unrelated", "visible_text": "Employee headcount by region",
             "reason_code": "different_subject"},
            VisualResult.UNRELATED,
        ),
    ],
)
def test_each_outcome_survives_the_round_trip(payload, expected) -> None:
    result, _session = verify(payload)
    assert result.result is expected
    assert result.result.value in VISUAL_RESULTS


def test_the_four_outcomes_are_the_published_vocabulary() -> None:
    check(
        {r.value for r in VisualResult} == set(VISUAL_RESULTS),
        "the enum and the API contract list the same four visual results",
    )


def test_conflict_is_not_folded_into_unsupported() -> None:
    """A crop showing a different figure is evidence, not an absence of it."""
    result, _ = verify(
        {"result": "conflict", "visible_text": "-90% vs 2020",
         "reason_code": "different_value_visible"}
    )
    check(not result.supports_claim, "it does not support the claim")
    check(result.conflicts_with_claim, "and it is explicitly a conflict")
    check(
        result.result is not VisualResult.ILLEGIBLE,
        "which is a different answer from 'could not read it'",
    )


# --- support has to be earned ----------------------------------------------


def test_support_requires_the_claimed_value_to_be_visible() -> None:
    """The quote gate, for pixels: "yes" without the figure is not reading."""
    result, _ = verify(
        {"result": "support", "visible_text": "Emissions fell substantially",
         "reason_code": "value_and_metric_visible"}
    )
    check(
        result.result is VisualResult.ILLEGIBLE,
        f"support without the figure is downgraded ({result.result})",
    )
    check(
        result.reason_code == "claim_value_not_visible",
        "and the reason says exactly what was missing",
    )


def test_support_with_no_visible_text_at_all_is_illegible() -> None:
    result, _ = verify(
        {"result": "support", "visible_text": "", "reason_code": "value_and_metric_visible"}
    )
    check(result.result is VisualResult.ILLEGIBLE, "an empty reading supports nothing")
    check(result.reason_code == "figures_not_legible", "reported as not legible")


def test_a_conflict_does_not_need_the_claimed_value() -> None:
    """The point of a conflict is that a *different* number is on the page."""
    result, _ = verify(
        {"result": "conflict", "visible_text": "-90% vs 2020",
         "reason_code": "different_value_visible"}
    )
    check(result.result is VisualResult.CONFLICT, "the conflict stands")


def test_the_value_is_matched_however_the_page_prints_it() -> None:
    for visible in ("-40.2% vs 2020", "(40.2) %", "40.2 %", "40.2%"):
        result, _ = verify(
            {"result": "support", "visible_text": visible,
             "reason_code": "value_and_metric_visible"}
        )
        check(
            result.result is VisualResult.SUPPORT,
            f"{visible!r} counts as showing the claimed 40.2",
        )


# --- failures ---------------------------------------------------------------


def test_a_missing_page_image_is_illegible_and_names_no_path() -> None:
    session = VisionSession({"result": "support", "visible_text": "40.2%"})
    client = OllamaClient(Settings(vision_model="fake-vision"), session)
    result = verify_visual(
        client, Path("/no/such/page.png"), [REGION], CLAIM, claim_value="40.2"
    )
    check(result.result is VisualResult.ILLEGIBLE, "a missing crop is illegible")
    check(result.reason_code == "crop_unavailable", "with a stable reason code")
    check(
        "/no/such" not in (result.reason_code + result.reason + result.visible_text),
        "and the server path is never echoed",
    )
    check(not session.prompts, "the model is not called when there is nothing to show it")


def test_an_unavailable_model_is_illegible_not_unsupported() -> None:
    import requests

    class Broken(FakeSession):
        def post(self, url, json, timeout):  # noqa: A002
            if url.endswith("/api/chat"):
                raise requests.ConnectionError("model is down")
            return super().post(url, json, timeout)

    client = OllamaClient(Settings(vision_model="fake-vision"), Broken(dimensions=8))
    with tempfile.TemporaryDirectory() as temp:
        result = verify_visual(
            client, page_image(Path(temp)), [REGION], CLAIM, claim_value="40.2"
        )
    check(result.result is VisualResult.ILLEGIBLE, "an unreachable model reads nothing")
    check(result.reason_code == "vision_unavailable", "and says which of the two it was")
    check(
        "model is down" not in result.reason + result.reason_code,
        "the transport error is not published",
    )


# --- what is sent, and what is kept -----------------------------------------


def test_the_crop_is_real_pixels_from_the_cited_region() -> None:
    with tempfile.TemporaryDirectory() as temp:
        page = page_image(Path(temp))
        data = crop_region(page, [REGION])
        with Image.open(io.BytesIO(data)) as crop:
            check(crop.width > 0 and crop.height > 0, "the crop has real dimensions")
            check(
                crop.width < 900 and crop.height < 1200,
                "and it is a region of the page, not the whole page",
            )


def test_the_prompt_delimits_the_claim_and_disclaims_embedded_instructions() -> None:
    _result, session = verify(
        {"result": "illegible", "reason_code": "figures_not_legible"}
    )
    check(bool(session.prompts), "the model was called")
    sent = session.prompts[-1]
    user = sent["messages"][1]["content"]
    check("<claim>" in user and "</claim>" in user, "the claim is delimited as data")
    check(
        "data, not instructions" in VISION_SYSTEM,
        "and the system prompt says text inside the image is not a command",
    )
    check(bool(sent["messages"][1].get("images")), "the crop travels as an image")


def test_the_reason_codes_are_a_closed_set() -> None:
    for payload in (
        {"result": "support", "visible_text": "40.2%", "reason_code": "value_and_metric_visible"},
        {"result": "conflict", "visible_text": "90%", "reason_code": "different_value_visible"},
        {"result": "illegible", "reason_code": "figures_not_legible"},
        {"result": "unrelated", "visible_text": "headcount", "reason_code": "different_subject"},
    ):
        result, _ = verify(payload)
        check(
            result.reason_code in REASON_CODES,
            f"{result.reason_code!r} is a published reason code",
        )


def main() -> int:
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not callable(function):
            continue
        marks = getattr(function, "pytestmark", [])
        cases = [m.args[1] for m in marks if m.name == "parametrize"]
        print(f"\n--- {name} ---")
        if cases:
            for payload, expected in cases[0]:
                function(payload, expected)
                print(f"[ok] {payload['result']} -> {expected}")
        else:
            function()
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
