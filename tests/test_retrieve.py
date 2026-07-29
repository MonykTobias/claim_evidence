"""Rank fusion, query building, and evidence cropping. No database, no model."""

from __future__ import annotations

import tempfile
from decimal import Decimal
from pathlib import Path


from PIL import Image

from claim_evidence.facts import heuristic_claim
from claim_evidence.models import (
    GeometryPrecision,
    ParsedClaim,
    Region,
    VisualVerification,
)
from claim_evidence.retrieve import (
    exact_tokens,
    fuse,
    lexical_query,
)
from claim_evidence.vision import CROP_PADDING, crop_region, verify_visual

CLAIM = "Danone reduced Scope 1 and 2 energy and industry emissions by 40.2% in 2025 versus 2020."


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


def row(evidence_id: int, text: str) -> dict:
    return {"id": evidence_id, "source_text": text}


def test_lexical_query_ors_terms_and_quotes_numbers() -> None:
    query = lexical_query(CLAIM, ["scope 1 and 2"])
    check(" OR " in query, "terms are OR'd so a sentence still matches")
    check('"40.2"' in query, "exact number quoted as a phrase")
    check('"2020"' in query, "years quoted without sentence punctuation")
    check('"2020."' not in query, "trailing period stripped from a number token")
    check('"scope 1 and 2"' in query, "a multi-word term is quoted, not AND'd")
    check(lexical_query("", []) == "", "empty query stays empty")


def test_exact_tokens_include_value_and_periods() -> None:
    tokens = exact_tokens(heuristic_claim(CLAIM), CLAIM)
    check("40.2" in tokens, "claimed value is an exact token")
    check("2025" in tokens and "2020" in tokens, "both periods are exact tokens")
    check("%" in tokens, "unit is an exact token")


def test_rrf_rewards_agreement_across_retrievers() -> None:
    fused = fuse(
        {
            "lexical": [row(1, "a"), row(2, "b")],
            "vector": [row(2, "b"), row(3, "c")],
            "graph": [row(2, "b")],
        }
    )
    check([int(f["row"]["id"]) for f in fused][0] == 2, "candidate found by all three wins")
    check(fused[0]["lexical_rank"] == 2, "lexical rank recorded")
    check(fused[0]["vector_rank"] == 1, "vector rank recorded")
    check(fused[0]["graph_rank"] == 1, "graph rank recorded")
    check(fused[-1]["lexical_rank"] is None, "vector-only candidate has no lexical rank")


def test_exact_number_bonus_beats_rank_alone() -> None:
    claim = heuristic_claim(CLAIM)
    # The top-ranked row is topically close but has the wrong number.
    fused = fuse(
        {
            "lexical": [
                row(1, "Scope 1 & 2 emissions vs. 2020 in 2025: (34.5) %"),
                row(2, "Scope 1 & 2 emissions vs. 2020 in 2025: (40.2) %"),
            ]
        },
        claim=claim,
        claim_text=CLAIM,
    )
    check(int(fused[0]["row"]["id"]) == 2, "the row with the exact number is promoted")


def test_scope_bonus_applies() -> None:
    claim = heuristic_claim(CLAIM)
    fused = fuse(
        {"lexical": [row(1, "total emissions fell"), row(2, "Scope 1 & 2 emissions fell")]},
        claim=claim,
        claim_text=CLAIM,
    )
    check(int(fused[0]["row"]["id"]) == 2, "matching scope tokens are promoted")


def test_fuse_is_deterministic_on_ties() -> None:
    first = fuse({"lexical": [row(5, "a"), row(3, "a")]})
    second = fuse({"lexical": [row(5, "a"), row(3, "a")]})
    check(
        [f["row"]["id"] for f in first] == [f["row"]["id"] for f in second],
        "identical input yields identical order",
    )


def test_crop_uses_the_region_union_with_padding() -> None:
    with tempfile.TemporaryDirectory() as temp:
        page = Path(temp) / "page.png"
        Image.new("RGB", (1000, 2000), "white").save(page)
        regions = [
            Region(bbox=(0.2, 0.4, 0.3, 0.45), role="value", precision=GeometryPrecision.CELL),
            Region(bbox=(0.1, 0.5, 0.5, 0.55), role="descriptor", precision=GeometryPrecision.CELL),
        ]
        data = crop_region(page, regions)
        with Image.open(__import__("io").BytesIO(data)) as crop:
            expected_w = round((0.5 + CROP_PADDING - (0.1 - CROP_PADDING)) * 1000)
            check(abs(crop.width - expected_w) <= 2, f"crop spans both regions ({crop.width})")
            check(crop.height > 0, "crop has height")


def test_crop_clamps_to_the_page() -> None:
    with tempfile.TemporaryDirectory() as temp:
        page = Path(temp) / "page.png"
        Image.new("RGB", (100, 100), "white").save(page)
        region = Region(bbox=(0.0, 0.0, 1.0, 1.0), role="page", precision=GeometryPrecision.PAGE)
        with Image.open(__import__("io").BytesIO(crop_region(page, [region]))) as crop:
            check(crop.size == (100, 100), "padding does not overflow the page")


def test_missing_page_image_verifies_as_unsupported() -> None:
    region = Region(bbox=(0.1, 0.1, 0.2, 0.2), role="visual", precision=GeometryPrecision.CROP)
    result = verify_visual(None, Path("does-not-exist.png"), [region], CLAIM)
    check(isinstance(result, VisualVerification), "a missing crop still returns a verdict")
    check(not result.supports_claim, "an unreadable crop never supports a claim")
    check("crop unavailable" in result.reason, "reason explains the failure")


def test_claim_without_a_number_has_no_value_tokens() -> None:
    claim = ParsedClaim(metric="emissions fell", scope="scope 1 and 2")
    check(exact_tokens(claim, "emissions fell") == set(), "nothing exact to boost")
    check(claim.value_decimal is None, "no value invented")
    check(Decimal("0") != claim.value_decimal, "value stays absent")


def main() -> int:
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
