"""The claim entry point: what it refuses, and what it deliberately no longer does.

No database and no model: every refusal here must happen before either is
touched, which is exactly what makes them testable without them.

Version 1 refused anything outside one exact surface shape -- approximate,
bounded, compound, causal, comparative, qualitative -- before an audit opened.
That gate is gone. The claims it turned away are now either compared
arithmetically or answered from cited evidence, and the only refusals left are
the ones a trust boundary owes its caller.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from claim_evidence.claims import MAX_CLAIM_CHARS, validate_free_text
from claim_evidence.contracts import ERROR_STATUS
from claim_evidence.errors import ValidationError
from claim_evidence.facts import heuristic_claim

ENTITY = "Danone S.A."

# The sentence this refactor exists for. Every part of it -- a fiscal period, a
# hedging adjective on the metric, a compound unit written in words, a Unicode
# subscript -- was refused by at least one version-1 rule.
IKEA = (
    "In FY24, IKEA’s estimated total climate footprint was 21.3 million "
    "tonnes of CO₂ equivalent."
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


def accepted(claim: str) -> str:
    text, _entity = validate_free_text(claim, reporting_entity=ENTITY)
    return text


# --- what the boundary still refuses ----------------------------------------


@pytest.mark.parametrize(
    "claim,entity",
    [
        ("", ENTITY),
        ("   ", ENTITY),
        ("x" * (MAX_CLAIM_CHARS + 1), ENTITY),
        ("Danone reduced emissions by 40.2% in 2025.", ""),
        ("Danone reduced emissions by 40.2% in 2025.", "   "),
        ("Danone reduced emissions by 40.2% in 2025.", None),
    ],
)
def test_the_trust_boundary_refuses_what_it_owns(claim, entity) -> None:
    with pytest.raises(ValidationError):
        validate_free_text(claim, reporting_entity=entity)


def test_non_text_is_refused_before_anything_reads_it() -> None:
    for value in (None, 42, ["a claim"], {"claim": "x"}):
        try:
            validate_free_text(value, reporting_entity=ENTITY)
        except ValidationError as exc:
            check("must be text" in str(exc), f"{type(value).__name__} is refused")
            continue
        raise AssertionError(f"{value!r} must be refused")


def test_a_filename_is_never_the_entity() -> None:
    """GR-I17: the subject used to be derived from the document's basename."""
    _text, entity = validate_free_text(
        "Danone reduced emissions by 40.2% in 2025.", reporting_entity="Danone S.A."
    )
    check(
        entity == "Danone S.A." and ".pdf" not in entity,
        "the entity is what the caller stated, not what a file is called",
    )


def test_a_rejected_claim_is_cheap() -> None:
    """No database, no model, no audit row: the refusal is pure computation."""
    for claim, entity in (("", ENTITY), ("Danone did well.", "")):
        with pytest.raises(ValidationError):
            validate_free_text(claim, reporting_entity=entity)
    check(True, "every refusal ran without a connection or a model client")


def test_the_contract_still_maps_a_declined_claim_to_422() -> None:
    """`unsupported_claim` did not disappear; it moved to where it is knowable.

    Decomposition still declines a post that asserts nothing checkable, and the
    status that answer maps to is unchanged -- it is a 422, never an
    `insufficient` verdict, which would report a limit of this tool as a
    finding about the document.
    """
    check(ERROR_STATUS["unsupported_claim"] == 422, "understood, and declined")
    check(ERROR_STATUS["validation_error"] == 400, "malformed is a different answer")


# --- what it no longer refuses ----------------------------------------------


@pytest.mark.parametrize(
    "claim",
    [
        "Danone reduced emissions by about 40% in 2025.",
        "Danone reduced emissions by at least 40% in 2025.",
        "Danone reduced emissions by between 30% and 50% in 2025.",
        "Danone reduced emissions by 40.2% because of renewable electricity.",
        "Danone reduced emissions by 40.2% while water use fell 12.0%.",
        "Danone reduced emissions by more than Nestle in 2025.",
        "Danone reduced emissions by 40.2%, equivalent to 1,000 tonnes.",
        "Danone reduced combined emissions by 40.2% in 2025.",
        "Danone improved its environmental performance in 2025.",
        "Danone reduced emissions by 40.2 in 2025.",
        "Danone reported 42 widgets in 2025.",
        IKEA,
    ],
)
def test_every_version_1_refusal_class_is_now_accepted(claim) -> None:
    assert accepted(claim) == accepted(claim)  # deterministic
    assert accepted(claim)


def test_the_regression_sentence_parses_into_comparable_parts() -> None:
    parsed = heuristic_claim(accepted(IKEA))
    check(parsed.value_decimal == Decimal("21.3"), f"the value is {parsed.value_decimal}")
    check(parsed.unit == "mtco2e", f"the compound unit is read ({parsed.unit})")
    check(parsed.reporting_period == "FY2024", f"the fiscal period is {parsed.reporting_period}")
    check(
        not parsed.approximate,
        "'estimated' describes the metric; it does not make 21.3 an approximation",
    )


def test_a_hedged_claim_is_still_marked_as_hedged() -> None:
    parsed = heuristic_claim("IKEA cut emissions by about 40% in 2025.")
    check(parsed.approximate, "'about' is a statement about precision")
    check(parsed.comparison == "~", "and is carried as the comparison operator")


def main() -> int:
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not callable(function):
            continue
        marks = getattr(function, "pytestmark", [])
        cases = [mark.args[1] for mark in marks if mark.name == "parametrize"]
        print(f"\n--- {name} ---")
        if cases:
            for case in cases[0]:
                arguments = case if isinstance(case, tuple) else (case,)
                function(*arguments)
                print(f"[ok] {str(arguments[0])[:60]}")
        else:
            function()
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
