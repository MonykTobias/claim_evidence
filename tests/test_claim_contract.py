"""The version-1 claim gate: what is accepted, and what is refused before work.

No database and no model: every refusal here must happen before either is
touched, which is exactly what makes them testable without them.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from claim_evidence.claims import SUPPORTED_UNITS, validate_claim
from claim_evidence.contracts import ERROR_STATUS
from claim_evidence.errors import UnsupportedClaimError, ValidationError

ENTITY = "Danone S.A."


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


def refused(claim: str) -> str:
    try:
        validate_claim(claim, reporting_entity=ENTITY)
    except UnsupportedClaimError as exc:
        return exc.reason_code
    raise AssertionError(f"claim was accepted but must be refused: {claim!r}")


# --- accepted ---------------------------------------------------------------


def test_the_supported_shape_is_accepted() -> None:
    parsed = validate_claim(
        "Danone reduced Scope 1 and 2 energy and industry emissions by 40.2% "
        "in 2025 versus 2020.",
        reporting_entity=ENTITY,
    )
    check(parsed.value == Decimal("40.2"), "the value is an exact Decimal")
    check(parsed.unit == "%", "the unit is normalized")
    check(parsed.reporting_period == "2025", "the reporting period is read")
    check(parsed.baseline_period == "2020", "and the baseline it is measured against")
    check(parsed.reporting_entity == ENTITY, "the stated entity is carried through")


def test_a_metric_name_containing_digits_is_not_a_value() -> None:
    """"Scope 1 and 2" is a metric, and reading it as the value 1 is a wrong answer."""
    parsed = validate_claim(
        "Danone reduced Scope 1 and 2 emissions by 40.2% in 2025.",
        reporting_entity=ENTITY,
    )
    check(parsed.value == Decimal("40.2"), "the value is the figure with the unit")


def test_thousands_separators_and_signs_parse_exactly() -> None:
    for text, expected in (
        ("Danone reported 1,044 tonnes of waste in 2025.", Decimal("1044")),
        ("Danone reported -12.5% water intensity in 2025.", Decimal("-12.5")),
        ("Danone reported 1,234,567 tonnes in 2025.", Decimal("1234567")),
    ):
        parsed = validate_claim(text, reporting_entity=ENTITY)
        check(parsed.value == expected, f"{text[:40]}... -> {parsed.value}")


def test_a_period_range_is_not_a_value_range() -> None:
    parsed = validate_claim(
        "Danone reduced emissions by 90% from 2020 to 2025.", reporting_entity=ENTITY
    )
    check(
        parsed.reporting_period == "2025" and parsed.baseline_period == "2020",
        "'from 2020 to 2025' is a baseline and a reporting year, not a range",
    )


# --- refused ----------------------------------------------------------------


@pytest.mark.parametrize(
    "claim,expected",
    [
        ("Danone reduced emissions by about 40% in 2025.", "approximate_claim"),
        ("Danone reduced emissions by roughly 40% in 2025.", "approximate_claim"),
        ("Danone reduced emissions by at least 40% in 2025.", "bounded_claim"),
        ("Danone reduced emissions by no more than 40% in 2025.", "bounded_claim"),
        ("Danone reduced emissions by between 30% and 50% in 2025.", "range_claim"),
        ("Danone reduced emissions by 30-50% in 2025.", "range_claim"),
        (
            "Danone reduced emissions by 40.2% because of renewable electricity.",
            "causal_claim",
        ),
        (
            "Danone reduced emissions by 40.2% while water use fell 12.0%.",
            "compound_claim",
        ),
        (
            "Danone reduced emissions by more than Nestle in 2025.",
            "bounded_claim",
        ),
        (
            "Danone reduced emissions by 40.2%, equivalent to 1,000 tonnes.",
            "unit_conversion_claim",
        ),
        (
            "Danone reduced combined emissions by 40.2% in 2025.",
            "external_calculation_claim",
        ),
        ("Danone improved its environmental performance in 2025.", "qualitative_claim"),
        ("Danone reduced emissions by 40.2 in 2025.", "missing_unit_claim"),
        (
            "Danone reduced emissions by 40.2% and water use by 12.0% in 2025.",
            "multi_value_claim",
        ),
        ("Danone reduced emissions by 40,2% in 2025.", "ambiguous_number"),
        ("Danone reduced emissions by 40·2% in 2025.", "ambiguous_number"),
    ],
)
def test_unsupported_classes_are_refused_by_category(claim, expected) -> None:
    assert refused(claim) == expected


def test_the_refusal_is_a_422_category_not_an_insufficient_verdict() -> None:
    """PD-08: `insufficient` is a statement about the sources, not about us."""
    try:
        validate_claim("Danone did well in 2025.", reporting_entity=ENTITY)
    except UnsupportedClaimError as exc:
        check(
            ERROR_STATUS["unsupported_claim"] == 422,
            "the contract maps an unsupported claim to HTTP 422",
        )
        check(
            "insufficient" not in str(exc).lower(),
            "the refusal never describes itself as an insufficient verdict",
        )
        check(
            "version 1" in str(exc),
            f"the message says what version 1 does accept ({exc})",
        )
        return
    raise AssertionError("a qualitative claim must be refused")


def test_an_unsupported_unit_is_refused_rather_than_guessed() -> None:
    code = refused("Danone reported 42 widgets in 2025.")
    check(code == "missing_unit_claim", f"an unknown unit is not a comparable one ({code})")
    check(
        "widget" not in SUPPORTED_UNITS,
        "the supported unit vocabulary is closed, not open-ended",
    )


# --- the entity -------------------------------------------------------------


def test_the_reporting_entity_must_be_stated() -> None:
    for entity in ("", "   ", None):
        try:
            validate_claim("Danone reduced emissions by 40.2% in 2025.", reporting_entity=entity)
        except ValidationError as exc:
            check("reporting_entity is required" in str(exc), f"{entity!r} is refused")
            continue
        raise AssertionError(f"reporting_entity {entity!r} must be refused")
    check(True, "an omitted reporting entity is a validation error")


def test_a_filename_is_never_the_entity() -> None:
    """GR-I17: the subject used to be derived from the document's basename."""
    parsed = validate_claim(
        "Danone reduced emissions by 40.2% in 2025.",
        reporting_entity="Danone S.A.",
    )
    check(
        parsed.reporting_entity == "Danone S.A.",
        "the entity is what the caller stated, not what a file happens to be called",
    )
    check(
        ".pdf" not in parsed.reporting_entity,
        "and it is not a filename",
    )


def test_a_rejected_claim_is_cheap() -> None:
    """No database, no model, no audit row: the refusal is pure computation."""
    for claim in (
        "Danone reduced emissions by about 40% in 2025.",
        "Danone did well.",
    ):
        with pytest.raises((UnsupportedClaimError, ValidationError)):
            validate_claim(claim, reporting_entity=ENTITY)
    check(True, "every refusal ran without a connection or a model client")


def main() -> int:
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not callable(function):
            continue
        marks = getattr(function, "pytestmark", [])
        cases = [
            mark.args[1] for mark in marks if mark.name == "parametrize"
        ]
        print(f"\n--- {name} ---")
        if cases:
            for claim, expected in cases[0]:
                assert refused(claim) == expected, claim
                print(f"[ok] {expected:30} {claim[:52]}")
        else:
            function()
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
