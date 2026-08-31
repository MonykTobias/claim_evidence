"""Geometry, value, and quote normalization checks."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path


from claim_evidence.models import GeometryPrecision
from claim_evidence.normalize import (
    all_periods,
    contains_quote,
    detect_direction,
    normalize_bbox,
    normalize_period,
    normalize_unit,
    parse_period,
    parse_value,
    region_from,
    scopes_comparable,
    signed_change,
    union_bbox,
    unit_conversion,
    values_agree,
)

PAGE_W = 609.45
PAGE_H = 793.7


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


def test_bottom_left_normalization() -> None:
    # Real page-1 heading block: BOTTOMLEFT means t is the larger y value.
    bbox = {"l": 51.09, "t": 762.55, "r": 367.575, "b": 642.783, "origin": "BOTTOMLEFT"}
    norm, source, origin = normalize_bbox(bbox, PAGE_W, PAGE_H)
    check(origin == "BOTTOMLEFT", "origin preserved")
    check(source == (51.09, 762.55, 367.575, 642.783), "source coordinates preserved")
    check(abs(norm[1] - (PAGE_H - 762.55) / PAGE_H) < 1e-6, "top flipped to top-left")
    check(norm[1] < norm[3], "top above bottom after flip")
    check(0.0 <= min(norm) and max(norm) <= 1.0, "normalized inside the unit square")


def test_top_left_normalization() -> None:
    # Real page-359 value cell for "(40.2) %".
    bbox = {
        "l": 335.603,
        "t": 386.679,
        "r": 355.523,
        "b": 394.851,
        "origin": "TOPLEFT",
    }
    norm, _, _ = normalize_bbox(bbox, PAGE_W, PAGE_H)
    check(abs(norm[1] - 386.679 / PAGE_H) < 1e-6, "top-left top used as-is")
    check(norm[1] < norm[3], "row height positive")


def test_pre_normalized_bbox_passthrough() -> None:
    norm, source, origin = normalize_bbox([0.08, 0.136, 0.918, 0.852], 0, 0)
    check(norm == (0.08, 0.136, 0.918, 0.852), "already-normalized bbox kept")
    check(origin == "NORMALIZED", "normalized origin recorded")
    check(source == (0.08, 0.136, 0.918, 0.852), "source retained")


def test_region_carries_precision() -> None:
    region = region_from(
        [0.1, 0.2, 0.3, 0.4], 0, 0, role="value", precision=GeometryPrecision.CELL
    )
    check(region.precision is GeometryPrecision.CELL, "precision carried on region")
    check(region.role == "value", "role carried on region")


def test_union_bbox() -> None:
    check(union_bbox([]) is None, "empty union is None")
    merged = union_bbox([(0.1, 0.2, 0.3, 0.4), (0.05, 0.25, 0.6, 0.35)])
    check(merged == (0.05, 0.2, 0.6, 0.4), "union covers both boxes")


def test_parenthesized_negative() -> None:
    check(parse_value("(40.2) %") == (Decimal("-40.2"), "%"), "(40.2) % is -40.2")
    check(parse_value("(20.7)%") == (Decimal("-20.7"), "%"), "(20.7)% is -20.7")
    check(parse_value("85.7%") == (Decimal("85.7"), "%"), "plain percent positive")
    check(parse_value("1,234.5") == (Decimal("1234.5"), None), "thousands separator")
    check(parse_value("15") == (Decimal("15"), None), "bare integer")
    check(parse_value("") == (None, None), "empty cell has no value")
    check(parse_value("Yes") == (None, None), "textual cell has no value")


def test_unit_normalization() -> None:
    check(normalize_unit("Percentage of variation") == "%", "percentage folds to %")
    check(normalize_unit("%") == "%", "percent sign folds to %")
    check(normalize_unit("Number") == "number", "other units lowercased")
    check(normalize_unit(None) is None, "missing unit stays missing")


def test_co2_equivalent_spellings_fold_together() -> None:
    for spelling in (
        "million tonnes of CO₂ equivalent",
        "million tonnes of CO2 equivalent",
        "MtCO2e",
        "Mt CO2-eq",
        "megatonnes CO2 equivalent",
    ):
        check(normalize_unit(spelling) == "mtco2e", f"{spelling!r} -> mtco2e")
    check(normalize_unit("tCO2e") == "tco2e", "the unprefixed form too")
    check(
        normalize_unit("million tonnes of CO2") == "million tonnes of co2",
        "but a bare CO2 figure is left alone: it is not the same claim as CO2e",
    )


def test_only_same_dimension_units_convert_and_exactly() -> None:
    check(unit_conversion("mtco2e", "tco2e") == Decimal(1_000_000), "Mt -> t is exact")
    check(unit_conversion("t", "kt") == Decimal("0.001"), "and exact the other way")
    check(
        Decimal("21.3") * unit_conversion("mtco2e", "tco2e") == Decimal("21300000.0"),
        "21.3 MtCO2e converts without a floating-point remainder",
    )
    check(unit_conversion("t", "tco2e") is None, "a tonne is not a tonne of CO2e")
    check(unit_conversion("kwh", "t") is None, "and energy is not mass")
    check(unit_conversion("widgets", "t") is None, "an unknown unit converts to nothing")
    check(unit_conversion("%", "%") == 1, "the same unit is a no-op")


def test_fiscal_periods_normalize_but_never_become_calendar_years() -> None:
    check(normalize_period("FY24") == "FY2024", "a two-digit fiscal year expands")
    check(normalize_period("FY 2024") == "FY2024", "and a spaced one folds")
    check(normalize_period("FY2024") == "FY2024", "the canonical form is stable")
    check(
        normalize_period("2024") != normalize_period("FY24"),
        "a fiscal year is not the calendar year it overlaps",
    )
    check(
        all_periods("In FY24, versus FY2020 and 2019") == ["FY2024", "FY2020", "2019"],
        "every period is read in the order it was stated",
    )


def test_period_and_direction() -> None:
    check(parse_period("Performance history 2025") == "2025", "year read from header")
    check(parse_period("Unit") is None, "no year means no period")
    check(detect_direction("Danone reduced emissions") == "decrease", "reduce")
    check(detect_direction("revenue grew") == "increase", "grew")
    check(detect_direction("emissions were 40.2%") == "unknown", "no direction word")


def test_signed_change_and_tolerance() -> None:
    check(signed_change(Decimal("40.2"), "decrease") == Decimal("-40.2"), "signed down")
    check(signed_change(Decimal("40.2"), "increase") == Decimal("40.2"), "signed up")
    check(
        values_agree(Decimal("-40.2"), Decimal("-40.2"), approximate=False),
        "exact match supported",
    )
    check(
        not values_agree(Decimal("-40.2"), Decimal("-40.3"), approximate=False),
        "exact claim rejects a near miss",
    )
    check(
        values_agree(Decimal("-40"), Decimal("-40.2"), approximate=True),
        "'about 40%' is satisfied at the precision it was written to",
    )
    check(
        not values_agree(Decimal("-40.0"), Decimal("-40.2"), approximate=True),
        "'about 40.0%' was written to one decimal, and 40.2 differs there",
    )
    check(
        values_agree(Decimal("21.3"), Decimal("21.34"), approximate=True),
        "'about 21.3' accepts 21.34",
    )
    check(
        not values_agree(Decimal("21.3"), Decimal("21.5"), approximate=True),
        "and still refuses 21.5, which no percentage tolerance would have",
    )
    check(
        not values_agree(Decimal("-40.0"), Decimal("-90.0"), approximate=True),
        "hedged claim still rejects a different number",
    )


def test_scope_comparability() -> None:
    fact = "Scope 1 & 2 energy and industry emissions (market-based) vs. 2020"
    check(
        scopes_comparable("Scope 1 and 2 energy and industry emissions", fact),
        "matching scope markers compare",
    )
    check(
        not scopes_comparable("all carbon emissions", fact),
        "vague scope is not comparable",
    )
    check(
        not scopes_comparable("Scope 3 emissions", fact),
        "different scope boundary is not comparable",
    )


def test_quote_substring_check() -> None:
    # The extractor emits a non-breaking hyphen inside "market-based".
    source = "Scope 1 & 2 energy and industry emissions (market‑based) vs. 2020"
    check(contains_quote(source, "market-based"), "unicode dash folds for matching")
    check(contains_quote(source, "Scope 1 & 2  energy"), "whitespace collapses")
    check(not contains_quote(source, "Scope 3 emissions"), "absent quote rejected")
    check(not contains_quote(source, ""), "empty quote rejected")


def main() -> int:
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
