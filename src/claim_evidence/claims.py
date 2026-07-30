"""The version-1 claim gate.

Version 1 audits exactly one shape of claim: one reporting entity, one metric,
one exact numeric value, one directly stated unit, and optional period, scope,
and geography qualifiers that must match directly. Everything else is refused
here -- before an audit row is written, before retrieval runs, before a model is
called.

Refusing early is the point. A claim this package cannot compare exactly does
not become a cautious ``insufficient``: that verdict is a statement about the
*sources* ("they were searched and say nothing comparable"), and using it for
"we could not understand the question" quietly turns a limitation of the tool
into a finding about the document.

Each rule carries a stable reason code, so a caller branches on the category
rather than on wording, and the message says what version 1 does accept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from .errors import UnsupportedClaimError, ValidationError
from .normalize import all_years, clean_text

MAX_CLAIM_CHARS = 10_000

# Optional sign, ASCII digits, optional decimal point, and correctly grouped
# comma thousands. Deliberately strict: "1.234,5" and "40·2" are numbers in
# some locales and typos in others, and guessing which is how a verdict ends up
# resting on a misread figure.
NUMBER = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")
_NUMERIC_TOKEN = re.compile(r"[+-]?[\d][\d,.· ]*")

# The units version 1 can compare, and how each is spelled in the index. A
# closed vocabulary rather than "whatever word follows the number": with an
# open one, "Scope 1 and 2 emissions" parses as the value 1 in the unit "and",
# and a metric name becomes a figure. A unit outside this set is refused by
# name, which is a better answer than a confident comparison of two strings
# that were never the same quantity.
SUPPORTED_UNITS: dict[str, str] = {
    "%": "%", "percent": "%", "percentage": "%",
    "percentage point": "percentage points", "percentage points": "percentage points",
    "pp": "percentage points",
    "t": "t", "tonne": "t", "tonnes": "t", "ton": "t", "tons": "t",
    "kt": "kt", "mt": "mt",
    "tco2e": "tco2e", "ktco2e": "ktco2e", "mtco2e": "mtco2e",
    "kwh": "kwh", "mwh": "mwh", "gwh": "gwh", "twh": "twh",
    "mj": "mj", "gj": "gj", "tj": "tj",
    "m3": "m3", "m³": "m3", "litre": "l", "litres": "l",
    "liter": "l", "liters": "l", "l": "l", "hl": "hl", "ml": "ml",
    "km": "km", "kg": "kg", "g": "g",
}
_UNIT_ALTERNATION = "|".join(
    sorted((re.escape(u) for u in SUPPORTED_UNITS), key=len, reverse=True)
)
# A number with one of those units directly attached. The lookahead stops "t"
# from matching the start of "tonnes"; it deliberately allows a following "."
# so a unit at the end of a sentence still counts.
_VALUE_WITH_UNIT = re.compile(
    rf"(?P<number>[+-]?\d[\d,.]*?)\s*(?P<unit>{_UNIT_ALTERNATION})(?![A-Za-z0-9])",
    re.I,
)
# Any number at all, so a claim that states a figure without a supported unit
# is told which of the two problems it has.
_ANY_NUMBER = re.compile(r"[+-]?\d[\d,.·]*")


@dataclass(frozen=True)
class SupportedClaim:
    """One claim that version 1 can compare exactly."""

    text: str
    reporting_entity: str
    value: Decimal
    unit: str
    reporting_period: str | None = None
    baseline_period: str | None = None


def _refuse(code: str, message: str) -> None:
    raise UnsupportedClaimError(message, reason_code=code)


# Each rule is (reason code, pattern, explanation). Order matters only for
# which message a doubly-unsupported claim gets; every one of them is a refusal.
_WORD_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "approximate_claim",
        re.compile(r"\b(about|approximately|around|roughly|circa|nearly|almost|some|~)\b|~", re.I),
        "version 1 compares exact values only, and this claim is approximate",
    ),
    (
        "bounded_claim",
        re.compile(
            r"\b(at least|at most|no more than|no less than|more than|less than|"
            r"up to|over|under|greater than|fewer than|minimum|maximum)\b",
            re.I,
        ),
        "version 1 compares exact values only, not minimums, maximums, or bounds",
    ),
    (
        "causal_claim",
        re.compile(
            r"\b(because|due to|owing to|thanks to|caused|causing|led to|leads to|"
            r"resulted in|resulting in|driven by)\b",
            re.I,
        ),
        "version 1 checks whether a figure is stated, not why it changed",
    ),
    (
        "compound_claim",
        re.compile(r"\bwhile\b|\bwhereas\b|;| as well as ", re.I),
        "version 1 audits one assertion at a time; split this into separate claims",
    ),
    (
        "multi_entity_claim",
        re.compile(r"\b(than|compared (?:to|with)|relative to)\s+[A-Z]", re.I),
        "version 1 audits one reporting entity, not a comparison between two",
    ),
    (
        "unit_conversion_claim",
        re.compile(r"\b(equivalent to|equals|equal to|i\.e\.|that is|which is)\b", re.I),
        "version 1 does not convert units or restate a figure in other terms",
    ),
    (
        "external_calculation_claim",
        re.compile(
            r"\b(combined|sum of|total of|average of|on average|per capita|"
            r"annualis|annualiz|extrapolat)\w*\b",
            re.I,
        ),
        "version 1 compares figures the source states, and performs no calculation",
    ),
)


def validate_free_text(
    text: str, *, reporting_entity: str, label: str = "claim"
) -> tuple[str, str]:
    """The trust-boundary check every text entry point shares.

    Type, emptiness, length, and a stated reporting entity -- nothing about what
    the text says. Returns the cleaned text and entity so a caller never works
    from the raw input.
    """
    if not isinstance(text, str):
        raise ValidationError(f"{label} must be text")
    cleaned = clean_text(text)
    if not cleaned:
        raise ValidationError(f"{label} is required")
    if len(cleaned) > MAX_CLAIM_CHARS:
        raise ValidationError(f"{label} must be at most {MAX_CLAIM_CHARS} characters")

    entity = clean_text(reporting_entity or "")
    if not entity:
        # The filename is a display label, not a fact about who reported what.
        # Deriving the subject from it made "danoneurdaccessible.pdf" an entity
        # and quietly attributed every figure in the document to that string.
        raise ValidationError(
            "reporting_entity is required: an audit is about one named entity, "
            "and a document's filename is not one"
        )
    return cleaned, entity


def validate_claim(claim: str, *, reporting_entity: str) -> SupportedClaim:
    """Parse one claim, or refuse it with a stable reason code.

    Nothing is persisted and no model is called on the way through: a rejected
    claim must cost a validation error and nothing else.
    """
    if not isinstance(claim, str):
        raise ValidationError("claim must be text")
    text = clean_text(claim)
    if not text:
        raise ValidationError("claim is required")
    if len(text) > MAX_CLAIM_CHARS:
        raise ValidationError(f"claim must be at most {MAX_CLAIM_CHARS} characters")

    entity = clean_text(reporting_entity or "")
    if not entity:
        # The filename is a display label, not a fact about who reported what.
        # Deriving the subject from it made "danoneurdaccessible.pdf" an entity
        # and quietly attributed every figure in the document to that string.
        raise ValidationError(
            "reporting_entity is required: version 1 audits one named entity, "
            "and a document's filename is not one"
        )

    for code, pattern, explanation in _WORD_RULES:
        if pattern.search(text):
            _refuse(code, explanation)
    if _states_a_value_range(text):
        _refuse("range_claim", "version 1 compares one exact value, not a range")

    value, unit = _one_exact_value(text)
    reporting, baseline = _periods(text)
    return SupportedClaim(
        text=text,
        reporting_entity=entity,
        value=value,
        unit=unit,
        reporting_period=reporting,
        baseline_period=baseline,
    )


_RANGE = re.compile(
    r"\bbetween\s+(?P<low>[+-]?\d[\d,.]*)\s*\S*\s*\band\s+(?P<high>[+-]?\d[\d,.]*)"
    r"|(?P<a>[+-]?\d[\d,.]*)\s*(?:-|–|\bto\b)\s*(?P<b>[+-]?\d[\d,.]*)",
    re.I,
)


def _states_a_value_range(text: str) -> bool:
    """True for "between 30% and 50%", false for "from 2020 to 2025".

    A period range is how a claim states its baseline and is entirely
    supported; a *value* range is what version 1 cannot compare exactly. The
    difference is which numbers the range joins, so the years have to be
    excluded rather than the wording matched.
    """
    years = set(all_years(text))
    for match in _RANGE.finditer(text):
        pair = [
            match.group("low") or match.group("a"),
            match.group("high") or match.group("b"),
        ]
        cleaned = [value.strip(".,") for value in pair if value]
        if len(cleaned) == 2 and not all(value in years for value in cleaned):
            return True
    return False


def _one_exact_value(text: str) -> tuple[Decimal, str]:
    """Exactly one numeric value with a directly attached supported unit.

    A number inside a metric name -- "Scope 1 and 2", "CO2" -- is not a value,
    which is why a value has to carry a unit from the closed vocabulary to
    count as one.
    """
    years = set(all_years(text))
    # Every number in the claim has to be one this package can read exactly.
    # Checked before anything else: with a non-greedy match, "40·2%" otherwise
    # parses as the value 2, and the audit answers a question nobody asked.
    for token in _ANY_NUMBER.findall(text):
        cleaned = token.strip(".,")
        if cleaned and cleaned not in years and not NUMBER.match(cleaned):
            _refuse(
                "ambiguous_number",
                "the number is not in the supported form: an optional sign, ASCII "
                "digits, an optional decimal point, and comma thousands separators",
            )

    candidates = [
        match
        for match in _VALUE_WITH_UNIT.finditer(text)
        if match.group("number").strip(".,") not in years
    ]
    if not candidates:
        if not [
            token for token in _ANY_NUMBER.findall(text)
            if token.strip(".,") and token.strip(".,") not in years
        ]:
            _refuse(
                "qualitative_claim",
                "version 1 audits a claim that states a number; this one states none",
            )
        _refuse(
            "missing_unit_claim",
            "version 1 needs a supported unit stated next to the value, for "
            "example '40.2%' or '1,044 tonnes'",
        )

    values = {
        (m.group("number").strip(".,"), SUPPORTED_UNITS[m.group("unit").lower()])
        for m in candidates
    }
    if len({value for value, _unit in values}) > 1:
        _refuse(
            "multi_value_claim",
            f"version 1 audits one numeric value; this claim states "
            f"{len({v for v, _ in values})}",
        )
    if len({unit for _value, unit in values}) > 1:
        _refuse(
            "multi_metric_claim",
            "version 1 audits one metric in one unit at a time",
        )

    match = candidates[0]
    raw = match.group("number")
    if not NUMBER.match(raw):
        _refuse(
            "ambiguous_number",
            "the number is not in the supported form: an optional sign, ASCII "
            "digits, an optional decimal point, and comma thousands separators",
        )
    try:
        value = Decimal(raw.replace(",", ""))
    except InvalidOperation:
        _refuse("ambiguous_number", "the number could not be read exactly")

    unit = SUPPORTED_UNITS[match.group("unit").lower()]
    if _mixes_percent_and_points(text):
        _refuse(
            "percentage_point_conversion",
            "version 1 does not convert between percentages and percentage points",
        )
    return value, unit


def _mixes_percent_and_points(text: str) -> bool:
    has_percent = re.search(r"\d\s*%|\bpercent\b", text, re.I) is not None
    has_points = re.search(r"\bpercentage points?\b|\bpp\b", text, re.I) is not None
    return has_percent and has_points


def _periods(text: str) -> tuple[str | None, str | None]:
    """The reporting year and, when the claim states one, its baseline."""
    years = all_years(text)
    if not years:
        return None, None
    if len(years) == 1:
        return years[0], None
    reporting, baseline = years[0], years[-1]
    if re.search(r"\b(from|since)\s+" + re.escape(years[0]), text, re.I):
        reporting, baseline = years[-1], years[0]
    return reporting, baseline


__all__ = [
    "MAX_CLAIM_CHARS",
    "NUMBER",
    "SupportedClaim",
    "validate_claim",
    "validate_free_text",
]
