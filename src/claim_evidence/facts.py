"""Claim-focused facts: deterministic from tables, LLM-extracted from prose.

The graph is intentionally narrow -- organizations, metrics, values, units,
periods, scope, geography -- because open-ended relation extraction produces
plausible edges that nothing in the source actually states.

Table facts are derived arithmetically and need no model. Narrative facts go
through the LLM but must echo an exact quote from their own evidence unit, so
a fabricated number is rejected before it can ever be cited.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any, Iterable, Literal

from .models import EvidenceKind, EvidenceUnit, Fact, ParsedClaim
from .normalize import (
    all_years,
    clean_text,
    contains_quote,
    content_tokens,
    detect_direction,
    is_approximate,
    normalize_for_match,
    normalize_unit,
    parse_value,
    scopes_comparable,
    signed_change,
    values_agree,
)

Comparison = Literal["match", "conflict", "incomparable"]

# A block worth sending to the fact extractor states something checkable: a
# quantity, a year, a target, or a comparison. Everything else is prose.
_CLAIM_LIKE = re.compile(
    r"\d|\bpercent\b|\btarget\b|\bcommit\w*\b|\bby 20\d\d\b|\bversus\b|\bvs\.?\b"
    r"|\bbaseline\b|\breduc\w*\b|\bincreas\w*\b|\breach\w*\b|\bachiev\w*\b",
    re.IGNORECASE,
)
_METRIC_CONTAINMENT = 0.5


def is_claim_like(text: str) -> bool:
    """Cheap gate before the LLM fact extractor; keeps token spend on prose that
    could actually be audited."""
    return bool(_CLAIM_LIKE.search(text)) and len(text.split()) >= 4


# --- deterministic table facts ---------------------------------------------


def table_fact(unit: EvidenceUnit, subject: str) -> Fact | None:
    """Derive one fact from a table-value evidence unit, or None if it has no
    comparable content."""
    if unit.kind is not EvidenceKind.TABLE_VALUE:
        return None
    context = unit.table_context
    descriptor = clean_text(context.get("descriptor") or "")
    header_path = [clean_text(h) for h in context.get("header_path") or []]
    raw_value = clean_text(context.get("value") or "")
    if not raw_value or not descriptor:
        return None

    value, inline_unit = parse_value(raw_value)
    unit_label = normalize_unit(context.get("unit") or "") or inline_unit

    header_text = " ".join(header_path)
    reporting = _last_year(header_text)
    # "... vs. 2020" in the row descriptor names the baseline the value is
    # measured against; without it a variation is not comparable to a claim.
    descriptor_years = all_years(descriptor)
    baseline = descriptor_years[-1] if descriptor_years else None
    if reporting is None and descriptor_years:
        reporting = None  # a year in the descriptor is a baseline, not a period

    metric = " | ".join(p for p in (descriptor, header_text) if p)
    direction = "unknown"
    if value is not None and unit_label == "%" and (baseline or "variation" in metric.casefold()):
        direction = "decrease" if value < 0 else "increase"

    return Fact(
        subject=subject,
        metric=metric,
        value_decimal=value,
        value_text=None if value is not None else raw_value,
        unit=unit_label,
        direction=direction,
        comparison="=",
        reporting_period=reporting,
        baseline_period=baseline,
        scope=descriptor,
        geography=None,
        qualifiers={"header_path": header_path, "displayed_value": raw_value},
        extraction_method="table",
        quote=raw_value,
        evidence_keys=[unit.unit_key],
    )


def _last_year(text: str) -> str | None:
    years = all_years(text)
    return years[-1] if years else None


def table_facts(units: Iterable[EvidenceUnit], subject: str) -> list[Fact]:
    return [fact for unit in units if (fact := table_fact(unit, subject))]


# --- LLM narrative facts ----------------------------------------------------

FACT_EXTRACTION_SYSTEM = """\
You extract auditable facts from one passage of a corporate report.

Rules:
- Only extract what the passage literally states. Never infer or complete a
  number that is not printed.
- `quote` must be copied verbatim from the passage, character for character.
- Use `value_decimal` for numbers and `value_text` only for non-numeric values.
- A reduction is a negative `value_decimal` with direction "decrease".
- Leave a field null when the passage does not state it. Do not guess a period,
  a scope, or a unit.
- Return an empty list when the passage states nothing auditable.
"""


def fact_extraction_prompt(unit: EvidenceUnit, subject: str) -> str:
    heading = " > ".join(unit.heading_path)
    return (
        f"Reporting entity: {subject}\n"
        f"Section: {heading or '(none)'}\n"
        f"Passage:\n{unit.text}\n"
    )


def accept_llm_facts(
    facts: Iterable[Fact], unit: EvidenceUnit, subject: str
) -> tuple[list[Fact], list[str]]:
    """Keep only facts whose quote really appears in the source evidence.

    This is the hallucination gate: the model may summarize, but it may not
    invent text, and a fact that cannot point at its own words is discarded.
    """
    kept: list[Fact] = []
    rejected: list[str] = []
    for fact in facts:
        if not fact.quote or not contains_quote(unit.text, fact.quote):
            rejected.append(f"{unit.unit_key}: quote not in source: {fact.quote!r}")
            continue
        if fact.value_decimal is None and not fact.value_text:
            rejected.append(f"{unit.unit_key}: fact has no value")
            continue
        kept.append(
            fact.model_copy(
                update={
                    "subject": fact.subject or subject,
                    "extraction_method": "llm",
                    "evidence_keys": [unit.unit_key],
                }
            )
        )
    return kept, rejected


# --- claim parsing ----------------------------------------------------------

CLAIM_PARSE_SYSTEM = """\
You decompose one atomic claim about a company report into comparable parts.

Rules:
- Copy the claim's own wording; do not normalize or expand abbreviations.
- `value_decimal` is the magnitude the claim states, unsigned. Direction is
  carried by `direction`.
- `reporting_period` is the period the value describes; `baseline_period` is
  the period it is compared against, if any.
- `scope` is the boundary the claim applies to, such as "Scope 1 and 2 energy
  and industry emissions". Leave it null when the claim does not name one.
- Set `comparison` to "~" and `approximate` to true only when the claim hedges
  with words such as about, roughly, or approximately.
- `key_terms` lists the distinctive words a search should preserve.
"""


def heuristic_claim(claim: str) -> ParsedClaim:
    """Deterministic parse used to seed retrieval and as an offline fallback."""
    text = clean_text(claim)
    years = all_years(text)
    value, unit = _claim_value(text)
    direction = detect_direction(text)
    approximate = is_approximate(text)
    reporting, baseline = (None, None)
    if len(years) >= 2:
        # "in 2025 versus 2020" -- the baseline is the one after the comparison.
        reporting, baseline = years[0], years[-1]
        if re.search(r"(from|since)\s+" + re.escape(years[0]), text, re.IGNORECASE):
            reporting, baseline = years[-1], years[0]
    elif years:
        reporting = years[0]
    return ParsedClaim(
        subject=text.split()[0] if text else "",
        metric=text,
        value_decimal=value,
        unit=unit,
        direction=direction,
        comparison="~" if approximate else "=",
        reporting_period=reporting,
        baseline_period=baseline,
        scope=text,
        approximate=approximate,
        key_terms=sorted(content_tokens(text)),
    )


def _claim_value(text: str) -> tuple[Decimal | None, str | None]:
    # `%` is not a word character, so a trailing \b would never match "40.2% in".
    match = re.search(
        r"(\d[\d,.]*)\s*(?:%|\b(?:percent|percentage points?|pp)\b)", text, re.I
    )
    if match:
        value, _ = parse_value(match.group(1))
        return value, "%"
    match = re.search(r"\b(\d[\d,.]*)\b", text)
    if match and match.group(1) not in all_years(text):
        value, _ = parse_value(match.group(1))
        return value, None
    return None, None


def merge_claim(parsed: ParsedClaim, fallback: ParsedClaim) -> ParsedClaim:
    """Fill gaps in the model's parse from the deterministic one."""
    update = {
        field: getattr(fallback, field)
        for field in ("value_decimal", "unit", "reporting_period", "baseline_period")
        if getattr(parsed, field) in (None, "")
    }
    if parsed.direction == "unknown":
        update["direction"] = fallback.direction
    if not parsed.key_terms:
        update["key_terms"] = fallback.key_terms
    if not parsed.metric:
        update["metric"] = fallback.metric
    if parsed.approximate or fallback.approximate:
        update["approximate"] = True
        update["comparison"] = "~"
    return parsed.model_copy(update=update)


# --- deterministic comparison ----------------------------------------------


def compare(claim: ParsedClaim, fact: dict[str, Any] | Fact) -> tuple[Comparison, str]:
    """Compare a parsed claim to one stored fact.

    Returns ``incomparable`` -- not ``conflict`` -- whenever a material
    qualifier does not line up. Refusing to compare is what keeps a vague claim
    from being forced into a confident contradiction against the nearest number.
    """
    row = fact if isinstance(fact, dict) else fact.model_dump()
    observed = row.get("value_decimal")
    if claim.value_decimal is None or observed is None:
        return "incomparable", "claim or fact has no numeric value"
    observed = Decimal(str(observed))

    claim_unit = normalize_unit(claim.unit)
    fact_unit = normalize_unit(row.get("unit"))
    if claim_unit != fact_unit:
        return "incomparable", f"unit {claim_unit!r} does not match {fact_unit!r}"

    fact_reporting = row.get("reporting_period")
    if claim.reporting_period and fact_reporting and claim.reporting_period != fact_reporting:
        return "incomparable", (
            f"reporting period {claim.reporting_period} does not match {fact_reporting}"
        )
    if claim.reporting_period and not fact_reporting:
        return "incomparable", "fact states no reporting period"

    fact_baseline = row.get("baseline_period")
    if claim.baseline_period and claim.baseline_period != fact_baseline:
        return "incomparable", (
            f"baseline {claim.baseline_period} does not match {fact_baseline}"
        )
    if fact_baseline and not claim.baseline_period:
        return "incomparable", "fact is measured against a baseline the claim omits"

    claim_scope = claim.scope or claim.metric
    fact_scope = row.get("scope") or row.get("metric") or ""
    if not scopes_comparable(claim_scope, fact_scope):
        return "incomparable", f"scope {claim_scope!r} is not comparable to {fact_scope!r}"

    if metric_containment(claim.metric, str(row.get("metric") or "")) < _METRIC_CONTAINMENT:
        return "incomparable", "metric wording does not overlap enough to compare"

    claimed = signed_change(claim.value_decimal, claim.direction)
    fact_direction = row.get("direction") or "unknown"
    if claim.direction == "unknown" and fact_direction != "unknown":
        claimed = signed_change(claim.value_decimal, fact_direction)
    if values_agree(claimed, observed, approximate=claim.approximate):
        return "match", f"claimed {claimed} matches reported {observed}"
    return "conflict", f"claimed {claimed} but the source reports {observed}"


def metric_containment(claim_metric: str, fact_metric: str) -> float:
    """Share of the claim's distinctive words that the fact also uses."""
    claim_tokens = content_tokens(claim_metric)
    if not claim_tokens:
        return 0.0
    fact_tokens = content_tokens(fact_metric)
    return len(claim_tokens & fact_tokens) / len(claim_tokens)


def organization_name(document_name: str, source_uri: str | None = None) -> str:
    """Best-effort reporting-entity label for a document.

    Conservative on purpose: it never merges two spellings into one entity, it
    just gives table facts a stable subject to hang off.
    """
    stem = re.sub(r"[_\-]+", " ", document_name).strip()
    stem = re.sub(r"\.(pdf|PDF)$", "", stem)
    return clean_text(stem) or (source_uri or document_name)


def normalized_name(name: str) -> str:
    return normalize_for_match(name)


__all__ = [
    "CLAIM_PARSE_SYSTEM",
    "FACT_EXTRACTION_SYSTEM",
    "accept_llm_facts",
    "compare",
    "fact_extraction_prompt",
    "heuristic_claim",
    "is_claim_like",
    "merge_claim",
    "metric_containment",
    "normalized_name",
    "organization_name",
    "table_fact",
    "table_facts",
]
