"""Table-fact derivation, hallucination rejection, and claim comparison."""

from __future__ import annotations

import tempfile
from decimal import Decimal
from pathlib import Path


from fixtures import block, kpi_table, write_output_root

from claim_evidence.facts import (
    accept_llm_facts,
    compare,
    compare_detailed,
    heuristic_claim,
    is_claim_like,
    merge_claim,
    metric_containment,
    table_fact,
)
from claim_evidence.models import (
    EvidenceKind,
    EvidenceUnit,
    Fact,
    ParsedClaim,
)
from claim_evidence.source import OutputReader, page_units

SUPPORTED = "Danone reduced Scope 1 and 2 energy and industry emissions by 40.2% in 2025 versus 2020."
CONTRADICTED = SUPPORTED.replace("40.2%", "90%")
VAGUE = "Danone reduced all carbon emissions by 90% from 2020 to 2025."


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


def kpi_units() -> list[EvidenceUnit]:
    with tempfile.TemporaryDirectory() as temp:
        root = write_output_root(
            Path(temp) / "run",
            pages=1,
            blocks=[block(1, 1, "Nature indicators", heading_path=["4.8.2"])],
            tables={1: [kpi_table()]},
        )
        reader = OutputReader(root)
        page = reader.validate()[0]
        return list(page_units(reader, page, reader.blocks_by_page().get(1, [])))


def emissions_fact() -> Fact:
    unit = next(
        u
        for u in kpi_units()
        if u.kind is EvidenceKind.TABLE_VALUE and u.table_context.get("value") == "(40.2) %"
    )
    fact = table_fact(unit, "Danone")
    assert fact is not None
    return fact


def test_table_fact_reads_sign_period_and_baseline() -> None:
    fact = emissions_fact()
    check(fact.value_decimal == Decimal("-40.2"), "parenthesized value is negative")
    check(fact.unit == "%", "percentage-of-variation unit folded to %")
    check(fact.reporting_period == "2025", "reporting period from the column header")
    check(fact.baseline_period == "2020", "baseline period from the row descriptor")
    check(fact.direction == "decrease", "negative variation is a decrease")
    check(fact.extraction_method == "table", "table facts need no model")
    check(fact.quote == "(40.2) %", "quote is the displayed value")


def test_textual_cell_becomes_value_text() -> None:
    unit = EvidenceUnit(
        unit_key="p0001:table_value:tc:r2:c6",
        page=1,
        kind=EvidenceKind.TABLE_VALUE,
        text="x",
        normalized_text="x",
        table_context={
            "descriptor": "Renewable electricity proportion",
            "header_path": ["Externally verified"],
            "value": "Yes",
            "unit": "",
        },
    )
    fact = table_fact(unit, "Danone")
    check(fact is not None and fact.value_text == "Yes", "non-numeric cell kept as text")
    check(fact.value_decimal is None, "no numeric value invented for a word")


def test_supported_claim_matches() -> None:
    verdict, reason = compare(heuristic_claim(SUPPORTED), emissions_fact())
    check(verdict == "match", f"exact claim matches ({reason})")


def test_contradicted_claim_conflicts() -> None:
    verdict, reason = compare(heuristic_claim(CONTRADICTED), emissions_fact())
    check(verdict == "conflict", f"90% conflicts with 40.2% ({reason})")


def test_vague_claim_is_incomparable_not_contradicted() -> None:
    verdict, reason = compare(heuristic_claim(VAGUE), emissions_fact())
    check(verdict == "incomparable", f"vague scope refuses comparison ({reason})")
    check("scope" in reason, "reason names the scope mismatch")


def test_detailed_comparison_explains_a_match() -> None:
    result = compare_detailed(heuristic_claim(SUPPORTED), emissions_fact())
    check(result.outcome == "match", "the wrapper and the detail agree on the outcome")
    check(
        compare(heuristic_claim(SUPPORTED), emissions_fact())
        == (result.outcome, result.reason),
        "compare() is exactly this result, narrowed",
    )
    status = {q.qualifier: q.status for q in result.qualifiers}
    for name in ("scope", "unit", "reporting_period", "baseline_period", "metric"):
        check(status[name] == "match", f"{name} is a match ({status[name]})")
    check(result.numeric.outcome == "match", "the arithmetic matched")
    check(result.numeric.source_value == "(40.2) %", "the source keeps its printed form")
    check(result.numeric.claim_value == "40.2", "the claim value is reported unsigned")
    check(result.numeric.claim_direction == "decrease", "the claim's direction is reported")


def test_detailed_comparison_never_claims_an_unestablished_match() -> None:
    result = compare_detailed(heuristic_claim(VAGUE), emissions_fact())
    status = {q.qualifier: q.status for q in result.qualifiers}
    check(result.outcome == "incomparable", "a vague scope stays incomparable")
    check(status["scope"] == "mismatch", "the scope is reported as a mismatch")
    check(
        result.numeric.outcome == "incomparable",
        f"and the numbers are not compared ({result.numeric.outcome})",
    )
    check(
        result.numeric.outcome != "conflict",
        "an incomparable scope is never a numeric contradiction",
    )


def test_detailed_comparison_invents_no_source_values() -> None:
    """A narrative fact with no number must not grow one in the explanation."""
    bare = emissions_fact().model_dump()
    bare.update({"value_decimal": None, "value_text": "improved", "qualifiers": {}})
    result = compare_detailed(heuristic_claim(SUPPORTED), bare)
    check(result.numeric.outcome == "not_applicable", "no arithmetic was attempted")
    check(result.numeric.source_value is None, "no source value was invented")

    missing_scope = emissions_fact().model_dump()
    missing_scope.update({"scope": None, "metric": ""})
    status = {
        q.qualifier: q.status
        for q in compare_detailed(heuristic_claim(SUPPORTED), missing_scope).qualifiers
    }
    check(status["scope"] == "missing", "an omitted source qualifier is missing, not match")
    check(status["geography"] == "missing", "an unstated geography is missing on both sides")


def test_two_facts_produce_two_independent_comparisons() -> None:
    matching = compare_detailed(heuristic_claim(SUPPORTED), emissions_fact())
    conflicting = compare_detailed(heuristic_claim(CONTRADICTED), emissions_fact())
    check(
        (matching.numeric.outcome, conflicting.numeric.outcome) == ("match", "conflict"),
        "each comparison keeps its own outcome",
    )
    check(
        matching.numeric.source_value == conflicting.numeric.source_value == "(40.2) %",
        "both cite the same source value",
    )
    check(
        matching.numeric.claim_value != conflicting.numeric.claim_value,
        "with the two different claim values",
    )


def test_an_approximate_claim_is_never_compared() -> None:
    """The tolerance is gone: version 1 compares values exactly or not at all.

    A hedged claim is refused by `claims.validate_claim` before an audit opens.
    If one reaches the comparator anyway, the answer is `incomparable` -- never
    a match won by a tolerance nobody asked for, and never a `conflict`, which
    would report a disagreement that was really a rounding rule.
    """
    hedged = (
        "Danone reduced Scope 1 and 2 energy and industry emissions by roughly "
        "40% in 2025 versus 2020."
    )
    parsed = heuristic_claim(hedged)
    check(parsed.approximate, "hedged wording is still detected")
    verdict, reason = compare(parsed, emissions_fact())
    check(verdict == "incomparable", f"an approximate claim is incomparable ({verdict})")
    check("exactly" in reason, f"and the reason says why ({reason})")

    exact = hedged.replace("roughly 40%", "40%")
    check(
        compare(heuristic_claim(exact), emissions_fact())[0] == "conflict",
        "the same number stated exactly is compared, and 40 is not 40.2",
    )


def test_bounded_claims_are_not_compared_either() -> None:
    fact = emissions_fact()  # a 40.2% reduction
    for wording, operator in (
        ("by at least 40%", ">="),
        ("by at least 50%", ">="),
        ("by no more than 50%", "<="),
        ("by no more than 30%", "<="),
    ):
        parsed = heuristic_claim(SUPPORTED.replace("by 40.2%", wording))
        check(parsed.comparison == operator, f"{wording!r} still parses as {operator}")
        verdict, _ = compare(parsed, fact)
        check(
            verdict == "incomparable",
            f"{wording!r} is incomparable, not silently satisfied ({verdict})",
        )


def test_an_exact_value_match_is_the_only_match() -> None:
    fact = emissions_fact()  # a 40.2% reduction
    check(
        compare(heuristic_claim(SUPPORTED), fact)[0] == "match",
        "the stated figure matches the reported one",
    )
    for wrong in ("40.3%", "40%", "41%", "402%"):
        parsed = heuristic_claim(SUPPORTED.replace("40.2%", wrong))
        check(
            compare(parsed, fact)[0] == "conflict",
            f"{wrong} is a conflict, not a near-enough match",
        )


def test_a_different_reporting_entity_blocks_the_comparison() -> None:
    """Two figures about two companies are not evidence for each other."""
    fact = emissions_fact()
    parsed = heuristic_claim(SUPPORTED).model_copy(update={"subject": "Nestle S.A."})
    verdict, reason = compare(parsed, fact)
    check(verdict == "incomparable", f"a different entity is incomparable ({verdict})")
    check("Nestle" in reason, f"and the reason names both sides ({reason})")


def test_mismatched_qualifiers_are_incomparable() -> None:
    fact = emissions_fact()
    wrong_year = heuristic_claim(SUPPORTED.replace("2025", "2024"))
    check(compare(wrong_year, fact)[0] == "incomparable", "different period refuses")

    no_baseline = ParsedClaim(
        metric="Scope 1 and 2 energy and industry emissions",
        scope="Scope 1 and 2 energy and industry emissions",
        value_decimal=Decimal("40.2"),
        unit="%",
        direction="decrease",
        reporting_period="2025",
    )
    check(
        compare(no_baseline, fact)[0] == "incomparable",
        "a claim without a baseline is not compared to a vs-baseline value",
    )

    wrong_unit = no_baseline.model_copy(update={"unit": "number", "baseline_period": "2020"})
    check(compare(wrong_unit, fact)[0] == "incomparable", "unit mismatch refuses")


def test_valueless_comparison_is_incomparable() -> None:
    claim = ParsedClaim(metric="emissions", scope="scope 1 and 2")
    check(compare(claim, emissions_fact())[0] == "incomparable", "no number to compare")


def test_metric_containment() -> None:
    fact = emissions_fact()
    high = metric_containment("Scope 1 and 2 energy and industry emissions", fact.metric)
    low = metric_containment("all carbon emissions", fact.metric)
    check(high > low, f"matching wording scores higher ({high} > {low})")
    check(metric_containment("", fact.metric) == 0.0, "empty metric scores zero")


def test_claim_like_gate() -> None:
    check(is_claim_like("We reduced emissions by 40.2% in 2025."), "quantified prose")
    check(is_claim_like("Our target is carbon neutrality by 2050."), "target prose")
    check(not is_claim_like("This section describes our approach."), "generic prose")
    check(not is_claim_like("2025"), "a bare token is not a claim")


def test_hallucinated_fact_is_rejected() -> None:
    unit = EvidenceUnit(
        unit_key="p0359:narrative:p0359-b0003",
        page=359,
        kind=EvidenceKind.NARRATIVE,
        text="Scope 1 and 2 emissions fell by 40.2% against the 2020 baseline.",
        normalized_text="",
    )
    facts = [
        Fact(subject="Danone", metric="scope 1 and 2", value_decimal=Decimal("-40.2"),
             quote="fell by 40.2%"),
        Fact(subject="Danone", metric="scope 3", value_decimal=Decimal("-61.0"),
             quote="Scope 3 emissions fell by 61%"),
        Fact(subject="Danone", metric="scope 1 and 2", quote="fell by 40.2%"),
    ]
    kept, rejected = accept_llm_facts(facts, unit, "Danone")
    check(len(kept) == 1, "only the grounded fact survives")
    check(kept[0].extraction_method == "llm", "surviving fact marked as llm-extracted")
    check(kept[0].evidence_keys == [unit.unit_key], "fact bound to its own evidence")
    check(any("quote not in source" in r for r in rejected), "invented quote rejected")
    check(any("no value" in r for r in rejected), "valueless fact rejected")


def test_merge_claim_fills_gaps_only() -> None:
    model = ParsedClaim(metric="scope 1 and 2 emissions", value_decimal=Decimal("40.2"))
    merged = merge_claim(model, heuristic_claim(SUPPORTED))
    check(merged.value_decimal == Decimal("40.2"), "model value kept")
    check(merged.reporting_period == "2025", "missing period filled from heuristic")
    check(merged.metric == "scope 1 and 2 emissions", "model metric not overwritten")


def test_from_to_wording_orders_periods() -> None:
    parsed = heuristic_claim(VAGUE)
    check(parsed.reporting_period == "2025", "'from 2020 to 2025' reports 2025")
    check(parsed.baseline_period == "2020", "'from 2020' is the baseline")

    versus = heuristic_claim(SUPPORTED)
    check(versus.reporting_period == "2025", "'in 2025 versus 2020' reports 2025")
    check(versus.baseline_period == "2020", "'versus 2020' is the baseline")


def main() -> int:
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
