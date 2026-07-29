"""The verdict truth table, and the wording the product is allowed to use.

`_deterministic` is exercised directly with fixture facts: the set rules in
PD-08 are arithmetic over comparison outcomes, and testing them through a
database and a model would test the plumbing instead of the rule.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from claim_evidence.audit import _collapse_duplicates, _deterministic
from claim_evidence.contracts import VERDICTS
from claim_evidence.facts import compare, heuristic_claim
from claim_evidence.models import Citation, EvidenceKind, EvidenceQuality, Verdict

ENTITY = "Danone S.A."
CLAIM = (
    "Danone reduced Scope 1 and 2 energy and industry emissions by 40.2% "
    "in 2025 versus 2020."
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


def parsed_claim(text: str = CLAIM):
    return heuristic_claim(text).model_copy(update={"subject": ENTITY})


def fact(
    value: str = "-40.2",
    *,
    fact_id: int = 1,
    evidence_id: int = 100,
    subject: str = ENTITY,
    unit: str = "%",
    reporting: str | None = "2025",
    baseline: str | None = "2020",
) -> dict[str, Any]:
    return {
        "id": fact_id,
        "evidence_id": evidence_id,
        "subject": subject,
        "metric": "Scope 1 & 2 energy and industry emissions vs. 2020",
        "value_decimal": Decimal(value),
        "unit": unit,
        "direction": "decrease",
        "comparison": "=",
        "reporting_period": reporting,
        "baseline_period": baseline,
        "scope": "Scope 1 & 2 energy and industry emissions",
        "geography": None,
        "qualifiers": {"displayed_value": f"({abs(Decimal(value))}) %"},
    }


def candidate(evidence_id: int, page: int = 359) -> dict[str, Any]:
    return {
        "row": {
            "id": evidence_id,
            "unit_key": f"p{page:04d}:table_row:tc1:r{evidence_id}",
            "kind": "table_row",
            "quality": "direct_table",
            "citable": True,
            "source_text": "Scope 1 & 2 energy and industry emissions | 2025: (40.2) %",
            "heading_path": [],
            "table_context": {},
            "artifact_path": f"page_{page:04d}/table_candidates.json",
            "geometry_precision": "cell",
            "source_order": evidence_id,
            "context_key": f"p{page:04d}:table:tc1:row:{evidence_id}",
            "pdf_page": page,
            "printed_page_label": None,
            "page_dir": f"page_{page:04d}",
            "document_id": 1,
            "document_name": "report.pdf",
            "sha256": "a" * 64,
            "source_uri": None,
        }
    }


class FakeConn:
    """Returns the scripted facts for whatever evidence ids are asked about."""

    def __init__(self, facts: list[dict[str, Any]]) -> None:
        self.facts = facts


def decide(facts: list[dict[str, Any]], claim_text: str = CLAIM):
    """Run the deterministic verdict over one scripted set of facts."""
    import claim_evidence.audit as audit_module

    candidates = [candidate(f["evidence_id"]) for f in facts]
    regions: dict[int, list] = {}
    reasons: dict[int, str] = {}
    comparisons: list = []

    original = audit_module.facts_for_evidence
    audit_module.facts_for_evidence = lambda conn, ids: [
        f for f in facts if f["evidence_id"] in set(ids)
    ]
    try:
        return audit_module._deterministic(
            None, parsed_claim(claim_text), candidates, regions, reasons, comparisons
        ), comparisons
    finally:
        audit_module.facts_for_evidence = original


# --- the truth table --------------------------------------------------------


@pytest.mark.parametrize(
    "values,expected",
    [
        (["-40.2"], Verdict.SUPPORTED),
        (["-40.2", "-40.2"], Verdict.SUPPORTED),
        (["-90"], Verdict.CONTRADICTED),
        (["-90", "-12"], Verdict.CONTRADICTED),
        (["-40.2", "-90"], Verdict.MIXED),
        (["-90", "-40.2"], Verdict.MIXED),
    ],
)
def test_the_verdict_set_rules(values, expected) -> None:
    facts = [
        fact(value, fact_id=index, evidence_id=100 + index)
        for index, value in enumerate(values, start=1)
    ]
    (verdict, rationale, citations, _missing, _scope, rule), _ = decide(facts)
    assert verdict is expected, f"{values} -> {verdict} (expected {expected})"
    assert rule, "every deterministic verdict names the rule that fired"
    assert citations, "a deterministic verdict always cites what it used"


def test_support_with_no_conflict_is_supported() -> None:
    (verdict, rationale, *_rest), _ = decide([fact("-40.2")])
    check(verdict is Verdict.SUPPORTED, "support and no conflict is supported")
    check(
        "indexed sources" in rationale,
        f"the wording qualifies the verdict by corpus ({rationale})",
    )


def test_conflict_with_no_support_is_contradicted() -> None:
    (verdict, rationale, *_rest), _ = decide([fact("-90")])
    check(verdict is Verdict.CONTRADICTED, "conflict and no support is contradicted")
    check("indexed sources" in rationale, "and it too is corpus-qualified")


def test_nothing_comparable_is_not_a_verdict() -> None:
    """`insufficient` is decided later, after the corpus was actually searched."""
    incomparable = fact("-40.2", subject="Nestle S.A.")
    (verdict, _rationale, citations, *_rest), comparisons = decide([incomparable])
    check(verdict is None, "an incomparable fact yields no deterministic verdict")
    check(citations == [], "and cites nothing")
    check(
        comparisons and comparisons[0].numeric.outcome == "incomparable",
        "the comparison is still reported, so the reason is visible",
    )


# --- duplicates -------------------------------------------------------------


def test_identical_evidence_counts_once() -> None:
    citation = Citation(
        evidence_id=1, document_id=1, document_name="report.pdf", pdf_page=359,
        source_kind=EvidenceKind.TABLE_ROW, quality=EvidenceQuality.DIRECT_TABLE,
        quote="(40.2) %", artifact_path="page_0359/table_candidates.json",
    )
    twin = citation.model_copy(update={"evidence_id": 2})
    elsewhere = citation.model_copy(update={"evidence_id": 3, "pdf_page": 360})

    collapsed = _collapse_duplicates(
        [(citation, "a"), (twin, "b"), (elsewhere, "c")]
    )
    check(len(collapsed) == 2, f"the same place counts once ({len(collapsed)})")
    check(
        [c.evidence_id for c, _ in collapsed] == [1, 3],
        "the first occurrence is kept and the other page survives",
    )


def test_a_duplicate_cannot_turn_a_conflict_into_a_disagreement() -> None:
    """The same figure printed twice is one source, not two that disagree."""
    facts = [
        fact("-90", fact_id=1, evidence_id=100),
        fact("-90", fact_id=2, evidence_id=100),
    ]
    (verdict, *_rest), _ = decide(facts)
    check(
        verdict is Verdict.CONTRADICTED,
        f"repeating the conflicting figure stays contradicted ({verdict})",
    )


# --- one comparison, one answer ---------------------------------------------


def test_the_verdict_and_the_explanation_come_from_one_comparison() -> None:
    facts = [fact("-40.2", evidence_id=100), fact("-90", evidence_id=101)]
    (verdict, _rationale, citations, *_rest), comparisons = decide(facts)
    outcomes = {c.evidence_id: c.numeric.outcome for c in comparisons}
    check(verdict is Verdict.MIXED, "both a match and a conflict were found")
    check(
        outcomes == {100: "match", 101: "conflict"},
        f"the explanation reports the same two outcomes ({outcomes})",
    )
    cited = {c.evidence_id for c in citations}
    check(
        cited == {100, 101},
        "and the verdict cites exactly the evidence those comparisons used",
    )


def test_the_comparator_and_the_verdict_agree_on_every_fact() -> None:
    for value, outcome in (("-40.2", "match"), ("-90", "conflict")):
        direct, _reason = compare(parsed_claim(), fact(value))
        (_verdict, _rationale, _citations, *_rest), comparisons = decide([fact(value)])
        check(
            direct == comparisons[0].numeric.outcome == outcome,
            f"{value}: compare() and the audit agree on {outcome}",
        )


# --- vocabulary -------------------------------------------------------------


def test_every_verdict_is_in_the_published_vocabulary() -> None:
    check(
        {v.value for v in Verdict} == set(VERDICTS),
        "the enum and the published contract list the same four verdicts",
    )


def test_no_verdict_claims_independent_truth() -> None:
    """PD-08: "proof" means citable support in the selected sources."""
    forbidden = ("proven true", "is true", "proves the claim", "fact-checked")
    for values in (["-40.2"], ["-90"], ["-40.2", "-90"]):
        facts = [
            fact(value, fact_id=i, evidence_id=100 + i)
            for i, value in enumerate(values, start=1)
        ]
        (_verdict, rationale, *_rest), _ = decide(facts)
        for phrase in forbidden:
            check(
                phrase not in rationale.lower(),
                f"{values} rationale avoids {phrase!r}",
            )
        check(
            "indexed sources" in rationale,
            f"{values} rationale names the corpus it is about",
        )


def main() -> int:
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not callable(function):
            continue
        marks = getattr(function, "pytestmark", [])
        cases = [m.args[1] for m in marks if m.name == "parametrize"]
        print(f"\n--- {name} ---")
        if cases:
            for values, expected in cases[0]:
                function(values, expected)
                print(f"[ok] {values} -> {expected}")
        else:
            function()
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
