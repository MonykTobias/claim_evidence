"""The frozen public API contract, checked against its own examples.

The same example files back this suite and `gw_detector_v2/tests/test_contract_v2.py`,
so backend and frontend cannot each believe a different shape is correct.

Run from the repository root with ``python tests/test_contract_v2.py``.
"""

from __future__ import annotations

import json
from typing import Any

import pydantic

from claim_evidence import contracts
from claim_evidence.errors import ValidationError
from claim_evidence.models import (
    AuditRequest,
    ClaimDecomposition,
    ClaimResult,
    ClaimVerification,
    DocumentSummary,
    EvidenceDetail,
    ProgressEvent,
    PublicError,
    Verdict,
)

MODELS: dict[str, Any] = {
    "audit_request": AuditRequest,
    "claim_decomposition": ClaimDecomposition,
    "claim_result": ClaimResult,
    "claim_verification": ClaimVerification,
    "document_summary": DocumentSummary,
    "error": PublicError,
    "evidence_detail": EvidenceDetail,
    "progress_event": ProgressEvent,
}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


def test_every_dto_has_examples() -> None:
    kinds = contracts.example_kinds()
    check(set(kinds) == set(MODELS), f"every example folder maps to a DTO: {kinds}")


def test_every_valid_example_parses() -> None:
    total = 0
    for kind, model in sorted(MODELS.items()):
        for name, payload in contracts.iter_examples(kind, valid=True):
            model(**payload)
            total += 1
    check(total >= 10, f"{total} checked-in valid examples parse through their DTO")


def test_every_invalid_example_is_rejected() -> None:
    total = 0
    for kind, model in sorted(MODELS.items()):
        for name, payload in contracts.iter_examples(kind, valid=False):
            try:
                model(**payload)
            except (ValidationError, pydantic.ValidationError):
                total += 1
                continue
            raise AssertionError(f"{kind}/{name} was accepted but must be rejected")
    check(total >= 18, f"{total} checked-in invalid examples are rejected")


def test_an_unknown_key_is_rejected_not_ignored() -> None:
    """A field a caller invents must not be silently dropped."""
    for kind in ("claim_result", "evidence_detail", "document_summary", "error"):
        _, payload = next(iter(contracts.iter_examples(kind, valid=False)))
        model = MODELS[kind]
        try:
            model(**payload)
        except (ValidationError, pydantic.ValidationError):
            continue
        raise AssertionError(f"{kind} accepted an unknown key")
    check(True, "unknown keys are rejected across the public DTOs")


def test_scope_is_explicit() -> None:
    check(
        AuditRequest(claim="c", scope="all").document_ids is None,
        "'all' resolves to every queryable document",
    )
    check(
        AuditRequest(claim="c", scope=[3, 1, 3]).document_ids == [1, 3],
        "a document list is de-duplicated and ordered",
    )
    for scope, why in (
        (None, "null"),
        ([], "an empty list"),
        ("everything", "an unknown word"),
    ):
        try:
            AuditRequest(claim="c", scope=scope)
        except (ValidationError, pydantic.ValidationError):
            continue
        raise AssertionError(f"{why} must not be accepted as a scope")
    try:
        AuditRequest(claim="c")
    except (ValidationError, pydantic.ValidationError):
        pass
    else:
        raise AssertionError("an omitted scope must not be accepted")
    check(True, "omitted, null, and empty scope are validation errors, never 'all'")


def test_an_empty_selection_never_broadens_to_all() -> None:
    """PV-014: the failure mode this rule exists to prevent."""
    try:
        request = AuditRequest(claim="c", scope=[])
    except (ValidationError, pydantic.ValidationError) as exc:
        check(
            "empty" in str(exc).lower(),
            "an empty selection says so instead of searching everything",
        )
        return
    raise AssertionError(
        f"an empty selection was accepted and resolved to {request.document_ids!r}"
    )


def test_vocabulary_matches_the_models() -> None:
    vocabulary = contracts.vocabulary()
    check(
        tuple(vocabulary["verdicts"]) == tuple(v.value for v in Verdict),
        "the published verdict list is the Verdict enum",
    )
    check(
        "insufficient" in vocabulary["verdicts"]
        and "insufficient_evidence" not in vocabulary["verdicts"],
        "the no-comparable-evidence verdict has exactly one literal: 'insufficient'",
    )
    check(
        set(vocabulary["error_status"]) == set(vocabulary["error_codes"]),
        "every error code maps to one HTTP status",
    )
    check(
        vocabulary["error_status"]["unsupported_claim"] == 422,
        "an unsupported claim is 422, decided before any audit work",
    )
    check(
        vocabulary["ingest_phases"][-1] == vocabulary["terminal_progress_phase"]
        and vocabulary["audit_phases"][-1] == vocabulary["terminal_progress_phase"],
        "both operations end on the same terminal phase",
    )
    check(
        set(vocabulary["terminal_job_statuses"]) <= set(vocabulary["job_statuses"]),
        "every terminal job status is a job status",
    )
    check(
        "cancelled" in vocabulary["job_statuses"],
        "cancellation is its own outcome, not a failure",
    )
    check(
        tuple(vocabulary["grounding_statuses"]) == ("token_grounded", "not_grounded")
        and tuple(vocabulary["entailment_outcomes"])[0] == "entailed",
        "both claim gates publish their own vocabulary",
    )
    check(
        "mixed" not in vocabulary["claim_batch_statuses"]
        and "mixed_outcomes" in vocabulary["claim_batch_statuses"],
        "a post whose claims came out differently is not a `mixed` verdict",
    )
    check(
        set(vocabulary["claim_batch_statuses"]) - set(vocabulary["verdicts"])
        == {"mixed_outcomes", "incomplete", "needs_review"},
        "and the post-level states that have no verdict equivalent are named",
    )
    check(
        vocabulary["max_atomic_claims"] == 20,
        "the claim limit is published, so a frontend keeps no second copy",
    )
    check(
        "interrupted" in vocabulary["version_statuses"]
        and "interrupted" in vocabulary["audit_statuses"],
        "interruption is a state a build and an audit can be reconciled into",
    )


def test_examples_carry_no_server_detail() -> None:
    """A public example is a specimen; a leak in one is a leak in the contract."""
    forbidden = ("C:\\", "postgresql://", "Traceback", "Bearer ", "/home/")
    for kind in contracts.example_kinds():
        for name, payload in contracts.iter_examples(kind, valid=True):
            text = json.dumps(payload)
            for needle in forbidden:
                check(
                    needle not in text,
                    f"valid example {kind}/{name} contains no {needle!r}",
                )


def test_document_summary_keeps_server_paths_out_of_the_public_shape() -> None:
    """The package returns them; the frontend allowlist is what withholds them."""
    _, payload = next(iter(contracts.iter_examples("document_summary", valid=True)))
    summary = DocumentSummary(**payload)
    check(
        summary.output_root and not summary.output_root.startswith("C:"),
        "the example's output root is repository-relative, not a machine path",
    )


def main() -> int:
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            print(f"\n--- {name} ---")
            function()
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
