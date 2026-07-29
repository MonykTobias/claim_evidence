"""The frozen public API contract.

One vocabulary for this backend and the ``gw_detector_v2`` frontend: the
verdicts, the states a document version, an audit, and a frontend job may be
in, the error codes a public surface may return, and the ordered progress
phases. The DTO *shapes* are the Pydantic models in :mod:`claim_evidence.models`;
this module is the enumerations they share with a consumer, plus the
checked-in examples both sides validate against.

A frontend that hard-codes these strings is a second, silently diverging copy
of the contract. It should read them from here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

CONTRACTS_DIR = Path(__file__).parent

API_CONTRACT = "claim_evidence/api"
API_CONTRACT_VERSION = "1.0"

# The four verdicts, exactly as PD-08 defines them. The no-comparable-evidence
# outcome is `insufficient`; `insufficient_evidence` appears in the acceptance
# checklist prose for the same state and is not an API literal.
VERDICTS = ("supported", "contradicted", "mixed", "insufficient")

# What a citation proves, not how confident anything is.
EVIDENCE_QUALITIES = (
    "direct_text",
    "direct_table",
    "verified_visual",
    "coarse_region",
    "none",
)

EVIDENCE_KINDS = ("narrative", "table_row", "table_value", "visual", "page_markdown")

# A build. `interrupted` is a reconciliation outcome, never a state a running
# process writes about itself.
VERSION_STATUSES = ("building", "ready", "degraded", "failed", "inactive", "interrupted")
QUERYABLE_VERSION_STATUSES = ("ready", "degraded")

AUDIT_STATUSES = ("running", "completed", "failed", "interrupted")
TERMINAL_AUDIT_STATUSES = ("completed", "failed", "interrupted")

# A frontend job. Cancellation is its own outcome: a job the user stopped and a
# job that broke are different events.
JOB_STATUSES = ("queued", "running", "completed", "failed", "cancelled", "interrupted")
TERMINAL_JOB_STATUSES = ("completed", "failed", "cancelled", "interrupted")

# Every code a default public surface may return. A caller branches on these;
# anything else is a bug, not a new category to render.
ERROR_CODES = (
    "validation_error",
    "unsupported_claim",
    "not_found",
    "index_not_ready",
    "dependency_unavailable",
    "internal_error",
)
# HTTP status each code maps to, so backend, CLI, and frontend agree.
ERROR_STATUS = {
    "validation_error": 400,
    "unsupported_claim": 422,
    "not_found": 404,
    "index_not_ready": 503,
    "dependency_unavailable": 503,
    "internal_error": 500,
}

# Ordered public phases. A phase that did not run is omitted, never reported as
# zero; `completed` is terminal for both operations.
INGEST_PHASES = (
    "validating_input",
    "building_evidence",
    "embedding_evidence",
    "extracting_facts",
    "building_indexes",
    "activating_version",
    "completed",
)
AUDIT_PHASES = (
    "parsing_claim",
    "retrieving_graph",
    "retrieving_full_text",
    "retrieving_vectors",
    "fusing_candidates",
    "expanding_context",
    "verifying_visuals",
    "deciding_verdict",
    "persisting_trace",
    "completed",
)
PROGRESS_STATUSES = ("start", "progress", "completed", "warning", "failed")
TERMINAL_PROGRESS_PHASE = "completed"

# Audit scope. `all` means every queryable document at audit start; a list means
# exactly those. Omitted, null, and empty are validation errors, never "all".
SCOPE_ALL = "all"

# Where a citation points. One closed vocabulary so a renderer can style a
# value cell differently from its descriptor without guessing.
REGION_ROLES = (
    "claim_text",
    "descriptor",
    "header",
    "unit",
    "value",
    "supporting_context",
    "visual_region",
    "unknown",
)
GEOMETRY_PRECISIONS = ("block", "cell", "row", "table", "crop", "page")
COORDINATE_SPACE = "normalized_top_left"

# A visual crop re-verification outcome.
VISUAL_RESULTS = ("support", "conflict", "illegible", "unrelated")


def vocabulary() -> dict[str, Any]:
    """The whole public vocabulary as plain data.

    Served to the frontend so the browser renders from the contract instead of
    from a second copy of these strings that drifts on its own schedule.
    """
    return {
        "contract": API_CONTRACT,
        "contract_version": API_CONTRACT_VERSION,
        "verdicts": list(VERDICTS),
        "evidence_qualities": list(EVIDENCE_QUALITIES),
        "evidence_kinds": list(EVIDENCE_KINDS),
        "version_statuses": list(VERSION_STATUSES),
        "queryable_version_statuses": list(QUERYABLE_VERSION_STATUSES),
        "audit_statuses": list(AUDIT_STATUSES),
        "terminal_audit_statuses": list(TERMINAL_AUDIT_STATUSES),
        "job_statuses": list(JOB_STATUSES),
        "terminal_job_statuses": list(TERMINAL_JOB_STATUSES),
        "error_codes": list(ERROR_CODES),
        "error_status": dict(ERROR_STATUS),
        "ingest_phases": list(INGEST_PHASES),
        "audit_phases": list(AUDIT_PHASES),
        "progress_statuses": list(PROGRESS_STATUSES),
        "terminal_progress_phase": TERMINAL_PROGRESS_PHASE,
        "scope_all": SCOPE_ALL,
        "region_roles": list(REGION_ROLES),
        "geometry_precisions": list(GEOMETRY_PRECISIONS),
        "coordinate_space": COORDINATE_SPACE,
        "visual_results": list(VISUAL_RESULTS),
    }


def example_path(*parts: str) -> Path:
    return CONTRACTS_DIR.joinpath(*parts)


def load_example(*parts: str) -> Any:
    return json.loads(example_path(*parts).read_text(encoding="utf-8"))


def iter_examples(kind: str, valid: bool) -> Iterable[tuple[str, Any]]:
    """Checked-in examples for one DTO, shared by backend and frontend tests."""
    folder = CONTRACTS_DIR / "api" / "examples" / ("valid" if valid else "invalid") / kind
    for path in sorted(folder.glob("*.json")):
        yield path.name, json.loads(path.read_text(encoding="utf-8"))


def example_kinds() -> list[str]:
    folder = CONTRACTS_DIR / "api" / "examples" / "valid"
    return sorted(child.name for child in folder.iterdir() if child.is_dir())


__all__ = [
    "API_CONTRACT",
    "API_CONTRACT_VERSION",
    "AUDIT_PHASES",
    "AUDIT_STATUSES",
    "COORDINATE_SPACE",
    "CONTRACTS_DIR",
    "ERROR_CODES",
    "ERROR_STATUS",
    "EVIDENCE_KINDS",
    "EVIDENCE_QUALITIES",
    "GEOMETRY_PRECISIONS",
    "INGEST_PHASES",
    "JOB_STATUSES",
    "PROGRESS_STATUSES",
    "QUERYABLE_VERSION_STATUSES",
    "REGION_ROLES",
    "SCOPE_ALL",
    "TERMINAL_AUDIT_STATUSES",
    "TERMINAL_JOB_STATUSES",
    "TERMINAL_PROGRESS_PHASE",
    "VERDICTS",
    "VERSION_STATUSES",
    "VISUAL_RESULTS",
    "example_kinds",
    "example_path",
    "iter_examples",
    "load_example",
    "vocabulary",
]
