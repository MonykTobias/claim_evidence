"""Claim auditing: retrieve, compare deterministically, adjudicate, cite.

Order matters. Arithmetic runs first and is authoritative whenever every
material qualifier lines up, because a model asked to compare two numbers will
occasionally agree with the wrong one. The LLM is only consulted for semantic
qualification, narrative evidence, and genuine ambiguity, and it can never
manufacture a citation the retriever did not return.

Fails closed throughout: no citable evidence means `insufficient`, never a
guess drawn from generated Markdown or an unverified image summary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import psycopg

from .config import Settings
from .db import (
    create_audit,
    facts_for_evidence,
    finish_audit,
    record_candidates,
    regions_for,
)
from .facts import (
    CLAIM_PARSE_SYSTEM,
    SCOPE_MISMATCH,
    compare,
    heuristic_claim,
    merge_claim,
)
from .models import (
    Adjudication,
    Citation,
    ClaimResult,
    EvidenceKind,
    EvidenceQuality,
    ParsedClaim,
    Verdict,
)
from .ollama import OllamaClient, OllamaError
from .retrieve import expand, retrieve, to_citation
from .vision import verify_visual

ADJUDICATION_SYSTEM = """\
You decide whether the cited evidence supports one atomic claim.

Rules:
- Judge only from the evidence passages given. Never use outside knowledge.
- "supported" requires every material qualifier -- metric, scope, unit,
  reporting period, baseline -- to match, backed by at least one passage.
- "contradicted" requires evidence about the same metric, scope, unit, and
  periods that reports an incompatible value.
- "mixed" is for comparable evidence that disagrees with itself.
- "insufficient" is the answer when the evidence does not name the claim's
  qualifiers, or only generated page context is available. Prefer it over a
  guess, and list what is missing in `missing_qualifiers`.
- `supporting_evidence_ids` may only contain ids present in the passages.
"""

MAX_PASSAGES = 15
PASSAGE_CHARS = 800

DIRECT_QUALITY = {
    EvidenceKind.NARRATIVE: EvidenceQuality.DIRECT_TEXT,
    EvidenceKind.TABLE_ROW: EvidenceQuality.DIRECT_TABLE,
    EvidenceKind.TABLE_VALUE: EvidenceQuality.DIRECT_TABLE,
}
_QUALITY_ORDER = [
    EvidenceQuality.DIRECT_TABLE,
    EvidenceQuality.DIRECT_TEXT,
    EvidenceQuality.VERIFIED_VISUAL,
    EvidenceQuality.COARSE_REGION,
    EvidenceQuality.NONE,
]


class AuditError(RuntimeError):
    """The audit could not reach a verdict; the caller gets an error, not a guess."""


def parse_claim(client: OllamaClient, claim: str) -> ParsedClaim:
    """Structured parse with a deterministic fallback and gap-fill."""
    fallback = heuristic_claim(claim)
    try:
        parsed = client.structured(ParsedClaim, CLAIM_PARSE_SYSTEM, claim)
    except OllamaError:
        return fallback
    return merge_claim(parsed, fallback)


def audit_claim(
    conn: psycopg.Connection,
    client: OllamaClient,
    settings: Settings,
    claim: str,
    *,
    document_ids: Sequence[int] | None = None,
    limit: int = 20,
) -> ClaimResult:
    parsed = parse_claim(client, claim)
    audit_id = create_audit(
        conn, claim, parsed.model_dump(mode="json"), settings.chat_model, settings.embed_model
    )

    try:
        embedding = client.embed([claim])[0]
    except OllamaError:
        embedding = None

    candidates = expand(
        conn, retrieve(conn, embedding, parsed, claim, document_ids=document_ids, limit=limit)
    )
    if not candidates:
        return _finish(
            conn, audit_id, claim, Verdict.INSUFFICIENT,
            "No evidence matched the claim.", EvidenceQuality.NONE, [], ["evidence"],
        )

    regions = regions_for(conn, [int(c["row"]["id"]) for c in candidates])
    citable = [c for c in candidates if c["row"]["citable"]]

    visual_status: dict[int, str] = {}
    reasons: dict[int, str] = {}
    verdict, rationale, citations, missing, scope_ambiguous = _deterministic(
        conn, parsed, citable, regions, reasons
    )
    decided_by = "comparison"
    if verdict is None:
        decided_by = "adjudicator"
        verdict, rationale, citations, missing = _adjudicate(
            conn, client, claim, parsed, citable, regions, visual_status,
            scope_ambiguous=scope_ambiguous,
        )

    selected_ids = {citation.evidence_id for citation in citations}
    record_candidates(
        conn,
        audit_id,
        [
            {
                "evidence_id": (evidence_id := int(c["row"]["id"])),
                "lexical_rank": c.get("lexical_rank"),
                "lexical_score": c.get("lexical_score"),
                "vector_rank": c.get("vector_rank"),
                "vector_score": c.get("vector_score"),
                "graph_rank": c.get("graph_rank"),
                "graph_score": c.get("graph_score"),
                "combined_rank": c.get("combined_rank"),
                "combined_score": c["score"],
                "expanded_from": c.get("expanded_from"),
                "visual_status": visual_status.get(evidence_id, "not_applicable"),
                "selected": evidence_id in selected_ids,
                "reason": _candidate_reason(
                    c, evidence_id, selected_ids, reasons, visual_status, decided_by
                ),
            }
            for c in candidates
        ],
    )
    return _finish(conn, audit_id, claim, verdict, rationale, _quality(citations),
                   citations, missing)


def _deterministic(
    conn: psycopg.Connection,
    parsed: ParsedClaim,
    candidates: Sequence[dict[str, Any]],
    regions: dict[int, list[dict[str, Any]]],
    reasons: dict[int, str],
) -> tuple[Verdict | None, str, list[Citation], list[str], bool]:
    """Arithmetic verdict when every material qualifier aligns.

    Returns ``None`` when nothing is comparable, handing the claim to the
    semantic verifier rather than inventing an answer. The trailing flag says
    the only thing standing between the claim and the evidence was scope --
    the claim names a boundary the report never reports on.
    """
    ids = [int(c["row"]["id"]) for c in candidates]
    rows = {int(c["row"]["id"]): c["row"] for c in candidates}
    matches: list[tuple[Citation, str]] = []
    conflicts: list[tuple[Citation, str]] = []
    scope_rejections = 0

    for fact in facts_for_evidence(conn, ids):
        evidence_id = int(fact["evidence_id"])
        row = rows.get(evidence_id)
        if row is None:
            continue
        outcome, reason = compare(parsed, fact)
        reasons.setdefault(evidence_id, f"{outcome}: {reason}")
        if outcome == "incomparable":
            if reason.startswith(SCOPE_MISMATCH):
                scope_rejections += 1
            continue
        reasons[evidence_id] = f"{outcome}: {reason}"
        citation = to_citation(row, regions.get(evidence_id, []))
        (matches if outcome == "match" else conflicts).append((citation, reason))

    if matches and conflicts:
        # Comparable, authoritative evidence that disagrees with itself.
        return (
            Verdict.MIXED,
            "Comparable evidence disagrees: "
            + "; ".join(reason for _, reason in (matches[:1] + conflicts[:1])),
            [c for c, _ in matches[:2] + conflicts[:2]],
            [],
            False,
        )
    if matches:
        return (
            Verdict.SUPPORTED,
            f"Source evidence agrees: {matches[0][1]}.",
            [c for c, _ in matches[:3]],
            [],
            False,
        )
    if conflicts:
        return (
            Verdict.CONTRADICTED,
            f"Source evidence disagrees: {conflicts[0][1]}.",
            [c for c, _ in conflicts[:3]],
            [],
            False,
        )
    return None, "", [], [], scope_rejections > 0


def _adjudicate(
    conn: psycopg.Connection,
    client: OllamaClient,
    claim: str,
    parsed: ParsedClaim,
    candidates: Sequence[dict[str, Any]],
    regions: dict[int, list[dict[str, Any]]],
    visual_status: dict[int, str],
    scope_ambiguous: bool = False,
) -> tuple[Verdict, str, list[Citation], list[str]]:
    usable = _verify_visuals(conn, client, claim, candidates, regions, visual_status)
    if not usable:
        return (
            Verdict.INSUFFICIENT,
            "No citable source evidence matched the claim's qualifiers.",
            [],
            _missing_qualifiers(parsed),
        )

    # Bounded on purpose: an over-long prompt is silently truncated by the
    # runtime, which drops evidence without any signal that it happened.
    passages = "\n\n".join(
        f"[{cid.evidence_id}] page {cid.pdf_page} ({cid.source_kind}, {cid.quality}): "
        f"{text[:PASSAGE_CHARS]}"
        for cid, text, _ in usable[:MAX_PASSAGES]
    )
    note = (
        "\n\nNote: the evidence reports on narrower or different boundaries than "
        "the claim names, so no passage is scope-comparable to it.\n"
        if scope_ambiguous
        else ""
    )
    try:
        decision = client.structured(
            Adjudication,
            ADJUDICATION_SYSTEM,
            f"Claim:\n{claim}\n\nParsed qualifiers:\n"
            f"{parsed.model_dump_json(exclude_none=True)}{note}\n\nEvidence:\n{passages}",
        )
    except OllamaError as exc:
        raise AuditError(f"adjudication failed: {exc}") from exc

    if scope_ambiguous and decision.verdict is Verdict.CONTRADICTED:
        # Every comparable-looking fact was rejected on scope, so the report
        # simply does not measure what the claim asserts. Reporting a
        # contradiction here would be a confidently wrong answer about a
        # question the source never addresses.
        return (
            Verdict.INSUFFICIENT,
            "No evidence shares the claim's scope, so it can be neither "
            "confirmed nor contradicted.",
            [],
            sorted({*decision.missing_qualifiers, "scope"}),
        )

    by_id = {cid.evidence_id: cid for cid, _, _ in usable[:MAX_PASSAGES]}
    cited = [by_id[i] for i in decision.supporting_evidence_ids if i in by_id]
    if decision.verdict is not Verdict.INSUFFICIENT and not cited:
        # A verdict the model could not attach to a real passage is not a
        # verdict; downgrade rather than emit an uncited claim.
        return (
            Verdict.INSUFFICIENT,
            f"{decision.rationale} (no citable evidence was identified)",
            [],
            decision.missing_qualifiers or _missing_qualifiers(parsed),
        )
    return decision.verdict, decision.rationale, cited, decision.missing_qualifiers


def _verify_visuals(
    conn: psycopg.Connection,
    client: OllamaClient,
    claim: str,
    candidates: Sequence[dict[str, Any]],
    regions: dict[int, list[dict[str, Any]]],
    visual_status: dict[int, str],
) -> list[tuple[Citation, str, dict[str, Any]]]:
    """Citations the adjudicator may use.

    Visual candidates are cropped and re-checked; one that fails verification
    is dropped entirely rather than offered as weak support. The outcome is
    recorded per candidate so the trace shows why a chart was not used.
    """
    usable: list[tuple[Citation, str, dict[str, Any]]] = []
    for candidate in candidates:
        row = candidate["row"]
        evidence_id = int(row["id"])
        citation = to_citation(row, regions.get(evidence_id, []))
        kind = EvidenceKind(row["kind"])
        if kind is not EvidenceKind.VISUAL:
            usable.append(
                (
                    citation.model_copy(
                        update={"quality": DIRECT_QUALITY.get(kind, citation.quality)}
                    ),
                    row["source_text"],
                    row,
                )
            )
            continue

        page_png = _page_png(conn, evidence_id)
        if page_png is None:
            visual_status[evidence_id] = "unavailable"
            continue
        result = verify_visual(client, page_png, citation.regions, claim)
        if not result.supports_claim:
            visual_status[evidence_id] = "rejected"
            continue
        visual_status[evidence_id] = "verified"
        usable.append(
            (
                citation.model_copy(update={"quality": EvidenceQuality.VERIFIED_VISUAL}),
                f"{row['source_text']} [verified crop shows: {result.visible_text}]",
                row,
            )
        )
    return usable


def _page_png(conn: psycopg.Connection, evidence_id: int) -> Path | None:
    row = conn.execute(
        """
        SELECT v.output_root, p.page_dir FROM evidence_unit e
        JOIN page p ON p.id = e.page_id
        JOIN document_version v ON v.id = e.version_id
        WHERE e.id = %s
        """,
        (evidence_id,),
    ).fetchone()
    if not row:
        return None
    path = Path(row["output_root"]) / row["page_dir"] / "page.png"
    return path if path.is_file() else None


def _missing_qualifiers(parsed: ParsedClaim) -> list[str]:
    missing = [
        name
        for name, value in (
            ("scope", parsed.scope),
            ("unit", parsed.unit),
            ("reporting_period", parsed.reporting_period),
            ("value", parsed.value_decimal),
        )
        if not value
    ]
    return missing or ["comparable_evidence"]


def _quality(citations: Sequence[Citation]) -> EvidenceQuality:
    """The strongest quality actually cited."""
    present = {c.quality for c in citations}
    for quality in _QUALITY_ORDER:
        if quality in present:
            return quality
    return EvidenceQuality.NONE


def _finish(
    conn: psycopg.Connection,
    audit_id: int,
    claim: str,
    verdict: Verdict,
    rationale: str,
    quality: EvidenceQuality,
    citations: Sequence[Citation],
    missing: Sequence[str],
) -> ClaimResult:
    # A supporting verdict without a direct or re-verified citation is a bug;
    # fail closed rather than emit it.
    if verdict is Verdict.SUPPORTED and quality in (
        EvidenceQuality.NONE,
        EvidenceQuality.COARSE_REGION,
    ):
        verdict = Verdict.INSUFFICIENT
        rationale = f"{rationale} (no direct or re-verified citation)"
        citations = []

    finish_audit(
        conn,
        audit_id,
        verdict=str(verdict),
        rationale=rationale,
        quality=str(quality),
        missing_qualifiers=list(missing),
        citations=[c.model_dump(mode="json") for c in citations],
    )
    return ClaimResult(
        claim=claim,
        verdict=verdict,
        rationale=rationale,
        evidence_quality=quality,
        citations=list(citations),
        missing_qualifiers=list(missing),
        audit_id=audit_id,
    )


def _candidate_reason(
    candidate: dict[str, Any],
    evidence_id: int,
    selected_ids: set[int],
    reasons: dict[int, str],
    visual_status: dict[int, str],
    decided_by: str,
) -> str:
    """One short sentence on why a candidate was used or set aside.

    Retrieval metadata, not model reasoning: a deterministic comparison result,
    a crop-verification outcome, or how the candidate entered the pool.
    """
    if evidence_id in reasons:
        return reasons[evidence_id]
    status = visual_status.get(evidence_id)
    if status == "rejected":
        return "visual: crop did not show the claim"
    if status == "unavailable":
        return "visual: page image unavailable"
    if status == "verified":
        return "visual: crop verified"
    if evidence_id in selected_ids:
        return f"selected by {decided_by}"
    if candidate.get("expanded_from"):
        return "context expansion from a higher-ranked candidate"
    if not candidate["row"].get("citable", True):
        return "retrieved as context; not citable"
    return "retrieved but not selected"


__all__ = [
    "ADJUDICATION_SYSTEM",
    "MAX_PASSAGES",
    "PASSAGE_CHARS",
    "AuditError",
    "audit_claim",
    "parse_claim",
]
