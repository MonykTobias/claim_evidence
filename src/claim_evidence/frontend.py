"""Read-only APIs the local frontend needs, so it never touches SQL.

Everything here answers a question the frontend would otherwise answer by
querying tables directly and re-deriving provenance rules. Keeping one
authoritative representation here is the point: a second implementation of
"which box belongs to which cell" is a second implementation to get wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import psycopg
import requests

from .config import Settings
from .db import (
    SCHEMA_VERSION,
    audit_candidates,
    audit_run,
    document_summaries,
    evidence_row,
    index_counts,
    regions_for,
    schema_state,
    vector_dimension,
)
from .errors import IndexNotReadyError, NotFoundError, ValidationError
from .models import (
    AuditTrace,
    DecisionExplanation,
    DocumentSummary,
    EvidenceDetail,
    EvidenceKind,
    EvidenceQuality,
    GeometryPrecision,
    HealthReport,
    IndexReference,
    ModelHealth,
    Region,
    RegionRole,
    TraceCandidate,
    VersionStatus,
)

PAGE_IMAGE_NAME = "page.png"
PAGE_MARKDOWN_NAME = "docling_final.md"


def as_id(value: int | str, field: str) -> int:
    """Accept the string ids a web frontend carries, reject anything else."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        raise ValidationError(f"{field} must be an integer id, got {value!r}") from None


# --- health -----------------------------------------------------------------


def health(
    conn: psycopg.Connection | None,
    settings: Settings,
    session: requests.Session | None = None,
) -> HealthReport:
    """System diagnostics that never leak a credential or a connection string."""
    report = HealthReport()
    if conn is None:
        report.problems.append("database connection is not open")
    else:
        try:
            version, pgvector = schema_state(conn)
            report.database_reachable = True
            report.schema_version = version
            report.schema_current = version == SCHEMA_VERSION
            report.pgvector_version = pgvector
            if pgvector is None:
                report.problems.append("pgvector extension is not installed")
            if version is None:
                report.problems.append("schema is not initialized; run db init")
            elif version != SCHEMA_VERSION:
                report.problems.append(
                    f"schema version {version} is older than {SCHEMA_VERSION}; run db init"
                )
            report.configured_embedding_dimensions = settings.embed_dimensions
            declared = vector_dimension(conn)
            report.schema_embedding_dimensions = declared
            if declared is not None and declared != settings.embed_dimensions:
                # Reachable, initialized, and unusable: every embedding written
                # from here would be the wrong width, so this is a readiness
                # failure rather than a warning.
                report.schema_current = False
                report.problems.append(
                    f"database vector dimension {declared} does not match configured "
                    f"dimension {settings.embed_dimensions}; use a fresh database or "
                    f"an explicit full reindex migration"
                )
            counts = index_counts(conn, settings.build_stale_minutes)
            report.documents_ready = counts["ready"]
            report.documents_building = counts["building"]
            report.documents_failed = counts["failed"]
            report.documents_interrupted = counts["interrupted"]
            report.documents_inactive = counts["inactive"]
            report.evidence_units = counts["evidence"]
            report.embeddings = counts["embeddings"]
            report.facts = counts["facts"]
            report.stored_evidence_units = counts["stored_evidence"]
        except psycopg.Error:
            conn.rollback()
            # The driver message can carry the host and user; report the class.
            report.problems.append("database query failed")

    installed = _installed_models(settings, session)
    report.ollama_reachable = installed is not None
    if installed is None:
        report.problems.append("ollama is not reachable")
    for role, name in (
        ("embed", settings.embed_model),
        ("chat", settings.chat_model),
        ("vision", settings.vision_model),
    ):
        available = bool(installed and name in installed)
        report.models.append(ModelHealth(role=role, name=name, available=available))
        if installed is not None and not available:
            report.problems.append(f"{role} model {name} is not pulled")
    return report


def _installed_models(
    settings: Settings, session: requests.Session | None
) -> set[str] | None:
    http = session or requests
    try:
        response = http.get(f"{settings.ollama_base_url}/api/tags", timeout=10)
        response.raise_for_status()
        return {m["name"] for m in response.json().get("models", [])}
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return None


# --- documents --------------------------------------------------------------


def to_summary(row: dict[str, Any]) -> DocumentSummary:
    return DocumentSummary(
        document_id=int(row["document_id"]),
        document_version_id=int(row["version_id"]),
        name=row["name"],
        source_uri=row.get("source_uri"),
        source_sha256=row.get("sha256"),
        status=VersionStatus(row["status"]),
        page_count=int(row["page_count"]),
        evidence_count=int(row["evidence_count"]),
        fact_count=int(row["fact_count"]),
        visual_evidence_count=int(row["visual_count"]),
        embedding_model=row["embed_model"],
        embedding_dimensions=int(row["embed_dim"]),
        indexed_at=row.get("ready_at"),
        output_root=row["output_root"],
        source_pdf=row.get("source_pdf"),
    )


def list_documents(conn: psycopg.Connection) -> list[DocumentSummary]:
    return [to_summary(row) for row in document_summaries(conn)]


def get_document(conn: psycopg.Connection, document_id: int | str) -> DocumentSummary:
    identifier = as_id(document_id, "document_id")
    rows = document_summaries(conn, identifier)
    if not rows:
        raise NotFoundError(f"no document with id {identifier}")
    return to_summary(rows[0])


def require_document(conn: psycopg.Connection, document_id: int | str) -> int:
    return get_document(conn, document_id).document_id


# --- audit trace ------------------------------------------------------------


def get_audit_trace(conn: psycopg.Connection, audit_id: int | str) -> AuditTrace:
    identifier = as_id(audit_id, "audit_id")
    run = audit_run(conn, identifier)
    if run is None:
        raise NotFoundError(f"no audit with id {identifier}")

    citations = run.get("citations") or []
    references = [IndexReference(**r) for r in run.get("index_references") or []]
    explanation = run.get("decision_explanation") or {}
    return AuditTrace(
        audit_id=identifier,
        claim=run["claim"],
        # The audited corpus, which for an insufficient verdict with no
        # citations is the only record of what was actually searched.
        document_ids=sorted(
            {int(c["document_id"]) for c in citations if "document_id" in c}
            | {r.document_id for r in references}
        ),
        created_at=run.get("created_at"),
        decision_explanation=(
            DecisionExplanation(**explanation) if explanation.get("decided_by") else None
        ),
        timings=run.get("timings") or {},
        index_references=references,
        parsed_claim=run.get("parsed_claim") or {},
        verdict=run.get("verdict"),
        rationale=run.get("rationale"),
        evidence_quality=run.get("evidence_quality"),
        missing_qualifiers=list(run.get("missing_qualifiers") or []),
        citation_ids=[int(c["evidence_id"]) for c in citations if "evidence_id" in c],
        candidates=[
            TraceCandidate(
                evidence_id=int(row["evidence_id"]),
                pdf_page=int(row["pdf_page"]),
                source_kind=EvidenceKind(row["kind"]),
                text=row["source_text"][:400],
                lexical_rank=row.get("lexical_rank"),
                lexical_score=row.get("lexical_score"),
                vector_rank=row.get("vector_rank"),
                vector_score=row.get("vector_score"),
                graph_rank=row.get("graph_rank"),
                graph_score=row.get("graph_score"),
                combined_rank=row.get("combined_rank"),
                combined_score=float(row.get("combined_score") or 0.0),
                expanded_from=row.get("expanded_from"),
                visual_status=row.get("visual_status") or "not_applicable",
                selected=bool(row["selected"]),
                reason=row.get("reason"),
            )
            for row in audit_candidates(conn, identifier)
        ],
        error=run.get("error"),
    )


# --- evidence detail --------------------------------------------------------


def get_evidence(conn: psycopg.Connection, evidence_id: int | str) -> EvidenceDetail:
    identifier = as_id(evidence_id, "evidence_id")
    row = evidence_row(conn, identifier)
    if row is None:
        raise NotFoundError(f"no evidence with id {identifier}")

    output_root = Path(row["output_root"])
    page_dir = output_root / row["page_dir"]
    kind = EvidenceKind(row["kind"])
    context = row.get("table_context") or {}

    regions = [
        Region(
            bbox=(r["left_norm"], r["top_norm"], r["right_norm"], r["bottom_norm"]),
            role=RegionRole(r["role"]) if r["role"] in set(RegionRole) else r["role"],
            precision=GeometryPrecision(r["precision"]),
            source_bbox=tuple(r["source_bbox"]) if r.get("source_bbox") else None,
            source_origin=r.get("source_origin"),
        )
        for r in regions_for(conn, [identifier]).get(identifier, [])
    ]

    return EvidenceDetail(
        evidence_id=identifier,
        document_id=int(row["document_id"]),
        document_version_id=int(row["version_id"]),
        document_name=row["document_name"],
        pdf_page=int(row["pdf_page"]),
        printed_page_label=row.get("printed_page_label"),
        source_kind=kind,
        evidence_quality=EvidenceQuality(row["quality"]),
        geometry_precision=GeometryPrecision(row["geometry_precision"]),
        text=row["source_text"],
        # A table value's proof is its cells, not a prose sentence.
        quote=row["source_text"] if kind is not EvidenceKind.TABLE_VALUE else None,
        table_context=context,
        heading_path=list(row.get("heading_path") or []),
        page_width=float(row["page_width"]),
        page_height=float(row["page_height"]),
        page_image_path=_existing(page_dir / PAGE_IMAGE_NAME),
        regions=regions,
        artifact_paths=[
            path
            for path in (
                _existing(output_root / row["artifact_path"]),
                _existing(page_dir / PAGE_MARKDOWN_NAME),
            )
            if path
        ],
    )


def _existing(path: Path) -> str | None:
    """Resolve a stored artifact path, never one supplied by the caller."""
    try:
        return str(path) if path.is_file() else None
    except OSError:
        return None


def require_ready(conn: psycopg.Connection) -> None:
    """Fail with a typed error rather than returning a confidently empty result."""
    ready = conn.execute(
        "SELECT count(*) AS n FROM document_version WHERE status = 'ready'"
    ).fetchone()["n"]
    if not ready:
        raise IndexNotReadyError("no ready document version; ingest a document first")


__all__ = [
    "as_id",
    "get_audit_trace",
    "get_document",
    "get_evidence",
    "health",
    "list_documents",
    "require_ready",
    "to_summary",
]
