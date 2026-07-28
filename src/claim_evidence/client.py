"""The one public entry point.

External agents call `search_evidence()` or `audit_claim()`. There is no agent
loop here on purpose: this package answers questions about evidence, it does
not decide what to ask.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import psycopg

from .audit import audit_claim as _audit_claim
from .config import Settings
from .db import connect, delete_document, ready_documents
from .errors import ValidationError
from .facts import heuristic_claim
from .frontend import (
    as_id,
    get_audit_trace,
    get_document,
    get_evidence,
    health,
    list_documents,
)
from .ingest import ensure_schema, ingest_document
from .models import (
    AuditTrace,
    ClaimResult,
    DocumentSummary,
    EvidenceDetail,
    EvidenceMatch,
    HealthReport,
    IngestReport,
    RemovalReport,
)
from .ollama import OllamaClient, OllamaError
from .retrieve import retrieve, to_matches


class ClaimEvidence:
    def __init__(
        self,
        settings: Settings,
        conn: psycopg.Connection | None = None,
        client: OllamaClient | None = None,
    ) -> None:
        self.settings = settings
        self.conn = conn or connect(settings.database_url)
        self.ollama = client or OllamaClient(settings)

    @classmethod
    def from_env(cls) -> "ClaimEvidence":
        return cls(Settings.from_env())

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "ClaimEvidence":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- setup --------------------------------------------------------------

    def init_db(self) -> None:
        ensure_schema(self.conn, self.settings)

    def initialize_database(self) -> HealthReport:
        """Apply the idempotent schema, then report the same diagnostics as
        `health()` so a caller sees in one call whether it can proceed."""
        ensure_schema(self.conn, self.settings)
        return self.health()

    def health(self) -> HealthReport:
        return health(self.conn, self.settings, self.ollama.session)

    def documents(self) -> list[dict]:
        return ready_documents(self.conn)

    def list_documents(self) -> list[DocumentSummary]:
        return list_documents(self.conn)

    def get_document(self, document_id: int | str) -> DocumentSummary:
        return get_document(self.conn, document_id)

    def remove_document(
        self, document_id: int | str, *, confirm_document_id: int | str
    ) -> RemovalReport:
        """Delete one document's index rows.

        Requires the id twice: an accidental call with the wrong argument
        should be a validation error, not a silent re-index of 494 pages.
        Source PDFs, output directories, and page images are never touched.
        """
        identifier = as_id(document_id, "document_id")
        confirmation = as_id(confirm_document_id, "confirm_document_id")
        if identifier != confirmation:
            raise ValidationError(
                "confirm_document_id must equal document_id; nothing was removed"
            )
        summary = get_document(self.conn, identifier)
        deleted = delete_document(self.conn, identifier)
        self.conn.commit()
        return RemovalReport(
            document_id=identifier, name=summary.name, deleted=deleted
        )

    def get_audit_trace(self, audit_id: int | str) -> AuditTrace:
        return get_audit_trace(self.conn, audit_id)

    def get_evidence(self, evidence_id: int | str) -> EvidenceDetail:
        return get_evidence(self.conn, evidence_id)

    # --- ingestion ----------------------------------------------------------

    def ingest_document(
        self,
        output_root: str | Path,
        *,
        source_pdf: str | Path | None = None,
        source_uri: str | None = None,
        force: bool = False,
        extract_narrative_facts: bool = True,
    ) -> IngestReport:
        """Index a completed output root.

        Unchanged sources are a no-op. ``force=True`` rebuilds anyway, keeping
        the current version queryable until the replacement passes its checks.
        """
        return ingest_document(
            self.conn,
            self.ollama,
            self.settings,
            output_root,
            source_pdf=source_pdf,
            source_uri=source_uri,
            force=force,
            extract_narrative_facts=extract_narrative_facts,
        )

    # --- query --------------------------------------------------------------

    def search_evidence(
        self,
        query: str,
        *,
        document_ids: Sequence[int] | None = None,
        limit: int = 20,
    ) -> list[EvidenceMatch]:
        parsed = heuristic_claim(query)
        try:
            embedding = self.ollama.embed([query])[0]
        except OllamaError:
            embedding = None
        candidates = retrieve(
            self.conn, embedding, parsed, query, document_ids=document_ids, limit=limit
        )
        return to_matches(self.conn, candidates)

    def audit_claim(
        self,
        claim: str,
        *,
        document_ids: Sequence[int] | None = None,
        limit: int = 20,
    ) -> ClaimResult:
        return _audit_claim(
            self.conn,
            self.ollama,
            self.settings,
            claim,
            document_ids=document_ids,
            limit=limit,
        )


__all__ = ["ClaimEvidence"]
