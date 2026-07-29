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
from .claims import validate_claim
from .config import Settings
from .db import (
    connect,
    delete_document,
    document_scope,
    queryable_version,
    ready_documents,
    reconcile_interrupted,
)
from .errors import IndexNotReadyError, NotFoundError, ValidationError
from .facts import heuristic_claim
from .frontend import (
    as_id,
    get_audit_trace,
    get_document,
    get_evidence,
    health,
    list_documents,
)
from .ingest import ensure_schema, ingest_document, retry_failed_facts
from .models import (
    AuditRequest,
    AuditTrace,
    ClaimResult,
    DocumentSummary,
    EvidenceDetail,
    EvidenceMatch,
    HealthReport,
    IndexReference,
    IngestReport,
    RemovalReport,
)
from .ollama import OllamaClient, OllamaError
from .progress import ProgressCallback
from .retrieve import retrieve, to_matches


class ClaimEvidence:
    def __init__(
        self,
        settings: Settings,
        conn: psycopg.Connection | None = None,
        client: OllamaClient | None = None,
    ) -> None:
        self.settings = settings
        self.conn = conn or connect(
            settings.database_url, settings.database_connect_timeout
        )
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

    def init_db(self) -> str:
        """``"initialized"`` or ``"unchanged"``; a mismatch raises instead."""
        return ensure_schema(self.conn, self.settings)

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
        reporting_entity: str,
        source_pdf: str | Path | None = None,
        source_uri: str | None = None,
        force: bool = False,
        extract_narrative_facts: bool = True,
        progress: ProgressCallback | None = None,
    ) -> IngestReport:
        """Index a completed output root, attributed to one reporting entity.

        ``reporting_entity`` is required and is what every stored fact is
        attributed to. It is not the document's filename: deriving it from
        one turned "danoneurdaccessible.pdf" into a company.

        Unchanged sources are a no-op. ``force=True`` rebuilds anyway, keeping
        the current version queryable until the replacement passes its checks.
        """
        return ingest_document(
            self.conn,
            self.ollama,
            self.settings,
            output_root,
            reporting_entity=reporting_entity,
            source_pdf=source_pdf,
            source_uri=source_uri,
            force=force,
            extract_narrative_facts=extract_narrative_facts,
            progress=progress,
        )

    def retry_facts(self, document_id: int | str) -> IngestReport:
        """Re-run only the narrative fact candidates that failed.

        Evidence, provenance, and embeddings are left exactly as they are: they
        are already correct, and rebuilding them to recover a handful of
        optional facts is minutes of work to fix seconds of it. A version whose
        last failure clears is promoted from degraded back to ready.
        """
        identifier = as_id(document_id, "document_id")
        version = queryable_version(self.conn, identifier)
        if version is None:
            raise IndexNotReadyError(
                f"document {identifier} has no queryable version to retry"
            )
        return retry_failed_facts(
            self.conn, self.ollama, self.settings, int(version["id"])
        )

    def reconcile_interrupted(self) -> dict[str, int]:
        """Mark work left running by a dead process as interrupted.

        Call once at startup. Never invents a completed or failed outcome for
        work whose end nobody saw, and never touches data that is already
        queryable.
        """
        counts = reconcile_interrupted(self.conn)
        self.conn.commit()
        return counts

    # --- query --------------------------------------------------------------

    def _resolve_scope(
        self, document_ids: Sequence[int] | None
    ) -> tuple[list[int] | None, list[IndexReference]]:
        """Turn a requested document selection into a real, ready corpus.

        Runs before anything expensive or persistent. An unknown or unindexed
        document must be an error the caller can act on -- passing it straight
        to SQL retrieves nothing, and "nothing retrieved" reads as `insufficient`,
        which is a factual statement about the report rather than about the
        request.

        ``None`` here means the caller asked for ``scope="all"``. It is not the
        same as an empty selection, which the request model rejects before this
        is reached.
        """
        if not document_ids:
            rows = document_scope(self.conn, None)
            if not rows:
                raise IndexNotReadyError(
                    "no ready document version; ingest a document first"
                )
            return None, [_reference(row) for row in rows]

        wanted: list[int] = []
        for value in document_ids:
            # bool is an int in Python; True is not document 1.
            if isinstance(value, bool):
                raise ValidationError(
                    f"document_ids must be integer ids, got {value!r}"
                )
            wanted.append(as_id(value, "document_ids"))
        unique = sorted(set(wanted))

        rows = document_scope(self.conn, unique)
        found = {int(row["document_id"]) for row in rows}
        missing = [i for i in unique if i not in found]
        if missing:
            raise NotFoundError(f"no document with id {missing[0]}")
        unready = [
            int(row["document_id"])
            for row in rows
            if row["document_version_id"] is None
        ]
        if unready:
            raise IndexNotReadyError(
                f"document {unready[0]} has no ready version to query"
            )
        return unique, [_reference(row) for row in rows]

    def search_evidence(
        self,
        query: str,
        *,
        document_ids: Sequence[int] | None = None,
        limit: int = 20,
    ) -> list[EvidenceMatch]:
        scope, _references = self._resolve_scope(document_ids)
        parsed = heuristic_claim(query)
        try:
            embedding = self.ollama.embed([query])[0]
        except OllamaError:
            embedding = None
        candidates, _channels = retrieve(
            self.conn, embedding, parsed, query, document_ids=scope, limit=limit
        )
        return to_matches(self.conn, candidates)

    def audit_claim(
        self,
        claim: str,
        *,
        scope: str | Sequence[int],
        reporting_entity: str,
        limit: int = 20,
        progress: ProgressCallback | None = None,
    ) -> ClaimResult:
        """Audit one claim against an explicitly chosen corpus.

        ``scope`` is ``"all"`` or a non-empty list of document ids. There is no
        value meaning "whatever you think best": an omitted selection used to
        widen silently to every document, which turns a frontend that had not
        finished loading into a much larger query whose answer looks normal.

        The claim is validated against the version-1 grammar *before* anything
        is written or any model is called, so an unsupported claim costs a typed
        error and nothing else.
        """
        request = AuditRequest(
            claim=claim, scope=list(scope) if not isinstance(scope, str) else scope,
            limit=limit,
        )
        supported = validate_claim(request.claim, reporting_entity=reporting_entity)
        selection, references = self._resolve_scope(request.document_ids)
        return _audit_claim(
            self.conn,
            self.ollama,
            self.settings,
            supported.text,
            document_ids=selection,
            index_references=references,
            limit=request.limit,
            reporting_entity=supported.reporting_entity,
            progress=progress,
        )


def _reference(row: dict) -> IndexReference:
    return IndexReference(
        document_id=int(row["document_id"]),
        document_version_id=int(row["document_version_id"]),
        embedding_model=row["embed_model"],
        embedding_dimensions=int(row["embed_dim"]),
    )


__all__ = ["ClaimEvidence"]
