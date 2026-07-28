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
from .db import connect, ready_documents
from .ingest import ensure_schema, ingest_document
from .models import ClaimResult, EvidenceMatch, IngestReport
from .ollama import OllamaClient, OllamaError
from .retrieve import expand, retrieve, to_matches
from .facts import heuristic_claim


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

    def documents(self) -> list[dict]:
        return ready_documents(self.conn)

    # --- ingestion ----------------------------------------------------------

    def ingest_document(
        self,
        output_root: str | Path,
        *,
        source_pdf: str | Path | None = None,
        source_uri: str | None = None,
        extract_narrative_facts: bool = True,
    ) -> IngestReport:
        return ingest_document(
            self.conn,
            self.ollama,
            self.settings,
            output_root,
            source_pdf=source_pdf,
            source_uri=source_uri,
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
