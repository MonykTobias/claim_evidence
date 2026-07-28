"""Ingestion: completed output root -> queryable, ready document version.

Resumable by construction. Evidence, embeddings, and facts all upsert on stable
keys, so a re-run picks up where an interrupted one stopped instead of
duplicating rows. The version only flips to ``ready`` after the integrity
checks pass, so a half-built index is never visible to a query.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import psycopg

from .config import Settings
from .db import (
    activate_version,
    add_alias,
    find_version,
    init_schema,
    set_embeddings,
    start_version,
    units_missing_embeddings,
    upsert_document,
    upsert_entity,
    upsert_evidence,
    upsert_fact,
    upsert_page,
)
from .facts import (
    FACT_EXTRACTION_SYSTEM,
    accept_llm_facts,
    fact_extraction_prompt,
    is_claim_like,
    normalized_name,
    organization_name,
    table_facts,
)
from .models import (
    EvidenceKind,
    EvidenceUnit,
    Fact,
    FactExtraction,
    IngestReport,
    VersionStatus,
)
from .normalize import normalize_for_match
from .ollama import OllamaClient, OllamaError
from .source import OutputReader, page_units

# Table values are reached through their row, their fact, and lexical search.
# Embedding 11k near-identical "(40.2) %" strings buys nothing semantically and
# costs the bulk of ingestion time.
# ponytail: skip value-cell embeddings; embed them too if vector recall on
# bare cell text ever proves necessary.
EMBEDDED_KINDS = (
    str(EvidenceKind.NARRATIVE),
    str(EvidenceKind.TABLE_ROW),
    str(EvidenceKind.VISUAL),
    str(EvidenceKind.PAGE_MARKDOWN),
)


class IngestionError(RuntimeError):
    """Ingestion could not complete; the version stays invisible to queries."""


def index_fingerprint(source_fingerprint: str, settings: Settings) -> str:
    """Bind the index identity to the embedding model as well as the source."""
    digest = hashlib.sha256(source_fingerprint.encode())
    for part in settings.index_fingerprint_parts:
        digest.update(b"\x1f" + part.encode())
    return digest.hexdigest()


def ingest_document(
    conn: psycopg.Connection,
    client: OllamaClient,
    settings: Settings,
    output_root: str | Path,
    *,
    source_pdf: str | Path | None = None,
    source_uri: str | None = None,
    force: bool = False,
    extract_narrative_facts: bool = True,
) -> IngestReport:
    reader = OutputReader(output_root)
    pages = reader.validate()

    source_pdf_path = Path(source_pdf) if source_pdf else None
    document_name = (source_pdf_path or reader.root).name
    sha256 = reader.fingerprint(source_pdf_path) if source_pdf_path else None
    fingerprint = index_fingerprint(reader.fingerprint(source_pdf_path), settings)

    document_id = upsert_document(conn, document_name, sha256, source_uri)
    existing = find_version(conn, document_id, fingerprint)
    if existing and existing["status"] == VersionStatus.READY and not force:
        # Same source, same embedding model: nothing to rebuild. Report the
        # stored counts anyway, so a no-op run does not read as an empty one.
        return IngestReport(
            document_id=document_id,
            version_id=int(existing["id"]),
            status=VersionStatus.READY,
            fingerprint=fingerprint,
            reused_existing=True,
            pages=len(pages),
            warnings=reader.warnings,
            **_stored_counts(conn, int(existing["id"])),
        )

    version_id = start_version(
        conn,
        document_id,
        fingerprint,
        embed_model=settings.embed_model,
        embed_dim=settings.embed_dimensions,
        output_root=str(reader.root),
        source_pdf=str(source_pdf_path) if source_pdf_path else None,
        force=force,
    )

    subject = organization_name(document_name, source_uri)
    subject_entity = upsert_entity(conn, "organization", subject, normalized_name(subject))
    add_alias(conn, subject_entity, document_name, normalized_name(document_name))
    conn.commit()

    blocks_by_page = reader.blocks_by_page()
    evidence_ids: dict[str, int] = {}
    all_units: list[EvidenceUnit] = []
    unit_count = 0

    for page in pages:
        units = list(page_units(reader, page, blocks_by_page.get(page.page, [])))
        page_id = upsert_page(
            conn, version_id, page.page, page.width, page.height, page.page_dir.name
        )
        evidence_ids.update(upsert_evidence(conn, version_id, page_id, units))
        conn.commit()
        all_units.extend(units)
        unit_count += len(units)

    embedded = _embed_pending(conn, client, settings, version_id)
    facts, rejected = _build_facts(
        conn,
        client,
        version_id,
        subject,
        subject_entity,
        all_units,
        evidence_ids,
        reader,
        extract_narrative_facts,
    )

    # Only now is the replacement allowed to take over. If _verify raises,
    # the previous version is still 'ready' and still serving queries.
    _verify(conn, version_id, unit_count, settings)
    activate_version(conn, version_id)
    conn.commit()

    return IngestReport(
        document_id=document_id,
        version_id=version_id,
        status=VersionStatus.READY,
        fingerprint=fingerprint,
        pages=len(pages),
        evidence_units=unit_count,
        embedded_units=embedded,
        facts=facts,
        warnings=reader.warnings,
        skipped_artifacts=sorted(set(reader.skipped)),
        rejected_facts=rejected,
    )


def _stored_counts(conn: psycopg.Connection, version_id: int) -> dict[str, int]:
    row = conn.execute(
        """
        SELECT
            (SELECT count(*) FROM evidence_unit WHERE version_id = %(v)s) AS units,
            (SELECT count(*) FROM evidence_unit
             WHERE version_id = %(v)s AND embedding IS NOT NULL) AS embedded,
            (SELECT count(*) FROM fact WHERE version_id = %(v)s) AS facts
        """,
        {"v": version_id},
    ).fetchone()
    return {
        "evidence_units": int(row["units"]),
        "embedded_units": int(row["embedded"]),
        "facts": int(row["facts"]),
    }


def _embed_pending(
    conn: psycopg.Connection,
    client: OllamaClient,
    settings: Settings,
    version_id: int,
) -> int:
    """Embed everything not yet embedded, one batch per HTTP call.

    The database transaction is committed between batches, so a slow model
    never holds a write transaction open.
    """
    pending = units_missing_embeddings(conn, version_id, EMBEDDED_KINDS)
    embedded = 0
    size = max(1, settings.embed_batch_size)
    for start in range(0, len(pending), size):
        chunk = pending[start : start + size]
        vectors = client.embed([row["normalized_text"] or " " for row in chunk])
        set_embeddings(conn, [(int(r["id"]), v) for r, v in zip(chunk, vectors)])
        embedded += len(chunk)
    return embedded


def _build_facts(
    conn: psycopg.Connection,
    client: OllamaClient,
    version_id: int,
    subject: str,
    subject_entity: int,
    units: Sequence[EvidenceUnit],
    evidence_ids: dict[str, int],
    reader: OutputReader,
    extract_narrative_facts: bool,
) -> tuple[int, list[str]]:
    stored = 0
    rejected: list[str] = []

    for fact in table_facts(units, subject):
        stored += _store_fact(conn, version_id, fact, subject_entity, evidence_ids)
    conn.commit()

    if not extract_narrative_facts:
        return stored, rejected

    for unit in units:
        if unit.kind is not EvidenceKind.NARRATIVE or not is_claim_like(unit.text):
            continue
        try:
            extraction = client.structured(
                FactExtraction,
                FACT_EXTRACTION_SYSTEM,
                fact_extraction_prompt(unit, subject),
            )
        except OllamaError as exc:
            reader.warnings.append(f"fact extraction failed for {unit.unit_key}: {exc}")
            continue
        kept, dropped = accept_llm_facts(extraction.facts, unit, subject)
        rejected.extend(dropped)
        for fact in kept:
            stored += _store_fact(conn, version_id, fact, subject_entity, evidence_ids)
        conn.commit()

    return stored, rejected


def _store_fact(
    conn: psycopg.Connection,
    version_id: int,
    fact: Fact,
    subject_entity: int,
    evidence_ids: dict[str, int],
) -> int:
    ids = [evidence_ids[key] for key in fact.evidence_keys if key in evidence_ids]
    if not ids:
        return 0
    metric_entity = upsert_entity(
        conn, "metric", fact.metric, normalized_name(fact.metric)
    )
    upsert_fact(
        conn,
        version_id,
        fact,
        normalize_for_match(fact.metric),
        subject_entity,
        ids,
    )
    del metric_entity  # registered for graph traversal, not needed inline
    return 1


def _verify(
    conn: psycopg.Connection, version_id: int, expected_units: int, settings: Settings
) -> None:
    """Count, foreign-key, citation-path, and embedding-dimension checks."""
    stored = conn.execute(
        "SELECT count(*) AS n FROM evidence_unit WHERE version_id = %s", (version_id,)
    ).fetchone()["n"]
    if stored != expected_units:
        raise IngestionError(f"stored {stored} evidence units, expected {expected_units}")

    orphans = conn.execute(
        """
        SELECT count(*) AS n FROM evidence_unit e
        LEFT JOIN page p ON p.id = e.page_id
        WHERE e.version_id = %s AND p.id IS NULL
        """,
        (version_id,),
    ).fetchone()["n"]
    if orphans:
        raise IngestionError(f"{orphans} evidence units have no page")

    uncited = conn.execute(
        """
        SELECT count(*) AS n FROM evidence_unit e
        WHERE e.version_id = %s AND e.citable
          AND NOT EXISTS (SELECT 1 FROM evidence_region r WHERE r.evidence_id = e.id)
        """,
        (version_id,),
    ).fetchone()["n"]
    if uncited:
        raise IngestionError(f"{uncited} citable units have no region to highlight")

    dim = conn.execute(
        """
        SELECT vector_dims(embedding) AS d FROM evidence_unit
        WHERE version_id = %s AND embedding IS NOT NULL LIMIT 1
        """,
        (version_id,),
    ).fetchone()
    if dim and int(dim["d"]) != settings.embed_dimensions:
        raise IngestionError(
            f"stored embeddings have dimension {dim['d']}, "
            f"expected {settings.embed_dimensions}"
        )


def ensure_schema(conn: psycopg.Connection, settings: Settings) -> None:
    init_schema(conn, settings.embed_dimensions)


__all__ = [
    "EMBEDDED_KINDS",
    "IngestionError",
    "ensure_schema",
    "index_fingerprint",
    "ingest_document",
]
