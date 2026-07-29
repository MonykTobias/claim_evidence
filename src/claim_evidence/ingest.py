"""Ingestion: completed output root -> queryable, ready document version.

Resumable by construction. Evidence, embeddings, and facts all upsert on stable
keys, so a re-run picks up where an interrupted one stopped instead of
duplicating rows. The version only flips to ``ready`` after the integrity
checks pass, so a half-built index is never visible to a query.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Sequence

import psycopg

log = logging.getLogger(__name__)

from .config import Settings
from .db import (
    activate_version,
    add_alias,
    delete_facts,
    delete_stale_pages,
    find_version,
    init_schema,
    mark_version_failed,
    note_progress,
    set_embeddings,
    start_version,
    units_missing_embeddings,
    upsert_document,
    upsert_entity,
    upsert_evidence,
    upsert_fact,
    upsert_page,
    vector_dimension,
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
from .progress import ProgressCallback, ProgressReporter, classify_error
from .normalize import normalize_for_match
from .errors import ClaimEvidenceError, IndexNotReadyError
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


class IngestionError(ClaimEvidenceError):
    """Ingestion could not complete; the version stays invisible to queries."""


# The integrity gate runs a fixed set of checks, which is what makes
# `building_indexes` a measurable phase rather than a spinner.
VERIFY_STEPS = (
    "evidence count",
    "page coverage",
    "page links",
    "citation paths",
    "embedding coverage",
    "embedding dimension",
    "fact links",
)


def identity_key(
    root: Path, sha256: str | None, source_uri: str | None
) -> str:
    """The document's internal identity: hash of a namespace-tagged basis.

    Strongest available source wins -- the PDF's content hash, then a logical
    source URI, then the canonical output root. The tag is inside the hash so
    a URI and a path that happen to read the same are still different
    documents. `root` is already resolved, which on Windows also settles its
    casing, so there is nothing left to normalize here.
    """
    if sha256 is not None:
        basis = f"pdf:{sha256}"
    elif source_uri:
        basis = f"uri:{source_uri}"
    else:
        basis = f"output:{root}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


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
    progress: ProgressCallback | None = None,
) -> IngestReport:
    reporter = ProgressReporter(progress, "ingest")
    # Set once a version exists, so a failure before that point has nothing to
    # mark and a failure after it marks exactly one row.
    building: list[int] = []
    try:
        return _ingest(
            conn,
            client,
            settings,
            output_root,
            source_pdf=source_pdf,
            source_uri=source_uri,
            force=force,
            extract_narrative_facts=extract_narrative_facts,
            reporter=reporter,
            building=building,
        )
    except BaseException as exc:
        # One safe terminal event, then the original error reaches the caller
        # unchanged. A failed version is never activated.
        reporter.fail(exc)
        if building:
            _record_failure(conn, building[0], exc, reporter.phase)
        raise


def _record_failure(
    conn: psycopg.Connection, version_id: int, exc: BaseException, phase: str
) -> None:
    """Persist why the build stopped, without ever masking the original error.

    The exception on its way out is the caller's answer; a database that is
    itself unhappy must not replace it with a second, less useful one.
    """
    code, _retryable, _message = classify_error(exc)
    try:
        # The failing statement may have poisoned the transaction; a rollback
        # is what makes the status write possible at all.
        conn.rollback()
        mark_version_failed(conn, version_id, code, phase or "unknown")
    except Exception:  # noqa: BLE001 - never mask the original failure
        log.exception("could not record the failure of version %s", version_id)


def _ingest(
    conn: psycopg.Connection,
    client: OllamaClient,
    settings: Settings,
    output_root: str | Path,
    *,
    source_pdf: str | Path | None,
    source_uri: str | None,
    force: bool,
    extract_narrative_facts: bool,
    reporter: ProgressReporter,
    building: list[int],
) -> IngestReport:
    reader = OutputReader(output_root)
    reporter.start("validating_input", "Validating the extraction output")
    pages = reader.validate()
    reporter.done(
        "validating_input",
        f"Validated {len(pages)} pages",
        completed=len(pages),
        total=len(pages),
    )

    source_pdf_path = Path(source_pdf) if source_pdf else None
    document_name = (source_pdf_path or reader.root).name
    sha256 = reader.fingerprint(source_pdf_path) if source_pdf_path else None
    fingerprint = index_fingerprint(reader.fingerprint(source_pdf_path), settings)

    document_id = upsert_document(
        conn,
        document_name,
        sha256,
        source_uri,
        identity_key(reader.root, sha256, source_uri),
    )
    reporter.document_id = document_id
    existing = find_version(conn, document_id, fingerprint)
    if existing and existing["status"] == VersionStatus.READY and not force:
        # Same source, same embedding model: nothing to rebuild. Report the
        # stored counts anyway, so a no-op run does not read as an empty one.
        report = IngestReport(
            document_id=document_id,
            version_id=int(existing["id"]),
            status=VersionStatus.READY,
            fingerprint=fingerprint,
            reused_existing=True,
            pages=len(pages),
            warnings=reader.warnings,
            **_stored_counts(conn, int(existing["id"])),
        )
        reporter.done(
            "completed",
            "Document is already indexed",
            completed=1,
            total=1,
            details=_completion_details(report, reporter),
        )
        return report

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
    building.append(version_id)

    subject = organization_name(document_name, source_uri)
    subject_entity = upsert_entity(conn, "organization", subject, normalized_name(subject))
    add_alias(conn, subject_entity, document_name, normalized_name(document_name))
    conn.commit()

    blocks_by_page = reader.blocks_by_page()
    evidence_ids: dict[str, int] = {}
    all_units: list[EvidenceUnit] = []
    unit_count = 0
    warned = 0

    reporter.start("building_evidence", "Building evidence units", total=len(pages))
    for index, page in enumerate(pages, start=1):
        units = list(page_units(reader, page, blocks_by_page.get(page.page, [])))
        page_id = upsert_page(
            conn, version_id, page.page, page.width, page.height, page.rel
        )
        evidence_ids.update(upsert_evidence(conn, version_id, page_id, units))
        note_progress(conn, version_id)
        conn.commit()
        all_units.extend(units)
        unit_count += len(units)
        reporter.step(
            "building_evidence",
            f"Indexed page {page.page}",
            completed=index,
            total=len(pages),
            current_item=f"page {page.page}",
        )
        warned = _warn_new(reporter, "building_evidence", reader, warned)
    # A resumed attempt can still hold pages an earlier manifest listed.
    delete_stale_pages(conn, version_id, [p.page for p in pages])
    conn.commit()
    reporter.done(
        "building_evidence",
        f"Built {unit_count} evidence units",
        completed=len(pages),
        total=len(pages),
    )

    _embed_pending(conn, client, settings, version_id, reporter)
    # Drain the reader's warnings here, before fact extraction: _build_facts
    # emits its own, and draining afterwards would send each of those twice.
    _warn_new(reporter, "embedding_evidence", reader, warned)
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
        reporter,
    )

    embedded = _verify(conn, version_id, unit_count, len(pages), settings, reporter)

    reporter.start("activating_version", "Activating the new version")
    activate_version(conn, version_id)
    conn.commit()
    reporter.done("activating_version", "Version activated", completed=1, total=1)

    report = IngestReport(
        document_id=document_id,
        version_id=version_id,
        status=VersionStatus.READY,
        fingerprint=fingerprint,
        pages=len(pages),
        evidence_units=unit_count,
        visual_evidence_units=sum(
            1 for unit in all_units if unit.kind is EvidenceKind.VISUAL
        ),
        embedded_units=embedded,
        facts=facts,
        warnings=reader.warnings,
        skipped_artifacts=sorted(set(reader.skipped)),
        rejected_facts=rejected,
    )
    reporter.done(
        "completed",
        f"Indexed {report.pages} pages into {report.evidence_units} evidence units",
        completed=1,
        total=1,
        details=_completion_details(report, reporter),
    )
    return report


def _completion_details(report: IngestReport, reporter: ProgressReporter) -> dict[str, Any]:
    """Counts the caller already paid for, echoed onto the terminal event.

    Every value comes off the finished report, so showing completion statistics
    costs no extra queries.
    """
    return {
        "document_version_id": report.version_id,
        "page_count": report.pages,
        "evidence_count": report.evidence_units,
        "visual_evidence_count": report.visual_evidence_units,
        "embedding_count": report.embedded_units,
        "fact_count": report.facts,
        "warning_count": len(report.warnings),
        "elapsed_seconds": reporter.elapsed_seconds,
        "no_op": report.reused_existing,
    }


def _warn_new(
    reporter: ProgressReporter, phase: str, reader: OutputReader, already: int
) -> int:
    """Surface recoverable fallbacks the reader recorded, once each."""
    for warning in reader.warnings[already:]:
        reporter.warn(phase, warning)
    return len(reader.warnings)


def _stored_counts(conn: psycopg.Connection, version_id: int) -> dict[str, int]:
    row = conn.execute(
        """
        SELECT
            (SELECT count(*) FROM evidence_unit WHERE version_id = %(v)s) AS units,
            (SELECT count(*) FROM evidence_unit
             WHERE version_id = %(v)s AND kind = 'visual') AS visual,
            (SELECT count(*) FROM evidence_unit
             WHERE version_id = %(v)s AND embedding IS NOT NULL) AS embedded,
            (SELECT count(*) FROM fact WHERE version_id = %(v)s) AS facts
        """,
        {"v": version_id},
    ).fetchone()
    return {
        "evidence_units": int(row["units"]),
        "visual_evidence_units": int(row["visual"]),
        "embedded_units": int(row["embedded"]),
        "facts": int(row["facts"]),
    }


def _embed_pending(
    conn: psycopg.Connection,
    client: OllamaClient,
    settings: Settings,
    version_id: int,
    reporter: ProgressReporter,
) -> int:
    """Embed everything not yet embedded, one batch per HTTP call.

    The database transaction is committed between batches, so a slow model
    never holds a write transaction open.
    """
    pending = units_missing_embeddings(conn, version_id, EMBEDDED_KINDS)
    embedded = 0
    size = max(1, settings.embed_batch_size)
    batches = -(-len(pending) // size)  # ceil, and 0 when nothing is pending
    reporter.start("embedding_evidence", "Embedding evidence", total=batches)
    for number, start in enumerate(range(0, len(pending), size), start=1):
        chunk = pending[start : start + size]
        vectors = client.embed([row["normalized_text"] or " " for row in chunk])
        set_embeddings(conn, [(int(r["id"]), v) for r, v in zip(chunk, vectors)])
        note_progress(conn, version_id)
        conn.commit()
        embedded += len(chunk)
        reporter.step(
            "embedding_evidence",
            f"Embedded batch {number} of {batches}",
            completed=number,
            total=batches,
            current_item=f"batch {number}",
        )
    reporter.done(
        "embedding_evidence",
        f"Embedded {embedded} evidence units",
        completed=batches,
        total=batches,
    )
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
    reporter: ProgressReporter,
) -> tuple[int, list[str]]:
    stored = 0
    rejected: list[str] = []

    # Rebuild rather than resume: every current unit is reprocessed below, so
    # keeping an earlier attempt's facts only risks stale values, qualifiers,
    # and evidence links surviving into a ready version.
    delete_facts(conn, version_id)
    conn.commit()

    candidates = (
        [
            unit
            for unit in units
            if unit.kind is EvidenceKind.NARRATIVE and is_claim_like(unit.text)
        ]
        if extract_narrative_facts
        else []
    )
    reporter.start("extracting_facts", "Extracting facts", total=len(candidates))

    for fact in table_facts(units, subject):
        stored += _store_fact(conn, version_id, fact, subject_entity, evidence_ids)
    conn.commit()

    if not extract_narrative_facts:
        reporter.done(
            "extracting_facts",
            f"Derived {stored} table facts",
            completed=0,
            total=0,
        )
        return stored, rejected

    for index, unit in enumerate(candidates, start=1):
        try:
            extraction = client.structured(
                FactExtraction,
                FACT_EXTRACTION_SYSTEM,
                fact_extraction_prompt(unit, subject),
            )
        except OllamaError:
            # One passage the model could not process is not a failed index.
            reader.warnings.append(f"fact extraction failed for {unit.unit_key}")
            reporter.warn(
                "extracting_facts", "A passage could not be processed by the model"
            )
            continue
        kept, dropped = accept_llm_facts(extraction.facts, unit, subject)
        rejected.extend(dropped)
        for fact in kept:
            stored += _store_fact(conn, version_id, fact, subject_entity, evidence_ids)
        note_progress(conn, version_id)
        conn.commit()
        reporter.step(
            "extracting_facts",
            f"Examined {index} of {len(candidates)} passages",
            completed=index,
            total=len(candidates),
            current_item=unit.unit_key,
        )

    reporter.done(
        "extracting_facts",
        f"Stored {stored} facts",
        completed=len(candidates),
        total=len(candidates),
    )
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
    conn: psycopg.Connection,
    version_id: int,
    expected_units: int,
    expected_pages: int,
    settings: Settings,
    reporter: ProgressReporter,
) -> int:
    """The gate between a built version and a queryable one.

    Everything a resumed attempt could leave inconsistent is checked here:
    counts, stale pages, orphaned links, missing or wrong-width embeddings, and
    fact links that reach outside this version. Any failure raises, which
    leaves the version in ``building`` and whatever was ready still serving.

    Returns the version's embedding count, taken from the check query it
    already runs: a resumed build only writes the embeddings that were still
    missing, so "written this run" understates the finished index.
    """
    total = len(VERIFY_STEPS)
    reporter.start("building_indexes", "Checking index integrity", total=total)

    def passed(step: int, message: str) -> None:
        reporter.step(
            "building_indexes", message, completed=step, total=total,
            current_item=VERIFY_STEPS[step - 1],
        )

    counts = conn.execute(
        "SELECT count(*) AS n, count(embedding) AS embedded"
        " FROM evidence_unit WHERE version_id = %s",
        (version_id,),
    ).fetchone()
    stored = counts["n"]
    if stored != expected_units:
        raise IngestionError(f"stored {stored} evidence units, expected {expected_units}")
    passed(1, "Evidence count verified")

    pages = conn.execute(
        "SELECT count(*) AS n FROM page WHERE version_id = %s", (version_id,)
    ).fetchone()["n"]
    if pages != expected_pages:
        raise IngestionError(f"stored {pages} pages, expected {expected_pages}")
    passed(2, "Page coverage verified")

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
    passed(3, "Page links verified")

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
    passed(4, "Citation paths verified")

    unembedded = conn.execute(
        """
        SELECT count(*) AS n FROM evidence_unit
        WHERE version_id = %s AND kind = ANY(%s) AND embedding IS NULL
        """,
        (version_id, list(EMBEDDED_KINDS)),
    ).fetchone()["n"]
    if unembedded:
        raise IngestionError(f"{unembedded} embeddable units have no embedding")
    passed(5, "Embedding coverage verified")

    wrong = conn.execute(
        """
        SELECT count(*) AS n FROM evidence_unit
        WHERE version_id = %s AND embedding IS NOT NULL
          AND vector_dims(embedding) <> %s
        """,
        (version_id, settings.embed_dimensions),
    ).fetchone()["n"]
    if wrong:
        raise IngestionError(
            f"{wrong} stored embeddings are not {settings.embed_dimensions}-dimensional"
        )
    passed(6, "Embedding dimension verified")

    crossed = conn.execute(
        """
        SELECT count(*) AS n FROM fact f
        JOIN fact_evidence fe ON fe.fact_id = f.id
        JOIN evidence_unit e ON e.id = fe.evidence_id
        WHERE f.version_id = %s AND e.version_id <> f.version_id
        """,
        (version_id,),
    ).fetchone()["n"]
    if crossed:
        raise IngestionError(f"{crossed} fact links point outside this version")

    reporter.done(
        "building_indexes", "Index checks passed", completed=total, total=total
    )
    return int(counts["embedded"])


def ensure_schema(conn: psycopg.Connection, settings: Settings) -> None:
    """Apply the schema, then refuse to proceed on a dimension mismatch.

    ``vector(N)`` is templated in only when the table is first created, so
    re-initializing an existing database with a different configured dimension
    silently leaves the old column in place. Caught here, before the first
    embedding write, rather than at query time on a half-built index.
    """
    init_schema(conn, settings.embed_dimensions)
    declared = vector_dimension(conn)
    if declared is not None and declared != settings.embed_dimensions:
        # Not altered automatically: rewriting a populated vector column
        # discards every embedding in the database.
        raise IndexNotReadyError(
            f"database vector dimension {declared} does not match configured "
            f"dimension {settings.embed_dimensions}; use a fresh database or an "
            f"explicit full reindex migration"
        )


__all__ = [
    "EMBEDDED_KINDS",
    "VERIFY_STEPS",
    "IngestionError",
    "ensure_schema",
    "identity_key",
    "index_fingerprint",
    "ingest_document",
]
