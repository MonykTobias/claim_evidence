"""PostgreSQL access layer.

Versioned SQL files run directly; there is no ORM and no migration framework.
Every read path filters on ``document_version.status = 'ready'``, which is what
keeps an interrupted build invisible to queries rather than partially visible.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Sequence

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .config import DEFAULT_DATABASE_CONNECT_TIMEOUT
from .errors import DependencyUnavailableError
from .models import EvidenceUnit, Fact, VersionStatus

SQL_DIR = Path(__file__).parent / "sql"
EMBED_DIM_TOKEN = "EMBED_DIM"
SCHEMA_VERSION = 4


def connect(
    database_url: str, connect_timeout: float = DEFAULT_DATABASE_CONNECT_TIMEOUT
) -> psycopg.Connection:
    """Open a bounded connection, or fail with a safe typed error.

    libpq takes whole seconds and treats anything under 2 as 2, so the
    configured value is rounded up rather than passed through verbatim. The
    driver's own message names the host, user, and sometimes the password, so
    it is kept as ``__cause__`` for the server log and never surfaced.
    """
    try:
        conn = psycopg.connect(
            database_url,
            row_factory=dict_row,
            autocommit=False,
            connect_timeout=max(1, int(connect_timeout)),
        )
    except psycopg.OperationalError as exc:
        raise DependencyUnavailableError("PostgreSQL is unavailable.") from exc
    try:
        register_vector(conn)
    except psycopg.ProgrammingError:
        # The extension is created by `db init`; on a fresh database the vector
        # type does not exist yet and registration succeeds after init_schema.
        conn.rollback()
    return conn


def init_schema(conn: psycopg.Connection, embed_dim: int) -> None:
    for path in sorted(SQL_DIR.glob("*.sql")):
        sql = path.read_text(encoding="utf-8").replace(EMBED_DIM_TOKEN, str(int(embed_dim)))
        conn.execute(sql)
    conn.commit()
    register_vector(conn)


def normalize_embedding(vector: Sequence[float]) -> Vector:
    """L2-normalize so the HNSW inner-product index ranks like cosine.

    Returns a pgvector ``Vector`` rather than a list: a bare Python list is
    adapted as ``double precision[]``, which has no ``<#>`` operator and fails
    at query time instead of at write time.
    """
    norm = math.sqrt(sum(v * v for v in vector))
    values = list(vector) if norm == 0 else [v / norm for v in vector]
    return Vector(values)


# --- documents and versions -------------------------------------------------


def upsert_document(
    conn: psycopg.Connection,
    name: str,
    sha256: str | None,
    source_uri: str | None,
    identity_key: str,
) -> int:
    """Find or create one document by its internal identity.

    The conflict target is ``identity_key`` alone, never the display name: two
    reports whose output roots happen to share a basename are two documents,
    and removing one must not remove the other.
    """
    row = conn.execute(
        """
        INSERT INTO document (name, sha256, source_uri, identity_key)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (identity_key) DO UPDATE
            SET source_uri = COALESCE(EXCLUDED.source_uri, document.source_uri),
                sha256 = COALESCE(EXCLUDED.sha256, document.sha256)
        RETURNING id
        """,
        (name, sha256, source_uri, identity_key),
    ).fetchone()
    return int(row["id"])


def find_version(
    conn: psycopg.Connection, document_id: int, fingerprint: str
) -> dict[str, Any] | None:
    """The newest attempt at this fingerprint, ready ones first."""
    return conn.execute(
        """
        SELECT * FROM document_version
        WHERE document_id = %s AND fingerprint = %s
        ORDER BY (status = 'ready') DESC, attempt DESC
        LIMIT 1
        """,
        (document_id, fingerprint),
    ).fetchone()


def start_version(
    conn: psycopg.Connection,
    document_id: int,
    fingerprint: str,
    *,
    embed_model: str,
    embed_dim: int,
    output_root: str,
    source_pdf: str | None,
    force: bool = False,
) -> int:
    """Open a building version.

    Normally this resumes the existing attempt for the fingerprint. A forced
    rebuild opens the next attempt instead, so the version currently serving
    queries stays untouched until the replacement passes its checks.
    """
    attempt = 1
    if force:
        row = conn.execute(
            "SELECT max(attempt) AS a FROM document_version"
            " WHERE document_id = %s AND fingerprint = %s",
            (document_id, fingerprint),
        ).fetchone()
        attempt = int(row["a"] or 0) + 1
    else:
        row = conn.execute(
            "SELECT attempt FROM document_version"
            " WHERE document_id = %s AND fingerprint = %s AND status = 'building'"
            " ORDER BY attempt DESC LIMIT 1",
            (document_id, fingerprint),
        ).fetchone()
        if row:
            attempt = int(row["attempt"])

    row = conn.execute(
        """
        INSERT INTO document_version
            (document_id, fingerprint, embed_model, embed_dim, output_root,
             source_pdf, attempt)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (document_id, fingerprint, attempt) DO UPDATE
            SET status = 'building', ready_at = NULL
        RETURNING id
        """,
        (document_id, fingerprint, embed_model, embed_dim, output_root,
         source_pdf, attempt),
    ).fetchone()
    conn.commit()
    return int(row["id"])


def activate_version(conn: psycopg.Connection, version_id: int) -> None:
    """Flip one version to ready and retire the others in the same transaction."""
    with conn.transaction():
        conn.execute(
            """
            UPDATE document_version SET status = 'inactive'
            WHERE document_id = (SELECT document_id FROM document_version WHERE id = %s)
              AND id <> %s AND status = 'ready'
            """,
            (version_id, version_id),
        )
        conn.execute(
            "UPDATE document_version SET status = 'ready', ready_at = now() WHERE id = %s",
            (version_id,),
        )


def version_status(conn: psycopg.Connection, version_id: int) -> VersionStatus:
    row = conn.execute(
        "SELECT status FROM document_version WHERE id = %s", (version_id,)
    ).fetchone()
    return VersionStatus(row["status"])


def upsert_page(
    conn: psycopg.Connection,
    version_id: int,
    pdf_page: int,
    width: float,
    height: float,
    page_dir: str,
) -> int:
    row = conn.execute(
        """
        INSERT INTO page (version_id, pdf_page, width, height, page_dir)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (version_id, pdf_page) DO UPDATE SET page_dir = EXCLUDED.page_dir
        RETURNING id
        """,
        (version_id, pdf_page, width, height, page_dir),
    ).fetchone()
    return int(row["id"])


# --- evidence ---------------------------------------------------------------


# Reconciliation only ever touches the version currently being built. A ready
# version keeps serving queries while its replacement is assembled, and an
# inactive one is somebody's rollback target.
_ONLY_BUILDING = (
    " AND version_id IN"
    " (SELECT id FROM document_version WHERE id = %s AND status = 'building')"
)


def upsert_evidence(
    conn: psycopg.Connection,
    version_id: int,
    page_id: int,
    units: Iterable[EvidenceUnit],
) -> dict[str, int]:
    """Make this page's stored evidence match the source exactly; key -> id.

    Authoritative, not additive. Every source-derived column is overwritten on
    conflict, units the source no longer produces are deleted, and an embedding
    is dropped the moment its normalized text changes -- a resumed build that
    kept the old vector would pair new text with an embedding of the old.
    """
    ids: dict[str, int] = {}
    for unit in units:
        row = conn.execute(
            """
            INSERT INTO evidence_unit (
                version_id, page_id, unit_key, kind, quality, citable,
                source_text, normalized_text, heading_path, table_context,
                artifact_path, geometry_precision, truncated_source
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (version_id, unit_key) DO UPDATE SET
                page_id = EXCLUDED.page_id,
                kind = EXCLUDED.kind,
                quality = EXCLUDED.quality,
                citable = EXCLUDED.citable,
                source_text = EXCLUDED.source_text,
                normalized_text = EXCLUDED.normalized_text,
                heading_path = EXCLUDED.heading_path,
                table_context = EXCLUDED.table_context,
                artifact_path = EXCLUDED.artifact_path,
                geometry_precision = EXCLUDED.geometry_precision,
                truncated_source = EXCLUDED.truncated_source,
                embedding = CASE
                    WHEN evidence_unit.normalized_text
                         IS DISTINCT FROM EXCLUDED.normalized_text
                    THEN NULL ELSE evidence_unit.embedding END
            RETURNING id
            """,
            (
                version_id,
                page_id,
                unit.unit_key,
                str(unit.kind),
                str(unit.quality),
                unit.citable,
                unit.text,
                unit.normalized_text,
                Jsonb(unit.heading_path),
                Jsonb(unit.table_context),
                unit.artifact_path,
                str(unit.geometry_precision),
                unit.truncated_source,
            ),
        ).fetchone()
        evidence_id = int(row["id"])
        ids[unit.unit_key] = evidence_id
        conn.execute("DELETE FROM evidence_region WHERE evidence_id = %s", (evidence_id,))
        for ordinal, region in enumerate(unit.regions):
            conn.execute(
                """
                INSERT INTO evidence_region (
                    evidence_id, ordinal, role, precision,
                    left_norm, top_norm, right_norm, bottom_norm,
                    source_bbox, source_origin
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    evidence_id,
                    ordinal,
                    region.role,
                    str(region.precision),
                    *region.bbox,
                    Jsonb(list(region.source_bbox)) if region.source_bbox else None,
                    region.source_origin,
                ),
            )
    # `= ANY('{}')` is false for every row, so a page that now yields nothing
    # correctly loses all of its earlier units rather than keeping them.
    conn.execute(
        "DELETE FROM evidence_unit WHERE version_id = %s AND page_id = %s"
        f" AND NOT (unit_key = ANY(%s)){_ONLY_BUILDING}",
        (version_id, page_id, list(ids), version_id),
    )
    return ids


def delete_stale_pages(
    conn: psycopg.Connection, version_id: int, pdf_pages: Sequence[int]
) -> int:
    """Drop pages an earlier attempt indexed that the manifest no longer lists.

    Their evidence, regions, and fact links go with them through the cascade.
    """
    return conn.execute(
        "DELETE FROM page WHERE version_id = %s AND NOT (pdf_page = ANY(%s))"
        f"{_ONLY_BUILDING}",
        (version_id, list(pdf_pages), version_id),
    ).rowcount


def delete_facts(conn: psycopg.Connection, version_id: int) -> int:
    """Clear a building version's facts so they are rebuilt from current units.

    Fact construction reprocesses every unit anyway, so rebuilding is both
    cheaper and more exact than diffing: an updated qualifier, a dropped value,
    and a stale ``fact_evidence`` link all disappear for free via the cascade.
    """
    return conn.execute(
        f"DELETE FROM fact WHERE version_id = %s{_ONLY_BUILDING}",
        (version_id, version_id),
    ).rowcount


def units_missing_embeddings(
    conn: psycopg.Connection, version_id: int, kinds: Sequence[str]
) -> list[dict[str, Any]]:
    return conn.execute(
        """
        SELECT id, normalized_text FROM evidence_unit
        WHERE version_id = %s AND embedding IS NULL AND kind = ANY(%s)
        ORDER BY id
        """,
        (version_id, list(kinds)),
    ).fetchall()


def set_embeddings(
    conn: psycopg.Connection, rows: Sequence[tuple[int, Sequence[float]]]
) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE evidence_unit SET embedding = %s WHERE id = %s",
            [(normalize_embedding(vec), evidence_id) for evidence_id, vec in rows],
        )
    conn.commit()


# --- entities and facts -----------------------------------------------------


def upsert_entity(
    conn: psycopg.Connection, kind: str, canonical_name: str, normalized_name: str
) -> int:
    row = conn.execute(
        """
        INSERT INTO entity (kind, canonical_name, normalized_name)
        VALUES (%s, %s, %s)
        ON CONFLICT (kind, normalized_name) DO UPDATE
            SET canonical_name = entity.canonical_name
        RETURNING id
        """,
        (kind, canonical_name, normalized_name),
    ).fetchone()
    return int(row["id"])


def add_alias(conn: psycopg.Connection, entity_id: int, alias: str, normalized: str) -> None:
    conn.execute(
        """
        INSERT INTO entity_alias (entity_id, alias, normalized_alias)
        VALUES (%s, %s, %s) ON CONFLICT DO NOTHING
        """,
        (entity_id, alias, normalized),
    )


def upsert_fact(
    conn: psycopg.Connection,
    version_id: int,
    fact: Fact,
    normalized_metric: str,
    subject_entity_id: int | None,
    evidence_ids: Sequence[int],
) -> int:
    row = conn.execute(
        """
        INSERT INTO fact (
            version_id, fact_key, subject_entity_id, subject, metric, normalized_metric,
            value_decimal, value_text, unit, direction, comparison,
            reporting_period, baseline_period, scope, geography,
            qualifiers, extraction_method, quote
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (version_id, fact_key) DO UPDATE SET quote = EXCLUDED.quote
        RETURNING id
        """,
        (
            version_id,
            fact.fact_key,
            subject_entity_id,
            fact.subject,
            fact.metric,
            normalized_metric,
            fact.value_decimal,
            fact.value_text,
            fact.unit,
            fact.direction,
            fact.comparison,
            fact.reporting_period,
            fact.baseline_period,
            fact.scope,
            fact.geography,
            Jsonb(fact.qualifiers),
            fact.extraction_method,
            fact.quote,
        ),
    ).fetchone()
    fact_id = int(row["id"])
    for evidence_id in evidence_ids:
        conn.execute(
            "INSERT INTO fact_evidence (fact_id, evidence_id) VALUES (%s, %s)"
            " ON CONFLICT DO NOTHING",
            (fact_id, evidence_id),
        )
    return fact_id


# --- retrieval --------------------------------------------------------------

_EVIDENCE_COLUMNS = """
    e.id, e.unit_key, e.kind, e.quality, e.citable, e.source_text,
    e.heading_path, e.table_context, e.artifact_path, e.geometry_precision,
    p.pdf_page, p.printed_page_label, p.page_dir,
    d.id AS document_id, d.name AS document_name, d.sha256, d.source_uri
"""

_EVIDENCE_FROM = """
    FROM evidence_unit e
    JOIN page p ON p.id = e.page_id
    JOIN document_version v ON v.id = e.version_id
    JOIN document d ON d.id = v.document_id
"""

_EVIDENCE_SELECT = f"SELECT {_EVIDENCE_COLUMNS} {_EVIDENCE_FROM}"

_READY = "v.status = 'ready'"


def _document_filter(document_ids: Sequence[int] | None) -> tuple[str, list[Any]]:
    if not document_ids:
        return "", []
    return " AND d.id = ANY(%s)", [list(document_ids)]


def lexical_search(
    conn: psycopg.Connection,
    query: str,
    document_ids: Sequence[int] | None,
    limit: int,
) -> list[dict[str, Any]]:
    clause, params = _document_filter(document_ids)
    return conn.execute(
        f"""
        SELECT {_EVIDENCE_COLUMNS},
               ts_rank_cd(e.text_search, websearch_to_tsquery('english', %s))
                   AS lexical_score
        {_EVIDENCE_FROM}
        WHERE {_READY} AND e.text_search @@ websearch_to_tsquery('english', %s){clause}
        ORDER BY lexical_score DESC, e.id
        LIMIT %s
        """,
        [query, query, *params, limit],
    ).fetchall()


def vector_search(
    conn: psycopg.Connection,
    embedding: Sequence[float],
    document_ids: Sequence[int] | None,
    limit: int,
) -> list[dict[str, Any]]:
    clause, params = _document_filter(document_ids)
    vector = normalize_embedding(embedding)
    return conn.execute(
        f"""
        SELECT {_EVIDENCE_COLUMNS}, (e.embedding <#> %s) * -1 AS vector_score
        {_EVIDENCE_FROM}
        WHERE {_READY} AND e.embedding IS NOT NULL{clause}
        ORDER BY e.embedding <#> %s
        LIMIT %s
        """,
        [vector, *params, vector, limit],
    ).fetchall()


def exact_vector_search(
    conn: psycopg.Connection,
    embedding: Sequence[float],
    document_ids: Sequence[int] | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Same ranking with the HNSW index disabled, for recall measurement."""
    with conn.transaction():
        conn.execute("SET LOCAL enable_indexscan = off")
        conn.execute("SET LOCAL enable_bitmapscan = off")
        return vector_search(conn, embedding, document_ids, limit)


def graph_search(
    conn: psycopg.Connection,
    *,
    metric_terms: Sequence[str],
    reporting_period: str | None,
    baseline_period: str | None,
    document_ids: Sequence[int] | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Evidence reachable through facts that match the claim's qualifiers.

    Metric terms are OR'd, not AND'd. A claim carries words the source never
    uses ("Danone reduced ..."), so requiring every term matches nothing and
    quietly removes the graph leg from the fusion. The period filters do the
    narrowing; ts_rank orders what survives.
    """
    clause, params = _document_filter(document_ids)
    pattern = " | ".join(t for t in metric_terms if t) or None
    return conn.execute(
        f"""
        SELECT {_EVIDENCE_COLUMNS},
               max(ts_rank(to_tsvector('english', f.normalized_metric),
                           to_tsquery('english', COALESCE(%s, 'x')))) AS graph_score
        {_EVIDENCE_FROM}
        JOIN fact_evidence fe ON fe.evidence_id = e.id
        JOIN fact f ON f.id = fe.fact_id
        WHERE {_READY}{clause}
          AND (%s::text IS NULL
               OR to_tsvector('english', f.normalized_metric)
                  @@ to_tsquery('english', %s))
          AND (%s::text IS NULL OR f.reporting_period = %s)
          AND (%s::text IS NULL OR f.baseline_period IS NULL OR f.baseline_period = %s)
        GROUP BY e.id, p.pdf_page, p.printed_page_label, p.page_dir,
                 d.id, d.name, d.sha256, d.source_uri
        ORDER BY graph_score DESC, e.id
        LIMIT %s
        """,
        [
            pattern,
            *params,
            pattern,
            pattern,
            reporting_period,
            reporting_period,
            baseline_period,
            baseline_period,
            limit,
        ],
    ).fetchall()


def facts_for_evidence(
    conn: psycopg.Connection, evidence_ids: Sequence[int]
) -> list[dict[str, Any]]:
    if not evidence_ids:
        return []
    return conn.execute(
        """
        SELECT f.*, fe.evidence_id FROM fact f
        JOIN fact_evidence fe ON fe.fact_id = f.id
        WHERE fe.evidence_id = ANY(%s)
        """,
        (list(evidence_ids),),
    ).fetchall()


def regions_for(conn: psycopg.Connection, evidence_ids: Sequence[int]) -> dict[int, list[dict]]:
    if not evidence_ids:
        return {}
    rows = conn.execute(
        """
        SELECT * FROM evidence_region WHERE evidence_id = ANY(%s)
        ORDER BY evidence_id, ordinal
        """,
        (list(evidence_ids),),
    ).fetchall()
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["evidence_id"]), []).append(row)
    return grouped


def neighbours(
    conn: psycopg.Connection, evidence_id: int, radius: int = 1
) -> list[dict[str, Any]]:
    """Adjacent source blocks and sibling rows used to expand a candidate."""
    return conn.execute(
        f"""
        {_EVIDENCE_SELECT}
        WHERE {_READY} AND e.page_id = (SELECT page_id FROM evidence_unit WHERE id = %s)
          AND e.id <> %s AND e.citable
        ORDER BY abs(e.id - %s)
        LIMIT %s
        """,
        (evidence_id, evidence_id, evidence_id, radius * 4),
    ).fetchall()


def document_scope(
    conn: psycopg.Connection, document_ids: Sequence[int] | None
) -> list[dict[str, Any]]:
    """Classify a document selection in one round trip.

    With ``document_ids``, one row per *existing* requested document: an id
    that is absent from the result does not exist, and a row with a null
    ``document_version_id`` exists but has nothing ready to query. With
    ``None``, the ready versions of every document, which is the corpus an
    unscoped query actually searches.
    """
    version = """
        SELECT id, embed_model, embed_dim FROM document_version
        WHERE document_id = d.id AND status = 'ready'
        ORDER BY id DESC LIMIT 1
    """
    if document_ids is None:
        return conn.execute(
            f"""
            SELECT d.id AS document_id, d.name,
                   v.id AS document_version_id, v.embed_model, v.embed_dim
            FROM document d JOIN LATERAL ({version}) v ON true
            ORDER BY d.id
            """
        ).fetchall()
    return conn.execute(
        f"""
        SELECT d.id AS document_id, d.name,
               v.id AS document_version_id, v.embed_model, v.embed_dim
        FROM document d LEFT JOIN LATERAL ({version}) v ON true
        WHERE d.id = ANY(%s)
        ORDER BY d.id
        """,
        (list(document_ids),),
    ).fetchall()


def ready_documents(conn: psycopg.Connection) -> list[dict[str, Any]]:
    return conn.execute(
        """
        SELECT d.id, d.name, d.sha256, v.id AS version_id, v.ready_at
        FROM document d JOIN document_version v ON v.document_id = d.id
        WHERE v.status = 'ready' ORDER BY d.id
        """
    ).fetchall()


# --- audit trace ------------------------------------------------------------


def create_audit(
    conn: psycopg.Connection,
    claim: str,
    parsed: dict[str, Any],
    chat_model: str,
    embed_model: str,
) -> int:
    row = conn.execute(
        """
        INSERT INTO audit_run (claim, parsed_claim, chat_model, embed_model)
        VALUES (%s, %s, %s, %s) RETURNING id
        """,
        (claim, Jsonb(_jsonable(parsed)), chat_model, embed_model),
    ).fetchone()
    conn.commit()
    return int(row["id"])


def record_candidates(
    conn: psycopg.Connection, audit_id: int, candidates: Sequence[dict[str, Any]]
) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO audit_candidate (
                audit_id, evidence_id,
                lexical_rank, lexical_score, vector_rank, vector_score,
                graph_rank, graph_score, combined_rank, combined_score,
                expanded_from, visual_status, selected, reason
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (audit_id, evidence_id) DO UPDATE SET
                visual_status = EXCLUDED.visual_status,
                selected = EXCLUDED.selected,
                reason = EXCLUDED.reason
            """,
            [
                (
                    audit_id,
                    c["evidence_id"],
                    c.get("lexical_rank"),
                    c.get("lexical_score"),
                    c.get("vector_rank"),
                    c.get("vector_score"),
                    c.get("graph_rank"),
                    c.get("graph_score"),
                    c.get("combined_rank"),
                    c.get("combined_score", 0.0),
                    c.get("expanded_from"),
                    c.get("visual_status", "not_applicable"),
                    c.get("selected", False),
                    c.get("reason"),
                )
                for c in candidates
            ],
        )
    conn.commit()


def finish_audit(
    conn: psycopg.Connection,
    audit_id: int,
    *,
    verdict: str | None,
    rationale: str,
    quality: str,
    missing_qualifiers: Sequence[str],
    citations: Sequence[dict[str, Any]],
    error: str | None = None,
    decision_explanation: dict[str, Any] | None = None,
    timings: dict[str, Any] | None = None,
    index_references: Sequence[dict[str, Any]] = (),
) -> None:
    conn.execute(
        """
        UPDATE audit_run SET verdict = %s, rationale = %s, evidence_quality = %s,
            missing_qualifiers = %s, citations = %s, error = %s,
            decision_explanation = %s, timings = %s, index_references = %s
        WHERE id = %s
        """,
        (
            verdict,
            rationale,
            quality,
            Jsonb(list(missing_qualifiers)),
            Jsonb(_jsonable(list(citations))),
            error,
            Jsonb(_jsonable(decision_explanation or {})),
            Jsonb(_jsonable(timings or {})),
            Jsonb(_jsonable(list(index_references))),
            audit_id,
        ),
    )
    conn.commit()


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_default))


def _default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    return str(value)


__all__ = [
    "activate_version",
    "add_alias",
    "connect",
    "create_audit",
    "delete_facts",
    "delete_stale_pages",
    "document_scope",
    "exact_vector_search",
    "facts_for_evidence",
    "find_version",
    "finish_audit",
    "graph_search",
    "init_schema",
    "lexical_search",
    "neighbours",
    "normalize_embedding",
    "ready_documents",
    "record_candidates",
    "regions_for",
    "set_embeddings",
    "start_version",
    "units_missing_embeddings",
    "upsert_document",
    "upsert_entity",
    "upsert_evidence",
    "upsert_fact",
    "upsert_page",
    "vector_search",
    "version_status",
]


# --- frontend support -------------------------------------------------------


def schema_state(conn: psycopg.Connection) -> tuple[int | None, str | None]:
    """Applied schema version and the installed pgvector version."""
    try:
        row = conn.execute("SELECT version FROM schema_meta WHERE id = 1").fetchone()
        version = int(row["version"]) if row else None
    except psycopg.errors.UndefinedTable:
        conn.rollback()
        version = None
    extension = conn.execute(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    ).fetchone()
    return version, (extension["extversion"] if extension else None)


def index_counts(conn: psycopg.Connection) -> dict[str, int]:
    row = conn.execute(
        """
        SELECT
            (SELECT count(*) FROM document_version WHERE status = 'ready')    AS ready,
            (SELECT count(*) FROM document_version WHERE status = 'building') AS building,
            (SELECT count(*) FROM document_version WHERE status = 'inactive') AS inactive,
            (SELECT count(*) FROM evidence_unit)                              AS evidence,
            (SELECT count(*) FROM evidence_unit WHERE embedding IS NOT NULL)  AS embeddings,
            (SELECT count(*) FROM fact)                                       AS facts
        """
    ).fetchone()
    return {k: int(v) for k, v in row.items()}


_DOCUMENT_SUMMARY = """
    SELECT d.id AS document_id, v.id AS version_id, d.name, d.source_uri,
           d.sha256, v.status, v.embed_model, v.embed_dim, v.ready_at,
           v.output_root, v.source_pdf,
           (SELECT count(*) FROM page WHERE version_id = v.id) AS page_count,
           (SELECT count(*) FROM evidence_unit WHERE version_id = v.id) AS evidence_count,
           (SELECT count(*) FROM fact WHERE version_id = v.id) AS fact_count,
           (SELECT count(*) FROM evidence_unit
             WHERE version_id = v.id AND kind = 'visual') AS visual_count
    FROM document d
    JOIN document_version v ON v.document_id = d.id
"""


def document_summaries(
    conn: psycopg.Connection, document_id: int | None = None
) -> list[dict[str, Any]]:
    """One row per document: its ready version, else its newest attempt."""
    clause = " WHERE d.id = %s" if document_id is not None else ""
    params = [document_id] if document_id is not None else []
    return conn.execute(
        f"""
        SELECT DISTINCT ON (document_id) * FROM ({_DOCUMENT_SUMMARY}{clause}) s
        ORDER BY document_id, (status = 'ready') DESC, version_id DESC
        """,
        params,
    ).fetchall()


def delete_document(conn: psycopg.Connection, document_id: int) -> dict[str, int]:
    """Delete one document's index rows, returning per-table counts.

    Counted before the delete, because the foreign keys cascade and a cascaded
    row reports nothing back. Source PDFs, output directories, and page images
    are never touched -- re-ingestion is the whole recovery path.
    """
    with conn.transaction():
        counts = conn.execute(
            """
            SELECT
              (SELECT count(*) FROM document_version WHERE document_id = %(d)s)
                  AS document_versions,
              (SELECT count(*) FROM page p
                JOIN document_version v ON v.id = p.version_id
                WHERE v.document_id = %(d)s) AS pages,
              (SELECT count(*) FROM evidence_unit e
                JOIN document_version v ON v.id = e.version_id
                WHERE v.document_id = %(d)s) AS evidence_units,
              (SELECT count(*) FROM evidence_region r
                JOIN evidence_unit e ON e.id = r.evidence_id
                JOIN document_version v ON v.id = e.version_id
                WHERE v.document_id = %(d)s) AS evidence_regions,
              (SELECT count(*) FROM fact f
                JOIN document_version v ON v.id = f.version_id
                WHERE v.document_id = %(d)s) AS facts,
              (SELECT count(*) FROM fact_evidence fe
                JOIN fact f ON f.id = fe.fact_id
                JOIN document_version v ON v.id = f.version_id
                WHERE v.document_id = %(d)s) AS fact_evidence,
              (SELECT count(*) FROM audit_candidate ac
                JOIN evidence_unit e ON e.id = ac.evidence_id
                JOIN document_version v ON v.id = e.version_id
                WHERE v.document_id = %(d)s) AS audit_candidates
            """,
            {"d": document_id},
        ).fetchone()
        conn.execute("DELETE FROM document WHERE id = %s", (document_id,))
    return {"documents": 1, **{k: int(v) for k, v in counts.items()}}


def audit_run(conn: psycopg.Connection, audit_id: int) -> dict[str, Any] | None:
    return conn.execute("SELECT * FROM audit_run WHERE id = %s", (audit_id,)).fetchone()


def audit_candidates(conn: psycopg.Connection, audit_id: int) -> list[dict[str, Any]]:
    return conn.execute(
        """
        SELECT ac.*, e.kind, e.source_text, p.pdf_page
        FROM audit_candidate ac
        JOIN evidence_unit e ON e.id = ac.evidence_id
        JOIN page p ON p.id = e.page_id
        WHERE ac.audit_id = %s
        ORDER BY ac.combined_score DESC, ac.evidence_id
        """,
        (audit_id,),
    ).fetchall()


def evidence_row(conn: psycopg.Connection, evidence_id: int) -> dict[str, Any] | None:
    """One evidence unit with everything needed to render it, at any status."""
    return conn.execute(
        f"""
        SELECT {_EVIDENCE_COLUMNS}, e.version_id, e.truncated_source,
               p.width AS page_width, p.height AS page_height,
               v.output_root, v.status AS version_status
        {_EVIDENCE_FROM}
        WHERE e.id = %s
        """,
        (evidence_id,),
    ).fetchone()
