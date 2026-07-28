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
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .models import EvidenceUnit, Fact, VersionStatus

SQL_DIR = Path(__file__).parent / "sql"
EMBED_DIM_TOKEN = "EMBED_DIM"


def connect(database_url: str) -> psycopg.Connection:
    conn = psycopg.connect(database_url, row_factory=dict_row, autocommit=False)
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


def normalize_embedding(vector: Sequence[float]) -> list[float]:
    """L2-normalize so the HNSW inner-product index ranks like cosine."""
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return list(vector)
    return [v / norm for v in vector]


# --- documents and versions -------------------------------------------------


def upsert_document(
    conn: psycopg.Connection, name: str, sha256: str | None, source_uri: str | None
) -> int:
    row = conn.execute(
        """
        INSERT INTO document (name, sha256, source_uri)
        VALUES (%s, %s, %s)
        ON CONFLICT (name, sha256) DO UPDATE
            SET source_uri = COALESCE(EXCLUDED.source_uri, document.source_uri)
        RETURNING id
        """,
        (name, sha256, source_uri),
    ).fetchone()
    return int(row["id"])


def find_version(
    conn: psycopg.Connection, document_id: int, fingerprint: str
) -> dict[str, Any] | None:
    return conn.execute(
        "SELECT * FROM document_version WHERE document_id = %s AND fingerprint = %s",
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
) -> int:
    row = conn.execute(
        """
        INSERT INTO document_version
            (document_id, fingerprint, embed_model, embed_dim, output_root, source_pdf)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (document_id, fingerprint) DO UPDATE
            SET status = 'building', ready_at = NULL
        RETURNING id
        """,
        (document_id, fingerprint, embed_model, embed_dim, output_root, source_pdf),
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


def upsert_evidence(
    conn: psycopg.Connection,
    version_id: int,
    page_id: int,
    units: Iterable[EvidenceUnit],
) -> dict[str, int]:
    """Insert or refresh evidence units and their regions; returns key -> id."""
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
                source_text = EXCLUDED.source_text,
                normalized_text = EXCLUDED.normalized_text,
                quality = EXCLUDED.quality,
                citable = EXCLUDED.citable,
                heading_path = EXCLUDED.heading_path,
                table_context = EXCLUDED.table_context,
                geometry_precision = EXCLUDED.geometry_precision,
                truncated_source = EXCLUDED.truncated_source
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
    return ids


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

_EVIDENCE_SELECT = """
    SELECT e.id, e.unit_key, e.kind, e.quality, e.citable, e.source_text,
           e.heading_path, e.table_context, e.artifact_path, e.geometry_precision,
           p.pdf_page, p.printed_page_label, p.page_dir,
           d.id AS document_id, d.name AS document_name, d.sha256, d.source_uri
    FROM evidence_unit e
    JOIN page p ON p.id = e.page_id
    JOIN document_version v ON v.id = e.version_id
    JOIN document d ON d.id = v.document_id
"""

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
        {_EVIDENCE_SELECT}
        WHERE {_READY} AND e.text_search @@ websearch_to_tsquery('english', %s){clause}
        ORDER BY ts_rank_cd(e.text_search, websearch_to_tsquery('english', %s)) DESC, e.id
        LIMIT %s
        """,
        [query, *params, query, limit],
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
        {_EVIDENCE_SELECT}
        WHERE {_READY} AND e.embedding IS NOT NULL{clause}
        ORDER BY e.embedding <#> %s
        LIMIT %s
        """,
        [*params, vector, limit],
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
    """Evidence reachable through facts that match the claim's qualifiers."""
    clause, params = _document_filter(document_ids)
    pattern = " & ".join(t for t in metric_terms if t) or None
    return conn.execute(
        f"""
        {_EVIDENCE_SELECT}
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
        ORDER BY e.id
        LIMIT %s
        """,
        [
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
                audit_id, evidence_id, lexical_rank, vector_rank, graph_rank,
                combined_score, selected, note
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (audit_id, evidence_id) DO UPDATE SET
                selected = EXCLUDED.selected, note = EXCLUDED.note
            """,
            [
                (
                    audit_id,
                    c["evidence_id"],
                    c.get("lexical_rank"),
                    c.get("vector_rank"),
                    c.get("graph_rank"),
                    c.get("combined_score", 0.0),
                    c.get("selected", False),
                    c.get("note"),
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
) -> None:
    conn.execute(
        """
        UPDATE audit_run SET verdict = %s, rationale = %s, evidence_quality = %s,
            missing_qualifiers = %s, citations = %s, error = %s
        WHERE id = %s
        """,
        (
            verdict,
            rationale,
            quality,
            Jsonb(list(missing_qualifiers)),
            Jsonb(_jsonable(list(citations))),
            error,
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
