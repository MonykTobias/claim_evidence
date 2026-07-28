"""End-to-end checks against the Compose PostgreSQL, with a fake Ollama.

Skips cleanly when the database is unreachable, so the deterministic suite
still runs on a machine with no Docker.

    docker compose up -d
    python tests/test_integration.py
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import psycopg  # noqa: E402
from fake_ollama import FakeSession  # noqa: E402
from fixtures import block, image_summary, kpi_table, write_output_root  # noqa: E402

from claim_evidence import ClaimEvidence, Settings  # noqa: E402
from claim_evidence.db import (  # noqa: E402
    connect,
    exact_vector_search,
    normalize_embedding,
    vector_search,
)
from claim_evidence.models import (  # noqa: E402
    EvidenceKind,
    EvidenceQuality,
    GeometryPrecision,
    Verdict,
    VersionStatus,
)
from claim_evidence.ollama import OllamaClient  # noqa: E402

ADMIN_URL = os.environ.get(
    "CLAIM_EVIDENCE_DATABASE_URL",
    "postgresql://claim_evidence:claim_evidence@localhost:5433/claim_evidence",
)
TEST_DB = "claim_evidence_test"
DIMENSIONS = 8

SUPPORTED = "Danone reduced Scope 1 and 2 energy and industry emissions by 40.2% in 2025 versus 2020."
CONTRADICTED = SUPPORTED.replace("40.2%", "90%")
VAGUE = "Danone reduced all carbon emissions by 90% from 2020 to 2025."


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


# --- harness ----------------------------------------------------------------


def test_database_url() -> str:
    return ADMIN_URL.rsplit("/", 1)[0] + f"/{TEST_DB}"


def reset_database() -> None:
    with psycopg.connect(ADMIN_URL, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{TEST_DB}" WITH (FORCE)')
        conn.execute(f'CREATE DATABASE "{TEST_DB}"')


def settings() -> Settings:
    return Settings(
        database_url=test_database_url(),
        embed_dimensions=DIMENSIONS,
        embed_batch_size=16,
        chat_model="fake-chat",
        vision_model="fake-vision",
    )


def parsed_claim_reply(payload: dict[str, Any]) -> dict[str, Any]:
    """A deliberately thin parse, so the heuristic gap-fill is exercised."""
    claim = payload["messages"][1]["content"]
    scope = None
    if "Scope 1 and 2" in claim:
        scope = "Scope 1 and 2 energy and industry emissions"
    elif "all carbon" in claim:
        scope = "all carbon emissions"
    return {"subject": "Danone", "metric": claim, "scope": scope, "direction": "decrease"}


def adjudication_reply(_: dict[str, Any]) -> dict[str, Any]:
    return {
        "verdict": "insufficient",
        "rationale": "The evidence does not state a comparable scope.",
        "supporting_evidence_ids": [],
        "missing_qualifiers": ["scope"],
    }


def build_root(root: Path, *, with_visual: bool = False) -> Path:
    blocks = [
        block(1, 1, "4.8.2 NATURE INDICATORS", kind="section_header",
              heading_path=["4.8.2 NATURE INDICATORS"]),
        block(1, 2, "The table below reports performance against the 2020 baseline.",
              heading_path=["4.8.2 NATURE INDICATORS"]),
        block(2, 1, "Water withdrawal intensity improved during the year.",
              heading_path=["4.8.3 WATER"]),
    ]
    return write_output_root(
        root,
        pages=2,
        blocks=blocks,
        tables={1: [kpi_table()]},
        images={2: [image_summary(2, 1, "Chart of emissions reduction")]} if with_visual else None,
    )


def make_client(session: FakeSession, conn: psycopg.Connection | None = None) -> ClaimEvidence:
    config = settings()
    return ClaimEvidence(
        config,
        conn or connect(config.database_url),
        OllamaClient(config, session),
    )


def default_session(**router: Any) -> FakeSession:
    return FakeSession(
        dimensions=DIMENSIONS,
        chat_router={
            "ParsedClaim": parsed_claim_reply,
            "FactExtraction": lambda _: {"facts": []},
            "Adjudication": adjudication_reply,
            "VisualVerification": lambda _: {"supports_claim": False, "reason": "unreadable"},
            **router,
        },
    )


# --- checks -----------------------------------------------------------------


def check_schema_is_repeatable(tmp: Path) -> None:
    with make_client(default_session()) as client:
        client.init_db()
        client.init_db()
        tables = client.conn.execute(
            "SELECT count(*) AS n FROM pg_tables WHERE schemaname = 'public'"
        ).fetchone()["n"]
        check(tables == 11, f"all 11 tables present after a repeated init ({tables})")


def check_ingestion_is_idempotent(tmp: Path) -> None:
    root = build_root(tmp / "run")
    with make_client(default_session()) as client:
        first = client.ingest_document(root, source_pdf=None, source_uri="urn:test")
        check(first.status is VersionStatus.READY, "version reaches ready")
        check(first.pages == 2, "both pages indexed")
        check(first.evidence_units > 0, "evidence units stored")
        check(first.embedded_units > 0, "embeddings written")
        check(first.facts >= 4, f"table facts derived ({first.facts})")

        second = client.ingest_document(root, source_pdf=None, source_uri="urn:test")
        check(second.reused_existing, "re-ingesting the same fingerprint is a no-op")
        check(second.version_id == first.version_id, "the same version is reused")

        units = client.conn.execute(
            "SELECT count(*) AS n FROM evidence_unit WHERE version_id = %s",
            (first.version_id,),
        ).fetchone()["n"]
        check(units == first.evidence_units, "no duplicate rows after a second run")


def check_interrupted_build_is_invisible(tmp: Path) -> None:
    root = build_root(tmp / "interrupted")
    config = settings()
    conn = connect(config.database_url)
    with make_client(default_session(), conn) as client:
        # Simulate a crash between "building" and activation.
        from claim_evidence.db import start_version, upsert_document, upsert_page, upsert_evidence
        from claim_evidence.source import OutputReader, page_units

        document_id = upsert_document(conn, "interrupted.pdf", "deadbeef", None)
        version_id = start_version(
            conn, document_id, "never-finished", embed_model="fake",
            embed_dim=DIMENSIONS, output_root=str(root), source_pdf=None,
        )
        reader = OutputReader(root)
        page = reader.validate()[0]
        page_id = upsert_page(conn, version_id, 1, page.width, page.height, page.page_dir.name)
        upsert_evidence(
            conn, version_id, page_id,
            list(page_units(reader, page, reader.blocks_by_page().get(1, []))),
        )
        conn.commit()

        status = conn.execute(
            "SELECT status FROM document_version WHERE id = %s", (version_id,)
        ).fetchone()["status"]
        check(status == "building", "interrupted version stays in building")

        matches = client.search_evidence("Scope 1 and 2 emissions", limit=50)
        leaked = [m for m in matches if m.citation.document_name == "interrupted.pdf"]
        check(not leaked, "a building version never appears in query results")


def check_indexes_are_used(tmp: Path) -> None:
    with make_client(default_session()) as client:
        conn = client.conn
        conn.execute("SET enable_seqscan = off")
        fts = "\n".join(
            str(r)
            for r in conn.execute(
                """
                EXPLAIN SELECT id FROM evidence_unit
                WHERE text_search @@ websearch_to_tsquery('english', 'emissions')
                """
            ).fetchall()
        )
        check("evidence_unit_fts_idx" in fts, "GIN full-text index is used")

        vector = "\n".join(
            str(r)
            for r in conn.execute(
                "EXPLAIN SELECT id FROM evidence_unit WHERE embedding IS NOT NULL "
                "ORDER BY embedding <#> %s LIMIT 10",
                (normalize_embedding([1.0] * DIMENSIONS),),
            ).fetchall()
        )
        check("evidence_unit_embedding_idx" in vector, "HNSW index is used for ordering")
        conn.execute("SET enable_seqscan = on")


def check_hybrid_search_finds_exact_and_semantic(tmp: Path) -> None:
    with make_client(default_session()) as client:
        matches = client.search_evidence(SUPPORTED, limit=25)
        check(bool(matches), "hybrid search returns candidates")
        texts = [m.text for m in matches]
        check(any("(40.2) %" in t for t in texts), "the exact value is retrieved")
        check(
            any(m.lexical_rank is not None for m in matches)
            and any(m.vector_rank is not None for m in matches),
            "both lexical and vector retrievers contribute",
        )
        top = matches[0].citation
        check(top.pdf_page in (1, 2), "citation carries a 1-based pdf page")
        check(bool(top.regions), "every citation carries at least one region")


def check_vector_recall(tmp: Path) -> None:
    config = settings()
    with make_client(default_session()) as client:
        conn = client.conn
        session = FakeSession(dimensions=DIMENSIONS)
        queries = [
            "Scope 1 and 2 emissions versus 2020",
            "renewable electricity in the energy mix",
            "water withdrawal intensity",
            "performance history 2025",
            "nature indicators table",
        ]
        hits = 0
        total = 0
        for query in queries:
            embedding = session.vector(query)
            approximate = [r["id"] for r in vector_search(conn, embedding, None, 10)]
            exact = [r["id"] for r in exact_vector_search(conn, embedding, None, 10)]
            if not exact:
                continue
            hits += len(set(approximate) & set(exact))
            total += len(exact)
        recall = hits / total if total else 1.0
        check(recall >= 0.9, f"approximate top-10 recall is {recall:.0%} (>= 90%)")
        del config


def check_supported_claim_cites_the_table(tmp: Path) -> None:
    with make_client(default_session()) as client:
        result = client.audit_claim(SUPPORTED)
        check(result.verdict is Verdict.SUPPORTED, f"exact claim supported ({result.rationale})")
        check(
            result.evidence_quality is EvidenceQuality.DIRECT_TABLE,
            "supported by direct table evidence",
        )
        check(bool(result.citations), "verdict carries citations")
        cite = result.citations[0]
        check(cite.source_kind is EvidenceKind.TABLE_VALUE, "the cited unit is a table value")
        check(cite.geometry_precision is GeometryPrecision.CELL, "cell-precision geometry")
        roles = {r.role for r in cite.regions}
        check(
            {"descriptor", "header", "unit", "value"} <= roles,
            f"descriptor, header, unit, and value regions cited ({roles})",
        )
        check("(40.2) %" in " ".join(cite.table_cells), "the displayed value is quoted")
        check(result.audit_id is not None, "audit persisted")

        candidates = client.conn.execute(
            "SELECT count(*) AS n FROM audit_candidate WHERE audit_id = %s",
            (result.audit_id,),
        ).fetchone()["n"]
        check(candidates > 0, "retrieval candidates persisted for the audit trace")
        selected = client.conn.execute(
            "SELECT count(*) AS n FROM audit_candidate WHERE audit_id = %s AND selected",
            (result.audit_id,),
        ).fetchone()["n"]
        check(selected > 0, "selected citations flagged in the trace")


def check_contradicted_claim(tmp: Path) -> None:
    with make_client(default_session()) as client:
        result = client.audit_claim(CONTRADICTED)
        check(result.verdict is Verdict.CONTRADICTED, f"90% contradicted ({result.rationale})")
        check("40.2" in result.rationale, "rationale names the reported value")
        check(bool(result.citations), "contradiction is cited")


def check_vague_claim_is_insufficient(tmp: Path) -> None:
    with make_client(default_session()) as client:
        result = client.audit_claim(VAGUE)
        check(
            result.verdict is Verdict.INSUFFICIENT,
            f"vague scope is not forced into a contradiction ({result.verdict})",
        )
        check(bool(result.missing_qualifiers), "missing qualifiers reported")


def check_verdict_fails_closed_without_citable_evidence(tmp: Path) -> None:
    """An adjudicator that claims support without naming a passage is downgraded."""
    session = default_session(
        Adjudication=lambda _: {
            "verdict": "supported",
            "rationale": "It looks right to me.",
            "supporting_evidence_ids": [],
            "missing_qualifiers": [],
        }
    )
    with make_client(session) as client:
        result = client.audit_claim("Danone employs many people across its operations.")
        check(
            result.verdict is not Verdict.SUPPORTED,
            f"uncited support is refused ({result.verdict})",
        )
        check(not result.citations, "no citations invented")


def check_visual_evidence_needs_crop_verification(tmp: Path) -> None:
    root = build_root(tmp / "visual", with_visual=True)
    claim = "The chart shows emissions falling by 40.2%."

    refusing = default_session()
    with make_client(refusing) as client:
        client.ingest_document(root, source_uri="urn:visual")
        result = client.audit_claim(claim)
        check(
            result.evidence_quality is not EvidenceQuality.VERIFIED_VISUAL,
            "an unverified crop cannot become verified visual evidence",
        )
        check(
            all(c.source_kind is not EvidenceKind.VISUAL for c in result.citations),
            "a failed crop check drops the visual candidate entirely",
        )

    accepting = default_session(
        VisualVerification=lambda _: {
            "supports_claim": True,
            "visible_text": "-40.2% vs 2020",
            "reason": "the figure is legible",
        },
        Adjudication=lambda payload: {
            "verdict": "supported",
            "rationale": "The verified crop shows the figure.",
            "supporting_evidence_ids": _visual_ids(payload),
            "missing_qualifiers": [],
        },
    )
    with make_client(accepting) as client:
        result = client.audit_claim(claim)
        visual = [c for c in result.citations if c.source_kind is EvidenceKind.VISUAL]
        if visual:
            check(
                visual[0].quality is EvidenceQuality.VERIFIED_VISUAL,
                "a verified crop is labelled verified_visual",
            )
        else:
            check(
                result.verdict is not Verdict.SUPPORTED
                or result.evidence_quality is not EvidenceQuality.COARSE_REGION,
                "no verdict rests on a coarse region",
            )


def _visual_ids(payload: dict[str, Any]) -> list[int]:
    import re

    prompt = payload["messages"][1]["content"]
    return [int(m) for m in re.findall(r"\[(\d+)\] page \d+ \(visual", prompt)]


def main() -> int:
    try:
        reset_database()
    except psycopg.OperationalError as exc:
        print(f"[skip] postgres unavailable ({exc.__class__.__name__}); run: docker compose up -d")
        return 0

    import tempfile

    random.seed(0)
    checks = [
        check_schema_is_repeatable,
        check_ingestion_is_idempotent,
        check_indexes_are_used,
        check_hybrid_search_finds_exact_and_semantic,
        check_vector_recall,
        check_supported_claim_cites_the_table,
        check_contradicted_claim,
        check_vague_claim_is_insufficient,
        check_verdict_fails_closed_without_citable_evidence,
        check_interrupted_build_is_invisible,
        check_visual_evidence_needs_crop_verification,
    ]
    with tempfile.TemporaryDirectory() as temp:
        tmp = Path(temp)
        for func in checks:
            print(f"\n-- {func.__name__}")
            func(tmp)
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
