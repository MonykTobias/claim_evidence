"""End-to-end checks against the Compose PostgreSQL, with a fake Ollama.

Skips cleanly when the database is unreachable, so the deterministic suite
still runs on a machine with no Docker.

    docker compose up -d
    python tests/test_integration.py
"""

from __future__ import annotations

import json
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
    SCHEMA_VERSION,
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
from claim_evidence.audit import MAX_PASSAGES, PASSAGE_CHARS  # noqa: E402
from claim_evidence.errors import (  # noqa: E402
    IndexNotReadyError,
    NotFoundError,
    ValidationError,
)
from claim_evidence.ingest import identity_key  # noqa: E402
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
        tables = {
            row["tablename"]
            for row in client.conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            ).fetchall()
        }
        expected = {
            "document", "document_version", "page", "evidence_unit", "evidence_region",
            "entity", "entity_alias", "fact", "fact_evidence", "audit_run",
            "audit_candidate",
        }
        check(expected <= tables, f"all 11 domain tables present ({sorted(expected - tables)})")
        version = client.conn.execute("SELECT version FROM schema_meta").fetchone()
        check(version["version"] == SCHEMA_VERSION, "schema version recorded")


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
        check(
            second.evidence_units == first.evidence_units and second.facts == first.facts,
            "a no-op run reports the stored counts, not zeros",
        )

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
        from claim_evidence.ingest import identity_key
        from claim_evidence.source import OutputReader, page_units

        document_id = upsert_document(
            conn, "interrupted.pdf", "deadbeef", None, identity_key(root, "deadbeef", None)
        )
        version_id = start_version(
            conn, document_id, "never-finished", embed_model="fake",
            embed_dim=DIMENSIONS, output_root=str(root), source_pdf=None,
        )
        reader = OutputReader(root)
        page = reader.validate()[0]
        page_id = upsert_page(conn, version_id, 1, page.width, page.height, page.rel)
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
        # A claim says "Danone reduced ..."; the table never uses those words.
        # AND'ing the metric terms silently removed this leg from the fusion.
        check(
            any(m.graph_rank is not None for m in matches),
            "the graph leg contributes on a claim worded unlike the source",
        )


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


def check_adjudicator_prompt_is_bounded(tmp: Path) -> None:
    """An over-long prompt is truncated by the runtime with no signal, so the
    passage list is capped before it is sent."""
    seen: list[str] = []

    def capture(payload: dict[str, Any]) -> dict[str, Any]:
        seen.append(payload["messages"][1]["content"])
        return adjudication_reply(payload)

    with make_client(default_session(Adjudication=capture)) as client:
        client.audit_claim(VAGUE)
    check(bool(seen), "the adjudicator was consulted")
    passages = seen[0].split("Evidence:\n", 1)[1].strip().split("\n\n")
    check(len(passages) <= MAX_PASSAGES, f"passage count capped ({len(passages)})")
    longest = max(len(p) for p in passages)
    check(longest <= PASSAGE_CHARS + 200, f"each passage is truncated ({longest} chars)")


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


# --- H-1: a selected page range keeps its original PDF page numbers ---------


def check_middle_range_keeps_pdf_pages(tmp: Path) -> None:
    root = write_output_root(
        tmp / "middle",
        page_numbers=[10, 11],
        blocks=[
            block(10, 1, "4.8.2 NATURE INDICATORS", kind="section_header",
                  heading_path=["4.8.2 NATURE INDICATORS"]),
            block(11, 1, "Water withdrawal intensity improved during the year.",
                  heading_path=["4.8.3 WATER"]),
        ],
        tables={10: [kpi_table()]},
    )
    with make_client(default_session()) as client:
        report = client.ingest_document(root, source_uri="urn:middle")
        check(report.status is VersionStatus.READY, "a range starting after page 1 indexes")

        matches = client.search_evidence(SUPPORTED, document_ids=[report.document_id])
        check(bool(matches), "the middle-range document is searchable")
        pages = {m.citation.pdf_page for m in matches}
        check(pages <= {10, 11}, f"search citations keep the pdf page numbers ({pages})")

        result = client.audit_claim(SUPPORTED, document_ids=[report.document_id])
        check(result.verdict is Verdict.SUPPORTED, f"claim supported ({result.rationale})")
        check(
            all(c.pdf_page in (10, 11) for c in result.citations),
            f"audit citations keep the pdf page numbers "
            f"({[c.pdf_page for c in result.citations]})",
        )
        detail = client.get_evidence(result.citations[0].evidence_id)
        check(detail.pdf_page in (10, 11), "get_evidence reports the pdf page")
        check(detail.page_image_path is not None, "the page image still resolves")


# --- H-3: document identity -------------------------------------------------


def check_source_less_documents_stay_separate(tmp: Path) -> None:
    """Two unrelated roots with the same basename, no PDF, and no URI."""
    first = build_root(tmp / "left" / "report")
    second = build_root(tmp / "right" / "report")
    with make_client(default_session()) as client:
        a = client.ingest_document(first)
        b = client.ingest_document(second)
        check(a.document_id != b.document_id, "same basename, different documents")

        ready = {int(r["id"]) for r in client.documents()}
        check({a.document_id, b.document_id} <= ready, "both are ready at once")
        for report in (a, b):
            hits = client.search_evidence(SUPPORTED, document_ids=[report.document_id])
            check(bool(hits), f"document {report.document_id} is queryable on its own")

        again = client.ingest_document(second)
        check(
            again.document_id == b.document_id and again.version_id == b.version_id,
            "re-ingesting the same root reuses its document and ready version",
        )
        check(again.reused_existing, "and is still a no-op")

        client.remove_document(a.document_id, confirm_document_id=a.document_id)
        survivors = {int(r["id"]) for r in client.documents()}
        check(a.document_id not in survivors, "the removed document is gone")
        check(b.document_id in survivors, "removing one leaves the other ready")
        check(
            bool(client.search_evidence(SUPPORTED, document_ids=[b.document_id])),
            "the surviving document still returns evidence",
        )


def check_shared_source_uri_is_one_document(tmp: Path) -> None:
    copies = [build_root(tmp / "copy_a" / "run"), build_root(tmp / "copy_b" / "run")]
    with make_client(default_session()) as client:
        ids = {client.ingest_document(root, source_uri="urn:same").document_id
               for root in copies}
        check(len(ids) == 1, "an explicit logical source_uri makes them one document")


def check_schema_migrates_a_legacy_document(tmp: Path) -> None:
    """A pre-identity row keeps its id and gets a deterministic key."""
    with make_client(default_session()) as client:
        conn = client.conn
        conn.execute("DROP INDEX IF EXISTS document_identity_idx")
        conn.execute("ALTER TABLE document DROP COLUMN IF EXISTS identity_key")
        legacy = int(
            conn.execute(
                "INSERT INTO document (name, sha256, source_uri)"
                " VALUES ('legacy.pdf', 'cafe1234', NULL) RETURNING id"
            ).fetchone()["id"]
        )
        conn.commit()

        client.init_db()
        client.init_db()

        row = conn.execute(
            "SELECT id, identity_key FROM document WHERE id = %s", (legacy,)
        ).fetchone()
        check(int(row["id"]) == legacy, "migration does not rewrite document ids")
        check(
            row["identity_key"] == identity_key(tmp, "cafe1234", None),
            "a stored pdf hash backfills to the same key ingestion would compute",
        )
        nulls = conn.execute(
            "SELECT count(*) AS n FROM document WHERE identity_key IS NULL"
        ).fetchone()["n"]
        check(nulls == 0, "every document row is backfilled")
        conn.execute("DELETE FROM document WHERE id = %s", (legacy,))
        conn.commit()


# --- H-4: an interrupted attempt is reconciled, not merged into ------------


def check_resume_reconciles_the_building_version(tmp: Path) -> None:
    from claim_evidence.db import (
        set_embeddings,
        start_version,
        upsert_document,
        upsert_evidence,
        upsert_fact,
        upsert_page,
    )
    from claim_evidence.ingest import identity_key, index_fingerprint
    from claim_evidence.models import Fact
    from claim_evidence.source import OutputReader, page_units

    root = build_root(tmp / "resume")
    config = settings()
    reader = OutputReader(root)
    pages = reader.validate()
    page = pages[0]
    units = list(page_units(reader, page, reader.blocks_by_page().get(1, [])))
    changed, unchanged = units[0], units[1]
    marker = [1.0] + [0.0] * (DIMENSIONS - 1)

    embedded_texts: list[str] = []
    session = default_session()
    session.embed_hook = lambda texts: embedded_texts.extend(texts) and None

    conn = connect(config.database_url)
    with make_client(session, conn) as client:
        # Seed an attempt that died before activation, holding stale text, a
        # unit the source no longer produces, a page that is gone, and a fact
        # built from all of it.
        fingerprint = index_fingerprint(reader.fingerprint(None), config)
        document_id = upsert_document(
            conn, reader.root.name, None, "urn:resume",
            identity_key(reader.root, None, "urn:resume"),
        )
        version_id = start_version(
            conn, document_id, fingerprint, embed_model=config.embed_model,
            embed_dim=DIMENSIONS, output_root=str(reader.root), source_pdf=None,
        )
        stale = changed.model_copy(
            update={"text": "OBSOLETE TEXT", "normalized_text": "obsolete text"}
        )
        ghost = changed.model_copy(
            update={
                "unit_key": "p0001:narrative:ghost",
                "text": "A paragraph the source no longer contains.",
                "normalized_text": "a paragraph the source no longer contains",
            }
        )
        page_id = upsert_page(
            conn, version_id, page.page, page.width, page.height, page.rel
        )
        seeded = upsert_evidence(conn, version_id, page_id, [stale, unchanged, ghost])
        gone_page = upsert_page(conn, version_id, 99, page.width, page.height, "page_0099")
        upsert_evidence(
            conn, version_id, gone_page,
            [changed.model_copy(update={"unit_key": "p0099:narrative:1", "page": 99})],
        )
        set_embeddings(
            conn,
            [
                (seeded[stale.unit_key], session.vector("obsolete text")),
                (seeded[unchanged.unit_key], marker),
                (seeded[ghost.unit_key], session.vector(ghost.normalized_text)),
            ],
        )
        upsert_fact(
            conn, version_id,
            Fact(subject="Danone", metric="obsolete metric", value_text="obsolete",
                 quote="OBSOLETE TEXT", evidence_keys=[stale.unit_key]),
            "obsolete metric", None, [seeded[stale.unit_key]],
        )
        conn.commit()
        stale_fact_id = conn.execute(
            "SELECT id FROM fact WHERE version_id = %s", (version_id,)
        ).fetchone()["id"]

        embedded_texts.clear()
        report = client.ingest_document(root, source_uri="urn:resume")
        check(report.version_id == version_id, "the interrupted attempt is resumed")
        check(report.status is VersionStatus.READY, "it activates once checks pass")

        row = conn.execute(
            "SELECT id, source_text, embedding FROM evidence_unit"
            " WHERE version_id = %s AND unit_key = %s",
            (version_id, changed.unit_key),
        ).fetchone()
        check(row["source_text"] == changed.text, "changed text is overwritten")
        check(
            changed.normalized_text in embedded_texts,
            "the changed unit was re-embedded",
        )
        distance = conn.execute(
            "SELECT embedding <#> %s AS d FROM evidence_unit WHERE id = %s",
            (normalize_embedding(session.vector(changed.normalized_text)), row["id"]),
        ).fetchone()["d"]
        check(
            abs(float(distance) + 1.0) < 1e-6,
            f"its stored embedding represents the new text ({distance})",
        )

        kept = conn.execute(
            "SELECT embedding <#> %s AS d FROM evidence_unit"
            " WHERE version_id = %s AND unit_key = %s",
            (normalize_embedding(marker), version_id, unchanged.unit_key),
        ).fetchone()["d"]
        check(
            abs(float(kept) + 1.0) < 1e-6, "an unchanged unit keeps its embedding"
        )
        check(
            unchanged.normalized_text not in embedded_texts,
            "and was not re-embedded",
        )

        leftovers = conn.execute(
            "SELECT count(*) AS n FROM evidence_unit"
            " WHERE version_id = %s AND unit_key = ANY(%s)",
            (version_id, [ghost.unit_key, "p0099:narrative:1"]),
        ).fetchone()["n"]
        check(leftovers == 0, "removed units do not survive the resume")
        stale_pages = conn.execute(
            "SELECT count(*) AS n FROM page WHERE version_id = %s AND pdf_page = 99",
            (version_id,),
        ).fetchone()["n"]
        check(stale_pages == 0, "a page the manifest no longer lists is deleted")

        old_fact = conn.execute(
            "SELECT count(*) AS n FROM fact WHERE id = %s", (stale_fact_id,)
        ).fetchone()["n"]
        check(old_fact == 0, "facts from the earlier attempt are rebuilt, not kept")
        links = conn.execute(
            "SELECT count(*) AS n FROM fact_evidence fe"
            " LEFT JOIN fact f ON f.id = fe.fact_id WHERE f.id IS NULL"
        ).fetchone()["n"]
        check(links == 0, "no orphaned fact_evidence links remain")
        quotes = conn.execute(
            "SELECT count(*) AS n FROM fact WHERE version_id = %s AND quote = %s",
            (version_id, "OBSOLETE TEXT"),
        ).fetchone()["n"]
        check(quotes == 0, "no rebuilt fact carries the obsolete quote")


# --- H-5: an invalid scope is an error, never a verdict ---------------------


def _expect(error: type[Exception], call, message: str) -> None:
    try:
        call()
    except error as exc:
        check(True, f"{message} ({type(exc).__name__}: {exc})")
        return
    except Exception as exc:  # noqa: BLE001 - the wrong type is the failure
        raise AssertionError(f"{message}: raised {type(exc).__name__} instead") from exc
    raise AssertionError(f"{message}: nothing was raised")


def check_document_scope_is_validated(tmp: Path) -> None:
    root = build_root(tmp / "scoped")
    with make_client(default_session()) as client:
        indexed = client.ingest_document(root, source_uri="urn:scoped")
        ready = indexed.document_id

        check(bool(client.search_evidence(SUPPORTED)), "None searches every ready document")
        check(
            bool(client.search_evidence(SUPPORTED, document_ids=[])),
            "an empty list searches every ready document",
        )
        scoped = client.search_evidence(SUPPORTED, document_ids=[ready])
        check(
            bool(scoped) and {m.citation.document_id for m in scoped} == {ready},
            "an explicit document restricts the results to itself",
        )
        check(
            len(client.search_evidence(SUPPORTED, document_ids=[ready, ready])) == len(scoped),
            "a duplicated id is harmless",
        )

        unknown = 10_000_000
        for call, label in (
            (lambda: client.search_evidence(SUPPORTED, document_ids=[unknown]), "search"),
            (lambda: client.audit_claim(SUPPORTED, document_ids=[unknown]), "audit"),
        ):
            _expect(NotFoundError, call, f"{label}: an unknown document is not found")
        _expect(
            NotFoundError,
            lambda: client.audit_claim(SUPPORTED, document_ids=[ready, unknown]),
            "a mixed selection fails as a whole rather than searching the valid part",
        )
        for value in (True, "abc", 1.5):
            _expect(
                ValidationError,
                lambda v=value: client.search_evidence(SUPPORTED, document_ids=[v]),
                f"{value!r} is a validation error",
            )

        # A document whose only version is still building has nothing to query.
        building = client.conn.execute(
            "INSERT INTO document (name, identity_key) VALUES ('halfbuilt', 'k-halfbuilt')"
            " RETURNING id"
        ).fetchone()["id"]
        client.conn.execute(
            "INSERT INTO document_version"
            " (document_id, fingerprint, embed_model, embed_dim, output_root)"
            " VALUES (%s, 'fp-halfbuilt', 'fake', %s, %s)",
            (building, DIMENSIONS, str(root)),
        )
        client.conn.commit()
        _expect(
            IndexNotReadyError,
            lambda: client.audit_claim(SUPPORTED, document_ids=[int(building)]),
            "a building-only document is not ready",
        )
        _expect(
            IndexNotReadyError,
            lambda: client.search_evidence(SUPPORTED, document_ids=[int(building)]),
            "and search says so too",
        )

        removed = client.ingest_document(build_root(tmp / "removed"), source_uri="urn:gone")
        client.remove_document(removed.document_id, confirm_document_id=removed.document_id)
        _expect(
            NotFoundError,
            lambda: client.audit_claim(SUPPORTED, document_ids=[removed.document_id]),
            "a removed document is not found",
        )
        client.conn.execute("DELETE FROM document WHERE id = %s", (building,))
        client.conn.commit()


def check_invalid_scope_costs_nothing(tmp: Path) -> None:
    """Validation runs before Ollama and before the audit row is written."""
    session = default_session()
    with make_client(session) as client:
        before = client.conn.execute("SELECT count(*) AS n FROM audit_run").fetchone()["n"]
        calls = len(session.requests)
        _expect(
            NotFoundError,
            lambda: client.audit_claim(SUPPORTED, document_ids=[10_000_001]),
            "an unknown scope is rejected",
        )
        check(len(session.requests) == calls, "the model was never called")
        after = client.conn.execute("SELECT count(*) AS n FROM audit_run").fetchone()["n"]
        check(after == before, "no audit row was created")


def check_empty_index_is_not_an_insufficient_verdict(tmp: Path) -> None:
    with make_client(default_session()) as client:
        client.conn.execute("DELETE FROM document")
        client.conn.commit()
        _expect(
            IndexNotReadyError,
            lambda: client.audit_claim(SUPPORTED),
            "an empty index raises rather than returning insufficient",
        )
        _expect(
            IndexNotReadyError,
            lambda: client.search_evidence(SUPPORTED),
            "and search raises rather than returning nothing",
        )


# --- H-7: one authoritative structured explanation --------------------------


def _qualifiers(comparison) -> dict[str, Any]:
    return {q.qualifier: q for q in comparison.qualifiers}


def check_supported_claim_explains_itself(tmp: Path) -> None:
    with make_client(default_session()) as client:
        result = client.audit_claim(SUPPORTED)
        explanation = result.decision_explanation
        check(explanation is not None, "a verdict carries a structured explanation")
        check(
            explanation.decided_by == "deterministic_comparison",
            f"decided by arithmetic ({explanation.decided_by})",
        )
        check(
            explanation.verdict_rule == "exact_numeric_match",
            f"named rule ({explanation.verdict_rule})",
        )

        matching = [
            c for c in explanation.evidence_comparisons if c.numeric.outcome == "match"
        ]
        check(bool(matching), "the matching comparison is reported")
        target = matching[0]
        quals = _qualifiers(target)
        for name in ("scope", "unit", "reporting_period", "baseline_period"):
            check(
                quals[name].status == "match",
                f"{name} matched ({quals[name].status}: {quals[name].source_value})",
            )
        check("40.2" in (target.numeric.source_value or ""), "the source value is quoted")
        check(target.numeric.claim_value == "40.2", "the claim value is quoted")
        check(
            target.evidence_id in {c.evidence_id for c in result.citations},
            "the comparison points at cited evidence",
        )
        check(target.pdf_page in (1, 2), f"and at its page ({target.pdf_page})")

        check(
            result.timings.get("total") is not None
            and result.timings["retrieval"] is not None,
            f"timings are measured ({result.timings})",
        )
        check(
            set(result.timings) == {
                "parsing", "retrieval", "fusion_context", "visual_verification",
                "verdict", "persistence", "total",
            },
            f"every public timing group is present ({sorted(result.timings)})",
        )
        check(bool(result.index_references), "the audited index versions are reported")
        reference = result.index_references[0]
        check(
            reference.embedding_dimensions == DIMENSIONS
            and reference.embedding_model == client.settings.embed_model,
            "the reference names the embedding model and width",
        )


def check_contradicted_claim_explains_the_conflict(tmp: Path) -> None:
    with make_client(default_session()) as client:
        result = client.audit_claim(CONTRADICTED)
        explanation = result.decision_explanation
        check(
            explanation.verdict_rule == "comparable_numeric_conflict",
            f"named rule ({explanation.verdict_rule})",
        )
        conflicting = [
            c for c in explanation.evidence_comparisons if c.numeric.outcome == "conflict"
        ]
        check(bool(conflicting), "the conflicting comparison is reported")
        target = conflicting[0]
        check(target.numeric.claim_value == "90", "the claim's 90 is shown")
        check("40.2" in (target.numeric.source_value or ""), "against the source's 40.2")
        quals = _qualifiers(target)
        check(
            all(quals[n].status == "match" for n in ("scope", "unit", "reporting_period")),
            "every other material qualifier still matches",
        )


def check_vague_claim_explains_the_scope_gap(tmp: Path) -> None:
    with make_client(default_session()) as client:
        result = client.audit_claim(VAGUE)
        explanation = result.decision_explanation
        check(
            explanation.verdict_rule in ("scope_not_comparable", "missing_material_qualifier"),
            f"named rule ({explanation.verdict_rule})",
        )
        check(
            explanation.decided_by == "semantic_adjudication",
            "arithmetic refused, so the semantic path decided",
        )
        scoped = [
            _qualifiers(c)["scope"]
            for c in explanation.evidence_comparisons
            if "scope" in _qualifiers(c)
        ]
        check(bool(scoped), "scope comparisons are reported")
        check(
            all(q.status in ("mismatch", "missing") for q in scoped),
            "no scope is claimed to match",
        )
        check(
            all(
                c.numeric.outcome != "conflict"
                for c in explanation.evidence_comparisons
                if _qualifiers(c)["scope"].status != "match"
            ),
            "a scope that does not line up is never a numeric contradiction",
        )


def check_explanation_round_trips_through_the_trace(tmp: Path) -> None:
    with make_client(default_session()) as client:
        result = client.audit_claim(SUPPORTED)
        trace = client.get_audit_trace(result.audit_id)
        check(
            trace.decision_explanation.verdict_rule
            == result.decision_explanation.verdict_rule,
            "the stored rule matches the returned one",
        )
        check(
            len(trace.decision_explanation.evidence_comparisons)
            == len(result.decision_explanation.evidence_comparisons),
            "every comparison survives persistence",
        )
        check(trace.timings.get("total") is not None, "timings persist")
        check(
            [r.document_version_id for r in trace.index_references]
            == [r.document_version_id for r in result.index_references],
            "index references persist",
        )

        insufficient = client.audit_claim(VAGUE)
        empty_trace = client.get_audit_trace(insufficient.audit_id)
        check(not insufficient.citations, "the vague claim cites nothing")
        check(
            bool(empty_trace.index_references),
            "an insufficient audit still records what it searched",
        )
        check(
            bool(empty_trace.document_ids),
            "and reports the audited document ids without citations",
        )


def check_explanation_carries_no_model_text(tmp: Path) -> None:
    with make_client(default_session()) as client:
        result = client.audit_claim(VAGUE)
        stored = client.conn.execute(
            "SELECT decision_explanation, timings, index_references"
            " FROM audit_run WHERE id = %s",
            (result.audit_id,),
        ).fetchone()
        blob = json.dumps(
            [stored["decision_explanation"], stored["timings"], stored["index_references"]]
        )
        for secret in (
            "You decide whether",  # the adjudication system prompt
            "You decompose one atomic claim",  # the claim-parse prompt
            "does not state a comparable scope",  # the model's own rationale
            "postgresql://",
            str(tmp),
        ):
            check(secret not in blob, f"persisted explanation omits {secret[:32]!r}")


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
        check_adjudicator_prompt_is_bounded,
        check_contradicted_claim,
        check_vague_claim_is_insufficient,
        check_verdict_fails_closed_without_citable_evidence,
        check_interrupted_build_is_invisible,
        check_visual_evidence_needs_crop_verification,
        check_middle_range_keeps_pdf_pages,
        check_source_less_documents_stay_separate,
        check_shared_source_uri_is_one_document,
        check_schema_migrates_a_legacy_document,
        check_resume_reconciles_the_building_version,
        check_supported_claim_explains_itself,
        check_contradicted_claim_explains_the_conflict,
        check_vague_claim_explains_the_scope_gap,
        check_explanation_round_trips_through_the_trace,
        check_explanation_carries_no_model_text,
        check_document_scope_is_validated,
        check_invalid_scope_costs_nothing,
        # Last: it empties the index the checks above rely on.
        check_empty_index_is_not_an_insufficient_verdict,
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
