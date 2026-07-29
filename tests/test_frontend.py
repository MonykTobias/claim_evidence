"""Frontend-support APIs: health, documents, removal, trace, evidence detail.

Needs the Compose PostgreSQL; skips cleanly without it. Ollama is faked.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import psycopg  # noqa: E402
from fake_ollama import FakeSession  # noqa: E402
from fixtures import block, image_summary, kpi_table, write_output_root  # noqa: E402
from test_integration import (  # noqa: E402
    DIMENSIONS,
    VAGUE,
    build_root,
    default_session,
    make_client,
    reset_database,
    settings,
)

from claim_evidence import (  # noqa: E402
    ClaimEvidence,
    NotFoundError,
    ValidationError,
    VersionStatus,
)
from claim_evidence.db import SCHEMA_VERSION  # noqa: E402
from claim_evidence.models import EvidenceKind, GeometryPrecision, RegionRole  # noqa: E402

SUPPORTED = "Danone reduced Scope 1 and 2 energy and industry emissions by 40.2% in 2025 versus 2020."


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


def expect(error: type[Exception], func, fragment: str, message: str) -> None:
    try:
        func()
    except error as exc:
        check(fragment in str(exc), f"{message} ({exc})")
        return
    raise AssertionError(f"{message}: no {error.__name__} raised")


# --- health -----------------------------------------------------------------


def check_initialize_is_idempotent(tmp: Path) -> None:
    with make_client(default_session()) as client:
        first = client.initialize_database()
        second = client.initialize_database()
        check(first.schema_version == SCHEMA_VERSION, "schema version reported")
        check(second.schema_current, "second init leaves the schema current")
        check(second.database_reachable, "database reported reachable")


def check_health_reports_missing_models_without_leaking(tmp: Path) -> None:
    with make_client(default_session()) as client:
        client.initialize_database()
        report = client.health()
        # The fake session has no /api/tags, so Ollama reads as unreachable.
        check(not report.ollama_reachable, "unreachable ollama is reported")
        check(
            any("ollama is not reachable" in p for p in report.problems),
            "problem names the dependency",
        )
        check(
            all(not m.available for m in report.models),
            "no model reads as available when ollama is down",
        )
        blob = report.model_dump_json()
        for secret in ("claim_evidence:claim_evidence", "postgresql://", "password"):
            check(secret not in blob, f"health output does not contain {secret!r}")
        check(not report.ok, "report is not ok while a dependency is down")


def check_health_reports_available_models(tmp: Path) -> None:
    class TagSession(FakeSession):
        def get(self, url: str, timeout: float) -> Any:  # noqa: A002
            from fake_ollama import FakeResponse

            return FakeResponse(
                {"models": [{"name": "fake-chat"}, {"name": "fake-vision"}]}
            )

    with make_client(TagSession(dimensions=DIMENSIONS)) as client:
        client.initialize_database()
        report = client.health()
        check(report.ollama_reachable, "reachable ollama is reported")
        by_role = {m.role: m for m in report.models}
        check(by_role["chat"].available, "pulled chat model is available")
        check(not by_role["embed"].available, "unpulled embed model is flagged")


def check_health_survives_a_missing_schema(tmp: Path) -> None:
    with psycopg.connect(settings().database_url, autocommit=True) as raw:
        raw.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
    with make_client(default_session()) as client:
        report = client.health()
        check(report.database_reachable, "database still reachable")
        check(report.schema_version is None, "missing schema reported as absent")
        check(not report.schema_current, "missing schema is not current")
        check(
            any("not initialized" in p for p in report.problems),
            "problem tells the caller to run db init",
        )
        client.initialize_database()


# --- documents --------------------------------------------------------------


def check_document_listing(tmp: Path) -> None:
    root = build_root(tmp / "listing")
    with make_client(default_session()) as client:
        report = client.ingest_document(root, source_uri="urn:listing")
        docs = client.list_documents()
        check(len(docs) == 1, "one document listed")
        doc = docs[0]
        check(doc.status is VersionStatus.READY, "status reported")
        check(doc.page_count == 2, "page count reported")
        check(doc.evidence_count == report.evidence_units, "evidence count matches")
        check(doc.fact_count == report.facts, "fact count matches")
        check(doc.source_uri == "urn:listing", "source uri carried")
        check(doc.embedding_dimensions == DIMENSIONS, "embedding dimensions carried")
        check(doc.output_root == str(Path(root).resolve()), "output root carried")
        check(doc.indexed_at is not None, "indexed timestamp set")

        same = client.get_document(str(doc.document_id))
        check(same.document_id == doc.document_id, "get_document accepts a string id")
        expect(NotFoundError, lambda: client.get_document(999999),
               "no document", "unknown document id rejected")
        expect(ValidationError, lambda: client.get_document("not-an-id"),
               "must be an integer id", "malformed document id rejected")


def check_visual_count_is_reported(tmp: Path) -> None:
    root = build_root(tmp / "visuals", with_visual=True)
    with make_client(default_session()) as client:
        client.ingest_document(root, source_uri="urn:visuals")
        doc = next(d for d in client.list_documents() if d.source_uri == "urn:visuals")
        check(doc.visual_evidence_count == 1, "visual evidence counted separately")


# --- forced re-index --------------------------------------------------------


def check_force_keeps_the_old_version_until_replacement(tmp: Path) -> None:
    root = build_root(tmp / "forced")
    with make_client(default_session()) as client:
        first = client.ingest_document(root, source_uri="urn:forced")
        plain = client.ingest_document(root, source_uri="urn:forced")
        check(plain.reused_existing, "an unchanged re-ingest is still a no-op")

        rebuilt = client.ingest_document(root, source_uri="urn:forced", force=True)
        check(not rebuilt.reused_existing, "force rebuilds an unchanged source")
        check(rebuilt.version_id != first.version_id, "a new version was built")
        check(rebuilt.status is VersionStatus.READY, "the replacement is ready")

        rows = {
            int(r["id"]): r["status"]
            for r in client.conn.execute(
                "SELECT id, status FROM document_version WHERE document_id = %s",
                (first.document_id,),
            ).fetchall()
        }
        check(rows[first.version_id] == "inactive", "the old version was retired")
        check(rows[rebuilt.version_id] == "ready", "the new version is active")

        docs = [d for d in client.list_documents() if d.source_uri == "urn:forced"]
        check(len(docs) == 1, "the document is still listed once")
        check(docs[0].document_version_id == rebuilt.version_id, "listing shows the new version")


def check_failed_rebuild_leaves_the_old_version_ready(tmp: Path) -> None:
    root = build_root(tmp / "failing")
    with make_client(default_session()) as client:
        good = client.ingest_document(root, source_uri="urn:failing")

        # Break the integrity check the way a partial write would.
        import claim_evidence.ingest as ingest_module

        original = ingest_module._verify
        ingest_module._verify = lambda *a, **k: (_ for _ in ()).throw(
            ingest_module.IngestionError("simulated integrity failure")
        )
        try:
            client.ingest_document(root, source_uri="urn:failing", force=True)
        except ingest_module.IngestionError:
            pass
        else:
            raise AssertionError("the simulated failure did not propagate")
        finally:
            ingest_module._verify = original

        status = client.conn.execute(
            "SELECT status FROM document_version WHERE id = %s", (good.version_id,)
        ).fetchone()["status"]
        check(status == "ready", "the working version is still ready after a failed rebuild")
        doc = next(d for d in client.list_documents() if d.source_uri == "urn:failing")
        check(doc.document_version_id == good.version_id, "queries still see the old version")
        check(doc.status is VersionStatus.READY, "the document is still queryable")

        matches = client.search_evidence("renewable electricity", limit=5)
        check(bool(matches), "search still works after a failed rebuild")


# --- removal ----------------------------------------------------------------


def check_removal_requires_exact_confirmation(tmp: Path) -> None:
    root = build_root(tmp / "confirm")
    with make_client(default_session()) as client:
        report = client.ingest_document(root, source_uri="urn:confirm")
        before = client.conn.execute(
            "SELECT count(*) AS n FROM evidence_unit WHERE version_id = %s",
            (report.version_id,),
        ).fetchone()["n"]

        expect(
            ValidationError,
            lambda: client.remove_document(
                report.document_id, confirm_document_id=report.document_id + 1
            ),
            "must equal document_id",
            "mismatched confirmation rejected",
        )
        after = client.conn.execute(
            "SELECT count(*) AS n FROM evidence_unit WHERE version_id = %s",
            (report.version_id,),
        ).fetchone()["n"]
        check(after == before, "a rejected removal changed nothing")


def check_removal_deletes_only_its_own_rows(tmp: Path) -> None:
    keep_root = build_root(tmp / "keep")
    drop_root = build_root(tmp / "drop")
    with make_client(default_session()) as client:
        keep = client.ingest_document(keep_root, source_uri="urn:keep")
        drop = client.ingest_document(drop_root, source_uri="urn:drop")
        client.audit_claim(SUPPORTED, document_ids=[drop.document_id])

        report = client.remove_document(
            drop.document_id, confirm_document_id=drop.document_id
        )
        check(report.document_id == drop.document_id, "removal reports the document")
        check(report.deleted["evidence_units"] == drop.evidence_units,
              f"evidence count reported ({report.deleted})")
        check(report.deleted["pages"] == 2, "page count reported")
        check(report.deleted["facts"] == drop.facts, "fact count reported")
        check(report.deleted["evidence_regions"] > 0, "region count reported")

        remaining = {d.document_id for d in client.list_documents()}
        check(drop.document_id not in remaining, "removed document is gone")
        check(keep.document_id in remaining, "the other document survives")

        kept = client.conn.execute(
            "SELECT count(*) AS n FROM evidence_unit WHERE version_id = %s",
            (keep.version_id,),
        ).fetchone()["n"]
        check(kept == keep.evidence_units, "the other document's evidence is untouched")
        orphans = client.conn.execute(
            "SELECT count(*) AS n FROM audit_candidate ac"
            " LEFT JOIN evidence_unit e ON e.id = ac.evidence_id WHERE e.id IS NULL"
        ).fetchone()["n"]
        check(orphans == 0, "audit candidates cascaded with their evidence")


def check_removal_leaves_source_files_alone(tmp: Path) -> None:
    root = build_root(tmp / "sources")
    before = sorted(p.relative_to(root).as_posix() for p in Path(root).rglob("*"))
    with make_client(default_session()) as client:
        report = client.ingest_document(root, source_uri="urn:sources")
        client.remove_document(report.document_id, confirm_document_id=report.document_id)
    after = sorted(p.relative_to(root).as_posix() for p in Path(root).rglob("*"))
    check(before == after, "every output file survives removal")
    check((Path(root) / "page_0001" / "page.png").is_file(), "page images survive")
    check((Path(root) / "manifest.json").is_file(), "the manifest survives")


def check_removed_document_leaves_query_scope(tmp: Path) -> None:
    root = build_root(tmp / "scope")
    with make_client(default_session()) as client:
        report = client.ingest_document(root, source_uri="urn:scope")
        check(bool(client.search_evidence("renewable electricity", limit=5)),
              "evidence is searchable before removal")
        client.remove_document(report.document_id, confirm_document_id=report.document_id)
        matches = client.search_evidence("renewable electricity", limit=5)
        check(
            all(m.citation.document_id != report.document_id for m in matches),
            "removed evidence never appears in search",
        )
        try:
            client.audit_claim(SUPPORTED, document_ids=[report.document_id])
        except NotFoundError as exc:
            # Not an empty verdict: "no evidence" would read as a statement
            # about the report rather than about the removed selection.
            check(str(report.document_id) in str(exc), f"the removed id is named ({exc})")
        else:
            raise AssertionError("auditing a removed document returned a verdict")


# --- trace and evidence -----------------------------------------------------


def check_trace_reports_every_channel(tmp: Path) -> None:
    root = build_root(tmp / "trace")
    with make_client(default_session()) as client:
        client.ingest_document(root, source_uri="urn:trace")
        result = client.audit_claim(SUPPORTED)
        trace = client.get_audit_trace(result.audit_id)

        check(trace.audit_id == result.audit_id, "trace is the audit that ran")
        check(trace.claim == SUPPORTED, "claim recorded")
        check(trace.verdict == result.verdict, "verdict recorded")
        check(trace.parsed_claim, "parsed claim fields recorded")
        check(bool(trace.candidates), "candidates recorded")
        check(bool(trace.lexical_candidates), "full-text candidates present")
        check(bool(trace.vector_candidates), "vector candidates present")
        check(bool(trace.graph_candidates), "graph/fact candidates present")
        check(
            any(c.combined_rank for c in trace.candidates), "fused rank recorded"
        )
        check(
            any(c.lexical_score is not None for c in trace.candidates),
            "channel score recorded alongside the rank",
        )
        selected = [c for c in trace.candidates if c.selected]
        check(bool(selected), "selected candidates flagged")
        check(
            sorted(c.evidence_id for c in selected) == sorted(trace.citation_ids),
            "selected candidates match the citation ids",
        )
        check(all(c.reason for c in trace.candidates), "every candidate has a reason")
        check(
            all(c.visual_status == "not_applicable" for c in trace.candidates),
            "non-visual candidates report no visual status",
        )
        expect(NotFoundError, lambda: client.get_audit_trace(999999),
               "no audit", "unknown audit id rejected")


def check_trace_carries_no_prompt_text(tmp: Path) -> None:
    root = build_root(tmp / "noprompt")
    with make_client(default_session()) as client:
        client.ingest_document(root, source_uri="urn:noprompt")
        result = client.audit_claim(VAGUE)
        blob = client.get_audit_trace(result.audit_id).model_dump_json()
        for leak in (
            "You decide whether",
            "You extract auditable facts",
            "You decompose one atomic claim",
            "Reply with valid JSON only",
        ):
            check(leak not in blob, f"trace contains no prompt text ({leak!r})")


def check_expansion_relationships_are_recorded(tmp: Path) -> None:
    root = build_root(tmp / "expansion")
    with make_client(default_session()) as client:
        client.ingest_document(root, source_uri="urn:expansion")
        result = client.audit_claim(SUPPORTED)
        trace = client.get_audit_trace(result.audit_id)
        expanded = [c for c in trace.candidates if c.expanded_from]
        check(bool(expanded), "context expansion recorded")
        known = {c.evidence_id for c in trace.candidates}
        check(
            all(c.expanded_from in known for c in expanded),
            "each expansion points at another candidate in the same trace",
        )


def check_evidence_detail(tmp: Path) -> None:
    root = build_root(tmp / "detail")
    with make_client(default_session()) as client:
        client.ingest_document(root, source_uri="urn:detail")
        result = client.audit_claim(SUPPORTED)
        cited = result.citations[0].evidence_id

        detail = client.get_evidence(str(cited))
        check(detail.evidence_id == cited, "evidence detail accepts a string id")
        check(detail.pdf_page == 1, "1-based pdf page returned")
        check(detail.page_width > 0 and detail.page_height > 0, "page size returned")
        check(detail.page_image_path is not None, "page image resolved")
        check(Path(detail.page_image_path).is_file(), "page image really exists")
        check(
            Path(detail.page_image_path).name == "page.png",
            "the page image is the registered page.png",
        )
        check(bool(detail.artifact_paths), "artifact paths returned")
        check(all(Path(p).is_file() for p in detail.artifact_paths),
              "artifact paths point at real files")

        roles = {r.role for r in detail.regions}
        check(
            {RegionRole.DESCRIPTOR, RegionRole.HEADER, RegionRole.UNIT, RegionRole.VALUE}
            <= roles,
            f"table evidence preserves all four cell roles ({roles})",
        )
        check(len(detail.regions) >= 4, "regions are not collapsed into one box")
        check(
            all(r.coordinate_space == "normalized_top_left" for r in detail.regions),
            "every region declares its coordinate space",
        )
        check(
            all(0.0 <= v <= 1.0 for r in detail.regions for v in r.bbox),
            "every region is normalized",
        )
        check(detail.geometry_precision is GeometryPrecision.CELL, "cell precision")
        check(detail.source_kind is EvidenceKind.TABLE_VALUE, "source kind returned")
        check(bool(detail.table_context), "table context returned")

        expect(NotFoundError, lambda: client.get_evidence(999999),
               "no evidence", "unknown evidence id rejected")
        expect(ValidationError, lambda: client.get_evidence("abc"),
               "must be an integer id", "malformed evidence id rejected")


def check_missing_cell_geometry_reports_fallback_precision(tmp: Path) -> None:
    root = build_root(tmp / "fallback")
    with make_client(default_session()) as client:
        report = client.ingest_document(root, source_uri="urn:fallback")
        row = client.conn.execute(
            """
            SELECT id FROM evidence_unit
            WHERE version_id = %s AND kind = 'table_value'
              AND table_context->>'value' = %s
            """,
            (report.version_id, "85.7%"),
        ).fetchone()
        check(row is not None, "the cell without geometry was indexed")
        detail = client.get_evidence(row["id"])
        check(
            detail.geometry_precision is GeometryPrecision.ROW,
            f"missing cell geometry reports row precision ({detail.geometry_precision})",
        )
        check(bool(detail.regions), "the fallback still yields a highlightable region")


def main() -> int:
    try:
        reset_database()
    except psycopg.OperationalError as exc:
        print(f"[skip] postgres unavailable ({exc.__class__.__name__})")
        return 0

    checks = [
        check_initialize_is_idempotent,
        check_health_reports_missing_models_without_leaking,
        check_health_reports_available_models,
        check_document_listing,
        check_visual_count_is_reported,
        check_force_keeps_the_old_version_until_replacement,
        check_failed_rebuild_leaves_the_old_version_ready,
        check_removal_requires_exact_confirmation,
        check_removal_deletes_only_its_own_rows,
        check_removal_leaves_source_files_alone,
        check_removed_document_leaves_query_scope,
        check_trace_reports_every_channel,
        check_trace_carries_no_prompt_text,
        check_expansion_relationships_are_recorded,
        check_evidence_detail,
        check_missing_cell_geometry_reports_fallback_precision,
        # Runs last: it drops and recreates the schema.
        check_health_survives_a_missing_schema,
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
