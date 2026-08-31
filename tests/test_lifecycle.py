"""Build lifecycle: degraded, retried, promoted, and honestly interrupted.

Needs the Compose PostgreSQL; the model is faked throughout.

    docker compose up -d
    python tests/test_lifecycle.py
"""

from __future__ import annotations

import os
from pathlib import Path

import psycopg
import pytest
from fake_ollama import FakeSession
from fixtures import block, kpi_table, write_output_root

from claim_evidence import ClaimEvidence, Settings
from claim_evidence.db import (
    QUERYABLE_STATUSES,
    connect,
    failed_fact_candidates,
    reconcile_interrupted,
)
from claim_evidence.models import VersionStatus
from claim_evidence.ollama import OllamaClient

ADMIN_URL = os.environ.get(
    "CLAIM_EVIDENCE_DATABASE_URL",
    "postgresql://claim_evidence:claim_evidence@localhost:5433/claim_evidence",
)
TEST_DB = "claim_evidence_lifecycle_test"
DIMENSIONS = 8
# Stated explicitly: version 1 never derives the entity from a filename.
ENTITY = "Danone S.A."


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


def database_url() -> str:
    return ADMIN_URL.rsplit("/", 1)[0] + f"/{TEST_DB}"


def available() -> bool:
    try:
        with psycopg.connect(ADMIN_URL, connect_timeout=3):
            return True
    except psycopg.OperationalError:
        return False


def fresh_database() -> None:
    with psycopg.connect(ADMIN_URL, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{TEST_DB}" WITH (FORCE)')
        conn.execute(f'CREATE DATABASE "{TEST_DB}"')


def settings() -> Settings:
    return Settings(
        database_url=database_url(),
        embed_dimensions=DIMENSIONS,
        chat_model="fake-chat",
        vision_model="fake-vision",
    )


class FailingFactSession(FakeSession):
    """Fails fact extraction for a chosen number of passages, then succeeds."""

    def __init__(self, failures: int, **kwargs) -> None:
        super().__init__(dimensions=DIMENSIONS, **kwargs)
        self.remaining_failures = failures
        self.fact_calls = 0

    def post(self, url: str, json: dict, timeout: float):  # noqa: A002
        if url.endswith("/api/chat") and (json.get("format") or {}).get(
            "title"
        ) == "FactExtraction":
            self.fact_calls += 1
            if self.remaining_failures > 0:
                self.remaining_failures -= 1
                import requests

                raise requests.ConnectionError("fake model outage")
            from fake_ollama import FakeResponse, reply

            return FakeResponse({"message": {"content": reply({"facts": []})}})
        return super().post(url, json, timeout)


def three_candidate_root(root: Path) -> Path:
    """Three narrative passages that all look claim-like, plus one table."""
    return write_output_root(
        root,
        pages=1,
        blocks=[
            block(1, 1, "Emissions fell by 40.2% in 2025 versus the 2020 baseline."),
            block(1, 2, "Renewable electricity reached 94.2% of the energy mix in 2025."),
            block(1, 3, "Water withdrawal intensity fell by 12.0% in 2025."),
        ],
        tables={1: [kpi_table()]},
    )


def client_with(session: FakeSession) -> ClaimEvidence:
    config = settings()
    return ClaimEvidence(config, connect(config.database_url), OllamaClient(config, session))


def prepared(session: FakeSession) -> ClaimEvidence:
    fresh_database()
    client = client_with(session)
    client.init_db()
    return client


# --- degraded ---------------------------------------------------------------


def test_a_partial_fact_failure_is_degraded_and_still_queryable(tmp_path: Path) -> None:
    if not available():
        pytest.skip("PostgreSQL is not reachable")
    root = three_candidate_root(tmp_path / "degraded")
    session = FailingFactSession(failures=1)
    client = prepared(session)
    try:
        report = client.ingest_document(root, source_uri="urn:degraded", reporting_entity=ENTITY)
        check(
            report.status is VersionStatus.DEGRADED,
            f"one failed candidate of three is degraded, not ready ({report.status})",
        )
        check(
            report.fact_candidates_total == 3
            and report.fact_candidates_succeeded == 2,
            f"coverage counted: {report.fact_candidates_succeeded}"
            f"/{report.fact_candidates_total}",
        )
        check(report.fact_coverage == round(2 / 3, 4), "coverage is reported as a fraction")
        check(
            len(report.failed_fact_candidates) == 1,
            "the failed candidate is named so a retry can find it",
        )
        check(
            "degraded" in QUERYABLE_STATUSES,
            "a degraded version is queryable by definition",
        )
        matches = client.search_evidence("emissions", document_ids=[report.document_id])
        check(bool(matches), "a degraded version answers queries")
        check(report.evidence_units > 0 and report.embedded_units > 0,
              "its evidence and embeddings are complete")
    finally:
        client.close()


def test_failed_candidates_store_keys_and_codes_only(tmp_path: Path) -> None:
    if not available():
        pytest.skip("PostgreSQL is not reachable")
    root = three_candidate_root(tmp_path / "safe-failures")
    client = prepared(FailingFactSession(failures=2))
    try:
        report = client.ingest_document(root, source_uri="urn:safe-failures", reporting_entity=ENTITY)
        rows = failed_fact_candidates(client.conn, report.version_id)
        check(len(rows) == 2, "both failures are recorded")
        for row in rows:
            check(
                row["reason_code"] == "model_unavailable",
                "the failure is a category, not a message",
            )
            check(
                "Emissions fell" not in row["unit_key"]
                and "outage" not in row["reason_code"],
                "neither the passage nor the model's error text is stored",
            )
        stored = client.conn.execute(
            "SELECT string_agg(unit_key || reason_code, ' ') AS blob"
            " FROM fact_candidate_failure"
        ).fetchone()["blob"]
        check(
            "fake model outage" not in stored and "ConnectionError" not in stored,
            "the raw failure never reaches the database",
        )
    finally:
        client.close()


# --- retry ------------------------------------------------------------------


def test_retry_processes_only_the_failures_and_promotes(tmp_path: Path) -> None:
    if not available():
        pytest.skip("PostgreSQL is not reachable")
    root = three_candidate_root(tmp_path / "retry")
    session = FailingFactSession(failures=1)
    client = prepared(session)
    try:
        report = client.ingest_document(root, source_uri="urn:retry", reporting_entity=ENTITY)
        check(report.status is VersionStatus.DEGRADED, "the build is degraded")
        calls_after_build = session.fact_calls
        evidence_before = client.conn.execute(
            "SELECT id, source_text FROM evidence_unit WHERE version_id = %s"
            " ORDER BY id",
            (report.version_id,),
        ).fetchall()

        retried = client.retry_facts(report.document_id)
        check(
            session.fact_calls == calls_after_build + 1,
            f"exactly one passage was re-processed "
            f"({session.fact_calls - calls_after_build})",
        )
        check(
            retried.status is VersionStatus.READY,
            f"clearing the last failure promotes to ready ({retried.status})",
        )
        check(retried.failed_fact_candidates == [], "no failures remain")
        check(
            retried.fact_candidates_succeeded == retried.fact_candidates_total == 3,
            "coverage is complete after the retry",
        )

        evidence_after = client.conn.execute(
            "SELECT id, source_text FROM evidence_unit WHERE version_id = %s"
            " ORDER BY id",
            (report.version_id,),
        ).fetchall()
        check(
            evidence_before == evidence_after,
            "unchanged evidence was not rebuilt: same ids, same text",
        )
    finally:
        client.close()


def test_a_retry_that_still_fails_stays_degraded(tmp_path: Path) -> None:
    if not available():
        pytest.skip("PostgreSQL is not reachable")
    root = three_candidate_root(tmp_path / "still-failing")
    session = FailingFactSession(failures=99)
    client = prepared(session)
    try:
        report = client.ingest_document(root, source_uri="urn:still-failing", reporting_entity=ENTITY)
        check(report.status is VersionStatus.DEGRADED, "every candidate failed")
        retried = client.retry_facts(report.document_id)
        check(
            retried.status is VersionStatus.DEGRADED,
            "a retry that fails again does not promote",
        )
        check(
            len(retried.failed_fact_candidates) == 3,
            "and still reports what is outstanding",
        )
    finally:
        client.close()


def test_reindexing_a_degraded_version_reuses_it(tmp_path: Path) -> None:
    """Rebuilding identical evidence to chase optional facts is the wrong tool."""
    if not available():
        pytest.skip("PostgreSQL is not reachable")
    root = three_candidate_root(tmp_path / "reuse-degraded")
    client = prepared(FailingFactSession(failures=1))
    try:
        first = client.ingest_document(root, source_uri="urn:reuse-degraded", reporting_entity=ENTITY)
        second = client.ingest_document(root, source_uri="urn:reuse-degraded", reporting_entity=ENTITY)
        check(second.reused_existing, "the degraded version is reused, not rebuilt")
        check(second.version_id == first.version_id, "and it is the same version")
        check(
            second.status is VersionStatus.DEGRADED
            and len(second.failed_fact_candidates) == 1,
            "the no-op run still reports the outstanding failure",
        )
    finally:
        client.close()


# --- interruption -----------------------------------------------------------


def test_restart_marks_prior_nonterminal_work_interrupted(tmp_path: Path) -> None:
    if not available():
        pytest.skip("PostgreSQL is not reachable")
    root = three_candidate_root(tmp_path / "interrupted")
    client = prepared(FailingFactSession(failures=0))
    try:
        ready = client.ingest_document(root, source_uri="urn:interrupted", reporting_entity=ENTITY)
        check(ready.status is VersionStatus.READY, "one good version exists")

        # A build and an audit that were still going when the process died.
        conn = client.conn
        conn.execute(
            """
            INSERT INTO document_version
                (document_id, fingerprint, embed_model, embed_dim, output_root,
                 status, attempt)
            VALUES (%s, 'half-built', 'fake', %s, %s, 'building', 9)
            """,
            (ready.document_id, DIMENSIONS, str(root)),
        )
        conn.execute("INSERT INTO audit_run (claim) VALUES ('a claim in flight')")
        conn.commit()

        counts = client.reconcile_interrupted()
        check(counts == {"builds": 1, "audits": 1}, f"one of each reconciled ({counts})")

        build = conn.execute(
            "SELECT status, failure_code FROM document_version"
            " WHERE fingerprint = 'half-built'"
        ).fetchone()
        check(build["status"] == "interrupted", "the dead build is interrupted")
        check(
            build["failure_code"] == "interrupted",
            "not 'failed': nobody observed it fail",
        )
        audit = conn.execute(
            "SELECT status, retryable FROM audit_run WHERE claim = 'a claim in flight'"
        ).fetchone()
        check(audit["status"] == "interrupted", "the dead audit is interrupted")
        check(audit["retryable"] is True, "and is marked retryable")

        survivor = conn.execute(
            "SELECT status FROM document_version WHERE id = %s", (ready.version_id,)
        ).fetchone()
        check(
            survivor["status"] == "ready",
            "the version that was already serving is untouched",
        )
        matches = client.search_evidence("emissions", document_ids=[ready.document_id])
        check(bool(matches), "and it is still queryable after the restart")
    finally:
        client.close()


def test_reconciliation_never_invents_a_completed_audit(tmp_path: Path) -> None:
    if not available():
        pytest.skip("PostgreSQL is not reachable")
    client = prepared(FailingFactSession(failures=0))
    try:
        conn = client.conn
        conn.execute(
            "INSERT INTO audit_run (claim, status, verdict, completed_at)"
            " VALUES ('finished', 'completed', 'supported', now())"
        )
        conn.execute("INSERT INTO audit_run (claim) VALUES ('in flight')")
        conn.commit()
        reconcile_interrupted(conn)
        conn.commit()
        rows = {
            r["claim"]: r["status"]
            for r in conn.execute("SELECT claim, status FROM audit_run").fetchall()
        }
        check(rows["finished"] == "completed", "a finished audit is left alone")
        check(rows["in flight"] == "interrupted", "an unfinished one becomes interrupted")
        check(
            "failed" not in rows.values(),
            "no audit is reported as failed on the strength of a restart",
        )
    finally:
        client.close()


def main() -> int:
    if not available():
        print("[skip] PostgreSQL is not reachable; run `docker compose up -d`")
        return 0
    import tempfile

    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            print(f"\n--- {name} ---")
            with tempfile.TemporaryDirectory() as temp:
                function(Path(temp))

    with psycopg.connect(ADMIN_URL, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{TEST_DB}" WITH (FORCE)')
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
