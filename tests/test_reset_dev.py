"""The guarded development reset: every refusal, and one real reset.

The refusal checks need no database at all -- that is the point: a guard that
only fires after connecting has already been given the chance to be wrong. The
one destructive check runs against a dedicated `_test` database and verifies
that source and configuration sentinels on disk survive it untouched.

    docker compose up -d
    python tests/test_reset_dev.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import psycopg
import pytest

from claim_evidence import Settings
from claim_evidence.db import connect, init_schema, schema_marker
from claim_evidence.reset import (
    CONFIRM_PHRASE,
    ResetRefused,
    app_is_running,
    check_name,
    reset_dev,
    target_of,
)
from fixtures import write_output_root

ADMIN_URL = os.environ.get(
    "CLAIM_EVIDENCE_DATABASE_URL",
    "postgresql://claim_evidence:claim_evidence@localhost:5433/claim_evidence",
)
TEST_DB = "claim_evidence_reset_test"
DIMENSIONS = 8


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


def url_for(database: str) -> str:
    return ADMIN_URL.rsplit("/", 1)[0] + f"/{database}"


def settings(database: str = TEST_DB, **overrides) -> Settings:
    return Settings(
        database_url=url_for(database),
        embed_dimensions=DIMENSIONS,
        environment=overrides.pop("environment", "development"),
        app_marker=overrides.pop("app_marker", str(Path(tempfile.gettempdir()) / "no-such-marker")),
        **overrides,
    )


def refuses(fn, fragment: str, message: str) -> None:
    try:
        fn()
    except ResetRefused as exc:
        check(fragment in str(exc), f"{message} ({exc})")
        return
    raise AssertionError(f"{message}: the reset was NOT refused")


def available() -> bool:
    try:
        with psycopg.connect(ADMIN_URL, connect_timeout=3):
            return True
    except psycopg.OperationalError:
        return False


# --- guards that never need a connection ------------------------------------


def test_environment_must_be_development() -> None:
    for environment in ("", "production", "Development", "dev"):
        refuses(
            lambda e=environment: reset_dev(
                settings(environment=e),
                confirm_database=TEST_DB,
                confirm_phrase=CONFIRM_PHRASE,
                dry_run=False,
            ),
            "CE_ENVIRONMENT",
            f"environment {environment!r} is refused",
        )


def test_only_dev_and_test_names_are_accepted() -> None:
    for name in ("claim_evidence", "app", "claim_evidence_devel", "test_claim_evidence"):
        refuses(
            lambda n=name: check_name(n), "does not end in", f"{name!r} is refused"
        )
    for name in ("claim_evidence_dev", "claim_evidence_test", "ce_test"):
        check_name(name)
    check(True, "only names ending _dev or _test are accepted")


def test_system_databases_are_never_targets() -> None:
    for name in ("postgres", "template0", "template1"):
        refuses(lambda n=name: check_name(n), "system database", f"{name!r} is refused")


def test_production_like_names_are_refused_even_with_a_dev_suffix() -> None:
    for name in ("prod_dev", "production_test", "staging_dev", "app_stage_test"):
        refuses(
            lambda n=name: check_name(n),
            "refusing in case",
            f"{name!r} is refused despite its suffix",
        )


def test_both_confirmations_are_required() -> None:
    refuses(
        lambda: reset_dev(
            settings(), confirm_database="", confirm_phrase=CONFIRM_PHRASE, dry_run=False
        ),
        "does not name the configured database",
        "a missing database confirmation is refused",
    )
    refuses(
        lambda: reset_dev(
            settings(),
            confirm_database="claim_evidence_dev",
            confirm_phrase=CONFIRM_PHRASE,
            dry_run=False,
        ),
        "does not name the configured database",
        "confirming a different database is refused",
    )
    refuses(
        lambda: reset_dev(
            settings(), confirm_database=TEST_DB, confirm_phrase="yes", dry_run=False
        ),
        "confirmation phrase",
        "a wrong confirmation phrase is refused",
    )
    refuses(
        lambda: reset_dev(
            settings(),
            confirm_database=TEST_DB,
            confirm_phrase=CONFIRM_PHRASE.lower(),
            dry_run=False,
        ),
        "confirmation phrase",
        "the confirmation phrase is case-sensitive",
    )


def test_a_running_application_blocks_the_reset() -> None:
    with tempfile.TemporaryDirectory() as temp:
        marker = Path(temp) / "claim_evidence_app.running"
        marker.write_text("pid 1234", encoding="utf-8")
        config = settings(app_marker=str(marker))
        check(app_is_running(config), "the marker file is detected")
        refuses(
            lambda: reset_dev(
                config,
                confirm_database=TEST_DB,
                confirm_phrase=CONFIRM_PHRASE,
                dry_run=False,
            ),
            "local application is running",
            "a reset is refused while the app holds its marker",
        )


def test_the_target_never_includes_the_password() -> None:
    target = target_of("postgresql://someone:hunter2@db.example:5433/claim_evidence_dev")
    check(target["database"] == "claim_evidence_dev", "the database name is read")
    check(target["user"] == "someone", "the user is read")
    check("hunter2" not in str(target), "the password is never part of the target")


# --- the real thing ---------------------------------------------------------


def prepare_database() -> None:
    if not available():
        pytest.skip("PostgreSQL is not reachable; the destructive checks need it")
    with psycopg.connect(ADMIN_URL, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{TEST_DB}" WITH (FORCE)')
        conn.execute(f'CREATE DATABASE "{TEST_DB}"')
    with connect(url_for(TEST_DB)) as conn:
        init_schema(conn, DIMENSIONS)
        conn.execute(
            "INSERT INTO document (name, identity_key) VALUES ('report.pdf', 'id-1')"
        )
        conn.execute("INSERT INTO audit_run (claim) VALUES ('a claim')")
        conn.commit()


def test_dry_run_is_the_default_and_changes_nothing() -> None:
    prepare_database()
    plan = reset_dev(
        settings(), confirm_database=TEST_DB, confirm_phrase=CONFIRM_PHRASE
    )
    check(plan.dry_run and not plan.performed, "the default is a dry run")
    check(plan.counts["document"] == 1 and plan.counts["audit_run"] == 1,
          "the dry run reports what would be lost")
    described = plan.describe()
    check(TEST_DB in described, "the dry run names the exact database")
    check("would delete" in described, "the dry run says it would delete, not that it did")
    check("claim_evidence:" not in described and "hunter" not in described,
          "the dry run prints no credential")
    check(
        "source PDFs, extraction output" in described,
        "the dry run states what is not touched",
    )
    with connect(url_for(TEST_DB)) as conn:
        rows = conn.execute("SELECT count(*) AS n FROM document").fetchone()["n"]
    check(rows == 1, "the dry run left every row in place")


def test_every_refusal_leaves_the_database_intact() -> None:
    prepare_database()
    attempts = [
        lambda: reset_dev(settings(environment="production"), confirm_database=TEST_DB,
                          confirm_phrase=CONFIRM_PHRASE, dry_run=False),
        lambda: reset_dev(settings(), confirm_database="wrong",
                          confirm_phrase=CONFIRM_PHRASE, dry_run=False),
        lambda: reset_dev(settings(), confirm_database=TEST_DB,
                          confirm_phrase="", dry_run=False),
    ]
    for attempt in attempts:
        try:
            attempt()
        except ResetRefused:
            pass
        else:
            raise AssertionError("expected a refusal")
    with connect(url_for(TEST_DB)) as conn:
        rows = conn.execute("SELECT count(*) AS n FROM document").fetchone()["n"]
        marker = schema_marker(conn)
    check(rows == 1 and marker is not None,
          "no refused attempt touched the schema or the rows")


def test_an_exactly_confirmed_target_resets_and_reinitializes() -> None:
    prepare_database()
    with tempfile.TemporaryDirectory() as temp:
        # Sentinels standing in for the things a reset must never reach.
        root = Path(temp)
        source_pdf = root / "report.pdf"
        source_pdf.write_bytes(b"%PDF-1.7\nsource bytes\n")
        config_file = root / "settings.env"
        config_file.write_text("CLAIM_EVIDENCE_EMBED_DIMENSIONS=8\n", encoding="utf-8")
        output_root = write_output_root(root / "extraction", pages=1)
        before = {
            path: path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

        plan = reset_dev(
            settings(),
            confirm_database=TEST_DB,
            confirm_phrase=CONFIRM_PHRASE,
            dry_run=False,
        )
        check(plan.performed and not plan.dry_run, "the confirmed reset ran")
        check(plan.counts["document"] == 1, "it reported what it destroyed")

        after = {
            path: path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
        check(before == after, f"{len(before)} source/config/extraction files are byte-identical")
        check(source_pdf.is_file() and output_root.is_dir(),
              "the source PDF and extraction output still exist")

    with connect(url_for(TEST_DB)) as conn:
        marker = schema_marker(conn)
        rows = conn.execute("SELECT count(*) AS n FROM document").fetchone()["n"]
        audits = conn.execute("SELECT count(*) AS n FROM audit_run").fetchone()["n"]
    check(marker is not None, "the current schema was reinstalled")
    check(rows == 0 and audits == 0, "index rows and audit history are gone")


def main() -> int:
    offline = [
        test_environment_must_be_development,
        test_only_dev_and_test_names_are_accepted,
        test_system_databases_are_never_targets,
        test_production_like_names_are_refused_even_with_a_dev_suffix,
        test_both_confirmations_are_required,
        test_a_running_application_blocks_the_reset,
        test_the_target_never_includes_the_password,
    ]
    for function in offline:
        print(f"\n--- {function.__name__} ---")
        function()

    if not available():
        print("\n[skip] PostgreSQL is not reachable; the destructive checks need it")
        return 0

    for function in (
        test_dry_run_is_the_default_and_changes_nothing,
        test_every_refusal_leaves_the_database_intact,
        test_an_exactly_confirmed_target_resets_and_reinitializes,
    ):
        print(f"\n--- {function.__name__} ---")
        function()

    with psycopg.connect(ADMIN_URL, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{TEST_DB}" WITH (FORCE)')
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
