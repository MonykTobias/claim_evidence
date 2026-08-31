"""Schema initialization: install once, recognize itself, refuse anything else.

Needs the Compose PostgreSQL; skips cleanly when it is unreachable.

    docker compose up -d
    python tests/test_db_init.py
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import psycopg
import pytest

from claim_evidence import Settings
from claim_evidence.db import (
    REQUIRED_COLUMNS,
    REQUIRED_TABLES,
    SCHEMA_PATH,
    SCHEMA_VERSION,
    SchemaMismatchError,
    connect,
    init_schema,
    is_empty,
    missing_objects,
    schema_marker,
    schema_sql_sha256,
)

ADMIN_URL = os.environ.get(
    "CLAIM_EVIDENCE_DATABASE_URL",
    "postgresql://claim_evidence:claim_evidence@localhost:5433/claim_evidence",
)
TEST_DB = "claim_evidence_init_test"
DIMENSIONS = 8


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


def database_url() -> str:
    return ADMIN_URL.rsplit("/", 1)[0] + f"/{TEST_DB}"


def fresh_database() -> None:
    with psycopg.connect(ADMIN_URL, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{TEST_DB}" WITH (FORCE)')
        conn.execute(f'CREATE DATABASE "{TEST_DB}"')


def settings() -> Settings:
    return Settings(database_url=database_url(), embed_dimensions=DIMENSIONS)


def available() -> bool:
    try:
        with psycopg.connect(ADMIN_URL, connect_timeout=3):
            return True
    except psycopg.OperationalError:
        return False


@contextmanager
def empty_database():
    """A freshly created, completely empty database, closed afterwards."""
    fresh_database()
    conn = connect(database_url())
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def initialized_database():
    with empty_database() as conn:
        init_schema(conn, DIMENSIONS)
        yield conn


@pytest.fixture()
def conn():
    if not available():
        pytest.skip("PostgreSQL is not reachable; run `docker compose up -d`")
    with empty_database() as connection:
        yield connection


@pytest.fixture()
def ready(conn):
    init_schema(conn, DIMENSIONS)
    return conn


# --- checks -----------------------------------------------------------------


def test_one_schema_asset() -> None:
    """PD-03: one current schema, not a numbered migration series."""
    files = sorted(p.name for p in SCHEMA_PATH.parent.glob("*.sql"))
    check(files == ["schema.sql"], f"exactly one schema asset is shipped: {files}")
    body = SCHEMA_PATH.read_text(encoding="utf-8")
    check(
        "ADD COLUMN IF NOT EXISTS" not in body,
        "the schema installs a shape rather than migrating an older one",
    )


def test_empty_initialization_records_the_marker(conn) -> None:
    check(is_empty(conn), "the test database starts empty")
    outcome = init_schema(conn, DIMENSIONS)
    check(outcome == "initialized", "an empty database reports that it was initialized")

    marker = schema_marker(conn)
    check(marker["version"] == SCHEMA_VERSION, "the schema version is recorded")
    check(
        marker["schema_sql_sha256"] == schema_sql_sha256(),
        "the schema file's own digest is recorded",
    )
    check(marker["initialized_at"] is not None, "the initialization time is recorded")
    check(missing_objects(conn) == [], "every required table and column exists")


def test_repeat_initialization_is_a_no_op(ready) -> None:
    conn = ready
    before = schema_marker(conn)
    conn.execute(
        "INSERT INTO document (name, identity_key) VALUES ('keep me', 'k1')"
    )
    conn.commit()

    check(init_schema(conn, DIMENSIONS) == "unchanged", "a repeat init is a no-op")
    after = schema_marker(conn)
    check(
        after["initialized_at"] == before["initialized_at"],
        "a no-op init does not rewrite the initialization time",
    )
    rows = conn.execute("SELECT count(*) AS n FROM document").fetchone()["n"]
    check(rows == 1, "a repeat init does not touch stored rows")


def test_a_missing_table_is_refused_read_only(ready) -> None:
    conn = ready
    conn.execute(
        "INSERT INTO document (name, identity_key) VALUES ('keep me', 'k1')"
    )
    conn.execute("DROP TABLE visual_verification")
    conn.commit()
    check(
        missing_objects(conn) == ["table visual_verification"],
        "the sanity list names exactly what is missing",
    )
    try:
        init_schema(conn, DIMENSIONS)
    except SchemaMismatchError as exc:
        check("visual_verification" in str(exc), "the refusal names the missing table")
        check("reset-dev" in str(exc), "the refusal gives the reset/rebuild path")
        check("upgrade" not in str(exc).lower() or "no in-place upgrade" in str(exc),
              "the refusal does not promise an upgrade path")
    else:
        raise AssertionError("a partial schema must be refused")

    rows = conn.execute("SELECT count(*) AS n FROM document").fetchone()["n"]
    check(rows == 1, "a refused init changed nothing")


def test_a_missing_column_is_refused(ready) -> None:
    conn = ready
    conn.execute("ALTER TABLE evidence_unit DROP COLUMN context_key")
    conn.commit()
    check(
        "column evidence_unit.context_key" in missing_objects(conn),
        "a missing required column is detected, not just a missing table",
    )
    try:
        init_schema(conn, DIMENSIONS)
    except SchemaMismatchError as exc:
        check("context_key" in str(exc), "the refusal names the missing column")
    else:
        raise AssertionError("a missing required column must be refused")


def test_a_different_schema_file_is_refused(ready) -> None:
    conn = ready
    conn.execute(
        "UPDATE schema_meta SET schema_sql_sha256 = %s WHERE id = 1", ("0" * 64,)
    )
    conn.commit()
    try:
        init_schema(conn, DIMENSIONS)
    except SchemaMismatchError as exc:
        check(
            "different version of the schema file" in str(exc),
            "a database installed from another schema file is refused",
        )
    else:
        raise AssertionError("a different schema digest must be refused")


def test_an_older_schema_version_is_refused(ready) -> None:
    conn = ready
    conn.execute("UPDATE schema_meta SET version = 5 WHERE id = 1")
    conn.commit()
    try:
        init_schema(conn, DIMENSIONS)
    except SchemaMismatchError as exc:
        check(str(SCHEMA_VERSION) in str(exc), "the refusal names the current version")
        check(
            "reset-dev" in str(exc) and "re-index" in str(exc),
            "the refusal says to reset and re-index rather than to migrate",
        )
    else:
        raise AssertionError("an older recorded version must be refused")


def test_a_foreign_database_is_refused() -> None:
    """Tables but no marker: someone else's database, not ours to install into."""
    fresh_database()
    with connect(database_url()) as conn:
        conn.execute("CREATE TABLE unrelated_application (id integer)")
        conn.commit()
        try:
            init_schema(conn, DIMENSIONS)
        except SchemaMismatchError as exc:
            check(
                "no claim_evidence schema marker" in str(exc),
                "a populated foreign database is refused",
            )
            check(
                "empty database" in str(exc),
                "the refusal offers an empty database as the way forward",
            )
        else:
            raise AssertionError("a foreign database must be refused")
        remaining = conn.execute(
            "SELECT count(*) AS n FROM information_schema.tables"
            " WHERE table_schema = current_schema()"
        ).fetchone()["n"]
        check(remaining == 1, "the refused database still holds only its own table")


def test_the_refusal_never_names_the_database() -> None:
    """A connection string is the caller's; errors travel further than they do."""
    fresh_database()
    with connect(database_url()) as conn:
        init_schema(conn, DIMENSIONS)
        conn.execute("UPDATE schema_meta SET version = 1 WHERE id = 1")
        conn.commit()
        try:
            init_schema(conn, DIMENSIONS)
        except SchemaMismatchError as exc:
            message = str(exc)
            check(TEST_DB not in message, "the refusal does not name the database")
            check("claim_evidence:" not in message, "nor any credential")
            check("localhost" not in message, "nor the host")
        else:
            raise AssertionError("expected a refusal")


def test_required_lists_describe_the_shipped_schema() -> None:
    body = SCHEMA_PATH.read_text(encoding="utf-8")
    for table in REQUIRED_TABLES:
        check(
            f"CREATE TABLE IF NOT EXISTS {table} " in body,
            f"required table {table} is created by the schema file",
        )
    for table, column in REQUIRED_COLUMNS:
        check(column in body, f"required column {table}.{column} appears in the schema")


def main() -> int:
    if not available():
        print("[skip] PostgreSQL is not reachable; run `docker compose up -d`")
        return 0

    for function in (test_one_schema_asset,
                     test_required_lists_describe_the_shipped_schema):
        print(f"\n--- {function.__name__} ---")
        function()

    for function in (test_empty_initialization_records_the_marker,):
        print(f"\n--- {function.__name__} ---")
        with empty_database() as connection:
            function(connection)

    for function in (test_repeat_initialization_is_a_no_op,
                     test_a_missing_table_is_refused_read_only,
                     test_a_missing_column_is_refused,
                     test_a_different_schema_file_is_refused,
                     test_an_older_schema_version_is_refused):
        print(f"\n--- {function.__name__} ---")
        with initialized_database() as connection:
            function(connection)

    for function in (test_a_foreign_database_is_refused,
                     test_the_refusal_never_names_the_database):
        print(f"\n--- {function.__name__} ---")
        function()

    with psycopg.connect(ADMIN_URL, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{TEST_DB}" WITH (FORCE)')
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
