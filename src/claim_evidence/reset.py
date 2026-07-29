"""The guarded development reset.

Development index data is disposable, and rebuilding it is the supported answer
to almost every schema question -- which makes an easy reset command genuinely
useful and genuinely dangerous. Every guard here exists because the same
command, pointed one character differently, would destroy the wrong thing.

What it removes: this database's schema objects, and therefore its index rows,
citations, and audit history. What it never touches: source PDFs, extraction
output directories, archives, and configuration. It issues SQL and reads one
marker file; it does not walk the filesystem at all, so there is no path for it
to delete a source root by mistake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psycopg
from psycopg.conninfo import conninfo_to_dict

from .config import Settings
from .db import connect, init_schema
from .errors import ValidationError

REQUIRED_ENVIRONMENT = "development"
CONFIRM_PHRASE = "RESET-INDEX-AND-AUDITS"
ALLOWED_SUFFIXES = ("_dev", "_test")
# Names that are never a disposable development target, whatever they end with.
FORBIDDEN_NAMES = frozenset({"postgres", "template0", "template1"})
FORBIDDEN_FRAGMENTS = ("prod", "production", "stage", "staging")

# Tables whose contents this reset destroys, counted before anything happens so
# the dry run can say what is at stake in the only units that matter.
COUNTED_TABLES = (
    "document",
    "document_version",
    "evidence_unit",
    "fact",
    "audit_run",
)


class ResetRefused(ValidationError):
    """A guard rejected this reset. Nothing was changed."""


@dataclass(frozen=True)
class ResetPlan:
    """What a reset would do, or has just done, against which target."""

    host: str
    port: str
    user: str
    database: str
    dry_run: bool
    performed: bool = False
    counts: dict[str, int] = field(default_factory=dict)

    def describe(self) -> str:
        """The target and the loss, in the terms the operator must confirm.

        The password is never part of this: it is not read, not stored, and not
        printed, so a pasted dry run cannot leak one.
        """
        rows = ", ".join(f"{n} {t}" for t, n in self.counts.items()) or "nothing"
        verb = "would delete" if self.dry_run else "deleted"
        return (
            f"target  host={self.host} port={self.port} user={self.user} "
            f"database={self.database}\n"
            f"data    {verb}: {rows}\n"
            f"kept    source PDFs, extraction output, archives, and configuration "
            f"are untouched"
        )


def target_of(database_url: str) -> dict[str, str]:
    """Host, port, user, and database name -- never the password."""
    info = conninfo_to_dict(database_url)
    return {
        "host": str(info.get("host") or "localhost"),
        "port": str(info.get("port") or "5432"),
        "user": str(info.get("user") or ""),
        "database": str(info.get("dbname") or ""),
    }


def check_name(database: str) -> None:
    """Refuse any database that is not obviously a disposable one."""
    if not database:
        raise ResetRefused("the configured database URL names no database")
    lowered = database.lower()
    if lowered in FORBIDDEN_NAMES:
        raise ResetRefused(f"{database!r} is a system database and is never a reset target")
    for fragment in FORBIDDEN_FRAGMENTS:
        if fragment in lowered:
            raise ResetRefused(
                f"{database!r} contains {fragment!r}; refusing in case this is not "
                f"a development database"
            )
    if not lowered.endswith(ALLOWED_SUFFIXES):
        raise ResetRefused(
            f"{database!r} does not end in "
            f"{' or '.join(ALLOWED_SUFFIXES)}; only a disposable development or "
            f"test database can be reset"
        )


def app_is_running(settings: Settings) -> bool:
    return Path(settings.app_marker).exists()


def reset_dev(
    settings: Settings,
    *,
    confirm_database: str = "",
    confirm_phrase: str = "",
    dry_run: bool = True,
    conn: psycopg.Connection | None = None,
) -> ResetPlan:
    """Reset one explicitly confirmed disposable database.

    Every guard runs before the connection is used for anything destructive, so
    a refusal leaves the database exactly as it was. ``dry_run`` defaults to
    true: the destructive form has to be asked for.
    """
    target = target_of(settings.database_url)

    if settings.environment != REQUIRED_ENVIRONMENT:
        raise ResetRefused(
            f"CE_ENVIRONMENT must be {REQUIRED_ENVIRONMENT!r} to reset a database; "
            f"it is {settings.environment or 'unset'}"
        )
    check_name(target["database"])
    if confirm_database != target["database"]:
        raise ResetRefused(
            "the confirmation does not name the configured database; nothing was reset"
        )
    if confirm_phrase != CONFIRM_PHRASE:
        raise ResetRefused(
            f"confirmation phrase must be exactly {CONFIRM_PHRASE!r}; nothing was reset"
        )
    if app_is_running(settings):
        raise ResetRefused(
            "the local application is running (its marker file exists); stop it "
            "before resetting, so a live process cannot write into a database "
            "that is being dropped"
        )

    owned = conn is None
    connection = conn or connect(settings.database_url, settings.database_connect_timeout)
    try:
        counts = _counts(connection)
        if dry_run:
            return ResetPlan(**target, dry_run=True, counts=counts)
        # One statement, one transaction: the schema either goes away and comes
        # back or neither happens. `public` is dropped rather than the database
        # itself, because dropping a database means connecting to another one
        # and that is a second target to get wrong.
        with connection.transaction():
            connection.execute("DROP SCHEMA public CASCADE")
            connection.execute("CREATE SCHEMA public")
        init_schema(connection, settings.embed_dimensions)
        return ResetPlan(**target, dry_run=False, performed=True, counts=counts)
    finally:
        if owned:
            connection.close()


def _counts(conn: psycopg.Connection) -> dict[str, int]:
    """How many rows the reset would destroy, per table that has one.

    A table that does not exist contributes nothing rather than raising: the
    point is to tell the operator what is at stake, and a half-installed
    database is exactly when they most want to be told.
    """
    counts: dict[str, int] = {}
    for table in COUNTED_TABLES:
        try:
            with conn.transaction():
                row = conn.execute(f"SELECT count(*) AS n FROM {table}").fetchone()
        except psycopg.errors.UndefinedTable:
            continue
        counts[table] = int(row["n"])
    return counts


__all__ = [
    "ALLOWED_SUFFIXES",
    "CONFIRM_PHRASE",
    "FORBIDDEN_FRAGMENTS",
    "FORBIDDEN_NAMES",
    "REQUIRED_ENVIRONMENT",
    "ResetPlan",
    "ResetRefused",
    "app_is_running",
    "check_name",
    "reset_dev",
    "target_of",
]
