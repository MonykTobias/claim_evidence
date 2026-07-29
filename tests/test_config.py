"""Settings validation and the bounded database connection.

Deterministic: psycopg is stubbed for the timeout checks, and the one live
check dials a closed local port rather than a real server.
"""

from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import psycopg  # noqa: E402

from claim_evidence import Settings  # noqa: E402
from claim_evidence.config import DEFAULT_DATABASE_CONNECT_TIMEOUT  # noqa: E402
from claim_evidence.db import connect  # noqa: E402
from claim_evidence.errors import DependencyUnavailableError, ValidationError  # noqa: E402

SECRET_URL = "postgresql://secret_user:hunter2@db.internal.example:5432/claim_evidence"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


class _StubConnect:
    """Records the kwargs psycopg.connect was called with."""

    def __init__(self, raises: BaseException | None = None) -> None:
        self.raises = raises
        self.kwargs: dict = {}

    def __call__(self, conninfo, **kwargs):
        self.kwargs = kwargs
        if self.raises is not None:
            raise self.raises
        raise AssertionError("stub was expected to raise")


def _with_stub(stub: _StubConnect, url: str = SECRET_URL, timeout: float = 7.0):
    original = psycopg.connect
    psycopg.connect = stub  # type: ignore[assignment]
    try:
        return connect(url, timeout)
    finally:
        psycopg.connect = original  # type: ignore[assignment]


def test_default_timeout_is_ten_seconds() -> None:
    check(
        Settings().database_connect_timeout == DEFAULT_DATABASE_CONNECT_TIMEOUT == 10.0,
        "the default connect timeout is 10 seconds",
    )


def test_timeout_reaches_psycopg() -> None:
    stub = _StubConnect(raises=psycopg.OperationalError("timeout expired"))
    try:
        _with_stub(stub, timeout=7.0)
    except DependencyUnavailableError:
        pass
    check(stub.kwargs.get("connect_timeout") == 7, "the configured timeout is passed through")

    stub = _StubConnect(raises=psycopg.OperationalError("timeout expired"))
    try:
        _with_stub(stub, timeout=0.4)
    except DependencyUnavailableError:
        pass
    check(
        stub.kwargs.get("connect_timeout") == 1,
        "a sub-second timeout becomes libpq's smallest whole second",
    )


def test_operational_error_becomes_a_typed_dependency_error() -> None:
    driver_message = (
        f'connection to server at "db.internal.example" (10.1.2.3), port 5432 failed: '
        f"FATAL: password authentication failed for user \"secret_user\" [{SECRET_URL}]"
    )
    stub = _StubConnect(raises=psycopg.OperationalError(driver_message))
    try:
        _with_stub(stub)
    except DependencyUnavailableError as exc:
        message = str(exc)
        check(message == "PostgreSQL is unavailable.", f"safe public message ({message})")
        for leak in ("secret_user", "hunter2", "db.internal.example", "10.1.2.3", "5432"):
            check(leak not in message, f"{leak!r} is not in the public message")
        check(
            isinstance(exc.__cause__, psycopg.OperationalError),
            "the driver error is preserved as __cause__ for the server log",
        )
        return
    raise AssertionError("no DependencyUnavailableError raised")


def test_invalid_timeout_configuration_is_a_validation_error() -> None:
    for value in (0, -1, -0.5):
        try:
            Settings(database_connect_timeout=value)
        except ValidationError as exc:
            check(
                "CLAIM_EVIDENCE_DATABASE_CONNECT_TIMEOUT" in str(exc),
                f"{value} rejected by name ({exc})",
            )
            continue
        raise AssertionError(f"{value} was accepted as a connect timeout")

    original = os.environ.get("CLAIM_EVIDENCE_DATABASE_CONNECT_TIMEOUT")
    os.environ["CLAIM_EVIDENCE_DATABASE_CONNECT_TIMEOUT"] = SECRET_URL
    try:
        Settings.from_env()
    except ValidationError as exc:
        check("hunter2" not in str(exc), "a misplaced value is never echoed back")
    else:
        raise AssertionError("an unparseable timeout was accepted")
    finally:
        if original is None:
            del os.environ["CLAIM_EVIDENCE_DATABASE_CONNECT_TIMEOUT"]
        else:
            os.environ["CLAIM_EVIDENCE_DATABASE_CONNECT_TIMEOUT"] = original


def test_unreachable_port_fails_within_the_timeout() -> None:
    """A closed local port: refused fast, and typed either way."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    # The socket is closed, so nothing is listening on `port`.
    started = time.monotonic()
    try:
        connect(f"postgresql://u:p@127.0.0.1:{port}/none", 3)
    except DependencyUnavailableError as exc:
        elapsed = time.monotonic() - started
        check(elapsed < 10.0, f"an unreachable port fails quickly ({elapsed:.1f}s)")
        check(str(exc) == "PostgreSQL is unavailable.", "and with the safe message")
        return
    raise AssertionError("connecting to a closed port succeeded")


def main() -> int:
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
