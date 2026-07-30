"""What the CLI prints when things go wrong, and what it never prints.

Default CLI output is a public surface: it is pasted into chat windows and bug
reports. No database and no model -- every failure here is injected.
"""

from __future__ import annotations

import contextlib
import io
import os
from unittest import mock

import pytest

from claim_evidence import cli as cli_module
from claim_evidence.audit import AuditError
from claim_evidence.db import SchemaMismatchError
from claim_evidence.errors import (
    DependencyUnavailableError,
    NotFoundError,
    UnsupportedClaimError,
    ValidationError,
)
from claim_evidence.ollama import OllamaError

# Things that must never appear in default CLI output, each standing for a
# class of leak rather than one string.
SENTINELS = {
    "model reply": "secret-sentinel-chain-of-thought",
    "credential": "hunter2",
    "connection string": "postgresql://ce:hunter2@db.internal:5432/prod",
    "server path": r"C:\Users\Tobia\Documents\secret\report.pdf",
    "source text": "Danone reduced Scope 1 and 2 emissions by 40.2%",
    "traceback": "Traceback (most recent call last)",
}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


def run_cli(exc: BaseException, *, debug: bool = False) -> tuple[int, str, str]:
    """Run `cli()` with `main()` replaced by one that raises, and capture both streams."""
    out, err = io.StringIO(), io.StringIO()
    environment = {cli_module.DEBUG_VARIABLE: "1"} if debug else {}
    with mock.patch.dict(os.environ, environment, clear=False):
        if not debug:
            os.environ.pop(cli_module.DEBUG_VARIABLE, None)
        with mock.patch.object(cli_module, "main", side_effect=exc):
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                try:
                    cli_module.cli()
                except SystemExit as exit_code:
                    code = int(exit_code.code or 0)
    return code, out.getvalue(), err.getvalue()


# --- what is never printed --------------------------------------------------


@pytest.mark.parametrize(
    "label,exc",
    [
        (
            "an Ollama error carrying the model's own reply",
            OllamaError(
                "/api/chat returned: {'thinking': "
                "'secret-sentinel-chain-of-thought'}"
            ),
        ),
        (
            "a wrapped adjudication failure",
            AuditError(
                "adjudication failed: /api/chat returned "
                "secret-sentinel-chain-of-thought"
            ),
        ),
        (
            "a driver error carrying a credential",
            RuntimeError(
                'connection to "db.internal" failed: password "hunter2" rejected'
            ),
        ),
        (
            "an unexpected bug holding source text",
            KeyError("Danone reduced Scope 1 and 2 emissions by 40.2%"),
        ),
        (
            "an error holding a server path",
            OSError(r"cannot open C:\Users\Tobia\Documents\secret\report.pdf"),
        ),
    ],
)
def test_default_output_leaks_nothing(label, exc) -> None:
    code, out, err = run_cli(exc)
    assert code == 1, label
    combined = out + err
    for name, sentinel in SENTINELS.items():
        assert sentinel not in combined, f"{label} leaked the {name}"
    assert "error [" in err, f"{label} still reports a category"


def test_a_failure_names_a_stable_code_and_a_safe_message() -> None:
    _code, _out, err = run_cli(OllamaError("secret-sentinel-chain-of-thought"))
    check("error [dependency_unavailable]" in err, f"the category is named ({err!r})")
    check(
        "A required service was unavailable." in err,
        "with a sentence written for a person",
    )
    check("this may succeed on a retry" in err, "and says whether a retry might help")


def test_the_cause_is_available_but_has_to_be_asked_for() -> None:
    _code, _out, quiet = run_cli(OllamaError("secret-sentinel-chain-of-thought"))
    check(
        "CLAIM_EVIDENCE_DEBUG=1" in quiet,
        "the default output says where the cause can be found",
    )
    check(
        "secret-sentinel-chain-of-thought" not in quiet,
        "and does not print it",
    )
    _code, _out, verbose = run_cli(
        OllamaError("secret-sentinel-chain-of-thought"), debug=True
    )
    check(
        "secret-sentinel-chain-of-thought" in verbose,
        "with the flag set, the raw cause is printed for local debugging",
    )
    check("Traceback" in verbose, "as a full traceback")


# --- what is printed --------------------------------------------------------


def test_the_packages_own_typed_errors_are_quoted() -> None:
    """Their messages are written for public consumption; that is the point."""
    for exc, expected in (
        (ValidationError("scope is required"), "scope is required"),
        (NotFoundError("no document with id 7"), "no document with id 7"),
        (
            UnsupportedClaimError(
                "version 1 compares exact values only", reason_code="approximate_claim"
            ),
            "version 1 compares exact values only",
        ),
        (
            DependencyUnavailableError("PostgreSQL is unavailable."),
            "PostgreSQL is unavailable.",
        ),
    ):
        _code, _out, err = run_cli(exc)
        check(expected in err, f"{type(exc).__name__} message is shown: {expected!r}")


def test_a_schema_mismatch_explains_the_way_forward() -> None:
    _code, _out, err = run_cli(
        SchemaMismatchError(
            "the target database does not match the current schema; reset the "
            "development database with `claim-evidence db reset-dev` and re-index"
        )
    )
    check("reset-dev" in err, "the refusal keeps its actionable guidance")
    check("postgresql://" not in err, "and still names no connection string")


def test_a_successful_run_exits_zero_and_says_nothing_on_stderr() -> None:
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(cli_module, "main", return_value=0):
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                cli_module.cli()
            except SystemExit as exit_code:
                code = int(exit_code.code or 0)
    stderr = err.getvalue()
    check(code == 0, "a successful run exits 0")
    check(stderr.strip() == "", f"and writes nothing to stderr ({stderr!r})")


# --- required arguments -----------------------------------------------------


def test_ingest_and_audit_require_the_reporting_entity() -> None:
    parser = cli_module.build_parser()
    for argv in (
        ["ingest", "some/output/root"],
        ["audit", "a claim", "--all-documents"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)
    check(True, "both entry points refuse to run without --entity")


def test_audit_scope_must_be_stated() -> None:
    parser = cli_module.build_parser()
    both = parser.parse_args(
        ["audit", "a claim", "--entity", "Danone S.A.", "--all-documents",
         "--document-id", "3"]
    )
    check(
        both.all_documents and both.document_ids == [3],
        "argparse accepts both flags; main() is what refuses the ambiguity",
    )
    neither = parser.parse_args(["audit", "a claim", "--entity", "Danone S.A."])
    check(
        not neither.all_documents and not neither.document_ids,
        "and neither flag leaves the scope unstated rather than defaulted",
    )


def main() -> int:
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not callable(function):
            continue
        marks = getattr(function, "pytestmark", [])
        cases = [m.args[1] for m in marks if m.name == "parametrize"]
        print(f"\n--- {name} ---")
        if cases:
            for label, exc in cases[0]:
                function(label, exc)
                print(f"[ok] {label}")
        else:
            function()
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
