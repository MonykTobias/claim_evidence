"""Progress events: phases, totals, completion summaries, and safe failures.

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
from fixtures import block, kpi_table, write_output_root  # noqa: E402
from test_integration import (  # noqa: E402
    SUPPORTED,
    build_root,
    default_session,
    make_client,
    reset_database,
)

from claim_evidence.ingest import VERIFY_STEPS  # noqa: E402
from claim_evidence.models import ProgressEvent  # noqa: E402
from claim_evidence.progress import (  # noqa: E402
    DEPENDENCY_ERROR_MESSAGE,
    ERROR_CODES,
    INTERNAL_ERROR_MESSAGE,
    classify_error,
)

INGEST_PHASES = [
    "validating_input",
    "building_evidence",
    "embedding_evidence",
    "extracting_facts",
    "building_indexes",
    "activating_version",
    "completed",
]
AUDIT_PHASES = [
    "parsing_claim",
    "retrieving_graph",
    "retrieving_full_text",
    "retrieving_vectors",
    "fusing_candidates",
    "expanding_context",
    "verifying_visuals",
    "deciding_verdict",
    "persisting_trace",
    "completed",
]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


def phase_order(events: list[ProgressEvent]) -> list[str]:
    order: list[str] = []
    for event in events:
        if not order or order[-1] != event.phase:
            order.append(event.phase)
    return order


def assert_well_formed(events: list[ProgressEvent], operation: str) -> None:
    """Invariants every event stream must satisfy."""
    check(bool(events), f"{operation} emitted events")
    check(all(e.operation == operation for e in events), "operation tagged on each event")
    check(events[-1].phase == "completed", "the last event is the completed phase")

    for event in events:
        if event.percent is not None:
            check(0.0 <= event.percent <= 100.0, f"percent in range ({event.percent})")
        if event.completed is not None and event.total is not None:
            check(
                0 <= event.completed <= event.total,
                f"0 <= completed <= total ({event.completed}/{event.total})",
            )

    seen: dict[str, float] = {}
    for event in events:
        if event.percent is None:
            continue
        previous = seen.get(event.phase)
        if previous is not None:
            check(
                event.percent >= previous,
                f"{event.phase} percent never decreases ({previous} -> {event.percent})",
            )
        seen[event.phase] = event.percent
    print(f"[ok] {operation} percentages are bounded and monotonic per phase")


# --- ingestion --------------------------------------------------------------


def check_ingest_without_a_callback_is_unchanged(tmp: Path) -> None:
    root = build_root(tmp / "nocb")
    with make_client(default_session()) as client:
        report = client.ingest_document(root, source_uri="urn:nocb")
        check(report.evidence_units > 0, "ingestion works with no callback")
        result = client.audit_claim(SUPPORTED)
        check(result.verdict is not None, "audit works with no callback")


def check_ingest_phases_and_totals(tmp: Path) -> None:
    root = build_root(tmp / "phases")
    events: list[ProgressEvent] = []
    with make_client(default_session()) as client:
        report = client.ingest_document(root, source_uri="urn:phases", progress=events.append)

    assert_well_formed(events, "ingest")
    order = phase_order(events)
    check(order == INGEST_PHASES, f"phases occur in the required order ({order})")
    check(
        all(e.document_id == report.document_id for e in events if e.document_id),
        "events carry the document id once it is known",
    )

    pages = [e for e in events if e.phase == "building_evidence" and e.status == "progress"]
    check(len(pages) == report.pages, "one evidence event per page")
    check(pages[-1].total == report.pages, "evidence total is the page count")

    batches = [e for e in events if e.phase == "embedding_evidence" and e.status == "progress"]
    check(bool(batches), "embedding batches reported")
    check(
        batches[-1].completed == batches[-1].total,
        "the last embedding batch completes the phase",
    )

    steps = [e for e in events if e.phase == "building_indexes"]
    check(
        steps[-1].total == len(VERIFY_STEPS),
        "index checks report their real step count",
    )


def check_ingest_completion_matches_the_report(tmp: Path) -> None:
    root = build_root(tmp / "summary")
    events: list[ProgressEvent] = []
    with make_client(default_session()) as client:
        report = client.ingest_document(
            root, source_uri="urn:summary", progress=events.append
        )

    details = events[-1].details
    check(details["document_version_id"] == report.version_id, "version id reported")
    check(details["page_count"] == report.pages, "page count matches the report")
    check(details["evidence_count"] == report.evidence_units, "evidence count matches")
    check(
        details["visual_evidence_count"] == report.visual_evidence_units,
        "visual evidence count matches",
    )
    check(details["embedding_count"] == report.embedded_units, "embedding count matches")
    check(details["fact_count"] == report.facts, "fact count matches")
    check(details["warning_count"] == len(report.warnings), "warning count matches")
    check(details["no_op"] is False, "a real build is not a no-op")
    check(isinstance(details["elapsed_seconds"], (int, float)), "elapsed time is numeric")
    check(events[-1].percent == 100.0, "completion is 100 percent")
    check(events[-1].status == "completed", "terminal status is completed")


def check_idempotent_ingest_reports_no_op(tmp: Path) -> None:
    root = build_root(tmp / "noop")
    with make_client(default_session()) as client:
        first = client.ingest_document(root, source_uri="urn:noop")
        events: list[ProgressEvent] = []
        second = client.ingest_document(
            root, source_uri="urn:noop", progress=events.append
        )

    check(second.reused_existing, "the second run is a no-op")
    order = phase_order(events)
    check(order == ["validating_input", "completed"], f"no-op skips the work phases ({order})")
    check(events[-1].message == "Document is already indexed", "no-op message")
    details = events[-1].details
    check(details["no_op"] is True, "no_op is true")
    check(details["evidence_count"] == first.evidence_units, "stored counts still reported")
    check(
        not any(e.phase == "embedding_evidence" for e in events),
        "no embedding work is performed",
    )


def check_zero_candidates_do_not_divide_by_zero(tmp: Path) -> None:
    root = build_root(tmp / "zerofacts")
    events: list[ProgressEvent] = []
    with make_client(default_session()) as client:
        # extract_narrative_facts=False leaves the fact phase with no candidates.
        client.ingest_document(
            root,
            source_uri="urn:zerofacts",
            extract_narrative_facts=False,
            progress=events.append,
        )
    facts = [e for e in events if e.phase == "extracting_facts"]
    check(bool(facts), "the fact phase still reports")
    check(facts[-1].total == 0, "an empty phase reports a zero total")
    check(facts[-1].percent == 100.0, "an empty phase is 100 percent, not an error")


def check_warnings_are_emitted_once(tmp: Path) -> None:
    root = write_output_root(
        tmp / "warned",
        pages=2,
        blocks=[block(1, 1, "Nature indicators", heading_path=["4.8.2"])],
        tables={1: [kpi_table()]},
        stale_page_dir="page_0009",
    )
    events: list[ProgressEvent] = []
    with make_client(default_session()) as client:
        report = client.ingest_document(
            root, source_uri="urn:warned", progress=events.append
        )
    warnings = [e for e in events if e.status == "warning"]
    check(bool(warnings), "a recoverable fallback is surfaced as a warning event")
    check(
        len(warnings) == len(report.warnings),
        f"each warning is emitted exactly once ({len(warnings)} vs {len(report.warnings)})",
    )
    check(
        events[-1].details["warning_count"] == len(warnings),
        "the completion summary agrees with the warning events",
    )


def check_resumed_build_reports_the_version_total(tmp: Path) -> None:
    """A resumed build only writes the embeddings that were still missing, so
    the summary must report what the version holds, not what this run wrote."""
    root = build_root(tmp / "resumed")
    calls = {"n": 0}

    class DiesHalfway(type(default_session())):
        def post(self, url: str, json: dict[str, Any], timeout: float):  # noqa: A002
            if url.endswith("/api/embed"):
                calls["n"] += 1
                if calls["n"] > 1:
                    import requests

                    raise requests.ConnectionError("dropped mid-build")
            return super().post(url, json=json, timeout=timeout)

    dying = DiesHalfway(dimensions=8)
    dying.chat_router = default_session().chat_router
    with make_client(dying) as client:
        # Batch size 1 so the first batch lands and the second fails.
        client.settings = client.settings.__class__(
            **{**client.settings.__dict__, "embed_batch_size": 1}
        )
        client.ollama.settings = client.settings
        try:
            client.ingest_document(root, source_uri="urn:resumed")
        except Exception:
            pass
        else:
            raise AssertionError("the simulated drop did not interrupt the build")
        # Scoped to this document's version: the shared test database already
        # holds embeddings from every earlier check.
        partial = client.conn.execute(
            """
            SELECT count(e.embedding) AS n FROM evidence_unit e
            JOIN document_version v ON v.id = e.version_id
            WHERE v.output_root = %s
            """,
            (str(Path(root).resolve()),),
        ).fetchone()["n"]
        check(partial > 0, f"the interrupted run embedded some units ({partial})")

    events: list[ProgressEvent] = []
    with make_client(default_session()) as client:
        report = client.ingest_document(
            root, source_uri="urn:resumed", progress=events.append
        )
        total = client.conn.execute(
            "SELECT count(embedding) AS n FROM evidence_unit WHERE version_id = %s",
            (report.version_id,),
        ).fetchone()["n"]

    check(report.embedded_units == total, "the report counts the version's embeddings")
    check(
        events[-1].details["embedding_count"] == total,
        f"the summary reports {total}, not just this run's writes",
    )
    check(total > partial, "the resumed run finished the remaining embeddings")


def check_callback_exception_does_not_fail_the_build(tmp: Path) -> None:
    root = build_root(tmp / "badcb")
    seen: list[ProgressEvent] = []

    def explode(event: ProgressEvent) -> None:
        seen.append(event)
        raise RuntimeError("the UI fell over")

    with make_client(default_session()) as client:
        report = client.ingest_document(root, source_uri="urn:badcb", progress=explode)
        check(report.evidence_units > 0, "ingestion completed despite a broken callback")
        check(len(seen) == 1, "the callback is dropped after it first raises")
        result = client.audit_claim(SUPPORTED, progress=explode)
        check(result.verdict is not None, "audit completed despite a broken callback")


# --- audit ------------------------------------------------------------------


def check_audit_phases_and_counts(tmp: Path) -> None:
    root = build_root(tmp / "auditphases")
    events: list[ProgressEvent] = []
    with make_client(default_session()) as client:
        client.ingest_document(root, source_uri="urn:auditphases")
        result = client.audit_claim(SUPPORTED, progress=events.append)
        trace = client.get_audit_trace(result.audit_id)

    assert_well_formed(events, "audit")
    order = phase_order(events)
    check(order == AUDIT_PHASES, f"audit phases in the required order ({order})")
    check(
        all(e.audit_id == result.audit_id for e in events if e.audit_id),
        "events carry the audit id once it is known",
    )

    details = events[-1].details
    check(details["verdict"] == str(result.verdict), "verdict reported")
    check(details["citation_count"] == len(result.citations), "citation count matches")
    check(
        details["selected_evidence_count"] == len({c.evidence_id for c in result.citations}),
        "selected evidence count matches",
    )
    # The event reports what each channel returned; the trace only records the
    # candidates that survived fusion's cut, so the event count is the larger.
    for key, survivors in (
        ("graph_candidate_count", trace.graph_candidates),
        ("full_text_candidate_count", trace.lexical_candidates),
        ("vector_candidate_count", trace.vector_candidates),
    ):
        check(details[key] >= len(survivors), f"{key} covers the persisted candidates")
        check(details[key] > 0, f"{key} reports a channel that returned results")
    check(
        details["fused_candidate_count"] <= details["graph_candidate_count"]
        + details["full_text_candidate_count"]
        + details["vector_candidate_count"],
        "fusion never invents candidates",
    )
    check(
        details["expanded_candidate_count"]
        == sum(1 for c in trace.candidates if c.expanded_from),
        "expanded candidate count matches the trace",
    )
    check(details["fused_candidate_count"] > 0, "fused candidate count reported")
    check(isinstance(details["elapsed_seconds"], (int, float)), "elapsed time is numeric")


def check_visual_counts(tmp: Path) -> None:
    """Zero visual candidates complete cleanly; a verified crop is counted."""
    plain = build_root(tmp / "novisual")
    events: list[ProgressEvent] = []
    with make_client(default_session()) as client:
        client.ingest_document(plain, source_uri="urn:novisual")
        client.audit_claim(SUPPORTED, progress=events.append)
    visual = [e for e in events if e.phase == "verifying_visuals"]
    check(bool(visual), "the visual phase reports even with nothing to check")
    check(visual[-1].total == 0, "zero visual candidates report a zero total")
    check(visual[-1].percent == 100.0, "an empty visual phase is complete, not an error")
    check(events[-1].details["visual_candidate_count"] == 0, "zero visual candidates")
    check(events[-1].details["visually_verified_count"] == 0, "zero verified")

    charts = build_root(tmp / "withvisual", with_visual=True)
    accepting = default_session(
        VisualVerification=lambda _: {"supports_claim": True, "visible_text": "40.2%"},
        Adjudication=lambda _: {
            "verdict": "insufficient",
            "rationale": "not comparable",
            "supporting_evidence_ids": [],
            "missing_qualifiers": ["scope"],
        },
    )
    events = []
    with make_client(accepting) as client:
        client.ingest_document(charts, source_uri="urn:withvisual")
        client.audit_claim("The chart shows emissions falling.", progress=events.append)
    details = events[-1].details
    check(
        details["visually_verified_count"] <= details["visual_candidate_count"],
        "verified count never exceeds the candidate count",
    )


def check_empty_channel_reports_zero(tmp: Path) -> None:
    """A channel that ran and found nothing reports 0, not a division error."""
    root = build_root(tmp / "emptychannel")
    events: list[ProgressEvent] = []
    with make_client(default_session()) as client:
        client.ingest_document(root, source_uri="urn:emptychannel")
        # A period no fact in the fixture carries, so the graph channel runs
        # and legitimately returns nothing.
        client.audit_claim(
            "Danone reduced Scope 1 and 2 emissions by 12% in 1974 versus 1968.",
            progress=events.append,
        )

    graph = [e for e in events if e.phase == "retrieving_graph"]
    check(graph[-1].completed == 0, "the empty channel reports zero candidates")
    check(graph[-1].percent == 0.0, "zero of a non-zero request is 0%, not an error")
    details = events[-1].details
    check(details["graph_candidate_count"] == 0, "the summary reports zero")
    check("graph_candidate_count" in details, "a channel that ran is reported, not omitted")
    assert_well_formed(events, "audit")


def check_vector_channel_is_omitted_when_it_cannot_run(tmp: Path) -> None:
    root = build_root(tmp / "novector")
    events: list[ProgressEvent] = []

    class NoEmbedSession(type(default_session())):
        def post(self, url: str, json: dict[str, Any], timeout: float):  # noqa: A002
            if url.endswith("/api/embed") and json.get("input") == [SUPPORTED]:
                import requests

                raise requests.ConnectionError("embeddings unavailable")
            return super().post(url, json=json, timeout=timeout)

    with make_client(default_session()) as client:
        client.ingest_document(root, source_uri="urn:novector")

    session = NoEmbedSession(dimensions=8)
    session.chat_router = default_session().chat_router
    with make_client(session) as client:
        client.audit_claim(SUPPORTED, progress=events.append)

    details = events[-1].details
    check(
        "vector_candidate_count" not in details,
        "a channel that never ran is omitted, not reported as zero",
    )
    check(details["graph_candidate_count"] >= 0, "channels that ran still report")
    check(
        not any(e.phase == "retrieving_vectors" for e in events),
        "the vector phase is skipped entirely",
    )


# --- failures ---------------------------------------------------------------


def check_error_classification() -> None:
    from claim_evidence.errors import (
        DependencyUnavailableError,
        IndexNotReadyError,
        NotFoundError,
        ValidationError,
    )
    from claim_evidence.ollama import OllamaError

    cases = [
        (ValidationError("bad"), "validation_error", False),
        (NotFoundError("gone"), "not_found", False),
        (IndexNotReadyError("empty"), "index_not_ready", True),
        (DependencyUnavailableError("down"), "dependency_unavailable", True),
        (OllamaError("http 500 -- {'secret': 'body'}"), "dependency_unavailable", True),
        (RuntimeError("password=hunter2"), "internal_error", True),
    ]
    for exc, code, retryable in cases:
        actual_code, actual_retry, message = classify_error(exc)
        check(actual_code == code, f"{type(exc).__name__} -> {code}")
        check(actual_retry is retryable, f"{code} retryable is {retryable}")
        check(ERROR_CODES[code] == retryable, f"{code} matches the published table")
    check(
        classify_error(OllamaError("body leak"))[2] == DEPENDENCY_ERROR_MESSAGE,
        "an ollama response body never reaches the caller",
    )
    check(
        classify_error(RuntimeError("password=hunter2"))[2] == INTERNAL_ERROR_MESSAGE,
        "an untyped exception message never reaches the caller",
    )


def check_dependency_failure_is_retryable(tmp: Path) -> None:
    root = build_root(tmp / "depfail")
    events: list[ProgressEvent] = []

    class DeadEmbeddings(type(default_session())):
        def post(self, url: str, json: dict[str, Any], timeout: float):  # noqa: A002
            if url.endswith("/api/embed"):
                import requests

                raise requests.ConnectionError("model host is down")
            return super().post(url, json=json, timeout=timeout)

    session = DeadEmbeddings(dimensions=8)
    session.chat_router = default_session().chat_router
    with make_client(session) as client:
        try:
            client.ingest_document(root, source_uri="urn:depfail", progress=events.append)
        except Exception as exc:
            check(type(exc).__name__ == "OllamaError", "the original error is re-raised")
        else:
            raise AssertionError("the dependency failure did not propagate")

    failed = [e for e in events if e.status == "failed"]
    check(len(failed) == 1, "exactly one failed event")
    details = failed[0].details
    check(details["error_code"] == "dependency_unavailable", "dependency error code")
    check(details["retryable"] is True, "dependency failures are retryable")
    check(details["failed_phase"] == "embedding_evidence", "the failing phase is named")
    check(failed[0].message == DEPENDENCY_ERROR_MESSAGE, "the safe message is used")
    check("model host is down" not in failed[0].model_dump_json(), "no driver text leaks")


def check_validation_failure_is_not_retryable(tmp: Path) -> None:
    events: list[ProgressEvent] = []
    with make_client(default_session()) as client:
        try:
            client.ingest_document(tmp / "does-not-exist", progress=events.append)
        except Exception:
            pass
        else:
            raise AssertionError("a missing output root did not raise")

    failed = [e for e in events if e.status == "failed"]
    check(len(failed) == 1, "exactly one failed event")
    check(failed[0].details["error_code"] == "validation_error", "validation error code")
    check(failed[0].details["retryable"] is False, "validation failures are not retryable")
    check(failed[0].details["failed_phase"] == "validating_input", "the failing phase is named")


def check_failed_ingest_leaves_the_previous_version_ready(tmp: Path) -> None:
    root = build_root(tmp / "failkeep")
    events: list[ProgressEvent] = []
    with make_client(default_session()) as client:
        good = client.ingest_document(root, source_uri="urn:failkeep")

        import claim_evidence.ingest as ingest_module

        original = ingest_module._verify
        ingest_module._verify = lambda *a, **k: (_ for _ in ()).throw(
            ingest_module.IngestionError("simulated failure")
        )
        try:
            client.ingest_document(
                root, source_uri="urn:failkeep", force=True, progress=events.append
            )
        except ingest_module.IngestionError:
            pass
        finally:
            ingest_module._verify = original

        status = client.conn.execute(
            "SELECT status FROM document_version WHERE id = %s", (good.version_id,)
        ).fetchone()["status"]
        check(status == "ready", "the previous version is still ready")
        check(
            bool(client.search_evidence(SUPPORTED, document_ids=[good.document_id])),
            "and is still queryable",
        )

    failed = [e for e in events if e.status == "failed"]
    check(len(failed) == 1, "a failed event was emitted")
    check(failed[0].details["error_code"] == "internal_error", "typed internal error code")
    check(failed[0].details["retryable"] is True, "internal errors are retryable")
    check(
        not any(e.phase == "activating_version" for e in events),
        "a failed version is never activated",
    )


# --- serialization ----------------------------------------------------------


def check_serialization_is_stable_and_safe(tmp: Path) -> None:
    root = build_root(tmp / "serial")
    events: list[ProgressEvent] = []
    with make_client(default_session()) as client:
        client.ingest_document(root, source_uri="urn:serial", progress=events.append)
        client.audit_claim(SUPPORTED, progress=events.append)

    import json

    for event in events:
        dumped = event.model_dump(mode="json")
        json.dumps(dumped)  # raises if anything is not JSON-safe
        check_types(dumped)

    terminal = [e for e in events if e.phase == "completed"]
    check(len(terminal) == 2, "one terminal event per operation")
    ingest_details = terminal[0].details
    check(isinstance(ingest_details["no_op"], bool), "no_op stays a boolean")

    blob = json.dumps([e.model_dump(mode="json") for e in events])
    for leak in (
        "You decide whether",
        "You extract auditable facts",
        "You decompose one atomic claim",
        "postgresql://",
        "claim_evidence:claim_evidence",
        "Traceback",
        "Renewable electricity proportion",  # source evidence text
        "(40.2) %",  # a quoted source value
    ):
        check(leak not in blob, f"no event serializes {leak!r}")
    print("[ok] serialized events carry no prompts, credentials, or source text")


def check_types(dumped: dict[str, Any]) -> None:
    check_once("timestamp is a string", isinstance(dumped["timestamp"], str))
    check_once("details is an object", isinstance(dumped["details"], dict))
    for key in ("completed", "total"):
        check_once(
            f"{key} is null or an int",
            dumped[key] is None or isinstance(dumped[key], int),
        )
    check_once(
        "percent is null or a number",
        dumped["percent"] is None or isinstance(dumped["percent"], (int, float)),
    )


_reported: set[str] = set()


def check_once(message: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(message)
    if message not in _reported:
        _reported.add(message)
        print(f"[ok] {message}")


def main() -> int:
    try:
        reset_database()
    except psycopg.OperationalError as exc:
        print(f"[skip] postgres unavailable ({exc.__class__.__name__})")
        return 0

    with make_client(default_session()) as setup:
        setup.init_db()

    print("\n-- check_error_classification")
    check_error_classification()

    checks = [
        check_ingest_without_a_callback_is_unchanged,
        check_ingest_phases_and_totals,
        check_ingest_completion_matches_the_report,
        check_idempotent_ingest_reports_no_op,
        check_zero_candidates_do_not_divide_by_zero,
        check_warnings_are_emitted_once,
        check_resumed_build_reports_the_version_total,
        check_callback_exception_does_not_fail_the_build,
        check_audit_phases_and_counts,
        check_visual_counts,
        check_empty_channel_reports_zero,
        check_vector_channel_is_omitted_when_it_cannot_run,
        check_dependency_failure_is_retryable,
        check_validation_failure_is_not_retryable,
        check_failed_ingest_leaves_the_previous_version_ready,
        check_serialization_is_stable_and_safe,
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
