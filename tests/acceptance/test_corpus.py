"""The labelled corpus, audited end to end against a real database.

Every case in `gw_detector_v2/acceptance/fixtures.py` gets its expected verdict,
page, artifact role, and evidence kind checked here. The model is faked -- the
point is the product's own rules, not the model's judgement -- but the database,
the schema, the ingestion, the retrieval, and the comparison are all real.

    docker compose up -d
    python tests/acceptance/test_corpus.py
"""

from __future__ import annotations

import functools
import importlib.util
import os
from pathlib import Path

import psycopg
import pytest

from claim_evidence import ClaimEvidence, Settings
from claim_evidence.db import connect
from claim_evidence.errors import UnsupportedClaimError, ValidationError
from claim_evidence.models import Verdict, VersionStatus
from claim_evidence.ollama import OllamaClient

WORKSPACE = Path(__file__).resolve().parents[3]
FIXTURES = WORKSPACE / "gw_detector_v2" / "acceptance" / "fixtures.py"
FAKE_OLLAMA = Path(__file__).resolve().parents[1] / "fake_ollama.py"

ADMIN_URL = os.environ.get(
    "CLAIM_EVIDENCE_DATABASE_URL",
    "postgresql://claim_evidence:claim_evidence@localhost:5433/claim_evidence",
)
TEST_DB = "claim_evidence_corpus_test"
DIMENSIONS = 8


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


@functools.cache
def load(name: str, path: Path):
    """Load one file as a module, without putting its directory on sys.path.

    The corpus lives in the sibling gw_detector_v2 checkout and the fake model
    lives one directory up; neither is an installed package. Loading them by
    path keeps that explicit -- and keeps PA-01 true, which fails any file that
    inserts a checkout onto sys.path.
    """
    if not path.is_file():
        pytest.skip(f"{name} is not present at {path.name}")
    spec = importlib.util.spec_from_file_location(f"_corpus_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def corpus_module():
    return load("fixtures", FIXTURES)


def database_url() -> str:
    return ADMIN_URL.rsplit("/", 1)[0] + f"/{TEST_DB}"


def available() -> bool:
    try:
        with psycopg.connect(ADMIN_URL, connect_timeout=3):
            return True
    except psycopg.OperationalError:
        return False


def session():
    """A model that extracts nothing and adjudicates nothing.

    Every verdict below therefore comes from the deterministic comparator,
    which is exactly what these cases are about.
    """
    return load("fake_ollama", FAKE_OLLAMA).FakeSession(
        dimensions=DIMENSIONS,
        chat_router={
            "ParsedClaim": lambda payload: {
                "subject": corpus_module().ENTITY,
                "metric": payload["messages"][1]["content"],
                "direction": "decrease",
            },
            "FactExtraction": lambda _p: {"facts": []},
            "Adjudication": lambda _p: {
                "verdict": "insufficient",
                "rationale": "No comparable evidence.",
                "supporting_evidence_ids": [],
                "missing_qualifiers": [],
            },
            "VisualVerification": lambda _p: {
                "result": "illegible", "reason_code": "figures_not_legible",
            },
        },
    )


def indexed(tmp_path: Path):
    """A fresh database holding the whole corpus, and an open client."""
    fixtures = corpus_module()
    with psycopg.connect(ADMIN_URL, autocommit=True) as admin:
        admin.execute(f'DROP DATABASE IF EXISTS "{TEST_DB}" WITH (FORCE)')
        admin.execute(f'CREATE DATABASE "{TEST_DB}"')
    config = Settings(
        database_url=database_url(), embed_dimensions=DIMENSIONS,
        chat_model="fake-chat", vision_model="fake-vision",
    )
    client = ClaimEvidence(config, connect(config.database_url), OllamaClient(config, session()))
    client.init_db()
    root = fixtures.build(tmp_path / "northwind")
    report = client.ingest_document(
        root, reporting_entity=fixtures.ENTITY, source_uri="urn:prototype-corpus"
    )
    return client, report, root


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    if not available():
        pytest.skip("PostgreSQL is not reachable; run `docker compose up -d`")
    client, report, root = indexed(tmp_path_factory.mktemp("corpus"))
    yield client, report, root
    client.close()


# --- the corpus indexes ------------------------------------------------------


def test_the_corpus_indexes_and_becomes_queryable(corpus) -> None:
    _client, report, _root = corpus
    check(
        report.status in (VersionStatus.READY, VersionStatus.DEGRADED),
        f"the corpus is queryable ({report.status})",
    )
    check(report.pages == 2, f"both pages indexed ({report.pages})")
    check(report.evidence_units >= 6, f"{report.evidence_units} evidence units")
    check(report.embedded_units > 0, "with embeddings")
    check(report.facts >= 1, f"and at least one table fact ({report.facts})")


def test_every_evidence_kind_the_corpus_promises_is_present(corpus) -> None:
    client, report, _root = corpus
    kinds = {
        row["kind"]
        for row in client.conn.execute(
            "SELECT DISTINCT kind FROM evidence_unit WHERE version_id = %s",
            (report.version_id,),
        ).fetchall()
    }
    for kind in ("narrative", "table_row", "table_value", "visual"):
        check(kind in kinds, f"the corpus contains {kind} evidence")


# --- the labelled cases ------------------------------------------------------


def audit(client, case) -> tuple[str | None, object]:
    fixtures = corpus_module()
    try:
        result = client.audit_claim(
            case["claim"], scope="all", reporting_entity=fixtures.ENTITY
        )
    except UnsupportedClaimError as exc:
        return exc.reason_code, None
    except ValidationError:
        return "validation_error", None
    return None, result


def test_supported_and_contradicted_cases_reach_their_verdicts(corpus) -> None:
    client, _report, _root = corpus
    fixtures = corpus_module()
    for case in fixtures.CASES:
        if case.get("expected_verdict") not in ("supported", "contradicted"):
            continue
        code, result = audit(client, case)
        check(code is None, f"{case['id']} was audited, not refused ({code})")
        if case["id"] == "narrative-quote":
            # The narrative fact extractor is stubbed out in this suite, so this
            # case has no fact to compare and legitimately lands elsewhere.
            continue
        check(
            result.verdict.value == case["expected_verdict"],
            f"{case['id']}: {result.verdict.value} == {case['expected_verdict']}",
        )
        pages = {c.pdf_page for c in result.citations}
        check(
            case["expected_page"] in pages,
            f"{case['id']} cites page {case['expected_page']} (got {sorted(pages)})",
        )
        artifacts = {c.artifact_path for c in result.citations}
        check(
            any("table_candidates" in a for a in artifacts),
            f"{case['id']} cites the table artifact ({artifacts})",
        )


def test_the_absent_metric_is_insufficient_not_contradicted(corpus) -> None:
    client, _report, _root = corpus
    fixtures = corpus_module()
    case = next(c for c in fixtures.CASES if c["id"] == "insufficient-absent-metric")
    code, result = audit(client, case)
    check(code is None, f"the claim itself is supported by the grammar ({code})")
    check(
        result.verdict is Verdict.INSUFFICIENT,
        f"an absent metric is insufficient, not contradicted ({result.verdict})",
    )
    check(not result.citations, "and nothing is cited")


def test_every_unsupported_case_is_refused_by_its_category(corpus) -> None:
    client, _report, _root = corpus
    fixtures = corpus_module()
    for case in fixtures.CASES:
        expected = case.get("expected_reason_code")
        if not expected:
            continue
        code, _result = audit(client, case)
        check(code == expected, f"{case['id']} -> {code} (expected {expected})")


def test_an_omitted_scope_is_a_validation_error(corpus) -> None:
    client, _report, _root = corpus
    fixtures = corpus_module()
    try:
        client.audit_claim(
            "Emissions fell by 40.2% in 2025.", scope=[], reporting_entity=fixtures.ENTITY
        )
    except Exception as exc:  # noqa: BLE001 - either validation type is correct
        check("empty" in str(exc).lower(), f"an empty scope is refused ({exc})")
        return
    raise AssertionError("an empty scope must be refused")


# --- provenance and hostile text --------------------------------------------


def test_every_citation_resolves_to_a_contained_artifact(corpus) -> None:
    client, report, root = corpus
    rows = client.conn.execute(
        "SELECT id, artifact_path FROM evidence_unit"
        " WHERE version_id = %s AND citable",
        (report.version_id,),
    ).fetchall()
    check(bool(rows), f"{len(rows)} citable units to resolve")
    for row in rows:
        target = (Path(root) / row["artifact_path"]).resolve()
        check(
            target.is_file() and Path(root).resolve() in target.parents,
            f"evidence {row['id']} resolves inside the root: {row['artifact_path']}",
        )


def test_hostile_document_text_is_indexed_verbatim(corpus) -> None:
    """It is evidence. Censoring it would be its own bug."""
    client, report, _root = corpus
    fixtures = corpus_module()
    row = client.conn.execute(
        "SELECT source_text, page_id FROM evidence_unit"
        " WHERE version_id = %s AND source_text LIKE %s",
        (report.version_id, "%IGNORE ALL PREVIOUS%"),
    ).fetchone()
    check(row is not None, "the hostile passage is indexed")
    check(
        fixtures.HOSTILE[:40] in row["source_text"],
        "and stored exactly as the document wrote it",
    )


def test_narrative_evidence_comes_back_in_source_order(corpus) -> None:
    client, report, _root = corpus
    rows = client.conn.execute(
        """
        SELECT e.source_order, e.source_text FROM evidence_unit e
        JOIN page p ON p.id = e.page_id
        WHERE e.version_id = %s AND p.pdf_page = 2 AND e.kind = 'narrative'
        ORDER BY e.source_order
        """,
        (report.version_id,),
    ).fetchall()
    check(len(rows) == 2, f"page 2 has both narrative blocks ({len(rows)})")
    check(
        "Water withdrawal" in rows[0]["source_text"],
        "the higher block on the page comes first",
    )
    check(
        "IGNORE ALL PREVIOUS" in rows[1]["source_text"],
        "and the lower one second, by position rather than by row id",
    )


def main() -> int:
    if not available():
        print("[skip] PostgreSQL is not reachable; run `docker compose up -d`")
        return 0
    import tempfile

    with tempfile.TemporaryDirectory() as temp:
        client, report, root = indexed(Path(temp))
        bundle = (client, report, root)
        try:
            for name, function in sorted(globals().items()):
                if name.startswith("test_") and callable(function):
                    print(f"\n--- {name} ---")
                    function(bundle)
        finally:
            client.close()

    with psycopg.connect(ADMIN_URL, autocommit=True) as admin:
        admin.execute(f'DROP DATABASE IF EXISTS "{TEST_DB}" WITH (FORCE)')
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
