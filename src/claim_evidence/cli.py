"""argparse CLI: db init, ingest, search, audit."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .audit import AuditError
from .client import ClaimEvidence
from .ingest import IngestionError
from .models import ClaimResult, EvidenceMatch, IngestReport
from .ollama import OllamaError
from .source import OutputValidationError

USER_ERRORS = (OutputValidationError, IngestionError, AuditError, OllamaError)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="claim-evidence",
        description="Audit claims against source-bound evidence from a document extraction.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    db = sub.add_parser("db", help="database maintenance").add_subparsers(
        dest="db_command", required=True
    )
    db.add_parser("init", help="create or update the schema")
    db.add_parser("documents", help="list ready documents")

    ingest = sub.add_parser("ingest", help="index a completed output root")
    ingest.add_argument("output_root")
    ingest.add_argument("--pdf", dest="source_pdf", help="source PDF for SHA-256 identity")
    ingest.add_argument("--source-uri", dest="source_uri")
    ingest.add_argument(
        "--no-narrative-facts",
        action="store_true",
        help="index evidence only; skip the LLM narrative fact extractor",
    )

    search = sub.add_parser("search", help="hybrid evidence search")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=20)
    search.add_argument("--document-id", type=int, action="append", dest="document_ids")
    search.add_argument("--json", action="store_true")

    audit = sub.add_parser("audit", help="audit one atomic claim")
    audit.add_argument("claim")
    audit.add_argument("--limit", type=int, default=20)
    audit.add_argument("--document-id", type=int, action="append", dest="document_ids")
    audit.add_argument("--json", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with ClaimEvidence.from_env() as client:
        if args.command == "db":
            if args.db_command == "init":
                client.init_db()
                print("schema ready")
            else:
                for row in client.documents():
                    print(f"{row['id']}\t{row['name']}\tversion {row['version_id']}")
            return 0

        if args.command == "ingest":
            report = client.ingest_document(
                args.output_root,
                source_pdf=args.source_pdf,
                source_uri=args.source_uri,
                extract_narrative_facts=not args.no_narrative_facts,
            )
            print(_format_ingest(report))
            return 0

        if args.command == "search":
            matches = client.search_evidence(
                args.query, document_ids=args.document_ids, limit=args.limit
            )
            print(
                json.dumps([m.model_dump(mode="json") for m in matches], indent=2)
                if args.json
                else _format_matches(matches)
            )
            return 0

        result = client.audit_claim(
            args.claim, document_ids=args.document_ids, limit=args.limit
        )
        print(
            json.dumps(result.model_dump(mode="json"), indent=2)
            if args.json
            else _format_result(result)
        )
        return 0


def _format_ingest(report: IngestReport) -> str:
    lines = [
        f"document {report.document_id} version {report.version_id} [{report.status}]",
        f"  fingerprint    {report.fingerprint[:16]}...",
        f"  pages          {report.pages}",
        f"  evidence units {report.evidence_units} ({report.embedded_units} embedded)",
        f"  facts          {report.facts}",
    ]
    if report.reused_existing:
        lines.append("  unchanged: existing ready version reused")
    for warning in report.warnings[:10]:
        lines.append(f"  warning: {warning}")
    if report.rejected_facts:
        lines.append(f"  rejected facts: {len(report.rejected_facts)}")
    return "\n".join(lines)


def _format_matches(matches: Sequence[EvidenceMatch]) -> str:
    if not matches:
        return "no matches"
    lines = []
    for match in matches:
        cite = match.citation
        lines.append(
            f"[{cite.evidence_id}] p.{cite.pdf_page} {cite.source_kind} "
            f"({cite.quality}, {cite.geometry_precision}) score={match.combined_score:.4f}"
        )
        lines.append(f"      {match.text[:180]}")
    return "\n".join(lines)


def _format_result(result: ClaimResult) -> str:
    lines = [
        f"verdict:  {result.verdict}",
        f"quality:  {result.evidence_quality}",
        f"rationale: {result.rationale}",
    ]
    if result.missing_qualifiers:
        lines.append(f"missing:  {', '.join(result.missing_qualifiers)}")
    for cite in result.citations:
        regions = ", ".join(
            f"{r.role}[{r.bbox[0]:.3f},{r.bbox[1]:.3f},{r.bbox[2]:.3f},{r.bbox[3]:.3f}]"
            for r in cite.regions
        )
        lines.append(
            f"  - {cite.document_name} p.{cite.pdf_page} ({cite.quality}, "
            f"{cite.geometry_precision})"
        )
        if cite.table_cells:
            lines.append(f"      cells: {' | '.join(cite.table_cells)}")
        elif cite.quote:
            lines.append(f"      quote: {cite.quote[:180]}")
        lines.append(f"      regions: {regions}")
    return "\n".join(lines)


def cli() -> None:
    try:
        raise SystemExit(main())
    except USER_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    cli()
