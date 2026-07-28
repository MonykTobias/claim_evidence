"""Manifest validation and evidence-unit construction."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import block, kpi_table, markdown_only_table, write_output_root  # noqa: E402

from claim_evidence.models import EvidenceKind, GeometryPrecision  # noqa: E402
from claim_evidence.source import (  # noqa: E402
    OutputReader,
    OutputValidationError,
    block_text,
    flatten_header,
    page_units,
)

REAL_ROOT = Path(
    r"C:\Users\Tobia\Documents\Tobi&Anna\gw_detector_v2\outputs_full_run\danoneurdaccessible"
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


def expect_error(func, fragment: str, message: str) -> None:
    try:
        func()
    except OutputValidationError as exc:
        check(fragment in str(exc), f"{message} ({exc})")
        return
    raise AssertionError(f"{message}: no OutputValidationError raised")


def test_complete_manifest_validates() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = write_output_root(Path(temp) / "run", pages=3)
        pages = OutputReader(root).validate()
        check([p.page for p in pages] == [1, 2, 3], "pages returned in order")
        check(pages[0].width == 600.0, "page size read from layout map")


def test_missing_manifest_rejected() -> None:
    with tempfile.TemporaryDirectory() as temp:
        expect_error(
            OutputReader(Path(temp)).validate, "missing manifest.json", "no manifest"
        )


def test_incomplete_page_rejected() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = write_output_root(Path(temp) / "run", pages=2, statuses={2: "failed"})
        expect_error(OutputReader(root).validate, "not 'completed'", "incomplete page")


def test_duplicate_page_rejected() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = write_output_root(Path(temp) / "run", pages=2, duplicate_page=1)
        expect_error(OutputReader(root).validate, "duplicate", "duplicate page")


def test_page_gap_rejected() -> None:
    with tempfile.TemporaryDirectory() as temp:
        # Manifest lists 2 pages but the checkpoints say the PDF had 4.
        root = write_output_root(Path(temp) / "run", pages=2, total_pages=4)
        expect_error(OutputReader(root).validate, "missing", "page coverage gap")


def test_missing_artifact_rejected() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = write_output_root(
            Path(temp) / "run", pages=2, drop_artifact=(2, "page.png")
        )
        expect_error(OutputReader(root).validate, "missing page.png", "missing artifact")


def test_stale_page_directory_ignored() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = write_output_root(Path(temp) / "run", pages=2, stale_page_dir="page_0009")
        reader = OutputReader(root)
        pages = reader.validate()
        check(len(pages) == 2, "stale directory not ingested")
        check(any("stale" in w for w in reader.warnings), "stale directory warned about")
        check("page_0009" in reader.skipped, "stale directory reported as skipped")


def test_fingerprint_is_stable_and_content_bound() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = write_output_root(Path(temp) / "run", pages=2)
        first = OutputReader(root).fingerprint()
        check(first == OutputReader(root).fingerprint(), "fingerprint is stable")
        (root / "blocks.jsonl").write_text('{"page": 1}', encoding="utf-8")
        check(OutputReader(root).fingerprint() != first, "content change moves it")

        pdf = Path(temp) / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.7 fixture")
        by_pdf = OutputReader(root).fingerprint(pdf)
        check(by_pdf != first, "a source PDF is a different identity")
        check(by_pdf == OutputReader(root).fingerprint(pdf), "pdf fingerprint stable")


def test_text_full_preferred_over_capped_text() -> None:
    long_text = "Danone " + "x" * 600
    with_full = block(1, 1, long_text, text_full=long_text)
    text, truncated = block_text(with_full)
    check(len(text) == len(long_text), "text_full used when present")
    check(not truncated, "untruncated block not flagged")

    capped = block(1, 1, long_text)
    text, truncated = block_text(capped)
    check(len(text) == 500, "falls back to capped text")
    check(truncated, "capped block flagged as truncated")

    short = block(1, 1, "Short block.")
    check(block_text(short) == ("Short block.", False), "short block is not truncated")


def test_header_flattening() -> None:
    rows = [
        ["", "", "Performance history", "Performance history"],
        ["KPI", "Unit", "2024", "2025"],
    ]
    check(flatten_header(rows, 2, 0) == ["KPI"], "empty header cell dropped")
    check(
        flatten_header(rows, 2, 3) == ["Performance history", "2025"],
        "spanning header flattened into a path",
    )
    check(flatten_header(rows, 2, 9) == [], "out-of-range column is empty")


def _units_for_table(table: dict) -> list:
    with tempfile.TemporaryDirectory() as temp:
        root = write_output_root(
            Path(temp) / "run",
            pages=1,
            blocks=[block(1, 1, "Nature indicators", heading_path=["4.8.2 NATURE INDICATORS"])],
            tables={1: [table]},
        )
        reader = OutputReader(root)
        page = reader.validate()[0]
        return list(page_units(reader, page, reader.blocks_by_page().get(1, [])))


def test_table_value_units_carry_every_region() -> None:
    units = _units_for_table(kpi_table())
    values = [u for u in units if u.kind is EvidenceKind.TABLE_VALUE]
    target = next(u for u in values if u.table_context["value"] == "(40.2) %")
    roles = {r.role for r in target.regions}
    check(roles == {"descriptor", "header", "unit", "value"}, f"four roles cited: {roles}")
    check(target.geometry_precision is GeometryPrecision.CELL, "cell precision")
    check(
        target.table_context["header_path"] == ["Performance history", "2025"],
        "value carries its flattened header path",
    )
    check(target.table_context["unit"] == "Percentage of variation", "unit column read")
    check("(40.2) %" in target.text, "displayed value kept verbatim in the text")

    rows = [u for u in units if u.kind is EvidenceKind.TABLE_ROW]
    check(len(rows) == 2, "one unit per body row")
    check(all(u.heading_path == ["4.8.2 NATURE INDICATORS"] for u in rows), "heading path")


def test_missing_cell_geometry_falls_back_to_the_row() -> None:
    units = _units_for_table(kpi_table())
    target = next(
        u
        for u in units
        if u.kind is EvidenceKind.TABLE_VALUE and u.table_context["value"] == "85.7%"
    )
    check(target.geometry_precision is GeometryPrecision.ROW, "row precision fallback")
    row_region = next(r for r in target.regions if r.role == "value")
    descriptor = next(r for r in target.regions if r.role == "descriptor")
    check(row_region.bbox[0] <= descriptor.bbox[0], "row union starts at the descriptor")
    check(row_region.bbox[2] >= descriptor.bbox[2], "row union spans past it")


def test_markdown_only_table_is_table_precision() -> None:
    units = _units_for_table(markdown_only_table())
    values = [u for u in units if u.kind is EvidenceKind.TABLE_VALUE]
    check(len(values) == 1, "one value parsed from the pipe table")
    check(values[0].table_context["value"] == "(12.0) %", "markdown value parsed")
    check(
        values[0].geometry_precision is GeometryPrecision.TABLE,
        "markdown-only evidence is table precision",
    )


def test_page_markdown_is_not_citable() -> None:
    units = _units_for_table(kpi_table())
    markdown = next(u for u in units if u.kind is EvidenceKind.PAGE_MARKDOWN)
    check(not markdown.citable, "generated page markdown is never citable")
    check(markdown.geometry_precision is GeometryPrecision.PAGE, "page-level geometry")


def test_real_danone_page_359() -> None:
    """Skips cleanly when the completed run is not on this machine."""
    if not (REAL_ROOT / "manifest.json").is_file():
        print("[skip] danone output root not present")
        return
    reader = OutputReader(REAL_ROOT)
    page = next(p for p in reader.validate() if p.page == 359)
    units = list(page_units(reader, page, reader.blocks_by_page().get(359, [])))
    target = next(
        u
        for u in units
        if u.kind is EvidenceKind.TABLE_VALUE and u.table_context["value"] == "(40.2) %"
    )
    check(target.page == 359, "40.2% value found on PDF page 359")
    check(
        {r.role for r in target.regions} == {"descriptor", "header", "unit", "value"},
        "real table value cites all four regions",
    )
    check(
        "Scope 1 & 2" in target.table_context["descriptor"],
        "descriptor is the scope 1 & 2 emissions row",
    )


def main() -> int:
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
