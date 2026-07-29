"""Read a completed ``document_extract`` output root into evidence units.

The manifest is the authoritative page list. Stale ``page_*`` directories left
behind by an earlier partial run are ignored rather than silently folded into
the index, and any page that is not ``completed`` aborts ingestion instead of
producing an index with invisible holes in it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from document_extract.contracts import (
    OPTIONAL_PAGE_ROLES,
    PAGE_ARTIFACT_ROLES,
    ROOT_ARTIFACT_ROLES,
    ContractError,
    validate_run,
)

from .errors import ValidationError
from .models import (
    EvidenceKind,
    EvidenceQuality,
    EvidenceUnit,
    GeometryPrecision,
    Region,
    RegionRole,
)
from .normalize import (
    clean_text,
    normalize_for_match,
    normalize_unit,
    region_from,
    union_bbox,
)

RUN_NAME = ROOT_ARTIFACT_ROLES["run"]
MANIFEST_NAME = ROOT_ARTIFACT_ROLES["manifest"]
BLOCKS_NAME = ROOT_ARTIFACT_ROLES["blocks"]
# Taken from the extraction contract rather than restated here: the producer
# declares which roles exist and what each is called, and a second copy of that
# list in the consumer is exactly how the two drift apart.
REQUIRED_PAGE_ARTIFACTS = tuple(
    name for role, name in PAGE_ARTIFACT_ROLES.items() if role not in OPTIONAL_PAGE_ROLES
)
# Everything a page's evidence, geometry, or visual verification depends on.
# An optional artifact is absorbed as "absent" rather than skipped, so a page
# that gains or loses one still changes the fingerprint.
FINGERPRINTED_ARTIFACTS = tuple(PAGE_ARTIFACT_ROLES.values())
# Block text is capped at 500 characters in older runs; `text_full` is the
# uncapped field and is preferred whenever the producing run emitted it.
LEGACY_TEXT_CAP = 500
NON_CITABLE_KINDS = frozenset({EvidenceKind.PAGE_MARKDOWN})


class OutputValidationError(ValidationError):
    """The output root cannot be indexed as-is.

    A package ValidationError, so a caller maps it to the same non-retryable
    category as any other bad input rather than to an internal fault.
    """


@dataclass(frozen=True)
class PageSource:
    page: int
    page_dir: Path
    width: float
    height: float
    # The manifest's page directory relative to the output root, as posix. This
    # is what gets stored and re-joined later; the absolute `page_dir` is only
    # used to read artifacts during ingestion.
    rel: str = ""

    def artifact(self, name: str) -> Path:
        return self.page_dir / name


class OutputReader:
    """Validating accessor over one completed output root."""

    def __init__(self, output_root: str | Path) -> None:
        self.root = Path(output_root).resolve()
        self.warnings: list[str] = []
        self.skipped: list[str] = []
        self._manifest: list[dict[str, Any]] | None = None
        self._pages: list[PageSource] | None = None
        self._run: dict[str, Any] | None = None

    # --- validation ---------------------------------------------------------

    @property
    def run(self) -> dict[str, Any]:
        """The extraction's own statement of what it produced.

        Required, and required to be the current contract version. There is no
        reader for output written before the contract existed: an old root's
        artifacts may be laid out differently, and guessing is how an index
        ends up citing the wrong page with complete confidence. The answer is
        always to re-extract, and the message says so.
        """
        if self._run is None:
            path = self.root / RUN_NAME
            if not path.is_file():
                raise OutputValidationError(
                    f"{RUN_NAME} is missing from the output root: this extraction "
                    f"predates the current run contract and cannot be indexed. "
                    f"Re-extract the document."
                )
            try:
                self._run = validate_run(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                raise OutputValidationError(f"{RUN_NAME} is not valid JSON") from None
            except ContractError as exc:
                # The contract's own message names the offending key or version
                # and never echoes a value, so it is safe to pass through.
                raise OutputValidationError(f"{RUN_NAME} is not usable: {exc}") from None
        return self._run

    @property
    def manifest(self) -> list[dict[str, Any]]:
        if self._manifest is None:
            path = self.root / MANIFEST_NAME
            if not path.is_file():
                raise OutputValidationError(f"missing {MANIFEST_NAME} in {self.root}")
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list) or not data:
                raise OutputValidationError(f"{path} is not a non-empty page list")
            self._manifest = data
        return self._manifest

    def validate(self) -> list[PageSource]:
        """Return the page list, raising on anything that would corrupt an index."""
        if self._pages is not None:
            return self._pages

        seen: dict[int, dict[str, Any]] = {}
        for entry in self.manifest:
            page = entry.get("page")
            if not isinstance(page, int) or isinstance(page, bool):
                raise OutputValidationError(f"manifest entry without a page: {entry!r}")
            if page < 1:
                raise OutputValidationError(f"manifest page {page} is not a 1-based page")
            if page in seen:
                raise OutputValidationError(f"duplicate manifest entry for page {page}")
            if entry.get("status") != "completed":
                raise OutputValidationError(
                    f"page {page} is {entry.get('status')!r}, not 'completed'"
                )
            if entry.get("failure"):
                raise OutputValidationError(
                    f"page {page} records a failure: {entry['failure']!r}"
                )
            seen[page] = entry

        pages: list[PageSource] = []
        for page in sorted(seen):
            entry = seen[page]
            page_dir = self._contained_dir(page, entry.get("page_dir"))
            if not page_dir.is_dir():
                raise OutputValidationError(f"page {page} has no page directory")
            for name in REQUIRED_PAGE_ARTIFACTS:
                if not (page_dir / name).is_file():
                    raise OutputValidationError(f"page {page} is missing {name}")
            width, height = self._page_size(page_dir)
            pages.append(
                PageSource(
                    page,
                    page_dir,
                    width,
                    height,
                    page_dir.relative_to(self.root).as_posix(),
                )
            )

        # The run contract states which pages were selected. The manifest must
        # cover exactly that range: a partial extraction is legitimate, but an
        # extraction that silently lost pages out of the middle of its own
        # selection is an index with invisible holes in it.
        selection = self.run["selection"]
        expected = set(range(selection["start_page"], selection["end_page"] + 1))
        missing = sorted(expected - seen.keys())
        extra = sorted(seen.keys() - expected)
        if missing or extra:
            raise OutputValidationError(
                f"manifest covers {len(seen)} pages but the run selected "
                f"{selection['page_count']} "
                f"({selection['start_page']}..{selection['end_page']}); "
                f"missing {missing[:10]}, unexpected {extra[:10]}"
            )
        # Page checkpoints carry the same count; a disagreement means the
        # directory mixes two runs.
        counts = {self._total_pages(p.page_dir) for p in pages}
        if counts != {selection["page_count"]}:
            raise OutputValidationError(
                f"page checkpoints disagree with the run selection: "
                f"{sorted(counts)} against {selection['page_count']}"
            )

        stale = sorted(
            p.name
            for p in self.root.glob("page_*")
            if p.is_dir() and p.name not in {s.page_dir.name for s in pages}
        )
        if stale:
            # Never ingested, but a caller should know the root is not clean.
            self.warnings.append(
                f"{len(stale)} stale page directories ignored: {stale[:5]}"
            )
            self.skipped.extend(stale)

        self._pages = pages
        return pages

    def _contained_dir(self, page: int, raw: Any) -> Path:
        """Resolve a manifest page directory, refusing anything outside the root.

        Runs before any artifact is touched, so an absolute path, a ``..``
        traversal, or a symlink pointing elsewhere never reaches a file read.
        The message names the page only: echoing the escaped target back to the
        caller would hand them the filesystem probe they were attempting.
        """
        relative = Path(str(raw or f"page_{page:04d}"))
        if relative.is_absolute() or relative.anchor:
            raise OutputValidationError(f"page {page} has an absolute page_dir")
        resolved = (self.root / relative).resolve()
        if self.root not in resolved.parents:
            raise OutputValidationError(
                f"page {page} has a page_dir outside the output root"
            )
        return resolved

    @staticmethod
    def _page_size(page_dir: Path) -> tuple[float, float]:
        data = json.loads(
            (page_dir / "layout_prompt_map.json").read_text(encoding="utf-8")
        )
        size = data.get("page_size")
        if not size or len(size) != 2:
            raise OutputValidationError(f"{page_dir} has no usable page_size")
        return float(size[0]), float(size[1])

    @staticmethod
    def _total_pages(page_dir: Path) -> int:
        state = json.loads((page_dir / "page_state.json").read_text(encoding="utf-8"))
        total = state.get("total_pages")
        if not isinstance(total, int) or total <= 0:
            raise OutputValidationError(f"{page_dir}/page_state.json has no total_pages")
        return total

    # --- identity -----------------------------------------------------------

    def artifact_digests(self) -> dict[str, str]:
        """SHA-256 of every artifact the run contract declares evidence-bearing.

        Keyed by role and page so the map says *what* changed, not only that
        something did. Hashing content rather than size is the whole point: a
        re-run that improves a table reconstruction or redraws a page image
        without changing its length would otherwise be indistinguishable from
        no change at all, and the better extraction would silently never be
        indexed.

        ``page_image`` counts. It is what visual re-verification crops from, so
        a changed page image changes what a verdict can rest on.

        Only manifest-listed pages contribute, so a stale directory the
        manifest has dropped cannot move the fingerprint. A declared-optional
        artifact that is absent is recorded as ``"absent"`` rather than
        skipped, so a page that gains or loses one is a different build.
        """
        roles = self.run["evidence_bearing_roles"]
        root_roles = self.run["artifacts"]["root"]
        page_roles = self.run["artifacts"]["page"]

        digests: dict[str, str] = {}
        for role in roles:
            if role in root_roles:
                digests[role] = _digest_of(self.root / root_roles[role])
        for page in self.validate():
            for role in roles:
                if role in page_roles:
                    digests[f"{role}@{page.page}"] = _digest_of(
                        page.artifact(page_roles[role])
                    )
        return digests

    def extraction_fingerprint(self) -> str:
        """One value over every declared evidence-bearing artifact's content."""
        return canonical_digest(self.artifact_digests())

    # --- artifacts ----------------------------------------------------------

    def blocks_by_page(self) -> dict[int, list[dict[str, Any]]]:
        path = self.root / BLOCKS_NAME
        by_page: dict[int, list[dict[str, Any]]] = {}
        if not path.is_file():
            self.warnings.append(f"{BLOCKS_NAME} is absent; narrative evidence skipped")
            return by_page
        wanted = {p.page for p in self.validate()}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                block = json.loads(line)
                page = block.get("page")
                if page in wanted:
                    by_page.setdefault(page, []).append(block)
        return by_page

    def table_candidates(self, page: PageSource) -> list[dict[str, Any]]:
        return _load_json(page.artifact("table_candidates.json"), default=[])

    def image_summaries(self, page: PageSource) -> list[dict[str, Any]]:
        path = page.artifact("image_summaries.jsonl")
        if not path.is_file():
            return []
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def page_markdown(self, page: PageSource) -> str:
        return page.artifact("docling_final.md").read_text(encoding="utf-8")


def sha256_file(path: Path) -> str:
    """The file's actual SHA-256, streamed. Verifiable with any standard tool."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    """The one canonical encoding: sorted keys, compact, UTF-8, not escaped.

    Two runs that agree on the facts must produce identical bytes, or a
    fingerprint is a record of formatting rather than of content.
    """
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _digest_of(path: Path) -> str:
    """Content digest, or ``"absent"`` for a declared-optional artifact.

    An absent artifact is recorded rather than skipped: a page that gains or
    loses its image summaries is a different build, and a skipped entry would
    make those two builds identical.
    """
    return sha256_file(path) if path.is_file() else "absent"


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


# --- evidence construction --------------------------------------------------


def block_text(block: dict[str, Any]) -> tuple[str, bool]:
    """Prefer uncapped ``text_full``; report whether only capped text exists."""
    full = block.get("text_full")
    if isinstance(full, str) and full:
        return full, False
    text = block.get("text") or ""
    return text, len(text) >= LEGACY_TEXT_CAP


def narrative_units(page: PageSource, blocks: list[dict[str, Any]]) -> list[EvidenceUnit]:
    units: list[EvidenceUnit] = []
    for block in blocks:
        text, truncated = block_text(block)
        text = clean_text(text)
        if not text or block.get("type") == "picture":
            continue
        bbox = block.get("bbox")
        if not bbox:
            continue
        region = region_from(
            bbox,
            page.width,
            page.height,
            role=RegionRole.CLAIM_TEXT,
            precision=GeometryPrecision.BLOCK,
        )
        heading_path = [clean_text(h) for h in block.get("heading_path") or []]
        units.append(
            EvidenceUnit(
                unit_key=f"p{page.page:04d}:narrative:{block['block_id']}",
                context_key=f"p{page.page:04d}:block:{block['block_id']}",
                page=page.page,
                kind=EvidenceKind.NARRATIVE,
                text=text,
                normalized_text=normalize_for_match(" > ".join(heading_path + [text])),
                quality=EvidenceQuality.DIRECT_TEXT,
                heading_path=heading_path,
                artifact_path=f"{page.rel}/{BLOCKS_NAME}",
                regions=[region],
                geometry_precision=GeometryPrecision.BLOCK,
                truncated_source=truncated,
            )
        )
    return units


def flatten_header(rows: list[list[str]], header_rows: int, column: int) -> list[str]:
    """Collapse a spanning multi-row header into one path for a column.

    Docling repeats a spanned header's text across every column it covers, so
    de-duplicating down the column turns the repeated cells back into a path
    like ``["Performance history", "2025"]``.
    """
    path: list[str] = []
    for row in rows[:header_rows]:
        if column >= len(row):
            continue
        value = clean_text(row[column])
        if value and value not in path:
            path.append(value)
    return path


def _cell_region(
    cell: dict[str, Any] | None,
    page: PageSource,
    role: RegionRole,
) -> Region | None:
    if not cell or not cell.get("bbox"):
        return None
    return region_from(
        cell["bbox"], page.width, page.height, role=role, precision=GeometryPrecision.CELL
    )


def table_units(
    page: PageSource,
    candidate: dict[str, Any],
    heading_path: list[str],
) -> list[EvidenceUnit]:
    """Rows and individual values from one table candidate.

    Prefers the reconstructed grid, which carries per-cell geometry. Candidate
    Markdown is only parsed when no grid exists, and those units are marked as
    table-precision because a Markdown pipe row has no coordinates of its own.
    """
    candidate_id = candidate.get("candidate_id") or "tc"
    table_region = region_from(
        candidate["bbox"],
        page.width,
        page.height,
        role=RegionRole.SUPPORTING_CONTEXT,
        precision=GeometryPrecision.TABLE,
    )
    grid = (candidate.get("stats") or {}).get("grid") or {}
    rows: list[list[str]] = grid.get("rows") or []
    if not rows:
        rows = _rows_from_markdown(candidate.get("markdown") or "")
        header_rows = 1 if rows else 0
        cells: dict[tuple[int, int], dict[str, Any]] = {}
    else:
        header_rows = int(grid.get("header_rows") or 1)
        cells = {(c["r"], c["c"]): c for c in grid.get("cells") or []}
    if not rows or header_rows >= len(rows):
        return []

    num_cols = max(len(r) for r in rows)
    headers = [flatten_header(rows, header_rows, c) for c in range(num_cols)]
    unit_column = next(
        (c for c, h in enumerate(headers) if normalize_for_match(" ".join(h)) == "unit"),
        None,
    )
    artifact = f"{page.rel}/table_candidates.json"
    units: list[EvidenceUnit] = []

    for r in range(header_rows, len(rows)):
        row = rows[r]
        descriptor = clean_text(row[0]) if row else ""
        values = [(c, clean_text(v)) for c, v in enumerate(row) if clean_text(v)]
        if not values:
            continue
        row_regions = [
            region
            for c in range(num_cols)
            if (region := _cell_region(cells.get((r, c)), page, RegionRole.SUPPORTING_CONTEXT))
        ]
        row_union = union_bbox([region.bbox for region in row_regions])
        row_unit = clean_text(row[unit_column]) if unit_column is not None and unit_column < len(row) else ""

        row_text = _row_text(descriptor, row_unit, headers, values, unit_column)
        row_fallback, row_precision = _fallback_regions(
            row_union, table_region, role=RegionRole.SUPPORTING_CONTEXT
        )
        # One key per table row. Its value cells share it, which is what makes
        # "expand this cell to its descriptor, header and unit" a lookup rather
        # than a guess about which rows happen to sit near it.
        row_context = f"p{page.page:04d}:table:{candidate_id}:row:{r}"
        units.append(
            EvidenceUnit(
                unit_key=f"p{page.page:04d}:table_row:{candidate_id}:r{r}",
                context_key=row_context,
                page=page.page,
                kind=EvidenceKind.TABLE_ROW,
                text=row_text,
                normalized_text=normalize_for_match(
                    " > ".join(heading_path + [row_text])
                ),
                quality=EvidenceQuality.DIRECT_TABLE,
                heading_path=heading_path,
                table_context={
                    "candidate_id": candidate_id,
                    "row_index": r,
                    "descriptor": descriptor,
                    "unit": row_unit,
                    "headers": headers,
                    "cells": [v for _, v in values],
                },
                artifact_path=artifact,
                regions=row_regions or row_fallback,
                geometry_precision=(
                    GeometryPrecision.CELL if row_regions else row_precision
                ),
            )
        )

        for c, value in values:
            if c == 0 or c == unit_column:
                continue
            header = headers[c] if c < len(headers) else []
            value_region = _cell_region(cells.get((r, c)), page, RegionRole.VALUE)
            regions: list[Region] = []
            for role, cell in (
                (RegionRole.DESCRIPTOR, cells.get((r, 0))),
                (RegionRole.HEADER, cells.get((header_rows - 1, c))),
                (RegionRole.UNIT, cells.get((r, unit_column)) if unit_column is not None else None),
            ):
                region = _cell_region(cell, page, role)
                if region:
                    regions.append(region)
            if value_region:
                regions.append(value_region)
                precision = GeometryPrecision.CELL
            else:
                # A cell without its own geometry still gets a highlightable
                # region: the row it sits in, and failing that the whole table.
                fallback, precision = _fallback_regions(
                    row_union, table_region, role=RegionRole.VALUE
                )
                regions.extend(fallback)
            text = _value_text(descriptor, header, row_unit, value)
            units.append(
                EvidenceUnit(
                    unit_key=f"p{page.page:04d}:table_value:{candidate_id}:r{r}:c{c}",
                    context_key=row_context,
                    page=page.page,
                    kind=EvidenceKind.TABLE_VALUE,
                    text=text,
                    normalized_text=normalize_for_match(text),
                    quality=EvidenceQuality.DIRECT_TABLE,
                    heading_path=heading_path,
                    table_context={
                        "candidate_id": candidate_id,
                        "row_index": r,
                        "column_index": c,
                        "descriptor": descriptor,
                        "header_path": header,
                        "unit": row_unit or normalize_unit(" ".join(header)) or "",
                        "value": value,
                    },
                    artifact_path=artifact,
                    regions=regions or [table_region],
                    geometry_precision=precision,
                )
            )
    return units


def _fallback_regions(
    row_union: tuple[float, float, float, float] | None,
    table_region: Region,
    *,
    role: RegionRole,
) -> tuple[list[Region], GeometryPrecision]:
    if row_union:
        return (
            [Region(bbox=row_union, role=role, precision=GeometryPrecision.ROW)],
            GeometryPrecision.ROW,
        )
    return [table_region], GeometryPrecision.TABLE


def _row_text(
    descriptor: str,
    unit: str,
    headers: list[list[str]],
    values: list[tuple[int, str]],
    unit_column: int | None,
) -> str:
    parts = [descriptor] if descriptor else []
    if unit:
        parts.append(f"unit: {unit}")
    for column, value in values:
        if column == 0 or column == unit_column:
            continue
        header = " ".join(headers[column]) if column < len(headers) else ""
        parts.append(f"{header}: {value}" if header else value)
    return " | ".join(parts)


def _value_text(descriptor: str, header: list[str], unit: str, value: str) -> str:
    head = " ".join(header)
    parts = [p for p in (descriptor, head) if p]
    label = " | ".join(parts)
    suffix = f" ({unit})" if unit else ""
    return f"{label}{suffix}: {value}" if label else value


def _rows_from_markdown(markdown: str) -> list[list[str]]:
    """Parse a pipe table when the candidate carries no reconstructed grid."""
    rows: list[list[str]] = []
    for line in markdown.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue
        rows.append([clean_text(c.replace("<br>", " ")) for c in cells])
    return rows


def visual_units(
    page: PageSource,
    summaries: list[dict[str, Any]],
    heading_path: list[str],
) -> list[EvidenceUnit]:
    """Chart/infographic regions.

    Summaries are retrieval hints only: the unit is stored at coarse precision
    and cannot support a verdict until the audit re-checks the crop.
    """
    units: list[EvidenceUnit] = []
    for summary in summaries:
        bbox = summary.get("norm_rect") or summary.get("bbox")
        if not bbox:
            continue
        text = clean_text(
            " ".join(
                filter(
                    None,
                    (
                        summary.get("caption") or "",
                        summary.get("summary") or "",
                        summary.get("classification") or "",
                    ),
                )
            )
        )
        index = summary.get("index")
        units.append(
            EvidenceUnit(
                unit_key=f"p{page.page:04d}:visual:{index}",
                context_key=f"p{page.page:04d}:visual:{index}",
                page=page.page,
                kind=EvidenceKind.VISUAL,
                text=text or f"visual region {index} on page {page.page}",
                normalized_text=normalize_for_match(text),
                quality=EvidenceQuality.COARSE_REGION,
                heading_path=heading_path,
                table_context={"rel_path": summary.get("rel_path") or ""},
                artifact_path=f"{page.rel}/image_summaries.jsonl",
                regions=[
                    region_from(
                        bbox,
                        page.width,
                        page.height,
                        role=RegionRole.VISUAL_REGION,
                        precision=GeometryPrecision.CROP,
                    )
                ],
                geometry_precision=GeometryPrecision.CROP,
            )
        )
    return units


def markdown_unit(page: PageSource, markdown: str) -> EvidenceUnit | None:
    text = clean_text(markdown)
    if not text:
        return None
    return EvidenceUnit(
        unit_key=f"p{page.page:04d}:page_markdown",
        page=page.page,
        kind=EvidenceKind.PAGE_MARKDOWN,
        text=text,
        normalized_text=normalize_for_match(text),
        citable=False,
        quality=EvidenceQuality.NONE,
        artifact_path=f"{page.rel}/docling_final.md",
        regions=[
            Region(
                bbox=(0.0, 0.0, 1.0, 1.0),
                role=RegionRole.SUPPORTING_CONTEXT,
                precision=GeometryPrecision.PAGE,
            )
        ],
        geometry_precision=GeometryPrecision.PAGE,
    )


def page_units(
    reader: OutputReader,
    page: PageSource,
    blocks: list[dict[str, Any]],
) -> Iterator[EvidenceUnit]:
    """Every evidence unit for one page, in reading order."""
    units: list[EvidenceUnit] = list(narrative_units(page, blocks))

    heading_by_block = {
        str(b["block_id"]).split("-")[-1]: [clean_text(h) for h in b.get("heading_path") or []]
        for b in blocks
    }
    default_heading = next(
        (h for h in reversed(list(heading_by_block.values())) if h), []
    )
    for candidate in reader.table_candidates(page):
        source_ids = candidate.get("source_block_ids") or []
        heading = next(
            (heading_by_block[s] for s in source_ids if s in heading_by_block),
            default_heading,
        )
        try:
            units.extend(table_units(page, candidate, heading))
        except (KeyError, TypeError, ValueError) as exc:
            reader.warnings.append(
                f"page {page.page} table {candidate.get('candidate_id')} skipped: {exc}"
            )
            reader.skipped.append(f"{page.rel}/table_candidates.json")

    units.extend(visual_units(page, reader.image_summaries(page), default_heading))

    unit = markdown_unit(page, reader.page_markdown(page))
    if unit:
        units.append(unit)

    yield from _in_reading_order(units)


def _in_reading_order(units: list[EvidenceUnit]) -> list[EvidenceUnit]:
    """Number the page's units top to bottom, artifact order breaking ties.

    A deterministic approximation of reading order, not a layout analysis: the
    top edge of a unit's highest region places it, and the order the artifacts
    produced it settles anything at the same height. That is enough for "what
    sits next to this on the page", and it is stable across re-ingestion in a
    way evidence ids are not.

    Generated page Markdown always sorts last. It covers the whole page, so by
    position it would rank first and become every candidate's nearest
    neighbour, which is precisely the thing it must never be.
    """
    def position(entry: tuple[int, EvidenceUnit]) -> tuple[float, int]:
        index, unit = entry
        if unit.kind is EvidenceKind.PAGE_MARKDOWN:
            return (2.0, index)
        top = min((region.bbox[1] for region in unit.regions), default=1.0)
        return (top, index)

    ordered = sorted(enumerate(units), key=position)
    return [
        unit.model_copy(update={"source_order": order})
        for order, (_, unit) in enumerate(ordered)
    ]


__all__ = [
    "FINGERPRINTED_ARTIFACTS",
    "canonical_digest",
    "canonical_json",
    "NON_CITABLE_KINDS",
    "OutputReader",
    "OutputValidationError",
    "PageSource",
    "block_text",
    "flatten_header",
    "markdown_unit",
    "narrative_units",
    "page_units",
    "sha256_file",
    "table_units",
    "visual_units",
]
