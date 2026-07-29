"""Synthetic output roots so the deterministic tests need no real extraction."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from document_extract.contracts import (
    COORDINATE_CONVENTION,
    EVIDENCE_BEARING_ROLES,
    PAGE_ARTIFACT_ROLES,
    ROOT_ARTIFACT_ROLES,
    RUN_CONTRACT,
    RUN_CONTRACT_VERSION,
)

PAGE_W = 600.0
PAGE_H = 800.0


def run_contract(
    *,
    numbers: list[int],
    selected_count: int | None = None,
    source_page_count: int | None = None,
    source_sha256: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """A valid ``run.json`` payload for a synthetic output root.

    ``selected_count`` is how many pages the run *selected*, which is not
    always how many the manifest ended up listing -- that difference is what
    the incomplete-range tests are about.
    """
    start = numbers[0]
    count = selected_count or len(numbers)
    end = start + count - 1
    total = max(source_page_count or end, end)
    payload: dict[str, Any] = {
        "contract": RUN_CONTRACT,
        "contract_version": RUN_CONTRACT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "filename": "fixture.pdf",
            "sha256": source_sha256
            or hashlib.sha256(str(numbers).encode()).hexdigest(),
            "page_count": total,
        },
        "selection": {
            "start_page": start,
            "end_page": end,
            "page_count": count,
            "complete_document": start == 1 and end == total,
        },
        "coordinate_convention": COORDINATE_CONVENTION,
        "artifacts": {
            "root": dict(ROOT_ARTIFACT_ROLES),
            "page": dict(PAGE_ARTIFACT_ROLES),
        },
        "evidence_bearing_roles": list(EVIDENCE_BEARING_ROLES),
        "settings": {
            "dpi": 200,
            "refine_mode": "auto",
            "visual_values_mode": "enforce",
            "vlm_model": "fixture-vlm",
            "num_ctx": 16384,
        },
        "pages": {"completed": len(numbers), "failed": 0},
        "warnings": {"total": 0, "categories": {}, "stale_page_dirs": []},
    }
    for path, value in overrides.items():
        target = payload
        parts = path.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    return payload


def write_output_root(
    root: Path,
    *,
    pages: int = 1,
    blocks: list[dict[str, Any]] | None = None,
    tables: dict[int, list[dict[str, Any]]] | None = None,
    total_pages: int | None = None,
    images: dict[int, list[dict[str, Any]]] | None = None,
    statuses: dict[int, str] | None = None,
    drop_artifact: tuple[int, str] | None = None,
    duplicate_page: int | None = None,
    stale_page_dir: str | None = None,
    page_numbers: list[int] | None = None,
    page_dirs: dict[int, str] | None = None,
    page_totals: dict[int, int] | None = None,
    run: dict[str, Any] | None = None,
    drop_run: bool = False,
) -> Path:
    """Build a minimal but structurally valid output root.

    ``page_numbers`` writes an arbitrary selected range (``[10, 11, 12]``)
    instead of ``1..pages``; ``page_dirs`` overrides the manifest's page_dir
    value without moving the directory it was written to, which is how the
    containment tests point a manifest entry outside the root.
    """
    numbers = page_numbers or list(range(1, pages + 1))
    root.mkdir(parents=True, exist_ok=True)
    statuses = statuses or {}
    manifest = []
    for page in numbers:
        name = f"page_{page:04d}"
        page_dir = root / name
        page_dir.mkdir(exist_ok=True)
        (page_dir / "docling_final.md").write_text(
            f"# Page {page}\n\nGenerated page context.\n", encoding="utf-8"
        )
        (page_dir / "layout_prompt_map.json").write_text(
            json.dumps({"page_size": [PAGE_W, PAGE_H], "blocks": []}), encoding="utf-8"
        )
        _write_page_png(page_dir / "page.png")
        (page_dir / "page_state.json").write_text(
            json.dumps(
                {
                    "page": page,
                    "total_pages": (page_totals or {}).get(
                        page, total_pages or len(numbers)
                    ),
                }
            ),
            encoding="utf-8",
        )
        (page_dir / "table_candidates.json").write_text(
            json.dumps((tables or {}).get(page, [])), encoding="utf-8"
        )
        page_images = (images or {}).get(page, [])
        if page_images:
            (page_dir / "image_summaries.jsonl").write_text(
                "\n".join(json.dumps(i) for i in page_images), encoding="utf-8"
            )
        if drop_artifact and drop_artifact[0] == page:
            (page_dir / drop_artifact[1]).unlink()
        manifest.append(
            {
                "page": page,
                "page_dir": (page_dirs or {}).get(page, name),
                "status": statuses.get(page, "completed"),
                "failure": None,
            }
        )
    if duplicate_page is not None:
        manifest.append(dict(manifest[numbers.index(duplicate_page)]))
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "blocks.jsonl").write_text(
        "\n".join(json.dumps(b) for b in (blocks or [])), encoding="utf-8"
    )
    if not drop_run:
        payload = run if run is not None else run_contract(
            numbers=numbers, selected_count=total_pages or len(numbers)
        )
        (root / "run.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    if stale_page_dir:
        (root / stale_page_dir).mkdir(exist_ok=True)
    return root


def _write_page_png(path: Path) -> None:
    """A real (if boring) page image, so crop verification has something to open."""
    from PIL import Image

    Image.new("RGB", (int(PAGE_W), int(PAGE_H)), "white").save(path)


def image_summary(page: int, index: int, summary: str) -> dict[str, Any]:
    return {
        "page": page,
        "index": index,
        "rel_path": f"images/picture_p{page:04d}_i{index:03d}.png",
        "norm_rect": [0.1, 0.55, 0.7, 0.8],
        "caption": "",
        "summary": summary,
        "classification": "chart",
    }


def block(
    page: int,
    index: int,
    text: str,
    *,
    text_full: str | None = None,
    heading_path: list[str] | None = None,
    kind: str = "paragraph",
    bbox: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "block_id": f"p{page:04d}-b{index:04d}",
        "doc_ref": f"#/texts/{index}",
        "page": page,
        "bbox": bbox or {"l": 60.0, "t": 700.0, "r": 540.0, "b": 660.0, "origin": "BOTTOMLEFT"},
        "type": kind,
        "heading_path": heading_path or [],
        "text": text[:500],
        "image_path": None,
        "caption": "",
    }
    if text_full is not None:
        row["text_full"] = text_full
    return row


def cell(
    r: int,
    c: int,
    text: str,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    header: bool = False,
) -> dict[str, Any]:
    return {
        "r": r,
        "c": c,
        "text": text,
        "bbox": (
            {"l": bbox[0], "t": bbox[1], "r": bbox[2], "b": bbox[3], "origin": "TOPLEFT"}
            if bbox
            else None
        ),
        "column_header": header,
    }


def kpi_table() -> dict[str, Any]:
    """A two-header-row table shaped like the real emissions factsheet."""
    rows = [
        ["", "", "Performance history", "Performance history"],
        ["Key Performance Indicators (KPIs)", "Unit", "2024", "2025"],
        [
            "Scope 1 & 2 energy and industry emissions (market-based) vs. 2020",
            "Percentage of variation",
            "(34.5) %",
            "(40.2) %",
        ],
        ["Renewable electricity proportion in the energy mix", "Percentage", "85.7%", "94.2%"],
    ]
    cells = [
        cell(0, 2, "Performance history", bbox=(200, 100, 340, 110), header=True),
        cell(0, 3, "Performance history", bbox=(200, 100, 340, 110), header=True),
        cell(1, 0, "Key Performance Indicators (KPIs)", bbox=(50, 115, 170, 125), header=True),
        cell(1, 1, "Unit", bbox=(200, 115, 240, 125), header=True),
        cell(1, 2, "2024", bbox=(260, 115, 285, 125), header=True),
        cell(1, 3, "2025", bbox=(300, 115, 325, 125), header=True),
        cell(2, 0, rows[2][0], bbox=(50, 380, 170, 400)),
        cell(2, 1, rows[2][1], bbox=(200, 380, 240, 400)),
        cell(2, 2, "(34.5) %", bbox=(260, 380, 285, 390)),
        cell(2, 3, "(40.2) %", bbox=(300, 380, 325, 390)),
        cell(3, 0, rows[3][0], bbox=(50, 410, 170, 430)),
        cell(3, 1, rows[3][1], bbox=(200, 410, 240, 430)),
        # 2024 renewable electricity has no cell geometry, exercising the
        # row-union fallback that the real run needs for 472 cells.
        cell(3, 2, "85.7%"),
        cell(3, 3, "94.2%", bbox=(300, 410, 325, 420)),
    ]
    return {
        "candidate_id": "tc001",
        "kind": "docling_table",
        "bbox": [0.08, 0.12, 0.92, 0.86],
        "source_block_ids": ["b0001"],
        "markdown": "| ignored |\n|---|\n",
        "stats": {"grid": {"rows": rows, "num_cols": 4, "header_rows": 2, "cells": cells}},
    }


def markdown_only_table() -> dict[str, Any]:
    """A candidate with no reconstructed grid, only Markdown."""
    return {
        "candidate_id": "tc002",
        "kind": "markdown_table",
        "bbox": [0.1, 0.5, 0.9, 0.7],
        "source_block_ids": [],
        "markdown": (
            "| Metric | 2025 |\n"
            "|---|---|\n"
            "| Water withdrawal intensity | (12.0) % |\n"
        ),
        "stats": {},
    }
