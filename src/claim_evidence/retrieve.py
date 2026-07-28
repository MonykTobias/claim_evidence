"""Hybrid retrieval: facts, full-text, and vectors merged by rank fusion.

The three retrievers run independently and are merged with reciprocal-rank
fusion, so a candidate that only one of them finds still surfaces. Exact
numbers, years, units, and scope tokens get an explicit bonus on top, because
a claim audit lives or dies on "40.2" versus "40.3" and embedding similarity
is indifferent to that difference.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import psycopg

from .db import (
    graph_search,
    lexical_search,
    neighbours,
    regions_for,
    vector_search,
)
from .models import (
    Citation,
    EvidenceKind,
    EvidenceMatch,
    EvidenceQuality,
    GeometryPrecision,
    ParsedClaim,
    Region,
)
from .normalize import all_years, content_tokens, normalize_for_match, scope_markers

RRF_K = 60
EXACT_TOKEN_BONUS = 0.5
SCOPE_TOKEN_BONUS = 0.3
_NUMBER_TOKEN = re.compile(r"\d[\d.,]*")


def lexical_query(text: str, key_terms: Sequence[str] = ()) -> str:
    """Build a recall-oriented websearch query.

    ``websearch_to_tsquery`` ANDs bare terms, which makes a whole sentence
    match nothing. Terms are OR'd instead and precision comes from ranking.
    """
    terms = list(dict.fromkeys([*key_terms, *sorted(content_tokens(text))]))
    # A multi-word term must be a quoted phrase; bare words next to OR are
    # parsed as an AND group and quietly drop the whole clause's recall.
    quoted = [f'"{t}"' if " " in t else t for t in terms if t]
    numbers = [f'"{value}"' for value in dict.fromkeys(_exact_numbers(text))]
    parts = [*numbers, *quoted]
    return " OR ".join(parts) if parts else text


def _exact_numbers(text: str) -> list[str]:
    """Numeric tokens with sentence punctuation stripped ("2020." -> "2020")."""
    return [match.strip(".,") for match in _NUMBER_TOKEN.findall(text) if match.strip(".,")]


def exact_tokens(claim: ParsedClaim, claim_text: str) -> set[str]:
    """Tokens whose literal presence is strong evidence of relevance."""
    tokens = set(_exact_numbers(claim_text))
    tokens |= set(all_years(claim_text))
    for period in (claim.reporting_period, claim.baseline_period):
        if period:
            tokens.add(period)
    if claim.value_decimal is not None:
        tokens.add(str(claim.value_decimal))
        tokens.add(str(abs(claim.value_decimal)))
    if claim.unit:
        tokens.add(claim.unit)
    return {t for t in tokens if t}


def fuse(
    ranked: dict[str, list[dict[str, Any]]],
    *,
    claim: ParsedClaim | None = None,
    claim_text: str = "",
) -> list[dict[str, Any]]:
    """Reciprocal-rank fusion plus an exact-token bonus."""
    tokens = exact_tokens(claim, claim_text) if claim else set()
    markers = scope_markers(claim_text) if claim_text else frozenset()

    merged: dict[int, dict[str, Any]] = {}
    for source, rows in ranked.items():
        for position, row in enumerate(rows, start=1):
            evidence_id = int(row["id"])
            entry = merged.setdefault(
                evidence_id,
                {"row": row, "score": 0.0, "lexical_rank": None,
                 "vector_rank": None, "graph_rank": None},
            )
            entry["score"] += 1.0 / (RRF_K + position)
            entry[f"{source}_rank"] = position

    for entry in merged.values():
        text = normalize_for_match(entry["row"]["source_text"])
        if tokens:
            hits = sum(1 for token in tokens if normalize_for_match(token) in text)
            entry["score"] += EXACT_TOKEN_BONUS * hits / len(tokens)
        if markers and scope_markers(entry["row"]["source_text"]) & markers:
            entry["score"] += SCOPE_TOKEN_BONUS

    return sorted(merged.values(), key=lambda e: (-e["score"], int(e["row"]["id"])))


def retrieve(
    conn: psycopg.Connection,
    query_embedding: Sequence[float] | None,
    claim: ParsedClaim,
    claim_text: str,
    *,
    document_ids: Sequence[int] | None = None,
    limit: int = 20,
    pool: int = 60,
) -> list[dict[str, Any]]:
    """Candidates for one claim, best first."""
    ranked: dict[str, list[dict[str, Any]]] = {
        "lexical": lexical_search(
            conn, lexical_query(claim_text, claim.key_terms), document_ids, pool
        )
    }
    if query_embedding is not None:
        ranked["vector"] = vector_search(conn, query_embedding, document_ids, pool)
    ranked["graph"] = graph_search(
        conn,
        metric_terms=sorted(content_tokens(claim.metric or claim_text))[:8],
        reporting_period=claim.reporting_period,
        baseline_period=claim.baseline_period,
        document_ids=document_ids,
        limit=pool,
    )
    return fuse(ranked, claim=claim, claim_text=claim_text)[:limit]


def expand(
    conn: psycopg.Connection, candidates: Sequence[dict[str, Any]], top: int = 5
) -> list[dict[str, Any]]:
    """Pull in each top candidate's page neighbours: table rows, headers, prose.

    A value cell alone rarely proves a claim; the row it sits in and the
    paragraph beside it are what make the qualifiers checkable.
    """
    seen = {int(c["row"]["id"]) for c in candidates}
    extra: list[dict[str, Any]] = []
    for candidate in candidates[:top]:
        for row in neighbours(conn, int(candidate["row"]["id"])):
            evidence_id = int(row["id"])
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            extra.append(
                {
                    "row": row,
                    "score": candidate["score"] * 0.25,
                    "lexical_rank": None,
                    "vector_rank": None,
                    "graph_rank": None,
                    "expanded_from": int(candidate["row"]["id"]),
                }
            )
    return [*candidates, *extra]


def to_citation(
    row: dict[str, Any],
    regions: Sequence[dict[str, Any]],
    *,
    quality: EvidenceQuality | None = None,
) -> Citation:
    kind = EvidenceKind(row["kind"])
    context = row.get("table_context") or {}
    cells = [str(v) for v in (context.get("cells") or []) if v]
    if kind is EvidenceKind.TABLE_VALUE:
        cells = [
            str(context.get("descriptor") or ""),
            " ".join(context.get("header_path") or []),
            str(context.get("unit") or ""),
            str(context.get("value") or ""),
        ]
    return Citation(
        evidence_id=int(row["id"]),
        document_id=int(row["document_id"]),
        document_name=row["document_name"],
        document_sha256=row.get("sha256"),
        source_uri=row.get("source_uri"),
        pdf_page=int(row["pdf_page"]),
        printed_page_label=row.get("printed_page_label"),
        source_kind=kind,
        quality=quality or EvidenceQuality(row["quality"]),
        quote=row["source_text"] if kind is not EvidenceKind.TABLE_VALUE else None,
        table_cells=[c for c in cells if c],
        heading_path=list(row.get("heading_path") or []),
        artifact_path=f"{row['page_dir']}/{row['artifact_path'].split('/')[-1]}",
        regions=[
            Region(
                bbox=(r["left_norm"], r["top_norm"], r["right_norm"], r["bottom_norm"]),
                role=r["role"],
                precision=GeometryPrecision(r["precision"]),
                source_bbox=tuple(r["source_bbox"]) if r.get("source_bbox") else None,
                source_origin=r.get("source_origin"),
            )
            for r in regions
        ],
        geometry_precision=GeometryPrecision(row["geometry_precision"]),
    )


def to_matches(
    conn: psycopg.Connection, candidates: Sequence[dict[str, Any]]
) -> list[EvidenceMatch]:
    ids = [int(c["row"]["id"]) for c in candidates]
    grouped = regions_for(conn, ids)
    return [
        EvidenceMatch(
            citation=to_citation(c["row"], grouped.get(int(c["row"]["id"]), [])),
            text=c["row"]["source_text"],
            lexical_rank=c.get("lexical_rank"),
            vector_rank=c.get("vector_rank"),
            graph_rank=c.get("graph_rank"),
            combined_score=round(c["score"], 6),
        )
        for c in candidates
    ]


__all__ = [
    "EXACT_TOKEN_BONUS",
    "RRF_K",
    "exact_tokens",
    "expand",
    "fuse",
    "lexical_query",
    "retrieve",
    "to_citation",
    "to_matches",
]
