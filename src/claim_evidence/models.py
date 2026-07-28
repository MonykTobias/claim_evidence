"""Public result types and the internal evidence/fact records they wrap.

Everything crossing the package boundary is a Pydantic model so an external
agent gets validated data and `model_dump()` JSON for free. Model responses are
parsed through the same models, which is what makes a hallucinated field a
validation error instead of a verdict.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _coerce_displayed_value(value: Any) -> Any:
    """Read a displayed value string into a Decimal.

    Models answer with the figure as the page prints it -- "-40.2%",
    "(40.2) %", "1,044" -- none of which is valid JSON number syntax. Parsing
    it exactly the way a table cell is parsed is faithful to the source; it is
    the quote check, not the number format, that guards against invention.
    """
    if isinstance(value, str):
        from .normalize import parse_value  # local: `normalize` imports this module

        parsed, _ = parse_value(value)
        return parsed
    return value


class EvidenceKind(StrEnum):
    NARRATIVE = "narrative"
    TABLE_ROW = "table_row"
    TABLE_VALUE = "table_value"
    VISUAL = "visual"
    PAGE_MARKDOWN = "page_markdown"


class EvidenceQuality(StrEnum):
    """What a citation actually proves. Deliberately not a confidence score."""

    DIRECT_TEXT = "direct_text"
    DIRECT_TABLE = "direct_table"
    VERIFIED_VISUAL = "verified_visual"
    COARSE_REGION = "coarse_region"
    NONE = "none"


class RegionRole(StrEnum):
    """What a region encloses. One closed vocabulary, so a renderer can style
    a value cell differently from the descriptor without guessing."""

    CLAIM_TEXT = "claim_text"
    DESCRIPTOR = "descriptor"
    HEADER = "header"
    UNIT = "unit"
    VALUE = "value"
    SUPPORTING_CONTEXT = "supporting_context"
    VISUAL_REGION = "visual_region"
    UNKNOWN = "unknown"


# Roles written by earlier builds, so an index created before the vocabulary
# was closed still reads back without a re-ingest.
_LEGACY_ROLES = {
    "block": RegionRole.CLAIM_TEXT,
    "cell": RegionRole.SUPPORTING_CONTEXT,
    "row": RegionRole.SUPPORTING_CONTEXT,
    "table": RegionRole.SUPPORTING_CONTEXT,
    "page": RegionRole.SUPPORTING_CONTEXT,
    "content": RegionRole.SUPPORTING_CONTEXT,
    "visual": RegionRole.VISUAL_REGION,
}


def _coerce_role(value: Any) -> Any:
    if isinstance(value, str) and value not in set(RegionRole):
        return _LEGACY_ROLES.get(value, RegionRole.UNKNOWN)
    return value


class GeometryPrecision(StrEnum):
    """How tightly a region encloses the cited content."""

    BLOCK = "block"
    CELL = "cell"
    ROW = "row"
    TABLE = "table"
    CROP = "crop"
    PAGE = "page"


class Verdict(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    MIXED = "mixed"
    INSUFFICIENT = "insufficient"


class VersionStatus(StrEnum):
    BUILDING = "building"
    READY = "ready"
    INACTIVE = "inactive"


class Region(BaseModel):
    """One highlightable rectangle on a page.

    ``bbox`` is always ``[left, top, right, bottom]`` normalized to 0-1 with a
    top-left origin. The pre-normalization coordinates stay in ``source_bbox``
    and ``source_origin`` so a caller can re-derive the original geometry.
    """

    model_config = ConfigDict(frozen=True)

    bbox: tuple[float, float, float, float]
    role: RegionRole = RegionRole.SUPPORTING_CONTEXT
    precision: GeometryPrecision = GeometryPrecision.BLOCK
    source_bbox: tuple[float, float, float, float] | None = None
    source_origin: str | None = None
    coordinate_space: Literal["normalized_top_left"] = "normalized_top_left"

    _coerce_role = field_validator("role", mode="before")(_coerce_role)


class Citation(BaseModel):
    evidence_id: int
    document_id: int
    document_name: str
    document_sha256: str | None = None
    source_uri: str | None = None
    pdf_page: int
    printed_page_label: str | None = None
    source_kind: EvidenceKind
    quality: EvidenceQuality
    quote: str | None = None
    table_cells: list[str] = Field(default_factory=list)
    heading_path: list[str] = Field(default_factory=list)
    artifact_path: str
    regions: list[Region] = Field(default_factory=list)
    geometry_precision: GeometryPrecision = GeometryPrecision.BLOCK


class EvidenceMatch(BaseModel):
    citation: Citation
    text: str
    lexical_rank: int | None = None
    vector_rank: int | None = None
    graph_rank: int | None = None
    combined_score: float = 0.0


class ClaimResult(BaseModel):
    claim: str
    verdict: Verdict
    rationale: str
    evidence_quality: EvidenceQuality
    citations: list[Citation] = Field(default_factory=list)
    missing_qualifiers: list[str] = Field(default_factory=list)
    audit_id: int | None = None


class IngestReport(BaseModel):
    document_id: int
    version_id: int
    status: VersionStatus
    fingerprint: str
    reused_existing: bool = False
    pages: int = 0
    evidence_units: int = 0
    embedded_units: int = 0
    facts: int = 0
    warnings: list[str] = Field(default_factory=list)
    skipped_artifacts: list[str] = Field(default_factory=list)
    rejected_facts: list[str] = Field(default_factory=list)


# --- Internal records -------------------------------------------------------


class EvidenceUnit(BaseModel):
    """One immutable, addressable piece of source evidence."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    unit_key: str
    page: int
    kind: EvidenceKind
    text: str
    normalized_text: str
    citable: bool = True
    quality: EvidenceQuality = EvidenceQuality.DIRECT_TEXT
    heading_path: list[str] = Field(default_factory=list)
    table_context: dict[str, Any] = Field(default_factory=dict)
    artifact_path: str = ""
    regions: list[Region] = Field(default_factory=list)
    geometry_precision: GeometryPrecision = GeometryPrecision.BLOCK
    truncated_source: bool = False


class Fact(BaseModel):
    """A claim-shaped assertion pinned to the evidence that states it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    subject: str
    metric: str
    value_decimal: Decimal | None = None
    value_text: str | None = None
    unit: str | None = None
    direction: Literal["increase", "decrease", "level", "unknown"] = "unknown"
    comparison: Literal["=", ">=", "<=", ">", "<", "~"] = "="
    reporting_period: str | None = None
    baseline_period: str | None = None
    scope: str | None = None
    geography: str | None = None
    qualifiers: dict[str, Any] = Field(default_factory=dict)
    extraction_method: Literal["table", "llm"] = "table"
    # Required, not defaulted: an optional field is omitted from the schema's
    # `required` list, and a model told a field is optional simply leaves it
    # out -- which made every extracted fact fail the quote check.
    quote: str = Field(
        description=(
            "Text copied verbatim from the passage, character for character, "
            "that states this fact."
        )
    )
    evidence_keys: list[str] = Field(default_factory=list)

    _parse_value = field_validator("value_decimal", mode="before")(_coerce_displayed_value)

    @property
    def fact_key(self) -> str:
        parts = (
            self.subject,
            self.metric,
            str(self.value_decimal),
            self.value_text or "",
            self.unit or "",
            self.reporting_period or "",
            self.baseline_period or "",
            self.scope or "",
            self.geography or "",
            "|".join(sorted(self.evidence_keys)),
        )
        return "\x1f".join(parts)


class ParsedClaim(BaseModel):
    """One atomic claim decomposed into comparable qualifiers."""

    subject: str = ""
    metric: str = ""
    value_decimal: Decimal | None = None
    value_text: str | None = None
    unit: str | None = None
    direction: Literal["increase", "decrease", "level", "unknown"] = "unknown"
    comparison: Literal["=", ">=", "<=", ">", "<", "~"] = "="
    reporting_period: str | None = None
    baseline_period: str | None = None
    scope: str | None = None
    geography: str | None = None
    approximate: bool = False
    key_terms: list[str] = Field(default_factory=list)

    _parse_value = field_validator("value_decimal", mode="before")(_coerce_displayed_value)

    @model_validator(mode="after")
    def _approximate_implies_tolerance(self) -> Self:
        if self.comparison == "~":
            object.__setattr__(self, "approximate", True)
        return self


class FactExtraction(BaseModel):
    """LLM fact-extraction response payload."""

    facts: list[Fact] = Field(default_factory=list)


class VisualVerification(BaseModel):
    """Vision model response for one evidence crop."""

    supports_claim: bool
    visible_text: str = ""
    reason: str = ""


class Adjudication(BaseModel):
    """Structured verdict from the semantic verifier."""

    verdict: Verdict
    rationale: str
    supporting_evidence_ids: list[int] = Field(default_factory=list)
    missing_qualifiers: list[str] = Field(default_factory=list)


__all__ = [
    "Adjudication",
    "Citation",
    "ClaimResult",
    "EvidenceKind",
    "EvidenceMatch",
    "EvidenceQuality",
    "EvidenceUnit",
    "Fact",
    "FactExtraction",
    "GeometryPrecision",
    "IngestReport",
    "ParsedClaim",
    "Region",
    "Verdict",
    "VersionStatus",
    "VisualVerification",
]
