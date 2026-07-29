"""Public result types and the internal evidence/fact records they wrap.

Everything crossing the package boundary is a Pydantic model so an external
agent gets validated data and `model_dump()` JSON for free. Model responses are
parsed through the same models, which is what makes a hallucinated field a
validation error instead of a verdict.
"""

from __future__ import annotations

from datetime import datetime, timezone
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


QualifierName = Literal[
    "subject", "metric", "scope", "unit",
    "reporting_period", "baseline_period", "geography",
]


class QualifierComparison(BaseModel):
    """How one material qualifier of the claim lined up with one stored fact.

    ``match`` means the package's own comparison established comparability --
    never that a value was merely parsed. A qualifier the source omits is
    ``missing``, and ``mismatch`` requires both sides present and disagreeing.
    """

    qualifier: QualifierName
    claim_value: str | None = None
    source_value: str | None = None
    status: Literal["match", "mismatch", "missing"]
    reason: str | None = None


class NumericComparison(BaseModel):
    """The arithmetic, in the terms the page and the claim each stated it."""

    claim_value: str | None = None
    claim_operator: str | None = None
    claim_direction: str | None = None
    source_value: str | None = None
    source_operator: str | None = None
    source_unit: str | None = None
    outcome: Literal["match", "conflict", "incomparable", "not_applicable"]
    reason: str | None = None


class EvidenceComparison(BaseModel):
    """One claim-versus-fact comparison, bound to the evidence it came from."""

    evidence_id: int
    fact_id: int | None = None
    pdf_page: int | None = None
    qualifiers: list[QualifierComparison] = Field(default_factory=list)
    numeric: NumericComparison


class DecisionExplanation(BaseModel):
    """Why this verdict, in operational terms.

    ``verdict_rule`` is a stable machine-readable name for the rule that fired,
    not model reasoning: the user-facing prose stays in ``rationale``. Nothing
    here is a prompt, a raw model reply, or hidden chain-of-thought.
    """

    decided_by: Literal[
        "deterministic_comparison", "semantic_adjudication", "no_evidence"
    ]
    verdict_rule: str
    evidence_comparisons: list[EvidenceComparison] = Field(default_factory=list)


class IndexReference(BaseModel):
    """Exactly which ready version answered one audit.

    Pinned before retrieval, so a trace read back later says what was searched
    rather than what happens to be ready now.
    """

    document_id: int
    document_version_id: int
    embedding_model: str
    embedding_dimensions: int


class ClaimResult(BaseModel):
    claim: str
    verdict: Verdict
    rationale: str
    evidence_quality: EvidenceQuality
    citations: list[Citation] = Field(default_factory=list)
    missing_qualifiers: list[str] = Field(default_factory=list)
    audit_id: int | None = None
    decision_explanation: DecisionExplanation | None = None
    # Elapsed seconds per public phase group; a phase that did not run is null
    # rather than zero.
    timings: dict[str, float | None] = Field(default_factory=dict)
    index_references: list[IndexReference] = Field(default_factory=list)


class IngestReport(BaseModel):
    document_id: int
    version_id: int
    status: VersionStatus
    fingerprint: str
    reused_existing: bool = False
    pages: int = 0
    evidence_units: int = 0
    visual_evidence_units: int = 0
    embedded_units: int = 0
    facts: int = 0
    warnings: list[str] = Field(default_factory=list)
    skipped_artifacts: list[str] = Field(default_factory=list)
    rejected_facts: list[str] = Field(default_factory=list)


def phase_percent(completed: int | None, total: int | None) -> float | None:
    """Phase-local percentage, or None when the work is not measurable.

    A total of zero is a finished phase with nothing to do, not a division
    error: "0 of 0 crops verified" is 100% done.
    """
    if completed is None or total is None:
        return None
    if total <= 0:
        return 100.0
    return round(min(100.0, max(0.0, 100.0 * completed / total)), 2)


class ProgressEvent(BaseModel):
    """One phase update from a running ingestion or audit.

    Carries counts and identifiers only. Never a prompt, a model response,
    source text, a connection string, or a stack trace -- a frontend renders
    these directly, and an operator may paste one into a bug report.
    """

    operation: Literal["ingest", "audit"]
    phase: str
    status: Literal["start", "progress", "completed", "warning", "failed"]
    message: str
    # Null whenever the work is genuinely not measurable; an invented
    # percentage is worse than an honest spinner.
    completed: int | None = None
    total: int | None = None
    percent: float | None = None
    document_id: int | None = None
    audit_id: int | None = None
    current_item: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _derive_percent(self) -> Self:
        if self.percent is None:
            object.__setattr__(
                self, "percent", phase_percent(self.completed, self.total)
            )
        return self


# --- Frontend-support types -------------------------------------------------


class ModelHealth(BaseModel):
    role: Literal["embed", "chat", "vision"]
    name: str
    available: bool


class HealthReport(BaseModel):
    """System diagnostics. Carries no credentials, connection strings, raw
    driver exceptions, or prompts -- only categories and safe sentences."""

    database_reachable: bool = False
    schema_version: int | None = None
    schema_current: bool = False
    pgvector_version: str | None = None
    ollama_reachable: bool = False
    models: list[ModelHealth] = Field(default_factory=list)
    schema_embedding_dimensions: int | None = None
    configured_embedding_dimensions: int | None = None
    documents_ready: int = 0
    # Recently active builds only. A build whose process died cannot write its
    # own failure, so it is classified by silence instead: still 'building'
    # with no progress for longer than the stale threshold is 'interrupted'.
    documents_building: int = 0
    documents_failed: int = 0
    documents_interrupted: int = 0
    documents_inactive: int = 0
    # The queryable index: rows belonging to a ready version, which is exactly
    # what retrieval can reach. Historical rows from failed, superseded and
    # half-built versions are counted separately.
    evidence_units: int = 0
    embeddings: int = 0
    facts: int = 0
    stored_evidence_units: int = 0
    problems: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.database_reachable and self.schema_current and not self.problems


class DocumentSummary(BaseModel):
    document_id: int
    document_version_id: int
    name: str
    source_uri: str | None = None
    source_sha256: str | None = None
    status: VersionStatus
    page_count: int = 0
    evidence_count: int = 0
    fact_count: int = 0
    visual_evidence_count: int = 0
    embedding_model: str
    embedding_dimensions: int
    indexed_at: datetime | None = None
    # Server-side path metadata. A frontend backend may use these after its own
    # allowed-root validation; they are not for an untrusted browser.
    output_root: str
    source_pdf: str | None = None


class RemovalReport(BaseModel):
    document_id: int
    name: str
    deleted: dict[str, int] = Field(default_factory=dict)


class TraceCandidate(BaseModel):
    evidence_id: int
    pdf_page: int
    source_kind: EvidenceKind
    text: str
    lexical_rank: int | None = None
    lexical_score: float | None = None
    vector_rank: int | None = None
    vector_score: float | None = None
    graph_rank: int | None = None
    graph_score: float | None = None
    combined_rank: int | None = None
    combined_score: float = 0.0
    expanded_from: int | None = None
    visual_status: Literal[
        "not_applicable", "verified", "rejected", "unavailable"
    ] = "not_applicable"
    selected: bool = False
    reason: str | None = None


class AuditTrace(BaseModel):
    """Operational retrieval metadata for one audit.

    How candidates were found and ranked -- not model reasoning. Only the
    structured output and the concise rationale the product already shows are
    stored, so a trace can be handed to a UI without leaking a prompt.
    """

    audit_id: int
    claim: str
    # The corpus that was searched, recorded when the audit opened. Not derived
    # from citations: an insufficient verdict cites nothing and still searched
    # something, and a document removed afterwards must not erase the record.
    document_ids: list[int] = Field(default_factory=list)
    status: Literal["running", "completed", "failed"] = "completed"
    created_at: datetime | None = None
    completed_at: datetime | None = None
    failed_at: datetime | None = None
    failure_code: str | None = None
    failure_phase: str | None = None
    retryable: bool | None = None
    parsed_claim: dict[str, Any] = Field(default_factory=dict)
    verdict: Verdict | None = None
    rationale: str | None = None
    evidence_quality: EvidenceQuality | None = None
    missing_qualifiers: list[str] = Field(default_factory=list)
    citation_ids: list[int] = Field(default_factory=list)
    candidates: list[TraceCandidate] = Field(default_factory=list)
    decision_explanation: DecisionExplanation | None = None
    timings: dict[str, float | None] = Field(default_factory=dict)
    index_references: list[IndexReference] = Field(default_factory=list)
    error: str | None = None

    @property
    def graph_candidates(self) -> list[TraceCandidate]:
        return [c for c in self.candidates if c.graph_rank is not None]

    @property
    def lexical_candidates(self) -> list[TraceCandidate]:
        return [c for c in self.candidates if c.lexical_rank is not None]

    @property
    def vector_candidates(self) -> list[TraceCandidate]:
        return [c for c in self.candidates if c.vector_rank is not None]


class EvidenceDetail(BaseModel):
    """Everything needed to render a cited region over its page image.

    The package renders nothing itself: it publishes one authoritative
    provenance representation and lets the caller draw.
    """

    evidence_id: int
    document_id: int
    document_version_id: int
    document_name: str
    pdf_page: int
    printed_page_label: str | None = None
    source_kind: EvidenceKind
    evidence_quality: EvidenceQuality
    geometry_precision: GeometryPrecision
    text: str
    quote: str | None = None
    table_context: dict[str, Any] = Field(default_factory=dict)
    heading_path: list[str] = Field(default_factory=list)
    page_width: float
    page_height: float
    page_image_path: str | None = None
    regions: list[Region] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)


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
    # Where this unit sits on its page, and what it belongs to. Context
    # expansion reads these instead of evidence ids, which record when a row
    # was inserted rather than what the page actually says.
    source_order: int | None = None
    context_key: str | None = None


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
    "AuditTrace",
    "Citation",
    "ClaimResult",
    "DocumentSummary",
    "EvidenceDetail",
    "EvidenceKind",
    "EvidenceMatch",
    "EvidenceQuality",
    "EvidenceUnit",
    "Fact",
    "FactExtraction",
    "GeometryPrecision",
    "HealthReport",
    "IngestReport",
    "ModelHealth",
    "ParsedClaim",
    "ProgressEvent",
    "phase_percent",
    "Region",
    "RegionRole",
    "RemovalReport",
    "TraceCandidate",
    "Verdict",
    "VersionStatus",
    "VisualVerification",
]
