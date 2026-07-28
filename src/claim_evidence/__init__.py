"""Evidence-first claim auditing over completed ``document_extract`` output."""

from .client import ClaimEvidence
from .config import Settings
from .models import (
    Citation,
    ClaimResult,
    EvidenceKind,
    EvidenceMatch,
    EvidenceQuality,
    GeometryPrecision,
    IngestReport,
    Region,
    Verdict,
    VersionStatus,
)

__version__ = "0.1.0"

__all__ = [
    "Citation",
    "ClaimEvidence",
    "ClaimResult",
    "EvidenceKind",
    "EvidenceMatch",
    "EvidenceQuality",
    "GeometryPrecision",
    "IngestReport",
    "Region",
    "Settings",
    "Verdict",
    "VersionStatus",
    "__version__",
]
