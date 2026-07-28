"""Evidence-first claim auditing over completed ``document_extract`` output."""

from .client import ClaimEvidence
from .config import Settings
from .errors import (
    ClaimEvidenceError,
    DependencyUnavailableError,
    IndexNotReadyError,
    NotFoundError,
    ValidationError,
)
from .progress import ProgressCallback, classify_error
from .models import (
    AuditTrace,
    Citation,
    ClaimResult,
    DocumentSummary,
    EvidenceDetail,
    EvidenceKind,
    EvidenceMatch,
    EvidenceQuality,
    GeometryPrecision,
    HealthReport,
    IngestReport,
    ModelHealth,
    ProgressEvent,
    Region,
    RegionRole,
    RemovalReport,
    TraceCandidate,
    Verdict,
    VersionStatus,
)

__version__ = "0.2.0"

__all__ = [
    "AuditTrace",
    "Citation",
    "ClaimEvidence",
    "ClaimEvidenceError",
    "ClaimResult",
    "DependencyUnavailableError",
    "DocumentSummary",
    "EvidenceDetail",
    "EvidenceKind",
    "EvidenceMatch",
    "EvidenceQuality",
    "GeometryPrecision",
    "HealthReport",
    "IndexNotReadyError",
    "IngestReport",
    "ModelHealth",
    "NotFoundError",
    "ProgressCallback",
    "ProgressEvent",
    "Region",
    "RegionRole",
    "RemovalReport",
    "Settings",
    "TraceCandidate",
    "ValidationError",
    "classify_error",
    "Verdict",
    "VersionStatus",
    "__version__",
]
