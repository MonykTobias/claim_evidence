"""The trust boundary every piece of caller text crosses.

Type, emptiness, length, and a stated reporting entity. Nothing about what the
text says.

This module used to hold a surface grammar that refused anything outside one
exact shape -- approximate, bounded, compound, causal, comparative, and
qualitative claims were all turned away before an audit opened. That was the
right answer while the only comparison available was exact equality on one
figure, because a claim the package could not compare must not be reported as
`insufficient`: that verdict is a statement about the *sources* ("they were
searched and say nothing comparable"), and using it for "we could not
understand the question" turns a limit of the tool into a finding about the
document.

The comparison is no longer only exact equality, and a post is no longer only
one sentence. Bounds, hedges, and unit conversions are compared arithmetically;
what stays incomparable falls through to cited semantic adjudication, which
answers from the evidence rather than from a regular expression. A claim that
is genuinely unauditable is now refused where that is actually knowable -- by
:mod:`claim_evidence.decompose` when a post asserts nothing checkable, and by
the audit itself when no evidence shares the claim's qualifiers.
"""

from __future__ import annotations

from .errors import ValidationError
from .normalize import clean_text

MAX_CLAIM_CHARS = 10_000


def validate_free_text(
    text: str, *, reporting_entity: str, label: str = "claim"
) -> tuple[str, str]:
    """The trust-boundary check every text entry point shares.

    Returns the cleaned text and entity so a caller never works from the raw
    input. Nothing is persisted and no model is called on the way through: a
    rejected request must cost a validation error and nothing else.
    """
    if not isinstance(text, str):
        raise ValidationError(f"{label} must be text")
    cleaned = clean_text(text)
    if not cleaned:
        raise ValidationError(f"{label} is required")
    if len(cleaned) > MAX_CLAIM_CHARS:
        raise ValidationError(f"{label} must be at most {MAX_CLAIM_CHARS} characters")

    entity = clean_text(reporting_entity or "")
    if not entity:
        # The filename is a display label, not a fact about who reported what.
        # Deriving the subject from it made "danoneurdaccessible.pdf" an entity
        # and quietly attributed every figure in the document to that string.
        raise ValidationError(
            "reporting_entity is required: an audit is about one named entity, "
            "and a document's filename is not one"
        )
    return cleaned, entity


__all__ = [
    "MAX_CLAIM_CHARS",
    "validate_free_text",
]
