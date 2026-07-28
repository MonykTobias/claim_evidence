"""Geometry, value, and text normalization.

Two things in this module decide whether a citation is trustworthy:

* ``normalize_bbox`` puts every coordinate system the extractor emits into one
  top-left 0-1 space, so a stored region can be highlighted without knowing
  which artifact produced it.
* ``parse_value`` reads displayed numbers the way a report prints them --
  ``(40.2) %`` is a 40.2 point decrease, not a positive 40.2 -- using
  ``Decimal`` so an exact-value claim is compared exactly.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from .models import GeometryPrecision, Region

# Dashes, minus signs, and non-breaking hyphens all render as "-" in a PDF but
# are distinct code points, which silently breaks substring quote checks.
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−­"), "-")
_QUOTES = {
    ord("‘"): "'",
    ord("’"): "'",
    ord("“"): '"',
    ord("”"): '"',
    ord(" "): " ",
    ord(" "): " ",
    ord(" "): " ",
}
_TRANSLATION = {**_DASHES, **_QUOTES}

_NUMBER_RE = re.compile(r"[-+]?\d[\d\s,._]*\d|\d")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_SCOPE_RE = re.compile(r"\bscope\s*([123])\b")
_APPROX_WORDS = ("about", "roughly", "approximately", "around", "circa", "~", "approx")
_DECREASE_WORDS = ("reduc", "decreas", "declin", "lower", "cut", "down by", "fell")
_INCREASE_WORDS = ("increas", "grew", "growth", "rose", "up by", "higher", "gain")

APPROXIMATE_TOLERANCE = Decimal("0.05")


def clean_text(value: str) -> str:
    """Unicode-fold punctuation and collapse whitespace, preserving case."""
    folded = unicodedata.normalize("NFKC", value).translate(_TRANSLATION)
    return re.sub(r"\s+", " ", folded).strip()


def normalize_for_match(value: str) -> str:
    """Case-insensitive form used for substring quote verification and search."""
    return clean_text(value).casefold()


def contains_quote(haystack: str, quote: str) -> bool:
    """Whether ``quote`` really appears in the source text.

    This is the gate that rejects a fabricated fact: the model must echo text
    that exists in the evidence unit it cited, not a paraphrase of it.
    """
    needle = normalize_for_match(quote)
    return bool(needle) and needle in normalize_for_match(haystack)


# --- Geometry ---------------------------------------------------------------


def normalize_bbox(
    bbox: Any,
    page_width: float,
    page_height: float,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float], str]:
    """Return ``(normalized, source, origin)`` for one extractor bbox.

    Accepts the ``{"l","t","r","b","origin"}`` dicts the extractor writes and
    already-normalized ``[l, t, r, b]`` sequences. The normalized tuple is
    top-left, 0-1, and ordered so left <= right and top <= bottom.
    """
    if bbox is None:
        raise ValueError("bbox is required")
    if isinstance(bbox, dict):
        left = float(bbox["l"])
        right = float(bbox["r"])
        top = float(bbox["t"])
        bottom = float(bbox["b"])
        origin = str(bbox.get("origin") or "TOPLEFT").upper()
    else:
        left, top, right, bottom = (float(v) for v in bbox)
        origin = "NORMALIZED"

    source = (left, top, right, bottom)

    if origin == "NORMALIZED":
        norm = (left, top, right, bottom)
    else:
        if page_width <= 0 or page_height <= 0:
            raise ValueError("page size is required to normalize a bbox")
        if origin == "BOTTOMLEFT":
            top, bottom = page_height - top, page_height - bottom
        norm = (
            left / page_width,
            top / page_height,
            right / page_width,
            bottom / page_height,
        )

    left, top, right, bottom = norm
    if left > right:
        left, right = right, left
    if top > bottom:
        top, bottom = bottom, top
    clamp = lambda v: min(1.0, max(0.0, round(v, 6)))  # noqa: E731
    return (clamp(left), clamp(top), clamp(right), clamp(bottom)), source, origin


def union_bbox(
    boxes: Iterable[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    boxes = list(boxes)
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def region_from(
    bbox: Any,
    page_width: float,
    page_height: float,
    *,
    role: str,
    precision: GeometryPrecision,
) -> Region:
    norm, source, origin = normalize_bbox(bbox, page_width, page_height)
    return Region(
        bbox=norm,
        role=role,
        precision=precision,
        source_bbox=source,
        source_origin=origin,
    )


# --- Values -----------------------------------------------------------------


def parse_value(raw: str) -> tuple[Decimal | None, str | None]:
    """Parse one displayed cell/inline value into ``(decimal, unit)``.

    Accounting parentheses mean negative, which matters because a
    "(40.2) %" variation is a *reduction* of 40.2 points.
    """
    text = clean_text(raw)
    if not text:
        return None, None

    negative = False
    stripped = text.strip()
    # "(40.2) %" and "(40.2%)" both mean a negative variation.
    paren = re.match(r"^\((.*?)\)\s*(.*)$", stripped)
    if paren:
        negative = True
        stripped = f"{paren.group(1)} {paren.group(2)}".strip()

    match = _NUMBER_RE.search(stripped)
    if not match:
        return None, unit_of(text)

    digits = re.sub(r"[\s,_]", "", match.group(0))
    # Trailing separators such as "1.234." are noise, not precision.
    digits = digits.rstrip(".")
    try:
        value = Decimal(digits)
    except InvalidOperation:
        return None, unit_of(text)
    if negative and value > 0:
        value = -value
    return value, unit_of(text)


def unit_of(text: str) -> str | None:
    lowered = clean_text(text).casefold()
    if "%" in lowered or "percent" in lowered or "pp" == lowered.strip():
        return "%"
    return None


def normalize_unit(raw: str | None) -> str | None:
    """Collapse the many ways a report spells the same unit."""
    if raw is None:
        return None
    lowered = clean_text(raw).casefold().strip(" .")
    if not lowered:
        return None
    if "%" in lowered or lowered.startswith("percentage") or lowered == "percent":
        return "%"
    return lowered


def parse_period(text: str) -> str | None:
    """Pick the reporting period out of a header or sentence."""
    years = _YEAR_RE.findall(clean_text(text))
    if not years:
        return None
    matches = _YEAR_RE.finditer(clean_text(text))
    found = [m.group(0) for m in matches]
    return found[-1] if len(found) == 1 else found[0]


def all_years(text: str) -> list[str]:
    return [m.group(0) for m in _YEAR_RE.finditer(clean_text(text))]


def detect_direction(text: str) -> str:
    lowered = clean_text(text).casefold()
    if any(word in lowered for word in _DECREASE_WORDS):
        return "decrease"
    if any(word in lowered for word in _INCREASE_WORDS):
        return "increase"
    return "unknown"


def is_approximate(text: str) -> bool:
    lowered = clean_text(text).casefold()
    return any(word in lowered for word in _APPROX_WORDS)


def signed_change(value: Decimal | None, direction: str) -> Decimal | None:
    """Express a magnitude plus a direction word as one signed number.

    A claim says "reduced by 40.2%"; the table prints "(40.2) %". Both mean
    -40.2, and only the signed form can be compared without re-reading prose.
    """
    if value is None:
        return None
    if direction == "decrease":
        return -abs(value)
    if direction == "increase":
        return abs(value)
    return value


def values_agree(
    claimed: Decimal, observed: Decimal, *, approximate: bool
) -> bool:
    """Exact displayed agreement, or 5% relative tolerance when hedged."""
    if claimed == observed:
        return True
    if not approximate:
        return False
    if claimed == 0:
        return observed == 0
    return abs(observed - claimed) / abs(claimed) <= APPROXIMATE_TOLERANCE


def scope_markers(text: str) -> frozenset[str]:
    """Scope tokens that must not conflict between a claim and a fact.

    "Scope 1 & 2" and "Scope 3" are different emissions boundaries; comparing
    a value across them produces a confidently wrong verdict.
    """
    lowered = clean_text(text).casefold()
    markers = {f"scope{n}" for n in _SCOPE_RE.findall(lowered)}
    if re.search(r"\bscope\s*1\s*(&|and|\+|,)\s*2\b", lowered):
        markers |= {"scope1", "scope2"}
    for phrase in ("market-based", "location-based", "gross", "net", "intensity"):
        if phrase in lowered:
            markers.add(phrase.replace("-", ""))
    return frozenset(markers)


def scopes_comparable(claim_text: str, fact_text: str) -> bool:
    """Whether two scope descriptions describe the same boundary.

    Fails closed: an empty claim scope is not treated as "matches anything",
    because a vague claim must land on `insufficient`, not on the nearest
    number that happens to exist.
    """
    claim_markers = scope_markers(claim_text)
    fact_markers = scope_markers(fact_text)
    if not claim_markers or not fact_markers:
        return False
    return claim_markers <= fact_markers or fact_markers <= claim_markers


def content_tokens(text: str) -> frozenset[str]:
    """Lowercase word tokens with stopwords dropped, for overlap scoring."""
    words = re.findall(r"[a-z0-9]+", normalize_for_match(text))
    return frozenset(w for w in words if len(w) > 2 and w not in _STOPWORDS)


_STOPWORDS = frozenset(
    """the and for with from that this than into over under between per was were
    are has have had been being its our their they them then when what which
    while about above below more most less least also such very can will
    would could should may might must not but out off""".split()
)


__all__ = [
    "APPROXIMATE_TOLERANCE",
    "all_years",
    "clean_text",
    "content_tokens",
    "contains_quote",
    "detect_direction",
    "is_approximate",
    "normalize_bbox",
    "normalize_for_match",
    "normalize_unit",
    "parse_period",
    "parse_value",
    "region_from",
    "scope_markers",
    "scopes_comparable",
    "signed_change",
    "union_bbox",
    "unit_of",
    "values_agree",
]
