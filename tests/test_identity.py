"""Document identity: what makes two extractions the same logical document.

No database and no model. Run from the repository root with
``python tests/test_identity.py``.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from claim_evidence.errors import ValidationError
from claim_evidence.ingest import (
    IDENTITY_VERSION,
    canonical_local_path,
    identity_basis,
    identity_key,
    normalize_source_uri,
)
from claim_evidence.source import canonical_digest

from fixtures import write_output_root


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[ok] {message}")


def test_the_pdf_is_the_strongest_basis() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = write_output_root(Path(temp) / "run", pages=1)
        kind, value = identity_basis(root, "AbC123", "urn:something")
        check(kind == "pdf_sha256", "a PDF hash outranks a URI and a path")
        check(value == "abc123", "the hash is compared in one case")


def test_a_uri_is_used_only_without_a_pdf() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = write_output_root(Path(temp) / "run", pages=1)
        kind, _ = identity_basis(root, None, "urn:report:2025")
        check(kind == "source_uri", "a URI outranks the output path")
        kind, _ = identity_basis(root, None, "   ")
        check(kind == "local_output_path", "a blank URI is not a URI")


def test_the_same_pdf_at_another_path_is_one_document() -> None:
    """PD-04's central promise: identity follows the bytes, not the location."""
    payload = b"%PDF-1.7 the same document\n"
    digest = hashlib.sha256(payload).hexdigest()
    with tempfile.TemporaryDirectory() as temp:
        first = write_output_root(Path(temp) / "monday" / "run", pages=1)
        second = write_output_root(Path(temp) / "tuesday" / "elsewhere", pages=1)
        check(
            identity_key(first, digest, None) == identity_key(second, digest, None),
            "the same PDF extracted twice into different folders is one document",
        )


def test_distinct_pdf_bytes_never_merge() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = write_output_root(Path(temp) / "run", pages=1)
        one = hashlib.sha256(b"report A").hexdigest()
        two = hashlib.sha256(b"report B").hexdigest()
        check(
            identity_key(root, one, None) != identity_key(root, two, None),
            "two different PDFs in the same folder stay two documents",
        )


def test_a_uri_and_a_path_that_read_alike_are_different_documents() -> None:
    """The basis kind is inside the digest, so the two namespaces cannot collide."""
    with tempfile.TemporaryDirectory() as temp:
        root = write_output_root(Path(temp) / "run", pages=1)
        as_path = identity_key(root, None, None)
        as_uri = identity_key(root, None, canonical_local_path(root))
        check(as_path != as_uri, "the same text in two roles is two identities")


def test_windows_path_spellings_settle_to_one_identity() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = write_output_root(Path(temp) / "Run", pages=1)
        spellings = [root, Path(str(root)), root.parent / root.name]
        keys = {identity_key(p, None, None) for p in spellings}
        check(len(keys) == 1, "equivalent spellings of one path are one document")
        if os.name == "nt":
            upper = Path(str(root).upper())
            check(
                identity_key(upper, None, None) == identity_key(root, None, None),
                "Windows case-insensitivity does not split a document",
            )
        forward = Path(str(root).replace("\\", "/"))
        check(
            identity_key(forward, None, None) == identity_key(root, None, None),
            "the separator a caller happened to type does not split a document",
        )


def test_a_path_identity_requires_the_path_to_exist() -> None:
    """A path-based identity for something that is not there identifies nothing."""
    with tempfile.TemporaryDirectory() as temp:
        missing = Path(temp) / "never-extracted"
        try:
            identity_key(missing, None, None)
        except ValidationError as exc:
            check("does not exist" in str(exc), f"a missing root is a validation error ({exc})")
            check(str(missing) not in str(exc), "and the error does not echo the path")
            return
    raise AssertionError("a missing output root must not produce an identity")


def test_uri_normalization_is_bounded() -> None:
    check(
        normalize_source_uri("HTTPS://Example.COM/Reports/A.pdf")
        == "https://example.com/Reports/A.pdf",
        "scheme and host are case-folded; the path is not",
    )
    check(
        normalize_source_uri("https://example.com/a/") == "https://example.com/a",
        "a trailing slash does not create a second document",
    )
    check(
        normalize_source_uri("  urn:report:2025  ") == "urn:report:2025",
        "a scheme without an authority is left alone but trimmed",
    )
    check(
        normalize_source_uri("https://example.com/A")
        != normalize_source_uri("https://example.com/a"),
        "two paths differing only in case stay two documents",
    )


def test_the_key_is_a_versioned_tagged_digest() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = write_output_root(Path(temp) / "run", pages=1)
        digest = hashlib.sha256(b"x").hexdigest()
        key = identity_key(root, digest, None)
        check(
            key
            == canonical_digest(
                {
                    "identity_version": IDENTITY_VERSION,
                    "basis": "pdf_sha256",
                    "value": digest,
                }
            ),
            "the key is the canonical digest of a versioned, tagged basis",
        )
        check(key != digest, "the identity key is never the public source hash")
        check(len(key) == 64, "and it is a full SHA-256")


def main() -> int:
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            print(f"\n--- {name} ---")
            function()
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
