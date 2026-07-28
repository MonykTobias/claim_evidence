"""Re-verify visual evidence from the cited crop.

An image summary written during extraction is a retrieval hint, not proof. A
chart may only support a verdict once the vision model has looked at the exact
region the citation points to and confirmed the claim's numbers are visible.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Sequence

from PIL import Image

from .models import Region, VisualVerification
from .ollama import OllamaClient, OllamaError

VISION_SYSTEM = """\
You inspect one cropped region of a report page.

Answer only from what is legible in the image. Copy visible numbers and labels
into `visible_text` exactly as printed. Set `supports_claim` true only when the
crop itself shows the claim's metric and value; a plausible-looking chart with
no readable figure does not support anything.
"""

# A tight cell crop is unreadable on its own; a little context keeps axis labels
# and units inside the frame.
CROP_PADDING = 0.02


def crop_region(page_png: Path, regions: Sequence[Region]) -> bytes:
    """Crop the page image to the union of the cited regions."""
    if not regions:
        raise ValueError("no region to crop")
    left = max(0.0, min(r.bbox[0] for r in regions) - CROP_PADDING)
    top = max(0.0, min(r.bbox[1] for r in regions) - CROP_PADDING)
    right = min(1.0, max(r.bbox[2] for r in regions) + CROP_PADDING)
    bottom = min(1.0, max(r.bbox[3] for r in regions) + CROP_PADDING)

    with Image.open(page_png) as image:
        width, height = image.size
        box = (
            int(left * width),
            int(top * height),
            max(int(right * width), int(left * width) + 1),
            max(int(bottom * height), int(top * height) + 1),
        )
        crop = image.convert("RGB").crop(box)
        buffer = io.BytesIO()
        crop.save(buffer, format="PNG")
        return buffer.getvalue()


def verify_visual(
    client: OllamaClient,
    page_png: Path,
    regions: Sequence[Region],
    claim: str,
) -> VisualVerification:
    """Check one crop against the claim. Failures verify as unsupported."""
    try:
        image = crop_region(page_png, regions)
    except (OSError, ValueError) as exc:
        return VisualVerification(
            supports_claim=False, reason=f"crop unavailable: {exc}"
        )
    try:
        return client.vision(
            VisualVerification,
            VISION_SYSTEM,
            f"Claim under audit:\n{claim}\n\nWhat does this crop show?",
            image,
        )
    except OllamaError as exc:
        return VisualVerification(
            supports_claim=False, reason=f"vision verification failed: {exc}"
        )


__all__ = ["CROP_PADDING", "VISION_SYSTEM", "crop_region", "verify_visual"]
