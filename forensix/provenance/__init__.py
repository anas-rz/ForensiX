"""Provenance: evidence a file carries about its origin, next to the detectors' judgement of how it looks.

- ``c2pa``: C2PA Content Credentials (signed provenance; tampering; declared AI).
- ``watermark-image``: the invisible watermark Stable Diffusion 1.x/2.x/SDXL/SVD embed.
- ``watermark-audio``: Meta's AudioSeal watermark.
- ``metadata``: generator traces, editing and re-encoding in EXIF/XMP/PNG chunks and container tags.

A signed AI declaration or a found watermark is stronger evidence than any
classifier score; their absence is not evidence of authenticity.

    report = forensix.provenance.check("upload.jpg")
    report["verdict"], report["label"], report["findings"]
"""

from pathlib import Path

from forensix.provenance import c2pa_check, metadata, watermark_audio, watermark_image  # noqa: F401  (registers models)
from forensix.provenance.evidence import Finding, media_kind

__all__ = ["CHECKS", "Finding", "check", "media_kind"]

# which checks apply to which media
CHECKS = {"image": ["c2pa", "watermark-image", "metadata"], "video": ["c2pa", "watermark-image", "metadata"],
          "audio": ["c2pa", "watermark-audio", "metadata"]}
# how decisive each verdict is, for the combined report
_RANK = {"tampered": 6, "ai_generated": 5, "watermark_found": 5, "verified_capture": 4, "partial_watermark": 3,
         "invalid_credentials": 2, "ai_enhanced": 2, "edited": 2}


def check(path: str | Path, models: list[str] | None = None, device: str = "cpu", **options) -> dict:
    """Run every applicable provenance check and combine them: the most decisive verdict
    leads (a found watermark or AI declaration over everything but tampering), and every
    finding is kept. ``options`` go to the checks, e.g. ``trust_anchors="c2pa-trust.pem"``.
    Checks whose optional packages aren't installed are reported as skipped."""
    import forensix

    kind = media_kind(path)
    names = models or CHECKS.get(kind, ["metadata"])
    results, skipped = {}, {}
    for name in names:
        try:
            opts = {k: v for k, v in options.items() if name == "c2pa" and k == "trust_anchors"}
            results[name] = forensix.load(name, device=device, **opts).predict(path).details
        except ImportError as e:
            skipped[name] = str(e)
    lead = max(results.items(), key=lambda kv: _RANK.get(kv[1]["verdict"], 0), default=(None, None))
    labels = {r["label"] for r in results.values() if r["label"]}
    label = "fake" if "fake" in labels else ("real" if labels == {"real"} else None)
    return {
        "media": kind, "label": label,
        "verdict": lead[1]["verdict"] if lead[0] and _RANK.get(lead[1]["verdict"]) else "no_decisive_evidence",
        "summary": lead[1]["summary"] if lead[0] and _RANK.get(lead[1]["verdict"]) else
        "No decisive provenance evidence (most files carry none; this is not evidence of authenticity).",
        "findings": [{"check": n, **f} for n, r in results.items() for f in r["findings"]],
        "checks": results, "skipped": skipped,
    }
