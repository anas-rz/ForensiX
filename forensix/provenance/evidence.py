"""The result structure shared by provenance checks.

Provenance checks don't estimate how fake something looks: they read evidence
the file carries (signed credentials, watermarks, metadata). Each returns a
``Prediction`` with ``score_type="provenance"`` and no threshold. Its ``details`` hold:

- ``verdict``: what the evidence says, e.g. ``ai_generated``, ``tampered``,
  ``verified_capture``, ``no_credentials``;
- ``label``: ``"fake"``, ``"real"`` or ``None``. ``None`` is common and means the
  file carries no decisive evidence either way, which is not evidence of authenticity;
- ``summary``: one sentence for people;
- ``findings``: each piece of evidence with a ``severity``: ``strong`` (decisive on its
  own), ``moderate`` (suspicious, needs corroboration) or ``info`` (context).

``fake_prob`` is 1.0 for ``label="fake"``, 0.0 for ``"real"`` and 0.5 otherwise, so
provenance can sit next to detector scores; read ``details`` rather than the number.
"""

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from forensix.types import Prediction

SEVERITIES = ("strong", "moderate", "info")


@dataclass
class Finding:
    code: str
    severity: str  # strong | moderate | info
    message: str
    evidence: dict = field(default_factory=dict)


def provenance_prediction(model: str, verdict: str, label: str | None, summary: str,
                          findings: list[Finding], **extra) -> Prediction:
    findings = sorted(findings, key=lambda f: SEVERITIES.index(f.severity))
    return Prediction(
        model=model, fake_prob={"fake": 1.0, "real": 0.0}.get(label, 0.5), threshold=None,
        score_type="provenance",
        details={"verdict": verdict, "label": label, "summary": summary,
                 "findings": [asdict(f) for f in findings], **extra},
    )


def media_kind(path: str | Path) -> str:
    """``image``, ``video`` or ``audio``, from the content (PIL, then ffprobe)."""
    from PIL import Image

    try:
        with Image.open(path) as img:
            if not getattr(img, "is_animated", False):
                return "image"
    except Exception:
        pass
    info = ffprobe(path)
    streams = info.get("streams", [])
    if any(s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic") for s in streams):
        return "video"
    if any(s.get("codec_type") == "audio" for s in streams):
        return "audio"
    return "audio" if Path(path).suffix.lower() in {".wav", ".flac", ".ogg"} else "unknown"


def ffprobe(path: str | Path) -> dict:
    """ffprobe's format and stream information with all tags, or {} without ffprobe."""
    if not shutil.which("ffprobe"):
        return {}
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
                             capture_output=True, text=True, timeout=60, check=True).stdout
        return json.loads(out)
    except (subprocess.SubprocessError, OSError, ValueError):
        return {}
