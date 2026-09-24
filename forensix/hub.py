"""Download and cache pretrained weights."""

import hashlib
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WeightFile:
    url: str
    sha256: str
    filename: str


def cache_dir() -> Path:
    """Weights cache: ``$FORENSIX_CACHE`` or ``~/.cache/forensix``."""
    root = os.environ.get("FORENSIX_CACHE")
    if root:
        return Path(root).expanduser()
    return Path.home() / ".cache" / "forensix"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(weights: WeightFile, force: bool = False) -> Path:
    """Return the local path to ``weights``, downloading and verifying if needed."""
    dest = cache_dir() / "weights" / weights.filename
    if dest.exists() and not force:
        if _sha256(dest) == weights.sha256:
            return dest
        dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(weights.url) as resp:
            shutil.copyfileobj(resp, out)
        actual = _sha256(Path(tmp))
        if actual != weights.sha256:
            raise RuntimeError(
                f"Checksum mismatch for {weights.url}: "
                f"expected {weights.sha256}, got {actual}"
            )
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return dest
