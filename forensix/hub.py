"""Download and cache pretrained weights."""

import hashlib
import json
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


def _stamp_path(path: Path) -> Path:
    return path.with_name(path.name + ".verified")


def _file_id(path: Path) -> dict:
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _verified(path: Path, sha256: str) -> bool:
    """Whether ``path`` has the expected hash. Hashing a 1.6 GB file takes seconds,
    so a successful check leaves a stamp (hash, size, mtime) next to the file and
    later checks of the unchanged file trust it. ``FORENSIX_VERIFY_HASH=always``
    rehashes every time."""
    stamp = _stamp_path(path)
    if os.environ.get("FORENSIX_VERIFY_HASH") != "always":
        try:
            if json.loads(stamp.read_text()) == {"sha256": sha256, **_file_id(path)}:
                return True
        except (OSError, ValueError):
            pass
    if _sha256(path) != sha256:
        return False
    try:
        stamp.write_text(json.dumps({"sha256": sha256, **_file_id(path)}))
    except OSError:  # read-only cache: verify again next time
        pass
    return True


def fetch(weights: WeightFile, force: bool = False) -> Path:
    """Return the local path to ``weights``, downloading and verifying if needed."""
    dest = cache_dir() / "weights" / weights.filename
    if dest.exists() and not force:
        if _verified(dest, weights.sha256):
            return dest
        dest.unlink()
        _stamp_path(dest).unlink(missing_ok=True)

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
        _stamp_path(dest).write_text(json.dumps({"sha256": actual, **_file_id(dest)}))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return dest


def torch_load(path: str | Path) -> dict:
    """``torch.load`` with ``weights_only=True``, also allowing the numpy
    types (arrays, scalars, dtypes) that some upstream checkpoints store next
    to their weights, e.g. training logs. Nothing else is unpickled."""
    import numpy as np
    import torch

    try:  # numpy >= 2 moved numpy.core to numpy._core; pickles use the old name
        from numpy._core import multiarray
    except ImportError:
        from numpy.core import multiarray
    allowed = [
        (multiarray.scalar, "numpy.core.multiarray.scalar"),
        (multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        np.ndarray,
        np.dtype,
        *{type(np.dtype(t)) for t in (np.float16, np.float32, np.float64, np.int32, np.int64, np.bool_)},
    ]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)
