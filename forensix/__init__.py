"""ForensiX: a model zoo for deepfake detection.

    import forensix
    forensix.list_models()              # ['aasist', 'aasist-l']
    detector = forensix.load("aasist")  # downloads weights on first use
    detector.predict("clip.wav")        # Prediction(fake_prob=..., ...)
"""

from pathlib import Path

from forensix import audio  # noqa: F401  (registers audio models)
from forensix.hub import cache_dir, fetch
from forensix.registry import ModelSpec, get_spec, list_models
from forensix.types import Prediction

__all__ = [
    "ModelSpec",
    "Prediction",
    "cache_dir",
    "download",
    "info",
    "list_models",
    "load",
]


def info(name: str) -> ModelSpec:
    """Metadata for a registered model: family, license, source, paper, ..."""
    return get_spec(name)


def download(name: str, force: bool = False) -> Path | None:
    """Download (and checksum) a model's pretrained weights into the cache."""
    spec = get_spec(name)
    return fetch(spec.weights, force=force) if spec.weights else None


def load(name: str, device: str = "cpu", weights_path: str | Path | None = None):
    """Build a detector with pretrained weights, downloading them if needed.

    Pass ``weights_path`` to use a local checkpoint (e.g. one you trained).
    """
    spec = get_spec(name)
    path = Path(weights_path) if weights_path else download(name)
    return spec.builder(path, device=device)
