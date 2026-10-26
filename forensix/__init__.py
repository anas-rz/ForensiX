"""ForensiX: a model zoo for deepfake detection.

    import forensix
    forensix.list_models()              # ['aasist', ..., 'univfd']
    detector = forensix.load("aasist")  # downloads weights on first use
    detector.predict("clip.wav")        # Prediction(fake_prob=..., ...)
"""

from pathlib import Path

from forensix import audio, audiovisual, identity, image, physics, specialized, video  # noqa: F401  (registers models)
from forensix.hub import cache_dir, fetch
from forensix.registry import ModelSpec, get_spec, list_models
from forensix.types import Prediction

person = specialized

__all__ = [
    "ModelSpec",
    "Prediction",
    "cache_dir",
    "download",
    "info",
    "list_models",
    "load",
    "person",
    "specialized",
]


def info(name: str) -> ModelSpec:
    """Metadata for a registered model: family, license, source, paper, ..."""
    return get_spec(name)


def download(name: str, force: bool = False) -> Path | None:
    """Download (and checksum) a model's pretrained weights into the cache.

    Returns the path to the model's own weights; shared backbones listed in
    ``extra_weights`` are fetched too.
    """
    spec = get_spec(name)
    for extra in spec.extra_weights:
        fetch(extra, force=force)
    return fetch(spec.weights, force=force) if spec.weights else None


def load(name: str, device: str = "cpu", weights_path: str | Path | None = None,
         calibration: str | Path | None = None):
    """Build a detector with pretrained weights, downloading them if needed.

    Pass ``weights_path`` to use a local checkpoint (e.g. one you trained), and
    ``calibration`` to apply a calibration table (see ``forensix.calibration``).
    """
    spec = get_spec(name)
    path = Path(weights_path) if weights_path else download(name)
    detector = spec.builder(path, device=device)
    if calibration is None:
        return detector
    from forensix.calibration import Calibration, CalibratedDetector

    table = Calibration.load(calibration)
    if table.model != name:
        raise ValueError(f"Calibration {calibration} is for {table.model!r}, not {name!r}")
    return CalibratedDetector(detector, table)
