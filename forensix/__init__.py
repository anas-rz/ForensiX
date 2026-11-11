"""ForensiX: a model zoo for deepfake detection.

    import forensix
    forensix.list_models()              # ['aasist', ..., 'univfd']
    detector = forensix.load("aasist")  # downloads weights on first use
    detector.predict("clip.wav")        # Prediction(fake_prob=..., ...)
"""

from pathlib import Path

from forensix import audio, audiovisual, identity, image, physics, provenance, specialized, video  # noqa: F401  (registers models)
from forensix.hub import cache_dir, fetch
from forensix.registry import ModelSpec, get_spec, list_models
from forensix.types import Prediction

person = specialized

__all__ = [
    "ModelSpec",
    "calibrate",
    "finetune",
    "Prediction",
    "cache_dir",
    "download",
    "info",
    "list_models",
    "load",
    "load_fusion",
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
         calibration: str | Path | None = None, **options):
    """Build a detector with pretrained weights, downloading them if needed.

    Pass ``weights_path`` to use a local checkpoint (e.g. one you trained), and
    ``calibration`` to apply a calibration table (see ``forensix.calibration``).
    Other keyword arguments are model options, e.g. ``clip_score="max-window"``
    for the window-based audio-visual detectors.
    """
    spec = get_spec(name)
    path = Path(weights_path) if weights_path else download(name)
    finetuned = None
    if weights_path:
        from forensix.finetuning import checkpoint_info

        if checkpoint_info(path):  # a forensix.finetune checkpoint: pretrained model + changed tensors
            finetuned, path = path, download(name)
    try:
        detector = spec.builder(path, device=device, **options)
    except TypeError as e:
        if options and "unexpected keyword" in str(e):
            raise TypeError(f"{name} doesn't take options {sorted(options)}") from e
        raise
    if finetuned is not None:
        from forensix.finetuning import apply

        apply(detector, finetuned)
    if calibration is None:
        return detector
    from forensix.calibration import Calibration, CalibratedDetector

    table = Calibration.load(calibration)
    if table.model != name:
        raise ValueError(f"Calibration {calibration} is for {table.model!r}, not {name!r}")
    return CalibratedDetector(detector, table)


def calibrate(model, real_media=None, fake_media=None, dataset=None, **kwargs):
    """Score labelled media with ``model`` and fit its calibration: ``(Calibration, report)``.
    ``dataset`` is a folder or ZIP in the ForensiX dataset format (``forensix.dataset``).
    See ``forensix.calibration.calibrate`` for the options."""
    from forensix.calibration import calibrate as _calibrate

    return _calibrate(model, real_media, fake_media, dataset, **kwargs)


def finetune(model, dataset=None, real_media=None, fake_media=None, out=None, **kwargs):
    """Fine-tune a pretrained detector on labelled media: ``(detector, report)``, saving the
    checkpoint to ``out``; ``forensix.load(model, weights_path=out)`` loads it later. See
    ``forensix.finetuning`` for the modes and options."""
    from forensix.finetuning import finetune as _finetune

    return _finetune(model, dataset, real_media, fake_media, out, **kwargs)


def load_fusion(path: str | Path, device: str = "cpu"):
    """Load every member of a fusion table (see ``forensix.fusion``) as one detector."""
    from forensix.fusion import FusedDetector, Fusion

    fusion = Fusion.load(path)
    detectors = [load(m.model, device=device, **m.options) for m in fusion.members]
    return FusedDetector(detectors, fusion, name=f"fusion:{Path(path).stem}")
