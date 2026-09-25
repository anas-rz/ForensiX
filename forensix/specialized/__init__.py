"""Person-specialized deepfake detection module for ForensiX.

Allows anyone to train teeny tiny models specialized for one individual person
on their CPUs in seconds from their real images, videos, audio recordings, or
audiovisual media.

Provides:
- ``TinyPersonModel``: Teeny tiny architecture (~20k parameters) with LoRA PEFT.
- ``train_person_model``: Train or PEFT fine-tune a model for one person on CPU.
- ``pretrain_base_model``: Pre-train base models on public real vs fake subsets.
- ``PersonDetector``: Standard ForensiX detector interface with ``predict()``.
"""

from pathlib import Path

from forensix.registry import ModelSpec, register
from forensix.specialized.detector import PersonDetector
from forensix.specialized.features import (
    DIM_AUDIO,
    DIM_AUDIOVISUAL,
    DIM_IMAGE,
    DIM_VIDEO,
    extract_audio_features,
    extract_audiovisual_features,
    extract_features,
    extract_image_features,
    extract_video_features,
    generate_synthetic_perturbations,
)
from forensix.specialized.model import LoRALinear, TinyPersonModel
from forensix.specialized.train import (
    load_person_model,
    pretrain_base_model,
    save_person_model,
    train_person_model,
)

__all__ = [
    "DIM_AUDIO",
    "DIM_AUDIOVISUAL",
    "DIM_IMAGE",
    "DIM_VIDEO",
    "LoRALinear",
    "PersonDetector",
    "TinyPersonModel",
    "extract_audio_features",
    "extract_audiovisual_features",
    "extract_features",
    "extract_image_features",
    "extract_video_features",
    "generate_synthetic_perturbations",
    "load_person_model",
    "pretrain_base_model",
    "save_person_model",
    "train_person_model",
]


def _build(weights_path: Path | str | None, device: str = "cpu") -> PersonDetector:
    if weights_path is None:
        raise ValueError(
            "person-detector requires a trained checkpoint. Train one with "
            "forensix.specialized.train_person_model(...) or scripts/train_person_model.py, "
            "then pass weights_path=..."
        )
    return load_person_model(weights_path, device=device)


register(
    ModelSpec(
        name="person-detector",
        family="specialized",
        builder=_build,
        weights=None,
        license="MIT (ForensiX). Face features: SFace / YuNet (Apache-2.0 / MIT).",
        source="",
        paper="",
        description="Teeny tiny deepfake detector specialized for one individual person. "
        "Trained on CPU in seconds with PEFT (LoRA) on the person's real media "
        "(images, video, audio, or audiovisual). Detects deepfakes and impostors.",
    )
)
