"""Inference detector for person-specialized deepfake detection.

Wraps a trained TinyPersonModel and provides the standard ForensiX ``predict()``
interface returning ``forensix.types.Prediction``.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import torch

from forensix.specialized.features import (
    extract_audio_features,
    extract_audiovisual_features,
    extract_image_features,
    extract_video_features,
)
from forensix.specialized.model import TinyPersonModel
from forensix.types import Prediction


class PersonDetector:
    """A lightweight detector specialized for one individual person.

    Evaluates whether input media (image, audio, video, or audiovisual) is real
    and belongs to the enrolled target person, or is a deepfake/impostor.
    """

    def __init__(
        self,
        model: TinyPersonModel,
        person_name: str,
        modality: str = "image",
        threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.name = "person-detector"
        self.model = model.to(device).eval()
        self.person_name = person_name
        self.modality = modality
        self.threshold = float(threshold)
        self.device = device

    def __repr__(self) -> str:
        params = self.model.get_param_count()
        return (
            f"PersonDetector(person={self.person_name!r}, modality={self.modality!r}, "
            f"params={params['total']:,}, threshold={self.threshold:.3f})"
        )

    def predict(self, media: Any) -> Prediction:
        """Run deepfake detection on the input media for this enrolled person.

        Args:
            media: Path to image, audio, or video file, or a NumPy array.

        Returns:
            Prediction object with ``fake_prob``, ``threshold``, ``is_fake``, and ``details``.
        """
        with torch.no_grad():
            if self.modality == "image":
                feat = extract_image_features(media)
                x_t = torch.as_tensor(feat, dtype=torch.float32, device=self.device)
                fake_prob_t, details = self.model.score(x_t)
                fake_prob = round(float(fake_prob_t.item()), 4)
                segment_probs: tuple[float, ...] = ()

            elif self.modality == "audio":
                feat = extract_audio_features(media)
                x_t = torch.as_tensor(feat, dtype=torch.float32, device=self.device)
                fake_prob_t, details = self.model.score(x_t)
                fake_prob = round(float(fake_prob_t.item()), 4)
                segment_probs = ()

            elif self.modality == "video":
                video_feat, frame_feats, times = extract_video_features(media)
                x_t = torch.as_tensor(video_feat, dtype=torch.float32, device=self.device)
                fake_prob_t, details = self.model.score(x_t)
                fake_prob = round(float(fake_prob_t.item()), 4)

                # Segment-level predictions for frames
                if frame_feats:
                    frames_t = torch.as_tensor(np.stack(frame_feats), dtype=torch.float32, device=self.device)
                    f_probs_t, _ = self.model.score(frames_t)
                    segment_probs = tuple(round(float(p), 4) for p in f_probs_t.tolist())
                    details["frame_times_s"] = times
                    details["frame_fake_probs"] = list(segment_probs)
                else:
                    segment_probs = ()

            elif self.modality == "audiovisual":
                av_feat, av_details = extract_audiovisual_features(media)
                x_t = torch.as_tensor(av_feat, dtype=torch.float32, device=self.device)
                fake_prob_t, details = self.model.score(x_t)
                fake_prob = round(float(fake_prob_t.item()), 4)
                details.update(av_details)
                segment_probs = ()

            else:
                raise ValueError(f"Unsupported modality: {self.modality!r}")

        details["target_person"] = self.person_name
        details["modality"] = self.modality
        details["threshold"] = self.threshold

        return Prediction(
            model=self.name,
            fake_prob=fake_prob,
            threshold=self.threshold,
            score_type="probability",
            segment_fake_probs=segment_probs,
            details=details,
        )

    def save(self, path: str | PathLike) -> Path:
        """Save this detector to disk as a safetensors file."""
        from forensix.specialized.train import save_person_model

        return save_person_model(path, self)

    @classmethod
    def load(cls, path: str | PathLike, device: str = "cpu") -> PersonDetector:
        """Load a detector from a safetensors file."""
        from forensix.specialized.train import load_person_model

        return load_person_model(path, device=device)
