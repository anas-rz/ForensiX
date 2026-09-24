from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn

from forensix.audio.io import AudioInput, load_audio
from forensix.types import Prediction


def split_windows(x: np.ndarray, window: int) -> np.ndarray:
    """Split a waveform into fixed-length windows, shape ``(n, window)``.

    Clips shorter than ``window`` are repeat-padded (as in the AASIST/RawNet
    evaluation code). Longer clips are cut into consecutive windows, with a
    final window aligned to the end so the tail is covered too.
    """
    if len(x) <= window:
        reps = window // len(x) + 1
        return np.tile(x, reps)[:window][None]
    starts = list(range(0, len(x) - window + 1, window))
    if starts[-1] + window < len(x):
        starts.append(len(x) - window)
    return np.stack([x[s : s + window] for s in starts])


class AudioDetector:
    """Base class for raw-waveform audio deepfake detectors.

    Subclasses implement ``_fake_probs`` for a batch of fixed-length windows.
    A clip's score is the mean fake probability over its windows.
    """

    sample_rate = 16000

    def __init__(
        self,
        name: str,
        network: nn.Module,
        window: int,
        device: str | torch.device = "cpu",
        threshold: float = 0.5,
    ):
        self.name = name
        self.window = window
        self.threshold = threshold
        self.device = torch.device(device)
        self.network = network.to(self.device).eval()

    def _fake_probs(self, windows: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, audio: AudioInput, sample_rate: int | None = None) -> Prediction:
        """Score one clip: a file path, or a waveform array plus ``sample_rate``."""
        return self.predict_batch([audio], sample_rate=sample_rate)[0]

    @torch.inference_mode()
    def predict_batch(
        self,
        inputs: Sequence[AudioInput],
        sample_rate: int | None = None,
        batch_size: int = 16,
    ) -> list[Prediction]:
        per_clip = [
            split_windows(load_audio(a, self.sample_rate, sample_rate), self.window)
            for a in inputs
        ]
        windows = np.concatenate(per_clip)
        probs = np.concatenate(
            [
                self._fake_probs(torch.from_numpy(chunk).to(self.device)).cpu().numpy()
                for chunk in _chunks(windows, batch_size)
            ]
        )
        preds, i = [], 0
        for clip in per_clip:
            seg = probs[i : i + len(clip)]
            i += len(clip)
            preds.append(
                Prediction(
                    model=self.name,
                    fake_prob=float(seg.mean()),
                    threshold=self.threshold,
                    segment_fake_probs=tuple(float(p) for p in seg),
                )
            )
        return preds


def _chunks(x: np.ndarray, size: int) -> Iterable[np.ndarray]:
    for i in range(0, len(x), size):
        yield x[i : i + size]
