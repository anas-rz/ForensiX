from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn

from forensix.audio.io import AudioInput, load_audio
from forensix.types import Prediction

CLIP_SCORES = ("mean", "first", "max-window")


def split_windows(x: np.ndarray, window: int, pad_short: bool = True) -> np.ndarray:
    """Split a waveform into fixed-length windows, shape ``(n, window)``.

    Clips shorter than ``window`` are repeat-padded (as in the AASIST/RawNet
    evaluation code), or returned whole if ``pad_short`` is False. Longer clips
    are cut into consecutive windows, with a final window aligned to the end so
    the tail is covered too.
    """
    if len(x) <= window:
        if not pad_short:
            return x[None]
        reps = window // len(x) + 1
        return np.tile(x, reps)[:window][None]
    starts = list(range(0, len(x) - window + 1, window))
    if starts[-1] + window < len(x):
        starts.append(len(x) - window)
    return np.stack([x[s : s + window] for s in starts])


class AudioDetector:
    """Base class for raw-waveform audio deepfake detectors.

    Subclasses implement ``_fake_probs`` for a batch of equal-length windows.
    A clip's score is the mean fake probability over its windows by default.
    ``clip_score="first"`` scores only the first window, as the upstream
    ASVspoof evaluation code does (on ASVspoof 2019 LA it halves ``aasist``'s
    EER, 1.58% -> 0.82%); ``"max-window"`` takes the most suspicious window. With
    ``pad_short=False`` (models scored on whole utterances), clips up to
    ``window`` samples are scored as they are and batched one clip at a time.
    """

    sample_rate = 16000

    def __init__(
        self,
        name: str,
        network: nn.Module,
        window: int,
        device: str | torch.device = "cpu",
        threshold: float = 0.5,
        pad_short: bool = True,
        clip_score: str = "mean",
    ):
        if clip_score not in CLIP_SCORES:
            raise ValueError(f"clip_score must be one of {CLIP_SCORES}, not {clip_score!r}")
        self.name = name
        self.clip_score = clip_score
        self.window = window
        self.pad_short = pad_short
        self.threshold = threshold
        self.device = torch.device(device)
        self.network = network.to(self.device).eval()

    def _fake_probs(self, windows: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def reduce(self, window_probs: np.ndarray) -> float:
        """The clip score from its windows' fake probabilities (see ``clip_score``)."""
        return float({"mean": np.mean, "first": lambda p: p[0], "max-window": np.max}[self.clip_score](window_probs))

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
            split_windows(
                load_audio(a, self.sample_rate, sample_rate), self.window, self.pad_short
            )
            for a in inputs
        ]
        # windows of different lengths can't share a batch
        groups = [np.concatenate(per_clip)] if self.pad_short else per_clip
        probs = np.concatenate(
            [
                self._fake_probs(torch.from_numpy(chunk).to(self.device)).cpu().numpy()
                for group in groups
                for chunk in _chunks(group, batch_size)
            ]
        )
        preds, i = [], 0
        for clip in per_clip:
            seg = probs[i : i + len(clip)]
            i += len(clip)
            preds.append(
                Prediction(
                    model=self.name,
                    fake_prob=self.reduce(seg),
                    threshold=self.threshold,
                    segment_fake_probs=tuple(float(p) for p in seg),
                )
            )
        return preds


def _chunks(x: np.ndarray, size: int) -> Iterable[np.ndarray]:
    for i in range(0, len(x), size):
        yield x[i : i + size]
