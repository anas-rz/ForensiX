from os import PathLike
from typing import Sequence

from forensix.types import Prediction

VideoInput = str | PathLike


class VideoDetector:
    """Base class for video deepfake detectors.

    Subclasses implement ``_score`` for one video file, returning the video's
    fake probability and per-segment (e.g. per-frame or per-face) scores.
    """

    def __init__(self, name: str, threshold: float = 0.5):
        self.name = name
        self.threshold = threshold

    def _score(self, video: VideoInput) -> tuple[float, Sequence[float]]:
        raise NotImplementedError

    def predict(self, video: VideoInput) -> Prediction:
        prob, segments = self._score(video)
        return Prediction(
            model=self.name,
            fake_prob=float(prob),
            threshold=self.threshold,
            segment_fake_probs=tuple(float(s) for s in segments),
        )

    def predict_batch(self, videos: Sequence[VideoInput]) -> list[Prediction]:
        return [self.predict(v) for v in videos]
