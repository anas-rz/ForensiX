"""Shared pieces of the physdf detectors."""

import numpy as np

from forensix.types import Prediction


def violation(value: float, start: float, full: float) -> float:
    """0 inside the expected range, rising linearly to 1 at ``full`` (works in
    either direction: ``full < start`` means lower values are worse)."""
    return float(np.clip((value - start) / (full - start), 0.0, 1.0))


def constraint_report(measured: dict[str, float], priors: dict[str, tuple[float, float]]) -> tuple[float, dict]:
    """Score = the largest violation among applicable constraints (one clearly
    broken physical law is evidence on its own), plus per-constraint evidence.
    A NaN measurement means the constraint doesn't apply to this clip (e.g. the
    mouth never moves); it is reported but left out of the score."""
    evidence, violations = {}, []
    for k, value in measured.items():
        start, full = priors[k]
        entry = {"expected_range_start": start, "prior": "provisional"}
        if np.isfinite(value):
            v = violation(value, start, full)
            violations.append(v)
            entry.update(measured=round(float(value), 4), violation=round(v, 3))
        else:
            entry.update(measured=None, violation=None, applicable=False)
        evidence[k] = entry
    return (float(max(violations)) if violations else 0.0), evidence


class PhysicsVideoDetector:
    """Base for physdf video detectors: ``analyze(frames, fps)`` returns a dict
    with ``score`` and evidence; ``predict(path)`` wraps it in a Prediction."""

    name = ""
    threshold = None  # anomaly score

    def analyze(self, frames: np.ndarray, fps: float, track=None) -> dict:
        raise NotImplementedError

    def predict(self, video) -> Prediction:
        from forensix.physics.io import read_video

        frames, fps = read_video(video)
        result = self.analyze(frames, fps)
        return Prediction(model=self.name, fake_prob=result["score"], threshold=None,
                          score_type="anomaly", details=result)

    def predict_batch(self, videos) -> list[Prediction]:
        return [self.predict(v) for v in videos]
