"""Shared pieces of the physdf detectors."""

import numpy as np

from forensix.types import Prediction


def violation(value: float, start: float, full: float) -> float:
    """0 inside the expected range, rising linearly to 1 at ``full`` (works in
    either direction: ``full < start`` means lower values are worse)."""
    return float(np.clip((value - start) / (full - start), 0.0, 1.0))


Prior = tuple  # (start, full) or (start, full, source)


def constraint_report(measured: dict[str, float], priors: dict[str, Prior]) -> tuple[float, dict]:
    """Score = the largest violation among applicable constraints (one clearly
    broken physical law is evidence on its own), plus per-constraint evidence.
    A NaN measurement means the constraint doesn't apply to this clip (e.g. the
    mouth never moves); it is reported but left out of the score. Each prior is
    ``(start, full)`` or ``(start, full, source)``, source "provisional" or "learned"."""
    evidence, violations = {}, []
    for k, value in measured.items():
        start, full, *source = priors[k]
        entry = {"expected_range_start": start, "worse": "higher" if full > start else "lower",
                 "prior": source[0] if source else "provisional"}
        if value is not None and np.isfinite(value):
            v = violation(value, start, full)
            violations.append(v)
            entry.update(measured=round(float(value), 4), violation=round(v, 3))
        else:
            entry.update(measured=None, violation=None, applicable=False)
        evidence[k] = entry
    return (float(max(violations)) if violations else 0.0), evidence


class _PriorsMixin:
    """How a detector turns measurements into a score.

    - A **reference** (``forensix.physics.reference``), fitted on real media: calibrated scores
      (0.99 = more unusual than 99% of real media) and a decision threshold. The packaged
      reference is used where it covers the detector.
    - Otherwise **priors**: the learned priors table, else the hand-set provisional ranges
      (``provisional_priors``); the score is the largest violation and there's no threshold.

    ``load_priors(path_or_table)`` switches: a reference table enables reference scoring, a
    priors table (``forensix.physics.priors``) forces range scoring."""

    name = ""
    provisional_priors: dict[str, tuple[float, float]] = {}

    def __init__(self, priors=None):
        self.load_priors(priors)

    def load_priors(self, priors=None) -> None:
        """``priors``: a reference or learned-priors JSON path or dict; None uses the packaged
        reference (else the packaged priors table, else the provisional ranges)."""
        from forensix.physics import reference
        from forensix.physics.priors import resolve

        is_ref = reference.is_reference(priors)
        table = reference.load(priors) if priors is None or is_ref else None
        self.reference = (table or {}).get("detectors", {}).get(self.name)
        self.priors = resolve(self.name, self.provisional_priors, None if is_ref else priors)

    @property
    def decision_threshold(self) -> float | None:
        from forensix.physics.reference import THRESHOLD

        return THRESHOLD if self.reference else None

    @property
    def scoring(self) -> str:
        return "reference" if self.reference else "provisional"

    def report(self, measured: dict[str, float]) -> tuple[float, dict]:
        if self.reference:
            from forensix.physics.reference import score

            return score(measured, self.reference, self.priors)
        return constraint_report(measured, self.priors)


class PhysicsVideoDetector(_PriorsMixin):
    """Base for physdf video detectors: ``analyze(frames, fps)`` returns a dict
    with ``score`` and evidence; ``predict(path)`` wraps it in a Prediction."""

    threshold = None  # anomaly score

    def analyze(self, frames: np.ndarray, fps: float, track=None) -> dict:
        raise NotImplementedError

    def predict(self, video) -> Prediction:
        from forensix.physics.io import read_video

        frames, fps = read_video(video)
        result = self.analyze(frames, fps)
        result.setdefault("scoring", self.scoring)
        return Prediction(model=self.name, fake_prob=result["score"], threshold=self.decision_threshold,
                          score_type="anomaly", details=result)

    def predict_batch(self, videos) -> list[Prediction]:
        return [self.predict(v) for v in videos]


class PhysicsAudioDetector(_PriorsMixin):
    """Base for physdf audio detectors: ``analyze(x, sr)`` on 16 kHz mono audio."""

    threshold = None
    sample_rate = 16000

    def analyze(self, x: np.ndarray, sr: int) -> dict:
        raise NotImplementedError

    def predict(self, audio, sample_rate: int | None = None) -> Prediction:
        from forensix.audio.io import load_audio

        x = load_audio(audio, self.sample_rate, sample_rate)
        result = self.analyze(x, self.sample_rate)
        result.setdefault("scoring", self.scoring)
        return Prediction(model=self.name, fake_prob=result["score"], threshold=self.decision_threshold,
                          score_type="anomaly", details=result)

    def predict_batch(self, inputs, sample_rate: int | None = None) -> list[Prediction]:
        return [self.predict(a, sample_rate) for a in inputs]
