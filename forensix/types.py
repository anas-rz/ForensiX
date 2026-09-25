from dataclasses import dataclass, field


@dataclass(frozen=True)
class Prediction:
    """Result of running a detector on one input.

    ``fake_prob`` is the detector's score: higher means more likely fake. With
    ``score_type="probability"`` it is the model's softmax/sigmoid output (not
    calibrated across models, so compare thresholds per model). With
    ``score_type="anomaly"`` it is an unbounded anomaly score, ``threshold`` is
    None and ``is_fake``/``label`` are None: pick a threshold on your own data.
    """

    model: str
    fake_prob: float
    threshold: float | None = 0.5
    segment_fake_probs: tuple[float, ...] = field(default_factory=tuple)
    score_type: str = "probability"

    @property
    def is_fake(self) -> bool | None:
        return None if self.threshold is None else self.fake_prob >= self.threshold

    @property
    def label(self) -> str | None:
        return None if self.threshold is None else ("fake" if self.is_fake else "real")
