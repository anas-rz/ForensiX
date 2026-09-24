from dataclasses import dataclass, field


@dataclass(frozen=True)
class Prediction:
    """Result of running a detector on one input.

    ``fake_prob`` is the model's softmax probability that the input is
    manipulated/synthetic. It is not calibrated across models, so compare
    thresholds per model rather than across models.
    """

    model: str
    fake_prob: float
    threshold: float = 0.5
    segment_fake_probs: tuple[float, ...] = field(default_factory=tuple)

    @property
    def is_fake(self) -> bool:
        return self.fake_prob >= self.threshold

    @property
    def label(self) -> str:
        return "fake" if self.is_fake else "real"
