"""Fuse several detectors into one: a logistic regression on their scores.

Detectors from different families look for different traces (lip-sync, face
artifacts, how the audio was made), so one often catches what another misses.
On ASVspoof 2019 LA, fusing a speaker check with a spoof detector cut the error
from 21-24% to 1.9% (identity_benchmark.md). ``fit_fusion`` learns the weights
from labelled scores; the result is a small JSON table, like a calibration.

Each member's score is put on a common scale first: ``logit(score)`` for
probabilities, the raw score for anomaly scores, then standardized with the
mean and std of the fitting data. A member that fails on an input (no face,
too short) gets 0 after standardization, its average, so it neither raises nor
lowers the fused score; the failure is listed in ``details``. The regression
weights classes equally, so the fused probability assumes a 50% fake rate and
0.5 is the balanced decision threshold; ``fake_prior`` moves it.

    fusion = fit_fusion({"stablesyncnet-max-window": s1, "gend-pe": s2}, labels,
                        members=[Member("stablesyncnet", {"clip_score": "max-window"}), Member("gend-pe")])
    fusion.save("fusions/av.json")
    det = forensix.load_fusion("fusions/av.json", device="cuda")
    det.predict("clip.mp4").details["members"]   # each member's score and contribution
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from forensix.calibration import _logit, _sigmoid
from forensix.types import Prediction

FORMAT = "forensix-fusion/1"


@dataclass
class Member:
    """One detector in a fusion: a registered model and its load options."""

    model: str
    options: dict = field(default_factory=dict)
    score_type: str = "probability"  # "raw" for anomaly scores (not in [0, 1])

    @property
    def key(self) -> str:
        """Column name of this member's scores, e.g. ``stablesyncnet-max-window``."""
        return "-".join([self.model, *(str(v) for v in self.options.values())])


@dataclass
class Fusion:
    members: list[Member]
    weights: list[float]  # per member, on standardized scores
    bias: float
    mean: list[float]  # standardization of each member's transformed score
    std: list[float]
    l2: float = 1.0
    source: str = ""
    created: str = field(default_factory=lambda: date.today().isoformat())
    format: str = FORMAT

    def features(self, scores: np.ndarray) -> np.ndarray:
        """(n, members) raw scores (NaN = member failed) -> standardized features."""
        scores = np.atleast_2d(np.asarray(scores, np.float64))
        x = np.column_stack([_logit(scores[:, j]) if m.score_type == "probability" else scores[:, j]
                             for j, m in enumerate(self.members)])
        z = (x - np.asarray(self.mean)) / np.asarray(self.std)
        return np.where(np.isnan(scores), 0.0, z)

    def logit(self, scores) -> np.ndarray:
        return self.features(scores) @ np.asarray(self.weights) + self.bias

    def prob(self, scores, fake_prior: float | None = None) -> np.ndarray:
        """Fused fake probability of (n, members) raw scores, at a 50% fake rate
        unless ``fake_prior`` is given."""
        z = self.logit(scores)
        if fake_prior is not None:
            z = z + _logit(fake_prior)
        return _sigmoid(z)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Fusion":
        if d.get("format") != FORMAT:
            raise ValueError(f"Not a fusion table (expected format {FORMAT!r})")
        d = dict(d)
        d["members"] = [Member(**m) for m in d["members"]]
        return cls(**d)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Fusion":
        return cls.from_dict(json.loads(Path(path).read_text()))


def fit_fusion(scores, labels, members: list[Member], l2: float = 1.0, source: str = "") -> Fusion:
    """Fit the fusion on raw member scores: ``scores`` (n, members) in the order
    of ``members`` (NaN where a member failed), ``labels`` 1 = fake, 0 = real.
    ``l2`` is the ridge penalty on the weights (in units of one example's loss),
    which keeps weights finite when members are correlated."""
    from scipy.optimize import minimize

    scores = np.atleast_2d(np.asarray(scores, np.float64))
    y = np.asarray(labels, np.float64)
    if scores.shape != (len(y), len(members)):
        raise ValueError(f"scores must be (n, {len(members)}), got {scores.shape}")
    if not (np.any(y == 1) and np.any(y == 0)):
        raise ValueError("Need fake and real examples")
    for j, m in enumerate(members):
        ok = scores[~np.isnan(scores[:, j]), j]
        if m.score_type == "probability" and (ok.min() < 0 or ok.max() > 1):
            raise ValueError(f"{m.key}: scores outside [0, 1]; use score_type='raw'")
    x = np.column_stack([_logit(scores[:, j]) if m.score_type == "probability" else scores[:, j]
                         for j, m in enumerate(members)])
    mean, std = np.nanmean(x, 0), np.nanstd(x, 0)
    std = np.where(std > 0, std, 1.0)
    z = np.where(np.isnan(x), 0.0, (x - mean) / std)
    w_cls = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))  # balanced classes

    def loss(p):
        t = z @ p[:-1] + p[-1]
        g = w_cls * (_sigmoid(t) - y)
        value = np.sum(w_cls * (np.logaddexp(0.0, t) - y * t)) + 0.5 * l2 * p[:-1] @ p[:-1]
        return value, np.r_[z.T @ g + l2 * p[:-1], g.sum()]

    p = minimize(loss, np.zeros(len(members) + 1), jac=True, method="L-BFGS-B").x
    return Fusion(members=list(members), weights=[float(v) for v in p[:-1]], bias=float(p[-1]),
                  mean=[float(v) for v in mean], std=[float(v) for v in std], l2=l2, source=source)


class FusedDetector:
    """Runs every member on the input and fuses their scores (``forensix.load_fusion``)."""

    def __init__(self, detectors: list, fusion: Fusion, name: str = "fusion"):
        self.detectors, self.fusion, self.name = detectors, fusion, name
        self.threshold = 0.5

    def predict(self, x, fake_prior: float | None = None) -> Prediction:
        raw, members = [], {}
        for m, det in zip(self.fusion.members, self.detectors):
            try:
                s = float(det.predict(x).fake_prob)
                members[m.key] = {"score": s}
            except Exception as e:  # no face, too short, no audio track
                s = float("nan")
                members[m.key] = {"error": f"{type(e).__name__}: {e}"}
            raw.append(s)
        if all(np.isnan(raw)):
            raise ValueError(f"Every fusion member failed on {x}: {members}")
        contrib = self.fusion.features(np.array([raw]))[0] * np.asarray(self.fusion.weights)
        for m, c in zip(self.fusion.members, contrib):
            members[m.key]["contribution"] = round(float(c), 4)  # to the fused logit
        p = float(self.fusion.prob(np.array([raw]), fake_prior)[0])
        return Prediction(model=self.name, fake_prob=p, threshold=self.threshold,
                          details={"members": members, "bias": self.fusion.bias})

    def predict_batch(self, xs) -> list[Prediction]:
        return [self.predict(x) for x in xs]
