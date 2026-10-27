"""Post-hoc calibration of any detector's score: probabilities, conformal
false-alarm control, and drift checks. The detector itself (weights,
preprocessing) is unchanged; a calibration is a small JSON table fitted on
scores of labelled media.

Three tools, fitted together by ``fit_calibration``:

- **Platt scaling** (Platt, 1999): ``p = sigmoid(a * logit(score) + b)``, fitted
  by logistic regression with Platt's smoothed targets. For detectors whose
  score isn't a probability (anomaly scores, sync confidences), fit with
  ``score_type="raw"``: ``p = sigmoid(a * score + b)``. Gives probabilities
  that match the fake rate of the calibration set, and can be moved to another
  base rate with ``fake_prior``.
- **Split-conformal threshold** (Vovk et al., 2005; Angelopoulos & Bates,
  2021): from the real calibration scores only. A new real input is flagged
  with probability at most ``alpha``, whatever the fakes look like, as long as
  real inputs are exchangeable with the calibration reals. No fakes are needed,
  so the guarantee survives new generators.
- **Conformal p-values and prediction sets** (class-conditional / Mondrian):
  ``p_real`` is how typical the score is among real calibration scores,
  ``p_fake`` the same among fakes. The prediction set holds every class whose
  p-value exceeds ``alpha``: ``{"fake"}``, ``{"real"}``, both (uncertain) or
  neither (unlike both). Each class is covered with probability >= 1 - alpha.

Calibration does not fix ranking: AUC is unchanged by any monotone map. A
detector that can't separate a new kind of fake needs new training data, not
recalibration; ``drift_check`` tells you when the old calibration no longer holds.

    cal = fit_calibration("gend", fake_scores, real_scores, alpha=0.05)
    cal.save("calibrations/gend.json")
    det = forensix.load("gend", device="cuda", calibration="calibrations/gend.json")
    det.predict("clip.mp4").details["prediction_set"]   # e.g. ["fake"]
"""

import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from forensix.types import Prediction

FORMAT = "forensix-calibration/1"
_EPS = 1e-6


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return np.log(p) - np.log1p(-p)


def _sigmoid(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def _platt_input(scores, score_type: str) -> np.ndarray:
    if score_type == "probability":
        return _logit(scores)
    if score_type == "raw":
        return np.asarray(scores, dtype=np.float64)
    raise ValueError(f"Unknown score_type {score_type!r}")


def fit_platt(fake_scores, real_scores, score_type: str = "probability") -> tuple[float, float]:
    """``(a, b)`` of ``sigmoid(a * x + b)`` with ``x = logit(score)`` (or the raw
    score for ``score_type="raw"``), by logistic regression on Platt's smoothed
    targets ``(n_fake + 1) / (n_fake + 2)`` and ``1 / (n_real + 2)``."""
    from scipy.optimize import minimize

    fake, real = np.asarray(fake_scores, np.float64), np.asarray(real_scores, np.float64)
    if fake.size == 0 or real.size == 0:
        raise ValueError("Need at least one fake and one real score")
    x = _platt_input(np.concatenate([fake, real]), score_type)
    t = np.concatenate([np.full(fake.size, (fake.size + 1) / (fake.size + 2)),
                        np.full(real.size, 1 / (real.size + 2))])

    def loss(w):
        z = w[0] * x + w[1]
        # log(1 + e^z) - t z, written stably; gradient (sigmoid(z) - t) [x, 1]
        value = np.sum(np.logaddexp(0.0, z) - t * z)
        g = _sigmoid(z) - t
        return value, np.array([np.sum(g * x), np.sum(g)])

    a, b = minimize(loss, x0=np.array([1.0, 0.0]), jac=True, method="L-BFGS-B").x
    return float(a), float(b)


def conformal_threshold(real_scores, alpha: float) -> float:
    """Score ``c`` such that flagging ``score > c`` flags a new exchangeable real
    input with probability at most ``alpha``. ``inf`` when there are too few
    real scores for that ``alpha`` (fewer than ``1 / alpha - 1``)."""
    real = np.sort(np.asarray(real_scores, np.float64))
    n = real.size
    m = math.floor(alpha * (n + 1)) - 1  # reals allowed above the threshold
    if m < 0:
        return math.inf
    return float(real[n - 1 - m])


def conformal_p_value(scores, calibration_scores, higher_is_typical: bool = False) -> np.ndarray:
    """Conformal p-value of each score against calibration scores of one class.

    For the real class (default), a score is atypical when it is high:
    ``p = (1 + #{cal >= s}) / (n + 1)``. With ``higher_is_typical`` (the fake
    class), ``p = (1 + #{cal <= s}) / (n + 1)``."""
    cal = np.sort(np.asarray(calibration_scores, np.float64))
    s = np.atleast_1d(np.asarray(scores, np.float64))
    if higher_is_typical:
        count = np.searchsorted(cal, s, side="right")
    else:
        count = cal.size - np.searchsorted(cal, s, side="left")
    return (1.0 + count) / (cal.size + 1.0)


def expected_calibration_error(probs, labels, bins: int = 10) -> float:
    """ECE with equal-width bins: mean |fake rate - mean prob|, weighted by bin size."""
    probs, labels = np.asarray(probs, np.float64), np.asarray(labels, np.float64)
    idx = np.minimum((probs * bins).astype(int), bins - 1)
    return float(sum(abs(labels[idx == k].mean() - probs[idx == k].mean()) * np.mean(idx == k)
                     for k in range(bins) if np.any(idx == k)))


@dataclass
class Calibration:
    """A fitted calibration for one detector (see the module docstring)."""

    model: str
    platt: tuple[float, float]
    alpha: float
    threshold: float  # conformal: flag raw score > threshold
    real_scores: list[float]
    fake_scores: list[float]
    source: str = ""
    created: str = field(default_factory=lambda: date.today().isoformat())
    format: str = FORMAT
    score_type: str = "probability"  # "raw": Platt on the score itself, not its logit

    @property
    def calibration_prior(self) -> float:
        """Fake rate of the calibration set (the base rate Platt probabilities assume)."""
        return len(self.fake_scores) / (len(self.fake_scores) + len(self.real_scores))

    def prob(self, scores, fake_prior: float | None = None) -> np.ndarray:
        """Calibrated fake probability; ``fake_prior`` moves it to another base rate."""
        a, b = self.platt
        z = a * _platt_input(scores, self.score_type) + b
        if fake_prior is not None:
            z += _logit(fake_prior) - _logit(self.calibration_prior)
        return _sigmoid(z)

    def p_values(self, scores) -> tuple[np.ndarray, np.ndarray]:
        """``(p_real, p_fake)`` conformal p-values of raw scores."""
        return (conformal_p_value(scores, self.real_scores),
                conformal_p_value(scores, self.fake_scores, higher_is_typical=True))

    def prediction_set(self, score: float, alpha: float | None = None) -> list[str]:
        alpha = self.alpha if alpha is None else alpha
        p_real, p_fake = (float(p[0]) for p in self.p_values(score))
        return [label for label, p in (("real", p_real), ("fake", p_fake)) if p > alpha]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["platt"] = list(self.platt)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        if d.get("format") != FORMAT:
            raise ValueError(f"Not a calibration table (expected format {FORMAT!r})")
        d = dict(d)
        d["platt"] = tuple(d["platt"])
        return cls(**d)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        return cls.from_dict(json.loads(Path(path).read_text()))


def fit_calibration(model: str, fake_scores, real_scores, alpha: float = 0.05,
                    source: str = "", score_type: str = "probability") -> Calibration:
    """Fit Platt scaling and the conformal threshold on labelled raw scores.
    ``score_type="raw"`` for scores that aren't probabilities."""
    fake = np.asarray(fake_scores, np.float64)
    real = np.asarray(real_scores, np.float64)
    fake, real = fake[np.isfinite(fake)], real[np.isfinite(real)]
    a, b = fit_platt(fake, real, score_type)
    if a <= 0:
        raise ValueError(f"{model}: scores rank real above fake on the calibration data; "
                         "recalibration can't fix that")
    return Calibration(model=model, platt=(a, b), alpha=alpha,
                       threshold=conformal_threshold(real, alpha),
                       real_scores=[round(float(s), 6) for s in np.sort(real)],
                       fake_scores=[round(float(s), 6) for s in np.sort(fake)], source=source,
                       score_type=score_type)


def drift_check(calibration: Calibration, fake_scores=(), real_scores=(), level: float = 0.01) -> dict:
    """Does new labelled data still match the calibration?

    - reals: false-alarm rate at the conformal threshold vs ``alpha``
      (one-sided binomial test) and the score distribution vs the calibration
      reals (two-sample KS test);
    - fakes: detection rate at the threshold vs the calibration fakes'
      (one-sided Fisher exact test, catches fakes the detector stopped seeing),
      and the score distribution (KS).

    ``recalibrate`` is True when any test rejects at ``level``. A drop in the
    detection rate that recalibration can't recover shows up as a low AUC on the
    new data: that needs a retrained or different detector.
    """
    from scipy.stats import binomtest, fisher_exact, ks_2samp

    out = {"level": level, "alpha": calibration.alpha, "threshold": calibration.threshold, "tests": {}}
    real = np.asarray(real_scores, np.float64)
    real = real[np.isfinite(real)]
    if real.size:
        flagged = int(np.sum(real > calibration.threshold))
        out["n_real"] = int(real.size)
        out["false_alarm_rate"] = round(flagged / real.size, 4)
        out["tests"]["false_alarms_above_alpha"] = binomtest(
            flagged, real.size, calibration.alpha, alternative="greater").pvalue
        out["tests"]["real_scores_shifted"] = ks_2samp(real, calibration.real_scores).pvalue
    fake = np.asarray(fake_scores, np.float64)
    fake = fake[np.isfinite(fake)]
    if fake.size:
        cal_fake = np.asarray(calibration.fake_scores)
        new_hit, old_hit = int(np.sum(fake > calibration.threshold)), int(np.sum(cal_fake > calibration.threshold))
        out["n_fake"] = int(fake.size)
        out["detection_rate"] = round(new_hit / fake.size, 4)
        out["calibration_detection_rate"] = round(old_hit / cal_fake.size, 4)
        table = [[new_hit, fake.size - new_hit], [old_hit, cal_fake.size - old_hit]]
        out["tests"]["detection_rate_dropped"] = fisher_exact(table, alternative="less").pvalue
        out["tests"]["fake_scores_shifted"] = ks_2samp(fake, cal_fake).pvalue
    out["tests"] = {k: round(float(v), 6) for k, v in out["tests"].items()}
    out["recalibrate"] = any(v < level for v in out["tests"].values())
    return out


def group_folds(groups: list[str], folds: int, seed: int) -> np.ndarray:
    unique = sorted(set(groups))
    random.Random(seed).shuffle(unique)
    fold_of = {g: k % folds for k, g in enumerate(unique)}
    return np.array([fold_of[g] for g in groups])


def _decisions(scores, is_fake, flagged) -> dict:
    real_acc = float(np.mean(~flagged[~is_fake]))
    fake_acc = float(np.mean(flagged[is_fake]))
    return {"balanced_acc": round((real_acc + fake_acc) / 2, 4), "false_alarm_rate": round(1 - real_acc, 4),
            "detection_rate": round(fake_acc, 4)}


def _prob_metrics(probs, is_fake) -> dict:
    p = np.clip(probs, 1e-6, 1 - 1e-6)
    return {"brier": round(float(np.mean((p - is_fake) ** 2)), 4),
            "log_loss": round(float(-np.mean(np.where(is_fake, np.log(p), np.log1p(-p)))), 4),
            "ece": round(expected_calibration_error(p, is_fake), 4)}


def cross_validate(scores, is_fake, groups, alpha, folds, seed, raw_threshold, score_type="probability") -> dict:
    """Out-of-fold raw vs calibrated metrics (each fold calibrated on the others)."""
    fold = group_folds(groups, folds, seed)
    prob, flagged, p_real, p_fake = (np.zeros(len(scores)) for _ in range(4))
    for k in range(folds):
        test, train = fold == k, fold != k
        if not test.any():
            continue
        cal = fit_calibration("cv", scores[train & is_fake], scores[train & ~is_fake], alpha,
                              score_type=score_type)
        prob[test] = cal.prob(scores[test])
        flagged[test] = scores[test] > cal.threshold
        p_real[test], p_fake[test] = cal.p_values(scores[test])
    flagged = flagged.astype(bool)
    in_real, in_fake = p_real > alpha, p_fake > alpha
    covered = np.where(is_fake, in_fake, in_real)
    size = in_real.astype(int) + in_fake.astype(int)
    raw = {"threshold": None}  # a raw score has no default threshold or probability
    if score_type == "probability":
        raw = {"threshold": raw_threshold, **_decisions(scores, is_fake, scores >= raw_threshold),
               **_prob_metrics(scores, is_fake)}
    return {
        "raw": raw,
        "platt": {**_decisions(scores, is_fake, prob >= 0.5), **_prob_metrics(prob, is_fake)},
        "conformal": {"alpha": alpha, **_decisions(scores, is_fake, flagged)},
        "prediction_sets": {"coverage_real": round(float(covered[~is_fake].mean()), 4),
                            "coverage_fake": round(float(covered[is_fake].mean()), 4),
                            "single_label": round(float(np.mean(size == 1)), 4),
                            "both_labels": round(float(np.mean(size == 2)), 4),
                            "empty": round(float(np.mean(size == 0)), 4)},
    }


def calibrate_scores(model: str, scores, is_fake, groups=None, alpha: float = 0.05, folds: int = 5,
                     seed: int = 0, source: str = "", score_type: str | None = None,
                     raw_threshold: float | None = 0.5) -> tuple["Calibration", dict]:
    """Fit a calibration on labelled raw scores, with an out-of-fold report.

    ``groups`` keeps related items in one cross-validation fold. ``score_type``
    defaults to ``"raw"`` when any score is outside [0, 1]. Returns
    ``(calibration, report)``; the report has the counts, AUC/AP, the
    ``cross_validate`` metrics and warnings about thin data.
    """
    from forensix.metrics import binary_summary

    scores, is_fake = np.asarray(scores, np.float64), np.asarray(is_fake, bool)
    groups = [str(i) for i in range(len(scores))] if groups is None else [str(g) for g in groups]
    ok = np.isfinite(scores)
    scores, is_fake, groups = scores[ok], is_fake[ok], [g for g, k in zip(groups, ok) if k]
    if score_type is None:
        score_type = "raw" if np.any((scores < 0) | (scores > 1)) else "probability"
    report = {"model": model, "n_real": int((~is_fake).sum()), "n_fake": int(is_fake.sum()),
              "score_type": score_type, "alpha": alpha, "warnings": []}
    if report["n_real"] < 2 or report["n_fake"] < 2:
        raise ValueError(f"{model}: need at least 2 scored real and 2 scored fake items, "
                         f"got {report['n_real']} and {report['n_fake']}")
    report.update({k: v for k, v in binary_summary(scores[is_fake], scores[~is_fake], None).items()
                   if k in ("auc", "ap")})
    n_groups = len(set(groups))
    if n_groups >= 2:
        try:
            report["cv"] = cross_validate(scores, is_fake, groups, alpha, min(folds, n_groups), seed,
                                          0.5 if raw_threshold is None else raw_threshold, score_type)
        except ValueError as e:  # a fold's training part lacks a class
            report["warnings"].append(f"cross-validation skipped: {e}")
    cal = fit_calibration(model, scores[is_fake], scores[~is_fake], alpha, source, score_type)
    report.update(conformal_threshold=None if math.isinf(cal.threshold) else cal.threshold, platt=list(cal.platt))
    if math.isinf(cal.threshold):
        report["warnings"].append(f"{report['n_real']} real items are too few for alpha={alpha}: "
                                  f"the conformal threshold never flags; add real items or raise alpha")
    if min(report["n_real"], report["n_fake"]) < 50:
        report["warnings"].append("fewer than 50 items in a class: probabilities and rates are rough")
    if report["auc"] < 0.6:
        report["warnings"].append(f"AUC {report['auc']:.2f}: the model barely separates this data; "
                                  "calibration can't fix that")
    return cal, report


def keep_window(new: np.ndarray, old: np.ndarray, window: int | None, seed: int = 0) -> np.ndarray:
    """Indices into ``old`` to keep next to all of ``new``: everything when ``window``
    is None, else a seeded random subset so that new + kept old <= ``window``."""
    if window is None or len(new) + len(old) <= window:
        return np.arange(len(old))
    keep = max(window - len(new), 0)
    return np.sort(np.random.default_rng(seed).choice(len(old), size=min(keep, len(old)), replace=False))


def update_calibration(base: "Calibration", fake_scores=(), real_scores=(), window: int | None = None,
                       alpha: float | None = None, seed: int = 0, source: str = "") -> tuple["Calibration", dict]:
    """Build on an existing calibration: refit on its stored scores plus new labelled
    scores of the same model. ``window`` caps the scores kept per class, keeping all
    new ones and a random subset of the old, so the calibration follows changing
    data. ``alpha`` defaults to the base's. Returns ``(calibration, report)``.

        new, report = update_calibration(Calibration.load("cal.json"), new_fake, new_real, window=2000)
    """
    fake_new, real_new = np.asarray(fake_scores, np.float64), np.asarray(real_scores, np.float64)
    fake_old, real_old = np.asarray(base.fake_scores), np.asarray(base.real_scores)
    fake_old = fake_old[keep_window(fake_new, fake_old, window, seed)]
    real_old = real_old[keep_window(real_new, real_old, window, seed + 1)]
    fake, real = np.concatenate([fake_new, fake_old]), np.concatenate([real_new, real_old])
    cal, report = calibrate_scores(base.model, np.concatenate([fake, real]),
                                   np.r_[np.ones(len(fake), bool), np.zeros(len(real), bool)],
                                   alpha=base.alpha if alpha is None else alpha,
                                   source=", ".join(x for x in (base.source, source) if x),
                                   score_type=base.score_type)
    report.update(n_new_real=int(len(real_new)), n_new_fake=int(len(fake_new)),
                  n_carried_real=int(len(real_old)), n_carried_fake=int(len(fake_old)))
    return cal, report


def calibrate(model, real_media=None, fake_media=None, dataset=None, alpha: float = 0.05, folds: int = 5,
              seed: int = 0, source: str = "", score_type: str | None = None, device: str = "cpu",
              **load_options) -> tuple["Calibration", dict]:
    """Score labelled media with a detector and fit its calibration in one call.

    Give either ``dataset`` (a folder or ZIP in the ForensiX dataset format, see
    ``forensix.dataset``) or ``real_media`` and ``fake_media`` lists. ``model`` is
    a registered name (loaded on ``device`` with ``load_options``) or a loaded,
    uncalibrated detector. Media the detector can't score are skipped and listed
    in the report.

        cal, report = forensix.calibrate("dfdet-clip", dataset="my_data.zip", device="cuda")
        cal.save("calibrations/dfdet-clip.json")
        det = forensix.load("dfdet-clip", device="cuda", calibration="calibrations/dfdet-clip.json")
    """
    import tempfile

    from forensix.dataset import Item, extract_dataset, read_dataset

    with tempfile.TemporaryDirectory() as tmp:
        if dataset is not None:
            root = Path(dataset)
            if root.is_file():
                root = extract_dataset(root, tmp)
            data = read_dataset(root)
            items, source = data.items, source or f"dataset:{Path(dataset).name}"
        elif real_media is not None and fake_media is not None:
            items = ([Item(Path(m), "real", str(m)) for m in real_media]
                     + [Item(Path(m), "fake", str(m)) for m in fake_media])
        else:
            raise ValueError("Give a dataset, or both real_media and fake_media")
        if isinstance(model, str):
            import forensix

            detector, name = forensix.load(model, device=device, **load_options), model
        else:
            detector, name = model, model.name
        scores, skipped, anomaly = np.full(len(items), np.nan), [], False
        for i, item in enumerate(items):
            try:
                pred = (detector.predict(item.path, identity=item.identity) if item.identity
                        else detector.predict(item.path))
            except Exception as e:  # undecodable, no face, too short
                skipped.append({"file": str(item.path), "error": f"{type(e).__name__}: {e}"})
                continue
            scores[i], anomaly = pred.fake_prob, anomaly or pred.score_type == "anomaly"
    cal, report = calibrate_scores(name, scores, [i.is_fake for i in items], [i.group for i in items], alpha,
                                   folds, seed, source, score_type or ("raw" if anomaly else None),
                                   getattr(detector, "threshold", None))
    report["skipped"] = skipped
    return cal, report


class CalibratedDetector:
    """A detector with a calibration applied to its scores.

    ``fake_prob`` becomes the calibrated probability (at ``fake_prior``, default
    the calibration set's fake rate). ``rule`` sets the decision:

    - ``"conformal"`` (default): fake when the raw score is above the conformal
      threshold, i.e. at most ``alpha`` false alarms on real media;
    - ``"probability"``: fake when the calibrated probability is >= 0.5.

    ``details`` holds the raw score, both conformal p-values and the
    prediction set; the wrapped detector's own details are kept under ``"detector"``.
    """

    def __init__(self, detector, calibration: Calibration, rule: str = "conformal",
                 fake_prior: float | None = None):
        if rule not in ("conformal", "probability"):
            raise ValueError(f"Unknown rule {rule!r}")
        self.detector, self.calibration, self.rule, self.fake_prior = detector, calibration, rule, fake_prior
        self.name = detector.name

    @property
    def threshold(self) -> float:
        if self.rule == "probability":
            return 0.5
        if math.isinf(self.calibration.threshold):
            return math.inf  # too few real scores for this alpha: never flag
        # score > c  <=>  prob(score) > prob(c), Platt being increasing
        return float(np.nextafter(float(self.calibration.prob(self.calibration.threshold, self.fake_prior)), 2.0))

    def calibrate(self, pred: Prediction) -> Prediction:
        cal, score = self.calibration, pred.fake_prob
        p_real, p_fake = (float(p[0]) for p in cal.p_values(score))
        details = {"raw_score": score, "p_real": round(p_real, 6), "p_fake": round(p_fake, 6),
                   "prediction_set": cal.prediction_set(score), "alpha": cal.alpha, "rule": self.rule,
                   "calibration": {"source": cal.source, "created": cal.created,
                                   "n_real": len(cal.real_scores), "n_fake": len(cal.fake_scores)}}
        if pred.details:
            details["detector"] = pred.details
        return Prediction(model=pred.model, fake_prob=float(cal.prob(score, self.fake_prior)),
                          threshold=self.threshold, segment_fake_probs=pred.segment_fake_probs,
                          score_type="probability", details=details)

    def predict(self, *args, **kwargs) -> Prediction:
        return self.calibrate(self.detector.predict(*args, **kwargs))

    def predict_batch(self, *args, **kwargs) -> list[Prediction]:
        return [self.calibrate(p) for p in self.detector.predict_batch(*args, **kwargs)]
