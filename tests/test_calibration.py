import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import forensix
from forensix.calibration import (
    Calibration,
    CalibratedDetector,
    conformal_p_value,
    conformal_threshold,
    drift_check,
    expected_calibration_error,
    fit_calibration,
    fit_platt,
)
from forensix.types import Prediction

SCRIPT = Path(__file__).parents[1] / "scripts" / "recalibrate.py"


def scores(rng, n, mean, sd=1.0):
    """Miscalibrated detector scores: sigmoid of a Gaussian logit."""
    return 1 / (1 + np.exp(-rng.normal(mean, sd, n)))


def test_conformal_threshold_controls_false_alarms():
    rng = np.random.default_rng(0)
    alpha, n_cal, trials = 0.1, 50, 2000
    rates = []
    for _ in range(trials):
        c = conformal_threshold(rng.standard_normal(n_cal), alpha)
        rates.append(np.mean(rng.standard_normal(1000) > c))
    # marginal guarantee: E[false alarm rate] <= alpha (here (n+1-k)/(n+1) = 5/51)
    assert np.mean(rates) <= alpha + 0.005
    assert np.mean(rates) > alpha - 0.02  # and not needlessly conservative
    assert conformal_threshold(np.arange(5.0), 0.1) == np.inf  # too few reals for alpha


def test_conformal_p_values():
    cal = np.arange(1.0, 10.0)  # 9 calibration scores
    assert conformal_p_value(100.0, cal)[0] == pytest.approx(1 / 10)  # above all reals: most atypical
    assert conformal_p_value(0.0, cal)[0] == pytest.approx(1.0)
    assert conformal_p_value(0.0, cal, higher_is_typical=True)[0] == pytest.approx(1 / 10)
    rng = np.random.default_rng(1)
    # valid p-values on exchangeable data: P(p <= 0.1) = 0.1, marginally over calibration draws
    rate = np.mean([np.mean(conformal_p_value(rng.standard_normal(200), rng.standard_normal(49)) <= 0.1)
                    for _ in range(1000)])
    assert abs(rate - 0.1) < 0.005


def test_platt_fixes_overconfident_scores():
    rng = np.random.default_rng(2)
    fake, real = scores(rng, 2000, 3.0, 4.0), scores(rng, 2000, -3.0, 4.0)  # overconfident
    a, b = fit_platt(fake, real)
    assert 0 < a < 1  # shrinks the logits
    cal = fit_calibration("m", fake, real)
    probs = np.concatenate([fake, real])
    labels = np.r_[np.ones(2000), np.zeros(2000)]
    test_fake, test_real = scores(rng, 2000, 3.0, 4.0), scores(rng, 2000, -3.0, 4.0)
    test = np.concatenate([test_fake, test_real])
    assert expected_calibration_error(cal.prob(test), labels) < expected_calibration_error(probs, labels) / 2
    # base-rate shift: 1% fakes lowers every probability
    assert np.all(cal.prob(test, fake_prior=0.01) < cal.prob(test))


def test_fit_rejects_inverted_detector():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="rank real above fake"):
        fit_calibration("m", scores(rng, 100, -2), scores(rng, 100, 2))


def test_save_load_and_prediction_sets(tmp_path):
    rng = np.random.default_rng(4)
    cal = fit_calibration("m", scores(rng, 200, 2), scores(rng, 200, -2), alpha=0.05, source="synthetic")
    cal.save(tmp_path / "m.json")
    loaded = Calibration.load(tmp_path / "m.json")
    assert loaded == cal
    assert cal.prediction_set(0.999) == ["fake"] and cal.prediction_set(0.001) == ["real"]
    with pytest.raises(ValueError, match="Not a calibration"):
        Calibration.from_dict({"format": "other"})


class _Fixed:
    name, threshold = "fixed", 0.5

    def __init__(self, score):
        self.score = score

    def predict(self, x):
        return Prediction(model=self.name, fake_prob=self.score, details={"own": 1})

    def predict_batch(self, xs):
        return [self.predict(x) for x in xs]


def test_calibrated_detector_decision_matches_conformal_threshold():
    rng = np.random.default_rng(5)
    cal = fit_calibration("fixed", scores(rng, 300, 1), scores(rng, 300, -1), alpha=0.05)
    for s in [cal.threshold - 1e-4, cal.threshold + 1e-4, 0.01, 0.99]:
        pred = CalibratedDetector(_Fixed(s), cal).predict("x")
        assert pred.is_fake == (s > cal.threshold)
        assert pred.details["raw_score"] == s and pred.details["detector"] == {"own": 1}
        assert set(pred.details["prediction_set"]) <= {"real", "fake"}
    pred = CalibratedDetector(_Fixed(0.7), cal, rule="probability").predict("x")
    assert pred.threshold == 0.5 and pred.is_fake == (float(cal.prob(0.7)) >= 0.5)


def test_load_with_calibration_checks_model(tmp_path):
    rng = np.random.default_rng(6)
    path = tmp_path / "other.json"
    fit_calibration("other-model", scores(rng, 50, 1), scores(rng, 50, -1)).save(path)
    with pytest.raises(ValueError, match="is for 'other-model'"):
        forensix.load("phys-voice", calibration=path)


def test_drift_check():
    rng = np.random.default_rng(7)
    cal = fit_calibration("m", scores(rng, 500, 2), scores(rng, 500, -2), alpha=0.05)
    same = drift_check(cal, scores(rng, 300, 2), scores(rng, 300, -2))
    assert not same["recalibrate"]
    shifted_reals = drift_check(cal, real_scores=scores(rng, 300, 0))  # e.g. a new camera
    assert shifted_reals["recalibrate"] and shifted_reals["false_alarm_rate"] > 0.05
    new_fakes = drift_check(cal, fake_scores=scores(rng, 300, -1))  # a generator it misses
    assert new_fakes["recalibrate"] and new_fakes["tests"]["detection_rate_dropped"] < 0.01


def _write_scores(path, rng, fake_mean, n=60):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "det"])
        for i in range(n):
            w.writerow([f"v{i}.mp4", "real", scores(rng, 1, -1)[0]])
            w.writerow([f"vs{i}.mp4", "fake", scores(rng, 1, fake_mean)[0]])


def test_recalibrate_script(tmp_path):
    rng = np.random.default_rng(8)
    _write_scores(tmp_path / "a.csv", rng, 1.5)
    cal_dir, report = tmp_path / "cal", tmp_path / "report.json"
    subprocess.run([sys.executable, SCRIPT, "fit", "--scores", tmp_path / "a.csv", "--group-regex", r"(\d+)",
                    "--out-dir", cal_dir, "--report", report], check=True)
    r = json.loads(report.read_text())["models"]["det"]
    assert r["n_real"] == r["n_fake"] == 60 and set(r["cv"]) == {"raw", "platt", "conformal", "prediction_sets"}
    assert Calibration.load(cal_dir / "det.json").model == "det"

    _write_scores(tmp_path / "same.csv", rng, 1.5)
    ok = subprocess.run([sys.executable, SCRIPT, "check", "--calibration-dir", cal_dir,
                         "--scores", tmp_path / "same.csv"])
    assert ok.returncode == 0
    _write_scores(tmp_path / "new.csv", rng, -1.0, n=200)  # fakes the detector no longer sees
    drifted = subprocess.run([sys.executable, SCRIPT, "check", "--calibration-dir", cal_dir,
                              "--scores", tmp_path / "new.csv"])
    assert drifted.returncode == 3
    subprocess.run([sys.executable, SCRIPT, "update", "--calibration-dir", cal_dir, "--scores", tmp_path / "new.csv",
                    "--window", "150", "--only-if-drift"], check=True)
    updated = Calibration.load(cal_dir / "det.json")
    assert len(updated.fake_scores) == len(updated.real_scores) == 150
    assert len(list((cal_dir / "history").glob("det-*.json"))) == 1
