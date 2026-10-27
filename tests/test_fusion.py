import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import forensix
from forensix.fusion import FusedDetector, Fusion, Member, fit_fusion
from forensix.metrics import roc_auc
from forensix.types import Prediction

ROOT = Path(__file__).resolve().parents[1]


def synthetic(n=400, seed=0):
    """Two complementary members: one sees half of the fakes, the other the other half."""
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n), np.zeros(n)]
    kind = rng.integers(0, 2, 2 * n)  # which trace each fake leaves
    a = rng.normal(0, 1, 2 * n) + 2.5 * (y == 1) * (kind == 0)  # anomaly score
    b = rng.normal(0, 1, 2 * n) + 2.5 * (y == 1) * (kind == 1)
    return np.column_stack([a, 1 / (1 + np.exp(-b))]), y  # second member: a probability


def test_fusion_beats_its_members_and_round_trips(tmp_path):
    members = [Member("a", score_type="raw"), Member("b", {"clip_score": "max-window"})]
    s, y = synthetic()
    fusion = fit_fusion(s, y, members)
    st, yt = synthetic(seed=1)
    auc = lambda v: roc_auc(v[yt == 1], v[yt == 0])  # noqa: E731
    assert auc(fusion.logit(st)) > max(auc(st[:, 0]), auc(st[:, 1])) + 0.05
    assert all(w > 0 for w in fusion.weights) and members[1].key == "b-max-window"
    p = fusion.prob(st)
    assert 0.4 < np.mean(p >= 0.5) < 0.6  # balanced classes: half the test set is fake
    assert np.all(fusion.prob(st, fake_prior=0.1) < p)
    fusion.save(tmp_path / "f.json")
    again = Fusion.load(tmp_path / "f.json")
    assert again.members == members and np.allclose(again.logit(st), fusion.logit(st))


def test_missing_member_score_is_neutral():
    s, y = synthetic()
    fusion = fit_fusion(s, y, [Member("a", score_type="raw"), Member("b")])
    row = s[:1].copy()
    row[0, 1] = np.nan
    only_a = fusion.features(row)[0]
    assert only_a[1] == 0.0 and np.isfinite(fusion.logit(row)).all()


def test_fit_rejects_probability_outside_unit_interval():
    s, y = synthetic()
    with pytest.raises(ValueError, match="score_type='raw'"):
        fit_fusion(s, y, [Member("a"), Member("b")])


class _Fixed:
    def __init__(self, score):
        self.score = score

    def predict(self, x):
        if self.score is None:
            raise ValueError("no face")
        return Prediction(model="m", fake_prob=self.score)


def test_fused_detector_reports_members_and_survives_a_failure():
    s, y = synthetic()
    fusion = fit_fusion(s, y, [Member("a", score_type="raw"), Member("b")])
    pred = FusedDetector([_Fixed(4.0), _Fixed(None)], fusion).predict("x.mp4")
    assert pred.is_fake and "error" in pred.details["members"]["b"]
    assert pred.details["members"]["b"]["contribution"] == 0.0
    with pytest.raises(ValueError, match="Every fusion member failed"):
        FusedDetector([_Fixed(None), _Fixed(None)], fusion).predict("x.mp4")


def test_load_rejects_unknown_options():
    with pytest.raises(TypeError, match="doesn't take options"):
        forensix.load("phys-voice", clip_score="max-window")


def test_fit_fusion_script(tmp_path):
    rows = ["file,label,a,b-max-window"]
    for split, seed in (("fit", 0), ("test", 1)):
        s, y = synthetic(seed=seed)
        lines = [f"{i}.mp4,{'fake' if l else 'real'},{a},{'' if i == 3 else b}" for i, (l, (a, b)) in enumerate(zip(y, s))]
        (tmp_path / f"{split}.csv").write_text("\n".join(rows + lines) + "\n")
    out = subprocess.run([sys.executable, str(ROOT / "scripts/fit_fusion.py"), "--fit", str(tmp_path / "fit.csv"),
                          "--test", str(tmp_path / "test.csv"), "--members", "a", "b:clip_score=max-window",
                          "--out", str(tmp_path / "f.json"), "--report", str(tmp_path / "r.json")],
                         capture_output=True, text=True, check=True).stdout
    assert "fusion AUC" in out and "on 799 files" in out
    fusion = Fusion.load(tmp_path / "f.json")
    assert [m.score_type for m in fusion.members] == ["raw", "probability"]


@pytest.mark.skipif(not os.environ.get("FORENSIX_TEST_HEAVY"), reason="large model; set FORENSIX_TEST_HEAVY=1")
def test_avad_scores_every_segment_of_a_long_track():
    d = forensix.load("avad")
    rng = np.random.default_rng(0)
    crops = rng.integers(0, 255, (110, 224, 224, 3), dtype=np.uint8)
    audio = (rng.standard_normal(110 * 640) * 3000).astype(np.float32)
    r = d.analyze_track(crops, audio)
    assert r["n_windows"] == 106 and r["segment_start_s"] == [0.0, 1.0, 2.0, 2.24]  # last segment ends at the track's end
    assert abs(r["first_segment_score"] - d.score_track(crops, audio)) < 1e-6  # segment 0 = upstream's score
    assert abs(r["max_segment_score"] - max(r["segment_score"])) < 1e-4
    with pytest.raises(ValueError, match="clip_score"):
        forensix.load("avad", clip_score="median")
