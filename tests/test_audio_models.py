"""Tests that need the pretrained weights (downloaded on first run).

The 316M-parameter XLS-R-based models (1.2 GB each) only run with FORENSIX_TEST_HEAVY=1.
"""

import os

import numpy as np
import pytest
import torch

import forensix

HEAVY = {"xlsr-aasist", "xlsr-aasist-df", "nes2net-x", "nes2net-x-itw"}
MODELS = [
    pytest.param(m, marks=pytest.mark.skipif(
        m in HEAVY and not os.environ.get("FORENSIX_TEST_HEAVY"),
        reason="large model; set FORENSIX_TEST_HEAVY=1",
    ))
    for m in forensix.list_models("audio")
]
PARAMS = {
    "aasist": 297_866,
    "aasist-l": 85_306,
    "rawgat-st": 440_810,
    "xlsr-aasist": 315_884_938,
    "xlsr-aasist-df": 315_884_938,
    "nes2net-x": 315_949_735,
    "nes2net-x-itw": 315_949_735,
}


@pytest.fixture(scope="module", params=MODELS)
def detector(request):
    return forensix.load(request.param)


def test_weights_load_and_param_count(detector):
    n = sum(p.numel() for p in detector.network.parameters())
    assert n == PARAMS[detector.name]
    assert not detector.network.training


def test_predict_array_and_file_agree(detector, tmp_path):
    import soundfile as sf

    x = (np.random.default_rng(0).standard_normal(48000) * 0.05).astype(np.float32)
    sf.write(tmp_path / "x.wav", x, 16000, subtype="FLOAT")
    a = detector.predict(x, sample_rate=16000)
    b = detector.predict(tmp_path / "x.wav")
    assert 0.0 <= a.fake_prob <= 1.0
    assert a.fake_prob == pytest.approx(b.fake_prob, abs=1e-6)
    assert a.model == detector.name


def test_long_clip_gives_one_score_per_window(detector):
    x = np.zeros(detector.window * 2 + 100, dtype=np.float32) + 0.01
    pred = detector.predict(x, sample_rate=16000)
    assert len(pred.segment_fake_probs) == 3
    assert pred.fake_prob == pytest.approx(np.mean(pred.segment_fake_probs))


def test_batch_matches_single(detector):
    rng = np.random.default_rng(1)
    clips = [(rng.standard_normal(n) * 0.05).astype(np.float32) for n in (16000, 70000)]
    batch = detector.predict_batch(clips, sample_rate=16000, batch_size=2)
    for clip, p in zip(clips, batch):
        single = detector.predict(clip, sample_rate=16000)
        assert p.fake_prob == pytest.approx(single.fake_prob, abs=1e-5)


def test_deterministic(detector):
    x = torch.randn(1, 64600).numpy() * 0.05
    p1 = detector.predict(x[0], sample_rate=16000).fake_prob
    p2 = detector.predict(x[0], sample_rate=16000).fake_prob
    assert p1 == p2


def test_clip_score_rules():
    rng = np.random.default_rng(2)
    x = (rng.standard_normal(64600 * 2 + 5000) * 0.05).astype(np.float32)
    seg = np.array(forensix.load("aasist").predict(x, sample_rate=16000).segment_fake_probs)
    for rule, expect in (("first", seg[0]), ("max-window", seg.max())):
        det = forensix.load("aasist", clip_score=rule)
        assert det.predict(x, sample_rate=16000).fake_prob == pytest.approx(expect, abs=1e-6)
    with pytest.raises(ValueError, match="clip_score"):
        forensix.load("aasist", clip_score="median")
