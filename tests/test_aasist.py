"""Tests that need the pretrained weights (downloaded on first run)."""

import numpy as np
import pytest
import torch

import forensix


@pytest.fixture(scope="module", params=["aasist", "aasist-l"])
def detector(request):
    return forensix.load(request.param)


def test_weights_load_and_param_count(detector):
    n = sum(p.numel() for p in detector.network.parameters())
    assert n == {"aasist": 297_866, "aasist-l": 85_306}[detector.name]
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
    x = np.zeros(64600 * 2 + 100, dtype=np.float32) + 0.01
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
