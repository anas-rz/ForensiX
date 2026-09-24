import numpy as np
import pytest

import forensix
from forensix.audio.detector import split_windows
from forensix.audio.io import load_audio
from forensix.metrics import compute_eer
from forensix.types import Prediction


def test_registry():
    assert {"aasist", "aasist-l"} <= set(forensix.list_models("audio"))
    assert forensix.list_models("video") == []
    assert forensix.info("aasist").license == "MIT"
    with pytest.raises(KeyError, match="Unknown model"):
        forensix.info("nope")


def test_prediction_label():
    assert Prediction("m", 0.7).label == "fake"
    assert Prediction("m", 0.2).label == "real"
    assert Prediction("m", 0.2, threshold=0.1).is_fake


def test_split_windows_short_clip_is_repeat_padded():
    w = split_windows(np.arange(3, dtype=np.float32), 7)
    assert w.tolist() == [[0, 1, 2, 0, 1, 2, 0]]


def test_split_windows_long_clip_covers_tail():
    w = split_windows(np.arange(10, dtype=np.float32), 4)
    assert w.shape == (3, 4)
    assert w[-1].tolist() == [6, 7, 8, 9]
    assert split_windows(np.arange(8, dtype=np.float32), 4).shape == (2, 4)


def test_load_audio_resamples_and_downmixes(tmp_path):
    import soundfile as sf

    stereo = np.random.default_rng(0).uniform(-0.5, 0.5, (8000, 2)).astype(np.float32)
    sf.write(tmp_path / "a.wav", stereo, 8000)
    x = load_audio(tmp_path / "a.wav", target_sr=16000)
    assert x.dtype == np.float32 and x.shape == (16000,)
    with pytest.raises(ValueError, match="sample_rate"):
        load_audio(stereo, target_sr=16000)


def test_eer():
    assert compute_eer([0.9, 0.8], [0.1, 0.2])[0] == 0.0
    assert compute_eer([0.1, 0.2], [0.9, 0.8])[0] == 1.0
    eer, _ = compute_eer([0.9, 0.3], [0.1, 0.5])
    assert eer == pytest.approx(0.5)
