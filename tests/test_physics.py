from pathlib import Path

import numpy as np
import pytest

import forensix
from forensix.physics.bloodflow import BloodFlowDetector, _violation
from forensix.physics.signals.rppg import chrom, coherence, heart_rate, pos

FACE = Path(__file__).parent / "data" / "face_public_domain.jpg"
FPS = 30.0


def _skin_trace(bpm, seconds=12, seed=0, amplitude=0.006):
    t = np.arange(int(FPS * seconds)) / FPS
    rng = np.random.default_rng(seed)
    illum = 1 + 0.02 * np.sin(2 * np.pi * 0.1 * t)  # slow lighting drift
    pulse = np.sin(2 * np.pi * bpm / 60 * t)[:, None] * np.array([0.33, 0.77, 0.53])  # PPG colour signature
    return np.array([180.0, 120.0, 100.0]) * illum[:, None] * (1 + amplitude * pulse) + rng.normal(0, 0.3, (len(t), 3))


@pytest.mark.parametrize("method", [pos, chrom])
@pytest.mark.parametrize("bpm", [55, 72, 110])
def test_rppg_recovers_heart_rate(method, bpm):
    hr, snr = heart_rate(method(_skin_trace(bpm), FPS), FPS)
    assert abs(hr - bpm) < 2.5 and snr > -3


def test_coherence_separates_shared_pulse_from_noise():
    a, b = _skin_trace(72, seed=1), _skin_trace(72, seed=2)
    noise = _skin_trace(72, seed=3, amplitude=0.0)  # same lighting and sensor noise, no pulse
    c, _ = coherence({"a": pos(a, FPS), "b": pos(b, FPS), "n": pos(noise, FPS)}, FPS)
    assert c[0, 1] > 0.5 and abs(c[0, 2]) < 0.3


def test_violation_ramp():
    assert _violation(0.8, 0.5, 0.0) == 0.0 and _violation(0.25, 0.5, 0.0) == 0.5
    assert _violation(30, 5, 20) == 1.0 and _violation(3, 5, 20) == 0.0


def test_physics_registry():
    assert "phys-bloodflow" in forensix.list_models("physics")


@pytest.fixture(scope="module")
def face_clips():
    pytest.importorskip("mediapipe")
    import cv2
    from PIL import Image

    from forensix.physics.face.tracker import track_face

    img = np.array(Image.open(FACE)).astype(np.float64)
    track = track_face(img.astype(np.uint8)[None], FPS)
    assert track.valid.all()
    hull = cv2.convexHull(track.landmarks[0, :, :2].astype(np.int32))
    face = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(face, [hull], 1)
    face = face.astype(bool)
    t = np.arange(int(FPS * 8)) / FPS
    rng = np.random.default_rng(0)

    def clip(pulse: bool):
        frames = []
        for ti in t:
            f = img.copy()
            if pulse:
                f[face] *= 1 + 0.008 * np.sin(2 * np.pi * 1.2 * ti) * np.array([0.33, 0.77, 0.53])
            frames.append(np.clip(f + rng.normal(0, 1.0, f.shape), 0, 255).astype(np.uint8))
        return np.stack(frames)

    return clip(True), clip(False)


def test_bloodflow_on_face_clips(face_clips):
    with_pulse, without = face_clips
    det = BloodFlowDetector()
    real = det.analyze(with_pulse, FPS)
    fake = det.analyze(without, FPS)
    assert abs(real["heart_rate_bpm"] - 72) < 3
    assert real["score"] < 0.1 < fake["score"]
    assert set(real["constraints"]) == {"mean_coherence", "hr_spread_bpm", "median_snr_db"}
    assert set(real["regions"]) == {"forehead", "left_cheek", "right_cheek", "nose", "chin"}


def test_bloodflow_rejects_short_video(face_clips):
    with pytest.raises(ValueError, match="at least"):
        BloodFlowDetector().analyze(face_clips[0][:60], FPS)
