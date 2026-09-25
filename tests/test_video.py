import os

import numpy as np
import pytest

import forensix
from forensix.video.dfdc_selim import confident_strategy, isotropically_resize, put_to_center
from forensix.video.io import read_frames

cv2 = pytest.importorskip("cv2")


def _write_video(path, n_frames, size=(64, 48)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, size)
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), i * 4 % 255, np.uint8))
    writer.release()


def test_read_frames_evenly_spaced(tmp_path):
    _write_video(tmp_path / "v.mp4", 100)
    frames, idx = read_frames(tmp_path / "v.mp4", 5)
    assert frames.shape == (5, 48, 64, 3) and frames.dtype == np.uint8
    assert idx == [0, 24, 49, 74, 99]


def test_read_frames_short_video_reads_each_frame_once(tmp_path):
    _write_video(tmp_path / "v.mp4", 10)
    frames, idx = read_frames(tmp_path / "v.mp4", 32)
    assert idx == list(range(10)) and len(frames) == 10


def test_read_frames_errors(tmp_path):
    (tmp_path / "bad.mp4").write_bytes(b"not a video")
    with pytest.raises(ValueError):
        read_frames(tmp_path / "bad.mp4", 4)


def test_resize_and_center():
    face = np.ones((100, 50, 3), np.uint8)
    resized = isotropically_resize(face, 380)
    assert resized.shape[:2] == (380, 190)
    out = put_to_center(resized, 380)
    assert out.shape == (380, 380, 3)
    assert out[:, :95].sum() == 0 and out[:, 95:285].all()


def test_confident_strategy():
    confident_fake = np.array([0.9] * 20 + [0.1] * 12)
    assert confident_strategy(confident_fake) == pytest.approx(0.9)
    confident_real = np.array([0.1] * 30 + [0.5] * 2)
    assert confident_strategy(confident_real) == pytest.approx(0.1)
    mixed = np.array([0.9] * 5 + [0.5] * 27)
    assert confident_strategy(mixed) == pytest.approx(mixed.mean())


def test_video_registry():
    assert "dfdc-selim" in forensix.list_models("video")
    assert len(forensix.info("dfdc-selim").extra_weights) == 6 + 3


@pytest.mark.skipif(not os.environ.get("FORENSIX_TEST_HEAVY"), reason="large model; set FORENSIX_TEST_HEAVY=1")
def test_dfdc_selim_no_face_video(tmp_path):
    _write_video(tmp_path / "v.mp4", 40, size=(320, 240))
    pred = forensix.load("dfdc-selim").predict(tmp_path / "v.mp4")
    assert pred.fake_prob == 0.5 and pred.segment_fake_probs == ()
