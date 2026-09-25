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
    assert set(forensix.list_models("video")) == {"dfdc-selim", "dfdc-wm", "gend", "gend-pe", "dfdet-clip"}
    assert len(forensix.info("dfdc-selim").extra_weights) == 6 + 3


@pytest.mark.skipif(not os.environ.get("FORENSIX_TEST_HEAVY"), reason="large model; set FORENSIX_TEST_HEAVY=1")
def test_dfdc_selim_no_face_video(tmp_path):
    _write_video(tmp_path / "v.mp4", 40, size=(320, 240))
    pred = forensix.load("dfdc-selim").predict(tmp_path / "v.mp4")
    assert pred.fake_prob == 0.5 and pred.segment_fake_probs == ()


def test_align_face_maps_landmarks_to_template():
    from forensix.video.faces import _TEMPLATE, align_face

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (400, 300, 3), dtype=np.uint8)
    # landmarks = template placed at 1.5x scale with a 30 degree rotation and shift
    theta = np.deg2rad(30)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    lms = (_TEMPLATE * 120) @ rot.T + [120, 150]
    face = align_face(img, lms, size=256, scale=1.3)
    assert face.shape == (256, 256, 3)
    # warping the landmarks with the same transform lands them on the scaled template
    dst = (_TEMPLATE * 256 + 256 * 0.15) * (256 / (256 * 1.3))
    matrix = cv2.estimateAffinePartial2D(lms.astype(np.float32), dst.astype(np.float32), method=cv2.LMEDS)[0]
    warped = cv2.transform(lms[None].astype(np.float32), matrix)[0]
    assert np.abs(warped - dst).max() < 1e-3


@pytest.mark.skipif(not os.environ.get("FORENSIX_TEST_HEAVY"), reason="large model; set FORENSIX_TEST_HEAVY=1")
@pytest.mark.parametrize("name, params", [("gend", 303_181_826), ("gend-pe", 316_104_706), ("dfdet-clip", 303_181_826)])
def test_gend_family(tmp_path, name, params):
    d = forensix.load(name)
    assert sum(p.numel() for p in d.network.parameters()) == params
    _write_video(tmp_path / "v.mp4", 40, size=(320, 240))
    assert d.predict(tmp_path / "v.mp4").fake_prob == 0.5
    faces = [np.random.default_rng(i).integers(0, 255, (256, 256, 3), dtype=np.uint8) for i in range(3)]
    probs = d.face_fake_probs(faces)
    assert probs.shape == (3,) and ((0 <= probs) & (probs <= 1)).all()


def test_similarity_transform_recovers_known_transform():
    from forensix.video.retinaface import ARCFACE_320, norm_crop, similarity_transform

    theta, scale, shift = np.deg2rad(-12), 0.7, np.array([35.0, -20.0])
    rot = scale * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    src = (ARCFACE_320 - shift) @ np.linalg.inv(rot).T  # points that map exactly onto the template
    matrix = similarity_transform(src, ARCFACE_320)
    assert np.allclose(matrix[:, :2], rot) and np.allclose(src @ matrix[:, :2].T + matrix[:, 2], ARCFACE_320)
    assert norm_crop(np.zeros((400, 400, 3), np.uint8), src).shape == (320, 320, 3)


@pytest.mark.skipif(not os.environ.get("FORENSIX_TEST_HEAVY"), reason="large model; set FORENSIX_TEST_HEAVY=1")
def test_dfdc_wm(tmp_path):
    d = forensix.load("dfdc-wm")
    counts = [sum(p.numel() for p in m.parameters()) for m in (d.xception, d.wsdan_x, d.wsdan_e, d.face_detector.net)]
    assert counts == [20_811_050, 20_856_120, 12_270_112, 27_293_600]
    _write_video(tmp_path / "v.mp4", 40, size=(320, 240))
    assert d.predict(tmp_path / "v.mp4").fake_prob == 0.5
    faces = [np.random.default_rng(i).integers(0, 255, (320, 320, 3), dtype=np.uint8) for i in range(3)]
    probs = d.face_fake_probs(faces)
    assert probs.shape == (3,) and ((0 <= probs) & (probs <= 1)).all()
