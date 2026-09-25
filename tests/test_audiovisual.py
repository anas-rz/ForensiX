import os
import shutil
import subprocess

import numpy as np
import pytest

import forensix
from forensix.audiovisual.avad.features import log_mel, mel_filterbank
from forensix.audiovisual.facetrack import crop_track, track_faces

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def _av_clip(path, seconds=2):
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=30:duration={seconds}",
                    "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={seconds}",
                    "-shortest", "-pix_fmt", "yuv420p", str(path)], check=True)


@needs_ffmpeg
def test_read_av_resamples_to_25fps_and_16khz(tmp_path):
    from forensix.audiovisual.io import read_av

    _av_clip(tmp_path / "c.mp4")
    frames, audio = read_av(tmp_path / "c.mp4")
    assert frames.shape[1:] == (120, 160, 3) and frames.dtype == np.uint8
    # ffmpeg can add a frame or two at the stream end; predict() trims to the audio length
    assert abs(len(frames) - 50) <= 3 and abs(len(audio) - 32000) <= 1600
    assert 1000 < np.abs(audio).max() <= 32768  # int16 scale, as AVAD expects


def test_track_faces_links_and_interpolates():
    box = lambda i: [100 + i, 50, 220 + i, 190]
    detections = [[{"frame": i, "bbox": box(i)}] if i % 3 else [] for i in range(30)]
    detections[10].append({"frame": 10, "bbox": [0, 0, 10, 10]})  # a small far-away box
    tracks = track_faces(detections, min_track=5, min_face_size=50)
    assert len(tracks) == 1
    t = tracks[0]
    assert t["frame"][0] == 1 and t["frame"][-1] == 29 and len(t["frame"]) == 29
    assert np.allclose(t["bbox"][2], box(3))  # interpolated missing frame


def test_crop_track_shape_and_padding():
    frames = np.zeros((8, 240, 320, 3), np.uint8)
    track = {"frame": np.arange(8), "bbox": np.tile([10.0, 10.0, 110.0, 110.0], (8, 1))}  # near the border
    crops = crop_track(frames, track)
    assert crops.shape == (8, 224, 224, 3)
    assert (crops == 110).any()  # upstream's grey padding shows outside the frame


def test_mel_filterbank_and_log_mel():
    import torch

    mel = mel_filterbank()
    assert mel.shape == (80, 257) and (mel >= 0).all() and (mel.sum(1) > 0).all()
    feats = log_mel(torch.randn(2, 3200) * 1000)
    assert feats.shape == (2, 20, 80)


def test_audiovisual_registry():
    assert forensix.list_models("audiovisual") == ["avad"]


@pytest.mark.skipif(not os.environ.get("FORENSIX_TEST_HEAVY"), reason="large model; set FORENSIX_TEST_HEAVY=1")
def test_avad_scores_a_track():
    d = forensix.load("avad")
    rng = np.random.default_rng(0)
    crops = rng.integers(0, 255, (12, 224, 224, 3), dtype=np.uint8)
    audio = (rng.standard_normal(12 * 640) * 3000).astype(np.float32)
    score = d.score_track(crops, audio)
    assert np.isfinite(score) and score >= 0
