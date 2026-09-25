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
    assert forensix.list_models("audiovisual") == ["avad", "avcons", "avid", "stablesyncnet", "syncnet"]


@pytest.mark.skipif(not os.environ.get("FORENSIX_TEST_HEAVY"), reason="large model; set FORENSIX_TEST_HEAVY=1")
def test_avad_scores_a_track():
    d = forensix.load("avad")
    rng = np.random.default_rng(0)
    crops = rng.integers(0, 255, (12, 224, 224, 3), dtype=np.uint8)
    audio = (rng.standard_normal(12 * 640) * 3000).astype(np.float32)
    score = d.score_track(crops, audio)
    assert np.isfinite(score) and score >= 0


def test_syncnet_mfcc_matches_python_speech_features():
    from forensix.audiovisual.syncnet import mfcc

    t = np.arange(8000) / 16000
    x = (3000 * np.sin(2 * np.pi * 220 * t) + 500 * np.sin(2 * np.pi * 1800 * t)).astype(np.int16)
    m = mfcc(x, 16000)
    # recorded from python_speech_features 0.6 (mfcc with defaults)
    ref = [[16.757903, 14.212825, -11.049129, 12.676961, 29.628861, -12.494635, -44.819697, -15.429081,
            6.359702, -28.607264, -50.440945, -19.699293, 2.79041],
           [16.766431, 16.672268, -12.103196, 11.77231, 31.622658, -9.946454, -43.408424, -14.46609,
            7.361561, -27.550775, -49.227266, -18.475373, 3.338462]]
    assert m.shape == (49, 13)
    assert np.allclose(m[[0, 20]], ref, atol=1e-5)


@pytest.mark.skipif(not os.environ.get("FORENSIX_TEST_HEAVY"), reason="downloads weights; set FORENSIX_TEST_HEAVY=1")
def test_syncnet_offsets_follow_the_audio():
    d = forensix.load("syncnet")
    rng = np.random.default_rng(0)
    crops = rng.integers(0, 255, (40, 224, 224, 3), dtype=np.uint8)
    audio = (rng.standard_normal(40 * 640) * 3000).astype(np.float32)
    r = d.analyze_track(crops, audio)
    assert r["n_windows"] == 35 and len(r["mean_dist_by_offset"]) == 31
    assert -15 <= r["offset_frames"] <= 15 and r["confidence"] >= 0


def test_stablesyncnet_mel_matches_latentsync():
    from forensix.audiovisual.stablesyncnet import melspectrogram

    t = np.arange(8000) / 16000
    x = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.sin(2 * np.pi * 1800 * t)).astype(np.float32)
    m = melspectrogram(x)
    # recorded from LatentSync latentsync/utils/audio.py (librosa 0.10.1)
    idx = [3, 4, 5, 40, 41, 42]
    assert m.shape == (80, 41)
    assert np.allclose(m[idx, 5], [1.005585, 1.10741, -0.731591, -0.916239, 1.127457, 0.088067], atol=1e-5)
    assert np.allclose(m[idx, 0], [0.828685, 0.870242, 0.29037, 0.119264, 0.935831, 0.462345], atol=1e-5)


def test_stablesyncnet_alignment_maps_points_to_template():
    from forensix.audiovisual.stablesyncnet import FACE_TEMPLATE, similarity_from_points

    angle, scale, shift = np.deg2rad(12), 0.6, np.array([40.0, -25.0])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = (FACE_TEMPLATE - FACE_TEMPLATE.mean(0)) @ rot.T * scale + FACE_TEMPLATE.mean(0) + shift
    m, bias = similarity_from_points(points)
    mapped = points @ m[:, :2].T + m[:, 2]
    # exact similarity onto the template, plus upstream's smoothing term: the nose's
    # residual in standardized (unrotated) coordinates, added to the translation
    assert np.allclose(mapped - bias, FACE_TEMPLATE, atol=1e-3)
    m0, bias0 = similarity_from_points(FACE_TEMPLATE)
    assert np.allclose(m0, [[1, 0, 0], [0, 1, 0]], atol=1e-5) and np.allclose(bias0, 0, atol=1e-6)
    # temporal smoothing: 0.2 of the previous frame's bias + 0.8 of this frame's
    _, b = similarity_from_points(points, p_bias=np.array([1.0, -1.0]))
    assert np.allclose(b, 0.2 * np.array([1.0, -1.0]) + 0.8 * bias, atol=1e-5)


@pytest.mark.skipif(not os.environ.get("FORENSIX_TEST_HEAVY"), reason="1.6 GB weights; set FORENSIX_TEST_HEAVY=1")
def test_stablesyncnet_scores_windows_and_offsets():
    d = forensix.load("stablesyncnet")
    rng = np.random.default_rng(0)
    faces = rng.integers(0, 255, (40, 256, 256, 3), dtype=np.uint8)
    wav = rng.standard_normal(40 * 640).astype(np.float32) * 0.1
    sims, offsets = d.sync_probabilities(faces, wav)
    assert offsets == list(range(-6, 7)) and sims.shape == (1, 13)
    assert ((sims >= 0) & (sims <= 1)).all()  # ReLU'd embeddings: cosine similarity in [0, 1]


# ---- avcons ---------------------------------------------------------------------

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from avcons_synth import D as SYNTH_D  # noqa: E402
from avcons_synth import clip as synth_clip  # noqa: E402

from forensix.audiovisual.avcons.model import AVConsistency, distance_profile  # noqa: E402
from forensix.audiovisual.avcons.negatives import SHIFT_RANGE, negative, positive  # noqa: E402


def test_avcons_negatives_and_labels():
    rng = np.random.default_rng(0)
    c, donor = synth_clip(0), synth_clip(1)
    p = positive(c, 125, rng)
    assert p.label == 0 and p.window_labels.sum() == 0 and len(p.v) == len(p.a) <= 125
    for kind in ("shift", "swap", "face_swap"):
        e = negative(c, donor, 125, rng, kind)
        assert e.label == 1 and e.window_labels.all() and len(e.v) == len(e.a)
    e = negative(c, donor, 200, rng, "drift")
    assert e.window_labels[0] == 0 and e.window_labels[-1] == 1  # the offset grows over time
    assert np.all(np.diff(e.window_labels) >= 0)
    for kind, stream in (("splice", "a"), ("visual_splice", "v")):
        e = negative(c, donor, 125, rng, kind)
        changed = np.any(getattr(e, stream) != (c.a if stream == "a" else c.v)[: len(e.v)], axis=1) \
            if len(c.v) == len(e.v) else None
        assert e.label == 1 and 0 < e.window_labels.mean() < 1
        if changed is not None:
            assert e.window_labels[changed].all()  # every altered window is labelled
    # a shift negative's best offset is at least SHIFT_RANGE[0] away
    e = negative(c, donor, 125, rng, "shift")
    d, valid = distance_profile(torch.from_numpy(e.v)[None], torch.from_numpy(e.a)[None], 15)
    mean = torch.where(valid, d, torch.nan).nanmean(1)[0]
    assert abs(int(torch.argmin(mean)) - 15) >= SHIFT_RANGE[0]


def test_avcons_distance_profile_and_padding():
    torch.manual_seed(0)
    v, a = torch.randn(1, 20, 8), torch.randn(1, 20, 8)
    d, valid = distance_profile(v, a, 3)
    assert torch.allclose(d[0, 5, 3 + 2], (v[0, 5] - a[0, 7]).norm()) and not valid[0, 0, 0] and valid[0, 5, 0]
    model = AVConsistency(emb_dim=8, dim=16, layers=1, heads=2, dropout=0.0).eval()
    clip_a, win_a = model(v, a)
    vp, ap = torch.cat([v, torch.randn(1, 7, 8)], 1), torch.cat([a, torch.randn(1, 7, 8)], 1)
    mask = torch.zeros(1, 27, dtype=torch.bool)
    mask[:, :20] = True
    clip_b, win_b = model(vp, ap, mask)
    assert torch.allclose(clip_a, clip_b, atol=1e-5) and torch.allclose(win_a, win_b[:, :20], atol=1e-5)


def test_avcons_learns_consistency_and_checkpoint(tmp_path):
    from forensix.audiovisual.avcons.train import evaluate, load, save, train

    torch.manual_seed(0)
    clips = [synth_clip(i) for i in range(40)]
    model, _ = train(clips, steps=150, batch=32, dim=32, layers=2, heads=2, emb_dim=SYNTH_D, lr=2e-3,
                     log_every=150, log=lambda m: None)
    r = evaluate(model, [synth_clip(1000 + i) for i in range(10)], per_kind=30)
    assert r["shift"]["auc"] > 0.9 and r["swap"]["auc"] > 0.9
    save(tmp_path / "m.safetensors", model, {"test": True})
    loaded, info = load(tmp_path / "m.safetensors")
    x = torch.from_numpy(clips[0].v[:50])[None], torch.from_numpy(clips[0].a[:50])[None]
    assert info == {"test": True} and torch.allclose(model(*x)[0], loaded(*x)[0])


def test_avcons_needs_trained_weights():
    with pytest.raises(ValueError, match="train_avcons"):
        forensix.load("avcons")


# ---- avid -----------------------------------------------------------------------

from avid_synth import clips as avid_clips  # noqa: E402


def _tone():
    t = np.arange(32000) / 16000
    return torch.from_numpy((0.3 * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
                             + 0.05 * np.sin(2 * np.pi * 2300 * t)).astype(np.float32))[None]


def test_avid_fbank_matches_speechbrain():
    from forensix.audiovisual.avid.ecapa import Fbank

    f = Fbank(n_mels=80)(_tone())
    # recorded from speechbrain 1.1.1 lobes.features.Fbank(n_mels=80)
    assert tuple(f.shape) == (1, 201, 80)
    assert np.allclose(f[0, [0, 50, 200], [3, 20, 60]].numpy(), [11.8341, -13.7875, -21.4415], atol=1e-3)


@pytest.mark.skipif(not os.environ.get("FORENSIX_TEST_HEAVY"), reason="downloads weights; set FORENSIX_TEST_HEAVY=1")
def test_avid_ecapa_matches_speechbrain():
    from forensix.audiovisual.avid import ECAPA_WEIGHTS
    from forensix.audiovisual.avid.ecapa import SpeakerEncoder
    from forensix.hub import fetch, torch_load

    m = SpeakerEncoder()
    m.embedding_model.load_state_dict(torch_load(fetch(ECAPA_WEIGHTS)))
    e = m.eval()(_tone())[0]
    # recorded from speechbrain 1.1.1 (Fbank, sentence InputNormalization, ECAPA_TDNN)
    assert np.allclose(e[[0, 1, 50, 191]].numpy(), [-78.2654, 23.8692, 62.1704, -3.0184], atol=1e-2)
    assert abs(float(e.norm()) - 379.396) < 0.05


def test_avid_negatives():
    from forensix.audiovisual.avid.train import negative

    rng = np.random.default_rng(0)
    a, b = avid_clips(2, per_speaker=1, segments=8)
    swap = negative(a, b, "voice_swap", rng)
    assert np.array_equal(swap.voice, b.voice) and np.array_equal(swap.face, a.face)
    for kind, stream in (("voice_splice", "voice"), ("face_splice", "face")):
        e = negative(a, b, kind, rng)
        same = np.all(getattr(e, stream) == getattr(a, stream), axis=1)
        assert same[0] and not same[-1] and np.all(np.diff(same.astype(int)) <= 0)  # one switch point
        other = "face" if stream == "voice" else "voice"
        assert np.array_equal(getattr(e, other), getattr(a, other))


def test_avid_learns_identity_and_checkpoint(tmp_path):
    from forensix.audiovisual.avid.train import evaluate, fit_scorer, load, save, train_heads

    torch.manual_seed(0)
    heads = train_heads(avid_clips(150), steps=200, batch=64, log=lambda m: None)
    scorer = fit_scorer(heads, avid_clips(30, offset=1000))
    r = evaluate(heads, scorer, avid_clips(30, offset=2000))
    assert r["matching_1_of_2"] > 0.8
    assert r["voice_swap"]["auc"] > 0.9 and r["voice_splice"]["auc"] > 0.8 and r["face_splice"]["auc"] > 0.8
    save(tmp_path / "a.safetensors", heads, scorer, {"test": 1})
    h2, s2, info = load(tmp_path / "a.safetensors")
    v, f = torch.randn(4, 192), torch.randn(4, 384)
    assert info == {"test": 1} and torch.allclose(heads.embed_voice(v), h2.embed_voice(v))
    x = torch.randn(3, 6)
    assert torch.allclose(scorer(x), s2(x))


def test_avid_needs_trained_weights():
    with pytest.raises(ValueError, match="train_avid"):
        forensix.load("avid")
