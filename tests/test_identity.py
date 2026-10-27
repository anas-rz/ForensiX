import os
from pathlib import Path

import numpy as np
import pytest

import forensix
from forensix.identity.gallery import Gallery, top_k_similarity

DATA = Path(__file__).parent / "data"
ARMSTRONG, ARMSTRONG_2, ALDRIN = DATA / "face_public_domain.jpg", DATA / "face2_public_domain.jpg", DATA / "other_face_public_domain.jpg"


def test_gallery_roundtrip_and_model_check(tmp_path):
    g = Gallery("model-a")
    rng = np.random.default_rng(0)
    g.add("alice", rng.standard_normal((4, 8)), ["a.jpg"] * 4)
    g.add("alice", rng.standard_normal((2, 8)), ["b.mp4"] * 2)
    g.add("bob", rng.standard_normal((3, 8)), ["c.jpg"] * 3)
    assert g.names() == ["alice", "bob"] and g.embeddings("alice").shape == (6, 8)
    assert np.allclose(np.linalg.norm(g.embeddings("bob"), axis=1), 1)
    g.save(tmp_path / "g.npz")
    back = Gallery.load(tmp_path / "g.npz", "model-a")
    assert back.names() == g.names() and np.allclose(back.embeddings("alice"), g.embeddings("alice"))
    assert back.people["alice"]["sources"] == ["a.jpg"] * 4 + ["b.mp4"] * 2
    with pytest.raises(ValueError, match="model-a"):
        Gallery.load(tmp_path / "g.npz", "model-b")
    with pytest.raises(KeyError, match="carol"):
        g.embeddings("carol")


def test_top_k_similarity():
    refs = np.eye(4, dtype=np.float32)
    q = np.array([[1.0, 1.0, 0, 0]])
    assert np.allclose(top_k_similarity(q, refs, k=2), 1 / np.sqrt(2))
    assert np.allclose(top_k_similarity(q, refs, k=4), 0.5 / np.sqrt(2))


@pytest.fixture(scope="module")
def id_face():
    return forensix.load("id-face")


def _video(path: Path, images: list[Path], seconds_each: float = 3.0, fps: int = 10) -> Path:
    import cv2

    frames = [cv2.resize(cv2.imread(str(p)), (320, 320)) for p in images]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 320))
    for img in frames:
        for t in range(int(seconds_each * fps)):
            m = cv2.getRotationMatrix2D((160, 160), 4 * np.sin(t / 5), 1.0)
            writer.write(cv2.warpAffine(img, m, (320, 320), borderMode=cv2.BORDER_REFLECT))
    writer.release()
    return path


def test_id_face_verifies_and_rejects(id_face):
    id_face.gallery = Gallery("sface-2021dec")
    id_face.enroll("armstrong", ARMSTRONG)
    same = id_face.predict(ARMSTRONG_2, identity="armstrong")
    other = id_face.predict(ALDRIN, identity="armstrong")
    assert same.is_fake is False and other.is_fake is True
    assert same.details["similarity"] > 0.5 and other.details["similarity"] < 0.3
    id_face.enroll("aldrin", ALDRIN)
    ranking = id_face.identify(ARMSTRONG_2)
    assert ranking[0][0] == "armstrong"
    assert other.details["frames_below_threshold"] == 1.0


def test_id_face_video_enrollment_and_face_switch(id_face, tmp_path):
    id_face.gallery = Gallery("sface-2021dec")
    info = id_face.enroll("armstrong", _video(tmp_path / "ref.avi", [ARMSTRONG]))
    assert info["n_embeddings"] >= 5
    switched = id_face.predict(_video(tmp_path / "switch.avi", [ARMSTRONG_2, ALDRIN]), identity="armstrong")
    sims = np.array(switched.details["frame_similarity"])
    half = len(sims) // 2
    assert sims[:half].min() > 0.363 and sims[half + 1 :].max() < 0.363  # the timeline shows the switch
    assert 0.35 < switched.details["frames_below_threshold"] < 0.65
    path = tmp_path / "gallery.npz"
    id_face.save_gallery(path)
    fresh = forensix.load("id-face")
    fresh.load_gallery(path)
    assert fresh.gallery.names() == ["armstrong"]


# ---- id-motion ------------------------------------------------------------------

import sys  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from motion_synth import clip as motion_clip  # noqa: E402


def _track(seconds=6.0, fps=30.0, yaw_deg=10.0, freq=0.5):
    from scipy.spatial.transform import Rotation

    from forensix.physics.face.tracker import FaceTrack

    t = np.arange(int(seconds * fps)) / fps
    rot = Rotation.from_euler("y", yaw_deg * np.sin(2 * np.pi * freq * t)[:, None], degrees=True)
    pose = np.tile(np.eye(4), (len(t), 1, 1))
    pose[:, :3, :3] = rot.as_matrix()
    pose[:, 2, 3] = -50.0
    lm = np.zeros((len(t), 478, 3))
    lm[:, 33, 0], lm[:, 263, 0] = 100.0, 160.0  # eye corners: face width 60 px
    lm[:, 1, 0] = 130 + 6 * np.sin(2 * np.pi * freq * t)  # nose tip sways by 0.1 face widths
    names = [f"bs{i}" for i in range(52)]
    bs = np.zeros((len(t), 52))
    bs[:, 3] = 0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * t)
    return FaceTrack(fps, np.ones(len(t), bool), lm, bs, names, pose)


def test_motion_features_resample_and_pose():
    from forensix.identity.motion.features import motion_features

    f, v = motion_features(_track())
    assert f.shape[1] == 64 and v.all() and abs(len(f) - 6.0 * 25) <= 1
    yaw = f[:, 1]  # rotation vector y (degrees), relative to the mean
    assert 9.0 < yaw.max() < 11.0 and -11.0 < yaw.min() < -9.0
    assert abs(f[:, 6].max() - 0.1) < 0.01  # nose sway in face widths
    assert 0.95 < f[:, 12 + 3].max() <= 1.0 and f[:, 12 + 3].min() < 0.05  # blendshape kept


def test_motion_encoder_masking_and_supcon():
    from forensix.identity.motion.model import MotionEncoder, supcon_loss

    torch.manual_seed(0)
    enc = MotionEncoder(channels=32, dilations=(1, 2)).eval()
    x = torch.randn(2, 50, 64)
    mask = torch.ones(2, 50, dtype=torch.bool)
    mask[:, 20] = False
    x2 = x.clone()
    x2[:, 20] = 99.0  # an untracked frame's values don't matter
    assert torch.allclose(enc(x, mask), enc(x2, mask), atol=1e-5)
    e = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)
    right = supcon_loss(e.repeat(2, 1), torch.arange(8).repeat(2))  # each item paired with its copy
    wrong = supcon_loss(e.repeat(2, 1), torch.cat([torch.arange(8), torch.arange(8).roll(1)]))  # with another item
    assert right < wrong


def test_id_motion_learns_behaviour_and_checkpoint(tmp_path):
    from forensix.identity.motion.train import evaluate, load, save, train

    torch.manual_seed(0)
    enc = train([motion_clip(p, v) for p in range(60) for v in range(3)], steps=150, people_per_batch=32, lr=2e-3,
                log=lambda m: None, channels=64, dilations=(1, 2, 4, 8))
    r = evaluate(enc, [motion_clip(p, v) for p in range(1000, 1015) for v in range(3)])
    assert r["eer"] < 0.2 and r["genuine_mean"] > r["impostor_mean"] + 0.2
    save(tmp_path / "m.safetensors", enc, r["threshold"], {"t": 1})
    enc2, thr, info = load(tmp_path / "m.safetensors")
    x = torch.randn(1, 100, 64)
    assert thr == r["threshold"] and info == {"t": 1} and torch.allclose(enc(x), enc2(x))

    det = forensix.load("id-motion", weights_path=tmp_path / "m.safetensors")
    assert det.tag.startswith("id-motion:") and det.threshold == thr
    with pytest.raises(ValueError, match="photo"):
        det.enroll("armstrong", ARMSTRONG)
    det.gallery.add("p", np.ones((2, 128)), ["x", "y"])
    det.save_gallery(tmp_path / "g.npz")
    other = forensix.load("id-face")
    with pytest.raises(ValueError, match="id-motion"):
        other.load_gallery(tmp_path / "g.npz")  # embeddings of another model are refused


def test_id_motion_needs_trained_weights():
    with pytest.raises(ValueError, match="train_id_motion"):
        forensix.load("id-motion")


def test_voice_segments():
    from forensix.identity.voice import SAMPLE_RATE, segments

    short = np.ones(4 * SAMPLE_RATE, np.float32)
    assert len(segments(short)[0]) == 1  # embedded whole
    long = np.r_[np.ones(6 * SAMPLE_RATE), np.zeros(4 * SAMPLE_RATE)].astype(np.float32)
    segs, starts = segments(long)
    assert starts == [0.0, 1.5, 3.0, 4.5] and all(len(s) == 3 * SAMPLE_RATE for s in segs)  # silence skipped


@pytest.mark.skipif(not os.environ.get("FORENSIX_TEST_HEAVY"), reason="downloads weights; set FORENSIX_TEST_HEAVY=1")
def test_id_voice_verifies_rejects_and_localizes(tmp_path):
    from forensix.audio.io import load_audio, resample

    det = forensix.load("id-voice")
    speech = load_audio(DATA / "speech_cc0.flac", 16000)  # one reader, 10 s
    other = resample(speech, 16000, 12000)  # played 1.33x faster: another pitch and vocal tract
    det.enroll("reader", speech[: 5 * 16000])
    same = det.predict(speech[5 * 16000 :], identity="reader")
    shifted = det.predict(other[: 5 * 16000], identity="reader")
    assert same.is_fake is False and shifted.is_fake is True
    assert same.details["similarity"] > 0.5 > 0.3 > shifted.details["similarity"]
    spliced = det.predict(np.r_[speech[5 * 16000 :], other[5 * 16000 :]], identity="reader")
    sims = spliced.details["segment_similarity"]
    assert spliced.details["segment_start_s"] == [0.0, 1.5, 3.0, 4.5] and sims[0] > 0.5 > 0.25 > sims[-1]
    det.enroll("faster", other[: 5 * 16000])
    assert det.identify(speech[5 * 16000 :])[0][0] == "reader"
    det.save_gallery(tmp_path / "voices.npz")
    det.gallery = Gallery("x")
    det.load_gallery(tmp_path / "voices.npz")
    assert det.gallery.names() == ["faster", "reader"]
