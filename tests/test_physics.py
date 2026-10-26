from pathlib import Path

import numpy as np
import pytest

import forensix
from forensix.physics.base import violation as _violation
from forensix.physics.bloodflow import BloodFlowDetector
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
    assert {"phys-bloodflow", "phys-motion", "phys-lighting", "phys-voice"} <= set(forensix.list_models("physics"))


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


# ---- phys-motion -------------------------------------------------------------

from scipy.spatial.transform import Rotation  # noqa: E402

from forensix.physics.base import constraint_report  # noqa: E402
from forensix.physics.motion import (  # noqa: E402
    RIGID_LANDMARKS,
    MotionDetector,
    angular_kinematics,
    hf_energy_ratio,
    jaw_lip_coupling,
    rigid_deformation,
)


def _poses(angles_deg_z):
    pose = np.tile(np.eye(4), (len(angles_deg_z), 1, 1))
    pose[:, :3, :3] = Rotation.from_euler("z", np.asarray(angles_deg_z)[:, None], degrees=True).as_matrix()
    return pose


def test_angular_kinematics_constant_rotation():
    velocity, acceleration = angular_kinematics(_poses(np.arange(60) * 2.0), FPS)  # 2 deg/frame
    assert np.allclose(np.linalg.norm(velocity, axis=1), 60.0)  # deg/s at 30 fps
    assert np.allclose(acceleration, 0, atol=1e-6)


def test_hf_energy_ratio_smooth_vs_jitter():
    t = np.arange(300) / FPS
    smooth = np.degrees(angular_kinematics(_poses(10 * np.sin(2 * np.pi * 0.5 * t)), FPS)[0])
    jitter = angular_kinematics(_poses(10 * np.sin(2 * np.pi * 0.5 * t) + np.random.default_rng(0).normal(0, 1, 300)), FPS)[0]
    assert hf_energy_ratio(smooth, FPS) < 0.05 < 0.5 < hf_energy_ratio(jitter, FPS)
    assert np.isnan(hf_energy_ratio(smooth, 12.0))  # Nyquist below the cutoff


def _landmark_template():
    rng = np.random.default_rng(0)
    lm = rng.normal(0, 30, (478, 3))
    lm[33], lm[263] = [-40, 0, 0], [40, 0, 0]  # outer eye corners
    return lm


def test_rigid_deformation_invariant_to_similarity_motion():
    base = _landmark_template()
    frames = []
    for i in range(20):
        r = Rotation.from_euler("xyz", [i, -i, 2 * i], degrees=True).as_matrix()
        frames.append((1 + 0.01 * i) * base @ r.T + [i, -2 * i, 3])
    assert rigid_deformation(np.stack(frames)).max() < 1e-9
    warped = np.stack(frames)
    warped[10, RIGID_LANDMARKS[0]] += [0, 20, 0]  # the forehead point moves off the skull
    d = rigid_deformation(warped)
    assert d[10] > 0.02 and np.delete(d, 10).max() < 0.02


def test_jaw_lip_coupling():
    t = np.linspace(0, 4 * np.pi, 120)
    base = _landmark_template()
    base[168], base[152], base[13], base[14] = [0, 20, 0], [0, 120, 0], [0, 80, 0], [0, 82, 0]
    coupled, lips_only = np.repeat(base[None], 120, 0), np.repeat(base[None], 120, 0)
    opening = 15 * (1 + np.sin(t))
    coupled[:, 14, 1] += opening
    coupled[:, 152, 1] += 0.8 * opening  # the jaw carries the chin down
    lips_only[:, 14, 1] += opening  # chin stays put
    lips_only[:, 152, 1] += np.random.default_rng(1).normal(0, 1, 120)
    assert jaw_lip_coupling(coupled) > 0.95
    assert jaw_lip_coupling(lips_only) < 0.3
    assert np.isnan(jaw_lip_coupling(np.repeat(base[None], 120, 0)))  # mouth never moves


def test_constraint_report_uses_worst_applicable_constraint():
    score, ev = constraint_report({"a": 0.2, "b": float("nan")}, {"a": (0.0, 0.4), "b": (0.0, 1.0)})
    assert score == 0.5 and ev["b"]["applicable"] is False and ev["b"]["violation"] is None


@pytest.fixture(scope="module")
def motion_clips():
    pytest.importorskip("mediapipe")
    import cv2
    from PIL import Image

    img = np.array(Image.open(FACE))
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rng = np.random.default_rng(0)

    def frame(angle, dx, stretch=0.0):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        m[0, 2] += dx
        f = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        if stretch:  # stretch the upper face about the eye line: non-rigid
            y0 = 0.42 * h
            weight = np.clip((y0 - yy) / (0.3 * h), 0, 1)
            f = cv2.remap(f, xx, (yy - weight * stretch * (yy - y0)).astype(np.float32), cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)
        return f

    t = np.arange(120) / FPS
    angle = 8 * np.sin(2 * np.pi * 0.3 * t)
    return {
        "smooth": np.stack([frame(a, 10 * np.sin(2 * np.pi * 0.2 * ti)) for a, ti in zip(angle, t)]),
        "jitter": np.stack([frame(a + rng.normal(0, 2.0), 0) for a in angle]),
        "nonrigid": np.stack([frame(a, 0, 0.10 * np.sin(2 * np.pi * 0.5 * ti)) for a, ti in zip(angle, t)]),
    }


def test_motion_on_face_clips(motion_clips):
    det = MotionDetector()
    results = {k: det.analyze(v, FPS) for k, v in motion_clips.items()}
    assert results["smooth"]["score"] < 0.1
    assert results["jitter"]["constraints"]["rotation_hf_energy_ratio"]["violation"] > 0.5
    rigid = {k: r["constraints"]["rigid_deformation"] for k, r in results.items()}
    assert rigid["nonrigid"]["measured"] > 2 * rigid["smooth"]["measured"]  # well above tracker noise
    assert rigid["nonrigid"]["violation"] > 0 and rigid["smooth"]["violation"] == 0
    assert results["jitter"]["score"] > 0.5 and results["nonrigid"]["score"] > 0
    assert len(results["smooth"]["timeline"]["rigid_deformation"]) == len(motion_clips["smooth"])


# ---- phys-lighting -----------------------------------------------------------

from forensix.physics.lighting import (  # noqa: E402
    LightingDetector,
    coupling,
    direction_jitter,
    sh_basis,
)


def test_sh_fit_recovers_known_lighting():
    rng = np.random.default_rng(0)
    normals = rng.normal(size=(2000, 3))
    normals[:, 2] = -np.abs(normals[:, 2])  # facing the camera
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    coeffs = np.array([100, 5, -8, 12, 1, -2, 3, 0.5, -1.5])
    lum = sh_basis(normals) @ coeffs + rng.normal(0, 0.5, 2000)
    fitted, *_ = np.linalg.lstsq(sh_basis(normals), lum, rcond=None)
    assert np.allclose(fitted, coeffs, atol=0.2)


def test_direction_jitter_and_coupling():
    steady = np.tile([0.0, 0.6, -0.8], (30, 1))
    assert direction_jitter(steady) < 1e-6
    jumpy = np.random.default_rng(0).normal(size=(30, 3))
    assert direction_jitter(jumpy / np.linalg.norm(jumpy, axis=1, keepdims=True)) > 45
    assert np.isnan(direction_jitter(np.full((30, 3), np.nan)))  # flat light: direction undefined
    scene = 100 * (1 + 0.2 * np.sin(np.linspace(0, 6, 60)))
    assert coupling(0.5 * scene, scene) > 0.99
    assert np.isnan(coupling(np.full(60, 50.0), np.full(60, 100.0)))  # static scene: not applicable


@pytest.fixture(scope="module")
def lighting_clips():
    pytest.importorskip("mediapipe")
    import cv2
    from PIL import Image
    from scipy.interpolate import griddata

    from forensix.physics.face.fit import fit_frame
    from forensix.physics.face.tracker import track_face
    from forensix.physics.lighting import skin_mask, skin_vertices

    img = np.array(Image.open(FACE)).astype(np.float64)
    h, w = img.shape[:2]
    track = track_face(img.astype(np.uint8)[None], 25.0)
    fit, lms = fit_frame(track, 0), track.landmarks[0, :, :2]
    idx = skin_vertices(fit, lms, (h, w), min_facing=0.0)
    face = skin_mask(lms, (h, w))
    ys, xs = np.nonzero(face)
    nmap = griddata(fit.vertices[idx, :2], fit.normals[idx], (xs, ys), method="linear")
    ok = ~np.isnan(nmap).any(1)
    ys, xs, nmap = ys[ok], xs[ok], nmap[ok] / np.linalg.norm(nmap[ok], axis=1, keepdims=True)
    l0 = np.array([0.3, -0.3, -1.0]) / np.linalg.norm([0.3, -0.3, -1.0])
    shade = lambda light: 0.45 + 0.55 * np.clip(nmap @ light, 0, None)  # Lambertian
    background = ~cv2.dilate(face.astype(np.uint8), np.ones((25, 25))).astype(bool)
    rng = np.random.default_rng(0)

    def clip(kind):
        frames = []
        for ti in np.arange(60) / 25.0:
            f = img.copy()
            gain = 1 + 0.15 * np.sin(2 * np.pi * 0.5 * ti)
            if kind == "consistent":  # the room light dims and brightens
                f *= gain
            elif kind == "jitter":  # face relit from a random direction every frame
                light = l0 + rng.normal(0, 0.6, 3) * [1, 1, 0.3]
                f[ys, xs] *= (shade(light / np.linalg.norm(light)) / shade(l0))[:, None]
            elif kind == "decoupled":  # only the surroundings change
                f[background] *= gain
            frames.append(np.clip(f + rng.normal(0, 1.0, f.shape), 0, 255).astype(np.uint8))
        return np.stack(frames)

    return {k: clip(k) for k in ("consistent", "jitter", "decoupled")}


def test_lighting_on_face_clips(lighting_clips):
    det = LightingDetector()
    r = {k: det.analyze(v, 25.0) for k, v in lighting_clips.items()}
    assert r["consistent"]["score"] < 0.1
    assert r["consistent"]["constraints"]["face_scene_coupling"]["measured"] > 0.9
    assert r["jitter"]["constraints"]["light_direction_jitter_deg"]["violation"] > 0.5
    assert r["decoupled"]["constraints"]["face_scene_coupling"]["violation"] > 0.5
    assert len(r["jitter"]["timeline"]["light_direction"]) == 60


# ---- phys-voice --------------------------------------------------------------

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from voice_synth import SR, tube_formants, utterance  # noqa: E402

from forensix.physics.signals import voice as V  # noqa: E402
from forensix.physics.voice import VoiceDetector, f0_jump_rate  # noqa: E402

SPEECH = Path(__file__).parent / "data" / "speech_cc0.flac"


@pytest.mark.parametrize("f0, length", [(120.0, 17.0), (210.0, 14.0)])
def test_voice_estimators_recover_ground_truth(f0, length):
    x = utterance(f0=f0, length_cm=length).astype(np.float64)
    f0_est, _ = V.yin_f0(x, SR)
    assert abs(np.nanmedian(f0_est) - f0) / f0 < 0.03
    formants = V.lpc_formants(x, SR, f0_est)
    assert np.allclose(np.nanmedian(formants, 0), tube_formants(length)[:4], rtol=0.05)
    assert abs(np.nanmedian(V.vocal_tract_length_cm(formants)) - length) < 1.0
    jitter, shimmer = V.jitter_shimmer(V.pitch_marks(x, SR, f0_est), SR)
    assert 0.3 < jitter < 1.2  # synthesized with 0.8%
    assert V.speech_runs_s(x, SR)[:2] == pytest.approx([2.5, 2.5], abs=0.1)


def test_f0_jump_rate_ignores_isolated_octave_errors():
    f0 = np.full(300, 120.0)
    aper = np.full(300, 0.05)
    errors = f0.copy()
    errors[[50, 51, 150, 220]] *= 2  # 1-2 frame octave errors, as trackers make
    assert f0_jump_rate(errors, aper) == 0.0
    shifted = f0 * np.where((np.arange(300) // 40) % 2, 2.0, 1.0)  # sustained octave shifts
    assert f0_jump_rate(shifted, aper) > 1.0


def test_voice_detector_synthetic_violations():
    det = VoiceDetector()
    assert det.analyze(utterance(), SR)["score"] == 0.0
    cases = {
        "jitter_percent": dict(jitter=0.0, shimmer=0.0, aspiration=0.0),
        "vtl_out_of_range_cm": dict(length_cm=28.0),
        "longest_pause_free_s": dict(seconds=20, pause_s=0),
    }
    for constraint, kw in cases.items():
        r = det.analyze(utterance(**kw), SR)
        assert r["constraints"][constraint]["violation"] > 0.5, constraint


def test_voice_detector_no_false_alarm_on_real_speech():
    pred = forensix.load("phys-voice").predict(SPEECH)
    assert pred.score_type == "anomaly" and pred.fake_prob == 0.0
    assert 15 < pred.details["voice"]["vocal_tract_length_cm"] < 20
    assert 80 < pred.details["voice"]["f0_median_hz"] < 180


# ---- phys-room ---------------------------------------------------------------

from voice_synth import in_room, mic, noise  # noqa: E402

from forensix.physics.room import RoomDetector  # noqa: E402
from forensix.physics.signals import room as Rm  # noqa: E402


def test_shape_change_catches_two_regimes():
    white = Rm.band_spectrum_db(noise(SR, -50), SR)
    brown = Rm.band_spectrum_db(noise(SR, -50, "brown"), SR)
    mixed = [white] * 4 + [brown] * 4
    assert Rm.shape_change_db(mixed) > 9.0  # full white-brown difference (~10 dB RMS)
    assert Rm.shape_variation_db(mixed) < 6.0  # deviation from the median only sees half of it
    assert Rm.shape_change_db([white] * 8) < 1.0


@pytest.fixture(scope="module")
def room_cases():
    dry = utterance(seconds=16, phrase_s=2.0, pause_s=0.6, floor=0.0).astype(np.float64)
    n, half = len(dry), len(dry) // 2
    room = in_room(dry, 0.5)

    def splice(a, b):
        return np.concatenate([a[:half], b[half:]])

    return {
        "one_room": room + noise(n, -60),
        "noise_floor_jump_db": room + splice(noise(n, -70), noise(n, -45)),
        "digital_silence_fraction": dry,
        "channel_variation_db": splice(room, mic(room, 400, 2500)) + noise(n, -60),
        "noise_spectrum_change_db": room + splice(noise(n, -55), noise(n, -55, "brown")),
    }


def test_room_detector_synthetic(room_cases):
    det = RoomDetector()
    r = det.analyze(room_cases["one_room"], SR)
    assert r["score"] == 0.0
    assert abs(r["room"]["rt60_s_median"] - 0.5) < 0.15
    for constraint in ["noise_floor_jump_db", "digital_silence_fraction", "channel_variation_db"]:
        r = det.analyze(room_cases[constraint], SR)
        assert r["constraints"][constraint]["violation"] > 0.5, constraint
    # a change in noise colour at the same level is a weak cue (margin over real speech is small)
    r = det.analyze(room_cases["noise_spectrum_change_db"], SR)
    assert r["constraints"]["noise_spectrum_change_db"]["violation"] > 0.1


def test_room_detector_chunk_at_end_of_file():
    # a 3 s chunk ending exactly at the end of the file has fewer frame levels than samples
    from forensix.audio.io import load_audio

    x = load_audio(SPEECH, SR)
    for seconds in (6.0, 6.00325, 9.0):
        RoomDetector().analyze(x[: int(seconds * SR)], SR)


def test_room_detector_no_false_alarm_on_real_speech():
    pred = forensix.load("phys-room").predict(SPEECH)
    assert pred.score_type == "anomaly" and pred.fake_prob == 0.0
    assert pred.details["constraints"]["digital_silence_fraction"]["measured"] == 0.0


# ---- learned priors (phase 3) -------------------------------------------------

from forensix.physics import DETECTORS  # noqa: E402
from forensix.physics.calibration import (  # noqa: E402
    in_holdout,
    measure_file,
    measurements_by_detector,
    score_rows,
)
from forensix.physics.priors import FORMAT, fit_constraint, fit_priors, provisional_priors, resolve  # noqa: E402


def test_fit_constraint_directions_and_limits():
    rng = np.random.default_rng(0)
    higher = rng.normal(1.0, 0.2, 2000)  # e.g. a jump rate; provisional (2, 6)
    e = fit_constraint(higher, (2.0, 6.0), quantile=0.99)
    assert e["start"] == pytest.approx(np.quantile(higher, 0.99))
    assert e["full"] > e["start"]
    lower = rng.normal(0.9, 0.2, 2000)  # e.g. jitter %; lower is worse, provisional (0.25, 0.1)
    e = fit_constraint(lower, (0.25, 0.1), quantile=0.99)
    assert e["start"] == pytest.approx(np.quantile(lower, 0.01))
    assert e["full"] == 0.1  # capped at the physical limit instead of ramping past it
    assert fit_constraint([1.0, np.nan, None] * 30, (2.0, 6.0), 0.99) is None  # too few finite values


def test_fit_priors_and_resolve():
    rng = np.random.default_rng(1)
    provisional = provisional_priors()
    measurements = {"phys-voice": {"hnr_db": list(rng.normal(20, 3, 500))}}
    table = fit_priors(measurements, provisional, alpha=0.01)
    assert table["format"] == FORMAT and list(table["detectors"]) == ["phys-voice"]
    assert table["detectors"]["phys-voice"]["hnr_db"]["quantile"] == pytest.approx(1 - 0.01 / 6)
    priors = resolve("phys-voice", provisional["phys-voice"], table)
    assert priors["hnr_db"][2] == "learned" and priors["jitter_percent"][2] == "provisional"
    flipped = {"format": FORMAT, "detectors": {"phys-voice": {"hnr_db": {"start": 30, "full": 20}}}}
    with pytest.raises(ValueError, match="wrong way"):
        resolve("phys-voice", provisional["phys-voice"], flipped)


def test_learned_priors_change_scores_and_are_labelled(tmp_path):
    rows = [{"file": "a", "group": "spk1", "label": "real", "detectors": {
        "phys-voice": {"measured": {"hnr_db": 30.0, "jitter_percent": 1.0}}}}]
    table = {"format": FORMAT, "detectors": {"phys-voice": {"hnr_db": {"start": 25.0, "full": 35.0}}}}
    assert score_rows(rows, "phys-voice", {"format": FORMAT})[0][0] == 0.0  # provisional: HNR fine below 35 dB
    assert score_rows(rows, "phys-voice", table)[0][0] == pytest.approx(0.5)
    path = tmp_path / "priors.json"
    path.write_text(__import__("json").dumps(table))
    det = DETECTORS["phys-voice"]()
    det.load_priors(path)
    evidence = det.predict(SPEECH).details["constraints"]
    assert evidence["hnr_db"]["prior"] == "learned" and evidence["jitter_percent"]["prior"] == "provisional"


def test_measure_file_and_split():
    detectors = {name: DETECTORS[name]() for name in ("phys-voice", "phys-room")}
    result = measure_file(SPEECH, "audio", detectors)
    assert set(result) == {"phys-voice", "phys-room"}
    assert result["phys-room"]["score"] == 0.0 and "noise_spectrum_change_db" in result["phys-room"]["measured"]
    assert measurements_by_detector([{"detectors": result}])["phys-voice"]["hnr_db"][0] is not None
    groups = [f"spk{i}" for i in range(2000)]
    held = [g for g in groups if in_holdout(g, 0.2)]
    assert 0.15 < len(held) / len(groups) < 0.25 and held == [g for g in groups if in_holdout(g, 0.2)]


# ---- physdf: graph prior, localization, report --------------------------------

import torch  # noqa: E402

from forensix.physics.graph import train as G  # noqa: E402
from forensix.physics.graph.features import GraphSequence, build_sequence  # noqa: E402
from forensix.physics.graph.model import PhysicsGNN, adjacency  # noqa: E402
from forensix.physics.graph.schema import EDGE_TYPES, EDGES, NODE_NAMES, NODES, RESIDUAL_GROUPS  # noqa: E402
from forensix.physics.physdf import PhysDF  # noqa: E402
from forensix.physics.report import build_report, verify_report  # noqa: E402
from graph_synth import sequence as graph_sequence  # noqa: E402


def test_graph_schema_is_consistent():
    assert all(u in NODES and v in NODES and r in EDGE_TYPES for u, v, r in EDGES)
    a = adjacency()
    assert a.shape == (len(EDGE_TYPES), len(NODE_NAMES), len(NODE_NAMES))
    assert torch.allclose(a.sum(-1)[a.sum(-1) > 0], torch.tensor(1.0))
    connected = (a.sum(0) > 0).any(1)
    assert connected.all()  # every node has a physical partner
    assert all(n in NODES for scored, hidden in RESIDUAL_GROUPS.values() for n in scored + (hidden or []))


def test_audio_graph_sequence_and_roundtrip(tmp_path):
    x = utterance(seconds=4).astype(np.float64)
    seq = build_sequence(audio=x, sr=SR)
    assert len(seq) == 100 and seq.valid["tract"].all() and not seq.valid["jaw"].any()
    voiced = seq.features["glottis"][:, 1]
    assert 0.3 < voiced.mean() < 0.95  # phrases with pauses
    seq.save(tmp_path / "s.npz")
    back = GraphSequence.load(tmp_path / "s.npz")
    assert all(np.array_equal(back.features[n], seq.features[n]) for n in NODE_NAMES)


def test_hidden_nodes_do_not_leak():
    torch.manual_seed(0)
    model = PhysicsGNN(dim=16, layers=2).eval()
    x = {n: torch.randn(1, 50, len(NODES[n][1])) for n in NODE_NAMES}
    visible = torch.ones(1, 50, len(NODE_NAMES), dtype=torch.bool)
    visible[:, :, NODE_NAMES.index("jaw")] = False
    before = model(x, visible)["jaw"][0]
    x2 = dict(x, jaw=torch.randn(1, 50, len(NODES["jaw"][1])) * 10)
    assert torch.equal(model(x2, visible)["jaw"][0], before)


@pytest.fixture(scope="module")
def synthetic_graph_prior():
    torch.manual_seed(0)
    seqs = [graph_sequence(seed=i) for i in range(40)]
    model, norm, history = G.train(seqs, steps=400, batch=16, dim=32, layers=4, lr=3e-3, log_every=400, log=lambda m: None)
    calibration = G.calibrate(model, norm, [graph_sequence(seed=100 + i) for i in range(12)])
    return model, norm, calibration, history


def _z(prior, seq, group):
    model, norm, calibration, _ = prior
    r = G.node_residuals(model, norm, seq)["groups"][group]
    c = calibration["audio_video"][group]
    return (G.clip_summary(G.window_scores(r)) - c["clip_mean"]) / c["clip_std"]


def test_graph_prior_detects_broken_couplings(synthetic_graph_prior):
    real = [_z(synthetic_graph_prior, graph_sequence(seed=200 + i), "graph_av_mouth_from_audio") for i in range(3)]
    lipsync = [_z(synthetic_graph_prior, graph_sequence(seed=210 + i, audio_seed=900 + i), "graph_av_mouth_from_audio")
               for i in range(3)]
    assert min(lipsync) > max(real) and min(lipsync) > 3.0
    real = [_z(synthetic_graph_prior, graph_sequence(seed=220 + i), "graph_bloodflow_residual") for i in range(3)]
    rogue = [_z(synthetic_graph_prior, graph_sequence(seed=230 + i, rogue_pulse="pulse_nose"), "graph_bloodflow_residual")
             for i in range(3)]
    assert min(rogue) > max(real)


def test_physdf_graph_checkpoint_localizes(synthetic_graph_prior, tmp_path):
    model, norm, calibration, _ = synthetic_graph_prior
    G.save(tmp_path / "g.safetensors", model, norm, calibration, {"data": "synthetic"})
    det = PhysDF(tmp_path / "g.safetensors")
    assert det.graph[2] == calibration  # checkpoint round trip, with the real residual distributions
    assert "graph_av_mouth_from_audio" in calibration["audio_video"]
    assert "graph_av_mouth_from_audio" not in calibration["audio_only"] and "graph_voice_residual" in calibration["audio_only"]
    # mismatched audio in the second half only
    seq = graph_sequence(seed=300, seconds=10)
    fake = graph_sequence(seed=301, seconds=10, audio_seed=77)
    half = len(seq) // 2
    for n in ("glottis", "tract", "breath"):
        seq.features[n][half:] = fake.features[n][half:]
    res = G.node_residuals(*det.graph[:2], seq)["groups"]["graph_av_mouth_from_audio"]
    windows = G.window_scores(res)
    assert np.nanmean(windows[len(windows) // 2 + 1 :]) > np.nanmean(windows[: len(windows) // 2 - 1])


def test_physdf_audio_only_report_without_graph():
    det = PhysDF()
    x = forensix.audio.io.load_audio(SPEECH, 16000)
    result = det.analyze(audio=x, sr=16000)
    report = result["report"]
    assert result["score"] == 0.0 and report["verification"]["passed"]
    assert result["status"]["phys-motion"] == "no video" and "no trained graph" in result["status"]["graph"]
    assert report["verdict"] in ("no physical violation found", "inconclusive")
    assert "# physdf forensic report" in report["markdown"] and report["counter_evidence"]


def test_report_claims_verdict_and_verification():
    def entry(v, measured=1.0, prior="provisional"):
        return {"expected_range_start": 0.5, "prior": prior, "measured": measured, "violation": v}

    constraints = {f"phys-voice/c{i}": entry(0.0) for i in range(6)}
    constraints["phys-motion/rigid_deformation"] = entry(0.9, 0.05, "learned")
    constraints["phys-room/rt60_variation"] = {"expected_range_start": 0.35, "prior": "provisional",
                                               "measured": None, "violation": None, "applicable": False}
    result = {"score": 0.9, "constraints": constraints, "status": {}}
    report = build_report(result)
    assert report["verdict"] == "physical violations found" and report["confidence"] == "high"
    assert [c["constraint"] for c in report["claims"]] == ["phys-motion/rigid_deformation"]
    assert "the skull is rigid" in report["markdown"].lower() and report["verification"]["passed"]
    tampered = dict(report, claims=[dict(report["claims"][0], measured=0.001)])
    assert verify_report(tampered, result)
    # too few applicable checks and only weak violations: abstain
    few = {"score": 0.2, "constraints": {"phys-voice/hnr_db": entry(0.2)}, "status": {}}
    assert build_report(few)["verdict"] == "inconclusive"
