"""person-anomaly (forensix.identity.anomaly) and the browser studio's shared file format."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

import forensix
from forensix.identity.anomaly import ENCODERS, FORMAT, THRESHOLDS, PersonAnomaly, kind_of

STUDIO = Path(forensix.__file__).parent / "server" / "static" / "studio"


def person(rng, center, n, spread=0.05):
    """Unit embeddings scattered around ``center``."""
    e = center + spread * rng.standard_normal((n, len(center)))
    return e / np.linalg.norm(e, axis=1, keepdims=True)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def unit(rng, d=128):
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


def test_enrolled_person_passes_and_others_are_flagged(rng):
    alice, bob = unit(rng), unit(rng)
    m = PersonAnomaly.from_embeddings("alice", "face", person(rng, alice, 40), groups=np.repeat(np.arange(4), 10))
    same = m.score_embeddings(person(rng, alice, 6))
    other = m.score_embeddings(person(rng, bob, 6))
    assert same["similarity"] > THRESHOLDS["face"] > other["similarity"]
    assert same["person_percentile"] > 0.05 and other["person_percentile"] == 0.0
    assert m.n_files == 4 and m.calibration["folds"] == 4          # leave one file out
    q = m.calibration["oof_similarity_quantiles"]
    assert len(q) == 21 and q == sorted(q)


def test_single_file_falls_back_to_blocks(rng):
    m = PersonAnomaly.from_embeddings("a", "voice", person(rng, unit(rng, 192), 9), groups=np.zeros(9))
    assert m.n_files == 1 and m.calibration["folds"] == 3


def test_too_few_samples(rng):
    with pytest.raises(ValueError, match="at least 2"):
        PersonAnomaly.from_embeddings("a", "face", person(rng, unit(rng), 1))


def test_round_trip_and_registry(tmp_path, rng):
    m = PersonAnomaly.from_embeddings("alice", "voice", person(rng, unit(rng, 192), 12))
    path = m.save(tmp_path / "alice.json")
    d = json.loads(path.read_text())
    assert d["format"] == FORMAT and d["encoder"] == ENCODERS["voice"] and d["trained_in"] == "python"
    loaded = forensix.load("person-anomaly", weights_path=path)
    assert isinstance(loaded, PersonAnomaly) and loaded.person == "alice"
    np.testing.assert_allclose(loaded.centroid, m.centroid, atol=1e-6)
    assert loaded.calibration == m.calibration and loaded.threshold == pytest.approx(1 - THRESHOLDS["voice"])
    with pytest.raises(ValueError, match="person file"):
        forensix.load("person-anomaly")


def test_loads_a_browser_made_file(tmp_path, rng):
    """The studio writes the same keys (engine.js PersonModel.toJSON)."""
    c = unit(rng)
    browser = {"format": FORMAT, "kind": "face", "person": "bob", "encoder": ENCODERS["face"], "created": "2026-11-11",
               "trained_in": "browser", "centroid": c.round(7).tolist(), "threshold_similarity": 0.363,
               "calibration": {"oof_similarity_quantiles": np.linspace(0.6, 0.95, 21).round(5).tolist(), "oof_n": 30, "folds": 3},
               "n_enroll": 30, "n_files": 5}
    p = tmp_path / "bob.face.forensix.json"; p.write_text(json.dumps(browser))
    m = PersonAnomaly.load(p)
    assert m.meta["trained_in"] == "browser" and m.n_files == 5
    assert m.person_percentile(0.5) == 0.0 and m.person_percentile(0.99) == 1.0
    assert m.person_percentile(0.775) == pytest.approx(0.5, abs=0.01)
    assert set(m.to_dict()) == set(browser)


def test_rejects_other_formats_and_encoders(tmp_path, rng):
    d = PersonAnomaly.from_embeddings("a", "face", person(rng, unit(rng), 5)).to_dict()
    (tmp_path / "x.json").write_text(json.dumps({**d, "format": "something/1"}))
    with pytest.raises(ValueError, match="not a"):
        PersonAnomaly.load(tmp_path / "x.json")
    (tmp_path / "y.json").write_text(json.dumps({**d, "encoder": "other-net"}))
    with pytest.raises(ValueError, match="encoder"):
        PersonAnomaly.load(tmp_path / "y.json")


def test_predict_uses_the_encoder(rng):
    alice = unit(rng)
    m = PersonAnomaly.from_embeddings("alice", "face", person(rng, alice, 20))

    class Stub:
        def __init__(self, e): self.e = e
        def embed_media(self, media): return self.e, [0.0, 0.5, 1.0], 4

    m._encoder = Stub(person(rng, alice, 3))
    p = m.predict("clip.mp4")
    assert p.model == "person-anomaly" and p.score_type == "anomaly" and not p.is_fake
    assert p.threshold == pytest.approx(1 - 0.363) and p.details["frames_without_face"] == 1
    assert len(p.segment_fake_probs) == 3
    m._encoder = Stub(person(rng, unit(rng), 3))
    assert m.predict("clip.mp4").is_fake
    m._encoder = Stub(np.zeros((0, 128), np.float32))
    with pytest.raises(ValueError, match="No face"):
        m.predict("clip.mp4")


def test_kind_of():
    assert kind_of("a.wav") == "voice" and kind_of("a.MP3") == "voice"
    assert kind_of("a.jpg") == "face" and kind_of("a.mp4") == "face"
    assert kind_of(np.zeros(16000)) == "voice" and kind_of(np.zeros((8, 8, 3))) == "face"


def test_studio_matches_python_constants():
    js = (STUDIO / "engine.js").read_text()
    assert f'FORMAT = "{FORMAT}"' in js
    for k, v in ENCODERS.items():
        assert re.search(rf'{k}: "{re.escape(v)}"', js)
    for k, v in THRESHOLDS.items():
        assert re.search(rf"{k}: {v}\b", js)
    html = (STUDIO / "index.html").read_text()
    assert 'src="studio.js"' in html and 'href="../style.css"' in html


def test_server_serves_studio_and_exports_ecapa_once(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import forensix.identity.anomaly as A
    from forensix.server import ServerConfig, create_app

    monkeypatch.setenv("FORENSIX_CACHE", str(tmp_path))
    calls = []

    def fake_export(path):
        calls.append(path); Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_bytes(b"onnx")
        return Path(path)

    monkeypatch.setattr(A, "export_ecapa_onnx", fake_export)
    c = TestClient(create_app(ServerConfig(device="cpu")))
    assert c.get("/ui/studio/").status_code == 200
    assert "ForensiX Studio" in c.get("/ui/studio/").text
    for _ in range(2):
        r = c.get("/ui/studio/models/ecapa.onnx")
        assert r.status_code == 200 and r.content == b"onnx"
    assert len(calls) == 1
