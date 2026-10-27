import io
import json
import time

import numpy as np
import pytest
import soundfile as sf
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient  # noqa: E402

import forensix  # noqa: E402
from forensix.audio.detector import AudioDetector  # noqa: E402
from forensix.hub import WeightFile, fetch  # noqa: E402
from forensix.image.detector import ImageDetector  # noqa: E402
from forensix.registry import _REGISTRY, ModelSpec, register  # noqa: E402
from forensix.server import ServerConfig, create_app  # noqa: E402
from forensix.server.app import to_jsonable  # noqa: E402
from forensix.server.manager import Batcher, Entry  # noqa: E402


class Brightness(ImageDetector):
    """fake_prob = mean pixel value of the 8x8 centre crop."""

    def _fake_probs(self, batch):
        return batch.mean(dim=(1, 2, 3))


class Loudness(AudioDetector):
    """fake_prob = mean absolute amplitude of each window."""

    def _fake_probs(self, windows):
        return windows.abs().mean(dim=1)


@pytest.fixture(autouse=True)
def toy_models():
    t = transforms.Compose([transforms.CenterCrop(8), transforms.ToTensor()])
    specs = [
        ModelSpec("toy-image", "image", lambda w, device="cpu": Brightness("toy-image", nn.Identity(), t, device),
                  None, "MIT", "", "", "toy"),
        ModelSpec("toy-audio", "audio", lambda w, device="cpu": Loudness("toy-audio", nn.Identity(), 16000, device),
                  None, "MIT", "", "", "toy"),
    ]
    for s in specs:
        register(s)
    yield
    for s in specs:
        _REGISTRY.pop(s.name)


def png(value: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def wav(amplitude: float) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.full(16000, amplitude, dtype=np.float32), 16000, format="WAV")
    return buf.getvalue()


def client(**settings) -> TestClient:
    settings.setdefault("default_models", {"image": ["toy-image"], "audio": ["toy-audio"]})
    return TestClient(create_app(ServerConfig(**settings)))


def detect(c, data: bytes, name="x.png", headers=None, **form):
    return c.post("/v1/detect", files={"file": (name, data)}, data=form, headers=headers or {})


def test_detect_image_and_audio_with_defaults():
    with client(preload=["toy-image"]) as c:
        assert c.get("/ready").json()["loaded"] == ["toy-image"]
        r = detect(c, png(255)).json()
        assert r["media"]["kind"] == "image"
        res = r["results"]["toy-image"]
        assert res["status"] == "ok" and res["score"] == pytest.approx(1.0) and res["label"] == "fake"
        r = detect(c, wav(0.25), name="clip.bin", segments="true").json()   # extension doesn't matter
        res = r["results"]["toy-audio"]
        assert r["media"]["kind"] == "audio" and res["score"] == pytest.approx(0.25, abs=1e-4)
        assert res["segments"] == [pytest.approx(0.25, abs=1e-4)]


def test_threshold_override_and_result_cache():
    with client(thresholds={"toy-image": 0.9}) as c:
        first = detect(c, png(204)).json()["results"]["toy-image"]   # 204/255 = 0.8
        assert first["threshold"] == 0.9 and first["label"] == "real" and not first["cached"]
        assert detect(c, png(204)).json()["results"]["toy-image"]["cached"]


def test_model_selection_and_errors():
    with client() as c:
        r = detect(c, wav(0.1), models="toy-image,toy-audio").json()["results"]
        assert r["toy-image"]["status"] == "not_applicable" and r["toy-audio"]["status"] == "ok"
        assert detect(c, png(0), models="nope").status_code == 404
        assert detect(c, b"not media at all" * 10).status_code == 415
        assert detect(c, b"").status_code == 400
        assert detect(c, wav(0.1), models="avcons").json()["results"]["avcons"]["status"] == "not_applicable"
    with client(max_upload_mb=0.001) as c:
        assert detect(c, wav(0.1)).status_code == 413     # 32 KB of audio over a 1 KB limit


def test_untrained_model_is_unavailable():
    with client() as c:
        mp4 = detect(c, png(0), models="avcons").json()   # an image: not applicable before availability
        assert mp4["results"]["avcons"]["status"] == "not_applicable"
        assert c.get("/v1/models/avcons").json()["available"] is False
        assert c.post("/v1/models/avcons/load").status_code == 409


def test_api_keys():
    with client(api_key="k", admin_key="admin") as c:
        assert c.get("/health").status_code == 200
        assert detect(c, png(0)).status_code == 401
        assert detect(c, png(0), headers={"X-API-Key": "k"}).status_code == 200
        assert detect(c, png(0), headers={"Authorization": "Bearer k"}).status_code == 200
        assert c.post("/v1/models/toy-image/load", headers={"X-API-Key": "k"}).status_code == 401
        assert c.post("/v1/models/toy-image/load", headers={"X-API-Key": "admin"}).status_code == 200


def test_batch_requests_share_forward_passes():
    with client(batch_wait_ms=300) as c:
        files = [("files", (f"{i}.png", png(i * 10))) for i in range(6)]
        r = c.post("/v1/detect/batch", files=files, data={"models": "toy-image"}).json()["results"]
        assert [x["results"]["toy-image"]["score"] for x in r] == pytest.approx([i * 10 / 255 for i in range(6)])
        stats = c.get("/v1/stats").json()["models"]["toy-image"]
        assert stats["calls"] == 6 and stats["mean_batch_size"] > 1


def test_lru_eviction_and_unload():
    with client(max_loaded=1) as c:
        c.post("/v1/models/toy-image/load")
        c.post("/v1/models/toy-audio/load")
        assert c.get("/ready").json()["loaded"] == ["toy-audio"]
        assert c.delete("/v1/models/toy-audio").json()["unloaded"] is True
        assert c.get("/ready").json()["loaded"] == []


def test_batcher_scores_one_by_one_when_a_file_fails(tmp_path):
    t = transforms.Compose([transforms.CenterCrop(8), transforms.ToTensor()])
    entry = Entry("toy", "image", Brightness("toy", nn.Identity(), t))
    batcher = Batcher(entry, max_batch=4, wait_ms=200)
    good = tmp_path / "good.png"
    good.write_bytes(png(255))
    futures = [batcher.submit(good), batcher.submit(tmp_path / "missing.png"), batcher.submit(good)]
    assert futures[0].result(5).fake_prob == pytest.approx(1.0)
    with pytest.raises(FileNotFoundError):
        futures[1].result(5)
    assert futures[2].result(5).fake_prob == pytest.approx(1.0)
    batcher.stop()


def test_to_jsonable():
    out = to_jsonable({1: np.float32(0.5), "a": (np.arange(2), float("nan")), "t": torch.device("cpu")})
    assert json.dumps(out) == '{"1": 0.5, "a": [[0, 1], null], "t": "cpu"}'


def test_config_layers(tmp_path, monkeypatch):
    path = tmp_path / "server.json"
    path.write_text(json.dumps({"device": "cuda", "max_loaded": 3, "thresholds": {"gend": 0.7}}))
    monkeypatch.setenv("FORENSIX_SERVER_PRELOAD", "gend, commfor")
    monkeypatch.setenv("FORENSIX_SERVER_MAX_LOADED", "4")
    monkeypatch.setenv("FORENSIX_SERVER_WARMUP", "false")
    cfg = ServerConfig.load(path, device="cpu")
    assert (cfg.device, cfg.max_loaded, cfg.preload, cfg.warmup) == ("cpu", 4, ["gend", "commfor"], False)
    assert cfg.thresholds == {"gend": 0.7}
    with pytest.raises(ValueError, match="Unknown"):
        ServerConfig.load(None, nonsense=1)


def test_fetch_stamp_skips_rehash_until_file_changes(tmp_path, monkeypatch):
    import hashlib

    import forensix.hub as hub

    src = tmp_path / "w.bin"
    src.write_bytes(b"weights")
    w = WeightFile(src.as_uri(), hashlib.sha256(b"weights").hexdigest(), "w.bin")
    monkeypatch.setenv("FORENSIX_CACHE", str(tmp_path / "cache"))
    dest = fetch(w)
    assert dest.read_bytes() == b"weights" and dest.with_name("w.bin.verified").exists()

    calls = []
    real = hub._sha256
    monkeypatch.setattr(hub, "_sha256", lambda p: calls.append(p) or real(p))
    fetch(w)
    assert calls == []                      # unchanged file: trusted from its stamp
    time.sleep(0.01)
    dest.write_bytes(b"tampered")           # size and mtime change: rehashed, re-downloaded
    assert fetch(w).read_bytes() == b"weights" and calls
    monkeypatch.setenv("FORENSIX_VERIFY_HASH", "always")
    calls.clear()
    fetch(w)
    assert calls                            # forced full verification


def test_toy_models_are_registered():
    assert "toy-image" in forensix.list_models("image")
