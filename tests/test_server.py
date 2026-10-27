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


def test_id_voice_takes_audio():
    from forensix.server.app import applicable
    from forensix.server.media import MediaInfo

    assert applicable("id-voice", "identity", MediaInfo("audio")) is None
    assert applicable("id-voice", "identity", MediaInfo("video", has_audio=True)) is None
    assert applicable("id-voice", "identity", MediaInfo("image")) is not None
    assert applicable("id-face", "identity", MediaInfo("audio")) is not None


def test_calibration_and_options_reach_the_model(tmp_path):
    from forensix.calibration import fit_calibration

    rng = np.random.default_rng(0)
    cal = fit_calibration("toy-image", rng.uniform(0.6, 1.0, 50), rng.uniform(0.0, 0.4, 50))
    cal.save(tmp_path / "cal.json")
    with client(calibrations={"toy-image": str(tmp_path / "cal.json")}) as c:
        res = detect(c, png(255), details="true").json()["results"]["toy-image"]
        assert res["status"] == "ok" and res["label"] == "fake"
        assert res["details"]["raw_score"] == pytest.approx(1.0) and "p_real" in res["details"]
    with client(options={"toy-image": {"bogus": 1}}) as c:
        assert detect(c, png(0)).json()["results"]["toy-image"]["status"] == "error"


def test_fusion_table_is_served_as_a_model(tmp_path):
    from forensix.fusion import Member, fit_fusion

    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.uniform(0.6, 1, (30, 1)), rng.uniform(0, 0.4, (30, 1))])
    labels = np.array([1] * 30 + [0] * 30)
    fit_fusion(scores, labels, [Member("toy-image")]).save(tmp_path / "f.json")
    with client(fusions={"img-fusion": str(tmp_path / "f.json")}) as c:
        assert [m["name"] for m in c.get("/v1/models?family=fusion").json()["models"]] == ["img-fusion"]
        r = detect(c, png(255), models="img-fusion").json()["results"]["img-fusion"]
        assert r["status"] == "ok" and r["family"] == "fusion" and r["label"] == "fake"
        r = detect(c, wav(0.1), models="img-fusion").json()["results"]["img-fusion"]
        assert r["status"] == "not_applicable"      # its member is an image model


# -- calibration from labelled datasets ------------------------------------------------

def make_dataset(root, n=30, layout="manifest", offset=0, fake_offset=0):
    """Toy-image data: reals dark, fakes bright, some overlap; pairs share a group.
    ``offset`` makes different files; ``fake_offset`` shifts the fakes (drift)."""
    root.mkdir(parents=True, exist_ok=True)
    rows = ["file,label,group"]
    for i in range(n):
        for label, value in (("real", 10 + 4 * i + offset), ("fake", 120 + 4 * i + offset + fake_offset)):
            rel = f"{label}/{i}.png" if layout == "folders" else f"media/{label}_{i}.png"
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(png(value))
            rows.append(f"{rel},{label},pair{i}")
    if layout == "manifest":
        (root / "manifest.csv").write_text("\n".join(rows) + "\n")
    return root


def zip_folder(folder, dest):
    import shutil

    return shutil.make_archive(str(dest), "zip", root_dir=folder.parent, base_dir=folder.name)


def test_read_dataset_layouts_and_problems(tmp_path):
    from forensix.dataset import read_dataset

    d = make_dataset(tmp_path / "m", n=3)
    with open(d / "manifest.csv", "a") as f:
        f.write("media/missing.png,real,\nmedia/real_0.png,maybe,\n../outside.png,fake,\n")
    data = read_dataset(d)
    assert (data.n_real, data.n_fake, data.layout) == (3, 3, "manifest")
    assert {i.group for i in data.items} == {"pair0", "pair1", "pair2"}
    assert len(data.problems) == 3 and any("outside" in p for p in data.problems)
    folders = read_dataset(make_dataset(tmp_path / "f", n=2, layout="folders"))
    assert (folders.n_real, folders.n_fake, folders.layout) == (2, 2, "folders")
    with pytest.raises(ValueError):
        read_dataset(tmp_path / "f" / "real")      # no manifest, no real/ and fake/


def test_extract_dataset_rejects_unsafe_archives(tmp_path):
    import zipfile

    from forensix.dataset import extract_dataset

    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_dataset(bad, tmp_path / "out")
    root = extract_dataset(zip_folder(make_dataset(tmp_path / "nested", n=2), tmp_path / "ok"), tmp_path / "out2")
    assert (root / "manifest.csv").exists()        # found one level down


def test_library_calibrate_from_dataset(tmp_path):
    archive = zip_folder(make_dataset(tmp_path / "data"), tmp_path / "data")
    cal, report = forensix.calibrate("toy-image", dataset=archive)
    assert (report["n_real"], report["n_fake"]) == (30, 30) and report["auc"] > 0.99
    assert report["cv"]["conformal"]["false_alarm_rate"] <= 0.1 and report["skipped"] == []
    assert cal.prob(0.9) > 0.9 > 0.1 > cal.prob(0.05)


def wait_for(c, job_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = c.get(f"/v1/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.1)
    raise TimeoutError(job)


def test_calibrate_from_uploaded_dataset_and_predict_with_confidence(tmp_path):
    archive = zip_folder(make_dataset(tmp_path / "data"), tmp_path / "data")
    with client(calibration_dir=str(tmp_path / "cal")) as c:
        before = detect(c, png(200)).json()["results"]["toy-image"]
        assert before["calibration"] is None and "confidence" not in before

        with open(archive, "rb") as f:
            info = c.post("/v1/datasets", files={"file": ("data.zip", f)}, data={"name": "toy"}).json()
        assert (info["n_real"], info["n_fake"], info["n_groups"]) == (30, 30, 30)

        job = c.post("/v1/calibrations", json={"dataset": info["id"], "models": ["toy-image"]}).json()
        job = wait_for(c, job["id"])
        res = job["results"]["toy-image"]
        assert job["status"] == "done" and res["version"] == 1 and res["activated"] and res["auc"] > 0.99

        fake = detect(c, png(250)).json()["results"]["toy-image"]
        real = detect(c, png(20)).json()["results"]["toy-image"]
        assert fake["calibration"] == real["calibration"] == "v1"
        assert fake["label"] == "fake" and fake["confidence"]["verdict"] == "fake"
        assert real["label"] == "real" and real["confidence"]["verdict"] == "real"
        assert fake["confidence"]["fake_probability"] > 0.9 > 0.1 > real["confidence"]["fake_probability"]

        status = c.get("/v1/calibrations/toy-image").json()
        assert status["active_version"] == 1 and status["versions"][0]["n_real"] == 30
        table = c.get("/v1/calibrations/toy-image/versions/1").json()["table"]
        assert table["model"] == "toy-image" and table["format"] == "forensix-calibration/1"

        check = wait_for(c, c.post("/v1/calibrations/check", json={"dataset": info["id"]}).json()["id"])
        assert check["results"]["toy-image"]["recalibrate"] is False    # same data: no drift

        assert c.put("/v1/calibrations/toy-image/active", json={"version": None}).json()["in_use"] is None
        assert "confidence" not in detect(c, png(250)).json()["results"]["toy-image"]
        assert c.put("/v1/calibrations/toy-image/active", json={"version": 7}).status_code == 404

    with client(calibration_dir=str(tmp_path / "cal")) as c:        # versions survive a restart
        assert c.get("/v1/calibrations/toy-image").json()["versions"][0]["version"] == 1


def test_dataset_upload_errors(tmp_path):
    with client() as c:
        r = c.post("/v1/datasets", files={"file": ("x.zip", b"not a zip")})
        assert r.status_code == 422
        empty = tmp_path / "empty"
        (empty / "real").mkdir(parents=True)
        (empty / "fake").mkdir()
        (empty / "real" / "notes.txt").write_text("hi")
        with open(zip_folder(empty, tmp_path / "empty"), "rb") as f:
            r = c.post("/v1/datasets", files={"file": ("e.zip", f)})
        assert r.status_code == 422 and "no usable items" in r.json()["detail"]
        assert c.post("/v1/calibrations", json={"dataset": "nope", "models": ["toy-image"]}).status_code == 404


def test_update_calibration_builds_on_previous_scores():
    from forensix.calibration import fit_calibration, update_calibration

    rng = np.random.default_rng(0)
    base = fit_calibration("m", rng.uniform(0.5, 1, 40), rng.uniform(0, 0.5, 40))
    new, report = update_calibration(base, rng.uniform(0.5, 1, 10), rng.uniform(0, 0.5, 10), window=30)
    assert (report["n_new_fake"], report["n_carried_fake"]) == (10, 20)
    assert len(new.fake_scores) == len(new.real_scores) == 30 and new.alpha == base.alpha
    full, report = update_calibration(base, [0.9], [0.1])
    assert len(full.fake_scores) == 41 and report["n_carried_real"] == 40


def upload(c, folder, tmp_path, name):
    with open(zip_folder(folder, tmp_path / name), "rb") as f:
        return c.post("/v1/datasets", files={"file": (f"{name}.zip", f)}, data={"name": name}).json()["id"]


def calibrate(c, **body):
    r = c.post("/v1/calibrations", json={"models": ["toy-image"], **body})
    assert r.status_code == 202, r.text
    job = wait_for(c, r.json()["id"])
    assert job["status"] == "done", job
    return job["results"]["toy-image"]


def test_full_incremental_and_refit_calibration(tmp_path):
    with client(calibration_dir=str(tmp_path / "cal")) as c:
        a = upload(c, make_dataset(tmp_path / "a"), tmp_path, "a")
        b = upload(c, make_dataset(tmp_path / "b", offset=1), tmp_path, "b")
        drifted = upload(c, make_dataset(tmp_path / "d", offset=2, fake_offset=-90), tmp_path, "d")

        v1 = calibrate(c, mode="full", dataset=a)
        assert (v1["version"], v1["mode"], v1["parent"], v1["n_new"], v1["n_carried"]) == (1, "full", None, 60, 0)

        v2 = calibrate(c, mode="incremental", datasets=[b])           # builds on the active v1
        assert (v2["version"], v2["parent"], v2["n_new"], v2["n_carried"]) == (2, 1, 60, 60)
        assert [x["name"] for x in v2["lineage"]] == ["a", "b"]

        v3 = calibrate(c, mode="incremental", datasets=[a])           # a's files again: replaced, not doubled
        assert (v3["n_new"], v3["n_carried"], v3["n_real"] + v3["n_fake"]) == (60, 60, 120)

        v4 = calibrate(c, mode="incremental", datasets=[b], base_version=1, window=40, activate=False)
        assert (v4["parent"], v4["n_new"], v4["n_carried"], v4["n_dropped_old"]) == (1, 60, 20, 40)
        assert c.get("/v1/calibrations/toy-image").json()["active_version"] == 3

        calls = c.get("/v1/stats").json()["models"]["toy-image"]["calls"]
        v5 = calibrate(c, mode="refit", alpha=0.2)                     # base: active v3; nothing scored
        assert (v5["parent"], v5["n_new"], v5["n_carried"], v5["alpha"]) == (3, 0, 120, 0.2)
        v6 = calibrate(c, mode="full", datasets=[a, b])                # cached scores: nothing scored either
        assert v6["n_new"] == 120
        assert c.get("/v1/stats").json()["models"]["toy-image"]["calls"] == calls

        same = calibrate(c, mode="incremental", datasets=[b], only_if_drift=True)
        assert same["status"] == "skipped" and same["drift"]["recalibrate"] is False
        moved = calibrate(c, mode="incremental", datasets=[drifted], only_if_drift=True)
        assert moved["status"] == "done" and moved["version"] == 7

        listed = c.get("/v1/calibrations/toy-image").json()["versions"]
        assert [v["mode"] for v in listed] == ["full", "incremental", "incremental", "incremental", "refit",
                                                "full", "incremental"]
        assert listed[6]["parent"] == 6 and listed[6]["datasets"] == ["a", "b", "d"]


def test_incremental_refuses_scores_from_another_model_configuration(tmp_path):
    with client(calibration_dir=str(tmp_path / "cal")) as c:
        a = upload(c, make_dataset(tmp_path / "a"), tmp_path, "a")
        calibrate(c, mode="full", dataset=a)
        report = tmp_path / "cal" / "models" / "toy-image" / "v1.report.json"
        report.write_text(json.dumps({**json.loads(report.read_text()), "fingerprint": "something-else"}))
        job = wait_for(c, c.post("/v1/calibrations", json={"models": ["toy-image"], "mode": "refit"}).json()["id"])
        assert job["results"]["toy-image"]["status"] == "failed"
        assert "different model configuration" in job["results"]["toy-image"]["error"]


def test_calibration_request_validation(tmp_path):
    with client(calibration_dir=str(tmp_path / "cal")) as c:
        a = upload(c, make_dataset(tmp_path / "a", n=3), tmp_path, "a")
        post = lambda **b: c.post("/v1/calibrations", json={"models": ["toy-image"], **b}).status_code  # noqa: E731
        assert post(mode="full") == 422                               # no dataset
        assert post(mode="full", dataset=a, window=10) == 422          # window is for incremental
        assert post(mode="incremental", dataset=a) == 409              # nothing to build on yet
        assert post(mode="refit", dataset=a) == 422                    # refit takes no datasets
        assert post(mode="sideways", dataset=a) == 422


def test_ui_server_info_and_benchmarks(tmp_path):
    bench = tmp_path / "results"
    bench.mkdir()
    (bench / "b.json").write_text(json.dumps({"dataset": "Toy set", "models": {
        "toy-image": {"auc": 0.9, "eer": 0.1, "balanced_acc": 0.8, "n_real": 5, "n_fake": 5},
        "other": {"auc": 0.6}, "broken": "not a dict"}}))
    (bench / "not_results.json").write_text(json.dumps({"something": 1}))
    with client(benchmarks_dir=str(bench), api_key="k") as c:
        page = c.get("/ui/")
        assert page.status_code == 200 and "ForensiX Manager" in page.text
        assert c.get("/ui/app.js").status_code == 200 and c.get("/ui/style.css").status_code == 200
        assert c.get("/", follow_redirects=False).headers["location"] == "/ui/"
        assert c.get("/v1/server").status_code == 401                     # the page is public, the data isn't
        info = c.get("/v1/server", headers={"X-API-Key": "k"}).json()
        assert info["auth"] and info["ready"] and "api_key" not in json.dumps(info).replace('"auth"', "")
        rows = c.get("/v1/benchmarks?model=toy-image", headers={"X-API-Key": "k"}).json()["benchmarks"]
        assert rows == [{"source": "b.json", "dataset": "Toy set", "model": "toy-image",
                         "metrics": {"auc": 0.9, "eer": 0.1, "balanced_acc": 0.8, "n_real": 5, "n_fake": 5}}]
    with client(ui=False, benchmarks_dir="") as c:
        assert c.get("/ui/").status_code == 404
        assert c.get("/v1/benchmarks").json() == {"benchmarks": []}


# -- cloud / container runtime -----------------------------------------------------------

def test_metrics_endpoint():
    with client(preload=["toy-image"]) as c:
        detect(c, png(10))
        detect(c, png(10))                                   # cache hit
        text = c.get("/metrics").text
        assert "forensix_ready 1" in text and "# TYPE forensix_requests_total counter" in text
        assert 'forensix_requests_total{model="toy-image",family="image"} 1' in text
        assert 'forensix_cache_hits_total{model="toy-image",family="image"} 1' in text
    with client(metrics=False) as c:
        assert c.get("/metrics").status_code == 404


def test_prediction_only_replica_refuses_shared_writes(tmp_path):
    with client(admin_endpoints=False, calibration_dir=str(tmp_path / "cal")) as c:
        assert detect(c, png(255)).json()["results"]["toy-image"]["status"] == "ok"
        r = c.post("/v1/datasets", files={"file": ("x.zip", b"zip")})
        assert r.status_code == 409 and "predictions only" in r.json()["detail"]
        assert c.post("/v1/calibrations", json={"models": ["toy-image"], "dataset": "x"}).status_code == 409
        assert c.put("/v1/calibrations/toy-image/active", json={"version": None}).status_code == 409
        assert c.post("/v1/identities/alice", files={"files": ("a.png", png(1))}).status_code == 409
        assert c.get("/v1/calibrations").status_code == 200   # reads still work
        assert c.get("/v1/server").json()["admin_endpoints"] is False


def test_replicas_pick_up_calibrations_from_the_shared_dir(tmp_path):
    archive = zip_folder(make_dataset(tmp_path / "data"), tmp_path / "data")
    shared = {"calibration_dir": str(tmp_path / "cal")}
    with client(**shared) as manager, client(admin_endpoints=False, **shared) as replica:
        assert replica.post("/v1/detect", files={"file": ("x.png", png(250))}).json()["results"]["toy-image"]["calibration"] is None
        with open(archive, "rb") as f:
            ds = manager.post("/v1/datasets", files={"file": ("d.zip", f)}).json()["id"]
        calibrate(manager, dataset=ds)
        res = replica.post("/v1/detect", files={"file": ("x.png", png(250))}).json()["results"]["toy-image"]
        assert res["calibration"] == "v1" and res["confidence"]["verdict"] == "fake"


def test_gallery_changes_reach_other_instances(tmp_path, monkeypatch):
    """An instance reloads an identity gallery that another instance changed on disk."""
    from forensix.server.manager import Entry, ModelManager

    class FakeVerifier:
        def __init__(self):
            self.loaded = 0

        def load_gallery(self, path):
            self.loaded += 1

    m = ModelManager(ServerConfig(gallery_dir=str(tmp_path)))
    entry = Entry("id-face", "identity", FakeVerifier())
    (tmp_path / "id-face.npz").write_bytes(b"v1")
    m._sync_gallery(entry)
    assert (entry.detector.loaded, entry.gallery_version) == (1, 1)
    m._sync_gallery(entry)                                   # unchanged: no reload
    assert entry.detector.loaded == 1
    import os
    os.utime(tmp_path / "id-face.npz", (1, 1))               # another instance saved it
    m._sync_gallery(entry)
    assert (entry.detector.loaded, entry.gallery_version) == (2, 2)


def test_cli_warm_only_and_environment(tmp_path, monkeypatch):
    from forensix.server.__main__ import main

    monkeypatch.setenv("FORENSIX_SERVER_CALIBRATION_DIR", str(tmp_path / "cal"))
    with pytest.raises(SystemExit) as ok:
        main(["--warm-only", "--preload", "toy-image", "toy-audio"])
    assert ok.value.code == 0
    with pytest.raises(SystemExit) as bad:
        main(["--warm-only", "--preload", "toy-image", "avcons"])   # avcons needs trained weights
    assert bad.value.code == 1

    seen = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: seen.update(kw, app=app))
    monkeypatch.setenv("PORT", "9123")
    monkeypatch.setenv("FORENSIX_SERVER_PRELOAD", "toy-image toy-audio")
    main(["--log-format", "json"])
    assert seen["port"] == 9123 and seen["workers"] == 1 and seen["timeout_graceful_shutdown"] == 30
    assert seen["app"].state.config.preload == ["toy-image", "toy-audio"]


def test_json_log_lines():
    import logging

    from forensix.server.__main__ import JsonFormatter

    record = logging.LogRecord("forensix.server", logging.WARNING, __file__, 1, "loaded %s", ("x",), None)
    line = json.loads(JsonFormatter().format(record))
    assert (line["level"], line["logger"], line["message"]) == ("WARNING", "forensix.server", "loaded x")
    assert line["time"].endswith("Z")


def test_read_only_cache_error_is_actionable(tmp_path, monkeypatch):
    import os

    cache = tmp_path / "cache"
    (cache / "weights").mkdir(parents=True)
    os.chmod(cache / "weights", 0o555)
    monkeypatch.setenv("FORENSIX_CACHE", str(cache))
    try:
        with pytest.raises(RuntimeError, match="isn't writable"):
            fetch(WeightFile("file:///nonexistent", "0" * 64, "missing.bin"))
    finally:
        os.chmod(cache / "weights", 0o755)
