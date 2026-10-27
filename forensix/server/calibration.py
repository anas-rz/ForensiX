"""Calibration on the server: uploaded labelled datasets, fitted versions per model, and jobs.

Everything lives under ``ServerConfig.calibration_dir`` (a temporary folder, lost on
restart, when it isn't set):

    datasets/<id>/data/...             the extracted dataset (forensix.dataset format)
    datasets/<id>/info.json            name, upload time, summary
    models/<model>/v<N>.json           fitted calibration tables (forensix.calibration format)
    models/<model>/v<N>.report.json    the fit report: counts, AUC, out-of-fold metrics, warnings
    models/<model>/active              the active version number (absent: none)
    jobs/<id>.json                     finished jobs
"""

import json
import logging
import math
import queue
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from forensix.calibration import Calibration, calibrate_scores, drift_check
from forensix.dataset import Dataset, extract_dataset, read_dataset
from forensix.metrics import binary_summary
from forensix.server.media import applicable, bound_video, probe, sha256_file

log = logging.getLogger("forensix.server")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class CalibrationStore:
    def __init__(self, directory: str | Path | None = None):
        self._tmp = None if directory else tempfile.mkdtemp(prefix="forensix-calibration-")
        self.root = Path(directory or self._tmp)
        for sub in ("datasets", "models", "jobs"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # -- datasets ------------------------------------------------------------------

    def add_dataset(self, archive: Path, name: str, max_bytes: int) -> dict:
        """Extract and validate a dataset ZIP; returns its info."""
        dataset_id = uuid.uuid4().hex[:12]
        d = self.root / "datasets" / dataset_id
        try:
            root = extract_dataset(archive, d / "data", max_bytes=max_bytes)
            data = read_dataset(root)
        except Exception:
            shutil.rmtree(d, ignore_errors=True)
            raise
        info = {"id": dataset_id, "name": name, "uploaded": _now(), "root": str(root.relative_to(d)),
                **data.summary()}
        (d / "info.json").write_text(json.dumps(info, indent=1))
        return info

    def dataset_info(self, dataset_id: str) -> dict | None:
        p = self.root / "datasets" / dataset_id / "info.json"
        return json.loads(p.read_text()) if dataset_id.isalnum() and p.exists() else None

    def dataset(self, dataset_id: str) -> Dataset:
        info = self.dataset_info(dataset_id)
        if info is None:
            raise KeyError(f"Unknown dataset {dataset_id!r}")
        return read_dataset(self.root / "datasets" / dataset_id / info["root"])

    def datasets(self) -> list[dict]:
        infos = (self.dataset_info(p.name) for p in (self.root / "datasets").iterdir() if p.is_dir())
        return sorted((i for i in infos if i), key=lambda i: i["uploaded"])

    def delete_dataset(self, dataset_id: str) -> bool:
        if self.dataset_info(dataset_id) is None:
            return False
        shutil.rmtree(self.root / "datasets" / dataset_id)
        return True

    # -- fitted versions -----------------------------------------------------------

    def _model_dir(self, model: str) -> Path:
        d = self.root / "models" / model
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_version(self, model: str, cal: Calibration, report: dict) -> int:
        with self._lock:
            d = self._model_dir(model)
            n = max(self.versions(model), default=0) + 1
            cal.save(d / f"v{n}.json")
            (d / f"v{n}.report.json").write_text(json.dumps(report, indent=1))
            return n

    def versions(self, model: str) -> list[int]:
        d = self.root / "models" / model
        if not d.exists():
            return []
        return sorted(int(p.stem[1:]) for p in d.glob("v*.json") if p.stem[1:].isdigit())

    def version(self, model: str, n: int) -> tuple[Calibration, dict] | None:
        d = self.root / "models" / model
        if n not in self.versions(model):
            return None
        report = d / f"v{n}.report.json"
        return Calibration.load(d / f"v{n}.json"), (json.loads(report.read_text()) if report.exists() else {})

    def active(self, model: str) -> tuple[int, Calibration] | None:
        p = self.root / "models" / model / "active"
        if not p.exists():
            return None
        n = int(p.read_text().strip())
        found = self.version(model, n)
        return (n, found[0]) if found else None

    def set_active(self, model: str, n: int | None) -> None:
        with self._lock:
            p = self._model_dir(model) / "active"
            if n is None:
                p.unlink(missing_ok=True)
            elif n not in self.versions(model):
                raise KeyError(f"{model} has no calibration version {n}")
            else:
                p.write_text(f"{n}\n")

    def models(self) -> list[str]:
        return sorted(p.name for p in (self.root / "models").iterdir() if p.is_dir())


class Jobs:
    """Background calibration and drift-check jobs, run one at a time."""

    def __init__(self, store: CalibrationStore, manager, config):
        self.store, self.manager, self.config = store, manager, config
        self._jobs: dict[str, dict] = {}
        self._queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, name="calibration-jobs", daemon=True).start()

    def submit(self, kind: str, params: dict) -> dict:
        job = {"id": uuid.uuid4().hex[:12], "kind": kind, "status": "queued", "created": _now(),
               "params": params, "progress": {}, "results": {}}
        self._jobs[job["id"]] = job
        self._queue.put(job["id"])
        return job

    def get(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id)
        if job is None and job_id.isalnum() and (p := self.store.root / "jobs" / f"{job_id}.json").exists():
            job = json.loads(p.read_text())
        return job

    def all(self) -> list[dict]:
        return sorted(self._jobs.values(), key=lambda j: j["created"])

    def _run(self) -> None:
        while True:
            job = self._jobs[self._queue.get()]
            job.update(status="running", started=_now())
            try:
                self._execute(job)
                job["status"] = "done"
            except Exception as e:
                log.exception("job %s failed", job["id"])
                job.update(status="failed", error=f"{type(e).__name__}: {e}")
            job["finished"] = _now()
            (self.store.root / "jobs" / f"{job['id']}.json").write_text(json.dumps(job, indent=1))

    def _execute(self, job: dict) -> None:
        p = job["params"]
        info = self.store.dataset_info(p["dataset"])
        dataset = self.store.dataset(p["dataset"])
        with tempfile.TemporaryDirectory(dir=self.config.tmp_dir) as tmp:
            prepared = self._prepare(dataset, Path(tmp), job)
            for model in p["models"]:
                job["progress"] = {"model": model, "scored": 0, "total": len(prepared)}
                try:
                    scores, labels, groups, skipped, anomaly, threshold = self._score(model, prepared, job)
                    if job["kind"] == "calibrate":
                        result = self._fit(model, scores, labels, groups, anomaly, threshold, p, info)
                    else:
                        result = self._check(model, scores, labels, p)
                    result["skipped"] = skipped[:100]
                    result["n_skipped"] = len(skipped)
                except Exception as e:  # one model failing doesn't stop the others
                    log.warning("job %s, %s: %s", job["id"], model, e)
                    result = {"status": "failed", "error": str(e)}
                job["results"][model] = result
        job["progress"] = {}

    def _prepare(self, dataset: Dataset, tmp: Path, job: dict) -> list[dict]:
        """Probe (and bound, for long or large videos) every item once, for all models."""
        prepared = []
        for i, item in enumerate(dataset.items):
            job["progress"] = {"stage": "preparing", "done": i, "total": len(dataset.items)}
            entry = {"item": item, "path": item.path}
            try:
                entry["info"] = probe(item.path)
                if entry["info"].kind == "video":
                    sub = tmp / str(i)
                    sub.mkdir()
                    entry["path"], _ = bound_video(item.path, entry["info"], self.config.max_video_seconds,
                                                   self.config.max_video_height, sub)
                entry["sha256"] = sha256_file(item.path)
            except Exception as e:
                entry["error"] = f"unreadable: {e}"
            prepared.append(entry)
        return prepared

    def _score(self, model: str, prepared: list[dict], job: dict):
        family = self.manager.family(model)
        self.manager.get(model)  # load now, so a missing checkpoint fails the model, not every item
        scores, skipped = np.full(len(prepared), np.nan), []
        anomaly, threshold = False, None

        def one(k: int):
            e = prepared[k]
            if "error" in e:
                return k, None, e["error"]
            if reason := applicable(model, family, e["info"]):
                return k, None, reason
            if family == "identity" and not e["item"].identity:
                return k, None, "no identity in the manifest"
            try:
                pred, _, _ = self.manager.predict(model, e["path"], e["sha256"], e["item"].identity, raw=True)
            except Exception as ex:
                return k, None, f"{type(ex).__name__}: {ex}"
            return k, pred, None

        # concurrent submissions let image and audio models batch; others queue on the model lock
        with ThreadPoolExecutor(max_workers=max(1, self.config.max_batch)) as pool:
            for k, pred, error in pool.map(one, range(len(prepared))):
                if pred is None:
                    skipped.append({"file": str(prepared[k]["item"].path.name), "reason": error})
                else:
                    scores[k] = pred.fake_prob
                    anomaly = anomaly or pred.score_type == "anomaly"
                    threshold = pred.threshold
                job["progress"]["scored"] += 1
        labels = np.array([e["item"].is_fake for e in prepared])
        groups = [e["item"].group for e in prepared]
        return scores, labels, groups, skipped, anomaly, threshold

    def _fit(self, model, scores, labels, groups, anomaly, threshold, p, info) -> dict:
        source = f"dataset:{info['name']} ({info['id']})"
        score_type = (p.get("score_types") or {}).get(model) or ("raw" if anomaly else None)
        cal, report = calibrate_scores(model, scores, labels, groups, p["alpha"], p["folds"], p.get("seed", 0),
                                       source, score_type, threshold)
        report.update(dataset=info["id"], dataset_name=info["name"], created=_now())
        version = self.store.save_version(model, cal, report)
        if p.get("activate", True):
            self.store.set_active(model, version)
        return {"status": "done", "version": version, "activated": p.get("activate", True), **report}

    def _check(self, model, scores, labels, p) -> dict:
        active = self.store.active(model)
        if active is None:
            raise ValueError(f"{model} has no active calibration to check")
        ok = np.isfinite(scores)
        fake, real = scores[ok & labels], scores[ok & ~labels]
        result = {"status": "done", "version": active[0], **drift_check(active[1], fake, real, p.get("level", 0.01))}
        if fake.size and real.size:
            result["auc"] = binary_summary(fake, real, None)["auc"]
        result["threshold"] = None if math.isinf(result["threshold"]) else result["threshold"]
        return result
