"""Calibration on the server: uploaded labelled datasets, fitted versions per model, and jobs.

Everything lives under ``ServerConfig.calibration_dir`` (a temporary folder, lost on
restart, when it isn't set):

    datasets/<id>/data/...                    the extracted dataset (forensix.dataset format)
    datasets/<id>/info.json                   name, upload time, summary
    datasets/<id>/scores/<model>-<fp>.jsonl   cached raw scores of its items (per model fingerprint)
    models/<model>/v<N>.json                  fitted calibration table (forensix.calibration format)
    models/<model>/v<N>.report.json           fit report: mode, parent, lineage, counts, AUC, CV, warnings
    models/<model>/v<N>.items.jsonl           the exact scored items the version was fitted on
    models/<model>/active                     the active version number (absent: none)
    jobs/<id>.json                            finished jobs

Calibration modes (``mode`` of a calibrate job):

- ``full``: fit from scratch on one or more datasets;
- ``incremental``: build on a base version: its items plus the new datasets' items
  (a file seen before, by SHA-256, takes its new label and score), optionally capped
  per class with ``window`` (all new items kept, old ones subsampled), and optionally
  only when the new data fails the drift check against the base (``only_if_drift``);
- ``refit``: the base version's items again, with new settings (``alpha``, ``folds``),
  without scoring anything.

Items are only combined across versions when the model fingerprint (registered
weights or checkpoint, and load options) matches, so scores of different model
configurations are never mixed.
"""

import hashlib
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

from forensix.calibration import Calibration, calibrate_scores, drift_check, keep_window
from forensix.dataset import Dataset, extract_dataset, read_dataset
from forensix.metrics import binary_summary
from forensix.server.media import applicable, bound_video, probe, sha256_file

log = logging.getLogger("forensix.server")

MODES = ("full", "incremental", "refit")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(path)


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

    def cached_scores(self, dataset_id: str, model: str, fingerprint: str) -> list[dict] | None:
        p = self.root / "datasets" / dataset_id / "scores" / f"{model}-{fingerprint}.jsonl"
        return _read_jsonl(p) if p.exists() else None

    def cache_scores(self, dataset_id: str, model: str, fingerprint: str, rows: list[dict]) -> None:
        d = self.root / "datasets" / dataset_id / "scores"
        d.mkdir(exist_ok=True)
        _write_jsonl(d / f"{model}-{fingerprint}.jsonl", rows)

    # -- fitted versions -----------------------------------------------------------

    def _model_dir(self, model: str) -> Path:
        d = self.root / "models" / model
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_version(self, model: str, cal: Calibration, report: dict, items: list[dict]) -> int:
        with self._lock:
            d = self._model_dir(model)
            n = max(self.versions(model), default=0) + 1
            _write_jsonl(d / f"v{n}.items.jsonl", items)
            (d / f"v{n}.report.json").write_text(json.dumps({**report, "version": n}, indent=1))
            cal.save(d / f"v{n}.json")  # written last: a version exists once its table does
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

    def version_items(self, model: str, n: int) -> list[dict]:
        """The scored items version ``n`` was fitted on; rebuilt from its table (without
        file hashes or groups) for versions saved before item records existed."""
        p = self.root / "models" / model / f"v{n}.items.jsonl"
        if p.exists():
            return _read_jsonl(p)
        cal, _ = self.version(model, n)
        return ([{"sha256": None, "label": "real", "group": f"v{n}:r{i}", "score": s, "dataset": None}
                 for i, s in enumerate(cal.real_scores)]
                + [{"sha256": None, "label": "fake", "group": f"v{n}:f{i}", "score": s, "dataset": None}
                   for i, s in enumerate(cal.fake_scores)])

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
            job["progress"] = {}
            (self.store.root / "jobs" / f"{job['id']}.json").write_text(json.dumps(job, indent=1, default=str))

    def _execute(self, job: dict) -> None:
        p = job["params"]
        with tempfile.TemporaryDirectory(dir=self.config.tmp_dir) as tmp:
            prepared: dict[str, list[dict]] = {}  # per dataset, probed once for all models
            for model in p["models"]:
                try:
                    fingerprint = self.manager.fingerprint(model)
                    new = self._dataset_items(model, fingerprint, p.get("datasets", []), prepared, Path(tmp), job)
                    if job["kind"] == "check":
                        result = self._check(model, new, p)
                    else:
                        result = self._calibrate(model, fingerprint, new, p)
                except Exception as e:  # one model failing doesn't stop the others
                    log.warning("job %s, %s: %s", job["id"], model, e)
                    result = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
                job["results"][model] = result

    # -- scoring -------------------------------------------------------------------

    def _dataset_items(self, model, fingerprint, dataset_ids, prepared, tmp, job) -> dict:
        """Raw scores of every item of the datasets (cached per model fingerprint):
        ``{"rows": scored items, "skipped": [...], "anomaly": bool, "threshold": float|None}``."""
        rows, skipped, anomaly, threshold = [], [], False, None
        for dataset_id in dataset_ids:
            cached = self.store.cached_scores(dataset_id, model, fingerprint)
            if cached is None:
                if dataset_id not in prepared:
                    prepared[dataset_id] = self._prepare(dataset_id, tmp / dataset_id, job)
                cached = self._score(model, dataset_id, prepared[dataset_id], job)
                self.store.cache_scores(dataset_id, model, fingerprint, cached)
            for r in cached:
                if r.get("score") is None:
                    skipped.append({"dataset": dataset_id, "file": r["file"], "reason": r["error"]})
                else:
                    rows.append(r)
                    anomaly = anomaly or r.get("score_type") == "anomaly"
                    threshold = r.get("threshold", threshold)
        return {"rows": rows, "skipped": skipped, "anomaly": anomaly, "threshold": threshold}

    def _prepare(self, dataset_id: str, tmp: Path, job: dict) -> list[dict]:
        """Probe (and bound, for long or large videos) every item of a dataset once."""
        dataset = self.store.dataset(dataset_id)
        tmp.mkdir(parents=True, exist_ok=True)
        prepared = []
        for i, item in enumerate(dataset.items):
            job["progress"] = {"stage": "preparing", "dataset": dataset_id, "done": i, "total": len(dataset.items)}
            entry = {"item": item, "path": item.path, "file": str(item.path.relative_to(dataset.root))}
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

    def _score(self, model: str, dataset_id: str, prepared: list[dict], job: dict) -> list[dict]:
        family = self.manager.family(model)
        self.manager.get(model)  # load now, so a missing checkpoint fails the model, not every item
        job["progress"] = {"stage": "scoring", "model": model, "dataset": dataset_id, "scored": 0,
                           "total": len(prepared)}

        def one(e: dict) -> dict:
            item = e["item"]
            row = {"file": e["file"], "sha256": e.get("sha256"), "label": item.label,
                   "group": f"{dataset_id}:{item.group}", "dataset": dataset_id}
            reason = e.get("error") or applicable(model, family, e["info"])
            if not reason and family == "identity" and not item.identity:
                reason = "no identity in the manifest"
            if not reason:
                try:
                    pred, _, _ = self.manager.predict(model, e["path"], e["sha256"], item.identity, raw=True)
                    row.update(score=float(pred.fake_prob), score_type=pred.score_type, threshold=pred.threshold)
                except Exception as ex:
                    reason = f"{type(ex).__name__}: {ex}"
            if reason:
                row.update(score=None, error=reason)
            job["progress"]["scored"] += 1
            return row

        # concurrent submissions let image and audio models batch; others queue on the model lock
        with ThreadPoolExecutor(max_workers=max(1, self.config.max_batch)) as pool:
            return list(pool.map(one, prepared))

    # -- fitting -------------------------------------------------------------------

    def _base(self, model: str, p: dict, fingerprint: str) -> tuple[int, Calibration, dict]:
        """The version an incremental or refit job builds on: ``base_version``, else the
        active one, else the latest."""
        n = p.get("base_version")
        if n is None:
            active = self.store.active(model)
            n = active[0] if active else max(self.store.versions(model), default=None)
        if n is None or (found := self.store.version(model, n)) is None:
            raise LookupError(f"{model} has no calibration version {'to build on' if n is None else n}; "
                              "run a full calibration first")
        cal, report = found
        if report.get("fingerprint") and report["fingerprint"] != fingerprint:
            raise ValueError(f"{model} v{n} was fitted with a different model configuration (weights or "
                             "options changed); its scores can't be combined with new ones. Run a full calibration")
        return n, cal, report

    def _calibrate(self, model: str, fingerprint: str, new: dict, p: dict) -> dict:
        mode = p.get("mode", "full")
        rows, warnings = _dedupe(new["rows"])
        base_n = base_cal = None
        lineage = [{"dataset": d, "name": (self.store.dataset_info(d) or {}).get("name")} for d in p.get("datasets", [])]
        carried, dropped_old = [], 0
        alpha, score_type = p.get("alpha"), (p.get("score_types") or {}).get(model)
        if mode in ("incremental", "refit"):
            base_n, base_cal, base_report = self._base(model, p, fingerprint)
            alpha = base_cal.alpha if alpha is None else alpha
            score_type = score_type or base_cal.score_type
            old = self.store.version_items(model, base_n)
            if mode == "refit":
                rows, old = old, []
            elif p.get("only_if_drift"):
                drift = _drift(base_cal, rows, p.get("level", 0.01))
                if not drift["recalibrate"]:
                    return {"status": "skipped", "reason": f"no drift from v{base_n}", "base_version": base_n,
                            "drift": drift, "n_skipped": len(new["skipped"]), "skipped": new["skipped"][:100]}
            new_hashes = {r["sha256"] for r in rows if r.get("sha256")}
            old = [r for r in old if not (r.get("sha256") and r["sha256"] in new_hashes)]
            for label in ("real", "fake"):
                old_l = [r for r in old if r["label"] == label]
                keep = keep_window(np.array([r for r in rows if r["label"] == label]), np.array(old_l),
                                   p.get("window"), p.get("seed", 0) + (label == "fake"))
                carried += [old_l[i] for i in keep]
                dropped_old += len(old_l) - len(keep)
            lineage = base_report.get("lineage", []) + lineage
        if alpha is None:
            alpha = 0.05
        items = rows + carried
        if not items:
            raise ValueError("no scored items to fit on")
        scores = np.array([r["score"] for r in items])
        is_fake = np.array([r["label"] == "fake" for r in items])
        info_names = ", ".join(x["name"] or x["dataset"] for x in lineage if x.get("dataset"))
        cal, report = calibrate_scores(
            model, scores, is_fake, [r["group"] for r in items], alpha, p.get("folds", 5), p.get("seed", 0),
            f"{mode}: {info_names}" if info_names else mode,
            score_type or ("raw" if new["anomaly"] else None), new["threshold"])
        n_new = len(rows) if mode != "refit" else 0
        report.update(
            mode=mode, parent=base_n, fingerprint=fingerprint, lineage=lineage, created=_now(),
            datasets=p.get("datasets", []), window=p.get("window"),
            n_new=n_new, n_carried=len(carried) if mode == "incremental" else (len(rows) if mode == "refit" else 0),
            n_dropped_old=dropped_old)
        report["warnings"] = warnings + report["warnings"]
        version = self.store.save_version(model, cal, report, items)
        if p.get("activate", True):
            self.store.set_active(model, version)
        return {"status": "done", "version": version, "activated": p.get("activate", True), **report,
                "n_skipped": len(new["skipped"]), "skipped": new["skipped"][:100]}

    def _check(self, model: str, new: dict, p: dict) -> dict:
        active = self.store.active(model)
        if active is None:
            raise ValueError(f"{model} has no active calibration to check")
        rows, _ = _dedupe(new["rows"])
        return {"status": "done", "version": active[0], **_drift(active[1], rows, p.get("level", 0.01)),
                "n_skipped": len(new["skipped"]), "skipped": new["skipped"][:100]}


def _dedupe(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """One row per file (SHA-256) across datasets; a file labelled both ways is dropped."""
    by_hash, out, warnings = {}, [], []
    for r in rows:
        h = r.get("sha256")
        if h is None:
            out.append(r)
        elif h not in by_hash:
            by_hash[h] = r
        elif by_hash[h] is not None and by_hash[h]["label"] != r["label"]:
            warnings.append(f"{r['file']} is labelled both real and fake across datasets; left out")
            by_hash[h] = None
    out += [r for r in by_hash.values() if r is not None]
    return out, warnings


def _drift(cal: Calibration, rows: list[dict], level: float) -> dict:
    scores = np.array([r["score"] for r in rows], float)
    is_fake = np.array([r["label"] == "fake" for r in rows], bool)
    result = drift_check(cal, scores[is_fake], scores[~is_fake], level)
    if is_fake.any() and (~is_fake).any():
        result["auc"] = binary_summary(scores[is_fake], scores[~is_fake], None)["auc"]
    result["threshold"] = None if math.isinf(result["threshold"]) else result["threshold"]
    return result


def fingerprint_of(parts: dict) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:12]
