"""Loaded models: kept in memory, warmed up, shared safely between requests."""

import logging
import queue
import threading
import time
from collections import OrderedDict
from types import SimpleNamespace
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import forensix
from forensix.calibration import Calibration, CalibratedDetector
from forensix.server.calibration import CalibrationStore
from forensix.server.config import ServerConfig
from forensix.types import Prediction

log = logging.getLogger("forensix.server")

# families whose detectors batch on the device; requests to them are batched dynamically
BATCHED_FAMILIES = {"image", "audio"}
# models that only work after training, so they need a checkpoint in ServerConfig.weights
NEEDS_WEIGHTS = {"avcons", "avid", "id-motion", "person-detector"}


@dataclass
class Entry:
    name: str
    family: str
    detector: object
    lock: threading.Lock = field(default_factory=threading.Lock)
    batcher: "Batcher | None" = None
    load_seconds: float = 0.0
    loaded_at: float = field(default_factory=time.time)
    gallery_version: int = 0
    calls: int = 0
    errors: int = 0
    cache_hits: int = 0
    busy_seconds: float = 0.0
    batches: int = 0
    batched_items: int = 0


class Batcher:
    """Collects concurrent requests for one image or audio model into batches.

    A request waits at most ``wait_ms`` for others to join its batch, so a lone
    request is barely delayed while bursts share one forward pass."""

    def __init__(self, entry: Entry, max_batch: int, wait_ms: float):
        self.entry, self.max_batch, self.wait_s = entry, max_batch, wait_ms / 1000
        self.queue: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, name=f"batcher-{entry.name}", daemon=True)
        self.thread.start()

    def submit(self, path: Path) -> Future:
        future: Future = Future()
        self.queue.put((path, future))
        return future

    def stop(self) -> None:
        self.queue.put(None)

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            if item is None:
                return
            batch = [item]
            deadline = time.monotonic() + self.wait_s
            while len(batch) < self.max_batch:
                try:
                    nxt = self.queue.get(timeout=max(0.0, deadline - time.monotonic()))
                except queue.Empty:
                    break
                if nxt is None:
                    self.queue.put(None)  # stop after this batch
                    break
                batch.append(nxt)
            self._score(batch)

    def _score(self, batch) -> None:
        paths = [p for p, _ in batch]
        self.entry.batches += 1
        self.entry.batched_items += len(batch)
        with self.entry.lock:
            try:
                preds = self.entry.detector.predict_batch(paths, batch_size=self.max_batch)
            except Exception:
                preds = None  # one bad file fails the whole batch: score one by one
            if preds is not None:
                for (_, future), pred in zip(batch, preds):
                    future.set_result(pred)
                return
            for path, future in batch:
                try:
                    future.set_result(self.entry.detector.predict(path))
                except Exception as e:
                    future.set_exception(e)


class ModelManager:
    def __init__(self, config: ServerConfig):
        self.config = config
        self._entries: "OrderedDict[str, Entry]" = OrderedDict()
        self._lock = threading.Lock()
        self._loading: dict[str, threading.Lock] = {}
        self._results: "OrderedDict[tuple, Prediction]" = OrderedDict()
        self._results_lock = threading.Lock()
        # detectors stay uncalibrated; calibrations apply to their predictions, so a new
        # table takes effect without reloading the model and calibration data is scored raw
        self.store = CalibrationStore(config.calibration_dir)
        self._static = {name: Calibration.load(path) for name, path in config.calibrations.items()}

    def calibration(self, name: str) -> tuple[str, Calibration] | None:
        """``(version, table)`` in use for ``name``: the active fitted version, else the config's table."""
        if active := self.store.active(name):
            return f"v{active[0]}", active[1]
        if name in self._static:
            return "config", self._static[name]
        return None

    # -- loading -----------------------------------------------------------------

    def loaded(self) -> list[str]:
        with self._lock:
            return list(self._entries)

    def get(self, name: str) -> Entry:
        """The loaded model, loading it first if needed."""
        with self._lock:
            entry = self._entries.get(name)
            if entry is not None:
                self._entries.move_to_end(name)
                return entry
            load_lock = self._loading.setdefault(name, threading.Lock())
        with load_lock:  # one load per model, even with concurrent first requests
            with self._lock:
                if name in self._entries:
                    return self._entries[name]
            entry = self._load(name)
            with self._lock:
                self._entries[name] = entry
                evicted = []
                while len(self._entries) > max(1, self.config.max_loaded):
                    evicted.append(self._entries.popitem(last=False)[1])
            for old in evicted:
                self._release(old)
            return entry

    def family(self, name: str) -> str:
        """The model's family; ``"fusion"`` for the configured fusion tables."""
        return "fusion" if name in self.config.fusions else forensix.get_spec(name).family

    def _load(self, name: str) -> Entry:
        start = time.perf_counter()
        if name in self.config.fusions:
            detector = forensix.load_fusion(self.config.fusions[name], device=self.config.device)
        else:
            weights = self.config.weights.get(name)
            if name in NEEDS_WEIGHTS and not weights:
                raise LookupError(f"{name} has no pretrained weights; train it and set weights[{name!r}] "
                                  "in the server config")
            detector = forensix.load(name, device=self.config.device, weights_path=weights,
                                     **self.config.options.get(name, {}))
        family = self.family(name)
        entry = Entry(name, family, detector)
        if family == "identity":
            gallery = self._gallery_path(name)
            if gallery and gallery.exists():
                detector.load_gallery(gallery)
        if family in BATCHED_FAMILIES and self.config.max_batch > 1:
            entry.batcher = Batcher(entry, self.config.max_batch, self.config.batch_wait_ms)
        if self.config.warmup:
            self._warmup(entry)
        entry.load_seconds = round(time.perf_counter() - start, 2)
        log.info("loaded %s in %.1fs", name, entry.load_seconds)
        return entry

    def _warmup(self, entry: Entry) -> None:
        """One prediction on synthetic input, so one-time costs (CUDA kernels, lazy
        imports, cuDNN autotuning) are paid before the first real request."""
        rng = np.random.default_rng(0)
        try:
            if entry.family == "image":
                entry.detector.predict(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))
            elif entry.family == "audio":
                entry.detector.predict((rng.standard_normal(32000) * 0.1).astype(np.float32), sample_rate=16000)
        except Exception as e:  # warm-up is best effort
            log.warning("warm-up of %s failed: %s", entry.name, e)

    def unload(self, name: str) -> bool:
        with self._lock:
            entry = self._entries.pop(name, None)
        if entry is None:
            return False
        self._release(entry)
        return True

    def _release(self, entry: Entry) -> None:
        # requests already holding the entry finish normally; the model is freed after them
        if entry.batcher:
            entry.batcher.stop()
        log.info("unloaded %s", entry.name)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # -- scoring -----------------------------------------------------------------

    def predict(self, name: str, path: Path, sha256: str, identity: str | None = None,
                raw: bool = False) -> tuple[Prediction, bool, str | None]:
        """``(prediction, from_cache, calibration_version)``; ``raw`` skips calibration.
        Blocking: call it from a worker thread."""
        pred, cached = self._predict_raw(name, path, sha256, identity)
        cal = None if raw else self.calibration(name)
        if cal is None:
            return pred, cached, None
        return CalibratedDetector(SimpleNamespace(name=name), cal[1]).calibrate(pred), cached, cal[0]

    def _predict_raw(self, name: str, path: Path, sha256: str, identity: str | None) -> tuple[Prediction, bool]:
        entry = self.get(name)
        key = (sha256, name, self.config.weights.get(name), identity, entry.gallery_version)
        cached = self._cache_get(key)
        if cached is not None:
            entry.cache_hits += 1
            return cached, True
        start = time.perf_counter()
        try:
            if entry.batcher is not None:
                pred = entry.batcher.submit(path).result()
            elif entry.family == "identity":
                if not identity:
                    raise ValueError(f"{name} needs an identity to check against")
                with entry.lock:
                    pred = entry.detector.predict(path, identity=identity)
            else:
                with entry.lock:
                    pred = entry.detector.predict(path)
        except Exception:
            entry.errors += 1
            raise
        finally:
            entry.calls += 1
            entry.busy_seconds += time.perf_counter() - start
        self._cache_put(key, pred)
        return pred, False

    def _cache_get(self, key) -> Prediction | None:
        if self.config.result_cache_size <= 0:
            return None
        with self._results_lock:
            pred = self._results.get(key)
            if pred is not None:
                self._results.move_to_end(key)
            return pred

    def _cache_put(self, key, pred: Prediction) -> None:
        if self.config.result_cache_size <= 0:
            return
        with self._results_lock:
            self._results[key] = pred
            while len(self._results) > self.config.result_cache_size:
                self._results.popitem(last=False)

    # -- identity galleries ------------------------------------------------------

    def _gallery_path(self, name: str) -> Path | None:
        return Path(self.config.gallery_dir) / f"{name}.npz" if self.config.gallery_dir else None

    def enroll(self, name: str, identity: str, paths: list[Path]) -> dict:
        entry = self.get(name)
        if entry.family != "identity":
            raise ValueError(f"{name} is not an identity model")
        with entry.lock:
            result = entry.detector.enroll(identity, paths)
            entry.gallery_version += 1
            self._save_gallery(entry)
        return result

    def remove_identity(self, name: str, identity: str) -> bool:
        entry = self.get(name)
        with entry.lock:
            if identity not in entry.detector.gallery:
                return False
            entry.detector.gallery.remove(identity)
            entry.gallery_version += 1
            self._save_gallery(entry)
        return True

    def identities(self, name: str) -> list[str]:
        entry = self.get(name)
        with entry.lock:
            return entry.detector.gallery.names()

    def _save_gallery(self, entry: Entry) -> None:
        path = self._gallery_path(entry.name)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            entry.detector.save_gallery(path)

    # -- status ------------------------------------------------------------------

    def stats(self) -> dict:
        with self._lock:
            entries = list(self._entries.values())
        return {e.name: {"family": e.family, "load_seconds": e.load_seconds, "calls": e.calls,
                         "errors": e.errors, "cache_hits": e.cache_hits,
                         "mean_seconds": round(e.busy_seconds / e.calls, 3) if e.calls else None,
                         "mean_batch_size": round(e.batched_items / e.batches, 2) if e.batches else None}
                for e in entries}
