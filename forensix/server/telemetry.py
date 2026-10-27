"""In-memory telemetry for the dashboard: recent predictions and resource samples.

Bounded ring buffers, so memory stays flat however long the server runs; history
starts when the process does. Fleet-wide, long-term numbers belong in Prometheus
(/metrics); this is the live view of one instance.
"""

import os
import threading
import time
from collections import deque

import numpy as np

OUTCOMES = ("real", "fake", "uncertain", "error")
# dashboard windows (seconds) and their bucket sizes
BUCKETS = {900: 30, 3600: 60, 6 * 3600: 300, 24 * 3600: 900}


def outcome(result: dict) -> str:
    """real / fake from the calibrated verdict or the label; uncertain when there's no
    decision (uncalibrated anomaly scores, no face, uncertain or out-of-distribution verdicts)."""
    if result["status"] == "error":
        return "error"
    verdict = (result.get("confidence") or {}).get("verdict")
    label = verdict if verdict else result.get("label")
    return label if label in ("real", "fake") else "uncertain"


class Telemetry:
    def __init__(self, max_events: int = 50_000, sample_every: float = 10.0, max_samples: int = 8640):
        self.events: deque = deque(maxlen=max_events)
        self.provenance: deque = deque(maxlen=max_events)  # (t, check, verdict, label): evidence, kept apart
        self.samples: deque = deque(maxlen=max_samples)  # 24 h at 10 s
        self.sample_every = sample_every
        self._lock = threading.Lock()
        self._last_cpu = (time.monotonic(), _cpu_seconds())
        self._stop = threading.Event()
        self.started = time.time()

    # -- recording -------------------------------------------------------------------

    def record(self, model: str, family: str, media: str, result: dict) -> None:
        """One scored (or failed) model run; not_applicable/unavailable results aren't runs."""
        if result["status"] not in ("ok", "no_face", "error"):
            return
        if family == "provenance":
            verdict = (result.get("provenance") or {}).get("verdict") or result["status"]
            with self._lock:
                self.provenance.append((time.time(), model, verdict, result.get("label")))
            return
        score = result.get("score")
        event = (time.time(), model, family, media, outcome(result), bool(result.get("cached")),
                 float(result.get("seconds") or 0.0), score if isinstance(score, (int, float)) else None,
                 result.get("threshold") if isinstance(result.get("threshold"), (int, float)) else None,
                 result.get("calibration"))
        with self._lock:
            self.events.append(event)

    def start_sampler(self, extra=lambda: {}) -> None:
        def loop():
            while not self._stop.wait(self.sample_every):
                self.sample(extra())

        self.sample(extra())
        threading.Thread(target=loop, name="telemetry-sampler", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def sample(self, extra: dict | None = None) -> None:
        now, cpu = time.monotonic(), _cpu_seconds()
        last_t, last_cpu = self._last_cpu
        self._last_cpu = (now, cpu)
        cpu_pct = 100.0 * (cpu - last_cpu) / max(now - last_t, 1e-6) if now > last_t else 0.0
        sample = {"t": time.time(), "rss_mb": _rss_mb(), "cpu_pct": round(cpu_pct, 1),
                  "threads": threading.active_count(), **_gpu(), **(extra or {})}
        with self._lock:
            self.samples.append(sample)

    # -- summaries ---------------------------------------------------------------------

    def summary(self, window: int = 3600, model: str | None = None) -> dict:
        window = min(BUCKETS, key=lambda w: abs(w - window))
        bucket = BUCKETS[window]
        now = time.time()
        start = (now - window) // bucket * bucket + bucket  # aligned buckets, the last one current
        n_buckets = int(window // bucket)
        with self._lock:
            events = [e for e in self.events if e[0] >= start and (model is None or e[1] == model)]
            samples = [s for s in self.samples if s["t"] >= start]
            evidence = [e for e in self.provenance if e[0] >= start and (model is None or e[1] == model)]
        series = [{"t": start + i * bucket, "n": 0, "cached": 0, **{o: 0 for o in OUTCOMES}, "lat": []}
                  for i in range(n_buckets)]
        by_model: dict[str, dict] = {}
        by_media: dict[str, int] = {}
        latencies, scores, thresholds, cached, calibrations = [], [], [], 0, set()
        for t, name, family, media, out, hit, seconds, score, threshold, calibration in events:
            b = series[min(n_buckets - 1, int((t - start) // bucket))]
            b["n"] += 1
            b[out] += 1
            m = by_model.setdefault(name, {"family": family, "n": 0, "cached": 0, "lat": [], **{o: 0 for o in OUTCOMES}})
            m["n"] += 1
            m[out] += 1
            by_media[media] = by_media.get(media, 0) + 1
            if hit:
                b["cached"] += 1
                m["cached"] += 1
                cached += 1
            else:
                b["lat"].append(seconds)
                m["lat"].append(seconds)
                latencies.append(seconds)
            if model is not None and score is not None:
                scores.append(score)
                if threshold is not None:
                    thresholds.append(threshold)
                if calibration:
                    calibrations.add(calibration)
        for b in series:
            b.update(_percentiles(b.pop("lat")))
        for m in by_model.values():
            m.update(_percentiles(m.pop("lat")))
        n = len(events)
        out = {
            "window": window, "bucket": bucket, "now": now, "since": max(start, self.started),
            "history_starts": self.started, "model": model,
            "totals": {"n": n, "cached": cached, **{o: sum(b[o] for b in series) for o in OUTCOMES},
                       **_percentiles(latencies)},
            "series": series,
            "by_model": dict(sorted(by_model.items(), key=lambda kv: -kv[1]["n"])),
            "by_media": dict(sorted(by_media.items(), key=lambda kv: -kv[1])),
            "system": samples,
            "provenance": _provenance_summary(evidence),
        }
        if model is not None:
            out["scores"] = {"values": [round(s, 6) for s in scores[-5000:]],
                             "threshold": float(np.median(thresholds)) if thresholds else None,
                             "calibrated": bool(calibrations)}
        return out


def _provenance_summary(evidence: list) -> dict:
    by_verdict: dict[str, int] = {}
    by_check: dict[str, dict[str, int]] = {}
    for _, check, verdict, _ in evidence:
        by_verdict[verdict] = by_verdict.get(verdict, 0) + 1
        by_check.setdefault(check, {})[verdict] = by_check.setdefault(check, {}).get(verdict, 0) + 1
    return {"n": len(evidence), "fake": sum(1 for e in evidence if e[3] == "fake"),
            "real": sum(1 for e in evidence if e[3] == "real"),
            "by_verdict": dict(sorted(by_verdict.items(), key=lambda kv: -kv[1])), "by_check": by_check}


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"p50": None, "p95": None}
    p50, p95 = np.percentile(values, [50, 95])
    return {"p50": round(float(p50), 4), "p95": round(float(p95), 4)}


def _cpu_seconds() -> float:
    t = os.times()
    return t.user + t.system


def _rss_mb() -> float | None:
    try:  # Linux: resident pages
        with open("/proc/self/statm") as f:
            return round(int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20, 1)
    except (OSError, ValueError, AttributeError):
        try:
            import resource

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux, bytes on macOS
            return round(peak / (2**20 if os.uname().sysname == "Darwin" else 2**10), 1)
        except Exception:
            return None


def _gpu() -> dict:
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            return {"gpu_mb": round(torch.cuda.memory_allocated() / 2**20, 1),
                    "gpu_reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 1)}
    except Exception:
        pass
    return {}
