"""Measure physics constraints on media, and score or evaluate those measurements.

Measuring is the slow part (face tracking, speech analysis). It is done once
per clip, and the raw constraint values are stored. Priors can then be refitted
and re-evaluated from the stored values without re-running the detectors.
Used by ``scripts/calibrate_physics.py``.
"""

import zlib
from pathlib import Path

import numpy as np

from forensix.physics import AUDIO_DETECTORS, DETECTORS, VIDEO_DETECTORS
from forensix.physics.base import constraint_report
from forensix.physics.priors import resolve

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
AUDIO_MAX_SECONDS = 30.0


def _run(detector, *args, **kwargs) -> dict:
    """Constraint values of one detector on one clip. A clip the detector can't
    judge (too short, face lost, ...) is recorded as skipped, not as an error."""
    try:
        result = detector.analyze(*args, **kwargs)
    except ValueError as e:
        return {"skipped": str(e)}
    except Exception as e:  # noqa: BLE001  (keep going over a large dataset)
        return {"error": f"{type(e).__name__}: {e}"}
    return {"score": round(result["score"], 4),
            "measured": {k: v["measured"] for k, v in result["constraints"].items()}}


def measure_audio(x: np.ndarray, sr: int, detectors: dict) -> dict:
    x = x[: int(AUDIO_MAX_SECONDS * sr)]
    return {name: _run(detectors[name], x, sr) for name in AUDIO_DETECTORS if name in detectors}


def measure_file(path: Path, kind: str, detectors: dict, with_audio: bool = False) -> dict:
    """``{detector: {"score", "measured"} | {"skipped"} | {"error"}}`` for one file.
    ``kind`` is "audio" or "video"; with ``with_audio`` the video's sound
    track is measured too (needs ffmpeg)."""
    from forensix.audio.io import load_audio

    out = {}
    if kind == "video":
        from forensix.physics.face.tracker import track_face
        from forensix.physics.io import read_video

        try:
            frames, fps = read_video(path)
            track = track_face(frames, fps)
        except Exception as e:  # noqa: BLE001
            out.update({name: {"error": f"{type(e).__name__}: {e}"} for name in VIDEO_DETECTORS if name in detectors})
        else:
            for name in VIDEO_DETECTORS:
                if name in detectors:
                    out[name] = _run(detectors[name], frames, fps, track=track)
    if kind == "audio" or with_audio:
        try:
            x = load_audio(path, 16000)
        except Exception as e:  # noqa: BLE001  (e.g. a video without a sound track)
            status = "skipped" if kind == "video" else "error"
            out.update({name: {status: f"{type(e).__name__}: {e}"} for name in AUDIO_DETECTORS if name in detectors})
        else:
            out.update(measure_audio(x, 16000, detectors))
    return out


def in_holdout(group: str, fraction: float) -> bool:
    """Deterministic group-level split (speakers or channels never straddle it)."""
    return (zlib.crc32(group.encode()) % 10000) / 10000 < fraction


def measurements_by_detector(rows: list[dict]) -> dict[str, dict[str, list]]:
    """``{detector: {constraint: [value per clip]}}`` over rows where the detector ran."""
    out: dict[str, dict[str, list]] = {}
    for row in rows:
        for name, result in row["detectors"].items():
            for k, v in result.get("measured", {}).items():
                out.setdefault(name, {}).setdefault(k, []).append(v)
    return out


def score_rows(rows: list[dict], detector: str, priors=None) -> tuple[np.ndarray, np.ndarray]:
    """Anomaly scores of ``detector`` recomputed from stored measurements under ``priors`` (a
    reference or priors table or path, or None for the packaged default), and the row indices."""
    from forensix.physics import reference

    is_ref = reference.is_reference(priors)
    resolved = resolve(detector, DETECTORS[detector].provisional_priors, None if is_ref else priors)
    table = reference.load(priors) if priors is None or is_ref else None
    entry = (table or {}).get("detectors", {}).get(detector)
    scores, idx = [], []
    for i, row in enumerate(rows):
        measured = row["detectors"].get(detector, {}).get("measured")
        if measured is not None:
            m = {k: measured.get(k) for k in resolved}
            scores.append(reference.score(m, entry, resolved)[0] if entry else constraint_report(m, resolved)[0])
            idx.append(i)
    return np.array(scores), np.array(idx, dtype=int)


def false_alarm_summary(scores: np.ndarray) -> dict:
    """How a set of real clips scores: any violation, a strong one, mean score."""
    if not len(scores):
        return {"n": 0}
    return {"n": int(len(scores)), "any_violation": round(float(np.mean(scores > 0)), 4),
            "score_ge_0.5": round(float(np.mean(scores >= 0.5)), 4),
            "score_ge_0.95": round(float(np.mean(scores >= 0.95)), 4),
            "score_ge_0.99": round(float(np.mean(scores >= 0.99)), 4), "mean_score": round(float(scores.mean()), 4)}


def oriented(values, provisional: tuple[float, float]) -> np.ndarray:
    """Constraint values flipped where lower is worse, so higher = more anomalous."""
    sign = 1.0 if provisional[1] > provisional[0] else -1.0
    return sign * np.array([np.nan if v is None else v for v in values], dtype=np.float64)
