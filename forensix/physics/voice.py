"""phys-voice: a voice is produced by vocal folds, a vocal tract and lungs.

Measured with source-filter estimators (forensix.physics.signals.voice):
- F0 continuity: laryngeal muscles can't jump several semitones within 10 ms;
- jitter: real vocal folds are never perfectly periodic (a "too perfect" voice
  is suspicious; the estimator's floor is ~0.2%, so this check is weak);
- harmonics-to-noise ratio: implausibly clean phonation;
- vocal-tract length (formant dispersion) within human anatomy (~10-21 cm,
  children to large adults) and stable over time (one speaker, one tract);
- breathing: the lungs force pauses; long stretches without one are suspicious.

Phase-2 priors are provisional (marked in ``details``).
"""

import numpy as np

from forensix.physics.base import PhysicsAudioDetector, constraint_report
from forensix.physics.signals import voice as V
from forensix.registry import ModelSpec, register

MIN_VOICED_S = 1.5
VTL_RANGE_CM = (10.0, 21.0)
# Provisional. Where noted, anchored just above values measured on real read
# speech (two CC0 LibriVox readers, 10 x 20 s excerpts, 64 kb/s MP3).
PRIORS = {
    # real speech: 0-1.6 jumps/s after octave-error removal. Weak check: YIN on
    # compressed speech can't separate fast real pitch moves from fakes yet
    "f0_jump_rate_per_s": (2.0, 6.0),
    "jitter_percent": (0.25, 0.10),  # lower is worse (too periodic); real speech ~2%
    "hnr_db": (35.0, 45.0),  # real running speech ~7-9 dB
    "vtl_out_of_range_cm": (0.0, 4.0),  # real readers 16.8-18.7 cm
    "vtl_variation": (0.12, 0.30),  # real speech: 0.05-0.095 across ~1 s chunks
    "longest_pause_free_s": (12.0, 25.0),  # real readers: up to 8.2 s
}


def _voiced_runs(voiced: np.ndarray) -> list[tuple[int, int]]:
    runs, i = [], 0
    while i < len(voiced):
        if voiced[i]:
            j = i
            while j + 1 < len(voiced) and voiced[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def f0_jump_rate(f0: np.ndarray, aperiodicity: np.ndarray, max_step_st: float = 4.0,
                 window: int = 7, edge: int = 2, max_aper: float = 0.2) -> float:
    """Pitch jumps larger than ``max_step_st`` semitones between consecutive
    10 ms hops, per voiced second. Pitch-tracker octave errors are removed
    first: frames more than 6 semitones from a 70 ms rolling median are folded
    by whole octaves toward it (isolated errors vanish; sustained shifts stay),
    and only jumps between clearly periodic frames away from voicing onsets and
    offsets are counted."""
    voiced = ~np.isnan(f0)
    voiced_s = voiced.sum() * V.HOP_S
    if voiced_s == 0:
        return float("nan")
    st = 12 * np.log2(np.where(voiced, f0, 1.0))
    jumps = 0
    for a, b in _voiced_runs(voiced):
        run = st[a : b + 1].copy()
        half = window // 2
        med = np.array([np.median(run[max(0, k - half) : k + half + 1]) for k in range(len(run))])
        run -= 12 * np.round((run - med) / 12) * (np.abs(run - med) > 6)
        ok = aperiodicity[a : b + 1] < max_aper
        for k in range(edge, len(run) - edge - 1):
            if ok[k] and ok[k + 1] and abs(run[k + 1] - run[k]) > max_step_st:
                jumps += 1
    return float(jumps / voiced_s)


def vtl_estimates(formants: np.ndarray, min_frames: int = 30) -> tuple[float, float, np.ndarray]:
    """Median VTL (cm), its coefficient of variation across ~1 s chunks, and the
    per-frame values; NaN when too few frames show 4 formants."""
    vtl = V.vocal_tract_length_cm(formants)
    ok = np.isfinite(vtl) & (vtl > 5) & (vtl < 40)
    if ok.sum() < min_frames:
        return float("nan"), float("nan"), vtl
    chunk_medians = [np.median(c[np.isfinite(c) & (c > 5) & (c < 40)]) for c in np.array_split(vtl, max(1, len(vtl) // 100))
                     if (np.isfinite(c) & (c > 5) & (c < 40)).sum() >= 10]
    variation = float(np.std(chunk_medians) / np.mean(chunk_medians)) if len(chunk_medians) >= 3 else float("nan")
    return float(np.median(vtl[ok])), variation, vtl


class VoiceDetector(PhysicsAudioDetector):
    name = "phys-voice"

    def analyze(self, x: np.ndarray, sr: int) -> dict:
        x = np.asarray(x, dtype=np.float64)
        f0, aper = V.yin_f0(x, sr)
        voiced = ~np.isnan(f0)
        if voiced.sum() * V.HOP_S < MIN_VOICED_S:
            raise ValueError(f"Need at least {MIN_VOICED_S} s of voiced speech")
        jitter, shimmer = V.jitter_shimmer(V.pitch_marks(x, sr, f0), sr)
        formants = V.lpc_formants(x, sr, f0)
        vtl, vtl_var, _ = vtl_estimates(formants)
        runs = V.speech_runs_s(x, sr)
        lo, hi = VTL_RANGE_CM
        measured = {
            "f0_jump_rate_per_s": f0_jump_rate(f0, aper),
            "jitter_percent": jitter,
            "hnr_db": V.hnr_db(x, sr, f0),
            "vtl_out_of_range_cm": float(max(0.0, lo - vtl, vtl - hi)) if np.isfinite(vtl) else float("nan"),
            "vtl_variation": vtl_var,
            "longest_pause_free_s": float(max(runs)) if runs else float("nan"),
        }
        score, evidence = constraint_report(measured, PRIORS)
        return {
            "score": score,
            "constraints": evidence,
            "voice": {
                "f0_median_hz": round(float(np.nanmedian(f0)), 1),
                "shimmer_percent": None if np.isnan(shimmer) else round(shimmer, 2),
                "formants_median_hz": [None if np.isnan(v) else round(float(v)) for v in np.nanmedian(formants, axis=0)]
                if np.isfinite(formants).any() else None,
                "vocal_tract_length_cm": None if np.isnan(vtl) else round(vtl, 1),
                "voiced_fraction": round(float(voiced.mean()), 3),
                "speech_runs_s": [round(r, 2) for r in runs],
            },
            "timeline": {"f0_hz": [None if np.isnan(v) else round(float(v), 1) for v in f0]},
        }


register(
    ModelSpec(
        name="phys-voice",
        family="physics",
        builder=lambda weights_path, device="cpu": VoiceDetector(),
        weights=None,
        license="ForensiX (algorithmic: YIN, LPC, Boersma HNR, Fitch formant dispersion)",
        source="",
        paper="https://doi.org/10.1121/1.1458024",
        description="Voice-production physics (physdf): F0 continuity (jump rate), jitter, HNR, vocal-tract "
        "length from formant dispersion (range and stability) and breathing pauses. No model "
        "weights. Anomaly score in [0, 1] with per-constraint evidence. Priors are provisional "
        "until phase 3.",
    )
)
