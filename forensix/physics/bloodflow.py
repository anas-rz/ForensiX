"""phys-bloodflow: one heart drives the blood flow under all facial skin.

A real face shows a pulse (rPPG) in every skin region, at the same heart rate,
with signals that move together. Face swaps and synthesized faces often break
this: the pulse is missing, differs between regions, or is incoherent.

Phase-2 priors are provisional ranges from the rPPG literature (marked in
``details``); learned priors trained on real video replace them in phase 3.
"""

import numpy as np

from forensix.physics.face.regions import REGIONS, region_masks
from forensix.physics.face.tracker import FACE_LANDMARKER, FaceTrack, track_face
from forensix.physics.signals.rppg import coherence, heart_rate, pos
from forensix.registry import ModelSpec, register
from forensix.types import Prediction

MIN_SECONDS = 6.0
# provisional priors (phase 2): (value where the violation starts, value where it saturates)
PRIORS = {
    "mean_coherence": (0.5, 0.0),  # correlation between regions' pulses; lower is worse
    "hr_spread_bpm": (5.0, 20.0),  # std of region heart rates; higher is worse
    "median_snr_db": (-3.0, -9.0),  # pulse strength; lower is worse
}


def region_rgb(frames: np.ndarray, track: FaceTrack) -> dict[str, np.ndarray]:
    """Mean RGB per skin region per frame (NaN where the face is lost)."""
    out = {name: np.full((len(frames), 3), np.nan) for name in REGIONS}
    for i, frame in enumerate(frames):
        if not track.valid[i]:
            continue
        for name, mask in region_masks(track.landmarks[i, :, :2], frame.shape[:2]).items():
            if mask.any():
                out[name][i] = frame[mask].mean(axis=0)
    return out


def _violation(value: float, start: float, full: float) -> float:
    """0 inside the expected range, rising linearly to 1 at ``full``."""
    return float(np.clip((value - start) / (full - start), 0.0, 1.0))


class BloodFlowDetector:
    name = "phys-bloodflow"
    threshold = None  # anomaly score

    def analyze(self, frames: np.ndarray, fps: float, track: FaceTrack | None = None) -> dict:
        track = track or track_face(frames, fps)
        if track.valid.mean() < 0.9:
            raise ValueError("Face tracked in under 90% of frames; rPPG needs a continuously visible face")
        if len(frames) / fps < MIN_SECONDS:
            raise ValueError(f"Need at least {MIN_SECONDS:.0f} s of video for a pulse measurement")
        rgb = region_rgb(frames, track)
        pulses, regions = {}, {}
        for name, trace in rgb.items():
            valid = ~np.isnan(trace).any(axis=1)
            trace = trace.copy()
            for c in range(3):  # bridge short tracking gaps
                trace[~valid, c] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), trace[valid, c])
            pulses[name] = pos(trace, fps)
            hr, snr = heart_rate(pulses[name], fps)
            regions[name] = {"heart_rate_bpm": round(hr, 1), "snr_db": round(snr, 2)}
        corr, names = coherence(pulses, fps)
        off_diag = corr[~np.eye(len(names), dtype=bool)]
        measured = {
            "mean_coherence": float(off_diag.mean()),
            "hr_spread_bpm": float(np.std([r["heart_rate_bpm"] for r in regions.values()])),
            "median_snr_db": float(np.median([r["snr_db"] for r in regions.values()])),
        }
        violations = {k: _violation(v, *PRIORS[k]) for k, v in measured.items()}
        return {
            "score": float(np.mean(list(violations.values()))),
            "constraints": {
                k: {"measured": round(measured[k], 3), "expected_range_start": PRIORS[k][0],
                    "violation": round(violations[k], 3), "prior": "provisional (literature)"}
                for k in measured
            },
            "regions": regions,
            "coherence_matrix": {"regions": names, "values": corr.round(3).tolist()},
            "heart_rate_bpm": round(float(np.median([r["heart_rate_bpm"] for r in regions.values()])), 1),
        }

    def predict(self, video) -> Prediction:
        from forensix.physics.io import read_video

        frames, fps = read_video(video)
        result = self.analyze(frames, fps)
        return Prediction(model=self.name, fake_prob=result["score"], threshold=None,
                          score_type="anomaly", details=result)

    def predict_batch(self, videos) -> list[Prediction]:
        return [self.predict(v) for v in videos]


register(
    ModelSpec(
        name="phys-bloodflow",
        family="physics",
        builder=lambda weights_path, device="cpu": BloodFlowDetector(),
        weights=FACE_LANDMARKER,
        license="ForensiX (algorithmic: POS rPPG, Wang et al. 2017). Face landmarks: MediaPipe "
        "FaceLandmarker (Apache-2.0)",
        source="https://github.com/google-ai-edge/mediapipe",
        paper="https://doi.org/10.1109/TBME.2016.2609282",
        description="Blood-flow physics (physdf): POS pulse signals in 5 facial skin regions "
        "(MediaPipe mesh); scores missing, inconsistent or incoherent pulses. Anomaly score "
        "in [0, 1] with per-constraint evidence in Prediction.details. Needs >= 6 s of a "
        "visible face. Priors are provisional until learned priors (phase 3).",
    )
)
