"""avcons: audio-visual consistency learned from real videos only (ForensiX).

A temporal transformer over frozen SyncNet window embeddings (lips and audio,
every 5-frame window at 25 fps) and their local sync-distance profiles
(``model.py``). It is trained on real talking-face videos (e.g. VoxCeleb2, CC BY
4.0) against inconsistencies made from those same real clips: offsets, drift,
swapped voices and faces, and spliced audio or video segments
(``negatives.py``). No generator output is used, so it doesn't learn any
generator's fingerprints. It learns what matching speech and lips look like.

Output: the probability that the audio and the face don't belong together
(threshold 0.5), and per-window probabilities with the segments above it.

Weights: none are published; train them with ``scripts/train_avcons.py`` and
load with ``forensix.load("avcons", weights_path=...)``.
"""

from pathlib import Path

import numpy as np

from forensix.audiovisual.syncnet import AUD_FACT, FPS, SAMPLE_RATE
from forensix.audiovisual.syncnet import WEIGHTS as SYNCNET_WEIGHTS
from forensix.hub import fetch
from forensix.registry import ModelSpec, register
from forensix.types import Prediction
from forensix.video.dfdc_wm import WEIGHTS as _WM_WEIGHTS

SPEEDS = (0.95, 1.05)


def speed_change(audio: np.ndarray, speed: float) -> np.ndarray:
    """Audio played ``speed`` times as fast (resampled: tempo and pitch change together)."""
    from fractions import Fraction

    from scipy.signal import resample_poly

    f = Fraction(speed).limit_denominator(100)
    return resample_poly(np.asarray(audio, np.float64), f.denominator, f.numerator)


def track_embeddings(frames: np.ndarray, audio: np.ndarray, syncnet, speeds=()) -> list[dict]:
    """SyncNet embeddings of every face track: ``{"v", "a", "a_speed", "start"}``.
    ``frames`` RGB uint8 at 25 fps; ``audio`` int16-valued at 16 kHz, same start."""
    from forensix.audiovisual.facetrack import crop_track, detect_faces, track_faces

    detections = detect_faces(frames, syncnet.face_detector)
    tracks = track_faces(detections) or track_faces(detections, min_track=30, min_face_size=0)
    out = []
    for track in tracks:
        start = int(track["frame"][0])
        crops = crop_track(frames, track)
        seg = audio[start * AUD_FACT : (start + len(crops)) * AUD_FACT]
        try:
            v, a = syncnet.embed_track(crops, seg)
        except ValueError:
            continue
        a_speed = {s: syncnet.embed_audio(speed_change(seg, s), len(v)).numpy() for s in speeds}
        out.append({"v": v.numpy(), "a": a.numpy(), "a_speed": a_speed, "start": start})
    return out


def sync_offset(v: np.ndarray, a: np.ndarray, vshift: int = 15) -> tuple[int, float]:
    """SyncNet's offset search: the k for which audio window t + k matches video
    window t best, and the confidence (median - min of the mean distance)."""
    padded = np.pad(a, ((vshift, vshift), (0, 0)))
    mdist = np.array([np.linalg.norm(v - padded[k : k + len(v)], axis=1).mean() for k in range(2 * vshift + 1)])
    return int(np.argmin(mdist)) - vshift, float(np.median(mdist) - mdist.min())


def sync_confidence(v: np.ndarray, a: np.ndarray, vshift: int = 15) -> float:
    return sync_offset(v, a, vshift)[1]


MIN_TRAIN_CONFIDENCE = 3.0  # LatentSync's threshold for keeping a training clip
MAX_TRAIN_OFFSET = 6


def align_to_sync(v: np.ndarray, a: np.ndarray, a_speed: dict) -> tuple | None:
    """Re-align a real clip to its own sync (natural AV lag removed), as LatentSync
    prepares its data; None when the sync is unclear (confidence under 3,
    e.g. off-screen speech) or the lag is over 6 frames."""
    k, conf = sync_offset(v, a)
    if conf < MIN_TRAIN_CONFIDENCE or abs(k) > MAX_TRAIN_OFFSET:
        return None
    lo, hi = max(0, -k), len(v) - max(0, k)
    return v[lo:hi], a[lo + k : hi + k], {s: x[lo + k : hi + k] for s, x in a_speed.items()}, k, conf


class AVConsistencyDetector:
    def __init__(self, model, syncnet, device: str = "cpu"):
        self.name = "avcons"
        self.threshold = 0.5
        self.device = device
        self.model = model.to(device).eval()
        self.syncnet = syncnet

    def analyze(self, frames: np.ndarray, audio: np.ndarray) -> dict:
        from forensix.audiovisual.avcons.train import score

        tracks = track_embeddings(frames, audio, self.syncnet)
        if not tracks:
            raise ValueError("No face track")
        results = []
        for t in tracks:
            prob, windows = score(self.model, t["v"], t["a"], self.device)
            flags = windows > self.threshold
            segments, start = [], None
            for i, f in enumerate(list(flags) + [False]):
                if f and start is None:
                    start = i
                elif not f and start is not None:
                    segments.append((round((t["start"] + start) / FPS, 2), round((t["start"] + i + 4) / FPS, 2)))
                    start = None
            results.append({
                "fake_prob": round(prob, 4),
                "sync_confidence": round(sync_confidence(t["v"], t["a"]), 3),
                "start_s": round(t["start"] / FPS, 2),
                "end_s": round((t["start"] + len(t["v"]) + 4) / FPS, 2),
                "window_fake_prob": np.round(windows, 3).tolist(),
                "inconsistent_segments_s": segments,
            })
        speaker = max(results, key=lambda r: r["sync_confidence"])  # the face that is speaking
        return {**speaker, "tracks": results}

    def predict(self, video) -> Prediction:
        from forensix.audiovisual.io import read_av

        frames, audio = read_av(video, FPS, SAMPLE_RATE)
        n = min(len(frames), len(audio) // AUD_FACT)
        result = self.analyze(frames[:n], audio[: n * AUD_FACT])
        return Prediction(model=self.name, fake_prob=result["fake_prob"], threshold=self.threshold,
                          segment_fake_probs=result["window_fake_prob"], details=result)

    def predict_batch(self, videos) -> list[Prediction]:
        return [self.predict(v) for v in videos]


def build(weights_path: Path | None, device: str = "cpu") -> AVConsistencyDetector:
    from forensix.audiovisual.avcons.train import load
    from forensix.audiovisual.syncnet import build as build_syncnet

    if weights_path is None:
        raise ValueError("avcons has no published weights: train them with scripts/train_avcons.py and "
                         "pass forensix.load('avcons', weights_path=...)")
    model, _ = load(weights_path, device)
    return AVConsistencyDetector(model, build_syncnet(fetch(SYNCNET_WEIGHTS), device), device)


register(
    ModelSpec(
        name="avcons",
        family="audiovisual",
        builder=build,
        weights=None,  # trained by you on real videos (scripts/train_avcons.py)
        extra_weights=(SYNCNET_WEIGHTS, _WM_WEIGHTS["retinaface"]),
        license="ForensiX (model and training code); trained by you on real media. Features: SyncNet "
        "(MIT*). Face detector: RetinaFace (MIT)",
        source="",
        paper="",
        description="Audio-visual consistency learned from real videos only: a temporal transformer over "
        "SyncNet lip and audio embeddings and their sync-distance profiles, trained against "
        "offsets, drift, swapped and spliced audio or faces made from real clips. Probability "
        "that audio and face don't belong together, with per-window localization.",
    )
)
