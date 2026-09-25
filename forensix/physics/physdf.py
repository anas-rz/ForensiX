"""physdf: every physics check on a clip, the graph prior, localization and a report.

- The five constraint detectors run on the clip. The face is tracked once and
  shared by the video detectors.
- The graph prior (a GNN trained on real media only, ``scripts/train_physdf.py``)
  predicts each physical subsystem from the others. Its residuals become
  constraints in standard deviations above held-out real clips: skull, muscles,
  blood flow, lighting, voice, and the audio-visual articulation in both
  directions.
- Score: the most-violated constraint over everything, as in each detector.
- Localization: 1 s windows whose graph residual exceeds 99% of real windows,
  with the node, face region or modality that carries it.
- Report: a forensic report built only from the measured evidence and checked
  against it (``forensix.physics.report``).

Without a trained graph checkpoint (``forensix.load("physdf", weights_path=...)``),
the graph constraints are left out and the report says so.
"""

from pathlib import Path

import numpy as np

from forensix.physics import AUDIO_DETECTORS, DETECTORS, VIDEO_DETECTORS
from forensix.physics.base import _PriorsMixin
from forensix.physics.face.ict import ICT_FILES
from forensix.physics.face.tracker import FACE_LANDMARKER
from forensix.physics.graph.schema import NODES, RESIDUAL_GROUPS, STEP_S
from forensix.registry import ModelSpec, register
from forensix.types import Prediction

# graph residuals in standard deviations above held-out real clips
GRAPH_PRIORS = {g: (3.0, 8.0) for g in RESIDUAL_GROUPS}
GROUP_MODALITY = {
    "graph_skull_residual": "video", "graph_muscle_residual": "video", "graph_bloodflow_residual": "video",
    "graph_lighting_residual": "video", "graph_voice_residual": "audio",
    "graph_av_mouth_from_audio": "audio-visual", "graph_av_audio_from_video": "audio-visual",
}
DETECTOR_MODALITY = {**{n: "video" for n in VIDEO_DETECTORS}, **{n: "audio" for n in AUDIO_DETECTORS}}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _segments(flags: np.ndarray, window_s: float) -> list[tuple[float, float]]:
    out, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((round(start * window_s, 2), round(i * window_s, 2)))
            start = None
    return out


class PhysDF(_PriorsMixin):
    name = "physdf"
    threshold = None
    provisional_priors = GRAPH_PRIORS

    def __init__(self, graph_path: str | Path | None = None, device: str = "cpu", priors=None):
        self.detectors = {n: cls() for n, cls in DETECTORS.items()}
        super().__init__(priors)
        self.device = device
        self.graph = None
        if graph_path is not None:
            from forensix.physics.graph import train as G

            self.graph = G.load(graph_path, device)

    def load_priors(self, priors=None) -> None:
        super().load_priors(priors)
        for d in self.detectors.values():
            d.load_priors(priors)

    # -- graph ---------------------------------------------------------------

    def _graph(self, frames, fps, track, audio, sr) -> tuple[dict, dict]:
        from forensix.physics.graph import train as G
        from forensix.physics.graph.features import build_sequence

        model, norm, calibrations, _ = self.graph
        seq = build_sequence(frames, fps, track, audio, sr)
        cond = G.condition(seq)
        calibration = calibrations.get(cond, {})  # real residuals under the same input condition
        res = G.node_residuals(model, norm, seq, self.device)
        measured, localization = {}, {"window_s": G.WINDOW_S, "timeline_z": {}, "segments": [], "condition": cond}
        for group, series in res["groups"].items():
            cal = calibration.get(group)
            windows = G.window_scores(series)
            summary = G.clip_summary(windows)
            if cal is None or not np.isfinite(summary):
                measured[group] = float("nan")
                continue
            measured[group] = (summary - cal["clip_mean"]) / cal["clip_std"]
            z = (windows - cal["window_mean"]) / cal["window_std"]
            localization["timeline_z"][group] = [None if not np.isfinite(v) else round(float(v), 2) for v in z]
            flags = np.nan_to_num(windows, nan=-np.inf) > cal["window_p99"]
            for start, end in _segments(flags, G.WINDOW_S):
                a, b = int(start / STEP_S), int(end / STEP_S)
                nodes = {n: float(np.nanmean(r[a:b])) for n, r in res["nodes"][group].items() if np.isfinite(r[a:b]).any()}
                localization["segments"].append({
                    "start_s": start, "end_s": end, "constraint": group, "modality": GROUP_MODALITY[group],
                    "peak_z": round(float(np.nanmax(z[int(start / G.WINDOW_S) : int(end / G.WINDOW_S)])), 2),
                    "node": max(nodes, key=nodes.get) if nodes else None,
                })
            quantiles = cal["clip_quantiles"]
            exceeded = [float(q) for q, v in quantiles.items() if summary > v]
            localization.setdefault("real_rarity", {})[group] = max(exceeded) if exceeded else None
        localization["segments"].sort(key=lambda s: -s["peak_z"])
        return measured, localization

    # -- analysis ------------------------------------------------------------

    def analyze(self, frames: np.ndarray | None = None, fps: float | None = None, audio: np.ndarray | None = None,
                sr: int = 16000, track=None) -> dict:
        from forensix.physics.report import build_report

        if frames is None and audio is None:
            raise ValueError("Need video, audio or both")
        results, status = {}, {}
        if frames is not None:
            if track is None:
                from forensix.physics.face.tracker import track_face

                track = track_face(frames, fps)
            for n in VIDEO_DETECTORS:
                try:
                    results[n] = self.detectors[n].analyze(frames, fps, track=track)
                except ValueError as e:
                    status[n] = f"not applicable: {e}"
        if audio is not None:
            for n in AUDIO_DETECTORS:
                try:
                    results[n] = self.detectors[n].analyze(audio, sr)
                except ValueError as e:
                    status[n] = f"not applicable: {e}"
        for n in DETECTORS:
            status.setdefault(n, "ran" if n in results else "no " + ("video" if n in VIDEO_DETECTORS else "audio"))

        constraints = {f"{n}/{k}": v for n, r in results.items() for k, v in r["constraints"].items()}
        localization: dict = {"segments": []}
        if self.graph is not None:
            measured, localization = self._graph(frames, fps, track, audio, sr)
            _, evidence = self.report(measured)
            for k, e in evidence.items():
                e["real_rarity"] = localization.get("real_rarity", {}).get(k)
                constraints[f"physdf/{k}"] = e
            status["graph"] = "ran"
        else:
            status["graph"] = "no trained graph prior loaded (train one with scripts/train_physdf.py)"
        violations = [e["violation"] for e in constraints.values() if e.get("violation") is not None]
        score = float(max(violations)) if violations else 0.0

        modalities: dict[str, float] = {}
        for key, e in constraints.items():
            if e.get("violation") is None:
                continue
            det, k = key.split("/", 1)
            m = GROUP_MODALITY[k] if det == "physdf" else DETECTOR_MODALITY[det]
            modalities[m] = max(modalities.get(m, 0.0), e["violation"])
        result = {
            "score": score,
            "constraints": constraints,
            "modalities": modalities,
            "localization": localization,
            "status": status,
            "detectors": results,
        }
        result["report"] = build_report(result)
        return result

    def predict(self, media, audio_path=None) -> Prediction:
        """``media``: a video (its sound track is used if it has one) or an audio file."""
        from forensix.audio.io import load_audio

        path = Path(media)
        frames = fps = audio = None
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            from forensix.physics.io import read_video

            frames, fps = read_video(path)
        try:
            audio = load_audio(audio_path or path, 16000)
            if frames is not None:
                audio = audio[: int(len(frames) / fps * 16000)]
        except ValueError:
            if frames is None:
                raise
        result = self.analyze(frames, fps, audio, 16000)
        return Prediction(model=self.name, fake_prob=result["score"], threshold=None,
                          score_type="anomaly", details=result)

    def predict_batch(self, inputs) -> list[Prediction]:
        return [self.predict(x) for x in inputs]


register(
    ModelSpec(
        name="physdf",
        family="physics",
        builder=lambda weights_path, device="cpu": PhysDF(weights_path, device),
        weights=None,  # the graph prior is trained by the user (scripts/train_physdf.py); pass weights_path
        extra_weights=(FACE_LANDMARKER, *ICT_FILES.values()),
        license="ForensiX (algorithmic + graph prior trained on real media). Face tracking: MediaPipe "
        "FaceLandmarker (Apache-2.0). 3-D face: ICT-FaceKit (MIT)",
        source="",
        paper="",
        description="Physics-based deepfake detection: all five physics detectors plus a spatio-temporal "
        "graph prior (GNN trained on real media only) over face, pulse, lighting, voice and "
        "audio-visual articulation. Anomaly score in [0, 1], the violated constraints, localization "
        "in time, face region and modality, and a verified forensic report.",
    )
)

assert set(GROUP_MODALITY) == set(RESIDUAL_GROUPS) and all(n in NODES for g in RESIDUAL_GROUPS.values() for n in g[0])
