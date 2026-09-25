"""Node states of the physdf graph, sampled on a common 25 Hz time grid.

The video nodes reuse the detectors' measurements (MediaPipe pose, landmarks
and blendshapes; POS rPPG per region; SH lighting on the ICT-FaceKit face), and
so do the audio nodes (YIN F0 and aperiodicity, LPC formants, band energies,
energy-based pauses). Every node has a per-step validity mask (face lost,
lighting undefined, no audio).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from forensix.physics.graph.schema import AUDIO_NODES, NODE_NAMES, NODES, PULSE_NODES, STEP_S, VIDEO_NODES


@dataclass
class GraphSequence:
    """Per-node features ``(T, d_node)`` and validity ``(T,)`` on the step grid."""

    features: dict[str, np.ndarray]
    valid: dict[str, np.ndarray]
    step_s: float = STEP_S
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(next(iter(self.features.values())))

    def save(self, path: str | Path) -> None:
        arrays = {f"x_{n}": self.features[n] for n in self.features}
        arrays.update({f"v_{n}": self.valid[n] for n in self.valid})
        np.savez_compressed(path, step_s=self.step_s, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "GraphSequence":
        z = np.load(path)
        feats = {n: z[f"x_{n}"] for n in NODE_NAMES if f"x_{n}" in z}
        valid = {n: z[f"v_{n}"].astype(bool) for n in feats}
        return cls(feats, valid, float(z["step_s"]))


def _to_grid(values: np.ndarray, times: np.ndarray, n_steps: int) -> np.ndarray:
    """Linear interpolation of ``values`` (T, d) sampled at ``times`` onto the step grid."""
    grid = np.arange(n_steps) * STEP_S
    return np.stack([np.interp(grid, times, values[:, j]) for j in range(values.shape[1])], axis=1)


def _valid_to_grid(valid: np.ndarray, times: np.ndarray, n_steps: int) -> np.ndarray:
    grid = np.arange(n_steps) * STEP_S
    idx = np.clip(np.searchsorted(times, grid), 0, len(times) - 1)
    return valid[idx]


def _blend(track, names: list[str]) -> np.ndarray:
    index = {n: i for i, n in enumerate(track.blendshape_names)}
    return np.mean([track.blendshapes[:, index[n]] for n in names], axis=0)


def video_nodes(frames: np.ndarray, fps: float, track, n_steps: int, lighting: bool = True) -> tuple[dict, dict]:
    """Video node features and validity on the step grid."""
    from forensix.physics.bloodflow import region_rgb
    from forensix.physics.lighting import border_luminance, fit_lighting
    from forensix.physics.face.fit import fit_frame
    from forensix.physics.motion import (
        CHIN, EYE_OUTER, LOWER_LIP_INNER, NOSE_BRIDGE, UPPER_LIP_INNER, _interpolate, rigid_deformation, rotations,
    )
    from forensix.physics.signals.rppg import bandpass, pos

    t = len(track)
    times = np.arange(t) / fps
    valid = track.valid.copy()
    feats: dict[str, np.ndarray] = {}
    if valid.sum() < 2:
        return ({n: np.zeros((n_steps, len(NODES[n][1])), np.float32) for n in VIDEO_NODES},
                {n: np.zeros(n_steps, bool) for n in VIDEO_NODES})
    pose = _interpolate(track.pose, valid)
    lm = _interpolate(track.landmarks, valid)
    bs = _interpolate(track.blendshapes, valid)
    track_i = type(track)(track.fps, valid, lm, bs, track.blendshape_names, pose)

    rot = rotations(pose)
    median_rot = rot.mean()
    feats["head"] = np.concatenate([np.degrees((median_rot.inv() * rot).as_rotvec()),
                                    pose[:, :3, 3] - np.median(pose[:, :3, 3], axis=0)], axis=1)
    feats["skull"] = rigid_deformation(lm)[:, None]
    iod = np.linalg.norm(lm[:, EYE_OUTER[0]] - lm[:, EYE_OUTER[1]], axis=1)
    chin = np.linalg.norm(lm[:, CHIN] - lm[:, NOSE_BRIDGE], axis=1) / iod
    feats["jaw"] = np.stack([_blend(track_i, ["jawOpen"]), chin - np.median(chin),
                             _blend(track_i, ["jawLeft"]) - _blend(track_i, ["jawRight"]),
                             _blend(track_i, ["jawForward"])], axis=1)
    aperture = np.linalg.norm(lm[:, UPPER_LIP_INNER] - lm[:, LOWER_LIP_INNER], axis=1) / iod
    width = np.linalg.norm(lm[:, 61] - lm[:, 291], axis=1) / iod
    feats["lips"] = np.stack([aperture, width - np.median(width), _blend(track_i, ["mouthFunnel"]),
                              _blend(track_i, ["mouthPucker"]), _blend(track_i, ["mouthClose"]),
                              _blend(track_i, ["mouthSmileLeft", "mouthSmileRight"])], axis=1)
    feats["upper_face"] = np.stack([_blend(track_i, ["eyeBlinkLeft"]), _blend(track_i, ["eyeBlinkRight"]),
                                    _blend(track_i, ["browInnerUp"]),
                                    _blend(track_i, ["browDownLeft", "browDownRight"])], axis=1)

    # blood flow: POS pulse per region, normalized; and the raw green-channel change
    rgb = region_rgb(frames, track)
    for name, node in zip(rgb, PULSE_NODES):
        trace = rgb[name]
        ok = ~np.isnan(trace).any(axis=1)
        if ok.sum() < 2 or t < int(2 * fps):
            feats[node] = np.zeros((t, 2))
            continue
        trace = np.stack([np.interp(times, times[ok], trace[ok, c]) for c in range(3)], axis=1)
        pulse = bandpass(pos(trace, fps), fps)
        pulse = pulse / (pulse.std() + 1e-9)
        dlog = np.concatenate([[0.0], np.diff(np.log(np.maximum(trace[:, 1], 1e-3)))])
        feats[node] = np.stack([pulse, dlog / (dlog.std() + 1e-9)], axis=1)

    # lighting: SH fit on the ICT face per frame
    light = np.zeros((t, 6))
    light_ok = np.zeros(t, bool)
    if lighting:
        for i in range(t):
            light[i, 5] = np.log(max(border_luminance(frames[i]), 1e-3))
            if not valid[i]:
                continue
            fit = fit_lighting(frames[i], fit_frame(track, i), track.landmarks[i, :, :2])
            if np.isfinite(fit["residual"]) and fit["coeffs"][0] > 0:
                d = fit["direction"]
                light[i, :5] = [np.log(fit["coeffs"][0]), *np.nan_to_num(d), fit["residual"]]
                light_ok[i] = True
    feats["light"] = light

    out_x, out_v = {}, {}
    for n in VIDEO_NODES:
        out_x[n] = _to_grid(feats[n], times, n_steps).astype(np.float32)
        ok = light_ok if n == "light" else valid
        out_v[n] = _valid_to_grid(ok, times, n_steps)
    return out_x, out_v


def audio_nodes(x: np.ndarray, sr: int, n_steps: int) -> tuple[dict, dict]:
    """Audio node features and validity on the step grid (16 kHz mono in)."""
    from forensix.physics.signals import voice as V

    f0, aper = V.yin_f0(x, sr)
    voiced = ~np.isnan(f0)
    formants = V.lpc_formants(x, sr, f0)[:, :3]
    hop = int(V.HOP_S * sr)
    n = len(f0)

    ref_f0 = np.nanmedian(f0) if voiced.any() else 1.0
    log_f0 = np.where(voiced, np.log2(np.where(voiced, f0, 1.0) / ref_f0), 0.0)
    ref_f = np.nanmedian(formants, axis=0)
    log_f = np.where(np.isfinite(formants), np.log2(formants / np.where(np.isfinite(ref_f), ref_f, 1.0)), 0.0)

    win = int(0.025 * sr)  # 25 ms frames on YIN's 10 ms grid
    padded = np.pad(x, (0, win))
    frames = padded[np.arange(win)[None] + hop * np.arange(n)[:, None]] * np.hanning(win)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(win, 1 / sr)
    edges = np.geomspace(100, 7000, 7)
    bands = np.stack([10 * np.log10(power[:, (freqs >= lo) & (freqs < hi)].mean(1) + 1e-12)
                      for lo, hi in zip(edges[:-1], edges[1:])], axis=1)
    level = 10 * np.log10(power.sum(1) + 1e-12)
    shape = bands - bands.mean(axis=1, keepdims=True)
    level = np.maximum(level - level.max(), -60.0)
    speech = level > -35.0
    since = np.zeros(n)
    quiet = 0
    for i in range(n):  # time since the last pause of >= 0.25 s (breath)
        quiet = 0 if speech[i] else quiet + 1
        since[i] = 0.0 if quiet >= 25 else (since[i - 1] + V.HOP_S if i else 0.0)

    frame_feats = {
        "glottis": np.stack([log_f0, voiced.astype(float), aper], axis=1),
        "tract": np.concatenate([log_f, shape], axis=1),
        "breath": np.stack([level / 10.0, speech.astype(float), np.minimum(since, 10.0)], axis=1),
    }
    times = np.arange(n) * V.HOP_S
    out_x = {k: _to_grid(v, times, n_steps).astype(np.float32) for k, v in frame_feats.items()}
    covered = np.arange(n_steps) * STEP_S <= len(x) / sr
    return out_x, {k: covered.copy() for k in AUDIO_NODES}


def build_sequence(frames: np.ndarray | None = None, fps: float | None = None, track=None,
                   audio: np.ndarray | None = None, sr: int = 16000, lighting: bool = True) -> GraphSequence:
    """Graph states of a clip. Either side may be missing (its nodes are then invalid)."""
    durations = []
    if frames is not None:
        durations.append(len(frames) / fps)
    if audio is not None:
        durations.append(len(audio) / sr)
    if not durations:
        raise ValueError("Need video, audio or both")
    n_steps = int(min(durations) / STEP_S)
    feats, valid = {}, {}
    if frames is not None:
        if track is None:
            from forensix.physics.face.tracker import track_face

            track = track_face(frames, fps)
        fx, fv = video_nodes(frames, fps, track, n_steps, lighting)
        feats.update(fx); valid.update(fv)
    else:
        feats.update({n: np.zeros((n_steps, len(NODES[n][1])), np.float32) for n in VIDEO_NODES})
        valid.update({n: np.zeros(n_steps, bool) for n in VIDEO_NODES})
    if audio is not None:
        ax, av = audio_nodes(np.asarray(audio, dtype=np.float64), sr, n_steps)
        feats.update(ax); valid.update(av)
    else:
        feats.update({n: np.zeros((n_steps, len(NODES[n][1])), np.float32) for n in AUDIO_NODES})
        valid.update({n: np.zeros(n_steps, bool) for n in AUDIO_NODES})
    return GraphSequence({n: feats[n] for n in NODE_NAMES}, {n: valid[n] for n in NODE_NAMES})
