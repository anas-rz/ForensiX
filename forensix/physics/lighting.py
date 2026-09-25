"""phys-lighting: a face is lit by the same light as the scene it is in.

Per frame, the ICT face (driven by MediaPipe) gives a surface normal for every
visible skin point; a 2nd-order spherical-harmonic (SH) lighting model is
fitted to skin luminance by least squares (Ramamoorthi & Hanrahan 2001:
Lambertian shading is well captured by 9 SH coefficients). Checks:
- light-direction stability: with a fixed scene light the dominant direction
  (1st-order SH, camera frame) changes slowly; per-frame synthesized faces
  flicker;
- shading plausibility: skin shading is explained by one smooth light field
  (normalized fit residual);
- face-scene coupling: when the scene's brightness changes, the face's ambient
  light changes with it; composited faces don't follow.

Priors are provisional ranges (marked in ``details``) until learned priors,
fitted on real media with ``scripts/calibrate_physics.py``, are loaded.
"""

import numpy as np

from forensix.physics.base import PhysicsVideoDetector
from forensix.physics.face.fit import fit_frame
from forensix.physics.face.ict import ICT_FILES
from forensix.physics.face.tracker import FACE_LANDMARKER, FaceTrack, track_face
from forensix.registry import ModelSpec, register

MIN_SECONDS = 2.0
# MediaPipe face oval and the non-skin parts to exclude (eyes, brows, lips)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
             148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
EXCLUDE = {
    "left_eye": [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
    "right_eye": [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382],
    "left_brow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "right_brow": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    "lips": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146],
}
BORDER = 0.08  # outer band of the frame used as "the surroundings"

# Provisional; anchored above the baseline measured on a real photograph (tests/data)
# with a static light: direction noise 2-5 deg, shading residual ~0.14.
PRIORS = {
    "light_direction_jitter_deg": (6.0, 25.0),  # p90 frame-to-frame change
    "shading_residual": (0.20, 0.40),  # RMS residual / mean skin luminance
    "face_scene_coupling": (0.5, 0.0),  # correlation; lower is worse
}


def sh_basis(normals: np.ndarray) -> np.ndarray:
    """Real SH basis up to order 2 (9 columns) at unit normals (constants omitted:
    they are absorbed by the fitted coefficients)."""
    x, y, z = normals[:, 0], normals[:, 1], normals[:, 2]
    return np.stack([np.ones_like(x), y, z, x, x * y, y * z, 3 * z ** 2 - 1, x * z, x ** 2 - y ** 2], axis=1)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.2126, 0.7152, 0.0722])


def _bilinear(img: np.ndarray, xy: np.ndarray) -> np.ndarray:
    import cv2

    maps = xy.astype(np.float32).reshape(-1, 1, 2)
    return cv2.remap(img.astype(np.float32), maps[..., 0], maps[..., 1], cv2.INTER_LINEAR).reshape(len(xy), -1)


def skin_mask(landmarks_xy: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Pixels inside the face oval, outside eyes, brows and lips."""
    import cv2

    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(landmarks_xy[FACE_OVAL]).astype(np.int32)], 1)
    for idx in EXCLUDE.values():
        cv2.fillPoly(mask, [np.round(landmarks_xy[idx]).astype(np.int32)], 0)
    return mask.astype(bool)


def skin_vertices(fitted, landmarks_xy: np.ndarray, shape: tuple[int, int], min_facing: float = 0.2) -> np.ndarray:
    """Indices of ICT vertices on visible skin: facing the camera and inside the skin mask."""
    xy = np.round(fitted.vertices[:, :2]).astype(int)
    inside = (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < shape[1]) & (xy[:, 1] < shape[0])
    keep = inside & (fitted.normals[:, 2] < -min_facing)
    mask = skin_mask(landmarks_xy, shape)
    keep[keep] &= mask[xy[keep, 1], xy[keep, 0]]
    return np.flatnonzero(keep)


def fit_lighting(frame: np.ndarray, fitted, landmarks_xy: np.ndarray) -> dict:
    """SH lighting of one frame: coefficients (9,), normalized residual, dominant
    light direction (camera frame, unit or NaN when the light is flat)."""
    idx = skin_vertices(fitted, landmarks_xy, frame.shape[:2])
    if len(idx) < 200:
        return {"coeffs": np.full(9, np.nan), "residual": np.nan, "direction": np.full(3, np.nan), "n": len(idx)}
    lum = luminance(_bilinear(frame, fitted.vertices[idx, :2]))
    basis = sh_basis(fitted.normals[idx])
    coeffs, *_ = np.linalg.lstsq(basis, lum, rcond=None)
    residual = np.sqrt(np.mean((basis @ coeffs - lum) ** 2)) / max(lum.mean(), 1e-6)
    first = coeffs[[3, 1, 2]]  # x, y, z components
    strength = np.linalg.norm(first) / max(abs(coeffs[0]), 1e-6)
    direction = first / np.linalg.norm(first) if strength > 0.1 else np.full(3, np.nan)
    return {"coeffs": coeffs, "residual": float(residual), "direction": direction, "n": len(idx)}


def border_luminance(frame: np.ndarray) -> float:
    h, w = frame.shape[:2]
    by, bx = max(1, int(BORDER * h)), max(1, int(BORDER * w))
    band = np.concatenate([frame[:by].reshape(-1, 3), frame[-by:].reshape(-1, 3),
                           frame[:, :bx].reshape(-1, 3), frame[:, -bx:].reshape(-1, 3)])
    return float(luminance(band.astype(np.float64)).mean())


def direction_jitter(directions: np.ndarray) -> float:
    """p90 angle (deg) between consecutive defined light directions (NaN if too few)."""
    ok = ~np.isnan(directions).any(axis=1)
    d = directions[ok]
    if len(d) < 3:
        return float("nan")
    cos = np.clip((d[1:] * d[:-1]).sum(1), -1, 1)
    return float(np.percentile(np.degrees(np.arccos(cos)), 90))


def coupling(face_ambient: np.ndarray, scene: np.ndarray, min_rel_std: float = 0.02) -> float:
    """Correlation of log face ambient light and log scene brightness; NaN when
    the scene's brightness barely changes (nothing to check)."""
    ok = (face_ambient > 0) & (scene > 0)
    if ok.sum() < 10:
        return float("nan")
    a, s = np.log(face_ambient[ok]), np.log(scene[ok])
    if s.std() < min_rel_std or a.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, s)[0, 1])


class LightingDetector(PhysicsVideoDetector):
    name = "phys-lighting"
    provisional_priors = PRIORS

    def analyze(self, frames: np.ndarray, fps: float, track: FaceTrack | None = None) -> dict:
        track = track or track_face(frames, fps)
        if track.valid.mean() < 0.8:
            raise ValueError("Face tracked in under 80% of frames")
        if len(frames) / fps < MIN_SECONDS:
            raise ValueError(f"Need at least {MIN_SECONDS:.0f} s of video")
        coeffs, residuals, directions, scene = [], [], [], []
        for i, frame in enumerate(frames):
            scene.append(border_luminance(frame))
            if not track.valid[i]:
                coeffs.append(np.full(9, np.nan)); residuals.append(np.nan); directions.append(np.full(3, np.nan))
                continue
            light = fit_lighting(frame, fit_frame(track, i), track.landmarks[i, :, :2])
            coeffs.append(light["coeffs"]); residuals.append(light["residual"]); directions.append(light["direction"])
        coeffs, directions = np.array(coeffs), np.array(directions)
        residuals, scene = np.array(residuals), np.array(scene)
        measured = {
            "light_direction_jitter_deg": direction_jitter(directions),
            "shading_residual": float(np.nanmedian(residuals)) if np.isfinite(residuals).any() else float("nan"),
            "face_scene_coupling": coupling(coeffs[:, 0], scene),
        }
        score, evidence = self.report(measured)
        return {
            "score": score,
            "constraints": evidence,
            "timeline": {
                "light_direction": np.round(directions, 3).tolist(),
                "face_ambient": np.round(coeffs[:, 0], 2).tolist(),
                "scene_luminance": np.round(scene, 2).tolist(),
                "shading_residual": np.round(residuals, 4).tolist(),
            },
        }


register(
    ModelSpec(
        name="phys-lighting",
        family="physics",
        builder=lambda weights_path, device="cpu": LightingDetector(),
        weights=FACE_LANDMARKER,
        extra_weights=tuple(ICT_FILES.values()),
        license="ForensiX (algorithmic). Face tracking: MediaPipe FaceLandmarker (Apache-2.0). "
        "3-D face: ICT-FaceKit (MIT)",
        source="https://github.com/USC-ICT/ICT-FaceKit",
        paper="https://doi.org/10.1145/383259.383317",
        description="Lighting physics (physdf): 9-coefficient spherical-harmonic lighting fitted to "
        "skin on the ICT-FaceKit face (placed via MediaPipe); checks light-direction stability, "
        "shading plausibility and face-scene brightness coupling. Anomaly score in [0, 1] with "
        "evidence and per-frame timelines. Priors: learned from real media when available, else provisional.",
    )
)
