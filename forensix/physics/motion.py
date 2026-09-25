"""phys-motion: a head is a rigid, massive body and the jaw is a hinge.

Checks on the tracked face over time:
- smoothness: head rotation carries little energy above 8 Hz (inertia);
  per-frame generation adds jitter;
- kinematic limits: angular velocity and acceleration stay biomechanically plausible;
- skull rigidity: pose-normalized positions of rigid landmarks (forehead
  midline, nose bridge, eye corners) do not deform; swaps and identity drift warp them;
- jaw coupling: large mouth openings come with the chin dropping (the jaw is a
  hinge); lip-sync fakes often move the lips alone.

Priors are provisional ranges (marked in ``details``) until learned priors,
fitted on real media with ``scripts/calibrate_physics.py``, are loaded.
"""

import numpy as np
from scipy.signal import welch
from scipy.spatial.transform import Rotation

from forensix.physics.base import PhysicsVideoDetector
from forensix.physics.face.tracker import FACE_LANDMARKER, FaceTrack, track_face
from forensix.registry import ModelSpec, register

MIN_SECONDS = 2.0
HF_CUTOFF_HZ = 8.0
# forehead midline, nose bridge, inner/outer eye corners on the MediaPipe mesh
RIGID_LANDMARKS = [10, 151, 9, 8, 168, 6, 197, 195, 33, 133, 362, 263]
EYE_OUTER = (33, 263)
UPPER_LIP_INNER, LOWER_LIP_INNER, CHIN, NOSE_BRIDGE = 13, 14, 152, 168

PRIORS = {
    "rotation_hf_energy_ratio": (0.35, 0.7),
    "p99_angular_velocity_deg_s": (300.0, 800.0),
    "p99_angular_acceleration_deg_s2": (6000.0, 20000.0),
    # p90 RMS residual / inter-ocular distance. On synthetic rigid motion MediaPipe's
    # noise floor is ~0.006; a 10% periodic stretch of the upper face gives ~0.015.
    # Start at ~2x the noise floor; real-video noise is calibrated by learned priors.
    "rigid_deformation": (0.012, 0.03),
    "jaw_lip_coupling": (0.4, 0.0),  # correlation; lower is worse
}


def _interpolate(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill untracked frames by linear interpolation along time."""
    x = x.copy()
    flat = x.reshape(len(x), -1)
    idx = np.flatnonzero(valid)
    for j in range(flat.shape[1]):
        flat[~valid, j] = np.interp(np.flatnonzero(~valid), idx, flat[valid, j])
    return flat.reshape(x.shape)


def rotations(pose: np.ndarray) -> Rotation:
    """Head rotations from 4x4 poses (orthonormalized by SVD)."""
    u, _, vt = np.linalg.svd(pose[:, :3, :3])
    r = u @ vt
    r[np.linalg.det(r) < 0] *= -1
    return Rotation.from_matrix(r)


def angular_kinematics(pose: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Angular velocity (T-1, 3) and acceleration (T-2, 3), in deg/s and deg/s^2."""
    rot = rotations(pose)
    step = rot[1:] * rot[:-1].inv()  # frame-to-frame rotation
    velocity = np.degrees(step.as_rotvec()) * fps
    return velocity, np.diff(velocity, axis=0) * fps


def hf_energy_ratio(velocity: np.ndarray, fps: float, cutoff: float = HF_CUTOFF_HZ) -> float:
    """Share of angular-velocity power above ``cutoff`` Hz (NaN if fps is too low)."""
    if fps / 2 <= cutoff or len(velocity) < 16:
        return float("nan")
    freqs, psd = welch(velocity - velocity.mean(axis=0), fs=fps, nperseg=min(len(velocity), 64), axis=0)
    total = psd[freqs > 0].sum()
    return float(psd[freqs > cutoff].sum() / total) if total > 0 else 0.0


def _align(points: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Similarity (Procrustes) alignment of centred ``points`` onto centred ``target``."""
    p_c, t_c = points - points.mean(axis=0), target - target.mean(axis=0)
    u, s, vt = np.linalg.svd(p_c.T @ t_c)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1, 1, d]) @ vt
    scale = (s * [1, 1, d]).sum() / (p_c ** 2).sum()
    return scale * p_c @ rot


def rigid_deformation(landmarks: np.ndarray) -> np.ndarray:
    """Per-frame RMS residual of rigid landmarks against their consensus shape
    (generalized Procrustes: align every frame to the first, take the median
    aligned shape, align again), relative to the inter-ocular distance."""
    pts = landmarks[:, RIGID_LANDMARKS].astype(np.float64)
    reference = np.median([_align(p, pts[0]) for p in pts], axis=0)
    i0, i1 = RIGID_LANDMARKS.index(EYE_OUTER[0]), RIGID_LANDMARKS.index(EYE_OUTER[1])
    iod = np.linalg.norm(reference[i0] - reference[i1])
    return np.array([np.sqrt(((_align(p, reference) - reference) ** 2).sum(axis=1).mean()) / iod for p in pts])


def jaw_lip_coupling(landmarks: np.ndarray, min_aperture_std: float = 0.03) -> float:
    """Correlation between lip aperture and chin drop, both relative to the
    inter-ocular distance. NaN when the mouth barely moves (nothing to check)."""
    iod = np.linalg.norm(landmarks[:, EYE_OUTER[0]] - landmarks[:, EYE_OUTER[1]], axis=1)
    aperture = np.linalg.norm(landmarks[:, UPPER_LIP_INNER] - landmarks[:, LOWER_LIP_INNER], axis=1) / iod
    chin_drop = np.linalg.norm(landmarks[:, CHIN] - landmarks[:, NOSE_BRIDGE], axis=1) / iod
    if aperture.std() < min_aperture_std or chin_drop.std() == 0:
        return float("nan")
    return float(np.corrcoef(aperture, chin_drop)[0, 1])


class MotionDetector(PhysicsVideoDetector):
    name = "phys-motion"
    provisional_priors = PRIORS

    def analyze(self, frames: np.ndarray, fps: float, track: FaceTrack | None = None) -> dict:
        track = track or track_face(frames, fps)
        return self.analyze_track(track)

    def analyze_track(self, track: FaceTrack) -> dict:
        fps = track.fps
        if track.valid.mean() < 0.8:
            raise ValueError("Face tracked in under 80% of frames")
        if len(track) / fps < MIN_SECONDS:
            raise ValueError(f"Need at least {MIN_SECONDS:.0f} s of video")
        pose = _interpolate(track.pose, track.valid)
        landmarks = _interpolate(track.landmarks, track.valid)
        velocity, acceleration = angular_kinematics(pose, fps)
        speed = np.linalg.norm(velocity, axis=1)
        accel = np.linalg.norm(acceleration, axis=1)
        deformation = rigid_deformation(landmarks)
        measured = {
            "rotation_hf_energy_ratio": hf_energy_ratio(velocity, fps),
            "p99_angular_velocity_deg_s": float(np.percentile(speed, 99)),
            "p99_angular_acceleration_deg_s2": float(np.percentile(accel, 99)),
            # 90th percentile: warping comes in episodes; a median would hide them
            "rigid_deformation": float(np.percentile(deformation, 90)),
            "jaw_lip_coupling": jaw_lip_coupling(landmarks),
        }
        score, evidence = self.report(measured)
        worst = int(np.argmax(deformation))
        return {
            "score": score,
            "constraints": evidence,
            "timeline": {  # for localization: per-frame signals
                "angular_speed_deg_s": np.round(speed, 2).tolist(),
                "rigid_deformation": np.round(deformation, 4).tolist(),
            },
            "most_deformed_frame": worst,
        }


register(
    ModelSpec(
        name="phys-motion",
        family="physics",
        builder=lambda weights_path, device="cpu": MotionDetector(),
        weights=FACE_LANDMARKER,
        license="ForensiX (algorithmic). Face landmarks and head pose: MediaPipe FaceLandmarker (Apache-2.0)",
        source="https://github.com/google-ai-edge/mediapipe",
        paper="",
        description="Head and jaw biomechanics (physdf): smoothness of head rotation (energy "
        "above 8 Hz), angular velocity/acceleration limits, skull rigidity (Procrustes "
        "residual of rigid landmarks) and jaw-lip coupling. Anomaly score in [0, 1] with "
        "per-constraint evidence and per-frame timelines. Priors: learned from real media when available, else provisional.",
    )
)
