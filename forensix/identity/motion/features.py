"""Per-frame facial-motion features: how the head moves and the face expresses,
without who it looks like.

From the MediaPipe FaceLandmarker track (``forensix.physics.face.tracker``):
- head rotation relative to the clip's mean rotation (rotation vector, degrees),
  and angular velocity (degrees per second);
- head translation relative to the clip's median, in face widths, and its
  velocity (face widths per second);
- the 52 ARKit expression blendshapes (eyes, brows, jaw, mouth, cheeks).

Blendshapes and pose are expression and motion parameters, like the 3DMM
expression and pose coefficients ID-Reveal uses. The face's shape (identity
coefficients) is left out, so the features describe behaviour, not appearance.
Everything is resampled to 25 fps.
"""

import numpy as np

FPS = 25.0
N_POSE = 12


def motion_features(track, fps_out: float = FPS) -> tuple[np.ndarray, np.ndarray]:
    """(T, 64) features and (T,) validity at ``fps_out``."""
    from forensix.physics.motion import EYE_OUTER, _interpolate, rotations

    valid = track.valid
    if valid.sum() < 2:
        raise ValueError("Face tracked in fewer than 2 frames")
    pose = _interpolate(track.pose, valid)
    lm = _interpolate(track.landmarks, valid)
    bs = _interpolate(track.blendshapes, valid)
    rot = rotations(pose)
    rel = np.degrees((rot.mean().inv() * rot).as_rotvec())
    face_w = np.median(np.linalg.norm(lm[:, EYE_OUTER[0], :2] - lm[:, EYE_OUTER[1], :2], axis=1))
    trans_px = lm[:, 1, :2]  # nose tip in pixels, face widths below
    trans = np.concatenate([(trans_px - np.median(trans_px, 0)) / max(face_w, 1e-6),
                            (pose[:, 2:3, 3] - np.median(pose[:, 2, 3])) / 10.0], axis=1)  # depth in dm
    fps_in = track.fps
    ang_vel = np.gradient(rel, axis=0) * fps_in
    trans_vel = np.gradient(trans, axis=0) * fps_in
    feats = np.concatenate([rel, ang_vel, trans, trans_vel, bs], axis=1)
    t_in = np.arange(len(feats)) / fps_in
    t_out = np.arange(int(t_in[-1] * fps_out) + 1) / fps_out
    out = np.stack([np.interp(t_out, t_in, feats[:, j]) for j in range(feats.shape[1])], 1).astype(np.float32)
    idx = np.clip(np.round(t_out * fps_in).astype(int), 0, len(valid) - 1)
    return out, valid[idx]


def windows(n: int, length: int, hop: int) -> list[int]:
    """Start indices of ``length``-frame windows every ``hop`` frames (at least one if n >= length)."""
    return list(range(0, n - length + 1, hop)) if n >= length else []
