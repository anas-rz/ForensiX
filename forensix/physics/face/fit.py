"""Place the ICT face in each video frame: expression from MediaPipe
blendshapes, then a 3-D similarity transform (Umeyama) from ICT anchor vertices
onto MediaPipe's 3-D landmarks (x, y in pixels; z depth, larger = farther)."""

from dataclasses import dataclass

import numpy as np

from forensix.physics.face import ict
from forensix.physics.face.tracker import FaceTrack


def similarity_3d(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """(scale, R, t) minimizing ||s R src + t - dst|| (Umeyama 1991, proper rotation)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    u, sv, vt = np.linalg.svd(d_c.T @ s_c / len(src))
    d = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[2] = -1
    rot = u @ np.diag(d) @ vt
    scale = (sv * d).sum() / s_c.var(0).sum()
    return float(scale), rot, mu_d - scale * rot @ mu_s


@dataclass
class FittedFace:
    vertices: np.ndarray  # (9409, 3) image space: x, y pixels, z depth (larger = farther)
    normals: np.ndarray  # (9409, 3) unit, image space; facing the camera means normal z < 0
    anchor_rms: float  # anchor fit residual in pixels


def fit_frame(track: FaceTrack, i: int) -> FittedFace:
    weights = ict.mediapipe_to_ict_weights(track.blendshape_names, track.blendshapes[i])
    mesh = ict.face_mesh(weights)
    src = mesh[ict.LANDMARKS_68[[a for a, _ in ict.ANCHORS]]]
    dst = track.landmarks[i, [m for _, m in ict.ANCHORS]].astype(np.float64)
    scale, rot, t = similarity_3d(src, dst)
    verts = scale * mesh @ rot.T + t
    normals = ict.vertex_normals(mesh, ict.load_ict()["triangles"]) @ rot.T
    rms = float(np.sqrt(((scale * src @ rot.T + t - dst) ** 2).sum(1).mean()))
    return FittedFace(verts, normals, rms)
