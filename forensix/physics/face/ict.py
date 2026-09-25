"""ICT-FaceKit morphable face model (USC ICT, MIT, see LICENSE-ICT-FaceKit):
the face region (9,409 vertices) of the generic neutral mesh and its 53
ARKit-named expression shapes, driven by MediaPipe blendshapes.

The source OBJs (~140 MB, pinned commit, SHA-256-checked) are downloaded once
and converted to a compact cache file. Identity shapes are not used (yet):
lighting normals depend mostly on the generic face geometry.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from forensix.hub import WeightFile, cache_dir, fetch

FACE_VERTS = 9409  # "Face" geometry: vertices [0, 9409)
_MANIFEST = json.loads((Path(__file__).parent / "ict_manifest.json").read_text())
_URL = "https://raw.githubusercontent.com/USC-ICT/ICT-FaceKit/{}/FaceXModel/{}.obj"
ICT_FILES = {
    name: WeightFile(_URL.format(_MANIFEST["commit"], name), sha, f"physics/ict/{name}.obj")
    for name, sha in _MANIFEST["files"].items()
}
EXPRESSIONS: list[str] = _MANIFEST["expressions"]
# the 68 standard (iBUG) facial landmarks on the ICT mesh
LANDMARKS_68 = np.array(_MANIFEST["landmark_verts_68"])

# (iBUG 68 index, MediaPipe index) pairs for well-defined points: eye corners,
# nose bridge / tip / base, mouth corners, lip midpoints, inner lips, chin
ANCHORS = [(36, 33), (39, 133), (42, 362), (45, 263), (27, 168), (30, 1), (33, 2),
           (48, 61), (54, 291), (51, 0), (57, 17), (62, 13), (66, 14), (8, 152)]


def _read_obj(path: Path, with_faces: bool = False):
    verts, faces = [], []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            verts.append([float(v) for v in line.split()[1:4]])
        elif with_faces and line.startswith("f "):
            faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:]])
    return np.array(verts, np.float32), faces


def _build_cache(target: Path) -> None:
    neutral, polys = _read_obj(fetch(ICT_FILES["generic_neutral_mesh"]), with_faces=True)
    tris = []
    for p in polys:  # keep face-region polygons, fan-triangulate quads
        if max(p) < FACE_VERTS:
            tris += [[p[0], p[i], p[i + 1]] for i in range(1, len(p) - 1)]
    deltas = np.stack([_read_obj(fetch(ICT_FILES[e]))[0][:FACE_VERTS] - neutral[:FACE_VERTS] for e in EXPRESSIONS])
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, neutral=neutral[:FACE_VERTS], triangles=np.array(tris, np.int32),
                        expression_deltas=deltas.astype(np.float32))


@lru_cache(maxsize=1)
def load_ict() -> dict[str, np.ndarray]:
    """{"neutral": (9409, 3) cm, "triangles": (F, 3), "expression_deltas": (53, 9409, 3)}."""
    path = cache_dir() / "weights" / "physics" / f"ict_face_{_MANIFEST['commit'][:8]}.npz"
    if not path.exists():
        _build_cache(path)
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def mediapipe_to_ict_weights(names: list[str], scores: np.ndarray) -> np.ndarray:
    """Map MediaPipe (ARKit) blendshape scores (..., 52) to ICT expression weights (..., 53)."""
    out = np.zeros(scores.shape[:-1] + (len(EXPRESSIONS),), np.float32)
    index = {e: i for i, e in enumerate(EXPRESSIONS)}
    for j, name in enumerate(names):
        if name in index:  # e.g. jawOpen, jawLeft
            targets = [name]
        elif name.endswith("Left") and name[:-4] + "_L" in index:
            targets = [name[:-4] + "_L"]
        elif name.endswith("Right") and name[:-5] + "_R" in index:
            targets = [name[:-5] + "_R"]
        else:  # browInnerUp, cheekPuff: one MediaPipe shape drives both sides
            targets = [t for t in (name + "_L", name + "_R") if t in index]
        for t in targets:
            out[..., index[t]] = scores[..., j]
    return out


def face_mesh(expression_weights: np.ndarray) -> np.ndarray:
    """ICT face-region vertices (9409, 3) for one frame's expression weights."""
    ict = load_ict()
    return ict["neutral"] + np.tensordot(expression_weights, ict["expression_deltas"], axes=1)


def vertex_normals(verts: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals."""
    a, b, c = verts[triangles[:, 0]], verts[triangles[:, 1]], verts[triangles[:, 2]]
    face_n = np.cross(b - a, c - a)
    normals = np.zeros_like(verts)
    for k in range(3):
        np.add.at(normals, triangles[:, k], face_n)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
