"""Prototype: motion-compensated temporal noise, face vs ring (SDFVD pairs of one split).

For consecutive frames, warp frame t-1 onto t with Farneback optical flow and
take the high-passed residual. Camera sensor noise is temporally independent
and equal across the frame, so the residual energy inside the face skin should
match the ring around the face. A generated, temporally smooth face has less.
Features (median over frame pairs): log(face/ring) residual energy, and its
spread over time. Usage: python proto_temporal.py REPO SDFVD_DIR SPLIT_JSON SPLIT_NAME OUT_JSON
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

repo, root, split_path = map(Path, sys.argv[1:4])
split_name, out_path = sys.argv[4], Path(sys.argv[5])
sys.path.insert(0, str(repo))
from forensix.physics.face.tracker import track_face  # noqa: E402
from forensix.physics.lighting import FACE_OVAL, skin_mask  # noqa: E402

K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def read_all(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames), fps


def masks(xy, shape):
    face = skin_mask(xy, shape).astype(np.uint8)
    size = int(max(4, 0.05 * np.sqrt(face.sum())))
    face_in = cv2.erode(face, np.ones((size, size), np.uint8)).astype(bool)
    oval = np.zeros(shape, np.uint8)
    cv2.fillPoly(oval, [np.round(xy[FACE_OVAL]).astype(np.int32)], 1)
    ring = (cv2.dilate(oval, np.ones((4 * size, 4 * size), np.uint8))
            - cv2.dilate(oval, np.ones((size, size), np.uint8))).astype(bool)
    return face_in, ring


def features(path):
    frames, fps = read_all(path)
    track = track_face(frames, fps)
    ratios, spatial = [], []
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) for f in frames]
    h, w = gray[0].shape
    grid = np.dstack(np.meshgrid(np.arange(w), np.arange(h))).astype(np.float32)
    for t in range(1, len(frames)):
        if not (track.valid[t] and track.valid[t - 1]):
            continue
        face, ring = masks(track.landmarks[t, :, :2], (h, w))
        if face.sum() < 200 or ring.sum() < 200:
            continue
        flow = cv2.calcOpticalFlowFarneback(gray[t].astype(np.uint8), gray[t - 1].astype(np.uint8), None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        warped = cv2.remap(gray[t - 1], grid + flow, None, cv2.INTER_LINEAR)
        resid = np.abs(cv2.filter2D(gray[t] - warped, -1, K))
        grad = np.hypot(cv2.Sobel(gray[t], cv2.CV_32F, 1, 0), cv2.Sobel(gray[t], cv2.CV_32F, 0, 1))
        e = {}
        for name, m in (("face", face), ("ring", ring)):
            flat = m & (grad < np.quantile(grad[m], 0.5))
            e[name] = resid[flat].mean()
        if e["face"] > 0 and e["ring"] > 0:
            ratios.append(np.log(e["face"] / e["ring"]))
    if len(ratios) < 10:
        return None
    return {"temporal_noise_log_ratio": float(np.median(ratios)),
            "temporal_noise_ratio_spread": float(np.std(ratios)), "n_pairs": len(ratios)}


split = json.loads(split_path.read_text())
out = {}
for i in split[split_name]:
    for label, name in (("real", f"videos_real/v{i}.mp4"), ("fake", f"videos_fake/vs{i}.mp4")):
        out[f"{label}/{i}"] = features(root / name)
        print(label, i, out[f"{label}/{i}"], flush=True)
out_path.write_text(json.dumps(out, indent=1))
