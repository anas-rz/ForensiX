"""Experiment (not part of the package): "one camera" features on SDFVD pairs.

The dev/test split of the 53 pairs (split.json, seed 0) was fixed before any
feature was computed. Features were explored on dev only; one
(noise_ratio_spread, lower = fake) was then chosen and evaluated once on test.
See physics_benchmark.md.

Per frame: face skin (inner oval minus eyes/brows/lips, eroded) vs a ring
around the face oval. Features (log ratios face/ring, median over frames):
noise (Immerkaer estimator on flat pixels), sharpness (Laplacian energy on
flat pixels), flicker (temporal high-pass of mean luminance).
Usage: python camera_features.py REPO SDFVD_DIR split.json OUT_JSON [dev|test]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

repo, root, split_path, out_path = map(Path, sys.argv[1:5])
split_name = sys.argv[5] if len(sys.argv) > 5 else "dev"
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


def features(path):
    frames, fps = read_all(path)
    track = track_face(frames, fps)
    noise, sharp, lum_face, lum_ring = [], [], [], []
    for t in np.flatnonzero(track.valid):
        img = frames[t]
        y = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        xy = track.landmarks[t, :, :2]
        h, w = y.shape
        face = skin_mask(xy, (h, w)).astype(np.uint8)
        size = int(max(4, 0.05 * np.sqrt(face.sum())))
        face_in = cv2.erode(face, np.ones((size, size), np.uint8)).astype(bool)
        oval = np.zeros((h, w), np.uint8)
        cv2.fillPoly(oval, [np.round(xy[FACE_OVAL]).astype(np.int32)], 1)
        ring = (cv2.dilate(oval, np.ones((4 * size, 4 * size), np.uint8)) - cv2.dilate(oval, np.ones((size, size), np.uint8))).astype(bool)
        grad = np.hypot(cv2.Sobel(y, cv2.CV_32F, 1, 0), cv2.Sobel(y, cv2.CV_32F, 0, 1))
        hp = np.abs(cv2.filter2D(y, -1, K))
        lap = cv2.Laplacian(cv2.GaussianBlur(y, (0, 0), 1.0), cv2.CV_32F) ** 2
        vals = {}
        for name, m in (("face", face_in), ("ring", ring)):
            if m.sum() < 200:
                break
            flat = m & (grad < np.quantile(grad[m], 0.5))  # flattest half: less texture
            vals[name] = (np.sqrt(np.pi / 2) / 6 * hp[flat].mean(), lap[flat].mean(), y[m].mean())
        if len(vals) < 2:
            continue
        noise.append(np.log(vals["face"][0] / vals["ring"][0]))
        sharp.append(np.log(vals["face"][1] / vals["ring"][1]))
        lum_face.append(vals["face"][2])
        lum_ring.append(vals["ring"][2])
    if len(noise) < 10:
        return None

    def hf(x):
        x = np.log(np.asarray(x))
        return np.std(x - cv2.GaussianBlur(x.reshape(-1, 1), (1, 0), 3).ravel()) if len(x) > 5 else np.nan

    return {"noise_log_ratio": float(np.median(noise)), "sharpness_log_ratio": float(np.median(sharp)),
            "noise_ratio_spread": float(np.std(noise)), "flicker_log_ratio": float(np.log(hf(lum_face) / hf(lum_ring))),
            "n_frames": len(noise)}


split = json.loads(split_path.read_text())
out = {}
for i in split[split_name]:
    for label, name in (("real", f"videos_real/v{i}.mp4"), ("fake", f"videos_fake/vs{i}.mp4")):
        out[f"{label}/{i}"] = features(root / name)
        print(label, i, out[f"{label}/{i}"], flush=True)
out_path.write_text(json.dumps(out, indent=1))
