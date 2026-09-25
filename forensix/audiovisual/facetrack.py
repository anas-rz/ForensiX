"""Face tracks and 224x224 talking-face crops, adapted from syncnet_python's
run_pipeline.py (``track_shot``, ``crop_video``; MIT, see LICENSE-syncnet;
https://github.com/joonson/syncnet_python, commit 907c0b5).

Differences from upstream: faces come from RetinaFace-ResNet50 (MIT) instead of
S3FD, detected on frames downscaled to at most 640 px; no scene-cut detection
(the whole clip is one shot); crops are kept in memory rather than re-encoded
through JPEG and XVID.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import medfilt

IOU_THRESHOLD = 0.5
NUM_FAILED_DET = 25  # missed detections allowed before a track ends
MIN_TRACK = 100  # frames
MIN_FACE_SIZE = 100  # pixels
CROP_SCALE = 0.40
DETECT_MAX_SIDE = 640


def _iou(a, b) -> float:
    x_a, y_a, x_b, y_b = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x_b - x_a) * max(0, y_b - y_a)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def detect_faces(frames: np.ndarray, detector) -> list[list[dict]]:
    """Per frame: ``[{"frame": i, "bbox": [x1, y1, x2, y2]}, ...]`` in full-resolution pixels."""
    import cv2

    scale = min(1.0, DETECT_MAX_SIDE / max(frames.shape[1:3]))
    out = []
    for i, frame in enumerate(frames):
        small = frame if scale == 1.0 else cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        boxes, _ = detector.detect(small[:, :, ::-1].copy())  # RetinaFace takes BGR
        out.append([{"frame": i, "bbox": (b / scale).tolist()} for b in boxes.numpy()])
    return out


def track_faces(detections: list[list[dict]], min_track: int = MIN_TRACK, min_face_size: int = MIN_FACE_SIZE) -> list[dict]:
    """Link detections into tracks by IoU (upstream ``track_shot``); boxes are
    interpolated over missed frames. Returns ``{"frame": idx, "bbox": (n, 4)}``."""
    scenefaces = [list(f) for f in detections]
    tracks = []
    while True:
        track = []
        for framefaces in scenefaces:
            # iterates the live list while removing from it, exactly as upstream
            for face in framefaces:
                if not track:
                    track.append(face)
                    framefaces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= NUM_FAILED_DET:
                    if _iou(face["bbox"], track[-1]["bbox"]) > IOU_THRESHOLD:
                        track.append(face)
                        framefaces.remove(face)
                        continue
                else:
                    break
        if not track:
            break
        if len(track) > min_track:
            framenum = np.array([f["frame"] for f in track])
            bboxes = np.array([f["bbox"] for f in track])
            frame_i = np.arange(framenum[0], framenum[-1] + 1)
            bboxes_i = np.stack([interp1d(framenum, bboxes[:, j])(frame_i) for j in range(4)], axis=1)
            if max(np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]), np.mean(bboxes_i[:, 3] - bboxes_i[:, 1])) > min_face_size:
                tracks.append({"frame": frame_i, "bbox": bboxes_i})
    return tracks


def crop_track(frames: np.ndarray, track: dict, size: int = 224) -> np.ndarray:
    """Smoothed, padded square crops around the face (upstream ``crop_video``)."""
    import cv2

    boxes = track["bbox"]
    s = medfilt(np.maximum(boxes[:, 3] - boxes[:, 1], boxes[:, 2] - boxes[:, 0]) / 2, kernel_size=13)
    y = medfilt((boxes[:, 1] + boxes[:, 3]) / 2, kernel_size=13)
    x = medfilt((boxes[:, 0] + boxes[:, 2]) / 2, kernel_size=13)
    cs = CROP_SCALE
    crops = []
    for i, frame_idx in enumerate(track["frame"]):
        bs = s[i]
        bsi = int(bs * (1 + 2 * cs))
        padded = np.pad(frames[frame_idx], ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=110)
        my, mx = y[i] + bsi, x[i] + bsi
        face = padded[int(my - bs) : int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs))]
        crops.append(cv2.resize(face, (size, size)))
    return np.stack(crops)
