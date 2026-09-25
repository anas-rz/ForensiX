"""Frames from images and videos for enrollment and verification."""

from os import PathLike
from pathlib import Path

import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_image(path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def sample_frames(path: str | PathLike | np.ndarray, fps: float = 2.0, max_frames: int = 120) -> tuple[list[np.ndarray], list[float]]:
    """RGB uint8 frames and their times (s): an image gives one frame; a video
    gives ``fps`` frames per second, at most ``max_frames``. An array is taken
    as one RGB image."""
    import cv2

    if isinstance(path, np.ndarray):
        return [path], [0.0]
    if is_image(path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Can't read image {path}")
        return [img[:, :, ::-1].copy()], [0.0]
    cap = cv2.VideoCapture(str(path))
    native = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or not native or native <= 0:
        raise ValueError(f"Can't read video {path}")
    step = max(1, int(round(native / fps)))
    frames, times, i = [], [], 0
    try:
        while len(frames) < max_frames:
            ok = cap.grab()
            if not ok:
                break
            if i % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    frames.append(frame[:, :, ::-1].copy())
                    times.append(i / native)
            i += 1
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"No frames in {path}")
    return frames, times
