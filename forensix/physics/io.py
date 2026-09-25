from os import PathLike

import numpy as np


def read_video(path: str | PathLike, max_seconds: float = 30.0, max_side: int = 640) -> tuple[np.ndarray, float]:
    """All frames (RGB uint8, downscaled so the long side is at most ``max_side``)
    of the first ``max_seconds``, at the video's native frame rate. Physiological
    signals need every frame, unlike the frame-sampling video detectors."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or not fps or fps <= 0:
        raise ValueError(f"Can't read {path}")
    frames = []
    try:
        while len(frames) < max_seconds * fps:
            ok, frame = capture.read()
            if not ok:
                break
            scale = max_side / max(frame.shape[:2])
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return np.stack(frames), float(fps)
