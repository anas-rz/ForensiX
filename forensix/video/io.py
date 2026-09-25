from os import PathLike

import numpy as np


def read_frames(path: str | PathLike, num_frames: int) -> tuple[np.ndarray, list[int]]:
    """Read up to ``num_frames`` RGB frames evenly spaced over the video.

    Returns ``(frames, indices)`` with ``frames`` of shape ``(n, H, W, 3)``
    uint8. Frames are decoded sequentially (seeking is unreliable in many
    codecs), stopping early if the stream ends before the frame count says.
    Videos shorter than ``num_frames`` yield each frame once.
    """
    import cv2  # imported lazily

    capture = cv2.VideoCapture(str(path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise ValueError(f"Can't read frames from {path}")
        wanted = set(np.linspace(0, frame_count - 1, num_frames).astype(int).tolist())
        frames, indices = [], []
        for frame_idx in range(max(wanted) + 1):
            if not capture.grab():
                break
            if frame_idx in wanted:
                ok, frame = capture.retrieve()
                if not ok or frame is None:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                indices.append(frame_idx)
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return np.stack(frames), indices
