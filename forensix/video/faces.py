"""Face detection and 5-point alignment shared by face-based video detectors."""

import numpy as np
from PIL import Image

# 5-point template (eyes, nose tip, mouth corners) as fractions of the crop,
# from GenD's detector.py (MIT, https://github.com/yermandy/GenD)
_TEMPLATE = np.array(
    [[0.34, 0.46], [0.66, 0.46], [0.5, 0.64], [0.37, 0.82], [0.63, 0.82]], dtype=np.float32
)


def align_face(img: np.ndarray, landmarks: np.ndarray, size: int = 256, scale: float = 1.3) -> np.ndarray:
    """Similarity-warp a face so its 5 landmarks match the template, with
    ``scale`` times the template's extent as context. Adapted from GenD's
    ``align_face`` (MIT); same arithmetic for a fixed output size."""
    import cv2

    dst = _TEMPLATE * size
    margin = size * (scale - 1) / 2.0
    dst = (dst + margin) * (size / (size + 2 * margin))
    matrix = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), dst, method=cv2.LMEDS)[0]
    return cv2.warpAffine(img, matrix, (size, size), flags=cv2.INTER_LINEAR)


def largest_face_landmarks(frame: np.ndarray, detector) -> np.ndarray | None:
    """5 landmarks (MTCNN order: eyes, nose, mouth corners) of the largest face."""
    boxes, _, landmarks = detector.detect(Image.fromarray(frame), landmarks=True)
    if boxes is None or len(boxes) == 0:
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return landmarks[int(np.argmax(areas))]
