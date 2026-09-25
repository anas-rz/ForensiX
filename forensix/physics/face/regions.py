"""Skin regions on the MediaPipe 478-point face mesh, as landmark polygons."""

import numpy as np

REGIONS: dict[str, list[int]] = {
    "forehead": [10, 338, 297, 332, 333, 334, 296, 336, 9, 107, 66, 105, 104, 103, 67, 109],
    "left_cheek": [116, 117, 118, 119, 100, 126, 142, 36, 205, 187, 123],  # subject's right
    "right_cheek": [345, 346, 347, 348, 329, 355, 371, 266, 425, 411, 352],  # subject's left
    "nose": [168, 193, 245, 188, 174, 236, 198, 209, 49, 64, 98, 97, 2, 326, 327, 294, 279, 429, 420, 456, 399, 412, 465, 417],
    "chin": [194, 201, 200, 421, 418, 428, 396, 175, 171, 208],  # below the lower lip
}


def region_masks(landmarks_xy: np.ndarray, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Boolean pixel masks for each region, from (478, 2) pixel landmarks."""
    import cv2

    masks = {}
    for name, idx in REGIONS.items():
        mask = np.zeros(shape, np.uint8)
        cv2.fillPoly(mask, [np.round(landmarks_xy[idx]).astype(np.int32)], 1)
        masks[name] = mask.astype(bool)
    return masks
