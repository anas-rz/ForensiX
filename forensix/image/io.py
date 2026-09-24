from os import PathLike

import numpy as np
from PIL import Image

ImageInput = str | PathLike | Image.Image | np.ndarray


def load_image(image: ImageInput) -> Image.Image:
    """Return ``image`` as an RGB PIL image.

    ``image`` is a file path, a PIL image, or an ``(H, W, 3)`` uint8 RGB array.
    Pixels are not resized or re-encoded: detectors rely on generator traces
    that resampling and compression destroy.
    """
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Array images must be (H, W, 3) uint8 RGB")
        return Image.fromarray(image)
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    with Image.open(image) as img:
        return img.convert("RGB")
