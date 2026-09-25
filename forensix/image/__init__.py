"""Image deepfake / AI-generated image detectors."""

from forensix.image import rine, univfd  # noqa: F401  (registers models)
from forensix.image.detector import ImageDetector

__all__ = ["ImageDetector"]
