"""Image deepfake / AI-generated image detectors."""

from forensix.image import clipdet, commfor, rine, safe, univfd  # noqa: F401  (registers models)
from forensix.image.detector import ImageDetector

__all__ = ["ImageDetector"]
