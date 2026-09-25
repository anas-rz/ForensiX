"""Video deepfake detectors."""

from forensix.video import dfdc_selim, dfdc_wm, gend  # noqa: F401  (registers models)
from forensix.video.detector import VideoDetector

__all__ = ["VideoDetector"]
