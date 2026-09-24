"""Audio deepfake / spoofed-speech detectors."""

from forensix.audio import aasist, rawgat_st  # noqa: F401  (registers models)
from forensix.audio.detector import AudioDetector

__all__ = ["AudioDetector"]
