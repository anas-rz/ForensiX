"""Decode a video's frames at a fixed frame rate and its audio as 16 kHz mono."""

import json
import shutil
import subprocess
from os import PathLike

import numpy as np


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("Audio-visual detectors need ffmpeg on PATH")
    return exe


def _size(path: str | PathLike) -> tuple[int, int]:
    probe = shutil.which("ffprobe")
    if probe is None:
        raise RuntimeError("Audio-visual detectors need ffprobe on PATH")
    out = subprocess.run(
        [probe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "json", str(path)], capture_output=True, check=True)
    stream = json.loads(out.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def read_av(path: str | PathLike, fps: int = 25, sample_rate: int = 16000) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(frames, audio)``: RGB uint8 frames ``(T, H, W, 3)`` resampled to
    ``fps``, and int16-valued mono audio (float32) at ``sample_rate``."""
    width, height = _size(path)
    video = subprocess.run(
        [_ffmpeg(), "-nostdin", "-v", "error", "-i", str(path), "-an", "-r", str(fps),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    frames = np.frombuffer(video, np.uint8).reshape(-1, height, width, 3)
    audio = subprocess.run(
        [_ffmpeg(), "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
         "-ar", str(sample_rate), "-f", "s16le", "-acodec", "pcm_s16le", "-"],
        capture_output=True, check=True).stdout
    if not frames.size or not audio:
        raise ValueError(f"{path} needs both a video and an audio stream")
    return frames, np.frombuffer(audio, np.int16).astype(np.float32)
