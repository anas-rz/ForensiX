from math import gcd
from os import PathLike

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

AudioInput = str | PathLike | np.ndarray


def to_mono(x: np.ndarray) -> np.ndarray:
    # soundfile returns (frames, channels) for multichannel audio
    return x.mean(axis=1) if x.ndim == 2 else x


def resample(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return x
    g = gcd(orig_sr, target_sr)
    return resample_poly(x, target_sr // g, orig_sr // g).astype(np.float32)


def load_audio(
    audio: AudioInput, target_sr: int, sample_rate: int | None = None
) -> np.ndarray:
    """Return ``audio`` as a mono float32 waveform at ``target_sr``.

    ``audio`` is a file path or a waveform array, in which case ``sample_rate``
    is required. Files libsndfile can't read (m4a/aac, video containers, ...)
    are decoded with ffmpeg if it's installed.
    """
    if isinstance(audio, np.ndarray):
        if sample_rate is None:
            raise ValueError("sample_rate is required when passing a waveform array")
        x, sr = audio, sample_rate
    else:
        try:
            x, sr = sf.read(audio, dtype="float32", always_2d=False)
        except sf.LibsndfileError:
            # formats libsndfile can't decode (m4a/aac, video containers, ...)
            return _ffmpeg_decode(audio, target_sr)
    x = to_mono(np.asarray(x, dtype=np.float32))
    if x.size == 0:
        raise ValueError("Audio is empty")
    return resample(x, sr, target_sr)


def _ffmpeg_decode(path: str | PathLike, target_sr: int) -> np.ndarray:
    """Decode any ffmpeg-readable file to mono float32 at ``target_sr``."""
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        raise ValueError(f"Can't decode {path}: unsupported by libsndfile and ffmpeg isn't installed")
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
           "-f", "f32le", "-ac", "1", "-ar", str(target_sr), "-"]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise ValueError(f"ffmpeg failed on {path}: {result.stderr.decode(errors='replace').strip()}")
    x = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if x.size == 0:
        raise ValueError(f"No audio decoded from {path}")
    return x
