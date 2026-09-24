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

    ``audio`` is a file path (any format libsndfile reads: wav, flac, ogg, mp3)
    or a waveform array, in which case ``sample_rate`` is required.
    """
    if isinstance(audio, np.ndarray):
        if sample_rate is None:
            raise ValueError("sample_rate is required when passing a waveform array")
        x, sr = audio, sample_rate
    else:
        x, sr = sf.read(audio, dtype="float32", always_2d=False)
    x = to_mono(np.asarray(x, dtype=np.float32))
    if x.size == 0:
        raise ValueError("Audio is empty")
    return resample(x, sr, target_sr)
