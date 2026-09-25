"""Source-filter speech synthesis with known physical ground truth (for tests)."""

import numpy as np
from scipy.signal import lfilter, resample_poly

SR = 16000
C = 35000.0  # cm/s


def rosenberg(n_open: int, n_close: int) -> np.ndarray:
    t1 = np.arange(n_open) / n_open
    t2 = np.arange(n_close) / n_close
    return np.concatenate([0.5 * (1 - np.cos(np.pi * t1)), np.cos(0.5 * np.pi * t2)])


def glottal_source(f0_contour, jitter=0.008, shimmer=0.03, seed=0, oversample=4):
    """Glottal flow derivative for a per-sample F0 contour (Hz), with
    cycle-to-cycle period (jitter) and amplitude (shimmer) perturbation. Built at
    ``oversample`` x the sample rate so periods aren't quantized to whole samples."""
    rng = np.random.default_rng(seed)
    sr = SR * oversample
    contour = np.repeat(f0_contour, oversample)
    out = np.zeros(len(contour))
    t = 0
    while t < len(out) - 1:
        n = max(4, int(round(sr / contour[t] * (1 + jitter * rng.standard_normal()))))
        pulse = rosenberg(int(0.4 * n), int(0.16 * n))
        seg = np.diff(np.concatenate([[0], pulse]))[: len(out) - t]
        out[t : t + len(seg)] += (1 + shimmer * rng.standard_normal()) * seg
        t += n
    return resample_poly(out, 1, oversample)


def tube_formants(length_cm, max_hz=4900.0):
    """All resonances of a uniform tube closed at the glottis below ``max_hz``."""
    out, k = [], 1
    while (f := (2 * k - 1) * C / (4 * length_cm)) < max_hz:
        out.append(f)
        k += 1
    return out


def vocal_tract(x, formants):
    for f in formants:
        bw = 50 + 0.05 * f  # bandwidths widen with frequency
        r = np.exp(-np.pi * bw / SR)
        x = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * f / SR), r * r], x)
    return x


def utterance(seconds=8.0, phrase_s=2.5, pause_s=0.5, f0=120.0, length_cm=17.0, jitter=0.008,
              shimmer=0.03, aspiration=0.02, f0_jumps=False, seed=0):
    """Phrases of voiced speech separated by pauses, with a gently moving F0."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    contour = f0 * 2 ** (1.5 * np.sin(2 * np.pi * 0.4 * t) / 12)  # +-1.5 semitones intonation
    if f0_jumps:  # instantaneous octave jumps every 0.2 s: physically impossible
        contour = contour * np.where((t // 0.2) % 2, 2.0, 1.0)
    src = glottal_source(contour, jitter, shimmer, seed)
    src += aspiration * np.abs(src).max() * rng.standard_normal(n)
    y = vocal_tract(src, tube_formants(length_cm))
    envelope = np.ones(n) if pause_s == 0 else ((t % (phrase_s + pause_s)) < phrase_s).astype(float)
    y = 0.3 * y / np.abs(y).max()  # normalize before the recording noise floor
    return (y * envelope + 3e-4 * rng.standard_normal(n)).astype(np.float32)
