"""Remote photoplethysmography: the pulse signal in skin colour.

Implemented from the papers (no third-party rPPG code):
- POS: Wang, den Brinker, Stuijk & de Haan, "Algorithmic Principles of Remote
  PPG", IEEE TBME 2017 (sliding-window plane-orthogonal-to-skin projection).
- CHROM: de Haan & Jeanne, "Robust Pulse Rate From Chrominance-Based rPPG",
  IEEE TBME 2013.
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch

HR_BAND_HZ = (0.7, 3.0)  # 42-180 bpm


def bandpass(x: np.ndarray, fs: float, band=HR_BAND_HZ, order: int = 3) -> np.ndarray:
    b, a = butter(order, [band[0] / (fs / 2), band[1] / (fs / 2)], btype="band")
    return filtfilt(b, a, x, padlen=min(len(x) - 1, 3 * max(len(a), len(b))))


def pos(rgb: np.ndarray, fs: float, window_s: float = 1.6) -> np.ndarray:
    """POS pulse signal from mean skin RGB traces (T, 3)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    n = len(rgb)
    win = max(2, int(round(window_s * fs)))
    projection = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    h = np.zeros(n)
    for start in range(0, n - win + 1):
        c = rgb[start : start + win]
        cn = c / c.mean(axis=0)  # temporal normalization
        s = cn @ projection.T  # (win, 2)
        s1, s2 = s[:, 0], s[:, 1]
        std2 = s2.std()
        p = s1 + (s1.std() / std2 if std2 > 0 else 0.0) * s2
        h[start : start + win] += p - p.mean()  # overlap-add
    return h


def chrom(rgb: np.ndarray, fs: float) -> np.ndarray:
    """CHROM pulse signal from mean skin RGB traces (T, 3)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    cn = rgb / rgb.mean(axis=0)
    x = 3 * cn[:, 0] - 2 * cn[:, 1]
    y = 1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2]
    xf, yf = bandpass(x, fs), bandpass(y, fs)
    alpha = xf.std() / yf.std() if yf.std() > 0 else 0.0
    return xf - alpha * yf


def heart_rate(pulse: np.ndarray, fs: float) -> tuple[float, float]:
    """(bpm, SNR in dB) from the spectral peak in the heart-rate band. SNR is the
    power within 0.1 Hz of the peak and its first harmonic over the rest of the band."""
    x = bandpass(pulse - np.mean(pulse), fs)
    freqs, psd = welch(x, fs=fs, nperseg=min(len(x), int(fs * 8)), nfft=max(2048, len(x)))
    band = (freqs >= HR_BAND_HZ[0]) & (freqs <= HR_BAND_HZ[1])
    f_band, p_band = freqs[band], psd[band]
    f0 = f_band[np.argmax(p_band)]
    near = (np.abs(f_band - f0) <= 0.1) | (np.abs(f_band - 2 * f0) <= 0.1)
    signal, noise = p_band[near].sum(), p_band[~near].sum()
    snr = 10 * np.log10(signal / noise) if noise > 0 else np.inf
    return float(f0 * 60), float(snr)


def coherence(pulses: dict[str, np.ndarray], fs: float) -> tuple[np.ndarray, list[str]]:
    """Pairwise correlation of band-passed pulse signals between regions."""
    names = list(pulses)
    x = np.stack([bandpass(pulses[n] - pulses[n].mean(), fs) for n in names])
    return np.corrcoef(x), names
