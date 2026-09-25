"""Speech-production measurements, implemented from the papers (no GPL code):

- F0: YIN, de Cheveigne & Kawahara, JASA 2002.
- jitter / shimmer: local cycle-to-cycle period / peak-amplitude perturbation
  on pitch-synchronous marks (definitions as in Praat's voice report).
- HNR: autocorrelation method, Boersma, IFA Proceedings 1993.
- formants: LPC (autocorrelation method) root-finding.
- vocal-tract length: formant dispersion, Fitch, JASA 1997: VTL = c / (2 * Df),
  Df = (F4 - F1) / 3 for a uniform tube closed at the glottis.
"""

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import butter, filtfilt, resample_poly

SPEED_OF_SOUND_CM_S = 35000.0  # warm, humid air in the vocal tract
HOP_S = 0.01


def _frames(x: np.ndarray, length: int, hop: int) -> np.ndarray:
    n = 1 + max(0, (len(x) - length) // hop)
    idx = np.arange(length)[None] + hop * np.arange(n)[:, None]
    return x[idx]


def yin_f0(x: np.ndarray, sr: int, fmin: float = 50.0, fmax: float = 600.0,
           threshold: float = 0.15, voicing: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """F0 per 10 ms hop (Hz, NaN when unvoiced) and the aperiodicity d'(T0)."""
    hop, max_lag = int(sr * HOP_S), int(sr / fmin)
    min_lag = int(sr / fmax)
    frames = _frames(np.pad(x, (0, 2 * max_lag)), 2 * max_lag, hop)[: 1 + len(x) // hop]
    w = max_lag
    # difference function via FFT: d(t) = r(0) + r_t(0) - 2 r(t)
    spec = np.fft.rfft(frames, n=4 * max_lag)
    acf = np.fft.irfft(spec * np.conj(np.fft.rfft(frames[:, :w], n=4 * max_lag)), n=4 * max_lag)[:, : max_lag + 1]
    energy = np.cumsum(frames ** 2, axis=1)
    e0 = energy[:, w - 1]
    et = energy[:, w - 1 : w + max_lag] - np.concatenate([np.zeros((len(frames), 1)), energy[:, :max_lag]], axis=1)
    d = e0[:, None] + et - 2 * acf
    cmnd = np.ones_like(d)
    cum = np.cumsum(d[:, 1:], axis=1)
    cmnd[:, 1:] = d[:, 1:] * np.arange(1, max_lag + 1) / np.maximum(cum, 1e-12)
    f0 = np.full(len(frames), np.nan)
    aper = np.ones(len(frames))
    silent = e0 < 1e-6 * max(e0.max(), 1e-12)
    for i in np.flatnonzero(~silent):
        c = cmnd[i]
        below = np.flatnonzero(c[min_lag:max_lag] < threshold)
        tau = min_lag + (below[0] if len(below) else np.argmin(c[min_lag:max_lag]))
        while tau + 1 < max_lag and c[tau + 1] < c[tau]:
            tau += 1
        aper[i] = c[tau]
        if c[tau] < voicing:
            a, b, g = c[tau - 1], c[tau], c[tau + 1]  # parabolic refinement
            shift = 0.5 * (a - g) / (a - 2 * b + g) if (a - 2 * b + g) != 0 else 0.0
            f0[i] = sr / (tau + shift)
    return f0, aper


def pitch_marks(x: np.ndarray, sr: int, f0: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per voiced run: (glottal-cycle peak times in samples, peak amplitudes),
    found by stepping one local period at a time on a low-passed signal."""
    b, a = butter(4, 900 / (sr / 2))
    y = filtfilt(b, a, x)
    hop = int(sr * HOP_S)
    voiced = ~np.isnan(f0)
    runs, i = [], 0
    while i < len(f0):
        if voiced[i]:
            j = i
            while j + 1 < len(f0) and voiced[j + 1]:
                j += 1
            if j - i >= 5:  # at least 60 ms
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    out = []
    for start, end in runs:
        s, e = start * hop, min(len(y), (end + 1) * hop)
        period = sr / f0[start]
        t = s + int(np.argmax(y[s : s + int(period)]))
        marks = [t]
        while True:
            period = sr / np.interp(t / hop, np.arange(len(f0))[voiced], f0[voiced])
            lo, hi = int(t + 0.8 * period), int(t + 1.2 * period)
            if hi >= e:
                break
            t = lo + int(np.argmax(y[lo:hi]))
            marks.append(t)
        marks = np.array(marks)
        # parabolic sub-sample refinement of each peak (removes 1-sample quantization jitter)
        inner = (marks > 0) & (marks < len(y) - 1)
        m = marks[inner]
        a, b, c = y[m - 1], y[m], y[m + 1]
        denom = a - 2 * b + c
        shift = np.where(denom != 0, 0.5 * (a - c) / np.where(denom != 0, denom, 1), 0.0)
        refined = marks.astype(np.float64)
        refined[inner] += np.clip(shift, -0.5, 0.5)
        out.append((refined, np.abs(y[marks])))
    return out


def jitter_shimmer(marks: list[tuple[np.ndarray, np.ndarray]], sr: int) -> tuple[float, float]:
    """Local jitter and shimmer (%), over consecutive-period pairs with plausible
    periods (1.67-20 ms) and ratios <= 1.3."""
    dp, pp, da, aa = [], [], [], []
    for times, amps in marks:
        periods = np.diff(times) / sr
        for k in range(1, len(periods)):
            p0, p1 = periods[k - 1], periods[k]
            if 1 / 600 <= p0 <= 1 / 50 and 1 / 600 <= p1 <= 1 / 50 and max(p0, p1) / min(p0, p1) <= 1.3:
                dp.append(abs(p1 - p0)); pp.append(p0)
                da.append(abs(amps[k + 1] - amps[k])); aa.append(amps[k])
    if len(dp) < 20:
        return float("nan"), float("nan")
    return 100 * float(np.mean(dp) / np.mean(pp)), 100 * float(np.mean(da) / np.mean(aa))


def hnr_db(x: np.ndarray, sr: int, f0: np.ndarray) -> float:
    """Median harmonics-to-noise ratio over voiced frames (normalized
    autocorrelation r at the pitch period: HNR = 10 log10(r / (1 - r)))."""
    hop, win = int(sr * HOP_S), int(0.04 * sr)
    values = []
    for i in np.flatnonzero(~np.isnan(f0)):
        lag = int(round(sr / f0[i]))
        s = i * hop
        seg = x[s : s + win + lag]
        if len(seg) < win + lag:
            continue
        a, b = seg[:win], seg[lag : lag + win]
        r = float(a @ b / np.sqrt((a @ a) * (b @ b) + 1e-20))
        if 0 < r < 1:
            values.append(10 * np.log10(r / (1 - r)))
    return float(np.median(values)) if values else float("nan")


def lpc_formants(x: np.ndarray, sr: int, f0: np.ndarray, n_formants: int = 4) -> np.ndarray:
    """F1..F4 (Hz, NaN where missing) for voiced 10 ms hops, from 12th-order LPC
    on 25 ms Hamming frames of the signal resampled to 10 kHz."""
    fs = 10000
    y = resample_poly(x, fs, sr)
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])  # pre-emphasis
    hop, win, order = int(fs * HOP_S), int(0.025 * fs), 12
    window = np.hamming(win)
    out = np.full((len(f0), n_formants), np.nan)
    for i in np.flatnonzero(~np.isnan(f0)):
        seg = y[i * hop : i * hop + win]
        if len(seg) < win:
            continue
        seg = seg * window
        r = np.correlate(seg, seg, "full")[win - 1 : win + order]
        if r[0] <= 0:
            continue
        try:
            a = solve_toeplitz(r[:order], -r[1 : order + 1])
        except np.linalg.LinAlgError:
            continue
        roots = np.roots(np.concatenate([[1.0], a]))
        roots = roots[np.imag(roots) > 0]
        freqs = np.angle(roots) * fs / (2 * np.pi)
        bws = -np.log(np.abs(roots)) * fs / np.pi
        ok = (freqs > 90) & (freqs < 4900) & (bws < 500)
        found = np.sort(freqs[ok])[:n_formants]
        out[i, : len(found)] = found
    return out


def vocal_tract_length_cm(formants: np.ndarray) -> np.ndarray:
    """Per-frame VTL from formant dispersion (NaN where F1 or F4 is missing)."""
    dispersion = (formants[:, 3] - formants[:, 0]) / 3
    with np.errstate(invalid="ignore", divide="ignore"):
        return SPEED_OF_SOUND_CM_S / (2 * dispersion)


def speech_runs_s(x: np.ndarray, sr: int, floor_db: float = -35.0, min_pause_s: float = 0.25) -> list[float]:
    """Durations (s) of speech stretches between pauses of at least ``min_pause_s``
    (energy VAD: 20 ms frames above ``floor_db`` relative to the loudest)."""
    hop = int(0.02 * sr)
    frames = _frames(x, hop, hop)
    level = 10 * np.log10((frames ** 2).mean(1) + 1e-12)
    speech = level > level.max() + floor_db
    min_pause = int(round(min_pause_s / 0.02))
    runs, current, gap = [], 0, 0
    for s in speech:
        if s:
            current += 1 + (gap if 0 < gap < min_pause else 0)
            gap = 0
        else:
            gap += 1
            if gap >= min_pause and current:
                runs.append(current * 0.02); current = 0
    if current:
        runs.append(current * 0.02)
    return runs
