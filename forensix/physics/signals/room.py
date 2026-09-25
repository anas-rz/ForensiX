"""Recording-environment measurements: noise floor in pauses, long-term
spectra (channel / microphone colouration) and blind reverberation time.

Blind RT60 follows the free-decay idea of Lollmann et al. (ITG 2010): at speech
offsets the level decays at the room's rate; decay slopes are fitted on the
energy envelope and the slow end of their distribution reflects the room.
"""

import numpy as np
from scipy.signal import welch

FRAME_S, HOP_S = 0.02, 0.005
BANDS_HZ = np.geomspace(100, 7000, 17)  # 16 log-spaced bands


def frame_levels_db(x: np.ndarray, sr: int, frame_s: float = FRAME_S, hop_s: float = HOP_S) -> np.ndarray:
    frame, hop = int(frame_s * sr), int(hop_s * sr)
    n = 1 + max(0, (len(x) - frame) // hop)
    idx = np.arange(frame)[None] + hop * np.arange(n)[:, None]
    return 10 * np.log10((x[idx] ** 2).mean(1) + 1e-20)


def speech_mask(levels_db: np.ndarray, hop_s: float = HOP_S, above_noise_db: float = 15.0,
                window_s: float = 6.0) -> np.ndarray:
    """Speech frames: at least ``above_noise_db`` over a rolling noise-floor
    estimate (2nd percentile over ``window_s``, several phrases long), so pauses
    in a noisier stretch are still recognized as pauses."""
    from scipy.ndimage import percentile_filter

    noise = percentile_filter(levels_db, 2, size=max(3, int(window_s / hop_s)), mode="nearest")
    return levels_db > np.maximum(noise + above_noise_db, levels_db.max() - 50.0)


def pause_regions(speech: np.ndarray, min_len: int, trim: int) -> list[tuple[int, int]]:
    """(start, end) frame ranges of non-speech lasting >= ``min_len`` frames, with
    ``trim`` frames removed at both ends (reverberant tails, onsets)."""
    out, i = [], 0
    while i < len(speech):
        if not speech[i]:
            j = i
            while j + 1 < len(speech) and not speech[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                a, b = i + trim, j - trim
                if b > a:
                    out.append((a, b))
            i = j + 1
        else:
            i += 1
    return out


def band_spectrum_db(x: np.ndarray, sr: int) -> np.ndarray:
    """Power per log-spaced band (dB) from a Welch spectrum."""
    freqs, psd = welch(x, fs=sr, nperseg=min(len(x), 1024))
    out = []
    for lo, hi in zip(BANDS_HZ[:-1], BANDS_HZ[1:]):
        sel = (freqs >= lo) & (freqs < hi)
        out.append(10 * np.log10(psd[sel].mean() + 1e-20) if sel.any() else np.nan)
    return np.array(out)


def shape_variation_db(spectra: list[np.ndarray]) -> float:
    """p90 over items of the RMS (over bands) deviation of each level-normalized
    spectrum from their median shape; NaN with fewer than 3 items."""
    if len(spectra) < 3:
        return float("nan")
    s = np.array(spectra)
    s = s - np.nanmean(s, axis=1, keepdims=True)  # remove overall level
    dev = np.sqrt(np.nanmean((s - np.nanmedian(s, axis=0)) ** 2, axis=1))
    return float(np.percentile(dev, 90))


def shape_change_db(spectra: list[np.ndarray], min_side: int = 3) -> float:
    """Largest RMS (over bands) difference between the median level-normalized
    spectrum before and after any point in time (>= ``min_side`` items per side);
    catches a switch between two regimes, which ``shape_variation_db`` halves.
    NaN with too few items."""
    if len(spectra) < 2 * min_side:
        return float("nan")
    s = np.array(spectra)
    s = s - np.nanmean(s, axis=1, keepdims=True)
    return float(max(np.sqrt(np.nanmean((np.nanmedian(s[:i], axis=0) - np.nanmedian(s[i:], axis=0)) ** 2))
                     for i in range(min_side, len(s) - min_side + 1)))


def decay_times_s(levels_db: np.ndarray, hop_s: float = HOP_S, noise_db: float | None = None,
                  min_drop_db: float = 10.0, min_len_s: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    """60 dB decay times implied by free-decay segments, and their start frames.
    A segment is a run of falling (smoothed) level that starts within 25 dB of
    the loudest frame and drops at least ``min_drop_db`` above the noise floor."""
    env = np.convolve(levels_db, np.ones(5) / 5, mode="same")
    floor = np.percentile(env, 5) if noise_db is None else noise_db
    top = env.max()
    times, starts, i = [], [], 1
    while i < len(env) - 1:
        if env[i] < env[i - 1] and env[i - 1] > top - 25:
            j = i
            while j + 1 < len(env) and env[j + 1] < env[j] and env[j + 1] > floor + 5:
                j += 1
            seg = env[i - 1 : j + 1]
            if len(seg) * hop_s >= min_len_s and seg[0] - seg[-1] >= min_drop_db:
                slope = np.polyfit(np.arange(len(seg)) * hop_s, seg, 1)[0]  # dB/s
                if slope < 0:
                    times.append(60.0 / -slope)
                    starts.append(i - 1)
            i = j + 1
        else:
            i += 1
    return np.array(times), np.array(starts, dtype=int)


def rt60_by_time(times: np.ndarray, max_groups: int = 4) -> np.ndarray:
    """RT60 estimates for consecutive time groups of free decays (the slow end,
    p75, of each group once it has >= 4 decays: fast ones are speech offsets)."""
    k = min(max_groups, len(times) // 2)
    if k < 3:
        return np.array([])
    return np.array([np.percentile(g, 75) if len(g) >= 4 else np.median(g) for g in np.array_split(times, k)])
