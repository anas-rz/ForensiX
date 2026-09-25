"""Training examples made from real clips only: consistent positives and
physically inconsistent negatives, at the level of SyncNet window embeddings.

SyncNet window i covers frames i..i+4 on both streams, so shifting the audio by k
frames is exactly shifting the audio embedding sequence by k. A speed change
isn't a shift; its audio embeddings are computed at extraction
(``a_speed``). Every example comes with per-window labels (1 = this window's
audio and video don't belong together) for localization.

Manipulations (each mimics a way audio and face stop matching):
- ``shift``: constant offset of 6-15 frames (bad dubbing, careless re-muxing);
- ``drift``: audio played 3-8% fast or slow, offset growing over time;
- ``swap``: audio from another clip (voice replacement);
- ``face_swap``: face track from another clip over this audio;
- ``splice``: a 0.6-3 s audio segment replaced from another clip or from
  elsewhere in the same clip (edited speech, partial voice cloning);
- ``visual_splice``: the same for the face.

Clips are re-aligned to their own sync at extraction (``align_to_sync``), so
offsets are relative to true sync. Positives keep offsets of up to 3 frames:
real videos often carry that much AV lag from encoding, which is not
manipulation.
"""

from dataclasses import dataclass

import numpy as np

POSITIVE_MAX_SHIFT = 3
SHIFT_RANGE = (6, 15)
SPEEDS = (0.95, 1.05)  # a_speed variants cached at extraction
SPLICE_S = (0.6, 3.0)
FPS = 25


@dataclass
class Clip:
    v: np.ndarray  # (T, D) lip embeddings
    a: np.ndarray  # (T, D) audio embeddings
    a_speed: dict  # {speed: (T', D)} audio embeddings of the sped-up / slowed audio
    group: str = ""


@dataclass
class Example:
    v: np.ndarray
    a: np.ndarray
    window_labels: np.ndarray  # (T,) 0/1
    label: int
    kind: str


def _crop(x: np.ndarray, start: int, length: int) -> np.ndarray:
    return x[start : start + length]


def _shifted(clip: Clip, k: int, length: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Video windows t and audio windows t + k (k > 0: audio later), ``length`` long."""
    t = len(clip.v)
    lo, hi = max(0, -k), min(t, t - k)
    n = min(length, hi - lo)
    start = lo + rng.integers(0, max(1, hi - lo - n + 1))
    return clip.v[start : start + n], clip.a[start + k : start + k + n]


def positive(clip: Clip, length: int, rng) -> Example:
    k = int(rng.integers(-POSITIVE_MAX_SHIFT, POSITIVE_MAX_SHIFT + 1))
    v, a = _shifted(clip, k, length, rng)
    return Example(v, a, np.zeros(len(v)), 0, "real")


def negative(clip: Clip, donor: Clip, length: int, rng, kind: str | None = None) -> Example:
    kinds = ["shift", "drift", "swap", "face_swap", "splice", "visual_splice"]
    kind = kind or kinds[rng.integers(len(kinds))]
    if kind == "shift":
        k = int(rng.integers(SHIFT_RANGE[0], SHIFT_RANGE[1] + 1)) * (1 if rng.random() < 0.5 else -1)
        if len(clip.v) - abs(k) < 10:
            k = int(np.sign(k)) * SHIFT_RANGE[0]
        v, a = _shifted(clip, k, length, rng)
        return Example(v, a, np.ones(len(v)), 1, kind)
    if kind == "drift":
        speed = SPEEDS[rng.integers(len(SPEEDS))]
        a_sp = clip.a_speed[speed]
        n = min(length, len(clip.v), len(a_sp))
        # audio window t of the sped-up audio holds original time t * speed; misaligned
        # once the drift exceeds the positives' tolerance
        drift = np.abs(np.arange(n) * (speed - 1.0))
        labels = (drift > POSITIVE_MAX_SHIFT).astype(float)
        if labels.sum() == 0:
            return negative(clip, donor, length, rng, "shift")
        return Example(clip.v[:n], a_sp[:n], labels, 1, kind)
    if kind in ("swap", "face_swap"):
        n = min(length, len(clip.v), len(donor.v))
        s1 = rng.integers(0, len(clip.v) - n + 1)
        s2 = rng.integers(0, len(donor.v) - n + 1)
        if kind == "swap":
            v, a = clip.v[s1 : s1 + n], donor.a[s2 : s2 + n]
        else:
            v, a = donor.v[s2 : s2 + n], clip.a[s1 : s1 + n]
        return Example(v, a, np.ones(n), 1, kind)
    # splices
    n = min(length, len(clip.v))
    s = rng.integers(0, len(clip.v) - n + 1)
    v, a = clip.v[s : s + n].copy(), clip.a[s : s + n].copy()
    seg = int(rng.uniform(*SPLICE_S) * FPS)
    seg = min(seg, n - 2)
    if seg < 5:
        return negative(clip, donor, length, rng, "shift")
    start = int(rng.integers(0, n - seg + 1))
    source = donor if rng.random() < 0.5 or len(clip.v) < n + seg + 2 * SHIFT_RANGE[1] else clip
    if source is clip:  # elsewhere in the same clip, far enough not to be a small shift
        choices = [j for j in range(0, len(clip.v) - seg) if abs(j - (s + start)) > SHIFT_RANGE[1]]
        src = int(rng.choice(choices)) if choices else None
        if src is None:
            source, src = donor, None
    if source is donor:
        m = min(seg, len(donor.v))
        seg = m
        src = int(rng.integers(0, len(donor.v) - m + 1))
    stream = "a" if kind == "splice" else "v"
    target = a if stream == "a" else v
    target[start : start + seg] = getattr(source, stream)[src : src + seg]
    labels = np.zeros(n)
    # windows span 5 frames: those overlapping the splice's edges are mixed; count them
    labels[max(0, start - 4) : start + seg] = 1.0
    return Example(v, a, labels, 1, kind)


def sample(clips: list[Clip], length: int, rng, p_negative: float = 0.5) -> Example:
    i = int(rng.integers(len(clips)))
    clip = clips[i]
    if rng.random() >= p_negative:
        return positive(clip, length, rng)
    others = [j for j in range(len(clips)) if clips[j].group != clip.group] or [j for j in range(len(clips)) if j != i]
    donor = clips[int(rng.choice(others))] if others else clip
    return negative(clip, donor, length, rng)
