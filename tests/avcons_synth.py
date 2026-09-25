"""Synthetic SyncNet-like embedding clips (for tests): lips and audio are two
noisy projections of one smooth speech process, so they match only in sync."""

import numpy as np

from forensix.audiovisual.avcons.negatives import Clip

D = 64


def _process(n: int, rng) -> np.ndarray:
    z = rng.standard_normal((n + 40, 8))
    k = np.hanning(9)
    z = np.stack([np.convolve(z[:, j], k / k.sum(), mode="same") for j in range(8)], 1)
    return z[20 : 20 + n]


def clip(seed: int, n: int = 250, dim: int = D) -> Clip:
    rng = np.random.default_rng(seed)
    proj = np.random.default_rng(12345)  # shared "network": the same for every clip
    wv, wa = proj.standard_normal((8, dim)), proj.standard_normal((8, dim))
    z = _process(int(n * 1.1) + 10, rng)
    v = z[:n] @ wv + 0.3 * rng.standard_normal((n, dim))
    a = z[:n] @ wa + 0.3 * rng.standard_normal((n, dim))
    speeds = {}
    for s in (0.95, 1.05):  # audio played s times as fast: window t holds process time t * s
        t = np.arange(n) * s
        zs = np.stack([np.interp(t, np.arange(len(z)), z[:, j]) for j in range(8)], 1)
        speeds[s] = zs @ wa + 0.3 * rng.standard_normal((n, dim))
    return Clip(v.astype(np.float32), a.astype(np.float32), {k: x.astype(np.float32) for k, x in speeds.items()},
                group=f"spk{seed}")
