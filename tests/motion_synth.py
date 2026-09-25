"""Synthetic motion clips (for tests): each person has a personal motion style
(rhythms, amplitudes and couplings of head and expression features); each video
adds session effects (offsets, scale, noise), so identity has to come from behaviour."""

import numpy as np

from forensix.identity.motion.train import Clip

D = 64


def person_style(pid: int) -> dict:
    rng = np.random.default_rng(10_000 + pid)
    return {"freq": rng.uniform(0.2, 2.5, 6), "amp": rng.uniform(0.2, 1.5, 6),
            "mix": rng.standard_normal((6, D)) * rng.uniform(0.3, 1.0, (1, D))}


def clip(pid: int, video: int, seconds: float = 12.0, seed: int = 0) -> Clip:
    style = person_style(pid)
    rng = np.random.default_rng(pid * 1000 + video * 10 + seed)
    t = np.arange(int(seconds * 25)) / 25.0
    phases = rng.uniform(0, 2 * np.pi, 6)
    jitter = 1 + 0.05 * rng.standard_normal(6)  # small per-session tempo change
    sources = style["amp"] * np.sin(2 * np.pi * style["freq"] * jitter * t[:, None] + phases)
    x = sources @ style["mix"]
    session = rng.standard_normal(D) * 0.5 + x * rng.uniform(-0.2, 0.2)  # offset, scale
    x = x + session + 0.4 * rng.standard_normal(x.shape)
    return Clip(x.astype(np.float32), np.ones(len(t), bool), f"p{pid}", f"p{pid}/v{video}")
