"""Synthetic physdf graph sequences with known physical couplings (for tests)."""

import numpy as np

from forensix.physics.graph.features import GraphSequence
from forensix.physics.graph.schema import NODE_NAMES, NODES, PULSE_NODES, STEP_S


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    k = np.hanning(width)
    return np.convolve(x, k / k.sum(), mode="same")


def sequence(seconds: float = 8.0, seed: int = 0, audio_seed: int | None = None, rogue_pulse: str | None = None) -> GraphSequence:
    """Speech envelope drives jaw, lips, tract and breath; one heart drives every
    pulse region; the head moves smoothly. ``audio_seed`` takes the audio from
    another utterance (a lip-sync mismatch); ``rogue_pulse`` gives one region its
    own heart."""
    rng = np.random.default_rng(seed)
    t = int(seconds / STEP_S)
    time = np.arange(t) * STEP_S

    def envelope(r):
        syll = (r.random(t) < 0.18).astype(float)
        env = _smooth(syll, 7) * (_smooth(r.random(t), 60) > 0.45)
        return env / (env.max() + 1e-9)

    env_v = envelope(rng)
    env_a = envelope(np.random.default_rng(audio_seed)) if audio_seed is not None else env_v
    arng = np.random.default_rng(1000 + (audio_seed if audio_seed is not None else seed))
    x = {}
    head = np.cumsum(rng.normal(0, 0.05, (t, 6)), axis=0)
    x["head"] = np.stack([_smooth(head[:, j], 25) for j in range(6)], axis=1)
    x["skull"] = np.abs(rng.normal(0.004, 0.001, (t, 1)))
    x["jaw"] = np.stack([env_v, 0.1 * env_v, 0 * env_v, 0.02 * env_v], 1) + rng.normal(0, 0.02, (t, 4))
    x["lips"] = np.stack([0.3 * env_v, 0.05 * env_v, 0 * env_v, 0 * env_v, 1 - env_v, 0 * env_v], 1) + rng.normal(0, 0.02, (t, 6))
    x["upper_face"] = rng.normal(0, 0.02, (t, 4))
    hr = rng.uniform(1.0, 1.6)
    phase = rng.uniform(0, 2 * np.pi)
    for p in PULSE_NODES:
        f = rng.uniform(1.0, 1.6) if p == rogue_pulse else hr
        ph = rng.uniform(0, 2 * np.pi) if p == rogue_pulse else phase
        pulse = np.sin(2 * np.pi * f * time + ph) + rng.normal(0, 0.3, t)
        x[p] = np.stack([pulse, np.gradient(pulse)], 1)
    x["light"] = np.tile([4.0, 0.3, -0.2, -0.9, 0.12, 4.5], (t, 1)) + rng.normal(0, 0.01, (t, 6))
    voiced = (env_a > 0.2).astype(float)
    x["glottis"] = np.stack([voiced * _smooth(arng.normal(0, 0.1, t), 15), voiced, 1 - 0.8 * voiced], 1)
    x["tract"] = np.concatenate([np.outer(env_a, [0.2, 0.1, 0.05]),
                                 np.outer(env_a, [5, 3, 0, -2, -3, -3])], 1) + arng.normal(0, 0.3, (t, 9))
    level = -40 + 35 * env_a + arng.normal(0, 1, t)
    x["breath"] = np.stack([level / 10, (level > -30).astype(float), np.zeros(t)], 1)
    feats = {n: x[n].astype(np.float32) for n in NODE_NAMES}
    assert all(feats[n].shape[1] == len(NODES[n][1]) for n in NODE_NAMES)
    return GraphSequence(feats, {n: np.ones(t, bool) for n in NODE_NAMES})
