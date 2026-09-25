"""Synthetic identities for avid tests: a person's voice and face embeddings are
two noisy projections of one hidden identity, so voices predict faces only partly."""

import numpy as np

from forensix.audiovisual.avid.train import Clip

VOICE_D, FACE_D = 192, 384


def clips(n_speakers: int, per_speaker: int = 3, segments: int = 6, seed: int = 0, offset: int = 0) -> list[Clip]:
    proj = np.random.default_rng(777)  # the "encoders": shared by everyone
    wv, wf = proj.standard_normal((16, VOICE_D)), proj.standard_normal((16, FACE_D))
    out = []
    for s in range(n_speakers):
        rng = np.random.default_rng(seed * 100000 + offset + s)
        z = rng.standard_normal(16)
        for _ in range(per_speaker):
            voice = z @ wv + 0.8 * rng.standard_normal((segments, VOICE_D)) * 20
            face = z @ wf + 0.8 * rng.standard_normal((segments, FACE_D)) * 20
            out.append(Clip(voice.astype(np.float32), face.astype(np.float32), f"spk{offset + s}"))
    return out
