"""id-voice: is the voice in this audio (or video) the enrolled person's?

Reference-based speaker verification. Enroll a person from recordings of
their voice, then check new audio, or a video's audio track, against them.

- Embedding: ECAPA-TDNN (Desplanques et al., 2020), SpeechBrain's
  spkrec-ecapa-voxceleb (Apache-2.0, trained on VoxCeleb 1+2), 192-d; the
  network and features are the ones vendored for ``avid`` and checked against
  SpeechBrain there.
- A clip of up to ``WHOLE_MAX_S`` is embedded whole, as SpeechBrain does.
  Longer audio is split into ``SEGMENT_S`` segments with a ``HOP_S`` hop; near
  silent segments are skipped. Per segment, the score is the cosine to the
  mean of the person's reference embeddings (the standard ECAPA scoring); the
  clip similarity is the median over segments, so the timeline shows where the
  voice stops being the person.
- ``fake_prob`` is ``1 - similarity``, flagged at ``1 - 0.25``: 0.25 is the
  cosine threshold of SpeechBrain's ``SpeakerRecognition.verify_batch``.

It checks *whose* voice it is: it rejects other speakers and voice conversion
or TTS that doesn't reach the person's voice. A good voice clone is built to
sound like the person and can pass, so pair it with a countermeasure (an audio
deepfake detector) that checks *how* the audio was made.
"""

from pathlib import Path

import numpy as np
import torch

from forensix.audiovisual.avid import ECAPA_WEIGHTS
from forensix.hub import torch_load
from forensix.identity.gallery import Gallery
from forensix.registry import ModelSpec, register
from forensix.types import Prediction

SAMPLE_RATE = 16000
WHOLE_MAX_S = 6.0  # embed shorter clips whole
SEGMENT_S, HOP_S = 3.0, 1.5
SILENCE_DB = -30.0  # segments this far below the loudest are skipped
COSINE_THRESHOLD = 0.25  # speechbrain SpeakerRecognition.verify_batch default
MODEL_TAG = "ecapa-voxceleb-0f99f2d"


def segments(wav: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
    """The whole clip if it's short, else non-silent ``SEGMENT_S`` segments, with start times (s)."""
    if len(wav) <= WHOLE_MAX_S * SAMPLE_RATE:
        return [wav], [0.0]
    seg, hop = int(SEGMENT_S * SAMPLE_RATE), int(HOP_S * SAMPLE_RATE)
    starts = list(range(0, len(wav) - seg + 1, hop))
    segs = [wav[s : s + seg] for s in starts]
    levels = np.array([10 * np.log10(np.mean(x**2) + 1e-12) for x in segs])
    keep = [i for i, lv in enumerate(levels) if lv > levels.max() + SILENCE_DB]
    return [segs[i] for i in keep], [round(starts[i] / SAMPLE_RATE, 2) for i in keep]


class VoiceVerifier:
    def __init__(self, ecapa_path: Path, device: str = "cpu", gallery: Gallery | None = None):
        from forensix.audiovisual.avid.ecapa import SpeakerEncoder

        self.name = "id-voice"
        self.threshold = 1.0 - COSINE_THRESHOLD
        self.device = torch.device(device)
        self.encoder = SpeakerEncoder()
        self.encoder.embedding_model.load_state_dict(torch_load(ecapa_path))
        self.encoder.to(self.device).eval()
        self.gallery = gallery or Gallery(MODEL_TAG)

    # -- embeddings ----------------------------------------------------------------

    def load(self, media) -> np.ndarray:
        """16 kHz mono waveform of an audio file, a video's audio track, or an array (16 kHz)."""
        from forensix.audio.io import load_audio

        if isinstance(media, np.ndarray):
            return load_audio(media, SAMPLE_RATE, SAMPLE_RATE)
        return load_audio(media, SAMPLE_RATE)

    @torch.no_grad()
    def embed(self, wavs: list[np.ndarray]) -> np.ndarray:
        """(N, 192) ECAPA embeddings, one per waveform (each embedded on its own, no padding)."""
        return np.stack([self.encoder(torch.from_numpy(np.ascontiguousarray(w, np.float32))[None].to(self.device))
                         .cpu().numpy()[0] for w in wavs])

    def embed_media(self, media) -> tuple[np.ndarray, list[float]]:
        segs, starts = segments(self.load(media))
        return self.embed(segs), starts

    # -- enrollment ----------------------------------------------------------------

    def enroll(self, name: str, media) -> dict:
        """Add recordings of ``name`` (audio or video paths, or 16 kHz arrays)."""
        media = [media] if isinstance(media, (str, Path, np.ndarray)) else list(media)
        embeddings, sources = [], []
        for m in media:
            e, _ = self.embed_media(m)
            embeddings.append(e)
            sources.extend([str(m) if not isinstance(m, np.ndarray) else "array"] * len(e))
        self.gallery.add(name, np.concatenate(embeddings), sources)
        return {"name": name, "n_embeddings": int(len(self.gallery.embeddings(name)))}

    def save_gallery(self, path) -> None:
        self.gallery.save(path)

    def load_gallery(self, path) -> None:
        self.gallery = Gallery.load(path, MODEL_TAG)

    def _centroid(self, name: str) -> np.ndarray:
        c = self.gallery.embeddings(name).mean(0)
        return c / np.linalg.norm(c)

    # -- verification ----------------------------------------------------------------

    def similarities(self, embeddings: np.ndarray, identity: str) -> np.ndarray:
        """Cosine of each embedding to the person's mean reference embedding."""
        e = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return e @ self._centroid(identity)

    def verify(self, media, identity: str) -> dict:
        emb, starts = self.embed_media(media)
        sim = self.similarities(emb, identity)
        others = {n: round(float(np.median(self.similarities(emb, n))), 4) for n in self.gallery.names() if n != identity}
        return {
            "identity": identity,
            "similarity": round(float(np.median(sim)), 4),
            "threshold_similarity": COSINE_THRESHOLD,
            "segments_below_threshold": round(float(np.mean(sim < COSINE_THRESHOLD)), 4),
            "segment_similarity": np.round(sim, 4).tolist(),
            "segment_start_s": starts,
            "segment_s": None if len(starts) == 1 and starts[0] == 0.0 else SEGMENT_S,
            "other_enrolled_similarity": others,  # e.g. the voice sounds more like someone else enrolled
        }

    def predict(self, media, identity: str) -> Prediction:
        r = self.verify(media, identity)
        return Prediction(model=self.name, fake_prob=round(1.0 - r["similarity"], 4), threshold=self.threshold,
                          score_type="anomaly", segment_fake_probs=[round(1 - s, 4) for s in r["segment_similarity"]],
                          details=r)

    def identify(self, media) -> list[tuple[str, float]]:
        """Enrolled people ranked by similarity to the voice in ``media``."""
        emb, _ = self.embed_media(media)
        scores = {n: float(np.median(self.similarities(emb, n))) for n in self.gallery.names()}
        return sorted(scores.items(), key=lambda kv: -kv[1])


def build(weights_path: Path, device: str = "cpu") -> VoiceVerifier:
    return VoiceVerifier(weights_path, device=device)


register(
    ModelSpec(
        name="id-voice",
        family="identity",
        builder=build,
        weights=ECAPA_WEIGHTS,
        license="Apache-2.0 (SpeechBrain spkrec-ecapa-voxceleb). Trained on VoxCeleb 1+2 (CC BY 4.0; "
        "the videos' copyright stays with their owners)",
        source="https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/tree/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
        paper="https://arxiv.org/abs/2005.07143",
        description="Reference-based speaker verification: enroll a person from recordings of their voice, then "
        "check whether the voice in new audio or video is theirs (ECAPA-TDNN embeddings, cosine to the "
        "enrolled mean, SpeechBrain's threshold). Per-segment timeline; pair with an audio deepfake "
        "detector, since a good voice clone can pass.",
    )
)
