"""id-motion: does this person move like the enrolled person? (ForensiX)

A face swap gives the actor the person's face but keeps the actor's head
motion and expressions; a reenactment drives the person's face with someone
else's. How someone nods, tilts, blinks, raises their brows and moves their
mouth while talking is personal (behavioural biometrics). The method follows ID-Reveal
(Cozzolino et al., ICCV 2021), implemented independently here:
- per frame: head pose and 52 expression blendshapes from MediaPipe
  FaceLandmarker (``features.py``);
- a temporal ID network embeds every 4 s window (``model.py``), trained on real
  videos only so that windows of the same person, from different videos, lie
  together (``train.py``);
- a new video is compared window by window with the person's enrolled windows.

Enroll from videos (motion needs time; photos have none). ``fake_prob`` is
1 - similarity (an anomaly score), flagged at the threshold that gave equal
error rates on held-out people during training. Weights: train them with
``scripts/train_id_motion.py`` and ``forensix.load("id-motion", weights_path=...)``.
"""

import hashlib
from pathlib import Path

import numpy as np

from forensix.identity.gallery import Gallery
from forensix.physics.face.tracker import FACE_LANDMARKER
from forensix.registry import ModelSpec, register
from forensix.types import Prediction

MAX_SECONDS = 60.0


class MotionVerifier:
    def __init__(self, encoder, threshold: float, tag: str, device: str = "cpu"):
        self.name = "id-motion"
        self.encoder, self.device = encoder.to(device).eval(), device
        self.threshold = threshold  # on 1 - similarity
        self.tag = tag
        self.gallery = Gallery(tag)

    def features(self, video) -> tuple[np.ndarray, np.ndarray]:
        from forensix.identity.motion.features import motion_features
        from forensix.physics.face.tracker import track_face
        from forensix.physics.io import read_video

        frames, fps = read_video(video, max_seconds=MAX_SECONDS)
        return motion_features(track_face(frames, fps))

    def window_embeddings(self, video) -> tuple[np.ndarray, list[float]]:
        from forensix.identity.motion.train import embed

        feats, valid = self.features(video)
        emb, starts = embed(self.encoder, feats, valid, device=self.device)
        return emb, [round(s / 25.0, 2) for s in starts]

    def enroll(self, name: str, videos) -> dict:
        from forensix.identity.media import is_image

        videos = [videos] if isinstance(videos, (str, Path)) else list(videos)
        embs, sources = [], []
        for v in videos:
            if is_image(v):
                raise ValueError(f"{v} is a photo: id-motion enrolls from videos (motion needs time)")
            e, _ = self.window_embeddings(v)
            embs.append(e); sources.extend([str(v)] * len(e))
        embs = [e for e in embs if len(e)]
        if not embs:
            raise ValueError(f"No 4 s window with a tracked face in the videos for {name!r}")
        self.gallery.add(name, np.concatenate(embs), sources)
        return {"name": name, "n_windows": int(len(self.gallery.embeddings(name)))}

    def save_gallery(self, path) -> None:
        self.gallery.save(path)

    def load_gallery(self, path) -> None:
        self.gallery = Gallery.load(path, self.tag)

    def verify(self, video, identity: str) -> dict:
        from forensix.identity.motion.train import clip_similarity

        refs = self.gallery.embeddings(identity)
        emb, times = self.window_embeddings(video)
        if not len(emb):
            raise ValueError("No 4 s window with a tracked face")
        sim, per_window = clip_similarity(emb, refs)
        others = {n: round(clip_similarity(emb, self.gallery.embeddings(n))[0], 4)
                  for n in self.gallery.names() if n != identity}
        return {"identity": identity, "similarity": round(sim, 4), "window_similarity": np.round(per_window, 4).tolist(),
                "window_start_s": times, "window_s": 4.0, "other_enrolled_similarity": others}

    def predict(self, video, identity: str) -> Prediction:
        r = self.verify(video, identity)
        return Prediction(model=self.name, fake_prob=round(1.0 - r["similarity"], 4), threshold=self.threshold,
                          score_type="anomaly", segment_fake_probs=[round(1 - s, 4) for s in r["window_similarity"]],
                          details=r)


def build(weights_path: Path | None, device: str = "cpu") -> MotionVerifier:
    from forensix.identity.motion.train import load

    if weights_path is None:
        raise ValueError("id-motion has no published weights: train them with scripts/train_id_motion.py and "
                         "pass forensix.load('id-motion', weights_path=...)")
    enc, threshold, _ = load(weights_path, device)
    tag = "id-motion:" + hashlib.sha256(Path(weights_path).read_bytes()).hexdigest()[:12]
    return MotionVerifier(enc, threshold, tag, device)


register(
    ModelSpec(
        name="id-motion",
        family="identity",
        builder=build,
        weights=None,  # trained by you on real videos (scripts/train_id_motion.py)
        extra_weights=(FACE_LANDMARKER,),
        license="ForensiX (model and training code; trained by you on real videos). Face tracking: MediaPipe "
        "FaceLandmarker (Apache-2.0)",
        source="",
        paper="https://arxiv.org/abs/2012.02512",
        description="Behavioural identity: whether a person moves like the enrolled person (head motion and "
        "expressions over 4 s windows, embedded by a temporal network trained on real videos; the "
        "ID-Reveal idea). Catches face swaps and reenactments that carry someone else's motion. "
        "Enroll from videos.",
    )
)
