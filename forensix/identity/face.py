"""id-face: is the face in this video the enrolled person?

Reference-based face verification. Enroll a person from photos or videos, then
check new media against them, frame by frame.

- Detection and alignment: YuNet (OpenCV Zoo, MIT) finds faces and 5 landmarks;
  OpenCV's ``FaceRecognizerSF.alignCrop`` aligns them to 112x112.
- Embedding: SFace (Zhong et al., 2021; MobileFaceNet; OpenCV Zoo,
  Apache-2.0), 128-d. The pipeline and the cosine threshold 0.363 are OpenCV
  Zoo's (``demo.py``, ``sface.py``; 99.40% on LFW).

The score is ``1 - similarity``, where similarity is the median over sampled
frames of the best-matching face's mean top-3 cosine to the person's reference
embeddings. It is flagged at ``1 - 0.363``. It checks *who* the face is: it
catches impersonation and poor face swaps. A good face swap of the person is
meant to look like them, so it passes; motion, voice and consistency detectors
cover that.
"""

from pathlib import Path

import numpy as np

from forensix.hub import WeightFile, fetch
from forensix.identity.gallery import Gallery, top_k_similarity
from forensix.identity.media import sample_frames
from forensix.registry import ModelSpec, register
from forensix.types import Prediction

SFACE = WeightFile(
    "https://huggingface.co/opencv/face_recognition_sface/resolve/3d7082438a6e4551e840c9b2bb60b71e8da4b524/face_recognition_sface_2021dec.onnx",
    "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    "identity/face_recognition_sface_2021dec.onnx",
)
YUNET = WeightFile(
    "https://huggingface.co/opencv/face_detection_yunet/resolve/3cc26e7f1014a5ee5d74a42acee58bafc9d0a310/face_detection_yunet_2023mar.onnx",
    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    "identity/face_detection_yunet_2023mar.onnx",
)
COSINE_THRESHOLD = 0.363  # OpenCV Zoo sface.py (LFW)
MODEL_TAG = "sface-2021dec"


class FaceVerifier:
    def __init__(self, sface_path: Path, yunet_path: Path, gallery: Gallery | None = None,
                 conf_threshold: float = 0.9, sample_fps: float = 2.0):
        import cv2

        self.name = "id-face"
        self.threshold = 1.0 - COSINE_THRESHOLD
        self.recognizer = cv2.FaceRecognizerSF.create(str(sface_path), "")
        self.detector = cv2.FaceDetectorYN.create(str(yunet_path), "", (320, 320), conf_threshold, 0.3, 5000)
        self.gallery = gallery or Gallery(MODEL_TAG)
        self.sample_fps = sample_fps

    # -- faces -------------------------------------------------------------------

    def faces(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Detected faces (YuNet rows: box, 5 landmarks, score) and their SFace
        embeddings (N, 128) in one RGB image."""
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        self.detector.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, faces = self.detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return np.zeros((0, 15), np.float32), np.zeros((0, 128), np.float32)
        emb = np.stack([self.recognizer.feature(self.recognizer.alignCrop(bgr, f[:-1])).ravel() for f in faces])
        return faces, emb

    # -- enrollment ----------------------------------------------------------------

    def enroll(self, name: str, media, max_frames_per_video: int = 60) -> dict:
        """Add reference media of ``name`` (paths to images or videos, or RGB arrays).
        Each frame contributes its largest face. In videos, frames whose face
        doesn't look like the rest (another person, a bad detection) are dropped."""
        media = [media] if isinstance(media, (str, Path, np.ndarray)) else list(media)
        embeddings, sources = [], []
        for m in media:
            frames, _ = sample_frames(m, self.sample_fps, max_frames_per_video)
            per_frame = []
            for f in frames:
                faces, emb = self.faces(f)
                if len(faces):
                    per_frame.append(emb[int(np.argmax(faces[:, 2] * faces[:, 3]))])
            if not per_frame:
                continue
            e = np.stack(per_frame)
            if len(e) >= 3:  # keep frames consistent with the medoid
                n = e / np.linalg.norm(e, axis=1, keepdims=True)
                medoid = n[int(np.argmax((n @ n.T).sum(1)))]
                e = e[n @ medoid >= COSINE_THRESHOLD]
            embeddings.append(e)
            sources.extend([str(m) if not isinstance(m, np.ndarray) else "array"] * len(e))
        if not embeddings:
            raise ValueError(f"No face found in the reference media for {name!r}")
        self.gallery.add(name, np.concatenate(embeddings), sources)
        return {"name": name, "n_embeddings": int(len(self.gallery.embeddings(name)))}

    def save_gallery(self, path) -> None:
        self.gallery.save(path)

    def load_gallery(self, path) -> None:
        self.gallery = Gallery.load(path, MODEL_TAG)

    # -- verification ------------------------------------------------------------

    def verify(self, media, identity: str, max_frames: int = 120) -> dict:
        refs = self.gallery.embeddings(identity)
        frames, times = sample_frames(media, self.sample_fps, max_frames)
        per_frame, t_used, n_faces = [], [], []
        best_other: dict[str, list] = {n: [] for n in self.gallery.names() if n != identity}
        for f, t in zip(frames, times):
            faces, emb = self.faces(f)
            n_faces.append(len(faces))
            if not len(faces):
                continue
            sims = top_k_similarity(emb, refs)
            i = int(np.argmax(sims))  # the face in the frame that looks most like the person
            per_frame.append(float(sims[i])); t_used.append(round(t, 2))
            for other in best_other:
                best_other[other].append(float(top_k_similarity(emb[i : i + 1], self.gallery.embeddings(other))[0]))
        if not per_frame:
            raise ValueError("No face found")
        sim = np.array(per_frame)
        similarity = float(np.median(sim))
        closest = {n: round(float(np.median(v)), 4) for n, v in best_other.items() if v}
        return {
            "identity": identity,
            "similarity": round(similarity, 4),
            "threshold_similarity": COSINE_THRESHOLD,
            "frames_below_threshold": round(float(np.mean(sim < COSINE_THRESHOLD)), 4),
            "frame_similarity": np.round(sim, 4).tolist(),
            "frame_time_s": t_used,
            "frames_without_face": int(sum(n == 0 for n in n_faces)),
            "other_enrolled_similarity": closest,  # e.g. the face looks more like someone else enrolled
        }

    def predict(self, media, identity: str) -> Prediction:
        r = self.verify(media, identity)
        return Prediction(model=self.name, fake_prob=round(1.0 - r["similarity"], 4), threshold=self.threshold,
                          score_type="anomaly", segment_fake_probs=[round(1 - s, 4) for s in r["frame_similarity"]],
                          details=r)

    def identify(self, media, max_frames: int = 60) -> list[tuple[str, float]]:
        """Enrolled people ranked by similarity to the main face in ``media``."""
        frames, _ = sample_frames(media, self.sample_fps, max_frames)
        embs = []
        for f in frames:
            faces, emb = self.faces(f)
            if len(faces):
                embs.append(emb[int(np.argmax(faces[:, 2] * faces[:, 3]))])
        if not embs:
            raise ValueError("No face found")
        e = np.stack(embs)
        scores = {n: float(np.median(top_k_similarity(e, self.gallery.embeddings(n)))) for n in self.gallery.names()}
        return sorted(scores.items(), key=lambda kv: -kv[1])


def build(weights_path: Path, device: str = "cpu") -> FaceVerifier:
    return FaceVerifier(weights_path, fetch(YUNET))


register(
    ModelSpec(
        name="id-face",
        family="identity",
        builder=build,
        weights=SFACE,
        extra_weights=(YUNET,),
        license="Apache-2.0 (SFace, OpenCV Zoo). Face detector: YuNet (MIT, OpenCV Zoo). SFace was trained "
        "on CASIA-WebFace, VGGFace2 or MS-Celeb-1M (the release doesn't say which; all have "
        "research-only or non-commercial terms)",
        source="https://huggingface.co/opencv/face_recognition_sface/tree/3d7082438a6e4551e840c9b2bb60b71e8da4b524",
        paper="https://arxiv.org/abs/2205.12010",
        description="Reference-based face verification: enroll a person from photos or videos, then check "
        "whether the face in new media is them (SFace embeddings, YuNet alignment, OpenCV Zoo "
        "threshold). Frame-by-frame similarity timeline; catches impersonation and poor face swaps.",
    )
)
