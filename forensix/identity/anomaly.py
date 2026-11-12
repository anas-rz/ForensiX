"""person-anomaly: a per-person deepfake detector learned from real media only.

Give it real photos, videos or voice recordings of one person; it flags media
that isn't consistent with them. No fakes are needed, and nothing is trained
from scratch: the person is described by a frozen, pretrained identity encoder,
which tested far better than small networks trained on the person's media
(``person_anomaly_benchmark.md``).

- Faces: YuNet finds the face (on the frame scaled to fit 640x640, as the
  browser studio does, which also finds large faces that full-resolution
  detection misses), OpenCV's 5-point similarity alignment gives a 112x112 crop,
  and SFace embeds it (128-d). The person is the mean of their frames'
  embeddings. A clip's similarity is the median over sampled frames (2 per
  second, up to 60) of the cosine to that mean.
- Voice: ECAPA-TDNN (SpeechBrain spkrec-ecapa-voxceleb), embedded whole up to
  6 s, else in 3 s segments with a 1.5 s hop, skipping near-silent ones (as
  ``id-voice``). A clip's similarity is the median over segments.
- Decision: ``fake_prob = 1 - similarity``, flagged below the encoder's
  published population threshold (SFace 0.363, OpenCV Zoo on LFW; ECAPA 0.25,
  SpeechBrain). A person's own media is too narrow a sample to set a threshold
  reliably; it is used to report how typical a clip is for *them*
  (``details["person_percentile"]``: the share of their own held-out media that
  was less similar).

Models are small JSON files (``forensix-person-anomaly/1``), the same format the
browser studio (``/ui/studio/``) saves, so a person enrolled in the browser
loads here and the other way round.
"""

from __future__ import annotations

import json
from datetime import date
from os import PathLike
from pathlib import Path

import numpy as np

from forensix.identity.media import is_image, sample_frames
from forensix.registry import ModelSpec, register
from forensix.types import Prediction

FORMAT = "forensix-person-anomaly/1"
ENCODERS = {"face": "sface-2021dec", "voice": "ecapa-voxceleb-0f99f2d"}
THRESHOLDS = {"face": 0.363, "voice": 0.25}   # population cosine thresholds (OpenCV Zoo / SpeechBrain)
DETECT_SIZE = 640
FPS, MAX_FRAMES = 2.0, 60
AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}


# ------------------------------------------------------------------ encoders
class FaceEncoder:
    """Largest face per frame -> unit SFace embedding, with letterboxed detection (as the studio)."""

    def __init__(self):
        import cv2

        from forensix.hub import fetch
        from forensix.identity.face import SFACE, YUNET

        self.cv2 = cv2
        self.rec = cv2.FaceRecognizerSF.create(str(fetch(SFACE)), "")
        self.det = cv2.FaceDetectorYN.create(str(fetch(YUNET)), "", (DETECT_SIZE, DETECT_SIZE), 0.8, 0.3, 5000)

    def detect(self, rgb: np.ndarray) -> np.ndarray | None:
        """The largest face (YuNet row: box, 5 landmarks, score) in original pixel coordinates."""
        cv2 = self.cv2
        h, w = rgb.shape[:2]
        s = min(DETECT_SIZE / w, DETECT_SIZE / h)
        canvas = np.zeros((DETECT_SIZE, DETECT_SIZE, 3), np.uint8)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        canvas[:nh, :nw] = cv2.resize(np.ascontiguousarray(rgb[:, :, ::-1]), (nw, nh),
                                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        _, faces = self.det.detect(canvas)
        if faces is None or not len(faces):
            return None
        f = faces[int(np.argmax(faces[:, 2] * faces[:, 3]))].copy()
        f[:14] /= s
        return f

    def embed_frame(self, rgb: np.ndarray) -> np.ndarray | None:
        f = self.detect(rgb)
        if f is None:
            return None
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        e = self.rec.feature(self.rec.alignCrop(bgr, f[:-1])).ravel().astype(np.float32)
        return e / np.linalg.norm(e)

    def embed_media(self, media) -> tuple[np.ndarray, list[float], int]:
        """(embeddings of frames with a face, their times, number of frames sampled)."""
        frames, times = sample_frames(media, FPS, MAX_FRAMES)
        out, t_used = [], []
        for f, t in zip(frames, times):
            e = self.embed_frame(f)
            if e is not None:
                out.append(e); t_used.append(round(t, 2))
        return (np.stack(out) if out else np.zeros((0, 128), np.float32)), t_used, len(frames)


class VoiceEncoder:
    """Unit ECAPA embeddings of a clip's segments (the id-voice segmentation)."""

    def __init__(self, device: str = "cpu"):
        from forensix.audiovisual.avid import ECAPA_WEIGHTS
        from forensix.hub import fetch
        from forensix.identity.voice import VoiceVerifier

        self.v = VoiceVerifier(fetch(ECAPA_WEIGHTS), device=device)

    def embed_media(self, media) -> tuple[np.ndarray, list[float], int]:
        e, starts = self.v.embed_media(media)
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        return e.astype(np.float32), starts, len(starts)


def kind_of(media) -> str:
    """'face' for images and videos, 'voice' for audio files."""
    if isinstance(media, np.ndarray):
        return "voice" if media.ndim == 1 else "face"
    return "voice" if Path(str(media)).suffix.lower() in AUDIO_EXT else "face"


# ------------------------------------------------------------------ the model
class PersonAnomaly:
    """One person's identity profile with a population threshold and person-relative calibration."""

    def __init__(self, person: str, kind: str, centroid: np.ndarray, calibration: dict, n_enroll: int,
                 n_files: int, threshold: float | None = None, device: str = "cpu", meta: dict | None = None):
        if kind not in ENCODERS:
            raise ValueError(f"kind must be 'face' or 'voice', not {kind!r}")
        self.name = "person-anomaly"
        self.person, self.kind, self.device = person, kind, device
        self.centroid = np.asarray(centroid, np.float32) / np.linalg.norm(centroid)
        self.calibration = calibration
        self.n_enroll, self.n_files = n_enroll, n_files
        self.threshold_similarity = THRESHOLDS[kind] if threshold is None else float(threshold)
        self.meta = meta or {}
        self._encoder = None

    @property
    def threshold(self) -> float:
        return round(1.0 - self.threshold_similarity, 4)

    def __repr__(self) -> str:
        return (f"PersonAnomaly(person={self.person!r}, kind={self.kind!r}, n_enroll={self.n_enroll}, "
                f"files={self.n_files}, threshold_similarity={self.threshold_similarity})")

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = FaceEncoder() if self.kind == "face" else VoiceEncoder(self.device)
        return self._encoder

    # -- fitting -----------------------------------------------------------------
    @classmethod
    def fit(cls, person: str, real_media, kind: str = "auto", device: str = "cpu", folds: int = 3,
            verbose: bool = False) -> PersonAnomaly:
        """Enroll ``person`` from their real media (paths or arrays). ``kind`` is 'face', 'voice' or 'auto'."""
        media = [real_media] if isinstance(real_media, (str, PathLike, np.ndarray)) else list(real_media)
        if not media:
            raise ValueError("No real media given")
        kind = kind_of(media[0]) if kind == "auto" else kind
        enc = FaceEncoder() if kind == "face" else VoiceEncoder(device)
        embs, groups = [], []
        for i, m in enumerate(media):
            e, _, n = enc.embed_media(m)
            if verbose:
                print(f"[person-anomaly] {m if not isinstance(m, np.ndarray) else 'array'}: {len(e)}/{n} usable")
            embs.append(e); groups += [i] * len(e)
        E = np.concatenate(embs) if embs else np.zeros((0, 1), np.float32)
        model = cls.from_embeddings(person, kind, E, np.array(groups), folds=folds, device=device)
        model._encoder = enc
        return model

    @classmethod
    def from_embeddings(cls, person: str, kind: str, embeddings: np.ndarray, groups=None, folds: int = 3,
                        device: str = "cpu") -> PersonAnomaly:
        """Fit from precomputed unit embeddings; ``groups`` (e.g. source file per row) keep a file's
        samples in one calibration fold. With fewer files than ``folds``, contiguous blocks are used."""
        E = np.asarray(embeddings, np.float32)
        if len(E) < 2:
            raise ValueError(f"Need at least 2 usable samples (frames with a face, or speech segments); got {len(E)}")
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        g = np.arange(len(E)) if groups is None else np.asarray(groups)
        n_files = len(np.unique(g))
        if n_files < folds:
            g = (np.arange(len(E)) * folds) // len(E)
        oof = np.empty(len(E), np.float32)
        for f in np.unique(g):
            test = g == f
            c = E[~test].mean(0); c /= np.linalg.norm(c)
            oof[test] = E[test] @ c
        calibration = {"oof_similarity_quantiles": np.round(np.quantile(oof, np.linspace(0, 1, 21)), 5).tolist(),
                       "oof_n": int(len(oof)), "folds": int(len(np.unique(g)))}
        return cls(person, kind, E.mean(0), calibration, n_enroll=len(E), n_files=n_files, device=device)

    # -- scoring -----------------------------------------------------------------
    def person_percentile(self, similarity: float) -> float:
        """Share of the person's own held-out media that was less similar (0 = less like them than any of it)."""
        q = np.asarray(self.calibration["oof_similarity_quantiles"])
        return float(np.interp(similarity, q, np.linspace(0, 1, len(q)), left=0.0, right=1.0))

    def score_embeddings(self, E: np.ndarray) -> dict:
        sims = (np.asarray(E, np.float32) / np.linalg.norm(E, axis=1, keepdims=True)) @ self.centroid
        s = float(np.median(sims))
        return {"similarity": round(s, 4), "item_similarity": np.round(sims, 4).tolist(),
                "person_percentile": round(self.person_percentile(s), 3)}

    def predict(self, media) -> Prediction:
        E, times, n = self.encoder.embed_media(media)
        if not len(E):
            raise ValueError("No face found" if self.kind == "face" else "No speech found")
        r = self.score_embeddings(E)
        details = {
            "person": self.person, "kind": self.kind, **r,
            "threshold_similarity": self.threshold_similarity,
            "items_below_threshold": round(float(np.mean(np.array(r["item_similarity"]) < self.threshold_similarity)), 4),
            ("frame_time_s" if self.kind == "face" else "segment_start_s"): times,
        }
        if self.kind == "face":
            details["frames_without_face"] = n - len(E)
        return Prediction(model=self.name, fake_prob=round(1.0 - r["similarity"], 4), threshold=self.threshold,
                          score_type="anomaly", segment_fake_probs=[round(1 - s, 4) for s in r["item_similarity"]],
                          details=details)

    # -- files (shared with the browser studio) ------------------------------------
    def to_dict(self) -> dict:
        return {"format": FORMAT, "kind": self.kind, "person": self.person, "encoder": ENCODERS[self.kind],
                "created": self.meta.get("created", date.today().isoformat()), "trained_in": self.meta.get("trained_in", "python"),
                "centroid": [round(float(v), 7) for v in self.centroid], "threshold_similarity": self.threshold_similarity,
                "calibration": self.calibration, "n_enroll": self.n_enroll, "n_files": self.n_files}

    def save(self, path: str | PathLike) -> Path:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict()))
        return p

    @classmethod
    def load(cls, path: str | PathLike, device: str = "cpu") -> PersonAnomaly:
        d = json.loads(Path(path).read_text())
        if d.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} file")
        if d.get("encoder") != ENCODERS[d["kind"]]:
            raise ValueError(f"{path} was made with encoder {d.get('encoder')!r}; this version uses {ENCODERS[d['kind']]!r}")
        return cls(d["person"], d["kind"], np.array(d["centroid"]), d["calibration"], d["n_enroll"], d["n_files"],
                   threshold=d.get("threshold_similarity"), device=device,
                   meta={k: d.get(k) for k in ("created", "trained_in")})


def fit_person(person: str, real_media, kind: str = "auto", **kw) -> PersonAnomaly:
    """Enroll a person from their real media; see :class:`PersonAnomaly`."""
    return PersonAnomaly.fit(person, real_media, kind=kind, **kw)


# ------------------------------------------------------------------ browser assets
def export_ecapa_onnx(path: str | PathLike) -> Path:
    """Export the ECAPA network (log-filterbank input, (1, frames, 80)) to ONNX for the browser studio.
    The studio computes the filterbank itself; the network and weights are the ones id-voice uses."""
    import torch

    try:
        import onnx  # noqa: F401  (torch.onnx.export needs it)
    except ImportError:
        raise ImportError("Exporting the studio's voice model needs the onnx package: pip install onnx "
                          "(included in forensix[server])") from None

    from forensix.audiovisual.avid import ECAPA_WEIGHTS
    from forensix.audiovisual.avid.ecapa import SpeakerEncoder
    from forensix.hub import fetch, torch_load

    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    enc = SpeakerEncoder(); enc.embedding_model.load_state_dict(torch_load(fetch(ECAPA_WEIGHTS))); enc.eval()
    feats = torch.zeros(1, 300, 80)
    tmp = p.with_suffix(".tmp")
    torch.onnx.export(enc.embedding_model, (feats,), str(tmp), input_names=["feats"], output_names=["embedding"],
                      dynamic_axes={"feats": {1: "frames"}}, opset_version=17, dynamo=False)
    tmp.replace(p)
    return p


def _build(weights_path: Path | str | None, device: str = "cpu") -> PersonAnomaly:
    if weights_path is None:
        raise ValueError("person-anomaly needs a person file: enroll one with forensix.identity.anomaly.fit_person("
                         "'alice', real_files).save('alice.json') or in the browser studio (/ui/studio/), "
                         "then pass weights_path=...")
    return PersonAnomaly.load(weights_path, device=device)


register(
    ModelSpec(
        name="person-anomaly",
        family="identity",
        builder=_build,
        weights=None,
        license="MIT (ForensiX). Encoders: SFace (Apache-2.0; trained on research-only face datasets, see id-face), "
        "YuNet (MIT), ECAPA-TDNN (Apache-2.0, SpeechBrain; trained on VoxCeleb)",
        source="",
        paper="",
        description="Per-person deepfake detector learned from the person's real media only (photos, video or "
        "voice): distance from their identity profile under a frozen face or speaker encoder, with the encoder's "
        "population threshold. Enroll in Python or in the browser studio; same file format.",
    )
)


__all__ = ["FORMAT", "PersonAnomaly", "FaceEncoder", "VoiceEncoder", "fit_person", "export_ecapa_onnx", "kind_of", "is_image"]
