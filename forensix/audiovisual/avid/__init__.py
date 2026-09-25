"""avid: does this voice belong to this face? (ForensiX)

Voice replacement keeps the lips in sync when a lip-sync model is applied
afterwards, so sync checks (``syncnet``, ``stablesyncnet``, ``avcons``) miss it.
The identity doesn't match, though. avid compares who the voice sounds like with
who the face looks like, and checks that neither changes mid-clip.

- Voice: ECAPA-TDNN speaker embeddings (SpeechBrain, Apache-2.0; trained on
  VoxCeleb, CC BY 4.0), one per 3 s segment (1.5 s hop).
- Face: DINOv2 ViT-S/14 (Apache-2.0) embeddings of the talking-face crops, averaged
  per segment. This is a general visual backbone: face-recognition networks are
  trained on non-commercial face datasets (VGGFace2, MS-Celeb, CASIA-WebFace).
- A voice-face space learned from real videos only (``model.py``), identity
  continuity, and a scorer calibrated against voice swaps and splices made from
  real clips (``train.py``).

Output: the probability that the voice and face don't belong to the same person
(threshold 0.5), per-segment voice-face match for localization, and the
continuity measures. Weights for the heads and scorer: train them with
``scripts/train_avid.py``; ``forensix.load("avid", weights_path=...)``.
"""

from pathlib import Path

import numpy as np
import torch

from forensix.hub import WeightFile, fetch, torch_load
from forensix.registry import ModelSpec, register
from forensix.types import Prediction
from forensix.video.dfdc_wm import WEIGHTS as _WM_WEIGHTS

FPS, SAMPLE_RATE = 25, 16000
SEGMENT_S, HOP_S = 3.0, 1.5
FACE_STEP = 5  # frames between face samples (5 per second)
SILENCE_DB = -30.0  # segments this far below the loudest are skipped
MIN_SEGMENTS = 2

ECAPA_WEIGHTS = WeightFile(
    "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286/embedding_model.ckpt",
    "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2",
    "avid/embedding_model.ckpt",
)
DINO_WEIGHTS = WeightFile(
    "https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m/resolve/4610ca143709d58a633b6397a74412c2c3842454/model.safetensors",
    "04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081",
    "avid/dinov2_vits14_lvd142m.safetensors",
)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Encoders:
    """Frozen voice (ECAPA-TDNN) and face (DINOv2 ViT-S/14) encoders plus a face detector."""

    def __init__(self, device: str = "cpu"):
        import timm
        from safetensors.torch import load_file

        from forensix.audiovisual.avid.ecapa import SpeakerEncoder
        from forensix.video.retinaface import RetinaFaceDetector

        self.device = torch.device(device)
        self.speaker = SpeakerEncoder()
        self.speaker.embedding_model.load_state_dict(torch_load(fetch(ECAPA_WEIGHTS)))
        self.speaker.to(self.device).eval()
        self.dino = timm.create_model("vit_small_patch14_dinov2", pretrained=False, dynamic_img_size=True)
        self.dino.load_state_dict(load_file(str(fetch(DINO_WEIGHTS))))
        self.dino.to(self.device).eval()
        self.face_detector = RetinaFaceDetector(fetch(_WM_WEIGHTS["retinaface"]), device=device, nms_threshold=0.4)

    @torch.no_grad()
    def faces(self, crops: np.ndarray, batch: int = 32) -> np.ndarray:
        """(N, 384) DINOv2 class embeddings of 224x224 RGB uint8 crops."""
        out = []
        for i in range(0, len(crops), batch):
            x = torch.from_numpy(crops[i : i + batch]).permute(0, 3, 1, 2).float().div(255)
            out.append(self.dino(((x - _MEAN) / _STD).to(self.device)).cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def voices(self, segments: list[np.ndarray]) -> np.ndarray:
        """(N, 192) ECAPA embeddings of equal-length [-1, 1] segments at 16 kHz."""
        x = torch.from_numpy(np.stack(segments).astype(np.float32)).to(self.device)
        return self.speaker(x).cpu().numpy()


def segment_features(frames: np.ndarray, audio: np.ndarray, encoders: Encoders) -> list[dict]:
    """Per face track: time-aligned ``voice`` (S, 192) and ``face`` (S, 384)
    segment embeddings and segment start times. ``frames`` RGB uint8 at 25 fps,
    ``audio`` [-1, 1] at 16 kHz with the same start. Silent segments are skipped."""
    from forensix.audiovisual.facetrack import crop_track, detect_faces, track_faces

    detections = detect_faces(frames, encoders.face_detector)
    tracks = track_faces(detections) or track_faces(detections, min_track=int(SEGMENT_S * FPS), min_face_size=0)
    seg, hop = int(SEGMENT_S * FPS), int(HOP_S * FPS)
    out = []
    for track in tracks:
        start, n = int(track["frame"][0]), len(track["frame"])
        if n < seg:
            continue
        crops = crop_track(frames, track)
        face_idx = np.arange(0, n, FACE_STEP)
        face_emb = encoders.faces(crops[face_idx])
        starts = list(range(0, n - seg + 1, hop))
        audio_segs = [audio[int((start + s) / FPS * SAMPLE_RATE) : int((start + s + seg) / FPS * SAMPLE_RATE)]
                      for s in starts]
        keep = [i for i, a in enumerate(audio_segs) if len(a) == int(SEGMENT_S * SAMPLE_RATE)]
        if not keep:
            continue
        levels = np.array([10 * np.log10(np.mean(audio_segs[i] ** 2) + 1e-12) for i in keep])
        keep = [i for i, lv in zip(keep, levels) if lv > levels.max() + SILENCE_DB]
        if len(keep) < MIN_SEGMENTS:
            continue
        voice = encoders.voices([audio_segs[i] for i in keep])
        face = np.stack([face_emb[(face_idx >= starts[i]) & (face_idx < starts[i] + seg)].mean(0) for i in keep])
        out.append({"voice": voice, "face": face, "start_s": [round((start + starts[i]) / FPS, 2) for i in keep],
                    "track_start_s": round(start / FPS, 2), "n_frames": n})
    return out


class AVIdentityDetector:
    def __init__(self, heads, scorer, encoders: Encoders, device: str = "cpu"):
        self.name = "avid"
        self.threshold = 0.5
        self.heads, self.scorer, self.encoders = heads.eval(), scorer.eval(), encoders
        self.device = device

    @torch.no_grad()
    def analyze(self, frames: np.ndarray, audio: np.ndarray) -> dict:
        from forensix.audiovisual.avid.model import FEATURES, clip_features

        tracks = segment_features(frames, audio, self.encoders)
        if not tracks:
            raise ValueError(f"Need a face track and speech of at least {SEGMENT_S * MIN_SEGMENTS - HOP_S:.1f} s")
        results = []
        for t in tracks:
            feats, seg_match = clip_features(self.heads, torch.from_numpy(t["voice"]), torch.from_numpy(t["face"]))
            prob = float(self.scorer(feats[None])[0])
            results.append({
                "fake_prob": round(prob, 4),
                "features": {k: round(float(v), 4) for k, v in zip(FEATURES, feats)},
                "segment_match": np.round(seg_match.numpy(), 4).tolist(),
                "segment_start_s": t["start_s"],
                "segment_s": SEGMENT_S,
                "track_start_s": t["track_start_s"],
            })
        # the face the voice belongs to best is the candidate speaker; if even it doesn't match, flag
        best = min(results, key=lambda r: r["fake_prob"])
        return {**best, "tracks": results}

    def predict(self, video) -> Prediction:
        from forensix.audiovisual.io import read_av

        frames, audio = read_av(video, FPS, SAMPLE_RATE)
        n = min(len(frames), len(audio) // (SAMPLE_RATE // FPS))
        result = self.analyze(frames[:n], audio[: n * (SAMPLE_RATE // FPS)] / 32768.0)
        return Prediction(model=self.name, fake_prob=result["fake_prob"], threshold=self.threshold, details=result)

    def predict_batch(self, videos) -> list[Prediction]:
        return [self.predict(v) for v in videos]


def build(weights_path: Path | None, device: str = "cpu") -> AVIdentityDetector:
    from forensix.audiovisual.avid.train import load

    if weights_path is None:
        raise ValueError("avid has no published weights: train them with scripts/train_avid.py and "
                         "pass forensix.load('avid', weights_path=...)")
    heads, scorer, _ = load(weights_path, device)
    return AVIdentityDetector(heads, scorer, Encoders(device), device)


register(
    ModelSpec(
        name="avid",
        family="audiovisual",
        builder=build,
        weights=None,  # heads and scorer are trained by you (scripts/train_avid.py)
        extra_weights=(ECAPA_WEIGHTS, DINO_WEIGHTS, _WM_WEIGHTS["retinaface"]),
        license="ForensiX (heads and scorer, trained by you on real media). Voice: SpeechBrain "
        "ECAPA-TDNN (Apache-2.0, trained on VoxCeleb). Face: DINOv2 ViT-S/14 (Apache-2.0). Face "
        "detector: RetinaFace (MIT)",
        source="https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb",
        paper="https://arxiv.org/abs/2005.07143",
        description="Voice-face identity consistency: whether the voice belongs to the face (a "
        "voice-face space learned from real videos on ECAPA-TDNN voice and DINOv2 face embeddings), "
        "and whether either identity changes mid-clip. Catches voice replacement even when the lips "
        "are in sync. Probability of an identity mismatch, with per-segment localization.",
    )
)
