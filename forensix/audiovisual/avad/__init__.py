"""AVAD: Self-Supervised Video Forensics by Audio-Visual Anomaly Detection
(Feng, Chen & Owens, CVPR 2023). https://github.com/cfeng16/audio-visual-forensics
(MIT, see LICENSE in this directory).

Trained only on real talking-head videos (LRS2): an audio-visual
synchronization network scores 31 audio offsets (-15..+15 frames) for each
5-frame window, and an autoregressive model of those offset distributions
flags videos whose sync pattern is unusual. The output is an anomaly score
(KL divergence), not a probability: higher means more likely manipulated.

Scoring follows upstream detect.py with ``lam=0`` (offset distributions only;
the PCA-based feature branch needs a pickled scikit-learn object and is
omitted) and ``max_len=50`` windows. Upstream expects pre-cropped 25 fps face
tracks from syncnet_python; ForensiX builds them with forensix.audiovisual.facetrack.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from forensix.hub import WeightFile, fetch, torch_load
from forensix.registry import ModelSpec, register
from forensix.types import Prediction
from forensix.video.dfdc_wm import WEIGHTS as _WM_WEIGHTS

FPS, SAMPLE_RATE = 25, 16000
AUD_FACT = SAMPLE_RATE // FPS  # 640 audio samples per video frame
CLIP_LEN, MAX_SHIFT, N_SHIFTS = 5, 15, 31
MAX_LEN = 50

_REPO = "https://raw.githubusercontent.com/cfeng16/audio-visual-forensics/b6358e17738869fae3223cdd59e7fc09f0442161"
WEIGHTS = {
    "sync": WeightFile("https://drive.usercontent.google.com/download?export=download&confirm=t&id=1BxaPiZmpiOJDsbbq8ZIDHJU7--RJE7Br",
                       "b0df6260483e8b02173be7b9ed161a2f4edc9b68bf4e667aaf4a3ef8cc2585c3", "avad/sync_model.pth"),
    "dist": WeightFile(f"{_REPO}/dist_regressive_model.pth",
                       "e9158714f334688c95ce725e4c56d0bc5b7dc5cd4d7e275cfcb22d69fc873971", "avad/dist_regressive_model.pth"),
}


class SyncNetwork(nn.Module):
    """Upstream detect.py ``network`` (inference path): visual encoder, audio
    encoder and fusion transformer scoring one audio-video alignment."""

    def __init__(self):
        super().__init__()
        from forensix.audiovisual.avad.audio_process import AudioEncoder
        from forensix.audiovisual.avad.backbone import VisualEncoder
        from forensix.audiovisual.avad.model import MP_AViT

        self.vis_enc = VisualEncoder()
        self.aud_enc = AudioEncoder()
        self.transformer = MP_AViT(image_size=14, patch_size=0, num_classes=1, dim=512, depth=3, heads=4,
                                   mlp_dim=512, dim_head=128, dropout=0.1, emb_dropout=0.1,
                                   max_visual_len=5, max_audio_len=4)

    def offset_logits(self, clip: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """clip: (3, 5, 224, 224) raw 0-255; audio: (35 * 640,) samples starting 15
        frames before the clip. Returns 31 logits for offsets -15..+15."""
        from forensix.audiovisual.avad.features import log_mel

        windows = torch.stack([audio[i * AUD_FACT : (i + CLIP_LEN) * AUD_FACT] for i in range(N_SHIFTS)])
        fbank = log_mel(windows).permute(0, 2, 1)[:, None]  # (31, 1, 80, 20)
        # the visual embedding is identical for all 31 offsets: compute it once
        vid_emb = self.vis_enc(clip[None]).expand(N_SHIFTS, -1, -1, -1, -1).clone()
        return self.transformer(vid_emb, self.aud_enc(fbank)).flatten()


class AVADDetector:
    """Anomaly score for a talking-face video (higher = more likely fake)."""

    def __init__(self, sync: SyncNetwork, decoder: nn.Module, face_detector, device="cpu", max_len: int = MAX_LEN):
        self.name = "avad"
        self.device = torch.device(device)
        self.sync = sync.to(self.device).eval()
        self.decoder = decoder.to(self.device).eval()
        self.face_detector = face_detector
        self.max_len = max_len

    @torch.inference_mode()
    def score_track(self, crops: np.ndarray, audio: np.ndarray) -> float:
        """Score one aligned face track: ``crops`` (T, 224, 224, 3) RGB uint8 at
        25 fps and ``audio`` (T * 640,) int16-valued samples covering the same frames."""
        video = torch.from_numpy(crops).float().permute(3, 0, 1, 2).to(self.device)  # C, T, H, W (0-255)
        padded = np.pad(audio, (MAX_SHIFT * AUD_FACT, MAX_SHIFT * AUD_FACT))
        padded = torch.from_numpy(padded.astype(np.float32)).to(self.device)
        n = min(video.shape[1] - CLIP_LEN + 1, self.max_len)
        if n < 2:
            raise ValueError("Need at least 6 frames with a tracked face")
        logits = torch.stack([
            self.sync.offset_logits(video[:, k : k + CLIP_LEN], padded[k * AUD_FACT : (k + CLIP_LEN + 2 * MAX_SHIFT) * AUD_FACT])
            for k in range(n)
        ]).double().cpu()  # upstream accumulates in a float64 numpy array
        seq = F.pad(logits, (0, 0, 0, self.max_len - n))[None].to(self.device)
        mask = torch.zeros(1, self.max_len, device=self.device)
        mask[:, n:] = 1.0
        target = F.softmax(seq[:, 1:].float(), dim=2)
        log_pred = F.log_softmax(self.decoder(seq[:, :-1].float(), mask[:, :-1]), dim=2)
        kl = F.kl_div(log_pred, target, reduction="none")  # target * (log target - log_pred)
        return float(kl[0, : n - 1].sum(dim=1).mean())

    def predict(self, video) -> Prediction:
        from forensix.audiovisual.facetrack import crop_track, detect_faces, track_faces
        from forensix.audiovisual.io import read_av

        frames, audio = read_av(video, FPS, SAMPLE_RATE)
        n = min(len(frames), len(audio) // AUD_FACT)
        frames, audio = frames[:n], audio[: n * AUD_FACT]
        detections = detect_faces(frames, self.face_detector)
        tracks = track_faces(detections)
        if not tracks:  # shorter or smaller than upstream's defaults: use the longest usable track
            tracks = track_faces(detections, min_track=CLIP_LEN, min_face_size=0)
        if not tracks:
            raise ValueError(f"No face track in {video}")
        track = max(tracks, key=lambda t: len(t["frame"]))
        crops = crop_track(frames, track)
        start = int(track["frame"][0])
        score = self.score_track(crops, audio[start * AUD_FACT : (start + len(crops)) * AUD_FACT])
        return Prediction(model=self.name, fake_prob=score, threshold=None, score_type="anomaly")

    def predict_batch(self, videos) -> list[Prediction]:
        return [self.predict(v) for v in videos]


def build(weights_path: Path, device: str = "cpu") -> AVADDetector:
    from forensix.audiovisual.avad.transformer_component import transformer_decoder
    from forensix.video.retinaface import RetinaFaceDetector

    sync = SyncNetwork()
    sync.load_state_dict(torch_load(weights_path))
    decoder = transformer_decoder(input_dim_old=31, input_dim=256, compress_factor=1, num_heads=16,
                                  dropout_prob=0.1, max_len=49, layers=2)
    decoder.load_state_dict(torch_load(fetch(WEIGHTS["dist"])))
    face_detector = RetinaFaceDetector(fetch(_WM_WEIGHTS["retinaface"]), device=device, nms_threshold=0.4)
    return AVADDetector(sync, decoder, face_detector, device=device)


register(
    ModelSpec(
        name="avad",
        family="audiovisual",
        builder=build,
        weights=WEIGHTS["sync"],
        extra_weights=(WEIGHTS["dist"], _WM_WEIGHTS["retinaface"]),
        license="MIT (code, weights; sync model hosted on the authors' Google Drive). Face "
        "detector: RetinaFace (MIT). Trained on LRS2 (BBC, research-only terms)",
        source="https://github.com/cfeng16/audio-visual-forensics/tree/b6358e17738869fae3223cdd59e7fc09f0442161",
        paper="https://arxiv.org/abs/2301.01767",
        description="Self-supervised audio-visual anomaly detection: an A/V sync network trained "
        "on real videos only, plus an autoregressive model of its per-window offset "
        "distributions. Returns an anomaly score (higher = more likely fake), not a probability.",
    )
)
