"""SyncNet: Out of time: automated lip sync in the wild (Chung & Zisserman,
ACCV 2016 workshops). https://github.com/joonson/syncnet_python (MIT, see
../LICENSE-syncnet).

A two-stream network embeds 5 video frames of a talking face and the 0.2 s of
MFCCs under them; in sync, the embeddings are close. Following upstream
``SyncNetInstance.evaluate``, the audio is slid over the video by -15..+15
frames: the shift with the smallest mean distance is the AV offset, and the
confidence (LSE-C) is the median distance minus that minimum. Audio that doesn't
belong to the face (dubbing, voice swaps, splices) has no clear minimum.

Output: an anomaly score, ``-confidence`` (higher = worse sync), not a
probability. The best offset, LSE-D (the minimum distance), per-frame
confidence and every face track's results are in ``details``.
"""

from pathlib import Path

import numpy as np
import torch
from scipy.fft import dct
from scipy.signal import medfilt

from forensix.hub import WeightFile, fetch, torch_load
from forensix.registry import ModelSpec, register
from forensix.types import Prediction
from forensix.video.dfdc_wm import WEIGHTS as _WM_WEIGHTS

FPS, SAMPLE_RATE = 25, 16000
AUD_FACT = SAMPLE_RATE // FPS  # 640 samples per frame
VSHIFT = 15  # upstream demo default
BATCH = 20

WEIGHTS = WeightFile(
    "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model",
    "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442",
    "syncnet/syncnet_v2.model",
)


def mfcc(signal: np.ndarray, samplerate: int = SAMPLE_RATE) -> np.ndarray:
    """``python_speech_features.mfcc`` with its defaults (MIT; 13 cepstra, 26
    mel filters, 25 ms rectangular frames every 10 ms, NFFT 512, pre-emphasis
    0.97, lifter 22, log energy as c0). Returns (frames, 13)."""
    x = np.asarray(signal, dtype=np.float64)
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    frame_len, step, nfft, nfilt = int(0.025 * samplerate + 0.5), int(0.01 * samplerate + 0.5), 512, 26
    n = 1 if len(x) <= frame_len else 1 + int(np.ceil((len(x) - frame_len) / step))
    x = np.concatenate([x, np.zeros((n - 1) * step + frame_len - len(x))])
    frames = x[np.arange(frame_len)[None] + step * np.arange(n)[:, None]]
    pspec = np.abs(np.fft.rfft(frames, nfft)) ** 2 / nfft
    energy = pspec.sum(1)
    energy = np.where(energy == 0, np.finfo(float).eps, energy)

    mel = np.linspace(0, 2595 * np.log10(1 + samplerate / 2 / 700.0), nfilt + 2)
    bins = np.floor((nfft + 1) * (700 * (10 ** (mel / 2595.0) - 1)) / samplerate)
    fb = np.zeros((nfilt, nfft // 2 + 1))
    for j in range(nfilt):
        for i in range(int(bins[j]), int(bins[j + 1])):
            fb[j, i] = (i - bins[j]) / (bins[j + 1] - bins[j])
        for i in range(int(bins[j + 1]), int(bins[j + 2])):
            fb[j, i] = (bins[j + 2] - i) / (bins[j + 2] - bins[j + 1])
    feat = pspec @ fb.T
    feat = np.where(feat == 0, np.finfo(float).eps, feat)
    feat = dct(np.log(feat), type=2, axis=1, norm="ortho")[:, :13]
    feat = feat * (1 + 11.0 * np.sin(np.pi * np.arange(13) / 22))
    feat[:, 0] = np.log(energy)
    return feat


class SyncNetDetector:
    """Lip-sync anomaly score for a talking-face video (higher = worse sync)."""

    def __init__(self, net: torch.nn.Module, face_detector, device: str = "cpu", vshift: int = VSHIFT):
        self.name = "syncnet"
        self.threshold = None  # anomaly score
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()
        self.face_detector = face_detector
        self.vshift = vshift

    @torch.inference_mode()
    def embed_track(self, crops: np.ndarray, audio: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Lip and audio embeddings (windows, 1024) of one face track: ``crops``
        (T, 224, 224, 3) RGB uint8 at 25 fps, and ``audio`` int16-valued samples at
        16 kHz from the track's first frame. Window i covers frames i..i+4."""
        im = torch.from_numpy(np.ascontiguousarray(crops[..., ::-1])).float()  # upstream reads BGR (cv2)
        im = im.permute(3, 0, 1, 2)[None]  # 1, C, T, H, W
        cc = torch.from_numpy(mfcc(audio).T[None, None].copy()).float()  # 1, 1, 13, frames
        lastframe = min(len(crops), len(audio) // AUD_FACT) - 5
        if lastframe < 1:
            raise ValueError("Need at least 6 frames with a tracked face")
        im_feat, cc_feat = [], []
        for i in range(0, lastframe, BATCH):
            idx = range(i, min(lastframe, i + BATCH))
            im_feat.append(self.net.forward_lip(torch.cat([im[:, :, v : v + 5] for v in idx]).to(self.device)).cpu())
            cc_feat.append(self.net.forward_aud(torch.cat([cc[:, :, :, v * 4 : v * 4 + 20] for v in idx]).to(self.device)).cpu())
        return torch.cat(im_feat), torch.cat(cc_feat)

    @torch.inference_mode()
    def embed_audio(self, audio: np.ndarray, n_windows: int) -> torch.Tensor:
        """Audio embeddings alone (n_windows, 1024), as in ``embed_track``."""
        cc = torch.from_numpy(mfcc(audio).T[None, None].copy()).float()
        n = min(n_windows, (cc.shape[-1] - 20) // 4 + 1)
        out = [self.net.forward_aud(torch.cat([cc[:, :, :, v * 4 : v * 4 + 20]
                                               for v in range(i, min(n, i + BATCH))]).to(self.device)).cpu()
               for i in range(0, n, BATCH)]
        return torch.cat(out)

    @torch.inference_mode()
    def analyze_track(self, crops: np.ndarray, audio: np.ndarray) -> dict:
        """One face track: ``crops`` (T, 224, 224, 3) RGB uint8 at 25 fps, and
        ``audio`` int16-valued samples at 16 kHz from the track's first frame."""
        im_feat, cc_feat = self.embed_track(crops, audio)

        # upstream calc_pdist: distance of each video window to audio shifted by -vshift..+vshift
        win = 2 * self.vshift + 1
        padded = torch.nn.functional.pad(cc_feat, (0, 0, self.vshift, self.vshift))
        dists = torch.stack([torch.nn.functional.pairwise_distance(im_feat[[i]].repeat(win, 1), padded[i : i + win])
                             for i in range(len(im_feat))]).numpy()  # (frames, shifts)
        mdist = dists.mean(0)
        best = int(np.argmin(mdist))
        conf = float(np.median(mdist) - mdist[best])
        framewise = medfilt(np.median(mdist) - dists[:, best], kernel_size=9)
        return {
            "offset_frames": self.vshift - best,
            "confidence": round(conf, 4),  # LSE-C
            "min_dist": round(float(mdist[best]), 4),  # LSE-D
            "dist_at_zero_offset": round(float(mdist[self.vshift]), 4),
            "mean_dist_by_offset": {self.vshift - k: round(float(d), 4) for k, d in enumerate(mdist)},
            "framewise_confidence": np.round(framewise, 3).tolist(),
            "n_windows": int(len(im_feat)),
        }

    def predict(self, video) -> Prediction:
        from forensix.audiovisual.facetrack import crop_track, detect_faces, track_faces
        from forensix.audiovisual.io import read_av

        frames, audio = read_av(video, FPS, SAMPLE_RATE)
        n = min(len(frames), len(audio) // AUD_FACT)
        frames, audio = frames[:n], audio[: n * AUD_FACT]
        detections = detect_faces(frames, self.face_detector)
        tracks = track_faces(detections)
        if not tracks:  # shorter or smaller than upstream's defaults: accept any usable track
            tracks = track_faces(detections, min_track=6, min_face_size=0)
        if not tracks:
            raise ValueError(f"No face track in {video}")
        results = []
        for track in tracks:
            start = int(track["frame"][0])
            crops = crop_track(frames, track)
            r = self.analyze_track(crops, audio[start * AUD_FACT : (start + len(crops)) * AUD_FACT])
            r.update(start_s=round(start / FPS, 2), end_s=round((start + len(crops)) / FPS, 2))
            results.append(r)
        # the speaker: the track that is most in sync (upstream's active-speaker use)
        speaker = max(results, key=lambda r: r["confidence"])
        details = {k: v for k, v in speaker.items()}
        details["tracks"] = results
        return Prediction(model=self.name, fake_prob=-speaker["confidence"], threshold=None,
                          score_type="anomaly", details=details)

    def predict_batch(self, videos) -> list[Prediction]:
        return [self.predict(v) for v in videos]


def build(weights_path: Path, device: str = "cpu") -> SyncNetDetector:
    from forensix.audiovisual.syncnet.model import S
    from forensix.video.retinaface import RetinaFaceDetector

    net = S(num_layers_in_fc_layers=1024)
    state = net.state_dict()
    for name, param in torch_load(weights_path).items():  # upstream loadParameters: copy by name
        state[name].copy_(param)
    face_detector = RetinaFaceDetector(fetch(_WM_WEIGHTS["retinaface"]), device=device, nms_threshold=0.4)
    return SyncNetDetector(net, face_detector, device=device)


register(
    ModelSpec(
        name="syncnet",
        family="audiovisual",
        builder=build,
        weights=WEIGHTS,
        extra_weights=(_WM_WEIGHTS["retinaface"],),
        license="MIT* (code; weights hosted by the Oxford VGG group, linked from the MIT repo). "
        "Face detector: RetinaFace (MIT). Trained on BBC television (research-only terms)",
        source="https://github.com/joonson/syncnet_python/tree/907c0b579c2e2d83f0eae1b2ac9e720cde4e5623",
        paper="https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf",
        description="Audio-visual lip-sync network: finds the AV offset and how clearly the audio "
        "matches the face (LSE-C). Flags audio that doesn't belong to the face (dubbing, voice "
        "swaps, splices). Returns an anomaly score (-confidence; higher = worse sync), not a "
        "probability. Lip-sync generators trained against SyncNet (e.g. Wav2Lip) can pass it.",
    )
)
