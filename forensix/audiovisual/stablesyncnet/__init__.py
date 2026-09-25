"""StableSyncNet: the lip-sync expert of LatentSync (Li et al., ByteDance, 2024;
https://arxiv.org/abs/2412.09262, https://github.com/bytedance/LatentSync).

Two convolutional encoders with self-attention embed 16 frames of the lower
half of an aligned face (256x256) and the mel spectrogram under them (80x52).
Trained on real talking faces (VoxCeleb2, HDTF) with synchronized pairs as
positives and pairs from elsewhere in the same video as negatives, its cosine
similarity is a sync probability (upstream thresholds it at 0.5; 94% accuracy on
VoxCeleb2 and HDTF).

Detection: each 16-frame window is scored against the audio at offsets of
-6..+6 frames (LatentSync's training data was offset-corrected within that
range). ``fake_prob`` is 1 minus the mean sync probability at the best offset,
so it is the probability that the audio doesn't belong to the face.

Preprocessing is ForensiX's own. LatentSync aligns faces with InsightFace's
106-point landmarks (non-commercial models) and code adapted from StyleSync (no
license), and computes mels with code adapted from Wav2Lip (non-commercial).
Here the same three alignment points (both brow centres and the nose centre)
come from MediaPipe FaceLandmarker (Apache-2.0), and the similarity alignment
and mel spectrogram are written from their definitions.
"""

from pathlib import Path

import numpy as np
import torch

from forensix.hub import WeightFile, torch_load
from forensix.registry import ModelSpec, register
from forensix.types import Prediction

FPS, SAMPLE_RATE = 25, 16000
NUM_FRAMES, MEL_WINDOW = 16, 52  # 16 frames = 0.64 s = 51.2 mel frames, rounded up
MAX_OFFSET = 6  # LatentSync kept training clips within +-6 frames after offset correction
BATCH = 8
RESOLUTION = 256

CONFIG = {  # configs/syncnet/syncnet_16_pixel_attn.yaml
    "audio_encoder": {"in_channels": 1, "block_out_channels": [32, 64, 128, 256, 512, 1024, 2048],
                      "downsample_factors": [[2, 1], 2, 2, 1, 2, 2, [2, 3]], "attn_blocks": [0, 0, 0, 1, 1, 0, 0],
                      "dropout": 0.0},
    "visual_encoder": {"in_channels": 48, "block_out_channels": [64, 128, 256, 256, 512, 1024, 2048, 2048],
                       "downsample_factors": [[1, 2], 2, 2, 2, 2, 2, 2, 2], "attn_blocks": [0, 0, 0, 0, 1, 1, 0, 0],
                       "dropout": 0.0},
}

WEIGHTS = WeightFile(
    "https://huggingface.co/ByteDance/LatentSync-1.6/resolve/c42c7e6c8e9c213626389fa7d9a3c444b8536353/stable_syncnet.pt",
    "77678d13861a02d6e83d0b169962175b6266be773f1f6f85fda2b0182e225118",
    "stablesyncnet/stable_syncnet_1.6.pt",
)

# ---- audio: mel spectrogram (LatentSync configs/audio.yaml) ----------------------

N_FFT, HOP, N_MELS, FMIN, FMAX = 800, 200, 80, 55.0, 7600.0
MIN_LEVEL_DB, REF_LEVEL_DB, MAX_ABS = -100.0, 20.0, 4.0


def _hz_to_mel(f):
    """Slaney mel scale: linear below 1 kHz, logarithmic above."""
    f = np.asarray(f, dtype=np.float64)
    mel = f / (200.0 / 3)
    log_region = f >= 1000.0
    return np.where(log_region, 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0), mel)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    f = m * (200.0 / 3)
    return np.where(m >= 15.0, 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)), f)


def mel_filterbank(sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS,
                   fmin: float = FMIN, fmax: float = FMAX) -> np.ndarray:
    """Triangular Slaney-normalized mel filters (n_mels, 1 + n_fft // 2)."""
    fft_freqs = np.linspace(0, sr / 2, 1 + n_fft // 2)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0, np.minimum(lower, upper))
    return weights * (2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels]))[:, None]


def melspectrogram(wav: np.ndarray) -> np.ndarray:
    """(80, frames) normalized log-mel of a [-1, 1] waveform at 16 kHz: pre-emphasis
    0.97, centred STFT (800-sample Hann window, hop 200, zero padding), Slaney
    mels 55-7600 Hz, dB re 20, scaled to [-4, 4]."""
    x = np.asarray(wav, dtype=np.float64)
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    x = np.pad(x, (N_FFT // 2, N_FFT // 2))
    n = 1 + (len(x) - N_FFT) // HOP
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)  # periodic Hann
    frames = x[np.arange(N_FFT)[None] + HOP * np.arange(n)[:, None]] * window
    mag = np.abs(np.fft.rfft(frames, axis=1)).T  # (1 + n_fft/2, frames)
    s = 20 * np.log10(np.maximum(10 ** (MIN_LEVEL_DB / 20), mel_filterbank() @ mag)) - REF_LEVEL_DB
    return np.clip(2 * MAX_ABS * (s - MIN_LEVEL_DB) / -MIN_LEVEL_DB - MAX_ABS, -MAX_ABS, MAX_ABS).astype(np.float32)


# ---- video: 3-point similarity alignment ---------------------------------------

_RATIO = RESOLUTION / 256 * 2.8
FACE_TEMPLATE = np.array([[17.0, 20.0], [58.0, 20.0], [37.5, 40.0]]) * _RATIO  # brow, brow, nose
FACE_SIZE = (int(75 * _RATIO), int(100 * _RATIO))  # (w, h) = (210, 280)

# MediaPipe FaceLandmarker points standing in for InsightFace 2d106's brow and nose
# groups (image-left brow = the subject's right brow). The nose points are the
# nearest MediaPipe landmarks to 2d106's 74, 77, 83, 86; each brow uses its whole
# contour, whose centre is closer to 2d106's brow centre than any 5-point subset
# (see the README's verification table).
MP_BROW_IMAGE_LEFT = [70, 63, 105, 66, 107, 46, 53, 52, 65, 55]
MP_BROW_IMAGE_RIGHT = [300, 293, 334, 296, 336, 276, 283, 282, 295, 285]
MP_NOSE = [5, 129, 358, 4]


def alignment_points(landmarks_xy: np.ndarray) -> np.ndarray:
    """(3, 2) brow centres and nose centre, rounded to pixels as upstream does."""
    return np.round([landmarks_xy[MP_BROW_IMAGE_LEFT].mean(0), landmarks_xy[MP_BROW_IMAGE_RIGHT].mean(0),
                     landmarks_xy[MP_NOSE].mean(0)])


def similarity_from_points(points: np.ndarray, template: np.ndarray = FACE_TEMPLATE,
                           p_bias: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """2x3 similarity mapping ``points`` onto ``template`` (both standardized by
    their overall std, then Procrustes rotation), with LatentSync's temporal
    smoothing: the nose's residual in standardized units, averaged over frames
    (0.2 previous + 0.8 current), is added to the translation."""
    p1, p2 = points.astype(np.float32), template.astype(np.float32)
    c1, c2 = p1.mean(0), p2.mean(0)
    s1, s2 = np.std(p1 - c1, ddof=1), np.std(p2 - c2, ddof=1)  # torch.std is unbiased
    n1, n2 = (p1 - c1) / s1, (p2 - c2) / s2
    u, _, vt = np.linalg.svd(n1.T @ n2)
    v = vt.T
    r = v @ u.T
    if np.linalg.det(r) < 0:
        v[:, -1] = -v[:, -1]
        r = v @ u.T
    m = np.concatenate([(s2 / s1) * r, (c2 - (s2 / s1) * r @ c1)[:, None]], axis=1)
    bias = n2[2] - n1[2]
    bias = bias if p_bias is None else p_bias * 0.2 + bias * 0.8
    m[:, 2] += bias
    return m, bias


def align_faces(frames: np.ndarray, landmarks_xy: np.ndarray) -> np.ndarray:
    """Aligned RESOLUTION x RESOLUTION faces (T, H, W, 3) RGB uint8."""
    import cv2

    out, p_bias = [], None
    for frame, lm in zip(frames, landmarks_xy):
        m, p_bias = similarity_from_points(alignment_points(lm), p_bias=p_bias)
        face = cv2.warpAffine(frame, m, FACE_SIZE, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(127, 127, 127))
        out.append(cv2.resize(face, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_LANCZOS4))
    return np.stack(out)


# ---- detector -------------------------------------------------------------------


class StableSyncNetDetector:
    """Probability that a talking face's audio doesn't belong to it (lip-sync)."""

    def __init__(self, net: torch.nn.Module, device: str = "cpu", max_offset: int = MAX_OFFSET):
        self.name = "stablesyncnet"
        self.threshold = 0.5
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()
        self.max_offset = max_offset

    @torch.inference_mode()
    def sync_probabilities(self, faces: np.ndarray, wav: np.ndarray, offsets=None) -> tuple[np.ndarray, list[int]]:
        """(windows, offsets) cosine similarities for non-overlapping 16-frame
        windows of aligned ``faces`` (T, 256, 256, 3) RGB and the [-1, 1] ``wav``
        from the same first frame, with the audio shifted by each offset (frames)."""
        offsets = list(range(-self.max_offset, self.max_offset + 1)) if offsets is None else list(offsets)
        mel = torch.from_numpy(melspectrogram(wav))
        x = torch.from_numpy(faces).float().div(255).sub(0.5).div(0.5).permute(0, 3, 1, 2)  # T, C, H, W
        starts = [s for s in range(0, len(faces) - NUM_FRAMES + 1, NUM_FRAMES)
                  if all(0 <= int(80.0 * (s + o) / FPS) and int(80.0 * (s + o) / FPS) + MEL_WINDOW <= mel.shape[1]
                         for o in offsets)]
        if not starts:
            raise ValueError(f"Need at least {NUM_FRAMES + 2 * self.max_offset} frames of face and audio")
        sims = np.zeros((len(starts), len(offsets)))
        for b in range(0, len(starts), BATCH):
            chunk = starts[b : b + BATCH]
            video = torch.stack([x[s : s + NUM_FRAMES].reshape(-1, RESOLUTION, RESOLUTION)[:, RESOLUTION // 2 :]
                                 for s in chunk]).to(self.device)  # (b, 48, 128, 256): lower half
            vis = self.net.visual_encoder(video).flatten(1)
            vis = torch.nn.functional.normalize(vis, dim=1)
            for j, o in enumerate(offsets):
                audio = torch.stack([mel[:, int(80.0 * (s + o) / FPS) : int(80.0 * (s + o) / FPS) + MEL_WINDOW]
                                     for s in chunk])[:, None].to(self.device)
                aud = torch.nn.functional.normalize(self.net.audio_encoder(audio).flatten(1), dim=1)
                sims[b : b + len(chunk), j] = torch.nn.functional.cosine_similarity(vis, aud).cpu().numpy()
        return sims, offsets

    def analyze(self, frames: np.ndarray, wav: np.ndarray, track=None) -> dict:
        """``frames`` RGB uint8 at 25 fps, ``wav`` [-1, 1] at 16 kHz, same start."""
        from forensix.physics.face.tracker import track_face

        track = track if track is not None else track_face(frames, FPS)
        if track.valid.mean() < 0.9:
            raise ValueError("Face tracked in under 90% of frames")
        from forensix.physics.motion import _interpolate

        landmarks = _interpolate(track.landmarks, track.valid)[:, :, :2]
        faces = align_faces(frames, landmarks)
        sims, offsets = self.sync_probabilities(faces, wav)
        mean = sims.mean(0)
        best = int(np.argmax(mean))
        return {
            "sync_probability": round(float(mean[best]), 4),
            "offset_frames": offsets[best],
            "sync_probability_at_zero": round(float(mean[offsets.index(0)]), 4),
            "sync_by_offset": {o: round(float(v), 4) for o, v in zip(offsets, mean)},
            "window_sync_probability": np.round(sims[:, best], 4).tolist(),
            "window_s": NUM_FRAMES / FPS,
            "n_windows": int(len(sims)),
        }

    def predict(self, video) -> Prediction:
        from forensix.audiovisual.io import read_av

        frames, audio = read_av(video, FPS, SAMPLE_RATE)
        n = min(len(frames), len(audio) // (SAMPLE_RATE // FPS))
        frames, wav = frames[:n], audio[: n * (SAMPLE_RATE // FPS)] / 32768.0
        result = self.analyze(frames, wav)
        return Prediction(model=self.name, fake_prob=1.0 - result["sync_probability"], threshold=self.threshold,
                          details=result)

    def predict_batch(self, videos) -> list[Prediction]:
        return [self.predict(v) for v in videos]


def build(weights_path: Path, device: str = "cpu") -> StableSyncNetDetector:
    from forensix.audiovisual.stablesyncnet.model import StableSyncNet

    net = StableSyncNet(CONFIG)
    net.load_state_dict(torch_load(weights_path)["state_dict"])
    return StableSyncNetDetector(net, device=device)


def _face_landmarker():
    from forensix.physics.face.tracker import FACE_LANDMARKER

    return FACE_LANDMARKER


register(
    ModelSpec(
        name="stablesyncnet",
        family="audiovisual",
        builder=build,
        weights=WEIGHTS,
        extra_weights=(_face_landmarker(),),
        license="CreativeML OpenRAIL++-M (weights: commercial use allowed, with use-based restrictions "
        "that pass on to your users; among them, no fully automated decisions adversely affecting legal "
        "rights and no use for justice, law enforcement, immigration or asylum processes). Code: "
        "Apache-2.0 (LatentSync). Preprocessing: ForensiX + MediaPipe "
        "FaceLandmarker (Apache-2.0). Trained on VoxCeleb2 and HDTF",
        source="https://github.com/bytedance/LatentSync/tree/a229c3948406bc2cf6eaf4873e662e70c6a04746",
        paper="https://arxiv.org/abs/2412.09262",
        description="LatentSync's lip-sync expert (401M): the probability that a talking face's audio "
        "doesn't belong to it, from 16-frame windows of the aligned lower face and the mel "
        "spectrogram, at the best offset within +-6 frames. Flags dubbing, voice swaps and "
        "out-of-sync lip-sync generation.",
    )
)
