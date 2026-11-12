"""Lightweight feature extraction for person-specialized deepfake detection.

All extractors are optimized for fast CPU execution without GPU dependencies:
- Image: YuNet + SFace (128-d) + spatial/frequency artifact statistics (64-d) = 192-d.
- Audio: log-mel filterbank + spectral dynamics + pitch/harmonics statistics = 128-d.
- Video: temporally aggregated frame representations + motion dynamics = 192-d.
- Audiovisual: fused visual + acoustic cross-modal representation = 320-d.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.signal import correlate, spectrogram

from forensix.audio.io import load_audio
from forensix.hub import fetch
from forensix.identity.face import SFACE, YUNET
from forensix.identity.media import is_image, sample_frames
from forensix.image.io import load_image

DIM_IMAGE = 192
DIM_VIDEO = 192
DIM_AUDIO = 128
DIM_AUDIOVISUAL = 320

_RECOGNIZER = None
_DETECTOR = None


def _get_face_tools() -> tuple[Any, Any]:
    global _RECOGNIZER, _DETECTOR
    if _RECOGNIZER is None or _DETECTOR is None:
        sface_p = fetch(SFACE)
        yunet_p = fetch(YUNET)
        _RECOGNIZER = cv2.FaceRecognizerSF.create(str(sface_p), "")
        _DETECTOR = cv2.FaceDetectorYN.create(str(yunet_p), "", (320, 320), 0.8, 0.3, 5000)
    return _RECOGNIZER, _DETECTOR


def _extract_frequency_artifacts(img_rgb: np.ndarray) -> np.ndarray:
    """Extract 64-d frequency and gradient artifact features from an RGB image.

    Deepfake synthesis (GANs, diffusion, face swaps) leaves distinct high-frequency
    traces, blurring at blend boundaries, and unnatural spectral decay.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    if h < 16 or w < 16:
        gray = cv2.resize(gray, (64, 64))
        h, w = 64, 64

    # 1. 2D DCT / Fourier radial energy distribution (32 bins)
    dsize = (64, 64)
    small_gray = cv2.resize(gray, dsize)
    dct = cv2.dct(small_gray / 255.0)
    # Radial rings from low to high frequency
    y, x = np.ogrid[:64, :64]
    r = np.sqrt(x**2 + y**2).astype(int)
    radial_energy = np.zeros(32, dtype=np.float32)
    for i in range(32):
        mask = (r >= i * 2) & (r < (i + 1) * 2)
        if mask.any():
            radial_energy[i] = np.mean(np.abs(dct[mask]))

    # 2. Laplacian edge & gradient distribution (16 stats)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)

    grad_stats = np.array([
        np.mean(lap), np.std(lap), np.percentile(lap, 90), np.percentile(lap, 10),
        np.mean(grad_mag), np.std(grad_mag), np.max(grad_mag), np.median(grad_mag),
        np.mean(np.abs(lap)), np.std(np.abs(lap)),
        np.percentile(grad_mag, 75), np.percentile(grad_mag, 25),
        float(np.mean(grad_mag > 20)), float(np.mean(grad_mag > 50)),
        float(np.mean(np.abs(lap) > 15)), float(np.mean(np.abs(lap) > 30)),
    ], dtype=np.float32)

    # 3. Color channel residual & saturation stats (16 stats)
    blurred = cv2.GaussianBlur(img_rgb, (5, 5), 0).astype(np.float32)
    residual = np.abs(img_rgb.astype(np.float32) - blurred)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

    color_stats = np.array([
        np.mean(residual[:, :, 0]), np.std(residual[:, :, 0]),
        np.mean(residual[:, :, 1]), np.std(residual[:, :, 1]),
        np.mean(residual[:, :, 2]), np.std(residual[:, :, 2]),
        np.mean(hsv[:, :, 1]), np.std(hsv[:, :, 1]),  # saturation
        np.mean(hsv[:, :, 2]), np.std(hsv[:, :, 2]),  # value
        np.mean(residual), np.std(residual), np.max(residual),
        np.percentile(residual, 95),
        float(np.mean(hsv[:, :, 1] < 20)),  # washed-out pixel fraction
        float(np.mean(hsv[:, :, 1] > 200)),  # hyper-saturated pixel fraction
    ], dtype=np.float32)

    feat = np.concatenate([radial_energy, grad_stats, color_stats])
    # Normalize
    norm = np.linalg.norm(feat)
    return (feat / (norm + 1e-6)).astype(np.float32)


def extract_image_features(media_input: str | PathLike | np.ndarray) -> np.ndarray:
    """Extract a 192-d feature vector from an image.

    Returns:
        np.ndarray of shape (192,), float32.
    """
    if isinstance(media_input, np.ndarray):
        img_rgb = media_input.copy()
        if img_rgb.dtype != np.uint8:
            img_rgb = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)
    else:
        pil_img = load_image(media_input)
        img_rgb = np.array(pil_img)

    bgr = np.ascontiguousarray(img_rgb[:, :, ::-1])
    recognizer, detector = _get_face_tools()

    detector.setInputSize((bgr.shape[1], bgr.shape[0]))
    _, faces = detector.detect(bgr)

    if faces is not None and len(faces) > 0:
        # Choose largest face
        areas = faces[:, 2] * faces[:, 3]
        best_face = faces[int(np.argmax(areas))]
        aligned = recognizer.alignCrop(bgr, best_face[:-1])
        sface_emb = recognizer.feature(aligned).ravel().astype(np.float32)
        # Unit normalize face embedding
        norm = np.linalg.norm(sface_emb)
        sface_emb = sface_emb / (norm + 1e-6)
        face_crop = aligned[:, :, ::-1]
        freq_feat = _extract_frequency_artifacts(face_crop)
    else:
        # Fallback for images without a clear detected face: compute spatial & texture representation
        small = cv2.resize(img_rgb, (112, 112))
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        hist_rgb = cv2.calcHist([small], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256]).ravel()
        hist_hsv = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256]).ravel()
        fallback_emb = np.concatenate([hist_rgb, hist_hsv])[:128].astype(np.float32)
        if len(fallback_emb) < 128:
            fallback_emb = np.pad(fallback_emb, (0, 128 - len(fallback_emb)))
        norm = np.linalg.norm(fallback_emb)
        sface_emb = fallback_emb / (norm + 1e-6)
        freq_feat = _extract_frequency_artifacts(small)

    feat = np.concatenate([sface_emb, freq_feat]).astype(np.float32)
    return feat


def extract_audio_features(media_input: str | PathLike | np.ndarray, target_sr: int = 16000) -> np.ndarray:
    """Extract a 128-d acoustic feature vector from an audio recording.

    Returns:
        np.ndarray of shape (128,), float32.
    """
    if isinstance(media_input, np.ndarray):
        waveform = media_input.astype(np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
    else:
        waveform = load_audio(media_input, target_sr=target_sr)

    if len(waveform) < target_sr * 0.1:
        # Pad short audio
        waveform = np.pad(waveform, (0, int(target_sr * 0.1) - len(waveform)))

    # Compute spectrogram
    nperseg = min(512, len(waveform))
    frequencies, times, Sxx = spectrogram(waveform, fs=target_sr, nperseg=nperseg, noverlap=nperseg // 2)
    Sxx = np.maximum(Sxx, 1e-10)
    log_Sxx = np.log10(Sxx)

    # 1. Mel-scale filterbank energy summary (40 bands x 2 stats = 80 stats)
    n_freqs = len(frequencies)
    band_size = max(1, n_freqs // 40)
    mel_means = []
    mel_stds = []
    for b in range(40):
        start = b * band_size
        end = min(n_freqs, (b + 1) * band_size)
        if start < n_freqs:
            band_data = log_Sxx[start:end, :]
            mel_means.append(float(np.mean(band_data)))
            mel_stds.append(float(np.std(band_data)))
        else:
            mel_means.append(0.0)
            mel_stds.append(0.0)
    mel_means_arr = np.array(mel_means, dtype=np.float32)
    mel_means_arr = mel_means_arr - np.mean(mel_means_arr)
    mel_stds_arr = np.array(mel_stds, dtype=np.float32)
    mel_stds_arr = mel_stds_arr - np.mean(mel_stds_arr)
    mel_stats = np.concatenate([mel_means_arr, mel_stds_arr]).astype(np.float32)

    # 2. Spectral centroid, rolloff, flatness, zero crossings (16 stats)
    # Spectral centroid per time frame
    norm_s = Sxx / (np.sum(Sxx, axis=0, keepdims=True) + 1e-10)
    centroids = np.sum(frequencies[:, None] * norm_s, axis=0)
    flatness = np.exp(np.mean(np.log(Sxx), axis=0)) / (np.mean(Sxx, axis=0) + 1e-10)
    zcr = np.mean(np.abs(np.diff(np.sign(waveform))))

    spectral_stats = np.array([
        np.mean(centroids), np.std(centroids), np.max(centroids), np.min(centroids),
        np.mean(flatness), np.std(flatness), np.percentile(flatness, 90), np.percentile(flatness, 10),
        zcr, np.std(waveform), np.mean(np.abs(waveform)), np.max(np.abs(waveform)),
        float(np.mean(centroids > 2000)), float(np.mean(flatness < 0.1)),
        float(np.percentile(centroids, 75)), float(np.percentile(centroids, 25)),
    ], dtype=np.float32)

    # 3. Autocorrelation-based pitch / harmonic statistics (16 stats)
    max_lag = min(len(waveform) // 2, int(target_sr / 50))  # down to 50 Hz
    min_lag = max(1, int(target_sr / 500))  # up to 500 Hz
    head = waveform[: min(len(waveform), target_sr * 2)]
    corr = correlate(head, head, mode="full", method="fft")  # same as np.correlate, O(n log n)
    mid = len(corr) // 2
    r_xx = corr[mid + min_lag : mid + max_lag]
    if len(r_xx) >= 16:
        # Sample 16 lag energies
        indices = np.linspace(0, len(r_xx) - 1, 16).astype(int)
        pitch_stats = r_xx[indices].astype(np.float32)
        pitch_stats = pitch_stats / (np.max(np.abs(pitch_stats)) + 1e-6)
    else:
        pitch_stats = np.zeros(16, dtype=np.float32)

    # 4. Spectral flux and delta dynamics (16 stats)
    flux = np.sqrt(np.sum(np.diff(log_Sxx, axis=1)**2, axis=0)) if log_Sxx.shape[1] > 1 else np.zeros(1)
    flux_stats = np.array([
        np.mean(flux), np.std(flux), np.max(flux), np.min(flux),
        np.percentile(flux, 80) if len(flux) > 1 else 0.0,
        np.percentile(flux, 20) if len(flux) > 1 else 0.0,
        float(np.mean(flux > np.mean(flux))),
        float(np.median(flux)),
        # Frequency band energy ratios
        np.mean(log_Sxx[:n_freqs // 4]), np.mean(log_Sxx[n_freqs // 4 : n_freqs // 2]),
        np.mean(log_Sxx[n_freqs // 2 : 3 * n_freqs // 4]), np.mean(log_Sxx[3 * n_freqs // 4 :]),
        np.std(log_Sxx[:n_freqs // 4]), np.std(log_Sxx[n_freqs // 4 : n_freqs // 2]),
        np.std(log_Sxx[n_freqs // 2 : 3 * n_freqs // 4]), np.std(log_Sxx[3 * n_freqs // 4 :]),
    ], dtype=np.float32)

    feat = np.concatenate([mel_stats, spectral_stats, pitch_stats, flux_stats]).astype(np.float32)
    norm = np.linalg.norm(feat)
    return (feat / (norm + 1e-6)).astype(np.float32)


def extract_video_features(
    media_input: str | PathLike,
    sample_fps: float = 2.0,
    max_frames: int = 60,
) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    """Extract temporal video features.

    Returns:
        (video_feat, frame_feats, frame_times)
        video_feat: shape (192,), float32.
        frame_feats: list of shape (192,) per sampled frame.
        frame_times: timestamps of sampled frames in seconds.
    """
    frames, times = sample_frames(media_input, fps=sample_fps, max_frames=max_frames)
    frame_feats = [extract_image_features(f) for f in frames]
    frame_arr = np.stack(frame_feats)

    # Temporal pooling: mean + temporal change dynamics
    mean_feat = np.mean(frame_arr, axis=0)
    if len(frame_feats) > 1:
        diff = np.diff(frame_arr, axis=0)
        temporal_dyn = np.std(diff, axis=0)
        video_feat = 0.7 * mean_feat + 0.3 * temporal_dyn
    else:
        video_feat = mean_feat

    norm = np.linalg.norm(video_feat)
    video_feat = (video_feat / (norm + 1e-6)).astype(np.float32)
    return video_feat, frame_feats, times


def extract_audiovisual_features(
    media_input: str | PathLike,
    sample_fps: float = 2.0,
    max_frames: int = 60,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract fused 320-d audiovisual feature vector."""
    video_feat, frame_feats, times = extract_video_features(media_input, sample_fps=sample_fps, max_frames=max_frames)
    try:
        audio_feat = extract_audio_features(media_input)
    except Exception:
        # Fallback if container has no readable audio stream
        audio_feat = np.zeros(DIM_AUDIO, dtype=np.float32)

    fused = np.concatenate([video_feat, audio_feat]).astype(np.float32)
    norm = np.linalg.norm(fused)
    fused = (fused / (norm + 1e-6)).astype(np.float32)

    details = {
        "visual_dim": len(video_feat),
        "audio_dim": len(audio_feat),
        "frame_times": times,
        "n_frames": len(frame_feats),
    }
    return fused, details


def extract_features(media_input: Any, modality: str) -> np.ndarray:
    """Unified feature extractor dispatcher."""
    if modality == "image":
        return extract_image_features(media_input)
    if modality == "audio":
        return extract_audio_features(media_input)
    if modality == "video":
        v_feat, _, _ = extract_video_features(media_input)
        return v_feat
    if modality == "audiovisual":
        av_feat, _ = extract_audiovisual_features(media_input)
        return av_feat
    raise ValueError(f"Unknown modality: {modality!r}. Expected image, audio, video, or audiovisual.")


def generate_synthetic_perturbations(
    feat: np.ndarray,
    modality: str,
    count: int = 5,
    seed: int = 42,
    return_impostor_mask: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], list[bool]]:
    """Generate realistic synthetic manipulation / impostor negative features.

    Creates diverse negative samples in feature space:
    1. Identity shift (orthogonal perturbation outside identity sphere) - Impostor
    2. Extreme impostor: independent identity vector - Impostor
    3. High-frequency anomaly (high-frequency energy boost) - Manipulation artifact
    4. Low-pass smoothing (suppressed detail) - Manipulation artifact
    5. Boundary blend perturbation - Manipulation artifact
    """
    rng = np.random.default_rng(seed)
    dim = len(feat)
    negatives = []
    is_impostor = []

    # 1. Identity shift: add orthogonal perturbation
    ortho = rng.standard_normal(dim).astype(np.float32)
    ortho = ortho - np.dot(ortho, feat) * feat
    ortho = ortho / (np.linalg.norm(ortho) + 1e-6)
    shifted = 0.3 * feat + 0.954 * ortho  # Cosine ~ 0.3 with original
    negatives.append((shifted / np.linalg.norm(shifted)).astype(np.float32))
    is_impostor.append(True)

    # 2. Extreme impostor: independent identity vector
    rand_id = rng.standard_normal(dim).astype(np.float32)
    negatives.append((rand_id / np.linalg.norm(rand_id)).astype(np.float32))
    is_impostor.append(True)

    # 3. High-frequency artifact noise
    noisy = feat.copy()
    noisy[dim // 2 :] += rng.normal(0, 0.4, size=dim - dim // 2).astype(np.float32)
    negatives.append((noisy / np.linalg.norm(noisy)).astype(np.float32))
    is_impostor.append(False)

    # 4. Smoothing / compression attenuation
    smoothed = feat.copy()
    smoothed[dim // 2 :] *= 0.15
    negatives.append((smoothed / np.linalg.norm(smoothed)).astype(np.float32))
    is_impostor.append(False)

    # 5. Boundary blend perturbation
    noise_vec = rng.uniform(-1, 1, size=dim).astype(np.float32)
    blended = 0.6 * feat + 0.4 * noise_vec
    negatives.append((blended / np.linalg.norm(blended)).astype(np.float32))
    is_impostor.append(False)

    while len(negatives) < count:
        alpha = rng.uniform(0.2, 0.5)
        extra = (1 - alpha) * feat + alpha * rng.standard_normal(dim).astype(np.float32)
        negatives.append((extra / np.linalg.norm(extra)).astype(np.float32))
        is_impostor.append(False)

    if return_impostor_mask:
        return negatives[:count], is_impostor[:count]
    return negatives[:count]
