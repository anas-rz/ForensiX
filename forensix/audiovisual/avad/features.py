"""Log-mel filterbank features for AVAD's audio encoder (16 kHz, 512-point FFT,
20 ms Hann window, 10 ms hop, 80 Slaney mel bands). Equivalent to AVAD's
``wav2filterbanks`` (torch.stft magnitude + librosa's default mel filters),
written without librosa."""

import numpy as np
import torch

SAMPLE_RATE, N_FFT, WIN_LENGTH, HOP_LENGTH, N_MELS = 16000, 512, 320, 160, 80


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    # Slaney: linear below 1 kHz, logarithmic above
    f = np.asarray(f, dtype=np.float64)
    mel = f / (200.0 / 3)
    log_region = f >= 1000.0
    mel[log_region] = 15.0 + np.log(f[log_region] / 1000.0) / (np.log(6.4) / 27.0)
    return mel


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    f = m * (200.0 / 3)
    log_region = m >= 15.0
    f[log_region] = 1000.0 * np.exp((np.log(6.4) / 27.0) * (m[log_region] - 15.0))
    return f


def mel_filterbank(sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS) -> np.ndarray:
    """Slaney-style, area-normalized triangular filters, shape (n_mels, n_fft // 2 + 1)."""
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel_pts = _mel_to_hz(np.linspace(_hz_to_mel(np.array([0.0]))[0], _hz_to_mel(np.array([sr / 2]))[0], n_mels + 2))
    fdiff = np.diff(mel_pts)
    ramps = mel_pts[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_pts[2 : n_mels + 2] - mel_pts[:n_mels]))[:, None]
    return weights.astype(np.float32)


_MEL = torch.from_numpy(mel_filterbank())


def log_mel(wav: torch.Tensor) -> torch.Tensor:
    """(batch, samples) -> (batch, frames, 80); the last STFT frame is dropped, as upstream."""
    window = torch.hann_window(WIN_LENGTH, device=wav.device)
    spec = torch.stft(wav, N_FFT, HOP_LENGTH, WIN_LENGTH, window=window, center=True,
                      pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
    mag = spec[:, :, :-1].abs()
    return torch.log(_MEL.to(wav.device) @ mag + 1e-20).transpose(1, 2).contiguous()
