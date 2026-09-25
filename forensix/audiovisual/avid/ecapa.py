# ECAPA-TDNN speaker embeddings, from SpeechBrain 1.1.1 (Apache-2.0, see LICENSE-speechbrain):
#   speechbrain/lobes/models/ECAPA_TDNN.py (network, as upstream, docstrings removed),
#   speechbrain/lobes/features.py Fbank + processing/features.py STFT, spectral_magnitude,
#   Filterbank and InputNormalization (sentence mean), reduced to what
#   speechbrain/spkrec-ecapa-voxceleb uses.
# Copyright SpeechBrain contributors. Modifications by ForensiX: the nnet.CNN.Conv1d,
# nnet.normalization.BatchNorm1d and dataio.length_to_mask helpers are reimplemented here
# (same parameter names and numerics), so speechbrain isn't a dependency.
#
# Desplanques, Thienpondt & Demuynck, "ECAPA-TDNN: Emphasized Channel Attention,
# Propagation and Aggregation in TDNN Based Speaker Verification", Interspeech 2020.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def length_to_mask(length, max_len=None, dtype=None, device=None):
    if max_len is None:
        max_len = length.max().long().item()
    mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    return torch.as_tensor(mask, dtype=dtype or length.dtype, device=device)


class _SBConv1d(nn.Module):
    """speechbrain.nnet.CNN.Conv1d with padding="same" (reflect), channels first."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.kernel_size, self.dilation = kernel_size, dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, groups=groups)

    def forward(self, x):
        l_in = x.shape[-1]
        l_out = math.floor((l_in - self.dilation * (self.kernel_size - 1) - 1) / 1) + 1
        pad = math.floor((l_in - l_out) / 2)
        return self.conv(F.pad(x, (pad, pad), mode="reflect"))


class _SBBatchNorm1d(nn.Module):
    """speechbrain.nnet.normalization.BatchNorm1d (eps 1e-5, momentum 0.1), channels first."""

    def __init__(self, input_size):
        super().__init__()
        self.norm = nn.BatchNorm1d(input_size, eps=1e-05, momentum=0.1)

    def forward(self, x):
        return self.norm(x)


Conv1d = _SBConv1d
BatchNorm1d = _SBBatchNorm1d


class TDNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, activation=nn.ReLU, groups=1, dropout=0.0):
        super().__init__()
        self.conv = Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, groups=groups)
        self.activation = activation()
        self.norm = BatchNorm1d(input_size=out_channels)
        self.dropout = nn.Dropout1d(p=dropout)

    def forward(self, x):
        return self.dropout(self.norm(self.activation(self.conv(x))))


class Res2NetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1, dropout=0.0):
        super().__init__()
        assert in_channels % scale == 0
        assert out_channels % scale == 0
        in_channel = in_channels // scale
        hidden_channel = out_channels // scale
        self.blocks = nn.ModuleList(
            [TDNNBlock(in_channel, hidden_channel, kernel_size=kernel_size, dilation=dilation, dropout=dropout)
             for i in range(scale - 1)]
        )
        self.scale = scale

    def forward(self, x):
        y = []
        for i, x_i in enumerate(torch.chunk(x, self.scale, dim=1)):
            if i == 0:
                y_i = x_i
            elif i == 1:
                y_i = self.blocks[i - 1](x_i)
            else:
                y_i = self.blocks[i - 1](x_i + y_i)
            y.append(y_i)
        return torch.cat(y, dim=1)


class SEBlock(nn.Module):
    def __init__(self, in_channels, se_channels, out_channels):
        super().__init__()
        self.conv1 = Conv1d(in_channels, se_channels, 1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = Conv1d(se_channels, out_channels, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, lengths=None):
        L = x.shape[-1]
        if lengths is not None:
            mask = length_to_mask(lengths * L, max_len=L, device=x.device)
            mask = mask.unsqueeze(1)
            total = mask.sum(dim=2, keepdim=True)
            s = (x * mask).sum(dim=2, keepdim=True) / total
        else:
            s = x.mean(dim=2, keepdim=True)
        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))
        return s * x


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels, attention_channels=128, global_context=True):
        super().__init__()
        self.eps = 1e-12
        self.global_context = global_context
        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = Conv1d(attention_channels, channels, 1)

    def forward(self, x, lengths=None):
        L = x.shape[-1]

        def _compute_statistics(x, m, dim=2, eps=self.eps):
            mean = (m * x).sum(dim)
            std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps))
            return mean, std

        if lengths is None:
            lengths = torch.ones(x.shape[0], device=x.device)
        mask = length_to_mask(lengths * L, max_len=L, device=x.device)
        mask = mask.unsqueeze(1)
        if self.global_context:
            total = mask.sum(dim=2, keepdim=True).float()
            mean, std = _compute_statistics(x, mask / total)
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x
        attn = self.conv(self.tanh(self.tdnn(attn)))
        attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=2)
        mean, std = _compute_statistics(x, attn)
        pooled_stats = torch.cat((mean, std), dim=1)
        return pooled_stats.unsqueeze(2)


class SERes2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, res2net_scale=8, se_channels=128, kernel_size=1, dilation=1,
                 activation=torch.nn.ReLU, groups=1, dropout=0.0):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TDNNBlock(in_channels, out_channels, kernel_size=1, dilation=1, activation=activation,
                               groups=groups, dropout=dropout)
        self.res2net_block = Res2NetBlock(out_channels, out_channels, res2net_scale, kernel_size, dilation)
        self.tdnn2 = TDNNBlock(out_channels, out_channels, kernel_size=1, dilation=1, activation=activation,
                               groups=groups, dropout=dropout)
        self.se_block = SEBlock(out_channels, se_channels, out_channels)
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = Conv1d(in_channels, out_channels, 1)

    def forward(self, x, lengths=None):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)
        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x, lengths)
        return x + residual


class ECAPA_TDNN(torch.nn.Module):
    def __init__(self, input_size, lin_neurons=192, activation=torch.nn.ReLU,
                 channels=(512, 512, 512, 512, 1536), kernel_sizes=(5, 3, 3, 3, 1), dilations=(1, 2, 3, 4, 1),
                 attention_channels=128, res2net_scale=8, se_channels=128, global_context=True,
                 groups=(1, 1, 1, 1, 1), dropout=0.0):
        super().__init__()
        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(dilations)
        self.channels = channels
        self.blocks = nn.ModuleList()
        self.blocks.append(TDNNBlock(input_size, channels[0], kernel_sizes[0], dilations[0], activation, groups[0], dropout))
        for i in range(1, len(channels) - 1):
            self.blocks.append(SERes2NetBlock(channels[i - 1], channels[i], res2net_scale=res2net_scale,
                                              se_channels=se_channels, kernel_size=kernel_sizes[i],
                                              dilation=dilations[i], activation=activation, groups=groups[i],
                                              dropout=dropout))
        self.mfa = TDNNBlock(channels[-2] * (len(channels) - 2), channels[-1], kernel_sizes[-1], dilations[-1],
                             activation, groups=groups[-1], dropout=dropout)
        self.asp = AttentiveStatisticsPooling(channels[-1], attention_channels=attention_channels,
                                              global_context=global_context)
        self.asp_bn = BatchNorm1d(input_size=channels[-1] * 2)
        self.fc = Conv1d(channels[-1] * 2, lin_neurons, 1)

    def forward(self, x, lengths=None):
        x = x.transpose(1, 2)
        xl = []
        for layer in self.blocks:
            if isinstance(layer, TDNNBlock):
                x = layer(x)
            else:
                x = layer(x, lengths=lengths)
            xl.append(x)
        x = torch.cat(xl[1:], dim=1)
        x = self.mfa(x)
        x = self.asp(x, lengths=lengths)
        x = self.asp_bn(x)
        x = self.fc(x)
        return x.transpose(1, 2)


class Fbank(nn.Module):
    """speechbrain.lobes.features.Fbank(n_mels=80): 25 ms Hamming STFT every 10 ms
    (n_fft 400, centred, zero padding), power spectrum, triangular mel filters
    0-8000 Hz, 10 log10 with amin 1e-10 and an 80 dB floor below each utterance's max."""

    def __init__(self, n_mels=80, sample_rate=16000, n_fft=400, win_ms=25, hop_ms=10):
        super().__init__()
        self.n_fft = n_fft
        self.win = int(round(sample_rate / 1000.0 * win_ms))
        self.hop = int(round(sample_rate / 1000.0 * hop_ms))
        self.register_buffer("window", torch.hamming_window(self.win), persistent=False)
        to_mel = lambda hz: 2595 * math.log10(1 + hz / 700)  # noqa: E731
        mel = torch.linspace(to_mel(0), to_mel(sample_rate // 2), n_mels + 2)
        hz = 700 * (10 ** (mel / 2595) - 1)
        band = (hz[1:] - hz[:-1])[:-1]
        f_central = hz[1:-1]
        all_freqs = torch.linspace(0, sample_rate // 2, n_fft // 2 + 1).repeat(n_mels, 1)
        f_central_mat = f_central.repeat(all_freqs.shape[1], 1).transpose(0, 1)
        band_mat = band.repeat(all_freqs.shape[1], 1).transpose(0, 1)
        slope = (all_freqs - f_central_mat) / band_mat
        fb = torch.max(torch.zeros(1), torch.min(slope + 1.0, -slope + 1.0)).transpose(0, 1)
        self.register_buffer("fbank_matrix", fb, persistent=False)

    def forward(self, wav):
        stft = torch.stft(wav, self.n_fft, self.hop, self.win, self.window.to(wav.device), True, "constant",
                          False, True, return_complex=True)
        mag = torch.view_as_real(stft).transpose(2, 1).pow(2).sum(-1)  # (B, frames, freq)
        fbanks = torch.matmul(mag, self.fbank_matrix)
        x_db = 10 * torch.log10(torch.clamp(fbanks, min=1e-10))
        x_db -= 10 * math.log10(max(1e-10, 1.0))
        new_max = x_db.amax(dim=(-2, -1)) - 80.0
        return torch.max(x_db, new_max.view(x_db.shape[0], 1, 1))


class SpeakerEncoder(nn.Module):
    """speechbrain/spkrec-ecapa-voxceleb ``encode_batch`` (normalize=False):
    Fbank, sentence mean normalization, ECAPA-TDNN. 16 kHz [-1, 1] waveforms
    (B, samples) -> (B, 192) embeddings."""

    def __init__(self):
        super().__init__()
        self.fbank = Fbank(n_mels=80)
        self.embedding_model = ECAPA_TDNN(80, channels=[1024, 1024, 1024, 1024, 3072], kernel_sizes=[5, 3, 3, 3, 1],
                                          dilations=[1, 2, 3, 4, 1], attention_channels=128, lin_neurons=192)

    @torch.no_grad()
    def forward(self, wavs: torch.Tensor) -> torch.Tensor:
        feats = self.fbank(wavs.float())
        feats = feats - feats.mean(dim=1, keepdim=True)  # InputNormalization(norm_type="sentence", std_norm=False)
        return self.embedding_model(feats).squeeze(1)
