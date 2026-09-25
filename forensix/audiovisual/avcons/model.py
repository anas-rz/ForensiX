"""AV consistency model: a temporal transformer over SyncNet window embeddings.

Per window t the input is the lip embedding v_t, the audio embedding a_t, and
the sync-distance profile d_t(k) = ||v_t - a_{t+k}|| for k = -7..+7 (SyncNet's
offset search, done locally). The profile is centred on the clip's median
distance, as SyncNet's confidence is. A transformer relates the windows over
time. It outputs a logit per window (localization) and a clip logit
(attention pooling).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

MAX_OFFSET = 7


def distance_profile(v: torch.Tensor, a: torch.Tensor, max_offset: int = MAX_OFFSET,
                     mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, T, 2K+1) distances ||v_t - a_{t+k}|| and a validity mask: False where
    window t or t + k falls outside the sequence or on padding (``mask`` (B, T),
    True on real windows)."""
    b, t, _ = v.shape
    if mask is None:
        mask = torch.ones(b, t, dtype=torch.bool, device=v.device)
    pad = F.pad(a, (0, 0, max_offset, max_offset))
    pad_mask = F.pad(mask, (max_offset, max_offset), value=False)
    dists, valid = [], []
    for k in range(-max_offset, max_offset + 1):
        other = pad[:, max_offset + k : max_offset + k + t]
        dists.append(torch.linalg.vector_norm(v - other, dim=-1))
        valid.append(mask & pad_mask[:, max_offset + k : max_offset + k + t])
    return torch.stack(dists, -1), torch.stack(valid, -1)


def _sinusoid(t: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(t, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(t, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class AVConsistency(nn.Module):
    def __init__(self, emb_dim: int = 1024, dim: int = 256, layers: int = 4, heads: int = 4,
                 max_offset: int = MAX_OFFSET, dropout: float = 0.1):
        super().__init__()
        self.config = {"emb_dim": emb_dim, "dim": dim, "layers": layers, "heads": heads,
                       "max_offset": max_offset, "dropout": dropout}
        n_off = 2 * max_offset + 1
        self.max_offset = max_offset
        self.v_proj = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, dim))
        self.a_proj = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, dim))
        self.profile_proj = nn.Sequential(nn.Linear(2 * n_off, dim), nn.GELU(), nn.Linear(dim, dim))
        self.fuse = nn.Linear(3 * dim, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.window_head = nn.Linear(dim, 1)
        self.pool = nn.Linear(dim, 1)
        self.clip_head = nn.Linear(dim, 1)

    def forward(self, v: torch.Tensor, a: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """``v``, ``a``: (B, T, emb_dim); ``mask``: (B, T) True on real windows.
        Returns (clip logits (B,), window logits (B, T))."""
        b, t, _ = v.shape
        if mask is None:
            mask = torch.ones(b, t, dtype=torch.bool, device=v.device)
        dist, valid = distance_profile(v, a, self.max_offset, mask)
        # centre on the clip's median distance (SyncNet's confidence is median - min)
        med = torch.stack([d[m].median() if m.any() else d.new_tensor(0.0) for d, m in zip(dist, valid)])
        centred = torch.where(valid, dist - med[:, None, None], torch.zeros_like(dist))
        profile = torch.cat([centred, valid.float()], -1)
        h = self.fuse(torch.cat([self.v_proj(v), self.a_proj(a), self.profile_proj(profile)], -1))
        h = h + _sinusoid(t, h.shape[-1], v.device)
        h = self.norm(self.encoder(h, src_key_padding_mask=~mask))
        window = self.window_head(h).squeeze(-1)
        weights = self.pool(h).squeeze(-1).masked_fill(~mask, float("-inf")).softmax(-1)
        clip = self.clip_head((weights[..., None] * h).sum(1)).squeeze(-1)
        return clip, window
