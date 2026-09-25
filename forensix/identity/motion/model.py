"""Temporal ID network: a window of facial-motion features -> an identity embedding.

Following the idea of ID-Reveal's Temporal ID Network (Cozzolino et al., ICCV
2021; implemented independently here, as upstream has no license): a stack of
dilated residual 1-D convolutions over time, mean and standard-deviation
pooling, and a linear projection to a unit-norm embedding. Trained with a
supervised contrastive loss so that windows of the same person, from different
videos, lie close together.
"""

import torch
import torch.nn.functional as F
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, 1)
        self.norm1, self.norm2 = nn.GroupNorm(8, ch), nn.GroupNorm(8, ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.conv1(F.gelu(self.norm1(x)))
        h = self.conv2(self.drop(F.gelu(self.norm2(h))))
        return x + h


class MotionEncoder(nn.Module):
    def __init__(self, in_dim: int = 64, channels: int = 256, dilations=(1, 2, 4, 8, 16, 1), dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.config = {"in_dim": in_dim, "channels": channels, "dilations": list(dilations), "dim": dim,
                       "dropout": dropout}
        self.inp = nn.Conv1d(in_dim, channels, 1)
        self.blocks = nn.Sequential(*[ResBlock(channels, d, dropout) for d in dilations])
        self.norm = nn.GroupNorm(8, channels)
        self.out = nn.Linear(2 * channels, dim)
        # feature standardization (fitted on the training data, saved with the model)
        self.register_buffer("mean", torch.zeros(in_dim))
        self.register_buffer("std", torch.ones(in_dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """``x`` (B, T, in_dim) raw features; ``mask`` (B, T) True on valid frames.
        Returns (B, dim) unit-norm embeddings."""
        x = (x - self.mean) / self.std
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        x = x * mask[..., None]
        h = F.gelu(self.norm(self.blocks(self.inp(x.transpose(1, 2)))))  # (B, C, T)
        m = mask[:, None, :].float()
        n = m.sum(-1).clamp(min=1.0)
        mean = (h * m).sum(-1) / n
        std = (((h - mean[..., None]) ** 2 * m).sum(-1) / n).clamp(min=1e-6).sqrt()
        return F.normalize(self.out(torch.cat([mean, std], -1)), dim=-1)


def supcon_loss(emb: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al., 2020) over unit-norm embeddings."""
    sim = emb @ emb.T / temperature
    self_mask = torch.eye(len(emb), dtype=torch.bool, device=emb.device)
    sim = sim.masked_fill(self_mask, -1e9)
    pos = (labels[:, None] == labels[None, :]) & ~self_mask
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    has_pos = pos.any(1)
    return -((log_prob * pos).sum(1)[has_pos] / pos.sum(1)[has_pos]).mean()
