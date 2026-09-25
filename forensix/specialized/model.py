"""Teeny tiny neural network architecture for person-specialized deepfake detection.

Designed to be lightweight enough to run and train on any standard CPU in seconds:
- Under 25,000 parameters total (~100 KB footprint).
- Parameter-Efficient Fine-Tuning (PEFT) via Low-Rank Adaptation (LoRA): adapts
  fewer than 2,500 parameters per person.
- Multi-modal: supports image, video, audio, and audiovisual representations.
- Dual-path decision logic:
  1. Artifact detection: catches synthesis glitches, blending boundaries, and frequency anomalies.
  2. Identity verification: measures deviation from the target individual's genuine biometric prototype.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Linear layer with optional Low-Rank Adaptation (LoRA).

    When PEFT is active, the base weight remains frozen and only the low-rank
    decomposition matrices (A and B) receive gradient updates.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 8.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = (alpha / rank) if rank > 0 else 1.0

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        if rank > 0:
            self.lora_A = nn.Parameter(torch.empty(rank, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, rank))
            self.lora_enabled = True
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.lora_enabled = False

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        if self.rank > 0 and self.lora_A is not None and self.lora_B is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_enabled and self.rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_delta = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
            out = out + lora_delta
        return out


class TinyPersonModel(nn.Module):
    """Teeny tiny neural network specialized for one person's deepfake detection.

    Attributes:
        in_dim: Dimension of input feature vector (e.g. 192 for visual, 128 for audio, 320 for AV).
        hidden_dim: Intermediate channel width (default 64).
        embed_dim: Dimension of identity projection space (default 64).
        lora_rank: Rank of LoRA adaptation matrices (default 4).
        modality: Modality family ('image', 'video', 'audio', 'audiovisual').
    """

    def __init__(
        self,
        in_dim: int = 192,
        hidden_dim: int = 64,
        embed_dim: int = 64,
        lora_rank: int = 4,
        modality: str = "image",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.lora_rank = lora_rank
        self.modality = modality

        # Feature normalization parameters (fitted during training)
        self.register_buffer("feat_mean", torch.zeros(in_dim))
        self.register_buffer("feat_std", torch.ones(in_dim))

        # Lightweight encoder backbone
        self.fc1 = LoRALinear(in_dim, hidden_dim, rank=lora_rank)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = LoRALinear(hidden_dim, hidden_dim, rank=lora_rank)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(0.1)

        # Artifact classification head (outputs probability of manipulation artifacts)
        self.artifact_head = LoRALinear(hidden_dim, 1, rank=lora_rank)

        # Identity metric projection head (outputs unit-norm identity embedding)
        self.identity_head = LoRALinear(hidden_dim, embed_dim, rank=lora_rank)

        # Learned identity prototype of the target person
        self.register_buffer("prototype", torch.zeros(embed_dim))
        self.register_buffer("prototype_similarity_threshold", torch.tensor(0.25, dtype=torch.float32))
        self.register_buffer("decision_threshold", torch.tensor(0.5, dtype=torch.float32))

    def config(self) -> dict[str, Any]:
        return {
            "in_dim": self.in_dim,
            "hidden_dim": self.hidden_dim,
            "embed_dim": self.embed_dim,
            "lora_rank": self.lora_rank,
            "modality": self.modality,
        }

    def enable_peft(self) -> None:
        """Freeze base weights and only train parameter-efficient LoRA matrices."""
        for name, param in self.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def disable_peft(self) -> None:
        """Enable training on all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def get_param_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora = sum(p.numel() for n, p in self.named_parameters() if "lora" in n)
        return {"total": total, "trainable": trainable, "lora": lora}

    def set_prototype(self, embeddings: torch.Tensor | np.ndarray) -> None:
        """Set the target person's genuine identity prototype vector."""
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        norm_emb = F.normalize(embeddings, p=2, dim=-1)
        mean_proto = norm_emb.mean(dim=0)
        proto = F.normalize(mean_proto, p=2, dim=-1)
        self.prototype.copy_(proto.detach())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input feature tensor of shape (B, in_dim).

        Returns:
            artifact_prob: (B,) probability of deepfake/manipulation artifacts.
            identity_emb: (B, embed_dim) unit-norm identity embedding.
        """
        # Standardization
        x_norm = (x - self.feat_mean) / self.feat_std.clamp(min=1e-5)
        h = F.gelu(self.norm1(self.fc1(x_norm)))
        h = h + F.gelu(self.norm2(self.fc2(self.drop(h))))

        artifact_prob = torch.sigmoid(self.artifact_head(h)).squeeze(-1)
        identity_emb = F.normalize(self.identity_head(h), p=2, dim=-1)
        return artifact_prob, identity_emb

    def score(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute the overall deepfake probability and diagnostic evidence.

        A sample is classified as fake if either:
        1. It exhibits synthetic/manipulation artifacts (high artifact score).
        2. It diverges from the enrolled person's biometric prototype (identity mismatch).
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        artifact_prob, identity_emb = self.forward(x)

        # Identity similarity to target prototype
        has_proto = bool(torch.norm(self.prototype).item() > 0.1)
        if has_proto:
            sim = torch.mv(identity_emb, self.prototype).clamp(-1.0, 1.0)
            thr_sim = float(self.prototype_similarity_threshold.item())
            id_mismatch_prob = torch.sigmoid((thr_sim - sim) * 10.0)
            # Combine: fake if high artifact OR high identity mismatch
            combined_fake_prob = torch.clamp(
                1.0 - (1.0 - artifact_prob) * (1.0 - id_mismatch_prob),
                0.0,
                1.0,
            )
        else:
            sim = torch.ones_like(artifact_prob)
            id_mismatch_prob = torch.zeros_like(artifact_prob)
            combined_fake_prob = artifact_prob

        details = {
            "artifact_score": [round(float(v), 4) for v in artifact_prob.detach().cpu().tolist()],
            "identity_similarity": [round(float(v), 4) for v in sim.detach().cpu().tolist()],
            "identity_mismatch_score": [round(float(v), 4) for v in id_mismatch_prob.detach().cpu().tolist()],
            "has_prototype": has_proto,
        }
        return combined_fake_prob, details
