"""Voice-face identity model: a shared space for voices and faces, identity
continuity, and a calibrated scorer.

- Projection heads map ECAPA-TDNN voice embeddings (192) and DINOv2 face
  embeddings (384) into one 128-d space. They are trained contrastively on real
  videos: a speaker's voice sits close to their own face and far from other
  people's faces. What a face reveals about a voice is mostly sex, age, build
  and accent/ethnicity (Nagrani et al., CVPR 2018: about 80% on 1:2 voice-to-face
  matching), so a mismatch is evidence, not proof.
- Continuity: a clip should keep one voice and one face. The largest distance
  between non-overlapping segments of the same modality flags an identity switch
  mid-clip.
- Scorer: logistic regression on the standardized features, fitted on held-out
  real clips against voice swaps and splices made from real clips.
"""

import torch
import torch.nn.functional as F
from torch import nn

FEATURES = ["match", "match_min", "voice_jump", "face_jump", "voice_split", "face_split"]


class Heads(nn.Module):
    def __init__(self, voice_dim: int = 192, face_dim: int = 384, dim: int = 128, hidden: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.config = {"voice_dim": voice_dim, "face_dim": face_dim, "dim": dim, "hidden": hidden, "dropout": dropout}

        def mlp(d):
            return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, dim))

        self.voice, self.face = mlp(voice_dim), mlp(face_dim)
        self.log_temp = nn.Parameter(torch.tensor(2.66))  # exp = 1 / 0.07

    def embed_voice(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.voice(v), dim=-1)

    def embed_face(self, f: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.face(f), dim=-1)


def _cos(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(x, y, dim=-1)


def _max_jump(segs: torch.Tensor, gap: int = 2) -> torch.Tensor:
    """Largest 1 - cos between segments ``gap`` apart (non-overlapping at 50% hop)."""
    if len(segs) <= gap:
        return segs.new_tensor(0.0)
    return (1 - _cos(segs[:-gap], segs[gap:])).max()


def _split(segs: torch.Tensor) -> torch.Tensor:
    """1 - cos between the mean of the first and second half."""
    if len(segs) < 4:
        return segs.new_tensor(0.0)
    h = len(segs) // 2
    return 1 - _cos(segs[:h].mean(0), segs[h:].mean(0))


@torch.no_grad()
def clip_features(heads: Heads, voice: torch.Tensor, face: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Features of one clip (``voice`` (S, 192), ``face`` (S, 384), time-aligned
    segments) and the per-segment match (S,), for localization."""
    ev, ef = heads.embed_voice(voice), heads.embed_face(face)
    seg_match = _cos(ev, ef)
    match = _cos(F.normalize(ev.mean(0), dim=-1), F.normalize(ef.mean(0), dim=-1))
    k = max(1, len(seg_match) // 4)
    feats = torch.stack([match, seg_match.sort().values[:k].mean(), _max_jump(voice), _max_jump(face),
                         _split(voice), _split(face)])
    return feats, seg_match


class Scorer(nn.Module):
    """Logistic regression on standardized clip features -> P(identity mismatch)."""

    def __init__(self, n: int = len(FEATURES)):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("std", torch.ones(n))
        self.linear = nn.Linear(n, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear((feats - self.mean) / self.std)).squeeze(-1)
