"""Training avid on real clips: contrastive heads, then the calibrated scorer.

Speakers are split three ways: heads are trained on the first split, the scorer
is fitted on the second (so it sees the heads' behaviour on unseen people), and
the third is only for the report.

Negatives for the scorer, made from real clips of other speakers:
- ``voice_swap``: the whole voice from another speaker (voice replacement);
- ``voice_splice``: the voice switches to another speaker part way through;
- ``face_splice``: the face switches to another person part way through.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from forensix.audiovisual.avid.model import FEATURES, Heads, Scorer, clip_features
from forensix.metrics import binary_summary

FORMAT = "avid/1"
KINDS = ["voice_swap", "voice_splice", "face_splice"]


@dataclass
class Clip:
    voice: np.ndarray  # (S, 192) ECAPA per segment
    face: np.ndarray  # (S, 384) DINOv2 per segment, same time windows
    group: str = ""


def load_clip(path: str | Path, group: str = "") -> Clip:
    z = np.load(path)
    return Clip(z["voice"].astype(np.float32), z["face"].astype(np.float32), group)


def save_clip(path: str | Path, voice, face) -> None:
    np.savez_compressed(path, voice=np.asarray(voice, np.float16), face=np.asarray(face, np.float16))


def train_heads(clips: list[Clip], steps: int = 5000, batch: int = 128, lr: float = 1e-3, device: str = "cpu",
                seed: int = 0, log_every: int = 500, log=print, **kw) -> Heads:
    """Symmetric InfoNCE between voices and faces of one clip per speaker per batch."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    by_group: dict[str, list[Clip]] = {}
    for c in clips:
        by_group.setdefault(c.group, []).append(c)
    groups = list(by_group)
    heads = Heads(voice_dim=clips[0].voice.shape[1], face_dim=clips[0].face.shape[1], **kw).to(device)
    opt = torch.optim.AdamW(heads.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    running = []
    for step in range(steps):
        chosen = rng.choice(len(groups), min(batch, len(groups)), replace=False)
        vs, fs = [], []
        for g in chosen:
            c = by_group[groups[g]][rng.integers(len(by_group[groups[g]]))]
            n = len(c.voice)
            length = int(rng.integers(1, n + 1))
            s = int(rng.integers(0, n - length + 1))
            # voice and face from different random windows of the same clip: identity, not timing
            s2 = int(rng.integers(0, n - length + 1))
            vs.append(c.voice[s : s + length].mean(0)); fs.append(c.face[s2 : s2 + length].mean(0))
        v = torch.from_numpy(np.stack(vs)).to(device)
        f = torch.from_numpy(np.stack(fs)).to(device)
        logits = heads.log_temp.exp().clamp(max=100) * heads.embed_voice(v) @ heads.embed_face(f).T
        target = torch.arange(len(v), device=device)
        loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        running.append(loss.item())
        if (step + 1) % log_every == 0 or step + 1 == steps:
            log(f"step {step + 1}/{steps}  loss {np.mean(running):.4f}")
            running = []
    return heads.eval()


def negative(clip: Clip, donor: Clip, kind: str, rng) -> Clip:
    n = min(len(clip.voice), len(donor.voice))
    voice, face = clip.voice[:n].copy(), clip.face[:n].copy()
    if kind == "voice_swap":
        voice = donor.voice[:n].copy()
    else:
        cut = int(rng.integers(max(1, n // 4), max(2, 3 * n // 4) + 1)) if n >= 4 else n // 2
        target = voice if kind == "voice_splice" else face
        source = donor.voice if kind == "voice_splice" else donor.face
        target[cut:] = source[cut:n]
    return Clip(voice, face, clip.group)


def _features(heads: Heads, clip: Clip, device: str) -> torch.Tensor:
    v = torch.from_numpy(clip.voice).to(device)
    f = torch.from_numpy(clip.face).to(device)
    return clip_features(heads, v, f)[0].cpu()


def _examples(heads, clips, rng, device, per_kind: int | None = None):
    """Features of real clips and of each kind of negative (from other speakers)."""
    feats = {"real": [_features(heads, c, device) for c in clips]}
    for kind in KINDS:
        rows = []
        for i in range(per_kind or len(clips)):
            c = clips[i % len(clips)]
            others = [d for d in clips if d.group != c.group]
            if not others:
                continue
            rows.append(_features(heads, negative(c, others[int(rng.integers(len(others)))], kind, rng), device))
        feats[kind] = rows
    return {k: torch.stack(v) for k, v in feats.items() if v}


def fit_scorer(heads: Heads, clips: list[Clip], device: str = "cpu", seed: int = 0, steps: int = 500) -> Scorer:
    rng = np.random.default_rng(seed)
    ex = _examples(heads, clips, rng, device)
    real = ex.pop("real")
    fake = torch.cat(list(ex.values()))
    x = torch.cat([real, fake])
    y = torch.cat([torch.zeros(len(real)), torch.ones(len(fake))])
    w = torch.where(y == 1, len(real) / len(fake), 1.0)  # balance real against all negatives
    scorer = Scorer(len(FEATURES))
    scorer.mean.copy_(x.mean(0))
    scorer.std.copy_(x.std(0).clamp(min=1e-4))
    opt = torch.optim.LBFGS(scorer.linear.parameters(), max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = scorer.linear((x - scorer.mean) / scorer.std).squeeze(-1)
        loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).mean()
        loss = loss + 1e-3 * scorer.linear.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return scorer.eval()


@torch.no_grad()
def evaluate(heads: Heads, scorer: Scorer, clips: list[Clip], device: str = "cpu", seed: int = 1) -> dict:
    """AUC of real vs each negative kind, and 1:2 voice-to-face matching accuracy
    (the voice's own face vs another speaker's), the standard benchmark."""
    rng = np.random.default_rng(seed)
    ex = _examples(heads, clips, rng, device)
    real = scorer(ex.pop("real")).numpy()
    out = {"real_mean_prob": round(float(real.mean()), 4)}
    for kind, feats in ex.items():
        out[kind] = binary_summary(scorer(feats).numpy(), real, threshold=0.5)
    correct, total = 0, 0
    for c in clips:
        others = [d for d in clips if d.group != c.group]
        if not others:
            continue
        o = others[int(rng.integers(len(others)))]
        ev = heads.embed_voice(torch.from_numpy(c.voice).mean(0, keepdim=True))
        own = (ev * heads.embed_face(torch.from_numpy(c.face).mean(0, keepdim=True))).sum()
        other = (ev * heads.embed_face(torch.from_numpy(o.face).mean(0, keepdim=True))).sum()
        correct += int(own > other); total += 1
    out["matching_1_of_2"] = round(correct / max(total, 1), 4)
    return out


def save(path: str | Path, heads: Heads, scorer: Scorer, info: dict) -> None:
    from safetensors.torch import save_file

    state = {f"heads.{k}": v for k, v in heads.state_dict().items()}
    state.update({f"scorer.{k}": v for k, v in scorer.state_dict().items()})
    save_file({k: v.contiguous() for k, v in state.items()}, str(path),
              metadata={"format": FORMAT, "config": json.dumps(heads.config), "features": json.dumps(FEATURES),
                        "info": json.dumps(info)})


def load(path: str | Path, device: str = "cpu") -> tuple[Heads, Scorer, dict]:
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path} is not an avid checkpoint ({FORMAT})")
    if json.loads(meta["features"]) != FEATURES:
        raise ValueError(f"{path} was trained with other features: {meta['features']}")
    state = load_file(str(path))
    heads = Heads(**json.loads(meta["config"]))
    heads.load_state_dict({k[len("heads."):]: v for k, v in state.items() if k.startswith("heads.")})
    scorer = Scorer(len(FEATURES))
    scorer.load_state_dict({k[len("scorer."):]: v for k, v in state.items() if k.startswith("scorer.")})
    return heads.to(device).eval(), scorer.to(device).eval(), json.loads(meta["info"])
