"""Training and evaluating the AV consistency model on real clips (see negatives.py)."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from forensix.audiovisual.avcons.model import AVConsistency
from forensix.audiovisual.avcons.negatives import Clip, Example, negative, positive, sample
from forensix.metrics import binary_summary

FORMAT = "avcons/1"
KINDS = ["shift", "drift", "swap", "face_swap", "splice", "visual_splice"]


def load_clip(path: str | Path, group: str = "") -> Clip:
    z = np.load(path)
    speeds = {float(k[len("a_speed_"):]): z[k].astype(np.float32) for k in z.files if k.startswith("a_speed_")}
    return Clip(z["v"].astype(np.float32), z["a"].astype(np.float32), speeds, group)


def save_clip(path: str | Path, v, a, a_speed: dict) -> None:
    np.savez_compressed(path, v=np.asarray(v, np.float16), a=np.asarray(a, np.float16),
                        **{f"a_speed_{s}": np.asarray(x, np.float16) for s, x in a_speed.items()})


def collate(examples: list[Example]) -> tuple[torch.Tensor, ...]:
    t = max(len(e.v) for e in examples)
    d = examples[0].v.shape[1]
    v = torch.zeros(len(examples), t, d)
    a = torch.zeros(len(examples), t, d)
    mask = torch.zeros(len(examples), t, dtype=torch.bool)
    wl = torch.zeros(len(examples), t)
    for i, e in enumerate(examples):
        n = len(e.v)
        v[i, :n], a[i, :n] = torch.from_numpy(e.v), torch.from_numpy(e.a)
        mask[i, :n] = True
        wl[i, :n] = torch.from_numpy(e.window_labels)
    y = torch.tensor([float(e.label) for e in examples])
    return v, a, mask, wl, y


def loss_fn(model, batch, device, window_weight: float = 0.5) -> torch.Tensor:
    v, a, mask, wl, y = (x.to(device) for x in batch)
    clip, window = model(v, a, mask)
    loss = F.binary_cross_entropy_with_logits(clip, y)
    wloss = F.binary_cross_entropy_with_logits(window, wl, reduction="none")
    return loss + window_weight * (wloss * mask).sum() / mask.sum()


def train(clips: list[Clip], steps: int = 20000, batch: int = 64, window_s: float = 5.0, lr: float = 3e-4,
          device: str = "cpu", seed: int = 0, log_every: int = 500, log=print, **model_kw) -> tuple[AVConsistency, list]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = AVConsistency(**model_kw).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    length = int(window_s * 25)
    history, running = [], []
    model.train()
    for step in range(steps):
        examples = [sample(clips, length, rng) for _ in range(batch)]
        loss = loss_fn(model, collate(examples), device)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running.append(loss.item())
        if (step + 1) % log_every == 0 or step + 1 == steps:
            history.append(float(np.mean(running)))
            log(f"step {step + 1}/{steps}  loss {history[-1]:.4f}")
            running = []
    model.eval()
    return model, history


@torch.no_grad()
def score(model: AVConsistency, v: np.ndarray, a: np.ndarray, device: str = "cpu") -> tuple[float, np.ndarray]:
    """Clip probability and per-window probabilities of AV inconsistency."""
    n = min(len(v), len(a))
    vt = torch.from_numpy(np.asarray(v[:n], np.float32))[None].to(device)
    at = torch.from_numpy(np.asarray(a[:n], np.float32))[None].to(device)
    clip, window = model(vt, at)
    return float(torch.sigmoid(clip)[0]), torch.sigmoid(window)[0].cpu().numpy()


def evaluate(model: AVConsistency, clips: list[Clip], window_s: float = 5.0, per_kind: int = 200,
             device: str = "cpu", seed: int = 1) -> dict:
    """AUC of real vs each manipulation on held-out clips, plus window-level AUC
    for the splices (localization)."""
    rng = np.random.default_rng(seed)
    length = int(window_s * 25)
    real = [score(model, e.v, e.a, device)[0]
            for e in (positive(clips[int(rng.integers(len(clips)))], length, rng) for _ in range(per_kind))]
    out = {"real_mean_prob": round(float(np.mean(real)), 4)}
    for kind in KINDS:
        probs, w_scores, w_labels = [], [], []
        for _ in range(per_kind):
            i, j = rng.choice(len(clips), 2, replace=False) if len(clips) > 1 else (0, 0)
            e = negative(clips[i], clips[j], length, rng, kind)
            p, w = score(model, e.v, e.a, device)
            probs.append(p)
            w_scores.extend(w.tolist()); w_labels.extend(e.window_labels.tolist())
        out[kind] = binary_summary(probs, real, threshold=0.5)
        if kind in ("splice", "visual_splice", "drift"):
            w_scores, w_labels = np.array(w_scores), np.array(w_labels)
            if 0 < w_labels.mean() < 1:
                out[kind]["window_auc"] = binary_summary(w_scores[w_labels == 1], w_scores[w_labels == 0],
                                                         threshold=None)["auc"]
    return out


def save(path: str | Path, model: AVConsistency, info: dict) -> None:
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(path),
              metadata={"format": FORMAT, "config": json.dumps(model.config), "info": json.dumps(info)})


def load(path: str | Path, device: str = "cpu") -> tuple[AVConsistency, dict]:
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path} is not an avcons checkpoint ({FORMAT})")
    model = AVConsistency(**json.loads(meta["config"]))
    model.load_state_dict(load_file(str(path)))
    return model.to(device).eval(), json.loads(meta["info"])
