"""Training and evaluating the temporal ID network on real videos.

A person's positives come from *different videos* whenever possible, so the
network learns how the person moves rather than a recording's camera, framing
or session. Evaluation is verification on held-out people: enroll from one of
their videos, then score their other videos (genuine) and other people's videos
(impostors: a face swap carries another person's motion). The EER threshold on
that split is stored with the model as its decision threshold.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from forensix.identity.motion.features import windows
from forensix.identity.motion.model import MotionEncoder, supcon_loss
from forensix.metrics import binary_summary, compute_eer

FORMAT = "id-motion/1"
WINDOW, HOP = 100, 25  # 4 s windows every 1 s at 25 fps
MIN_VALID = 0.8
TOP_K = 5


@dataclass
class Clip:
    feats: np.ndarray  # (T, 64)
    valid: np.ndarray  # (T,)
    person: str
    video: str


def save_clip(path, feats, valid) -> None:
    np.savez_compressed(path, feats=np.asarray(feats, np.float16), valid=np.asarray(valid, bool))


def load_clip(path, person: str, video: str) -> Clip:
    z = np.load(path)
    return Clip(z["feats"].astype(np.float32), z["valid"], person, video)


def _window(clip: Clip, rng, length: int = WINDOW, augment: bool = True) -> tuple[np.ndarray, np.ndarray]:
    speed = rng.uniform(0.9, 1.1) if augment else 1.0
    span = min(len(clip.feats) - 1, int(np.ceil(length * speed)))
    start = int(rng.integers(0, max(1, len(clip.feats) - span)))
    t = start + np.arange(length) * speed
    t = np.clip(t, 0, len(clip.feats) - 1)
    x = np.stack([np.interp(t, np.arange(len(clip.feats)), clip.feats[:, j]) for j in range(clip.feats.shape[1])], 1)
    m = clip.valid[np.round(t).astype(int)].copy()
    if augment:
        m &= rng.random(length) > 0.05  # dropped frames
    return x.astype(np.float32), m


def train(clips: list[Clip], steps: int = 20000, people_per_batch: int = 64, per_person: int = 4, lr: float = 1e-3,
          device: str = "cpu", seed: int = 0, log_every: int = 500, log=print, **model_kw) -> MotionEncoder:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    clips = [c for c in clips if len(c.feats) > WINDOW * 1.15]
    by_person: dict[str, dict[str, list[Clip]]] = {}
    for c in clips:
        by_person.setdefault(c.person, {}).setdefault(c.video, []).append(c)
    people = [p for p in by_person if sum(map(len, by_person[p].values())) >= 2]
    enc = MotionEncoder(in_dim=clips[0].feats.shape[1], **model_kw)
    stacked = np.concatenate([c.feats[c.valid] for c in clips])
    enc.mean.copy_(torch.from_numpy(stacked.mean(0)))
    enc.std.copy_(torch.from_numpy(np.maximum(stacked.std(0), 1e-3)))
    enc.to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    running = []
    enc.train()
    for step in range(steps):
        chosen = rng.choice(len(people), min(people_per_batch, len(people)), replace=False)
        xs, ms, labels = [], [], []
        for i, p in enumerate(chosen):
            videos = list(by_person[people[p]].values())
            order = rng.permutation(len(videos))
            for k in range(per_person):  # cycle through different videos first
                vclips = videos[order[k % len(videos)]]
                x, m = _window(vclips[int(rng.integers(len(vclips)))], rng)
                x = x + rng.normal(0, 0.02, x.shape).astype(np.float32) * enc.std.cpu().numpy()
                xs.append(x); ms.append(m); labels.append(i)
        x = torch.from_numpy(np.stack(xs)).to(device)
        m = torch.from_numpy(np.stack(ms)).to(device)
        loss = supcon_loss(enc(x, m), torch.tensor(labels, device=device))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        opt.step()
        sched.step()
        running.append(loss.item())
        if (step + 1) % log_every == 0 or step + 1 == steps:
            log(f"step {step + 1}/{steps}  loss {np.mean(running):.4f}")
            running = []
    return enc.eval()


@torch.no_grad()
def embed(enc: MotionEncoder, feats: np.ndarray, valid: np.ndarray, hop: int = HOP, device: str = "cpu",
          batch: int = 256) -> tuple[np.ndarray, list[int]]:
    """Embeddings of every window with at least 80% tracked frames, and their start frames."""
    starts = [s for s in windows(len(feats), WINDOW, hop) if valid[s : s + WINDOW].mean() >= MIN_VALID]
    if not starts:
        return np.zeros((0, enc.config["dim"]), np.float32), []
    out = []
    for i in range(0, len(starts), batch):
        idx = starts[i : i + batch]
        x = torch.from_numpy(np.stack([feats[s : s + WINDOW] for s in idx])).float().to(device)
        m = torch.from_numpy(np.stack([valid[s : s + WINDOW] for s in idx])).to(device)
        out.append(enc(x, m).cpu().numpy())
    return np.concatenate(out), starts


def clip_similarity(query: np.ndarray, refs: np.ndarray, k: int = TOP_K) -> tuple[float, np.ndarray]:
    """Median over query windows of the mean of their k highest similarities to
    the reference windows, and the per-window values."""
    sims = query @ refs.T
    k = min(k, sims.shape[1])
    per_window = np.sort(sims, axis=1)[:, -k:].mean(1)
    return float(np.median(per_window)), per_window


def evaluate(enc: MotionEncoder, clips: list[Clip], device: str = "cpu", impostors_per_person: int = 20,
             seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    emb = {}
    for c in clips:
        e, _ = embed(enc, c.feats, c.valid, device=device)
        if len(e):
            emb.setdefault(c.person, {}).setdefault(c.video, []).append(e)
    people = [p for p in emb if len(emb[p]) >= 2]
    genuine, impostor = [], []
    for p in people:
        videos = sorted(emb[p])
        ref_video = videos[int(rng.integers(len(videos)))]
        refs = np.concatenate(emb[p][ref_video])
        for v in videos:
            if v != ref_video:
                genuine.extend(clip_similarity(q, refs)[0] for q in emb[p][v])
        others = [q for o in emb if o != p for qs in emb[o].values() for q in qs]
        for i in rng.choice(len(others), min(impostors_per_person, len(others)), replace=False):
            impostor.append(clip_similarity(others[i], refs)[0])
    if not genuine or not impostor:
        raise ValueError("Need held-out people with at least 2 videos each")
    genuine, impostor = np.array(genuine), np.array(impostor)
    # scores as anomaly: 1 - similarity (higher = not the person)
    eer, thr = compute_eer(1 - impostor, 1 - genuine)
    summary = binary_summary(1 - impostor, 1 - genuine, threshold=thr)
    summary.update(eer=round(eer, 4), threshold=round(float(thr), 4), n_people=len(people),
                   genuine_mean=round(float(genuine.mean()), 4), impostor_mean=round(float(impostor.mean()), 4))
    return summary


def save(path, enc: MotionEncoder, threshold: float, info: dict) -> None:
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in enc.state_dict().items()}, str(path),
              metadata={"format": FORMAT, "config": json.dumps(enc.config), "threshold": json.dumps(threshold),
                        "info": json.dumps(info)})


def load(path, device: str = "cpu") -> tuple[MotionEncoder, float, dict]:
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path} is not an id-motion checkpoint ({FORMAT})")
    enc = MotionEncoder(**json.loads(meta["config"]))
    enc.load_state_dict(load_file(str(path)))
    return enc.to(device).eval(), json.loads(meta["threshold"]), json.loads(meta["info"])
