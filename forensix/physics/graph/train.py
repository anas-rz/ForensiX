"""Training the physdf graph prior on real clips, residuals, and calibration.

Masked modelling: in every training window some node states are hidden (spans
of single nodes, whole nodes, the mouth, a whole modality) and the model
predicts them from what stays visible. The residual queries at test time
hide the same kinds of sets, so they're in-distribution for real media.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from forensix.physics.graph.features import GraphSequence
from forensix.physics.graph.model import PhysicsGNN, gaussian_nll
from forensix.physics.graph.schema import (
    AUDIO_NODES, MOUTH_NODES, NODE_NAMES, RESIDUAL_GROUPS, STEP_S, VIDEO_NODES, feature_dims,
)

FORMAT = "physdf-graph/1"
WINDOW_S = 1.0  # localization window
SUMMARY_QUANTILES = [0.5, 0.9, 0.95, 0.99, 0.999]


@dataclass
class Normalizer:
    mean: dict[str, np.ndarray]
    std: dict[str, np.ndarray]

    @classmethod
    def fit(cls, sequences: list[GraphSequence]) -> "Normalizer":
        mean, std = {}, {}
        for n in NODE_NAMES:
            rows = [s.features[n][s.valid[n]] for s in sequences if s.valid[n].any()]
            x = np.concatenate(rows) if rows else np.zeros((1, feature_dims()[n]))
            mean[n], std[n] = x.mean(0), np.maximum(x.std(0), 1e-3)
        return cls(mean, std)

    def __call__(self, seq: GraphSequence) -> dict[str, np.ndarray]:
        return {n: np.where(seq.valid[n][:, None], (seq.features[n] - self.mean[n]) / self.std[n], 0.0).astype(np.float32)
                for n in NODE_NAMES}

    def to_json(self) -> dict:
        return {n: {"mean": self.mean[n].tolist(), "std": self.std[n].tolist()} for n in NODE_NAMES}

    @classmethod
    def from_json(cls, d: dict) -> "Normalizer":
        return cls({n: np.array(d[n]["mean"]) for n in NODE_NAMES}, {n: np.array(d[n]["std"]) for n in NODE_NAMES})


def _valid_matrix(seq: GraphSequence) -> np.ndarray:
    return np.stack([seq.valid[n] for n in NODE_NAMES], axis=1)  # (T, N)


HIDE_SETS = [[n] for n in NODE_NAMES] + [MOUTH_NODES, AUDIO_NODES, VIDEO_NODES, ["jaw", "lips", "upper_face"]]


def sample_hidden(t: int, rng: np.random.Generator) -> np.ndarray:
    """(T, N) bool: node states to hide in one training window."""
    hidden = np.zeros((t, len(NODE_NAMES)), bool)
    r = rng.random()
    if r < 0.65:  # a whole set for the whole window (the test-time queries)
        for n in HIDE_SETS[rng.integers(len(HIDE_SETS))]:
            hidden[:, NODE_NAMES.index(n)] = True
    if r > 0.35:  # random spans of single nodes
        for i in range(len(NODE_NAMES)):
            if rng.random() < 0.3:
                length = int(rng.uniform(0.4, 1.6) / STEP_S)
                start = rng.integers(0, max(1, t - length))
                hidden[start : start + length, i] = True
    return hidden


def _batch(seqs: list[GraphSequence], norm: Normalizer, window: int, batch: int, rng: np.random.Generator):
    xs = {n: [] for n in NODE_NAMES}
    valid, hidden = [], []
    for _ in range(batch):
        s = seqs[rng.integers(len(seqs))]
        start = rng.integers(0, max(1, len(s) - window + 1))
        x = norm(s)
        v = _valid_matrix(s)[start : start + window]
        pad = window - len(v)
        for n in NODE_NAMES:
            xs[n].append(np.pad(x[n][start : start + window], ((0, pad), (0, 0))))
        valid.append(np.pad(v, ((0, pad), (0, 0))))
        hidden.append(sample_hidden(window, rng))
    return ({n: torch.from_numpy(np.stack(v)) for n, v in xs.items()},
            torch.from_numpy(np.stack(valid)), torch.from_numpy(np.stack(hidden)))


def masked_loss(model: PhysicsGNN, x: dict, valid: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    out = model(x, valid & ~hidden)
    target = (valid & hidden).float()
    total, count = 0.0, 0.0
    for i, n in enumerate(NODE_NAMES):
        mean, log_var = out[n]
        w = target[:, :, i : i + 1]
        total = total + (gaussian_nll(x[n], mean, log_var) * w).sum()
        count = count + w.sum() * mean.shape[-1]
    return total / count.clamp(min=1.0)


def train(sequences: list[GraphSequence], steps: int = 20000, window_s: float = 4.0, batch: int = 32,
          lr: float = 1e-3, dim: int = 64, layers: int = 6, device: str = "cpu", seed: int = 0,
          log_every: int = 500, log=print) -> tuple[PhysicsGNN, Normalizer, list[float]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    norm = Normalizer.fit(sequences)
    model = PhysicsGNN(dim=dim, layers=layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    window = int(window_s / STEP_S)
    history, running = [], []
    model.train()
    for step in range(steps):
        x, valid, hidden = _batch(sequences, norm, window, batch, rng)
        x = {n: v.to(device) for n, v in x.items()}
        loss = masked_loss(model, x, valid.to(device), hidden.to(device))
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
    return model, norm, history


def _queries() -> list[tuple[str, list[str], list[str]]]:
    """(group, hidden nodes, nodes scored in this pass) for every residual pass."""
    out = []
    for group, (scored, hide) in RESIDUAL_GROUPS.items():
        if hide is None:
            out.extend((group, [n], [n]) for n in scored)
        else:
            out.append((group, hide, scored))
    return out


@torch.no_grad()
def node_residuals(model: PhysicsGNN, norm: Normalizer, seq: GraphSequence, device: str = "cpu") -> dict:
    """``{group: (T,) residual}``: mean NLL per step over the scored, valid nodes
    of each group, predicted with the group's hidden set hidden (NaN where no
    scored node is valid). Also ``{group: {node: (T,)}}`` under "nodes"."""
    x = norm(seq)
    valid = torch.from_numpy(_valid_matrix(seq))
    queries = _queries()
    hidden = torch.zeros(len(queries), *valid.shape, dtype=torch.bool)
    for q, (_, hide, _) in enumerate(queries):
        for n in hide:
            hidden[q, :, NODE_NAMES.index(n)] = True
    xb = {n: torch.from_numpy(v).unsqueeze(0).expand(len(queries), -1, -1).to(device) for n, v in x.items()}
    visible = (valid.unsqueeze(0) & ~hidden).to(device)
    out = model(xb, visible)
    per_group: dict[str, list] = {}
    per_node: dict[str, dict] = {}
    for q, (group, _, scored) in enumerate(queries):
        for n in scored:
            mean, log_var = out[n]
            r = gaussian_nll(xb[n][q], mean[q], log_var[q]).mean(-1).cpu().numpy()
            r = np.where(seq.valid[n], r, np.nan)
            per_group.setdefault(group, []).append(r)
            per_node.setdefault(group, {})[n] = r
    with np.errstate(all="ignore"):
        groups = {g: np.nanmean(np.stack(rs), axis=0) if np.isfinite(np.stack(rs)).any() else np.full(len(seq), np.nan)
                  for g, rs in per_group.items()}
    return {"groups": groups, "nodes": per_node}


def window_scores(residual: np.ndarray, window_s: float = WINDOW_S) -> np.ndarray:
    """Mean residual per non-overlapping window (NaN where under half is valid)."""
    w = max(1, int(round(window_s / STEP_S)))
    out = []
    for k in range(0, len(residual) - w + 1, w):
        seg = residual[k : k + w]
        ok = np.isfinite(seg)
        out.append(float(seg[ok].mean()) if ok.sum() >= w // 2 else np.nan)
    return np.array(out)


def clip_summary(windows: np.ndarray) -> float:
    """A clip's residual for a group: the 90th percentile of its window scores
    (localized manipulation lifts a few windows; one noisy window doesn't)."""
    ok = windows[np.isfinite(windows)]
    return float(np.percentile(ok, 90)) if len(ok) >= 2 else float("nan")


CONDITIONS = ("audio_video", "audio_only", "video_only")
AV_GROUPS = ("graph_av_mouth_from_audio", "graph_av_audio_from_video")


def has_modality(seq: GraphSequence, nodes: list[str]) -> bool:
    return any(seq.valid[n].any() for n in nodes)


def condition(seq: GraphSequence) -> str | None:
    """Which modalities a clip has: "audio_video", "audio_only", "video_only" (None: neither)."""
    v, a = has_modality(seq, VIDEO_NODES), has_modality(seq, AUDIO_NODES)
    return "audio_video" if v and a else "audio_only" if a else "video_only" if v else None


def drop_modality(seq: GraphSequence, nodes: list[str]) -> GraphSequence:
    valid = {n: (np.zeros_like(v) if n in nodes else v) for n, v in seq.valid.items()}
    return GraphSequence(seq.features, valid, seq.step_s)


def _calibrate_condition(model, norm, sequences, device, groups) -> dict:
    clips: dict[str, list] = {g: [] for g in groups}
    windows: dict[str, list] = {g: [] for g in groups}
    for seq in sequences:
        res = node_residuals(model, norm, seq, device)["groups"]
        for g in groups:
            w = window_scores(res[g])
            windows[g].extend(w[np.isfinite(w)].tolist())
            s = clip_summary(w)
            if np.isfinite(s):
                clips[g].append(s)
    out = {}
    for g in groups:
        c, w = np.array(clips[g]), np.array(windows[g])
        if len(c) < 5:
            continue
        out[g] = {
            "n_clips": int(len(c)),
            "clip_mean": float(c.mean()),
            "clip_std": float(max(c.std(), 1e-3)),
            "clip_quantiles": dict(zip(map(str, SUMMARY_QUANTILES), np.quantile(c, SUMMARY_QUANTILES).tolist())),
            "window_mean": float(w.mean()),
            "window_std": float(max(w.std(), 1e-3)),
            "window_p99": float(np.quantile(w, 0.99)),
        }
    return out


def calibrate(model: PhysicsGNN, norm: Normalizer, sequences: list[GraphSequence], device: str = "cpu") -> dict:
    """Distribution of every group's residual on held-out real clips, per input
    condition: per clip (for z-scores and rarity) and per window (for
    localization). Single-modality conditions are also made by dropping a side
    of audio-visual clips; they have no audio-visual constraints."""
    by_condition = {c: [] for c in CONDITIONS}
    for seq in sequences:
        cond = condition(seq)
        if cond == "audio_video":
            by_condition["audio_video"].append(seq)
            by_condition["audio_only"].append(drop_modality(seq, VIDEO_NODES))
            by_condition["video_only"].append(drop_modality(seq, AUDIO_NODES))
        elif cond is not None:
            by_condition[cond].append(seq)
    out = {}
    for cond, seqs in by_condition.items():
        groups = [g for g in RESIDUAL_GROUPS if cond == "audio_video" or g not in AV_GROUPS]
        stats = _calibrate_condition(model, norm, seqs, device, groups) if seqs else {}
        if stats:
            out[cond] = stats
    return out


def save(path: str | Path, model: PhysicsGNN, norm: Normalizer, calibration: dict, info: dict | None = None) -> None:
    from safetensors.torch import save_file

    meta = {"format": FORMAT, "config": json.dumps(model.config), "normalizer": json.dumps(norm.to_json()),
            "calibration": json.dumps(calibration), "info": json.dumps(info or {})}
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(path), metadata=meta)


def load(path: str | Path, device: str = "cpu") -> tuple[PhysicsGNN, Normalizer, dict, dict]:
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path} is not a physdf graph checkpoint ({FORMAT})")
    model = PhysicsGNN(**json.loads(meta["config"]))
    model.load_state_dict(load_file(str(path)))
    model.to(device).eval()
    return model, Normalizer.from_json(json.loads(meta["normalizer"])), json.loads(meta["calibration"]), json.loads(meta["info"])
