"""Fine-tune a pretrained detector on your labelled data, and load the result.

Supported: every image detector, every audio detector, and the GenD video detectors (``gend``,
``gend-pe``, ``dfdet-clip``, which score aligned face crops). Data is a ForensiX dataset (see
``forensix.dataset``) or lists of real and fake files.

    det, report = forensix.finetune("commfor", dataset="my_data.zip", out="commfor-ft.safetensors",
                                    mode="norm", epochs=5, device="cuda")
    det = forensix.load("commfor", weights_path="commfor-ft.safetensors")   # anywhere later

What gets trained (``mode``):

- ``head``: only the final classifier layer. Fastest, needs the least data, can't overfit much.
- ``norm``: the normalization layers' scales and shifts plus the head (as GenD's own training,
  "LN-tuning"). The default: adapts the features to your media at a small fraction of the
  parameters.
- ``full``: every parameter, at a low learning rate. Needs thousands of items per class.
- ``lora``: low-rank adapters on the Linear layers plus the head, through Hugging Face PEFT
  (``pip install peft``; see docs/guides/peft.md). Close to ``full`` in what it can learn, at a
  fraction of the memory. The adapters are merged into the weights after training and again when
  loading, so serving needs no PEFT and runs at the original speed.

A held-out part of the data (``val_fraction``, split by ``group`` so near-duplicates stay on one
side) scores the base model first and the fine-tuned model after every epoch; the epoch with the
best held-out AUC is kept. ``head``, ``norm`` and ``lora`` checkpoints store only the tensors they
changed (for ``lora``, the adapter matrices), so they're small; ``forensix.load`` rebuilds the pretrained model and applies them.

Command line: ``python -m forensix.finetuning MODEL --dataset DATA.zip --out MODEL-ft.safetensors``.
"""

import argparse
import hashlib
import json
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

MODES = ("head", "norm", "lora", "full")
DEFAULT_LR = {"head": 1e-3, "norm": 1e-4, "lora": 2e-4, "full": 1e-5}
METADATA_KEY = "forensix.finetune"
_NORMS = (nn.LayerNorm, nn.GroupNorm, nn.modules.batchnorm._BatchNorm)


# -- loading --------------------------------------------------------------------------------------

def checkpoint_info(path: str | Path) -> dict | None:
    """The fine-tune metadata of ``path``, or None if it isn't a ForensiX fine-tune checkpoint."""
    path = Path(path)
    if path.suffix != ".safetensors" or not path.is_file():
        return None
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        meta = (f.metadata() or {}).get(METADATA_KEY)
    return json.loads(meta) if meta else None


def apply(detector, path: str | Path):
    """Load a fine-tune checkpoint into a detector built with its pretrained weights."""
    from safetensors.torch import load_file

    info = checkpoint_info(path)
    if info is None:
        raise ValueError(f"{path} is not a ForensiX fine-tune checkpoint")
    if info["model"] != detector.name:
        raise ValueError(f"{path} fine-tunes {info['model']!r}, not {detector.name!r}")
    tensors = load_file(str(path), device=str(detector.device))
    if info["mode"] == "lora":
        tensors = _merge_saved_lora(detector.network, tensors, info["lora"])
    result = detector.network.load_state_dict(tensors, strict=info["mode"] == "full")
    if result.unexpected_keys:
        raise ValueError(f"{path} has tensors {detector.name} doesn't: {result.unexpected_keys[:5]}")
    detector.network.eval()
    return detector


# -- what to train --------------------------------------------------------------------------------

def head_module(network: nn.Module) -> nn.Linear:
    """The output layer: the last Linear with at most two outputs (else the last Linear)."""
    linears = [m for m in network.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError("no Linear output layer found")
    small = [m for m in linears if m.out_features <= 2]
    return (small or linears)[-1]


def trainable_parameters(network: nn.Module, mode: str) -> dict[str, nn.Parameter]:
    if mode == "full":
        return dict(network.named_parameters())
    head = head_module(network)
    chosen = {id(p) for p in head.parameters()}
    if mode == "norm":
        chosen |= {id(p) for m in network.modules() if isinstance(m, _NORMS) for p in m.parameters()}
    return {n: p for n, p in network.named_parameters() if id(p) in chosen}


def _train_mode(network: nn.Module):
    network.train()
    for m in network.modules():  # batch statistics from a small dataset would wreck the features
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


# -- LoRA through PEFT ----------------------------------------------------------------------------

def lora_targets(network: nn.Module, pattern: str | None = None) -> list[str]:
    """Modules that get adapters: those whose name matches ``pattern`` (a regex), else every
    Linear layer but the head (and the Conv2d layers too when a network has almost no Linear)."""
    import re

    head = head_module(network)
    if pattern:
        names = [n for n, m in network.named_modules() if re.search(pattern, n) and m is not head
                 and isinstance(m, (nn.Linear, nn.Conv2d))]
    else:
        names = [n for n, m in network.named_modules() if isinstance(m, nn.Linear) and m is not head]
        if len(names) < 2:
            names += [n for n, m in network.named_modules() if isinstance(m, nn.Conv2d)]
    if not names:
        raise ValueError(f"no Linear or Conv2d layers match {pattern!r}")
    return names


def add_lora(network: nn.Module, rank: int, alpha: float, dropout: float, targets: list[str]) -> dict:
    """Inject PEFT LoRA adapters into ``network`` in place; returns the adapter config."""
    try:
        from peft import LoraConfig, inject_adapter_in_model
    except ImportError as e:
        raise ImportError("mode='lora' needs PEFT: pip install peft") from e
    import re

    # anchored full names, so 'fc' doesn't also match 'blocks.0.mlp.fc'
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                        target_modules="^(" + "|".join(re.escape(t) for t in targets) + ")$")
    inject_adapter_in_model(config, network)
    return {"rank": rank, "alpha": alpha, "dropout": dropout, "scaling": alpha / rank, "targets": targets}


def merge_lora(network: nn.Module) -> dict[str, torch.Tensor]:
    """Fold every PEFT LoRA layer into its base layer, put the plain layer back, and return the
    tensors a checkpoint needs to redo the merge without PEFT: ``<module>.lora_A`` and
    ``.lora_B`` for Linear layers (small), the merged delta ``.lora_delta`` for convolutions."""
    from peft.tuners.lora import LoraLayer

    saved = {}
    wrapped = [(n, m) for n, m in network.named_modules() if isinstance(m, LoraLayer)]
    for name, layer in wrapped:
        base = layer.get_base_layer()
        with torch.no_grad():
            delta = layer.get_delta_weight("default")
            if isinstance(base, nn.Linear):
                saved[f"{name}.lora_A"] = layer.lora_A["default"].weight.detach().cpu().clone()
                saved[f"{name}.lora_B"] = layer.lora_B["default"].weight.detach().cpu().clone()
            else:
                saved[f"{name}.lora_delta"] = delta.detach().cpu().clone()
            base.weight += delta.to(base.weight.dtype)
        parent, _, child = name.rpartition(".")
        setattr(network.get_submodule(parent) if parent else network, child, base)
    return saved


def _merge_saved_lora(network: nn.Module, tensors: dict, config: dict) -> dict:
    """Add saved adapters to the pretrained weights (no PEFT needed); return the other tensors."""
    rest = {}
    with torch.no_grad():
        for key, value in tensors.items():
            module, _, kind = key.rpartition(".")
            if kind == "lora_A":
                weight = network.get_submodule(module).weight
                b = tensors[f"{module}.lora_B"]
                weight += (b.float() @ value.float()).to(weight.dtype) * config["scaling"]
            elif kind == "lora_delta":
                weight = network.get_submodule(module).weight
                weight += value.to(weight.dtype)
            elif kind != "lora_B":
                rest[key] = value
    return rest


# -- data per family ------------------------------------------------------------------------------

def _augment_image(img, rng: random.Random):
    """Light augmentation that keeps forensic traces: flips and mild JPEG recompression."""
    import io

    from PIL import Image, ImageOps

    if rng.random() < 0.5:
        img = ImageOps.mirror(img)
    if rng.random() < 0.3:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=rng.randint(75, 98))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img


class _Image:
    """One sample per image; the detector's own evaluation transform."""

    def __init__(self, det):
        self.det = det

    def samples(self, items, log):
        return [(i, item.path) for i, item in enumerate(items)]

    def forward(self, payloads, train, rng):
        from forensix.image.io import load_image

        imgs = [load_image(p) for p in payloads]
        if train:
            imgs = [_augment_image(im, rng) for im in imgs]
        batch = torch.stack([self.det.transform(im) for im in imgs]).to(self.det.device)
        return self.det._fake_probs(batch)


class _Audio:
    """One sample per clip: a random window when training, every window (averaged) when scoring."""

    def __init__(self, det):
        self.det = det

    def samples(self, items, log):
        return [(i, item.path) for i, item in enumerate(items)]

    def forward(self, payloads, train, rng):
        from forensix.audio.detector import split_windows
        from forensix.audio.io import load_audio

        det, out = self.det, []
        if train:
            windows = []
            for p in payloads:
                w = split_windows(load_audio(p, det.sample_rate), det.window)
                x = w[rng.randrange(len(w))]
                windows.append(x * rng.uniform(0.5, 1.0))   # random gain
            return det._fake_probs(torch.from_numpy(np.stack(windows)).float().to(det.device))
        for p in payloads:
            w = split_windows(load_audio(p, det.sample_rate), det.window, det.pad_short)
            out.append(det._fake_probs(torch.from_numpy(w).float().to(det.device)).mean())
        return torch.stack(out)


class _GenD:
    """Aligned face crops, extracted once per video and cached; one sample per face. A video's
    score is the mean over its faces, as GenDDetector scores it."""

    def __init__(self, det, frames: int, cache: Path):
        self.det, self.frames, self.cache = det, frames, cache

    def samples(self, items, log):
        from forensix.video.gend import aligned_faces
        from forensix.video.io import read_frames

        out = []
        for i, item in enumerate(items):
            key = hashlib.sha1(f"{Path(item.path).resolve()}|{self.frames}".encode()).hexdigest()[:20]
            f = self.cache / f"{key}.npy"   # stable across runs, so a rerun reuses the crops
            if not f.exists():
                frames, _ = read_frames(item.path, self.frames)
                faces = aligned_faces(frames, self.det.face_detector)
                np.save(f, np.stack(faces) if faces else np.zeros((0, 256, 256, 3), np.uint8))
            faces = np.load(f, mmap_mode="r")
            if not len(faces):
                log(f"  no face in {item.path.name}; skipped")
            out += [(i, (f, k)) for k in range(len(faces))]
        return out

    def forward(self, payloads, train, rng):
        faces = [np.load(f, mmap_mode="r")[k] for f, k in payloads]
        if train:
            faces = [np.ascontiguousarray(x[:, ::-1]) if rng.random() < 0.5 else np.asarray(x) for x in faces]
        pixels = self.det.preprocess([np.asarray(x) for x in faces]).to(self.det.device)
        return self.det.network(pixels).softmax(dim=-1)[:, 1]


def _adapter(det, frames, cache):
    from forensix.audio.detector import AudioDetector
    from forensix.image.detector import ImageDetector
    from forensix.video.gend import GenDDetector

    if isinstance(det, ImageDetector):
        return _Image(det)
    if isinstance(det, AudioDetector):
        return _Audio(det)
    if isinstance(det, GenDDetector):
        return _GenD(det, frames, cache)
    raise ValueError(f"{det.name} can't be fine-tuned: supported are image and audio detectors "
                     "and the GenD video detectors (gend, gend-pe, dfdet-clip)")


# -- training -------------------------------------------------------------------------------------

def _split(items, val_fraction: float, seed: int):
    """Hold out whole groups until ``val_fraction`` of the items are held out."""
    groups = sorted({it.group for it in items})
    random.Random(seed).shuffle(groups)
    val_groups, n_val = set(), 0
    for g in groups:
        if n_val >= val_fraction * len(items):
            break
        val_groups.add(g)
        n_val += sum(it.group == g for it in items)
    return [it for it in items if it.group not in val_groups], [it for it in items if it.group in val_groups]


def _auc(scores, labels) -> float | None:
    from forensix.metrics import roc_auc

    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    s, y = s[np.isfinite(s)], y[np.isfinite(s)]   # videos without a face have no score
    return round(float(roc_auc(s[y], s[~y])), 4) if y.any() and (~y).any() else None


@torch.no_grad()
def _score_items(adapter, samples, n_items, batch_size) -> np.ndarray:
    """Per-item fake probability: the mean over an item's samples (one, or one per face)."""
    adapter.det.network.eval()
    sums, counts = np.zeros(n_items), np.zeros(n_items)
    for i in range(0, len(samples), batch_size):
        chunk = samples[i : i + batch_size]
        probs = adapter.forward([p for _, p in chunk], False, None).float().cpu().numpy()
        for (item, _), p in zip(chunk, probs):
            sums[item] += p
            counts[item] += 1
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def finetune(model, dataset=None, real_media=None, fake_media=None, out: str | Path | None = None,
             mode: str = "norm", epochs: int = 5, lr: float | None = None, batch_size: int = 16,
             val_fraction: float = 0.2, max_per_class: int | None = None, frames: int = 16,
             augment: bool = True, seed: int = 0, device: str = "cpu", cache_dir: str | Path | None = None,
             lora_rank: int = 8, lora_alpha: float = 16, lora_dropout: float = 0.05,
             lora_target_pattern: str | None = None, log=print):
    """Fine-tune ``model`` (a name, or a loaded detector) and return ``(detector, report)``.

    Saves the checkpoint to ``out`` (``.safetensors``) when given. See the module docstring.
    """
    import forensix
    from forensix.dataset import Item, extract_dataset, read_dataset

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    det = forensix.load(model, device=device) if isinstance(model, str) else model
    tmp = tempfile.TemporaryDirectory()
    if dataset is not None:
        root = Path(dataset)
        if root.is_file():
            root = extract_dataset(root, Path(tmp.name) / "data")
        items = read_dataset(root).items
    elif real_media is not None and fake_media is not None:
        items = ([Item(Path(m), "real", str(m)) for m in real_media]
                 + [Item(Path(m), "fake", str(m)) for m in fake_media])
    else:
        raise ValueError("give dataset=... or both real_media and fake_media")
    if max_per_class:
        items = list(items)
        rng.shuffle(items)
        items = [it for it in items if it.label == "real"][:max_per_class] + \
                [it for it in items if it.label == "fake"][:max_per_class]
    train_items, val_items = _split(items, val_fraction, seed)
    warnings = []
    for part, its in (("training", train_items), ("held-out", val_items)):
        if not {"real", "fake"} <= {it.label for it in its}:
            raise ValueError(f"the {part} split needs both real and fake items; add data or change val_fraction")
    if min(sum(it.label == c for it in val_items) for c in ("real", "fake")) < 20:
        warnings.append("fewer than 20 held-out items in a class: the held-out AUC is rough")

    cache = Path(cache_dir) if cache_dir else Path(tmp.name) / "faces"
    cache.mkdir(parents=True, exist_ok=True)
    adapter = _adapter(det, frames, cache)
    log(f"{det.name}: preparing {len(train_items)} training and {len(val_items)} held-out items")
    train = adapter.samples(train_items, log)
    val = adapter.samples(val_items, log)
    y_train = torch.tensor([train_items[i].is_fake for i, _ in train], dtype=torch.float32)
    y_val = [it.is_fake for it in val_items]

    base_auc = _auc(_score_items(adapter, val, len(val_items), batch_size), y_val)
    log(f"  base model, held-out AUC {base_auc}")

    lora = None
    if mode == "lora":
        head_names = set(trainable_parameters(det.network, "head"))
        lora = add_lora(det.network, lora_rank, lora_alpha, lora_dropout,
                        lora_targets(det.network, lora_target_pattern))
        log(f"  LoRA rank {lora_rank} on {len(lora['targets'])} layers")
        params = {n: p for n, p in det.network.named_parameters() if "lora_" in n or n in head_names}
    else:
        params = trainable_parameters(det.network, mode)
    initial = {n: p.detach().cpu().clone() for n, p in params.items()}   # restored if no epoch helps
    for p in det.network.parameters():
        p.requires_grad_(False)
    for p in params.values():
        p.requires_grad_(True)
    lr = lr or DEFAULT_LR[mode]
    optimizer = torch.optim.AdamW(params.values(), lr=lr, weight_decay=0.0 if mode == "norm" else 0.01)
    n_fake = float(y_train.sum())
    pos_weight = (len(y_train) - n_fake) / max(n_fake, 1.0)   # balance the classes
    use_amp = det.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best = {"auc": base_auc if base_auc is not None else -1.0, "epoch": 0, "state": None}
    history = []
    for epoch in range(1, epochs + 1):
        start, order, losses = time.time(), list(range(len(train))), []
        rng.shuffle(order)
        _train_mode(det.network)
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            y = y_train[idx].to(det.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                probs = adapter.forward([train[j][1] for j in idx], augment, rng)
            probs = probs.float().clamp(1e-6, 1 - 1e-6)
            weights = torch.where(y > 0.5, torch.full_like(y, pos_weight), torch.ones_like(y))
            loss = nn.functional.binary_cross_entropy(probs, y, weight=weights)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        val_auc = _auc(_score_items(adapter, val, len(val_items), batch_size), y_val)
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4), "val_auc": val_auc,
                        "seconds": round(time.time() - start, 1)})
        log(f"  epoch {epoch}/{epochs}: loss {history[-1]['train_loss']}, held-out AUC {val_auc}")
        if val_auc is not None and val_auc > best["auc"]:
            best = {"auc": val_auc, "epoch": epoch,
                    "state": {n: p.detach().cpu().clone() for n, p in params.items()}}

    for p in det.network.parameters():
        p.requires_grad_(False)
    if best["state"] is None:
        warnings.append("no epoch beat the base model on held-out data: the checkpoint keeps the base weights")
        best["state"] = initial
        best["auc"], best["epoch"] = base_auc, 0
    det.network.load_state_dict(best["state"], strict=False)
    saved = None
    if lora:   # fold the adapters in: the returned detector runs at the original speed
        saved = {n: v for n, v in best["state"].items() if "lora_" not in n} | merge_lora(det.network)
    elif mode != "full":
        saved = best["state"]
    det.network.eval()
    report = {
        "model": det.name, "mode": mode, "lr": lr, "epochs": epochs, "best_epoch": best["epoch"],
        "base_val_auc": base_auc, "val_auc": best["auc"],
        "n_train": {"real": sum(not it.is_fake for it in train_items), "fake": sum(it.is_fake for it in train_items)},
        "n_val": {"real": sum(not it.is_fake for it in val_items), "fake": sum(it.is_fake for it in val_items)},
        "trained_parameters": int(sum(p.numel() for p in params.values())),
        "lora": {k: v for k, v in lora.items() if k != "targets"} | {"n_targets": len(lora["targets"])} if lora else None,
        "history": history, "warnings": warnings,
    }
    if out:
        save(det, out, mode, report, saved, lora)
        report["checkpoint"] = str(out)
    tmp.cleanup()
    return det, report


def save(detector, path: str | Path, mode: str, report: dict, tensors: dict | None = None,
         lora: dict | None = None):
    """Write a fine-tune checkpoint: the changed tensors (``head``/``norm``), the head and adapters
    (``lora``), or every tensor (``full``)."""
    from safetensors.torch import save_file

    from forensix.registry import get_spec

    spec = get_spec(detector.name)
    state = tensors if tensors is not None else detector.network.state_dict()
    state = {k: v.detach().cpu().contiguous() for k, v in state.items()}
    info = {"model": detector.name, "mode": mode, "base_sha256": spec.weights.sha256 if spec.weights else None,
            "report": {k: v for k, v in report.items() if k != "history"}}
    if lora:
        info["lora"] = {k: v for k, v in lora.items() if k != "targets"}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(path), metadata={METADATA_KEY: json.dumps(info)})


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m forensix.finetuning",
                                     description="Fine-tune a pretrained ForensiX detector on labelled data.")
    parser.add_argument("model", help="e.g. commfor, univfd, aasist, nes2net-x-itw, dfdet-clip")
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--dataset", help="ForensiX dataset: folder or ZIP (manifest.csv, or real/ and fake/)")
    data.add_argument("--real", help="folder of real files (use with --fake)")
    parser.add_argument("--fake", help="folder of fake files")
    parser.add_argument("--out", required=True, help="checkpoint to write (.safetensors)")
    parser.add_argument("--mode", choices=MODES, default="norm")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-per-class", type=int)
    parser.add_argument("--frames", type=int, default=16, help="video: frames sampled per video")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--cache-dir", help="video: keep extracted face crops here (resumable)")
    parser.add_argument("--lora-rank", type=int, default=8, help="lora: adapter rank r")
    parser.add_argument("--lora-alpha", type=float, default=16, help="lora: scaling numerator (scale = alpha / r)")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", help="lora: regex of module names to adapt (default: every Linear but the head)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.out and not args.out.endswith(".safetensors"):
        parser.error("--out must end in .safetensors")
    real = fake = None
    if args.real:
        if not args.fake:
            parser.error("--real needs --fake")
        from forensix.dataset import MEDIA_SUFFIXES

        def files(d):
            return sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in MEDIA_SUFFIXES)
        real, fake = files(args.real), files(args.fake)
    _, report = finetune(args.model, dataset=args.dataset, real_media=real, fake_media=fake, out=args.out,
                         mode=args.mode, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                         val_fraction=args.val_fraction, max_per_class=args.max_per_class, frames=args.frames,
                         augment=args.augment, seed=args.seed, device=args.device, cache_dir=args.cache_dir,
                         lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                         lora_target_pattern=args.lora_targets)
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nheld-out AUC {report['base_val_auc']} -> {report['val_auc']} (epoch {report['best_epoch']}); "
          f"{args.out} and its report {Path(args.out).with_suffix('.json')}")
    for w in report["warnings"]:
        print("warning:", w)


if __name__ == "__main__":
    main()
