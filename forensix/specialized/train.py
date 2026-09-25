"""Training, PEFT fine-tuning, and serialization for person-specialized detectors.

Enables fast, CPU-friendly training of teeny tiny models specialized for one individual.
Supports training from images, videos, audio, or audiovisual media.
"""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from forensix.identity.media import is_image
from forensix.specialized.features import (
    DIM_AUDIO,
    DIM_AUDIOVISUAL,
    DIM_IMAGE,
    DIM_VIDEO,
    extract_features,
    generate_synthetic_perturbations,
)
from forensix.specialized.model import TinyPersonModel

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def detect_modality(media_item: Any) -> str:
    """Infer modality from file extension or structure."""
    if isinstance(media_item, np.ndarray):
        if media_item.ndim == 1:
            return "audio"
        if media_item.ndim == 3 and media_item.shape[2] == 3:
            return "image"
        if media_item.ndim == 4:
            return "video"
        return "image"

    path_str = str(media_item).lower()
    suffix = Path(path_str).suffix.lower()
    if is_image(path_str):
        return "image"
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    return "image"


def _prepare_media_list(media: Any) -> list[Any]:
    if isinstance(media, (str, Path, np.ndarray)):
        return [media]
    return list(media)


def pretrain_base_model(
    real_media: list[Any],
    fake_media: list[Any],
    modality: str = "image",
    epochs: int = 25,
    lr: float = 2e-3,
    hidden_dim: int = 64,
    embed_dim: int = 64,
    device: str = "cpu",
) -> TinyPersonModel:
    """Pre-train a base TinyPersonModel on general real vs deepfake datasets.

    Args:
        real_media: List of genuine media paths or arrays.
        fake_media: List of deepfake / manipulated media paths or arrays.
        modality: Modality ('image', 'audio', 'video', 'audiovisual').
        epochs: Number of training epochs on CPU.
        lr: Learning rate.
        hidden_dim: Hidden dimension (default 64).
        embed_dim: Embedding dimension (default 64).
        device: Device to train on ('cpu' by default).

    Returns:
        Trained base TinyPersonModel.
    """
    in_dim_map = {
        "image": DIM_IMAGE,
        "video": DIM_VIDEO,
        "audio": DIM_AUDIO,
        "audiovisual": DIM_AUDIOVISUAL,
    }
    in_dim = in_dim_map.get(modality, DIM_IMAGE)

    real_feats = [extract_features(m, modality) for m in real_media]
    fake_feats = [extract_features(m, modality) for m in fake_media]

    # Augment with positive jitter and synthetic perturbations to prevent overfitting on small sets
    augmented_pos = list(real_feats)
    augmented_neg = list(fake_feats)
    rng = np.random.default_rng(42)
    for rf in real_feats:
        for _ in range(8):
            jitter = rng.normal(0, 0.04, size=rf.shape).astype(np.float32)
            p_aug = rf + jitter
            augmented_pos.append((p_aug / (np.linalg.norm(p_aug) + 1e-6)).astype(np.float32))
        augmented_neg.extend(generate_synthetic_perturbations(rf, modality, count=6))

    X_real = np.stack(augmented_pos)
    X_fake = np.stack(augmented_neg)

    X = np.concatenate([X_real, X_fake], axis=0)
    y = np.concatenate([np.zeros(len(X_real), dtype=np.float32), np.ones(len(X_fake), dtype=np.float32)])

    model = TinyPersonModel(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        lora_rank=4,
        modality=modality,
    ).to(device)

    # Fit feature normalization statistics with safe standard deviation bounds
    mean_np = np.mean(X, axis=0)
    std_np = np.clip(np.std(X, axis=0), 0.05, None)
    model.feat_mean.copy_(torch.as_tensor(mean_np, dtype=torch.float32, device=device))
    model.feat_std.copy_(torch.as_tensor(std_np, dtype=torch.float32, device=device))

    model.disable_peft()  # Base pre-training trains all weights
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()

    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=device)

    n_samples = len(X)
    batch_size = min(32, n_samples)

    for _ in range(epochs):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            idx = perm[i : i + batch_size]
            bx, by = X_t[idx], y_t[idx]

            artifact_prob, _ = model(bx)
            loss = F.binary_cross_entropy(artifact_prob, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def train_person_model(
    person_name: str,
    real_media: list[Any] | Any,
    negative_media: list[Any] | Any | None = None,
    modality: str = "auto",
    base_model: TinyPersonModel | str | Path | None = None,
    use_peft: bool = True,
    lora_rank: int = 4,
    epochs: int = 30,
    lr: float = 2e-3,
    device: str = "cpu",
    synthetic_perturbations_per_sample: int = 6,
    verbose: bool = False,
):
    """Train or PEFT fine-tune a teeny tiny model specialized for one target individual.

    Can be trained entirely on CPU in seconds from even a few images, audio clips,
    videos, or audiovisual recordings.

    Args:
        person_name: Name or ID of the enrolled individual (e.g. 'neil_armstrong').
        real_media: Genuine media of the individual (paths or arrays).
        negative_media: Optional media of impostors or known deepfakes. If not provided
            or limited, realistic synthetic manipulations are automatically synthesized.
        modality: 'image', 'audio', 'video', 'audiovisual', or 'auto'.
        base_model: Optional pre-trained TinyPersonModel or path to base weights.
        use_peft: If True, uses Parameter-Efficient Fine-Tuning (LoRA), keeping base weights
            frozen and adapting only low-rank matrices. If False, fine-tunes all weights.
        lora_rank: LoRA rank (default 4).
        epochs: Training epochs on CPU (default 30).
        lr: Learning rate (default 2e-3).
        device: 'cpu' or 'cuda'.
        synthetic_perturbations_per_sample: Number of synthetic manipulation / impostor
            samples to generate per real sample.
        verbose: If True, prints progress.

    Returns:
        PersonDetector instance.
    """
    from forensix.specialized.detector import PersonDetector

    real_list = _prepare_media_list(real_media)
    if not real_list:
        raise ValueError(f"No real media provided for {person_name!r}")

    if modality == "auto":
        modality = detect_modality(real_list[0])

    in_dim_map = {
        "image": DIM_IMAGE,
        "video": DIM_VIDEO,
        "audio": DIM_AUDIO,
        "audiovisual": DIM_AUDIOVISUAL,
    }
    in_dim = in_dim_map.get(modality, DIM_IMAGE)

    if verbose:
        print(f"[forensix.specialized] Extracting features for {person_name!r} ({modality})...")

    # Extract real features
    real_feats = [extract_features(m, modality) for m in real_list]

    # Augment real features with mild positive variations to prevent single-sample memorization
    augmented_pos = list(real_feats)
    rng = np.random.default_rng(42)
    for rf in real_feats:
        for _ in range(8):
            jitter = rng.normal(0, 0.04, size=rf.shape).astype(np.float32)
            p_aug = rf + jitter
            augmented_pos.append((p_aug / (np.linalg.norm(p_aug) + 1e-6)).astype(np.float32))
    real_feats_arr = np.stack(augmented_pos)

    # Collect or generate negative features with impostor and artifact tracking
    neg_feats_list: list[np.ndarray] = []
    is_impostor_list: list[bool] = []
    target_art_list: list[float] = []

    if negative_media is not None:
        neg_list = _prepare_media_list(negative_media)
        for m in neg_list:
            neg_feats_list.append(extract_features(m, modality))
            is_impostor_list.append(True)
            # Provided negative media (e.g. impostor or synthetic sample)
            target_art_list.append(1.0)

    # Add synthetic manipulation & impostor perturbations
    if synthetic_perturbations_per_sample > 0 or len(neg_feats_list) == 0:
        count = max(synthetic_perturbations_per_sample, 4)
        for rf in real_feats:
            p_feats, p_imp = generate_synthetic_perturbations(rf, modality, count=count, return_impostor_mask=True)
            neg_feats_list.extend(p_feats)
            is_impostor_list.extend(p_imp)
            target_art_list.extend([0.0 if imp else 1.0 for imp in p_imp])

    neg_feats_arr = np.stack(neg_feats_list)
    is_impostor_mask = torch.as_tensor(is_impostor_list, dtype=torch.bool, device=device)
    target_art_neg = torch.as_tensor(target_art_list, dtype=torch.float32, device=device)

    # Initialize or load base model
    has_prefitted_stats = False
    if isinstance(base_model, (str, Path)):
        model = load_person_model(base_model, device=device).model
        has_prefitted_stats = True
    elif isinstance(base_model, TinyPersonModel):
        model = base_model.to(device)
        has_prefitted_stats = bool(torch.norm(model.feat_std - 1.0).item() > 0.01)
    else:
        model = TinyPersonModel(
            in_dim=in_dim,
            hidden_dim=64,
            embed_dim=64,
            lora_rank=lora_rank,
            modality=modality,
        ).to(device)

    # Feature normalization: update only if not already pre-calibrated by base model
    if not has_prefitted_stats:
        all_feats = np.concatenate([real_feats_arr, neg_feats_arr], axis=0)
        feat_mean = np.mean(all_feats, axis=0)
        feat_std = np.clip(np.std(all_feats, axis=0), 0.05, None)
        model.feat_mean.copy_(torch.as_tensor(feat_mean, dtype=torch.float32, device=device))
        model.feat_std.copy_(torch.as_tensor(feat_std, dtype=torch.float32, device=device))

    # Configure PEFT / LoRA
    if use_peft:
        model.enable_peft()
    else:
        model.disable_peft()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    # Tensors
    X_pos = torch.as_tensor(real_feats_arr, dtype=torch.float32, device=device)
    X_neg = torch.as_tensor(neg_feats_arr, dtype=torch.float32, device=device)

    model.train()
    for ep in range(epochs):
        # 1. Forward real samples
        pos_art, pos_emb = model(X_pos)
        curr_proto = F.normalize(pos_emb.mean(dim=0, keepdim=True), p=2, dim=-1)

        # 2. Forward negative samples
        neg_art, neg_emb = model(X_neg)

        # 3. Artifact classification loss: Real = 0 (clean), Manipulations = 1 (artifact)
        loss_art_pos = F.binary_cross_entropy(pos_art, torch.zeros_like(pos_art))
        loss_art_neg = F.binary_cross_entropy(neg_art, target_art_neg)
        loss_art = 0.5 * (loss_art_pos + loss_art_neg)

        # 4. Identity metric loss: Real should align with prototype
        sim_pos = (pos_emb @ curr_proto.T).squeeze(-1)
        loss_id_pos = F.relu(0.60 - sim_pos).mean()

        # Only impostors (different identities) should be pushed away from prototype
        if is_impostor_mask.any():
            sim_neg = (neg_emb[is_impostor_mask] @ curr_proto.T).squeeze(-1)
            loss_id_neg = F.relu(sim_neg - 0.15).mean()
        else:
            loss_id_neg = torch.tensor(0.0, device=device)

        loss = loss_art + 1.2 * (loss_id_pos + loss_id_neg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()

    # Register the final target person prototype
    with torch.no_grad():
        _, final_real_emb = model(X_pos)
        model.set_prototype(final_real_emb)

        # Calibrate prototype similarity threshold based on enrolled samples
        pos_sims = (final_real_emb @ model.prototype).tolist()
        mean_pos_sim = float(np.mean(pos_sims)) if isinstance(pos_sims, list) else float(pos_sims)
        calibrated_sim_thr = max(0.15, min(0.30, mean_pos_sim - 0.18))
        model.prototype_similarity_threshold.copy_(torch.tensor(calibrated_sim_thr, dtype=torch.float32))

        # Calibrate decision threshold on training sets
        scores_pos, _ = model.score(X_pos)
        scores_neg, _ = model.score(X_neg)

        calibrated_thr = 0.50
        model.decision_threshold.copy_(torch.tensor(calibrated_thr, dtype=torch.float32))

    if verbose:
        params_info = model.get_param_count()
        print(
            f"[forensix.specialized] Model trained for {person_name!r}! "
            f"Total params: {params_info['total']:,} (trainable: {params_info['trainable']:,}). "
            f"Calibrated threshold: {calibrated_thr}"
        )

    return PersonDetector(
        model=model,
        person_name=person_name,
        modality=modality,
        threshold=calibrated_thr,
        device=device,
    )


def save_person_model(path: str | PathLike, detector: Any) -> Path:
    """Save a specialized person detector to a safetensors file with embedded metadata."""
    from safetensors.torch import save_file

    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    model = detector.model
    state_dict = {k: v.cpu().contiguous() for k, v in model.state_dict().items()}

    metadata = {
        "person_name": str(detector.person_name),
        "modality": str(detector.modality),
        "threshold": str(detector.threshold),
        "in_dim": str(model.in_dim),
        "hidden_dim": str(model.hidden_dim),
        "embed_dim": str(model.embed_dim),
        "lora_rank": str(model.lora_rank),
        "forensix_type": "person-detector",
    }

    save_file(state_dict, str(target_path), metadata=metadata)
    return target_path


def load_person_model(path: str | PathLike, device: str = "cpu") -> Any:
    """Load a specialized person detector from a safetensors file."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    from forensix.specialized.detector import PersonDetector

    file_path = str(path)
    with safe_open(file_path, framework="pt") as f:
        meta = f.metadata() or {}

    in_dim = int(meta.get("in_dim", DIM_IMAGE))
    hidden_dim = int(meta.get("hidden_dim", 64))
    embed_dim = int(meta.get("embed_dim", 64))
    lora_rank = int(meta.get("lora_rank", 4))
    modality = str(meta.get("modality", "image"))
    person_name = str(meta.get("person_name", "unknown"))
    threshold = float(meta.get("threshold", 0.5))

    model = TinyPersonModel(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        lora_rank=lora_rank,
        modality=modality,
    )

    tensors = load_file(file_path, device=device)
    model.load_state_dict(tensors)
    model.to(device).eval()

    return PersonDetector(
        model=model,
        person_name=person_name,
        modality=modality,
        threshold=threshold,
        device=device,
    )
