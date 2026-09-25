from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from forensix.specialized import (
    DIM_AUDIO,
    DIM_IMAGE,
    TinyPersonModel,
    extract_features,
    train_person_model,
)

DATA = Path(__file__).parent.parent / "data"
ARMSTRONG = DATA / "face_public_domain.jpg"
ARMSTRONG_2 = DATA / "face2_public_domain.jpg"
ALDRIN = DATA / "other_face_public_domain.jpg"
SPEECH = DATA / "speech_cc0.flac"


def test_person_model_peft_gradient_flow_and_loss_decrease():
    """Verify that TinyPersonModel decreases loss and only updates LoRA parameters during PEFT."""
    torch.manual_seed(42)
    model = TinyPersonModel(in_dim=DIM_IMAGE, hidden_dim=64, embed_dim=64, lora_rank=4)
    model.enable_peft()

    # Create dummy positive and negative features
    rng = np.random.default_rng(42)
    pos_feat = rng.standard_normal((10, DIM_IMAGE)).astype(np.float32)
    pos_feat = pos_feat / np.linalg.norm(pos_feat, axis=1, keepdims=True)
    neg_feat = rng.standard_normal((10, DIM_IMAGE)).astype(np.float32)
    neg_feat = neg_feat / np.linalg.norm(neg_feat, axis=1, keepdims=True)

    X_pos = torch.tensor(pos_feat)
    X_neg = torch.tensor(neg_feat)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-3)

    losses = []
    for step in range(25):
        pos_art, pos_emb = model(X_pos)
        neg_art, neg_emb = model(X_neg)
        curr_proto = F.normalize(pos_emb.mean(dim=0, keepdim=True), p=2, dim=-1)

        loss_art = 0.5 * (
            F.binary_cross_entropy(pos_art, torch.zeros_like(pos_art))
            + F.binary_cross_entropy(neg_art, torch.ones_like(neg_art))
        )
        sim_pos = (pos_emb @ curr_proto.T).squeeze(-1)
        sim_neg = (neg_emb @ curr_proto.T).squeeze(-1)
        loss_id = F.relu(0.65 - sim_pos).mean() + F.relu(sim_neg - 0.15).mean()

        loss = loss_art + 1.2 * loss_id
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        # Check gradient flow:
        # Base weights must have no gradient (frozen by PEFT)
        assert model.fc1.weight.grad is None
        assert model.fc2.weight.grad is None
        assert model.artifact_head.weight.grad is None

        # LoRA weights must have active gradients
        assert model.fc1.lora_B.grad is not None
        assert model.fc1.lora_B.grad.norm().item() > 0
        if step > 0:
            assert model.fc1.lora_A.grad.norm().item() > 0

        optimizer.step()

    # The training loss must decrease significantly
    assert losses[-1] < losses[0] * 0.6, f"Loss did not decrease sufficiently: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_person_image_learning_separation():
    """Verify that train_person_model learns to separate genuine target from impostor on real media."""
    detector = train_person_model(
        person_name="armstrong",
        real_media=[ARMSTRONG],
        negative_media=[ALDRIN],
        modality="image",
        use_peft=True,
        epochs=25,
        device="cpu",
    )

    pred_real = detector.predict(ARMSTRONG_2)
    pred_impostor = detector.predict(ALDRIN)

    assert pred_real.fake_prob < detector.threshold
    assert pred_impostor.fake_prob >= detector.threshold
    assert pred_impostor.fake_prob - pred_real.fake_prob > 0.40


def test_person_audio_learning_separation():
    """Verify that train_person_model learns voice discrimination from genuine audio."""
    from forensix.audio.io import load_audio

    real_audio = load_audio(SPEECH, target_sr=16000)
    t = np.arange(len(real_audio)) / 16000.0
    impostor_audio = (np.sin(2 * np.pi * 320 * t) + np.random.normal(0, 0.4, len(real_audio))).astype(np.float32)

    detector = train_person_model(
        person_name="speaker_librivox",
        real_media=[real_audio],
        negative_media=[impostor_audio],
        modality="audio",
        use_peft=True,
        epochs=20,
        device="cpu",
    )

    p_real = detector.predict(real_audio)
    p_fake = detector.predict(impostor_audio)

    assert p_real.is_fake is False
    assert p_fake.is_fake is True
    assert p_fake.fake_prob > p_real.fake_prob + 0.35
