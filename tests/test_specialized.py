from pathlib import Path
import tempfile

import cv2
import numpy as np
import pytest
import torch

import forensix
from forensix.audio.io import load_audio
from forensix.specialized import (
    DIM_AUDIO,
    DIM_AUDIOVISUAL,
    DIM_IMAGE,
    DIM_VIDEO,
    PersonDetector,
    TinyPersonModel,
    extract_audio_features,
    extract_audiovisual_features,
    extract_features,
    extract_image_features,
    extract_video_features,
    generate_synthetic_perturbations,
    load_person_model,
    pretrain_base_model,
    save_person_model,
    train_person_model,
)

DATA = Path(__file__).parent / "data"
ARMSTRONG = DATA / "face_public_domain.jpg"
ARMSTRONG_2 = DATA / "face2_public_domain.jpg"
ALDRIN = DATA / "other_face_public_domain.jpg"
SPEECH = DATA / "speech_cc0.flac"


def _make_test_video(path: Path, img_path: Path, num_frames: int = 15) -> Path:
    img = cv2.imread(str(img_path))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (img.shape[1], img.shape[0]))
    for _ in range(num_frames):
        writer.write(img)
    writer.release()
    return path


def test_tiny_model_architecture_and_param_count():
    model = TinyPersonModel(in_dim=DIM_IMAGE, hidden_dim=64, embed_dim=64, lora_rank=4, modality="image")
    params = model.get_param_count()

    # Must be a teeny tiny model: under 30k total parameters
    assert params["total"] < 30_000
    # PEFT LoRA parameters must be under 3,000 parameters
    assert params["lora"] < 3_000

    x = torch.randn(2, DIM_IMAGE)
    art_prob, emb = model(x)
    assert art_prob.shape == (2,)
    assert (art_prob >= 0.0).all() and (art_prob <= 1.0).all()
    assert emb.shape == (2, 64)
    # Unit norm embeddings
    assert torch.allclose(torch.norm(emb, dim=-1), torch.ones(2), atol=1e-5)


def test_lora_peft_freezing():
    model = TinyPersonModel(in_dim=DIM_IMAGE, hidden_dim=64, embed_dim=64, lora_rank=4)

    model.enable_peft()
    for name, param in model.named_parameters():
        if "lora" in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False

    model.disable_peft()
    for param in model.parameters():
        assert param.requires_grad is True


def test_feature_extractors_dimensions():
    # Image
    img_feat = extract_image_features(ARMSTRONG)
    assert img_feat.shape == (DIM_IMAGE,)
    assert img_feat.dtype == np.float32

    # Audio
    aud_feat = extract_audio_features(SPEECH)
    assert aud_feat.shape == (DIM_AUDIO,)
    assert aud_feat.dtype == np.float32

    # Synthetic perturbations
    negs = generate_synthetic_perturbations(img_feat, "image", count=5)
    assert len(negs) == 5
    assert negs[0].shape == (DIM_IMAGE,)

    # Unified dispatcher
    assert extract_features(ARMSTRONG, "image").shape == (DIM_IMAGE,)
    assert extract_features(SPEECH, "audio").shape == (DIM_AUDIO,)


def test_image_person_training_and_verification(tmp_path):
    # Train specialized model for Neil Armstrong on CPU
    detector = train_person_model(
        person_name="armstrong",
        real_media=[ARMSTRONG],
        negative_media=[ALDRIN],
        modality="image",
        use_peft=True,
        epochs=25,
        device="cpu",
    )

    # Armstrong's second photo should verify as REAL
    pred_real = detector.predict(ARMSTRONG_2)
    assert pred_real.is_fake is False
    assert pred_real.label == "real"
    assert pred_real.model == "person-detector"
    assert pred_real.fake_prob < detector.threshold

    # Impostor Aldrin should verify as FAKE
    pred_fake = detector.predict(ALDRIN)
    assert pred_fake.is_fake is True
    assert pred_fake.label == "fake"
    assert pred_fake.fake_prob >= detector.threshold


def test_audio_person_training_and_verification():
    real_audio = load_audio(SPEECH, target_sr=16000)

    # Synthesize impostor voice waveform
    t = np.arange(len(real_audio)) / 16000.0
    fake_audio = (np.sin(2 * np.pi * 320 * t) + np.random.normal(0, 0.4, len(real_audio))).astype(np.float32)

    detector = train_person_model(
        person_name="speaker_a",
        real_media=[real_audio],
        negative_media=[fake_audio],
        modality="audio",
        use_peft=True,
        epochs=20,
        device="cpu",
    )

    p_real = detector.predict(real_audio)
    p_fake = detector.predict(fake_audio)

    assert p_real.is_fake is False
    assert p_fake.is_fake is True
    assert p_fake.fake_prob > p_real.fake_prob


def test_video_person_training_and_segment_predictions(tmp_path):
    video_path = _make_test_video(tmp_path / "armstrong.avi", ARMSTRONG, num_frames=12)

    detector = train_person_model(
        person_name="armstrong",
        real_media=[video_path],
        modality="video",
        epochs=15,
        device="cpu",
    )

    pred = detector.predict(video_path)
    assert pred.is_fake is False
    assert len(pred.segment_fake_probs) > 0
    assert "frame_times_s" in pred.details


def test_pretrain_base_and_peft_finetune():
    # 1. Pretrain base model on public real vs fake subsets
    base = pretrain_base_model(
        real_media=[ARMSTRONG],
        fake_media=[ALDRIN],
        modality="image",
        epochs=15,
        device="cpu",
    )

    # 2. Specialize with PEFT on target individual
    detector = train_person_model(
        person_name="armstrong",
        real_media=[ARMSTRONG],
        negative_media=[ALDRIN],
        modality="image",
        base_model=base,
        use_peft=True,
        epochs=25,
        device="cpu",
    )

    pred_real = detector.predict(ARMSTRONG_2)
    pred_fake = detector.predict(ALDRIN)

    assert pred_real.is_fake is False
    assert pred_fake.is_fake is True


def test_safetensors_serialization_and_forensix_load(tmp_path):
    detector = train_person_model(
        person_name="armstrong",
        real_media=[ARMSTRONG],
        negative_media=[ALDRIN],
        modality="image",
        epochs=15,
        device="cpu",
    )

    save_path = tmp_path / "model.safetensors"
    detector.save(save_path)
    assert save_path.exists()

    # Load via forensix.load
    loaded = forensix.load("person-detector", weights_path=save_path)
    assert isinstance(loaded, PersonDetector)
    assert loaded.person_name == "armstrong"
    assert loaded.modality == "image"

    pred_orig = detector.predict(ARMSTRONG_2)
    pred_loaded = loaded.predict(ARMSTRONG_2)

    assert pred_orig.is_fake == pred_loaded.is_fake
    assert abs(pred_orig.fake_prob - pred_loaded.fake_prob) < 1e-4

    # Calling forensix.load without weights_path raises clear ValueError
    with pytest.raises(ValueError, match="requires a trained checkpoint"):
        forensix.load("person-detector")


def test_forensix_person_alias_and_registry():
    assert forensix.person is forensix.specialized
    assert "person-detector" in forensix.list_models()
    assert "person-detector" in forensix.list_models(family="specialized")
    spec = forensix.info("person-detector")
    assert spec.family == "specialized"
