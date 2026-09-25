import numpy as np
import pytest

import forensix
from forensix.audio.detector import split_windows
from forensix.audio.io import load_audio
from forensix.metrics import compute_eer
from forensix.types import Prediction


def test_registry():
    assert {"aasist", "aasist-l"} <= set(forensix.list_models("audio"))
    assert forensix.list_models("graph") == []
    assert forensix.info("aasist").license == "MIT"
    with pytest.raises(KeyError, match="Unknown model"):
        forensix.info("nope")


def test_prediction_label():
    assert Prediction("m", 0.7).label == "fake"
    assert Prediction("m", 0.2).label == "real"
    assert Prediction("m", 0.2, threshold=0.1).is_fake


def test_split_windows_short_clip_is_repeat_padded():
    w = split_windows(np.arange(3, dtype=np.float32), 7)
    assert w.tolist() == [[0, 1, 2, 0, 1, 2, 0]]


def test_split_windows_long_clip_covers_tail():
    w = split_windows(np.arange(10, dtype=np.float32), 4)
    assert w.shape == (3, 4)
    assert w[-1].tolist() == [6, 7, 8, 9]
    assert split_windows(np.arange(8, dtype=np.float32), 4).shape == (2, 4)


def test_load_audio_resamples_and_downmixes(tmp_path):
    import soundfile as sf

    stereo = np.random.default_rng(0).uniform(-0.5, 0.5, (8000, 2)).astype(np.float32)
    sf.write(tmp_path / "a.wav", stereo, 8000)
    x = load_audio(tmp_path / "a.wav", target_sr=16000)
    assert x.dtype == np.float32 and x.shape == (16000,)
    with pytest.raises(ValueError, match="sample_rate"):
        load_audio(stereo, target_sr=16000)


def test_eer():
    assert compute_eer([0.9, 0.8], [0.1, 0.2])[0] == 0.0
    assert compute_eer([0.1, 0.2], [0.9, 0.8])[0] == 1.0
    eer, _ = compute_eer([0.9, 0.3], [0.1, 0.5])
    assert eer == pytest.approx(0.5)


def test_convert_fairseq_state_dict():
    from forensix.audio.xlsr import convert_fairseq_state_dict

    p = "ssl_model.model."
    fairseq = {
        "module.pos_S": 0,
        p + "feature_extractor.conv_layers.3.0.weight": 1,
        p + "feature_extractor.conv_layers.3.2.1.bias": 2,
        p + "layer_norm.weight": 3,
        p + "post_extract_proj.bias": 4,
        p + "encoder.pos_conv.0.weight_g": 5,
        p + "encoder.layers.7.self_attn.q_proj.weight": 6,
        p + "encoder.layers.7.self_attn_layer_norm.bias": 7,
        p + "encoder.layers.7.fc2.weight": 8,
        p + "encoder.layers.7.final_layer_norm.weight": 9,
        p + "encoder.layer_norm.bias": 10,
        p + "quantizer.vars": 11,
        p + "mask_emb": 12,
    }
    assert convert_fairseq_state_dict(fairseq) == {
        "pos_S": 0,
        p + "feature_extractor.conv_layers.3.conv.weight": 1,
        p + "feature_extractor.conv_layers.3.layer_norm.bias": 2,
        p + "feature_projection.layer_norm.weight": 3,
        p + "feature_projection.projection.bias": 4,
        p + "encoder.pos_conv_embed.conv.parametrizations.weight.original0": 5,
        p + "encoder.layers.7.attention.q_proj.weight": 6,
        p + "encoder.layers.7.layer_norm.bias": 7,
        p + "encoder.layers.7.feed_forward.output_dense.weight": 8,
        p + "encoder.layers.7.final_layer_norm.weight": 9,
        p + "encoder.layer_norm.bias": 10,
    }


def test_split_windows_without_padding_keeps_short_clips_whole():
    x = np.arange(3, dtype=np.float32)
    assert split_windows(x, 7, pad_short=False).tolist() == [[0, 1, 2]]
    assert split_windows(np.arange(10, dtype=np.float32), 4, pad_short=False).shape == (3, 4)


def test_roc_auc_and_average_precision():
    from forensix.metrics import average_precision, roc_auc

    assert roc_auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert roc_auc([0.5], [0.5]) == 0.5
    assert roc_auc([0.9, 0.3], [0.5, 0.1]) == 0.75
    assert average_precision([0.9, 0.8], [0.1]) == 1.0
    # ranking: fake 0.9, real 0.5, fake 0.3 -> AP = 0.5*1 + 0.5*(2/3)
    assert average_precision([0.9, 0.3], [0.5]) == pytest.approx(0.5 + 1 / 3)
