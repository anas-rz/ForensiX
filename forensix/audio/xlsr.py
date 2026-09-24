"""Shared wav2vec 2.0 XLS-R (300M) front end for SSL-based detectors.

Upstream detectors (SSL_Anti-spoofing, Nes2Net, ...) wrap fairseq's XLS-R in an
``SSLModel`` module. ``SSLModel`` here is a drop-in replacement built on Hugging
Face transformers' Wav2Vec2Model, and ``convert_fairseq_state_dict`` maps
upstream checkpoints onto it.
"""

import re

from torch import nn

_SSL = "ssl_model.model."

# fairseq wav2vec2 -> transformers Wav2Vec2Model (layer-norm feature extractor,
# stable/pre-LN encoder). Applied in order; first match wins.
_KEY_RULES = [
    (r"feature_extractor\.conv_layers\.(\d+)\.0\.", r"feature_extractor.conv_layers.\1.conv."),
    (r"feature_extractor\.conv_layers\.(\d+)\.2\.1\.", r"feature_extractor.conv_layers.\1.layer_norm."),
    (r"^layer_norm\.", "feature_projection.layer_norm."),
    (r"^post_extract_proj\.", "feature_projection.projection."),
    (r"encoder\.pos_conv\.0\.weight_g$", "encoder.pos_conv_embed.conv.parametrizations.weight.original0"),
    (r"encoder\.pos_conv\.0\.weight_v$", "encoder.pos_conv_embed.conv.parametrizations.weight.original1"),
    (r"encoder\.pos_conv\.0\.bias$", "encoder.pos_conv_embed.conv.bias"),
    (r"\.self_attn\.", ".attention."),
    (r"\.self_attn_layer_norm\.", ".layer_norm."),
    (r"\.fc1\.", ".feed_forward.intermediate_dense."),
    (r"\.fc2\.", ".feed_forward.output_dense."),
]
# fairseq pretraining-only parameters, unused when extracting features
_DROP = re.compile(r"^(quantizer|project_q|final_proj|mask_emb|label_embs_concat)")


def convert_fairseq_state_dict(state: dict) -> dict:
    """Map an upstream checkpoint (fairseq XLS-R under ``ssl_model.model.``)
    onto a network that uses ``SSLModel`` as ``ssl_model``."""
    out = {}
    for key, value in state.items():
        key = key.removeprefix("module.")  # saved from nn.DataParallel
        if not key.startswith(_SSL):
            out[key] = value
            continue
        sub = key[len(_SSL) :]
        if _DROP.match(sub):
            continue
        for pattern, repl in _KEY_RULES:
            sub = re.sub(pattern, repl, sub)
        out[_SSL + sub] = value
    return out


def xlsr_300m_config():
    """facebook/wav2vec2-xls-r-300m architecture, spec-augment masking disabled."""
    from transformers import Wav2Vec2Config

    return Wav2Vec2Config(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        feat_extract_norm="layer",
        feat_extract_activation="gelu",
        conv_dim=[512] * 7,
        conv_kernel=[10, 3, 3, 3, 3, 2, 2],
        conv_stride=[5, 2, 2, 2, 2, 2, 2],
        conv_bias=True,
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
        do_stable_layer_norm=True,
        layer_norm_eps=1e-5,
        apply_spec_augment=False,
        mask_time_prob=0.0,
        mask_feature_prob=0.0,
    )


class SSLModel(nn.Module):
    """Same interface as upstream's fairseq-based SSLModel."""

    def __init__(self, device=None):
        super().__init__()
        from transformers import Wav2Vec2Model

        self.model = Wav2Vec2Model(xlsr_300m_config())
        self.device = device
        self.out_dim = 1024

    def extract_feat(self, input_data):
        # input should be in shape (batch, length)
        if input_data.ndim == 3:
            input_data = input_data[:, :, 0]
        # [batch, length, dim]
        return self.model(input_data).last_hidden_state
