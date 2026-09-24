"""XLS-R + AASIST (Tak et al., Odyssey 2022), fine-tuned on ASVspoof 2019 LA.

Upstream checkpoints store the XLS-R front end in fairseq's layout; they are
converted to Hugging Face transformers' Wav2Vec2Model layout on load.
"""

import re
from pathlib import Path

import torch

from forensix.audio.detector import AudioDetector
from forensix.hub import WeightFile
from forensix.registry import ModelSpec, register

_SSL = "ssl_model.model."
_DRIVE = "https://drive.usercontent.google.com/download?export=download&confirm=t&id="

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
    """Map an upstream checkpoint's keys onto ``network.Model``."""
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


class XLSRAASISTDetector(AudioDetector):
    def _fake_probs(self, windows: torch.Tensor) -> torch.Tensor:
        logits = self.network(windows)
        # upstream label convention: class 1 = bonafide, class 0 = spoof
        return logits.softmax(dim=-1)[:, 0]


def _builder(name: str):
    def build(weights_path: Path, device: str = "cpu") -> XLSRAASISTDetector:
        # imported lazily: pulls in transformers
        from forensix.audio.xlsr_aasist.network import Model

        network = Model(None, device)
        state = torch.load(weights_path, map_location="cpu", weights_only=True, mmap=True)
        network.load_state_dict(convert_fairseq_state_dict(state))
        return XLSRAASISTDetector(name, network, window=64600, device=device)

    return build


_COMMON = dict(
    family="audio",
    license="MIT (code, weights); XLS-R base model Apache-2.0",
    source="https://github.com/TakHemlata/SSL_Anti-spoofing/tree/4acaa61dcef5f7610f43aa4d0b29c4559b970cd2",
    paper="https://arxiv.org/abs/2202.12233",
)

register(
    ModelSpec(
        name="xlsr-aasist",
        builder=_builder("xlsr-aasist"),
        weights=WeightFile(
            url=_DRIVE + "11vFBNKzYUtWqdK358_JEygawFzmmaZjO",
            sha256="bd6f36097259fe54e7004eb983651e5304d807be81156dbd04faccb70d91e10c",
            filename="xlsr-aasist.pth",
        ),
        description="wav2vec 2.0 XLS-R (300M) front end + AASIST back end, fine-tuned "
        "on ASVspoof 2019 LA with RawBoost (LA recipe, 316M params). Published EER "
        "0.82% on ASVspoof 2021 LA eval. Needs a GPU for practical speed.",
        **_COMMON,
    )
)

register(
    ModelSpec(
        name="xlsr-aasist-df",
        builder=_builder("xlsr-aasist-df"),
        weights=WeightFile(
            url=_DRIVE + "1JHBClArVdM-Cr1b8In1iakTV_TvC3HvG",
            sha256="1cf904f1d84c867c278cd42161df5367939d61cc28bfefd239bc995af59c2804",
            filename="xlsr-aasist-df.pth",
        ),
        description="XLS-R + AASIST fine-tuned on ASVspoof 2019 LA with the DF-track "
        "RawBoost recipe (316M params). Published EER 2.85% on ASVspoof 2021 DF eval, "
        "which covers compressed, vocoded deepfakes from many sources.",
        **_COMMON,
    )
)
