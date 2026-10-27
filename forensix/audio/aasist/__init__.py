"""AASIST and AASIST-L (Jung et al., ICASSP 2022), pretrained on ASVspoof 2019 LA."""

from pathlib import Path

import torch

from forensix.audio.aasist.network import Model
from forensix.audio.detector import AudioDetector
from forensix.hub import WeightFile
from forensix.registry import ModelSpec, register

_COMMIT = "a04c9863f63d44471dde8a6abcb3b082b07cd1d1"
_WEIGHTS_URL = f"https://raw.githubusercontent.com/clovaai/aasist/{_COMMIT}/models/weights"

# model_config blocks from config/AASIST.conf and config/AASIST-L.conf
AASIST_CONFIG = {
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}
AASIST_L_CONFIG = {
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
    "gat_dims": [24, 32],
    "pool_ratios": [0.4, 0.5, 0.7, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


class AASISTDetector(AudioDetector):
    def _fake_probs(self, windows: torch.Tensor) -> torch.Tensor:
        _, logits = self.network(windows)
        # upstream label convention: class 1 = bonafide, class 0 = spoof
        return logits.softmax(dim=-1)[:, 0]


def _builder(name: str, config: dict):
    def build(weights_path: Path, device: str = "cpu", clip_score: str = "mean") -> AASISTDetector:
        network = Model(config)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        network.load_state_dict(state)
        return AASISTDetector(name, network, window=config["nb_samp"], device=device, clip_score=clip_score)

    return build


_COMMON = dict(
    family="audio",
    license="MIT",
    source=f"https://github.com/clovaai/aasist/tree/{_COMMIT}",
    paper="https://arxiv.org/abs/2110.01200",
)

register(
    ModelSpec(
        name="aasist",
        builder=_builder("aasist", AASIST_CONFIG),
        weights=WeightFile(
            url=f"{_WEIGHTS_URL}/AASIST.pth",
            sha256="51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0",
            filename="aasist.pth",
        ),
        description="Spectro-temporal graph attention on raw waveforms (297K params). "
        "Trained on ASVspoof 2019 LA; published EER 0.83% on LA eval.",
        **_COMMON,
    )
)

register(
    ModelSpec(
        name="aasist-l",
        builder=_builder("aasist-l", AASIST_L_CONFIG),
        weights=WeightFile(
            url=f"{_WEIGHTS_URL}/AASIST-L.pth",
            sha256="814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a",
            filename="aasist-l.pth",
        ),
        description="Lightweight AASIST (85K params). Trained on ASVspoof 2019 LA; "
        "published EER 0.99% on LA eval.",
        **_COMMON,
    )
)
