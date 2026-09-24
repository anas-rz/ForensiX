"""RawGAT-ST (Tak et al., ASVspoof 2021 Workshop), pretrained on ASVspoof 2019 LA."""

from pathlib import Path

import torch

from forensix.audio.detector import AudioDetector
from forensix.audio.rawgat_st.network import RawGAT_ST
from forensix.hub import WeightFile
from forensix.registry import ModelSpec, register

_COMMIT = "e3b049f79b3cd9552e66f6d0ba844cbb9399c479"

# model block from model_config_RawGAT_ST.yaml
RAWGAT_ST_CONFIG = {
    "nb_samp": 64600,
    "out_channels": 70,
    "first_conv": 128,
    "in_channels": 1,
    "filts": [32, [32, 32], [32, 64], [64, 64]],
    "blocks": [2, 4],
    "nb_classes": 2,
}


class RawGATSTDetector(AudioDetector):
    def _fake_probs(self, windows: torch.Tensor) -> torch.Tensor:
        logits = self.network(windows)
        # upstream label convention: class 1 = bonafide, class 0 = spoof
        return logits.softmax(dim=-1)[:, 0]


def build(weights_path: Path, device: str = "cpu") -> RawGATSTDetector:
    # the sinc layer moves its filters to this device on every forward pass
    network = RawGAT_ST(RAWGAT_ST_CONFIG, torch.device(device))
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    network.load_state_dict(state)
    return RawGATSTDetector(
        "rawgat-st", network, window=RAWGAT_ST_CONFIG["nb_samp"], device=device
    )


register(
    ModelSpec(
        name="rawgat-st",
        family="audio",
        builder=build,
        weights=WeightFile(
            url=f"https://raw.githubusercontent.com/eurecom-asp/RawGAT-ST-antispoofing/"
            f"{_COMMIT}/Pre_trained_models/RawGAT_ST_mul/Best_epoch.pth",
            sha256="8a91ce8201d42119cbfb9ebca8850e5914f1a85959b3ca962819783738c2a9e2",
            filename="rawgat-st.pth",
        ),
        license="MIT",
        source=f"https://github.com/eurecom-asp/RawGAT-ST-antispoofing/tree/{_COMMIT}",
        paper="https://arxiv.org/abs/2107.12710",
        description="Spectral and temporal graph attention on sinc-conv features, "
        "fused by element-wise multiplication (441K params). Trained on ASVspoof "
        "2019 LA; published EER 1.06% on LA eval.",
    )
)
