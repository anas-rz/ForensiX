"""Nes2Net-X (Liu et al., IEEE T-IFS 2025) on XLS-R, fine-tuned on ASVspoof 2019 LA."""

from pathlib import Path
from types import SimpleNamespace

import torch

from forensix.audio.detector import AudioDetector
from forensix.audio.xlsr import convert_fairseq_state_dict
from forensix.hub import WeightFile
from forensix.registry import ModelSpec, register

_DRIVE = "https://drive.usercontent.google.com/download?export=download&confirm=t&id="

# settings from the upstream README's evaluation commands
NES2NET_X_ARGS = SimpleNamespace(
    n_output_logits=2, dilation=2, pool_func="mean", Nes_ratio=[8, 8], SE_ratio=[1]
)


class Nes2NetXDetector(AudioDetector):
    def _fake_probs(self, windows: torch.Tensor) -> torch.Tensor:
        logits = self.network(windows)
        # upstream label convention: class 1 = bonafide, class 0 = spoof
        return logits.softmax(dim=-1)[:, 0]


def _builder(name: str, window: int, pad_short: bool):
    def build(weights_path: Path, device: str = "cpu") -> Nes2NetXDetector:
        # imported lazily: pulls in transformers
        from forensix.audio.nes2net_x.network import wav2vec2_Nes2Net_no_Res_w_allT

        network = wav2vec2_Nes2Net_no_Res_w_allT(NES2NET_X_ARGS, device)
        state = torch.load(weights_path, map_location="cpu", weights_only=True, mmap=True)
        network.load_state_dict(convert_fairseq_state_dict(state))
        return Nes2NetXDetector(
            name, network, window=window, device=device, pad_short=pad_short
        )

    return build


_COMMON = dict(
    family="audio",
    license="Apache-2.0 (code and released models, per the upstream README; "
    "weights hosted on Google Drive). XLS-R base model Apache-2.0",
    source="https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW/tree/c9fe1177f44040f002cb40def928332d2e261f1e",
    paper="https://arxiv.org/abs/2504.05657",
)

register(
    ModelSpec(
        name="nes2net-x",
        builder=_builder("nes2net-x", window=64600, pad_short=True),
        weights=WeightFile(
            url=_DRIVE + "1JFGv_2TONMnTLGbiOIuHFfMvuo4SIIpg",
            sha256="4d7246d057bf1f312db52801bb213ab53cfea00dfd29354161ea9b1ef6396282",
            filename="nes2net-x.pth",
        ),
        description="Nested Res2Net back end (511K params) on XLS-R 300M, fine-tuned on "
        "ASVspoof 2019 LA; average of 5 checkpoints. Published EER 1.49% on ASVspoof "
        "2021 DF and 1.88% on 2021 LA. Scores 4 s windows.",
        **_COMMON,
    )
)

register(
    ModelSpec(
        name="nes2net-x-itw",
        # whole utterances, as upstream scores In-the-Wild; longer clips are split
        # into 30 s chunks to bound attention memory
        builder=_builder("nes2net-x-itw", window=30 * 16000, pad_short=False),
        weights=WeightFile(
            url=_DRIVE + "14YD8A54HLkXzlp7PWZ5iTW56WmQTIeEy",
            sha256="891de3f03cd846a5d6b69a193889440e26a5dde4028c7c4ff3bc80273987a024",
            filename="nes2net-x-itw.pth",
        ),
        description="Nes2Net-X on XLS-R, trained on ASVspoof 2019 LA with validation-set "
        "augmentation for out-of-domain robustness. Published EER 5.15% on In-the-Wild. "
        "Scores whole utterances.",
        **_COMMON,
    )
)
