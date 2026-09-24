"""UnivFD (Ojha et al., CVPR 2023): a linear probe on frozen CLIP ViT-L/14 features.

Trained on ProGAN images (ForenSynths) and shown to generalize to unseen GANs
and diffusion models. Upstream: https://github.com/WisconsinAIVision/UniversalFakeDetect
(MIT, see LICENSE in this directory); the linear head's weights come from there.
"""

from pathlib import Path

import torch
from torch import nn
from torchvision import transforms

from forensix.hub import WeightFile
from forensix.image.clip import CLIP_MEAN, CLIP_STD, CLIP_VIT_L14, clip_vit_l14
from forensix.image.detector import ImageDetector
from forensix.registry import ModelSpec, register

_COMMIT = "030495aea3300a8b54c0ec37ec7fe1dd7e63c619"

# upstream validate.py: center crop without resizing, CLIP normalization
TRANSFORM = transforms.Compose(
    [
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ]
)


class UnivFD(nn.Module):
    """CLIP image embedding -> linear layer -> fake logit (upstream CLIPModel)."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(768, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.backbone(pixel_values=x).image_embeds)


class UnivFDDetector(ImageDetector):
    def _fake_probs(self, batch: torch.Tensor) -> torch.Tensor:
        # upstream label convention: 1 = fake
        return self.network(batch).sigmoid().flatten()


def build(weights_path: Path, device: str = "cpu") -> UnivFDDetector:
    network = UnivFD(clip_vit_l14())
    network.fc.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True)
    )
    return UnivFDDetector("univfd", network, TRANSFORM, device=device)


register(
    ModelSpec(
        name="univfd",
        family="image",
        builder=build,
        weights=WeightFile(
            url=f"https://raw.githubusercontent.com/WisconsinAIVision/UniversalFakeDetect/"
            f"{_COMMIT}/pretrained_weights/fc_weights.pth",
            sha256="477100745713bcc957beb2b40859536859b6483fd6301b3b9293151b194c7847",
            filename="univfd-fc.pth",
        ),
        extra_weights=(CLIP_VIT_L14,),
        license="MIT (code, linear head). Backbone: OpenAI CLIP ViT-L/14, MIT; OpenAI's "
        "model card lists deployed use as out of scope (not a license term)",
        source=f"https://github.com/WisconsinAIVision/UniversalFakeDetect/tree/{_COMMIT}",
        paper="https://arxiv.org/abs/2302.10174",
        description="Linear probe on frozen CLIP ViT-L/14 image embeddings, trained on "
        "ProGAN vs real (ForenSynths). Generalizes to unseen GANs and diffusion "
        "models. Scores the central 224x224 crop without resizing.",
    )
)
