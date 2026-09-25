"""CLIP-based synthetic image detector (Cozzolino et al., CVPR-W 2024,
"Raising the Bar of AI-generated Image Detection with CLIP").

A linear classifier on L2-normalized, pre-projection OpenCLIP ViT-L/14
features (CommonPool.XL, MIT), trained on a few thousand latent-diffusion and
real images. Upstream: https://github.com/grip-unina/ClipBased-SyntheticImageDetection
(Apache-2.0, see LICENSE in this directory); the linear head comes from there
(weights/clipdet_latent10k_plus).
"""

from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from forensix.hub import WeightFile, fetch
from forensix.image.clip import CLIP_MEAN, CLIP_STD
from forensix.image.detector import ImageDetector
from forensix.registry import ModelSpec, register

_COMMIT = "c76ef7f5e158c5aba9e55b8b94ab0079720d281e"

OPENCLIP_L14_COMMONPOOL = WeightFile(
    url="https://huggingface.co/laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K/resolve/"
    "d979a50bf428edbb14b32c4dce7da2d49f7f46ed/open_clip_pytorch_model.safetensors",
    sha256="947d32621fa7da5c2fced632f85b50314ad581396628876f24d5a2e4ddebec0f",
    filename="openclip-vit-l14-commonpool-xl.safetensors",
)

# upstream main.py, patch_size "Clip224": bicubic resize of the short side, center crop
TRANSFORM = transforms.Compose(
    [
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ]
)


class ClipDet(nn.Module):
    """Upstream OpenClipLinear(next_to_last=True, normalize=True)."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        backbone.visual.proj = None  # features before the final projection
        self.backbone = backbone.visual
        self.fc = nn.Linear(1024, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = nn.functional.normalize(self.backbone(x), dim=-1)
        return self.fc(features)  # log-likelihood ratio; > 0 means synthetic


class ClipDetDetector(ImageDetector):
    def _fake_probs(self, batch: torch.Tensor) -> torch.Tensor:
        return self.network(batch).sigmoid().flatten()


def build(weights_path: Path, device: str = "cpu") -> ClipDetDetector:
    import open_clip  # imported lazily

    backbone = open_clip.create_model("ViT-L-14", pretrained=str(fetch(OPENCLIP_L14_COMMONPOOL)))
    network = ClipDet(backbone)
    head = torch.load(weights_path, map_location="cpu", weights_only=True)["model"]
    network.fc.load_state_dict({k.removeprefix("fc."): v for k, v in head.items()})
    return ClipDetDetector("clipdet", network, TRANSFORM, device=device)


register(
    ModelSpec(
        name="clipdet",
        family="image",
        builder=build,
        weights=WeightFile(
            url=f"https://media.githubusercontent.com/media/grip-unina/"
            f"ClipBased-SyntheticImageDetection/{_COMMIT}/weights/clipdet_latent10k_plus/weights.pth",
            sha256="1c9c9f48a47d975bde67e9924983aa4e4ccda96a15e7e983b6305f0c56d9ee18",
            filename="clipdet-latent10k-plus.pth",
        ),
        extra_weights=(OPENCLIP_L14_COMMONPOOL,),
        license="Apache-2.0 (code, linear head). Backbone: OpenCLIP ViT-L/14 "
        "CommonPool.XL (MIT)",
        source=f"https://github.com/grip-unina/ClipBased-SyntheticImageDetection/tree/{_COMMIT}",
        paper="https://arxiv.org/abs/2312.00195",
        description="Linear classifier on L2-normalized OpenCLIP ViT-L/14 features, "
        "trained on a small set of latent-diffusion and real images. Resizes the short "
        "side to 224 (bicubic) and scores the central crop.",
    )
)
