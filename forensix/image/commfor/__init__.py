"""Community Forensics (Park & Owens, CVPR 2025): a ViT-S/16 trained on images
from ~4,800 generators.

Upstream: https://github.com/JeongsooP/Community-Forensics (MIT, see LICENSE in
this directory); weights from the authors' Hugging Face repos (MIT). The
backbone was initialized from timm's AugReg ImageNet-21k ViT-S/16 (Apache-2.0).
"""

from pathlib import Path

import torch
from safetensors.torch import load_file
from torchvision import transforms

from forensix.hub import WeightFile
from forensix.image.detector import ImageDetector
from forensix.registry import ModelSpec, register

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def test_transform(input_size: int):
    """Upstream dataloader.get_transform(mode="test")."""
    resize = {224: 256, 384: 440}[input_size]
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class CommForDetector(ImageDetector):
    def _fake_probs(self, batch: torch.Tensor) -> torch.Tensor:
        # upstream label convention: 1 = fake
        return self.network(batch).sigmoid().flatten()


def _builder(name: str, input_size: int):
    def build(weights_path: Path, device: str = "cpu") -> CommForDetector:
        import timm  # imported lazily

        # upstream ViTClassifier: timm ViT-S/16 with a 1-logit head, stored as ``vit.*``
        network = timm.create_model(
            f"vit_small_patch16_{input_size}", pretrained=False, num_classes=1
        )
        state = load_file(weights_path)
        network.load_state_dict({k.removeprefix("vit."): v for k, v in state.items()})
        return CommForDetector(name, network, test_transform(input_size), device=device)

    return build


def _weights(input_size: int, revision: str, sha256: str) -> WeightFile:
    return WeightFile(
        url=f"https://huggingface.co/OwensLab/commfor-model-{input_size}/resolve/"
        f"{revision}/model.safetensors",
        sha256=sha256,
        filename=f"commfor-{input_size}.safetensors",
    )


_COMMON = dict(
    family="image",
    license="MIT (code and weights). Backbone init: timm AugReg ViT-S/16 (Apache-2.0)",
    source="https://github.com/JeongsooP/Community-Forensics/tree/ee5b71d43db0f3779e1edd64ee927b13f2dd6ad4",
    paper="https://arxiv.org/abs/2411.04125",
)

register(
    ModelSpec(
        name="commfor",
        builder=_builder("commfor", 384),
        weights=_weights(
            384,
            "6076002bf0d9dd37537f965ee2f06f826c333b61",
            "b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387",
        ),
        description="ViT-S/16 (22M params) trained on Community Forensics: 2.7M images "
        "from ~4,800 generators (mostly Stable Diffusion derivatives) plus real photos. "
        "Resizes the short side to 440 and scores the central 384x384 crop.",
        **_COMMON,
    )
)

register(
    ModelSpec(
        name="commfor-224",
        builder=_builder("commfor-224", 224),
        weights=_weights(
            224,
            "26afc31e6b40c312c3fd42c05a758be62446215b",
            "a6cc439d5a6d2dfadd60c77d27a2838ad55b34e601ecd30f46ad97266d6ac4e0",
        ),
        description="Community Forensics ViT-S/16 at 224 input (faster). Resizes the "
        "short side to 256 and scores the central 224x224 crop.",
        **_COMMON,
    )
)
