"""RINE (Koutlis & Papadopoulos, ECCV 2024): synthetic image detection from the
CLS tokens of every CLIP ViT-L/14 encoder block.

Adapted from https://github.com/mever-team/rine (Apache-2.0, see LICENSE in this
directory), src/models.py and src/utils.py at commit 9b7fd58. Changes: the
frozen backbone is Hugging Face transformers' CLIP (forensix.image.clip) instead
of the OpenAI ``clip`` package, with hooks on each encoder layer's
``layer_norm2`` (OpenAI's ``ln_2``). The trainable modules keep upstream's
structure and parameter names, so the released checkpoints load unchanged.
"""

from pathlib import Path

import torch
from torch import nn
from torchvision import transforms

from forensix.hub import WeightFile
from forensix.image.clip import CLIP_MEAN, CLIP_STD, CLIP_VIT_L14, clip_vit_l14
from forensix.image.detector import ImageDetector
from forensix.registry import ModelSpec, register

_COMMIT = "9b7fd5857cc205d0412be6aeee0d7611b95bd620"
_CKPT_URL = f"https://raw.githubusercontent.com/mever-team/rine/{_COMMIT}/ckpt"

_NORMALIZE = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
# upstream get_transforms(): transforms_test_1 (ProGAN protocol)
CENTER_CROP = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), _NORMALIZE])


def _pad_to(size: int):
    # TenCrop needs both sides >= size (upstream errors otherwise); pad like CenterCrop
    def pad(img):
        w, h = img.size
        if w >= size and h >= size:
            return img
        return transforms.CenterCrop((max(h, size), max(w, size)))(img)

    return pad


# upstream transforms_test_2 (Synthbuster protocol): 10 crops per image
TEN_CROP = transforms.Compose(
    [
        transforms.Lambda(_pad_to(224)),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([_NORMALIZE(transforms.ToTensor()(c)) for c in crops])),
    ]
)


class RINE(nn.Module):
    def __init__(self, backbone: nn.Module, nproj: int, proj_dim: int):
        super().__init__()
        self.backbone = backbone.vision_model
        self.backbone.requires_grad_(False)
        self._cls_tokens = []
        for layer in self.backbone.encoder.layers:
            layer.layer_norm2.register_forward_hook(
                lambda module, inputs, output: self._cls_tokens.append(output[:, 0])
            )

        # trainable part, same structure and names as upstream Model
        n_blocks = len(self.backbone.encoder.layers)
        self.alpha = nn.Parameter(torch.randn([1, n_blocks, proj_dim]))
        proj1_layers = [nn.Dropout()]
        for i in range(nproj):
            proj1_layers += [nn.Linear(1024 if i == 0 else proj_dim, proj_dim), nn.ReLU(), nn.Dropout()]
        self.proj1 = nn.Sequential(*proj1_layers)
        proj2_layers = [nn.Dropout()]
        for _ in range(nproj):
            proj2_layers += [nn.Linear(proj_dim, proj_dim), nn.ReLU(), nn.Dropout()]
        self.proj2 = nn.Sequential(*proj2_layers)
        self.head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim), nn.ReLU(), nn.Dropout(),
            nn.Linear(proj_dim, proj_dim), nn.ReLU(), nn.Dropout(),
            nn.Linear(proj_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fake logit per image, shape (batch, 1)."""
        self._cls_tokens.clear()
        with torch.no_grad():
            self.backbone(pixel_values=x)
        g = torch.stack(self._cls_tokens, dim=1)  # (batch, blocks, 1024)
        self._cls_tokens.clear()
        g = self.proj1(g.float())
        z = torch.sum(torch.softmax(self.alpha, dim=1) * g, dim=1)
        return self.head(self.proj2(z))


class RINEDetector(ImageDetector):
    def _fake_probs(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim == 5:  # (batch, crops, C, H, W): average crop logits, as upstream
            b, k = batch.shape[:2]
            return self.network(batch.flatten(0, 1)).view(b, k).mean(1).sigmoid()
        return self.network(batch).sigmoid().flatten()


def _builder(name: str, nproj: int, proj_dim: int, transform):
    def build(weights_path: Path, device: str = "cpu") -> RINEDetector:
        network = RINE(clip_vit_l14(), nproj=nproj, proj_dim=proj_dim)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        missing, unexpected = network.load_state_dict(state, strict=False)
        assert not unexpected and all(k.startswith("backbone.") for k in missing), (missing, unexpected)
        return RINEDetector(name, network, transform, device=device)

    return build


_COMMON = dict(
    family="image",
    extra_weights=(CLIP_VIT_L14,),
    license="Apache-2.0 (code, trainable weights). Backbone: OpenAI CLIP ViT-L/14, MIT; "
    "OpenAI's model card lists deployed use as out of scope (not a license term)",
    source=f"https://github.com/mever-team/rine/tree/{_COMMIT}",
    paper="https://arxiv.org/abs/2402.19091",
)

register(
    ModelSpec(
        name="rine",
        builder=_builder("rine", nproj=2, proj_dim=1024, transform=CENTER_CROP),
        weights=WeightFile(
            url=f"{_CKPT_URL}/model_4class_trainable.pth",
            sha256="6535ff6ecfa88d33c081e561c2cb2dcb594a1006b050eddb72a72ce005ed2ca1",
            filename="rine-4class.pth",
        ),
        description="Importance-weighted CLS tokens from all 24 CLIP ViT-L/14 blocks, "
        "trained on 4 ProGAN classes (6.3M trainable params). Scores the central "
        "224x224 crop without resizing.",
        **_COMMON,
    )
)

register(
    ModelSpec(
        name="rine-ldm",
        builder=_builder("rine-ldm", nproj=4, proj_dim=1024, transform=TEN_CROP),
        weights=WeightFile(
            url=f"{_CKPT_URL}/model_ldm_trainable.pth",
            sha256="13986ffc86e1bd9d57fd640f891358c4f14c0f8aea8e590e52c598541f69fdc1",
            filename="rine-ldm.pth",
        ),
        description="RINE trained on Latent Diffusion images (10.5M trainable params); "
        "better suited to diffusion-generated images. Averages logits over ten "
        "224x224 crops (corners, center and their flips), as upstream's Synthbuster evaluation.",
        **_COMMON,
    )
)
