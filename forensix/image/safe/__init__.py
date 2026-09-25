"""SAFE (Li et al., KDD 2025): a small ResNet on the diagonal wavelet
high-frequency band of the image.

Weights: https://github.com/Ouxiang-Li/SAFE (Apache-2.0, see LICENSE in this
directory), trained from scratch on ProGAN images (ForenSynths).

The network is implemented here from torchvision's ResNet building blocks
(BSD-3) rather than copied from upstream, whose models/resnet.py derives from
NPR's unlicensed code; parameter names match, so the checkpoint loads as is.
The single-level DWT replaces upstream's pytorch_wavelets call.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.models.resnet import Bottleneck, conv1x1
from torchvision.transforms.functional import resize

from forensix.hub import WeightFile
from forensix.image.detector import ImageDetector
from forensix.registry import ModelSpec, register

_COMMIT = "4e998724651b227def64f5be0cd60c0aa1552c35"

# upstream eval transform (--transform_mode crop --input_size 256): no resize, no normalization
TRANSFORM = transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor()])


def _symmetric_pad(x: torch.Tensor, left: int, right: int, dim: int) -> torch.Tensor:
    """Half-sample symmetric padding (numpy/pywt ``symmetric``) along ``dim``."""
    n = x.shape[dim]
    idx = torch.arange(-left, n + right, device=x.device) % (2 * n)
    idx = torch.where(idx >= n, 2 * n - 1 - idx, idx)
    return x.index_select(dim, idx)


def dwt_hh(x: torch.Tensor, wavelet: str = "bior1.3") -> torch.Tensor:
    """Diagonal (HH) detail band of a one-level 2-D DWT, per channel, with
    symmetric boundary handling; same output as pytorch_wavelets'
    ``DWTForward(J=1, mode="symmetric")`` ``Yh[0][:, :, 2]``."""
    import pywt  # imported lazily

    hi = torch.tensor(pywt.Wavelet(wavelet).dec_hi[::-1], dtype=x.dtype, device=x.device)
    length = hi.numel()
    channels = x.shape[1]
    for dim in (3, 2):  # filter rows, then columns
        n = x.shape[dim]
        out = (n + length - 1) // 2  # pywt.dwt_coeff_len(n, length, "symmetric")
        p = 2 * (out - 1) - n + length
        x = _symmetric_pad(x, p // 2, (p + 1) // 2, dim)
        shape = (channels, 1, 1, length) if dim == 3 else (channels, 1, length, 1)
        stride = (1, 2) if dim == 3 else (2, 1)
        x = F.conv2d(x, hi.view(1, 1, *shape[2:]).expand(shape), stride=stride, groups=channels)
    return x


class SAFENet(nn.Module):
    """First two stages of a ResNet-50 (3x3 stride-2 stem) on the DWT HH band."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inplanes = 64
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128 * Bottleneck.expansion, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * Bottleneck.expansion, stride),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion
        layers += [Bottleneck(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # HH band upsampled back to the input size (bilinear), as upstream
        x = resize(dwt_hh(x), list(x.shape[-2:]))
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer2(self.layer1(x))
        return self.fc1(torch.flatten(self.avgpool(x), 1))


class SAFEDetector(ImageDetector):
    def _fake_probs(self, batch: torch.Tensor) -> torch.Tensor:
        # upstream: softmax over (real, fake), class 1 = fake
        return self.network(batch).softmax(dim=-1)[:, 1]


def build(weights_path: Path, device: str = "cpu") -> SAFEDetector:
    network = SAFENet()
    state = torch.load(weights_path, map_location="cpu", weights_only=True)["model"]
    network.load_state_dict(state)
    return SAFEDetector("safe", network, TRANSFORM, device=device)


register(
    ModelSpec(
        name="safe",
        family="image",
        builder=build,
        weights=WeightFile(
            url=f"https://raw.githubusercontent.com/Ouxiang-Li/SAFE/{_COMMIT}/checkpoint/checkpoint-best.pth",
            sha256="b3f5ecfb46a154ed553aaaf4bf3ba59182310726ddb0cbb1fe42bd0e22d2f20e",
            filename="safe.pth",
        ),
        license="Apache-2.0 (weights; trained from scratch). Network reimplemented from "
        "torchvision ResNet blocks (BSD-3)",
        source=f"https://github.com/Ouxiang-Li/SAFE/tree/{_COMMIT}",
        paper="https://arxiv.org/abs/2408.06741",
        description="1.4M-param ResNet on the diagonal wavelet (bior1.3) high-frequency "
        "band, trained on ProGAN (ForenSynths) with crop-based augmentation. Scores the "
        "central 256x256 crop without resizing.",
    )
)
