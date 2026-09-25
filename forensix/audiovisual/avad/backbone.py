"""Visual encoder of AVAD's audio-visual synchronization network: an 18-layer
ResNet on 5-frame clips whose first two stages use 1x3x3 (per-frame 2-D) and
last two 3x3x3 (spatio-temporal) convolutions; layer4 keeps stride 1.

Written for ForensiX from torch building blocks; parameter names match the
released checkpoint (``vis_enc.*``).
"""

from torch import nn


def _conv(in_planes: int, out_planes: int, temporal: bool, stride: int = 1) -> nn.Conv3d:
    kernel = (3, 3, 3) if temporal else (1, 3, 3)
    padding = (1, 1, 1) if temporal else (0, 1, 1)
    return nn.Conv3d(in_planes, out_planes, kernel, stride=(1, stride, stride), padding=padding, bias=False)


class _Block(nn.Module):
    def __init__(self, inplanes: int, planes: int, temporal: bool, stride: int = 1):
        super().__init__()
        self.conv1 = _conv(inplanes, planes, temporal, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = _conv(planes, planes, temporal)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU()
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes, kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        identity = x if self.downsample is None else self.downsample(x)
        return self.relu(out + identity)


class VisualEncoder(nn.Module):
    """(B, 3, T, 224, 224) -> (B, 512, T, 14, 14)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._stage(64, 64, temporal=False, stride=1)
        self.layer2 = self._stage(64, 128, temporal=False, stride=2)
        self.layer3 = self._stage(128, 256, temporal=True, stride=2)
        self.layer4 = self._stage(256, 512, temporal=True, stride=1)

    @staticmethod
    def _stage(inplanes: int, planes: int, temporal: bool, stride: int) -> nn.Sequential:
        return nn.Sequential(_Block(inplanes, planes, temporal, stride), _Block(planes, planes, temporal))

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))
