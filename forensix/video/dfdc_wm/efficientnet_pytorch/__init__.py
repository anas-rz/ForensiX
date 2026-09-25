# Vendored from https://github.com/lukemelas/EfficientNet-PyTorch (commit d8481a539cc1f84ef0fe502f9c12dcc187669611,
# efficientnet_pytorch/__init__.py), Apache-2.0, see LICENSE in this directory. Unchanged: this is the
# version Team WM trained with; later releases pad convolutions differently.

__version__ = "0.6.3"
from .model import EfficientNet
from .utils import (
    GlobalParams,
    BlockArgs,
    BlockDecoder,
    efficientnet,
    get_model_params,
)

