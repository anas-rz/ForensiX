from typing import Callable, Sequence

import torch
from PIL import Image
from torch import nn

from forensix.image.io import ImageInput, load_image
from forensix.types import Prediction


class ImageDetector:
    """Base class for single-image deepfake / AI-generated image detectors.

    ``transform`` maps an RGB PIL image to a ``(C, H, W)`` tensor with the
    model's upstream evaluation preprocessing. Subclasses implement
    ``_fake_probs`` for a batch of transformed images.
    """

    def __init__(
        self,
        name: str,
        network: nn.Module,
        transform: Callable[[Image.Image], torch.Tensor],
        device: str | torch.device = "cpu",
        threshold: float = 0.5,
    ):
        self.name = name
        self.transform = transform
        self.threshold = threshold
        self.device = torch.device(device)
        self.network = network.to(self.device).eval()

    def _fake_probs(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, image: ImageInput) -> Prediction:
        """Score one image: a file path, a PIL image or an RGB uint8 array."""
        return self.predict_batch([image])[0]

    @torch.inference_mode()
    def predict_batch(
        self, images: Sequence[ImageInput], batch_size: int = 16
    ) -> list[Prediction]:
        probs = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack(
                [self.transform(load_image(im)) for im in images[i : i + batch_size]]
            )
            probs += self._fake_probs(batch.to(self.device)).cpu().tolist()
        return [
            Prediction(model=self.name, fake_prob=float(p), threshold=self.threshold)
            for p in probs
        ]
