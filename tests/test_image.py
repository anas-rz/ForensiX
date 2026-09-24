import os

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

import forensix
from forensix.image.detector import ImageDetector
from forensix.image.io import load_image


class MeanBrightness(ImageDetector):
    """Toy detector: fake_prob = mean pixel value of the transformed image."""

    def _fake_probs(self, batch):
        return self.network(batch).mean(dim=(1, 2, 3))


@pytest.fixture
def toy():
    t = transforms.Compose([transforms.CenterCrop(8), transforms.ToTensor()])
    return MeanBrightness("toy", nn.Identity(), t)


def test_load_image_inputs(tmp_path):
    arr = np.full((10, 12, 3), 200, dtype=np.uint8)
    Image.fromarray(arr).convert("L").save(tmp_path / "g.png")
    assert load_image(tmp_path / "g.png").mode == "RGB"
    assert load_image(arr).size == (12, 10)
    assert load_image(Image.fromarray(arr).convert("RGBA")).mode == "RGB"
    with pytest.raises(ValueError, match="uint8"):
        load_image(arr.astype(np.float32))


def test_image_detector_batching(toy, tmp_path):
    imgs = [np.full((16, 16, 3), v, dtype=np.uint8) for v in (0, 51, 255)]
    Image.fromarray(imgs[1]).save(tmp_path / "x.png")
    preds = toy.predict_batch([imgs[0], tmp_path / "x.png", imgs[2]], batch_size=2)
    assert [round(p.fake_prob, 3) for p in preds] == [0.0, 0.2, 1.0]
    assert [p.label for p in preds] == ["real", "real", "fake"]
    assert toy.predict(imgs[2]).model == "toy"


def test_image_registry():
    assert "univfd" in forensix.list_models("image")
    spec = forensix.info("univfd")
    assert spec.family == "image" and spec.extra_weights


heavy = pytest.mark.skipif(
    not os.environ.get("FORENSIX_TEST_HEAVY"), reason="large model; set FORENSIX_TEST_HEAVY=1"
)


@heavy
def test_univfd():
    d = forensix.load("univfd")
    assert sum(p.numel() for p in d.network.parameters()) == 303_966_977
    rng = np.random.default_rng(0)
    small = rng.integers(0, 255, (100, 150, 3), dtype=np.uint8)  # padded by CenterCrop
    large = rng.integers(0, 255, (600, 800, 3), dtype=np.uint8)
    batch = d.predict_batch([small, large], batch_size=2)
    for img, p in zip([small, large], batch):
        assert 0.0 <= p.fake_prob <= 1.0
        assert p.fake_prob == pytest.approx(d.predict(img).fake_prob, abs=1e-5)
    with torch.inference_mode():
        assert d.network(d.transform(load_image(large))[None]).shape == (1, 1)
