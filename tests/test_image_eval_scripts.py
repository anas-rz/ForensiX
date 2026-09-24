import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_spec = importlib.util.spec_from_file_location(
    "build_image_eval_set", Path(__file__).parents[1] / "scripts" / "build_image_eval_set.py"
)
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)


@pytest.mark.parametrize(
    "size, area, expected",
    [
        ((1024, 1024), 1024 * 1024, (1024, 1024)),
        ((1024, 682), 1024 * 1024, (1280, 832)),  # 3:2 landscape
        ((600, 900), 768 * 768, (640, 960)),  # 2:3 portrait
        ((4000, 500), 1024 * 1024, (1472, 704)),  # clamped to 2:1
    ],
)
def test_generation_size(size, area, expected):
    w, h = build.generation_size(*size, area)
    assert (w, h) == expected
    assert w % 64 == 0 and h % 64 == 0


@pytest.mark.parametrize("quality", [50, 75, 90, 95])
def test_estimate_jpeg_quality(quality):
    tables = build._jpeg_quality_tables()
    arr = np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    assert build.estimate_jpeg_quality(Image.open(buf), tables) == quality
    png = io.BytesIO()
    Image.fromarray(arr).save(png, format="PNG")
    assert build.estimate_jpeg_quality(Image.open(png), tables) is None


def test_generators_are_pinned_and_commercially_licensed():
    for name, g in build.GENERATORS.items():
        assert len(g["revision"]) == 40, name
        assert g["license"] in {"Apache-2.0", "CreativeML OpenRAIL++-M"}, name
