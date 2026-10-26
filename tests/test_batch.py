import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from forensix.audio.detector import AudioDetector
from forensix.batch import BatchRunner, _fusable
from forensix.image.detector import ImageDetector
from forensix.kernels import center_crop_u8, to_normalized_chw

SCRIPT = Path(__file__).parents[1] / "scripts" / "batch_infer.py"
MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
CUDA = torch.cuda.is_available()


@pytest.mark.parametrize("shape", [(720, 1280), (721, 1281), (100, 300), (223, 500), (224, 224)])
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if CUDA else []))
def test_crop_and_kernel_match_torchvision(shape, device):
    img = np.random.default_rng(0).integers(0, 256, (*shape, 3), dtype=np.uint8)
    ref = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(),
                              transforms.Normalize(MEAN, STD)])(Image.fromarray(img))
    crop = torch.from_numpy(center_crop_u8(img, 224))[None].to(device)
    assert torch.equal(to_normalized_chw(crop, MEAN, STD)[0].cpu(), ref)  # bit-identical


class _TinyImage(ImageDetector):
    def __init__(self, transform):
        torch.manual_seed(0)
        net = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(48, 1))
        super().__init__("tiny", net, transform, device="cuda" if CUDA else "cpu")

    def _fake_probs(self, batch):
        return self.network(batch).sigmoid().squeeze(1)


@pytest.mark.parametrize("transform, fused", [
    (transforms.Compose([transforms.CenterCrop(64), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]), True),
    (transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64), transforms.ToTensor()]), False),
])
def test_runner_matches_predict_batch_for_images(tmp_path, transform, fused):
    rng = np.random.default_rng(1)
    paths = []
    for i, size in enumerate([(80, 120), (64, 64), (50, 90), (200, 100), (70, 70)]):
        p = tmp_path / f"{i}.png"
        Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8)).save(p)
        paths.append(str(p))
    paths.append(str(tmp_path / "missing.png"))
    det = _TinyImage(transform)
    assert (_fusable(det.transform) is not None) == fused
    expected = {p: pred.fake_prob for p, pred in zip(paths[:-1], det.predict_batch(paths[:-1]))}
    got, errors = {}, {}
    for item, pred, error in BatchRunner(det, batch_size=2, num_workers=2).run(paths):
        (errors if error else got)[item] = error or pred.fake_prob
    assert set(errors) == {paths[-1]}  # an unreadable input is reported, the rest still scored
    assert got.keys() == expected.keys()
    assert max(abs(got[p] - expected[p]) for p in got) < 1e-6


class _TinyAudio(AudioDetector):
    def __init__(self):
        torch.manual_seed(0)
        super().__init__("tiny-audio", nn.Linear(1600, 1), window=1600, device="cuda" if CUDA else "cpu")

    def _fake_probs(self, windows):
        return self.network(windows.float()).sigmoid().squeeze(1)


def test_runner_matches_predict_batch_for_audio(tmp_path):
    rng = np.random.default_rng(2)
    paths = []
    for i, seconds in enumerate([0.05, 0.3, 0.1, 0.5]):  # shorter and longer than the window
        p = tmp_path / f"{i}.wav"
        sf.write(p, (rng.standard_normal(int(seconds * 16000)) * 0.1).astype(np.float32), 16000)
        paths.append(str(p))
    det = _TinyAudio()
    expected = {p: pred for p, pred in zip(paths, det.predict_batch(paths))}
    for item, pred, error in BatchRunner(det, batch_size=3, num_workers=2).run(paths):
        assert error is None
        assert pred.fake_prob == pytest.approx(expected[item].fake_prob, abs=1e-6)
        assert len(pred.segment_fake_probs) == len(expected[item].segment_fake_probs)


def test_batch_infer_script_resumes(tmp_path):
    rng = np.random.default_rng(3)
    audio = tmp_path / "audio"
    audio.mkdir()
    for i in range(4):
        sf.write(audio / f"{i}.flac", (rng.standard_normal(16000) * 0.05).astype(np.float32), 16000)
    out = tmp_path / "scores.jsonl"
    cmd = [sys.executable, SCRIPT, "--model", "aasist-l", "--inputs", audio, "--device", "cpu",
           "--workers", "0", "--out", out]
    subprocess.run(cmd, check=True)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 4 and all(0 <= r["fake_prob"] <= 1 for r in rows)
    resumed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert "4 already scored" in resumed.stdout
    assert len(out.read_text().splitlines()) == 4
