import json

import torch

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

import forensix
from forensix.finetuning import checkpoint_info, head_module, trainable_parameters


def image_dataset(root, n=24):
    """Textured images; the 'real' ones carry a checkerboard (upsampling) pattern. SAFE flags that
    pattern as fake, so the base model gets these labels backwards and fine-tuning has to learn."""
    from PIL import ImageFilter

    rng = np.random.default_rng(0)
    rows = ["file,label,group"]
    for i in range(n):
        noise = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
        x = np.asarray(noise.filter(ImageFilter.GaussianBlur(2))).astype(float)
        for label in ("real", "fake"):
            y = x.copy()
            if label == "real":
                y[::2, ::2] += 12
            Image.fromarray(np.clip(y, 0, 255).astype(np.uint8)).save(root / f"{label}{i}.png")
            rows.append(f"{label}{i}.png,{label},pair{i}")
    (root / "manifest.csv").write_text("\n".join(rows) + "\n")
    return root


@pytest.fixture(scope="module")
def tuned_image(tmp_path_factory):
    root = image_dataset(tmp_path_factory.mktemp("img"))
    out = root.parent / "safe-ft.safetensors"
    det, report = forensix.finetune("safe", dataset=root, out=out, mode="norm", epochs=4, lr=1e-3,
                                    batch_size=8, val_fraction=0.25, log=lambda *_: None)
    return root, out, det, report


def test_finetune_improves_and_reports(tuned_image):
    _, _, _, report = tuned_image
    assert report["base_val_auc"] < 0.2 and report["val_auc"] > 0.9
    assert report["n_val"] == {"real": 6, "fake": 6} and len(report["history"]) == 4
    assert report["mode"] == "norm" and 0 < report["trained_parameters"] < 100_000


def test_checkpoint_holds_only_changed_tensors_and_round_trips(tuned_image):
    root, out, det, _ = tuned_image
    info = checkpoint_info(out)
    assert info["model"] == "safe" and info["mode"] == "norm" and info["base_sha256"]
    from safetensors import safe_open

    with safe_open(str(out), framework="pt") as f:
        names = set(f.keys())
    assert names == set(trainable_parameters(det.network, "norm"))
    files = [str(root / "real0.png"), str(root / "fake0.png")]
    reloaded = forensix.load("safe", weights_path=out)
    tuned = [p.fake_prob for p in det.predict_batch(files)]
    assert [p.fake_prob for p in reloaded.predict_batch(files)] == pytest.approx(tuned, abs=1e-5)
    assert [p.fake_prob for p in forensix.load("safe").predict_batch(files)] != pytest.approx(tuned, abs=1e-3)


def test_audio_finetune(tmp_path):
    sr, rows = 16000, ["file,label"]
    t = np.arange(sr * 2) / sr
    rng = np.random.default_rng(1)
    for i in range(16):
        real = 0.3 * np.sin(2 * np.pi * rng.uniform(150, 300) * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
        fake = real + 0.05 * rng.standard_normal(t.size)       # a vocoder-like noise floor
        for label, x in (("real", real), ("fake", fake)):
            sf.write(tmp_path / f"{label}{i}.wav", x.astype(np.float32), sr)
            rows.append(f"{label}{i}.wav,{label}")
    (tmp_path / "manifest.csv").write_text("\n".join(rows) + "\n")
    out = tmp_path / "aasist-ft.safetensors"
    _, report = forensix.finetune("aasist", dataset=tmp_path, out=out, mode="norm", epochs=2,
                                  batch_size=8, log=lambda *_: None)
    assert checkpoint_info(out)["model"] == "aasist" and report["val_auc"] is not None
    assert forensix.load("aasist", weights_path=out).predict(str(tmp_path / "real0.wav")).fake_prob is not None


def test_head_selection_and_errors(tmp_path):
    det = forensix.load("aasist")
    assert head_module(det.network).out_features == 2
    assert set(trainable_parameters(det.network, "head")) < set(trainable_parameters(det.network, "norm"))
    with pytest.raises(ValueError, match="mode"):
        forensix.finetune(det, real_media=[], fake_media=[], mode="everything")
    with pytest.raises(ValueError, match="both real and fake"):
        forensix.finetune(det, real_media=["a.wav"], fake_media=["b.wav"], val_fraction=0.5)
    plain = tmp_path / "x.safetensors"
    from safetensors.torch import save_file
    import torch

    save_file({"w": torch.zeros(1)}, str(plain), metadata={"other": json.dumps({})})
    assert checkpoint_info(plain) is None


peft = pytest.importorskip("peft")


def test_lora_learns_merges_and_loads_without_peft(tmp_path):
    import subprocess
    import sys

    from peft.tuners.lora import LoraLayer

    (tmp_path / "img").mkdir()
    root = image_dataset(tmp_path / "img")
    out = tmp_path / "safe-lora.safetensors"
    det, report = forensix.finetune("safe", dataset=root, out=out, mode="lora", epochs=4, lr=1e-3,
                                    lora_rank=4, batch_size=8, val_fraction=0.25, log=lambda *_: None)
    assert report["base_val_auc"] < 0.2 and report["val_auc"] > 0.9
    assert not any(isinstance(m, LoraLayer) for m in det.network.modules())   # merged back into plain layers
    info = checkpoint_info(out)
    assert info["mode"] == "lora" and info["lora"]["rank"] == 4 and info["lora"]["scaling"] == 4.0
    from safetensors import safe_open

    with safe_open(str(out), framework="pt") as f:
        keys = list(f.keys())
    head = set(trainable_parameters(det.network, "head"))
    assert head < set(keys) and all(k in head or k.endswith((".lora_A", ".lora_B", ".lora_delta")) for k in keys)
    files = [str(root / "real0.png"), str(root / "fake0.png")]
    tuned = [p.fake_prob for p in det.predict_batch(files)]
    # serving must not need PEFT: load in a fresh process where importing it fails
    code = ("import sys, json; sys.modules['peft'] = None; import forensix; "
            f"d = forensix.load('safe', weights_path={str(out)!r}); "
            f"print(json.dumps([p.fake_prob for p in d.predict_batch({files!r})]))")
    reloaded = json.loads(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                         check=True).stdout.strip().splitlines()[-1])
    assert reloaded == pytest.approx(tuned, abs=1e-4)


def test_lora_on_linear_layers(tmp_path):
    from forensix.finetuning import lora_targets

    det = forensix.load("aasist")
    targets = lora_targets(det.network)
    assert targets and all(isinstance(det.network.get_submodule(t), torch.nn.Linear) for t in targets)
    assert head_module(det.network) not in [det.network.get_submodule(t) for t in targets]
    sr, rows = 16000, ["file,label"]
    rng = np.random.default_rng(2)
    for i in range(8):
        x = 0.3 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)
        for label, y in (("real", x), ("fake", x + 0.05 * rng.standard_normal(sr))):
            sf.write(tmp_path / f"{label}{i}.wav", y.astype(np.float32), sr)
            rows.append(f"{label}{i}.wav,{label}")
    (tmp_path / "manifest.csv").write_text("\n".join(rows) + "\n")
    out = tmp_path / "aasist-lora.safetensors"
    tuned, _ = forensix.finetune("aasist", dataset=tmp_path, out=out, mode="lora", epochs=1, batch_size=4,
                                 log=lambda *_: None)
    clip = str(tmp_path / "fake0.wav")
    assert forensix.load("aasist", weights_path=out).predict(clip).fake_prob == pytest.approx(
        tuned.predict(clip).fake_prob, abs=1e-4)


def test_no_improving_epoch_keeps_the_base_weights(tmp_path):
    root = image_dataset(tmp_path)
    out = tmp_path / "safe-none.safetensors"
    _, report = forensix.finetune("safe", dataset=root, out=out, mode="lora", epochs=0, log=lambda *_: None)
    assert report["best_epoch"] == 0 and any("keeps the base" in w for w in report["warnings"])
    files = [str(root / "real0.png"), str(root / "fake1.png")]
    base = [p.fake_prob for p in forensix.load("safe").predict_batch(files)]
    assert [p.fake_prob for p in forensix.load("safe", weights_path=out).predict_batch(files)] == pytest.approx(base, abs=1e-5)
