import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

SCRIPT = Path(__file__).parents[1] / "scripts" / "eval_deepfake_eval_2024.py"


def _run(data, modality, model, tmp_path):
    out = tmp_path / f"{modality}.json"
    subprocess.run([sys.executable, SCRIPT, "--data", data, "--modality", modality,
                    "--models", model, "--out", out], check=True)
    return json.loads(out.read_text())


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg for m4a")
def test_audio_with_m4a_and_detected_columns(tmp_path):
    data = tmp_path / "de2024"
    (data / "audio-data").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(6):
        name = f"clip{i}.wav"
        sf.write(data / "audio-data" / name, (rng.standard_normal(16000) * 0.1).astype(np.float32), 16000)
        if i == 0:  # one m4a, decoded through the ffmpeg fallback
            subprocess.run(["ffmpeg", "-v", "error", "-i", data / "audio-data" / name,
                            data / "audio-data" / "clip0.m4a"], check=True)
            (data / "audio-data" / name).unlink()
            name = "clip0.m4a"
        rows.append({"Duration": "1.0", "Filename": name, "Ground Truth": "Fake" if i % 2 else "Real"})
    rows.append({"Duration": "1.0", "Filename": "missing.wav", "Ground Truth": "Real"})
    _write_csv(data / "audio-metadata-publish.csv", rows)
    r = _run(data, "audio", "aasist-l", tmp_path)
    assert (r["n_fake"], r["n_real"]) == (3, 3)
    assert r["models"]["aasist-l"]["failed"] == 0 and 0 <= r["models"]["aasist-l"]["auc"] <= 1


def test_image_subset_and_corrupt_file(tmp_path):
    data = tmp_path / "de2024"
    (data / "image-data").mkdir(parents=True)
    rows = []
    for i in range(8):
        arr = np.random.default_rng(i).integers(0, 255, (300, 300, 3), dtype=np.uint8)
        Image.fromarray(arr).save(data / "image-data" / f"img{i}.png")
        rows.append({"id": f"img{i}", "label": "fake" if i < 4 else "real"})
    (data / "image-data" / "img0.png").write_bytes(b"corrupt")
    _write_csv(data / "image-metadata-publish.csv", rows)
    r = _run(data, "image", "safe", tmp_path)
    assert r["models"]["safe"]["failed"] == 1
    assert r["models"]["safe"]["n_fake"] == 3 and r["models"]["safe"]["n_real"] == 4
