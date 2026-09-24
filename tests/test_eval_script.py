import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SCRIPT = Path(__file__).parents[1] / "scripts" / "eval_asvspoof2019_la.py"


def test_eval_script_on_synthetic_protocol(tmp_path):
    rng = np.random.default_rng(0)
    audio = tmp_path / "flac"
    audio.mkdir()
    lines = []
    for i in range(6):
        attack, key = ("-", "bonafide") if i < 3 else (f"A{7 + i % 2:02d}", "spoof")
        utt = f"LA_E_{i:07d}"
        sf.write(audio / f"{utt}.flac", rng.standard_normal(16000).astype(np.float32) * 0.05, 16000)
        lines.append(f"LA_0000 {utt} - {attack} {key}")
    protocol = tmp_path / "protocol.txt"
    protocol.write_text("\n".join(lines) + "\n")

    out = tmp_path / "results.json"
    subprocess.run(
        [sys.executable, SCRIPT, "--protocol", protocol, "--audio-dir", audio,
         "--models", "aasist-l", "--n-bonafide", "2", "--n-per-attack", "1", "--out", out],
        check=True,
    )
    results = json.loads(out.read_text())
    assert results["subset"]["counts"] == {"A07": 1, "A08": 1, "bonafide": 2}
    r = results["models"]["aasist-l"]
    assert 0 <= r["eer_percent"] <= 100
    assert set(r["per_attack_eer_percent"]) == {"A07", "A08"}
