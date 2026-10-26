"""Throughput and score parity: the per-input path vs forensix.batch.BatchRunner.

Times the path used by the eval scripts (image detectors: predict_batch on 8
frames per video; audio: predict_batch; video: predict), then BatchRunner in
fp32, bf16, and for video detectors bf16 with fast_faces. Prints one JSON line
with inputs/s, speedup and the max score difference against the per-input path.

    python scripts/bench_throughput.py univfd 106 --sdfvd /content/sdfvd/SDFVD
    python scripts/bench_throughput.py aasist 400 --asvspoof /content/asvspoof/LA
"""
import argparse
import glob
import json
import time

import numpy as np
import torch

import forensix
from forensix.batch import BatchRunner
from forensix.metrics import roc_auc
from forensix.video.io import read_frames

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("n", type=int, help="number of inputs")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--batch-size", type=int, default=64)
ap.add_argument("--sdfvd", default="/content/sdfvd/SDFVD", help="SDFVD folder (image and video models)")
ap.add_argument("--asvspoof", default="/content/asvspoof/LA", help="ASVspoof 2019 LA folder (audio models)")
args = ap.parse_args()
name, n, workers, batch = args.model, args.n, args.workers, args.batch_size
family = forensix.get_spec(name).family
if family == "audio":
    rows = [l.split() for l in open(f"{args.asvspoof}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")]
    rows = rows[:: max(1, len(rows) // n)][:n]
    inputs = [f"{args.asvspoof}/ASVspoof2019_LA_eval/flac/{r[1]}.flac" for r in rows]
    labels = np.array([r[4] == "spoof" for r in rows])
else:
    real = sorted(glob.glob(f"{args.sdfvd}/videos_real/*.mp4"))
    fake = sorted(glob.glob(f"{args.sdfvd}/videos_fake/*.mp4"))
    inputs = (real[: n // 2] + fake[: n - n // 2])
    labels = np.array([("videos_fake" in p) for p in inputs])

det = forensix.load(name, device="cuda")


def baseline():
    with torch.inference_mode():
        if family == "image":  # scripts/eval_video_folder.py
            return [float(np.mean([p.fake_prob for p in det.predict_batch(list(read_frames(v, 8)[0]), batch_size=8)]))
                    for v in inputs]
        if family == "audio":
            return [p.fake_prob for p in det.predict_batch(inputs, batch_size=16)]
        return [det.predict(v).fake_prob for v in inputs]


def runner(precision, fast=False):
    r = BatchRunner(det, batch_size=batch, num_workers=workers, precision=precision, fast_faces=fast)
    out = {item: (pred.fake_prob if pred else np.nan) for item, pred, _ in r.run(inputs)}
    return [out[i] for i in inputs]


def timed(fn, *a):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn(*a)
    torch.cuda.synchronize()
    return np.array(out, float), time.perf_counter() - t


if __name__ == "__main__":
    runner("fp32")  # warm-up (cuDNN autotune, worker start, Triton compile)
    base, t_base = timed(baseline)
    fp32, t_fp32 = timed(runner, "fp32")
    bf16, t_bf16 = timed(runner, "bf16")
    fast, t_fast = timed(runner, "bf16", True) if family == "video" else (bf16, t_bf16)
    res = {"model": name, "n": len(inputs), "workers": workers, "batch": batch, "gpu": torch.cuda.get_device_name()}
    runs = [("baseline", base, t_base), ("runner_fp32", fp32, t_fp32), ("runner_bf16", bf16, t_bf16)]
    if family == "video":
        runs.append(("runner_bf16_fast_faces", fast, t_fast))
    for tag, s, t in runs:
        res[tag] = {"seconds": round(t, 2), "items_per_s": round(len(inputs) / t, 2),
                    "auc": round(roc_auc(s[labels], s[~labels]), 4)}
        if tag != "baseline":
            res[tag]["max_abs_diff_vs_baseline"] = float(np.nanmax(np.abs(s - base)))
            res[tag]["speedup"] = round(t_base / t, 2)
    print(json.dumps(res))
