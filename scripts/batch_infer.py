"""Score many files with one detector at GPU throughput (see forensix.batch).

    python scripts/batch_infer.py --model gend --inputs /data/videos --device cuda \\
        --precision bf16 --batch-size 64 --workers 8 --out scores/gend.jsonl

- ``--inputs``: files, folders (searched recursively for media of the model's
  kind) or ``.txt`` lists of paths.
- ``--out``: one JSON line per input, written as inputs finish; inputs already
  in the file are skipped, so a run can be resumed.
- ``--gpus 0,1,2,3``: one process per GPU, each on every n-th input; their
  outputs are merged into ``--out``.
- ``--precision bf16|fp16``: autocast; ``--compare-fp32 N`` first scores N
  inputs in both fp32 and the chosen precision and reports the score drift.
- ``--calibration table.json``: also write calibrated probabilities, the
  conformal decision and prediction set (see forensix.calibration).
- Detectors without a batching adapter run through ``predict``; CPU ones
  (physics) in ``--processes`` worker processes.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import forensix
from forensix.batch import VIDEO_EXTS, BatchRunner, adapter_for, parallel_predict

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac"}
KIND_EXTS = {"image": IMAGE_EXTS | VIDEO_EXTS, "audio": AUDIO_EXTS, "video": VIDEO_EXTS,
             "audiovisual": VIDEO_EXTS, "physics": VIDEO_EXTS | AUDIO_EXTS}


def collect(inputs: list[str], family: str) -> list[str]:
    exts = KIND_EXTS.get(family, VIDEO_EXTS | AUDIO_EXTS | IMAGE_EXTS)
    out = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            out += sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in exts)
        elif p.suffix == ".txt":
            out += [line.strip() for line in p.read_text().splitlines() if line.strip()]
        else:
            out.append(str(p))
    return list(dict.fromkeys(out))  # dedupe, keep order


def done_inputs(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["input"] for line in f if line.strip()}


def row(item, pred, error, calibrated) -> dict:
    if error is not None:
        return {"input": str(item), "error": error}
    r = {"input": str(item), "model": pred.model, "fake_prob": round(pred.fake_prob, 6), "threshold": pred.threshold,
         "label": pred.label}
    if pred.segment_fake_probs:
        r["segments"] = [round(p, 6) for p in pred.segment_fake_probs]
    if calibrated is not None:
        c = calibrated.calibrate(pred)
        r["calibrated"] = {"fake_prob": round(c.fake_prob, 6), "label": c.label,
                           "prediction_set": c.details["prediction_set"], "p_real": c.details["p_real"]}
    return r


def score(args, items: list[str]) -> None:
    calibrated = None
    if args.calibration:
        from forensix.calibration import Calibration, CalibratedDetector

        table = Calibration.load(args.calibration)
        calibrated = CalibratedDetector(SimpleNamespace(name=args.model), table)  # only calibrate() is used
    family = forensix.get_spec(args.model).family
    with open(args.out, "a", encoding="utf-8") as out:
        if family == "physics" or (args.processes and args.device == "cpu"):
            results = parallel_predict(args.model, items, args.processes or os.cpu_count(), args.device)
        else:
            detector = forensix.load(args.model, device=args.device)
            if adapter_for(detector) is None:
                print(f"{args.model}: no batching adapter, scoring one input at a time")
            results = BatchRunner(detector, batch_size=args.batch_size, num_workers=args.workers,
                                  precision=args.precision, frames=args.frames, fast_faces=args.fast_faces).run(items)
        start, n = time.perf_counter(), 0  # after model loading
        for item, pred, error in results:
            out.write(json.dumps(row(item, pred, error, calibrated)) + "\n")
            out.flush()
            n += 1
            if n % 100 == 0 or n == len(items):
                rate = n / (time.perf_counter() - start)
                print(f"  {n}/{len(items)}  {rate:.2f} inputs/s", flush=True)
    elapsed = time.perf_counter() - start
    print(f"{args.model}: {n} inputs in {elapsed:.1f} s ({n / max(elapsed, 1e-9):.2f} inputs/s) -> {args.out}")


def compare_fp32(args, items: list[str]) -> None:
    detector = forensix.load(args.model, device=args.device)
    runs = {}
    for precision in ("fp32", args.precision):
        r = BatchRunner(detector, batch_size=args.batch_size, num_workers=args.workers, precision=precision,
                        frames=args.frames, fast_faces=args.fast_faces)
        runs[precision] = {item: pred.fake_prob for item, pred, _ in r.run(items) if pred is not None}
    common = [i for i in items if i in runs["fp32"] and i in runs[args.precision]]
    diff = np.abs(np.array([runs["fp32"][i] - runs[args.precision][i] for i in common]))
    thr = getattr(detector, "threshold", 0.5)
    flips = sum((runs["fp32"][i] >= thr) != (runs[args.precision][i] >= thr) for i in common) if thr else 0
    print(f"{args.precision} vs fp32 on {len(common)} inputs: max |diff| {diff.max():.2e}, "
          f"mean {diff.mean():.2e}, decisions changed at threshold {thr}: {flips}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpus", help="comma-separated GPU ids: one process per GPU")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8, help="CPU decode/preprocess workers per GPU")
    ap.add_argument("--frames", type=int, default=8, help="frames per video for image detectors")
    ap.add_argument("--fast-faces", action="store_true", help="GenD models: batch MTCNN (small score drift)")
    ap.add_argument("--processes", type=int, help="worker processes for detectors scored via predict on CPU")
    ap.add_argument("--calibration", type=Path)
    ap.add_argument("--compare-fp32", type=int, metavar="N")
    ap.add_argument("--shard", default="0/1", help=argparse.SUPPRESS)  # set by --gpus
    args = ap.parse_args()

    items = collect(args.inputs, forensix.get_spec(args.model).family)
    k, n = map(int, args.shard.split("/"))
    items = items[k::n]
    if args.compare_fp32:
        compare_fp32(args, items[: args.compare_fp32])
        return
    done = done_inputs(args.out)
    todo = [i for i in items if i not in done]
    print(f"{len(items)} inputs ({len(items) - len(todo)} already scored)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not todo:
        return
    if not args.gpus or n > 1:
        score(args, todo)
        return
    # one process per GPU, each on every n-th remaining input; parts merged into --out
    gpus = args.gpus.split(",")
    todo_file = args.out.with_suffix(".todo.txt")
    todo_file.write_text("\n".join(todo) + "\n")
    keep = []  # this run's options, minus the ones set per GPU process
    skip = {"--gpus", "--inputs", "--out"}
    argv = sys.argv[1:]
    j = 0
    while j < len(argv):
        if argv[j] in skip:
            j += 1
            while j < len(argv) and not argv[j].startswith("--"):
                j += 1
        else:
            keep.append(argv[j])
            j += 1
    procs = []
    for i, gpu in enumerate(gpus):
        part = args.out.with_suffix(f".part{i}{args.out.suffix}")
        cmd = [sys.executable, __file__, *keep, "--inputs", str(todo_file), "--shard", f"{i}/{len(gpus)}",
               "--out", str(part)]
        procs.append((part, subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})))
    codes = [p.wait() for _, p in procs]
    with open(args.out, "a", encoding="utf-8") as out:
        for part, _ in procs:
            if part.exists():
                out.write(part.read_text())
                part.unlink()
    todo_file.unlink()
    if any(codes):
        sys.exit(f"a GPU process failed (exit codes {codes}); rerun the same command to resume")


if __name__ == "__main__":
    main()
