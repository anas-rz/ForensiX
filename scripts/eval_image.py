"""Evaluate image detectors on the ForensiX image evaluation set.

Build the set first with scripts/build_image_eval_set.py, then:

    python scripts/eval_image.py --data /content/forensix_image_eval \\
        --models univfd --device cuda --out results/image_eval.json

For each fake condition (``png``: generator output as is; ``jpeg``: fakes
re-encoded at the reals' JPEG quality) it reports ROC-AUC, average precision
and accuracy at the model's default threshold, pooled and per generator. The
``jpeg`` numbers are the fair ones: with ``png``, a detector can score well by
spotting "not JPEG".
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import forensix
from forensix.metrics import binary_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="folder with manifest.csv")
    parser.add_argument("--models", nargs="+", default=forensix.list_models("image"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, help="write results JSON here")
    parser.add_argument("--scores-out", type=Path, help="write per-image scores CSV here")
    args = parser.parse_args()

    with open(args.data / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    paths = [args.data / r["path"] for r in rows]
    counts = defaultdict(int)
    for r in rows:
        counts[r["condition"] if r["label"] == "fake" else "real"] += 1
    print(f"{len(rows)} images: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    results = {"data": str(args.data), "counts": dict(counts), "models": {}}
    all_scores = {}
    for name in args.models:
        detector = forensix.load(name, device=args.device)
        start = time.perf_counter()
        scores = []
        for i in range(0, len(paths), args.batch_size):
            preds = detector.predict_batch(paths[i : i + args.batch_size], batch_size=args.batch_size)
            scores += [p.fake_prob for p in preds]
            print(f"\r  {name}: {len(scores)}/{len(paths)}", end="", flush=True)
        print()
        scores = np.array(scores)
        all_scores[name] = scores

        real = scores[[r["label"] == "real" for r in rows]]
        model_result = {"seconds": round(time.perf_counter() - start, 1)}
        for condition in ("png", "jpeg"):
            is_cond = np.array([r["condition"] == condition for r in rows])
            if not is_cond.any():
                continue
            cond = binary_summary(scores[is_cond], real, detector.threshold)
            gens = sorted({r["generator"] for r in rows if r["condition"] == condition})
            cond["per_generator"] = {
                g: binary_summary(
                    scores[is_cond & np.array([r["generator"] == g for r in rows])],
                    real,
                    detector.threshold,
                )
                for g in gens
            }
            model_result[condition] = cond
            per_gen = ", ".join(f"{g} {v['auc']:.3f}" for g, v in cond["per_generator"].items())
            print(f"  {name} [{condition}] AUC {cond['auc']:.3f}  AP {cond['ap']:.3f}  "
                  f"bal.acc {cond.get('balanced_acc', float('nan')):.3f}  | AUC per generator: {per_gen}")
        results["models"][name] = model_result

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"Wrote {args.out}")
    if args.scores_out:
        args.scores_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.scores_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label", "condition", "generator", *all_scores])
            for i, r in enumerate(rows):
                writer.writerow([r["path"], r["label"], r["condition"], r["generator"],
                                 *(f"{s[i]:.6f}" for s in all_scores.values())])


if __name__ == "__main__":
    main()
