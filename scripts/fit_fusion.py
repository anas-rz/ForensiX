"""Fit a fusion of several detectors on one split's scores and test it on another's.

Works on per-file score CSVs (``file,label,<column>,...``), such as
``results/lavdf/lavdf_{dev,test}_scores.csv`` from ``scripts/eval_lavdf.py report``,
so models are never re-run. Each member is ``model[:option=value,...]``; its
column is the model name followed by the option values
(``stablesyncnet:clip_score=max-window`` reads column ``stablesyncnet-max-window``).

    python scripts/fit_fusion.py --fit results/lavdf/lavdf_dev_scores.csv \\
        --test results/lavdf/lavdf_test_scores.csv \\
        --members stablesyncnet:clip_score=max-window syncnet:clip_score=max-window avad \\
        --out fusions/lavdf_av.json --report results/lavdf/fusion_lavdf.json

Reports test AUC with stratified bootstrap CIs for the fusion and every member
alone, and the fusion's paired gain over the best member. A member's missing
score (it failed on that file) is imputed as that member's average, as
``FusedDetector`` does at inference time. Members alone are scored only on the
files they scored. Load the fitted table with ``forensix.load_fusion(path)``.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from forensix.fusion import Member, fit_fusion
from forensix.metrics import binary_summary, compute_eer, roc_auc


def parse_member(spec: str) -> Member:
    model, _, opts = spec.partition(":")
    options = {}
    for kv in filter(None, opts.split(",")):
        k, v = kv.split("=", 1)
        options[k] = v
    return Member(model, options)


def read_csv(path: Path, members: list[Member]) -> tuple[list[str], np.ndarray, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = [m.key for m in members if m.key not in rows[0]]
    if missing:
        raise SystemExit(f"{path}: no column {missing}; columns are {list(rows[0])[2:]}")
    files = [r["file"] for r in rows]
    labels = np.array([{"fake": 1, "real": 0}[r["label"]] for r in rows])
    scores = np.array([[float(r[m.key]) if r[m.key] != "" else np.nan for m in members] for r in rows])
    return files, labels, scores


def summary(scores: np.ndarray, labels: np.ndarray, rng_seed: int = 0, n_boot: int = 2000) -> dict:
    fake, real = scores[labels == 1], scores[labels == 0]
    s = binary_summary(fake, real, None)
    s["eer"] = round(compute_eer(fake, real)[0], 4)
    rng = np.random.default_rng(rng_seed)
    boots = [roc_auc(fake[rng.integers(0, fake.size, fake.size)], real[rng.integers(0, real.size, real.size)])
             for _ in range(n_boot)]
    s["auc_ci95"] = [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]
    return s


def paired_gain(a: np.ndarray, b: np.ndarray, labels: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """AUC(a) - AUC(b) on the same files, with a stratified paired bootstrap CI."""
    fi, ri = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        f, r = rng.choice(fi, fi.size), rng.choice(ri, ri.size)
        diffs.append(roc_auc(a[f], a[r]) - roc_auc(b[f], b[r]))
    gain = roc_auc(a[fi], a[ri]) - roc_auc(b[fi], b[ri])
    return {"gain": round(float(gain), 4),
            "ci95": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", type=Path, required=True, help="scores CSV to fit on")
    ap.add_argument("--test", type=Path, help="scores CSV to test on (another split)")
    ap.add_argument("--members", nargs="+", required=True, help="model[:option=value,...]")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--out", type=Path, help="fusion table (JSON)")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    members = [parse_member(s) for s in args.members]
    _, y_fit, s_fit = read_csv(args.fit, members)
    for j, m in enumerate(members):
        col = s_fit[~np.isnan(s_fit[:, j]), j]
        if col.min() < 0 or col.max() > 1:
            m.score_type = "raw"
    fusion = fit_fusion(s_fit, y_fit, members, l2=args.l2, source=str(args.fit))
    print("weights (standardized): " + ", ".join(f"{m.key} {w:+.3f}" for m, w in zip(members, fusion.weights))
          + f", bias {fusion.bias:+.3f}")
    report = {"fit": str(args.fit), "members": [m.key for m in members], "fusion": fusion.to_dict()}
    if args.out:
        fusion.save(args.out)
        print(f"-> {args.out}")
    if args.test:
        _, y, s = read_csv(args.test, members)
        fused = fusion.logit(s)
        test = {"file": str(args.test), "n": int(len(y)), "fusion": summary(fused, y), "members": {}}
        print(f"test ({len(y)} files): fusion AUC {test['fusion']['auc']:.3f} "
              f"[{test['fusion']['auc_ci95'][0]:.3f}, {test['fusion']['auc_ci95'][1]:.3f}]")
        for j, m in enumerate(members):
            ok = ~np.isnan(s[:, j])
            e = summary(s[ok, j], y[ok])
            e["n"] = int(ok.sum())
            e["fusion_gain_paired"] = paired_gain(fused[ok], s[ok, j], y[ok])
            test["members"][m.key] = e
            g = e["fusion_gain_paired"]
            print(f"  {m.key:28} AUC {e['auc']:.3f} [{e['auc_ci95'][0]:.3f}, {e['auc_ci95'][1]:.3f}] on {ok.sum()} "
                  f"files; fusion gain {g['gain']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}]")
        report["test"] = test
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1) + "\n")
        print(f"-> {args.report}")


if __name__ == "__main__":
    main()
