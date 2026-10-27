"""Evaluate audio-visual detectors on LAV-DF (localized audio-visual deepfakes).

Dataset: https://huggingface.co/datasets/ControlNet/LAV-DF (CC BY-NC 4.0,
non-commercial research and education only; built from VoxCeleb2, whose terms
also apply). Accept the terms on Hugging Face, then download it where you run
the evaluation (25.6 GB), not into this repo:

    hf download ControlNet/LAV-DF LAV-DF.tar --repo-type dataset --local-dir /content/lavdf
    tar -xf /content/lavdf/LAV-DF.tar -C /content/lavdf

Every fake replaces a few words (one or two segments of about 0.2-1.1 s) in
the visual track, the audio track or both. The four classes (real, audio,
video, both) are sampled equally.

1. score: per-clip scores and per-window scores (JSONL, resumable). Run
   several shards in parallel with ``--shard k/n``.

       python scripts/eval_lavdf.py score --data /content/lavdf/LAV-DF --split test \\
           --per-class 300 --models syncnet stablesyncnet avad --device cuda --out-dir results/lavdf

2. report: AUC with bootstrap CIs, overall and per manipulated modality,
   window-level localization (windows inside a fake segment vs windows of
   real clips), and an alternative clip score (the most suspicious window,
   column ``<model>-max-window``). Writes a ``file,label,<model>...`` CSV for
   scripts/recalibrate.py.
   ``--calibration-dir`` also scores calibration tables fitted on another
   split (out-of-sample false alarms, detection, ECE).
   The JSONL files can be gzipped to store them (``gzip *.jsonl``); report reads both,
   but score resumes only from uncompressed files.

       python scripts/eval_lavdf.py report --data /content/lavdf/LAV-DF --split test \\
           --out-dir results/lavdf --calibration-dir calibrations/lavdf
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np

CATEGORIES = ("real", "audio", "video", "both")  # what was modified
FPS = 25  # syncnet frame-wise confidences are per 25 fps frame


def category(row: dict) -> str:
    return {(False, False): "real", (False, True): "audio", (True, False): "video",
            (True, True): "both"}[(row["modify_video"], row["modify_audio"])]


def select(data: Path, split: str, per_class: int, seed: int) -> list[dict]:
    rows = [r for r in json.loads((data / "metadata.min.json").read_text()) if r["split"] == split]
    rng = random.Random(seed)
    out = []
    for c in CATEGORIES:
        pool = sorted((r for r in rows if category(r) == c), key=lambda r: r["file"])
        out += rng.sample(pool, min(per_class, len(pool)))
    return out


def windows(model: str, pred) -> list[list[float]] | None:
    """[start_s, end_s, fake score] of each window the detector scores, if it reports them."""
    d = pred.details or {}
    if model == "syncnet" and "framewise_confidence" in d:  # sync confidence per frame, higher = in sync
        return [[round(d["start_s"] + t / FPS, 3), round(d["start_s"] + (t + 1) / FPS, 3), -c]
                for t, c in enumerate(d["framewise_confidence"])]
    if model == "stablesyncnet" and "window_start_s" in d:
        return [[s, round(s + d["window_s"], 3), round(1 - p, 4)]
                for s, p in zip(d["window_start_s"], d["window_sync_probability"])]
    if model == "avad" and "segment_score" in d:  # overlapping 50-step segments, every 1 s
        return [[s, round(s + w, 3), v] for s, w, v in zip(d["segment_start_s"], d["segment_s"], d["segment_score"])]
    return None


def out_path(out_dir: Path, split: str, model: str, shard: str) -> Path:
    k, n = shard.split("/")
    suffix = "" if n == "1" else f".part{k}"
    return out_dir / split / f"{model}{suffix}.jsonl"


def cmd_score(args) -> None:
    import forensix

    items = select(args.data, args.split, args.per_class, args.seed)
    k, n = map(int, args.shard.split("/"))
    items = items[k::n]
    for model in args.models:
        path = out_path(args.out_dir, args.split, model, args.shard)
        path.parent.mkdir(parents=True, exist_ok=True)
        done = {json.loads(line)["file"] for line in path.open()} if path.exists() else set()
        todo = [r for r in items if r["file"] not in done]
        print(f"{model}: {len(items)} clips ({len(items) - len(todo)} already scored)", flush=True)
        if not todo:
            continue
        detector = forensix.load(model, device=args.device)
        start = time.perf_counter()
        with path.open("a") as f:
            for i, r in enumerate(todo, 1):
                row = {"file": r["file"], "category": category(r), "fake_periods": r["fake_periods"],
                       "duration": r["duration"]}
                try:
                    pred = detector.predict(args.data / r["file"])
                    row.update(score=float(pred.fake_prob), windows=windows(model, pred))
                except Exception as e:  # no face track, too short, undecodable
                    row["error"] = f"{type(e).__name__}: {e}"
                f.write(json.dumps(row) + "\n")
                f.flush()
                if i % 50 == 0 or i == len(todo):
                    rate = (time.perf_counter() - start) / i
                    print(f"  {model}: {i}/{len(todo)}  {rate:.2f} s/clip", flush=True)


def read_rows(out_dir: Path, split: str, model: str) -> list[dict]:
    """Scored clips of one model, from its JSONL files (gzipped or not)."""
    import gzip

    rows = {}
    for pattern in (f"{model}.jsonl", f"{model}.part*.jsonl"):
        for path in sorted((out_dir / split).glob(pattern)) + sorted((out_dir / split).glob(pattern + ".gz")):
            with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as f:
                for line in f:
                    r = json.loads(line)
                    rows[r["file"]] = r
    return list(rows.values())


def auc_ci(fake, real, n_boot: int = 2000, seed: int = 0) -> dict:
    from forensix.metrics import binary_summary, compute_eer, roc_auc

    fake, real = np.asarray(fake, np.float64), np.asarray(real, np.float64)
    s = binary_summary(fake, real, None)
    s["eer"] = round(compute_eer(fake, real)[0], 4)
    rng = np.random.default_rng(seed)
    boots = [roc_auc(fake[rng.integers(0, fake.size, fake.size)], real[rng.integers(0, real.size, real.size)])
             for _ in range(n_boot)]
    s["auc_ci95"] = [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]
    return s


def window_labels(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Scores of windows that overlap a fake segment (fake clips) and of all windows of real clips."""
    fake, real = [], []
    for r in rows:
        for s, e, score in r.get("windows") or []:
            if r["category"] == "real":
                real.append(score)
            elif any(s < pe and e > ps for ps, pe in r["fake_periods"]):
                fake.append(score)
    return np.array(fake), np.array(real)


def calibration_check(table_path: Path, fake: np.ndarray, real: np.ndarray) -> dict:
    """A table fitted on another split, applied to these scores."""
    from forensix.calibration import Calibration, expected_calibration_error

    cal = Calibration.load(table_path)
    scores = np.concatenate([fake, real])
    labels = np.r_[np.ones(fake.size), np.zeros(real.size)]
    prob = cal.prob(scores)
    p_real, p_fake = cal.p_values(scores)
    in_real, in_fake = p_real > cal.alpha, p_fake > cal.alpha
    return {
        "table": str(table_path), "source": cal.source, "alpha": cal.alpha,
        "false_alarm_rate": round(float(np.mean(real > cal.threshold)), 4),
        "detection_rate": round(float(np.mean(fake > cal.threshold)), 4),
        "platt_balanced_acc": round(float((np.mean(prob[labels == 1] >= 0.5) + np.mean(prob[labels == 0] < 0.5)) / 2), 4),
        "platt_ece": round(expected_calibration_error(prob, labels), 4),
        "coverage_real": round(float(in_real[labels == 0].mean()), 4),
        "coverage_fake": round(float(in_fake[labels == 1].mean()), 4),
        "single_label": round(float(np.mean(in_real != in_fake)), 4),
    }


def cmd_report(args) -> None:
    report = {"dataset": "LAV-DF", "split": args.split, "models": {}}
    table = {}
    columns = []
    for model in args.models:
        rows = read_rows(args.out_dir, args.split, model)
        if not rows:
            continue
        ok = [r for r in rows if "score" in r]
        by_cat = {c: np.array([r["score"] for r in ok if r["category"] == c]) for c in CATEGORIES}
        real = by_cat["real"]
        fake = np.concatenate([by_cat[c] for c in CATEGORIES[1:]])
        entry = {"n": len(rows), "failed": {c: sum(1 for r in rows if "error" in r and r["category"] == c)
                                            for c in CATEGORIES},
                 "clip": auc_ci(fake, real),
                 "clip_by_category": {c: auc_ci(by_cat[c], real) for c in CATEGORIES[1:]}}
        fake_files = {r["file"] for r in ok if r["category"] != "real"}
        with_windows = [r for r in ok if r.get("windows")]
        if with_windows:
            wf, wr = window_labels(with_windows)
            entry["window"] = auc_ci(wf, wr)
            peak = {r["file"]: max(w[2] for w in r["windows"]) for r in with_windows}
            pf = np.array([peak[r["file"]] for r in with_windows if r["category"] != "real"])
            pr = np.array([peak[r["file"]] for r in with_windows if r["category"] == "real"])
            entry["clip_max_window"] = auc_ci(pf, pr)
        variants = {model: (fake, real, {r["file"]: r["score"] for r in ok})}
        if with_windows:
            variants[f"{model}-max-window"] = (pf, pr, peak)
        if model == "avad" and with_windows:  # avad's other clip scores (forensix.load("avad", clip_score=...))
            for name, rule in (("first", lambda w: w[0][2]), ("mean", lambda w: float(np.mean([x[2] for x in w])))):
                by_file = {r["file"]: rule(r["windows"]) for r in with_windows}
                vf = np.array([by_file[r["file"]] for r in with_windows if r["category"] != "real"])
                vr = np.array([by_file[r["file"]] for r in with_windows if r["category"] == "real"])
                entry[f"clip_{name}_segment"] = auc_ci(vf, vr)
                variants[f"avad-{name}"] = (vf, vr, by_file)
        for name, (vf, vr, by_file) in variants.items():
            path = args.calibration_dir / f"{name}.json" if args.calibration_dir else None
            if path and path.exists():
                key = "calibration" if name == model else "calibration_" + name[len(model) + 1 :].replace("-", "_")
                entry[key] = calibration_check(path, vf, vr)
            columns.append(name)
            for file, score in by_file.items():
                table.setdefault(file, {"label": "fake" if file in fake_files else "real"})[name] = score
        report["models"][model] = entry
        c = entry["clip"]
        line = f"{model:14} clip AUC {c['auc']:.3f} [{c['auc_ci95'][0]:.3f}, {c['auc_ci95'][1]:.3f}]  " + "  ".join(
            f"{k} {v['auc']:.3f}" for k, v in entry["clip_by_category"].items())
        if "window" in entry:
            line += f"  | window AUC {entry['window']['auc']:.3f}, max-window clip AUC {entry['clip_max_window']['auc']:.3f}"
        for name in ("first", "mean"):
            if f"clip_{name}_segment" in entry:
                line += f", {name}-segment clip AUC {entry[f'clip_{name}_segment']['auc']:.3f}"
        print(line)
        for key in [k for k in entry if k.startswith("calibration")]:
            if key in entry:
                k = entry[key]
                print(f"{'':14} {key} ({k['source']}): false alarms {k['false_alarm_rate']:.1%} at alpha "
                      f"{k['alpha']:.0%}, detection {k['detection_rate']:.1%}, Platt ECE {k['platt_ece']:.3f}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"lavdf_{args.split}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    with open(args.out_dir / f"lavdf_{args.split}_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", *columns])
        for file, r in sorted(table.items()):
            w.writerow([file, r["label"], *(repr(r[m]) if m in r else "" for m in columns)])
    print(f"-> {out}, {args.out_dir / f'lavdf_{args.split}_scores.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("score", "report"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, required=True, help="LAV-DF folder (metadata.min.json, train/dev/test)")
        p.add_argument("--split", default="test", choices=["train", "dev", "test"])
        p.add_argument("--models", nargs="+", default=["syncnet", "stablesyncnet", "avad"])
        p.add_argument("--out-dir", type=Path, default=Path("results/lavdf"))
        if name == "score":
            p.add_argument("--per-class", type=int, default=300)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--device", default="cuda")
            p.add_argument("--shard", default="0/1", help="k/n: score every n-th clip from the k-th")
        else:
            p.add_argument("--calibration-dir", type=Path, help="tables fitted on another split")
    args = ap.parse_args()
    {"score": cmd_score, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
