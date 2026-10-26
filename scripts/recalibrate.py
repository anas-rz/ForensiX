"""Calibrate detectors from labelled scores, check for drift, and recalibrate.

Works on per-file score CSVs (``file,label,<model>,<model>,...``) such as the
``--scores-out`` of ``scripts/eval_video_folder.py``, so models are never re-run.
See ``forensix.calibration`` for the methods.

1. fit: cross-validated before/after report, then one calibration per model
   fitted on all rows.

       python scripts/recalibrate.py fit --scores results/sdfvd_scores.csv \\
           --group-regex '(\\d+)' --alpha 0.05 --out-dir calibrations/sdfvd \\
           --report results/calibration_sdfvd.json

   ``--group-regex`` keeps related files in the same fold (SDFVD: ``v12.mp4``
   and its face swap ``vs12.mp4`` are both group "12").

2. check: does new labelled data still match the calibrations? Exits with
   status 3 if any model needs recalibrating (for schedulers and CI).

       python scripts/recalibrate.py check --calibration-dir calibrations/sdfvd \\
           --scores new_scores.csv

3. update: refit on the stored scores plus new ones (newest kept first, older
   ones subsampled to ``--window`` per class so the calibration follows
   changing fakes). The previous table is kept in ``<dir>/history/``.

       python scripts/recalibrate.py update --calibration-dir calibrations/sdfvd \\
           --scores new_scores.csv --window 2000 --only-if-drift

A recalibration loop: score newly labelled media with eval_video_folder.py,
then run ``update --only-if-drift``. Load a table with
``forensix.load(name, calibration="calibrations/sdfvd/<name>.json")``.
"""

import argparse
import csv
import json
import random
import re
import shutil
import sys
from pathlib import Path

import numpy as np

from forensix.calibration import Calibration, drift_check, expected_calibration_error, fit_calibration
from forensix.metrics import binary_summary

DRIFT_EXIT_CODE = 3


def read_scores(paths: list[Path]) -> tuple[list[dict], list[str]]:
    """Rows ``{"file", "label", "scores": {model: float or nan}}`` and the model columns."""
    rows, models = [], []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = [c for c in reader.fieldnames if c not in ("file", "label")]
            models += [c for c in cols if c not in models]
            for r in reader:
                label = r["label"].strip().lower()
                if label not in ("real", "fake"):
                    continue
                rows.append({"file": r["file"], "label": label,
                             "scores": {c: float(r[c]) if r[c] not in ("", None) else np.nan for c in cols}})
    return rows, models


def labelled(rows: list[dict], model: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Scores, is-fake labels and row indices where ``model`` has a finite score."""
    idx = [i for i, r in enumerate(rows) if np.isfinite(r["scores"].get(model, np.nan))]
    scores = np.array([rows[i]["scores"][model] for i in idx])
    return scores, np.array([rows[i]["label"] == "fake" for i in idx]), idx


def group_folds(groups: list[str], folds: int, seed: int) -> np.ndarray:
    unique = sorted(set(groups))
    random.Random(seed).shuffle(unique)
    fold_of = {g: k % folds for k, g in enumerate(unique)}
    return np.array([fold_of[g] for g in groups])


def _decisions(scores, is_fake, flagged) -> dict:
    real_acc = float(np.mean(~flagged[~is_fake]))
    fake_acc = float(np.mean(flagged[is_fake]))
    return {"balanced_acc": round((real_acc + fake_acc) / 2, 4), "false_alarm_rate": round(1 - real_acc, 4),
            "detection_rate": round(fake_acc, 4)}


def _prob_metrics(probs, is_fake) -> dict:
    p = np.clip(probs, 1e-6, 1 - 1e-6)
    return {"brier": round(float(np.mean((p - is_fake) ** 2)), 4),
            "log_loss": round(float(-np.mean(np.where(is_fake, np.log(p), np.log1p(-p)))), 4),
            "ece": round(expected_calibration_error(p, is_fake), 4)}


def cross_validate(scores, is_fake, groups, alpha, folds, seed, raw_threshold) -> dict:
    """Out-of-fold raw vs calibrated metrics (each fold calibrated on the others)."""
    fold = group_folds(groups, folds, seed)
    prob, flagged, p_real, p_fake = (np.zeros(len(scores)) for _ in range(4))
    for k in range(folds):
        test, train = fold == k, fold != k
        if not test.any():
            continue
        cal = fit_calibration("cv", scores[train & is_fake], scores[train & ~is_fake], alpha)
        prob[test] = cal.prob(scores[test])
        flagged[test] = scores[test] > cal.threshold
        p_real[test], p_fake[test] = cal.p_values(scores[test])
    flagged = flagged.astype(bool)
    in_real, in_fake = p_real > alpha, p_fake > alpha
    covered = np.where(is_fake, in_fake, in_real)
    size = in_real.astype(int) + in_fake.astype(int)
    return {
        "raw": {"threshold": raw_threshold, **_decisions(scores, is_fake, scores >= raw_threshold),
                **_prob_metrics(scores, is_fake)},
        "platt": {**_decisions(scores, is_fake, prob >= 0.5), **_prob_metrics(prob, is_fake)},
        "conformal": {"alpha": alpha, **_decisions(scores, is_fake, flagged)},
        "prediction_sets": {"coverage_real": round(float(covered[~is_fake].mean()), 4),
                            "coverage_fake": round(float(covered[is_fake].mean()), 4),
                            "single_label": round(float(np.mean(size == 1)), 4),
                            "both_labels": round(float(np.mean(size == 2)), 4),
                            "empty": round(float(np.mean(size == 0)), 4)},
    }


def cmd_fit(args) -> None:
    rows, models = read_scores(args.scores)
    models = args.models or models
    pattern = re.compile(args.group_regex) if args.group_regex else None
    report = {"scores": [str(p) for p in args.scores], "alpha": args.alpha, "folds": args.folds,
              "seed": args.seed, "group_regex": args.group_regex, "models": {}}
    for model in models:
        scores, is_fake, idx = labelled(rows, model)
        files = [rows[i]["file"] for i in idx]
        groups = [m.group(1) if pattern and (m := pattern.search(f)) else f for f in files]
        entry = {"n_real": int((~is_fake).sum()), "n_fake": int(is_fake.sum()),
                 **{k: v for k, v in binary_summary(scores[is_fake], scores[~is_fake], None).items()
                    if k in ("auc", "ap")}}
        try:
            entry["cv"] = cross_validate(scores, is_fake, groups, args.alpha, args.folds, args.seed,
                                         args.raw_threshold)
            cal = fit_calibration(model, scores[is_fake], scores[~is_fake], args.alpha, args.source)
        except ValueError as e:
            entry["error"] = str(e)
            print(f"{model}: {e}")
            report["models"][model] = entry
            continue
        cal.save(args.out_dir / f"{model}.json")
        entry.update(platt=list(cal.platt), conformal_threshold=cal.threshold)
        report["models"][model] = entry
        cv = entry["cv"]
        print(f"{model:12} AUC {entry['auc']:.3f} | bal.acc raw {cv['raw']['balanced_acc']:.3f} -> "
              f"platt {cv['platt']['balanced_acc']:.3f}, conformal {cv['conformal']['balanced_acc']:.3f} "
              f"(false alarms {cv['conformal']['false_alarm_rate']:.1%}, alpha {args.alpha:.0%}) | "
              f"ECE {cv['raw']['ece']:.3f} -> {cv['platt']['ece']:.3f}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1) + "\n")
        print(f"report -> {args.report}")
    print(f"calibrations -> {args.out_dir}")


def _load_tables(directory: Path, models: list[str] | None) -> dict[str, Calibration]:
    tables = {p.stem: Calibration.load(p) for p in sorted(directory.glob("*.json"))}
    return {m: t for m, t in tables.items() if not models or m in models}


def _check(tables, rows, level) -> dict:
    out = {}
    for model, cal in tables.items():
        scores, is_fake, _ = labelled(rows, model)
        if not len(scores):
            continue
        r = drift_check(cal, scores[is_fake], scores[~is_fake], level)
        out[model] = r
        failed = [k for k, p in r["tests"].items() if p < level]
        print(f"{model:12} {'RECALIBRATE' if r['recalibrate'] else 'ok':11} "
              f"false alarms {r.get('false_alarm_rate', float('nan')):.1%} (alpha {cal.alpha:.0%}), "
              f"detection {r.get('detection_rate', float('nan')):.1%} "
              f"(was {r.get('calibration_detection_rate', float('nan')):.1%})"
              + (f"  [{', '.join(failed)}]" if failed else ""))
    return out


def cmd_check(args) -> None:
    rows, _ = read_scores(args.scores)
    report = _check(_load_tables(args.calibration_dir, args.models), rows, args.level)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")
    if any(r["recalibrate"] for r in report.values()):
        sys.exit(DRIFT_EXIT_CODE)


def _window(new: np.ndarray, old: list[float], window: int | None, rng: random.Random) -> np.ndarray:
    """All new scores, then older ones subsampled so the total is at most ``window``."""
    if window is None:
        return np.concatenate([new, old])
    keep = max(window - len(new), 0)
    return np.concatenate([new[-window:], rng.sample(list(old), min(keep, len(old)))])


def cmd_update(args) -> None:
    rows, _ = read_scores(args.scores)
    tables = _load_tables(args.calibration_dir, args.models)
    drift = _check(tables, rows, args.level)
    rng = random.Random(args.seed)
    for model, cal in tables.items():
        if model not in drift or (args.only_if_drift and not drift[model]["recalibrate"]):
            continue
        scores, is_fake, _ = labelled(rows, model)
        fake = _window(scores[is_fake], cal.fake_scores, args.window, rng)
        real = _window(scores[~is_fake], cal.real_scores, args.window, rng)
        source = ", ".join(s for s in (cal.source, args.source) if s)
        try:
            new = fit_calibration(model, fake, real, args.alpha or cal.alpha, source)
        except ValueError as e:
            print(f"{model}: not updated: {e}")
            continue
        history = args.calibration_dir / "history"
        history.mkdir(exist_ok=True)
        archived = history / f"{model}-{cal.created}.json"
        n = 1
        while archived.exists():
            archived, n = history / f"{model}-{cal.created}-{n}.json", n + 1
        shutil.copy(args.calibration_dir / f"{model}.json", archived)
        new.save(args.calibration_dir / f"{model}.json")
        print(f"{model:12} updated: {len(real)} real + {len(fake)} fake scores, threshold "
              f"{cal.threshold:.4g} -> {new.threshold:.4g} (previous table -> {archived})")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fit", help="cross-validated report and one calibration per model")
    f.add_argument("--scores", type=Path, nargs="+", required=True, help="per-file scores CSV(s)")
    f.add_argument("--models", nargs="+", help="model columns (default: all)")
    f.add_argument("--alpha", type=float, default=0.05, help="target false-alarm rate on real media")
    f.add_argument("--group-regex", help="regex whose first group names a file's group (kept in one fold)")
    f.add_argument("--folds", type=int, default=5)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--raw-threshold", type=float, default=0.5, help="the detectors' default threshold")
    f.add_argument("--source", default="", help="dataset tag stored in the calibrations")
    f.add_argument("--out-dir", type=Path, required=True)
    f.add_argument("--report", type=Path)
    f.set_defaults(func=cmd_fit)

    for name, func, help_ in (("check", cmd_check, f"drift check (exit {DRIFT_EXIT_CODE} if any model drifted)"),
                              ("update", cmd_update, "refit calibrations with new labelled scores")):
        c = sub.add_parser(name, help=help_)
        c.add_argument("--calibration-dir", type=Path, required=True)
        c.add_argument("--scores", type=Path, nargs="+", required=True)
        c.add_argument("--models", nargs="+")
        c.add_argument("--level", type=float, default=0.01, help="significance level of the drift tests")
        if name == "check":
            c.add_argument("--out", type=Path)
        else:
            c.add_argument("--window", type=int, help="max scores kept per class (newest first)")
            c.add_argument("--alpha", type=float, help="new target false-alarm rate (default: keep)")
            c.add_argument("--only-if-drift", action="store_true")
            c.add_argument("--source", default="")
            c.add_argument("--seed", type=int, default=0)
        c.set_defaults(func=func)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
