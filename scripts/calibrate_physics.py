"""Learn the physdf priors from real media, then evaluate them on real vs fake.

Three steps, meant for Colab (downloading the data is up to you; nothing here
fetches datasets):

1. measure: run the physics detectors over a folder of media once, storing the
   raw constraint values per clip (JSONL, resumable).

       python scripts/calibrate_physics.py measure --media /content/LibriSpeech/test-clean \\
           --kind audio --label real --source librispeech-test-clean \\
           --max-files 2000 --out measurements/librispeech_test_clean.jsonl
       python scripts/calibrate_physics.py measure --media /content/voxceleb2_test/mp4 \\
           --kind video --with-audio --label real --source voxceleb2-test \\
           --max-files 1000 --out measurements/voxceleb2_test.jsonl

2. fit: fit priors on the real clips only, report the false-alarm rate on a
   held-out set of speakers/channels, and write the table (use it with
   ``detector.load_priors(path)``, or copy it to
   ``forensix/physics/learned_priors.json`` to make it the default).

       python scripts/calibrate_physics.py fit --measurements measurements/*.jsonl \\
           --alpha 0.01 --out learned_priors.json

3. eval: real vs fake AUC per detector and per constraint, from stored
   measurements. Use real clips that were not used to fit.

       python scripts/calibrate_physics.py measure --media /content/de2024/video-data \\
           --kind video --with-audio --labels-csv /content/de2024/video-metadata-publish.csv \\
           --source deepfake-eval-2024 --out measurements/de2024.jsonl
       python scripts/calibrate_physics.py eval --measurements measurements/de2024.jsonl \\
           --priors learned_priors.json --out results/physics_de2024.json

Suggested real data (commercially usable licenses):
- audio: LibriSpeech (CC BY 4.0, openslr.org/12), read speech from many
  microphones. VCTK (CC BY 4.0) is studio speech, in utterances of about 3 s:
  too short for phys-room (6 s).
- video: VoxCeleb2 test (CC BY 4.0; the copyright of the videos stays with
  their owners), interviews with sound. Its clips are face crops, so the
  scene-coupling check of phys-lighting rarely applies.
Calibrate on media like the media you'll check: priors fitted on clean read
speech will flag noisy phone calls.

The group of a clip (the unit of the held-out split) is its first folder
under ``--media`` (the speaker in LibriSpeech, VCTK and VoxCeleb).
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from forensix.metrics import binary_summary
from forensix.physics import AUDIO_DETECTORS, DETECTORS, VIDEO_DETECTORS
from forensix.physics.calibration import (
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    false_alarm_summary,
    in_holdout,
    measure_file,
    measurements_by_detector,
    oriented,
    score_rows,
)
from forensix.physics.priors import fit_priors, provisional_priors

FAKE_WORDS = {"fake", "deepfake", "manipulated", "synthetic", "spoof", "1", "true"}
REAL_WORDS = {"real", "authentic", "genuine", "bonafide", "bona-fide", "0", "false"}


def read_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def read_labels(path: Path, file_col: str | None, label_col: str | None) -> dict[str, str]:
    """``{file name: "real" | "fake"}`` from a metadata CSV (columns detected, or given)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = list(rows[0])
    file_col = file_col or next(c for c in cols if any(w in c.lower() for w in ("file", "path", "name")))
    label_col = label_col or next(c for c in cols if any(w in c.lower() for w in ("label", "fake", "class")))
    print(f"labels: file column {file_col!r}, label column {label_col!r}")
    out = {}
    for r in rows:
        value = r[label_col].strip().lower()
        label = "fake" if value in FAKE_WORDS else "real" if value in REAL_WORDS else None
        if label:
            out[Path(r[file_col]).name] = label
    return out


def cmd_measure(args) -> None:
    extensions = AUDIO_EXTENSIONS if args.kind == "audio" else VIDEO_EXTENSIONS
    files = sorted(p for p in args.media.rglob("*") if p.suffix.lower() in extensions)
    labels = read_labels(args.labels_csv, args.file_col, args.label_col) if args.labels_csv else None
    if labels is not None:
        files = [p for p in files if p.name in labels]
    random.Random(args.seed).shuffle(files)
    files = files[: args.max_files] if args.max_files else files
    names = [n for n in DETECTORS if n in (args.detectors or DETECTORS)]
    if args.kind == "audio":
        names = [n for n in names if n in AUDIO_DETECTORS]
    elif not args.with_audio:
        names = [n for n in names if n in VIDEO_DETECTORS]
    detectors = {n: DETECTORS[n]() for n in names}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["file"] for r in read_rows([args.out])} if args.out.exists() else set()
    print(f"{len(files)} files ({len(done)} already measured), detectors: {', '.join(names)}")
    t0, n_new = time.time(), 0
    with open(args.out, "a", encoding="utf-8") as out:
        for i, path in enumerate(files):
            rel = path.relative_to(args.media).as_posix()
            if rel in done:
                continue
            row = {"file": rel, "group": rel.split("/")[0] if "/" in rel else rel,
                   "label": labels[path.name] if labels else args.label, "source": args.source,
                   "kind": args.kind, "detectors": measure_file(path, args.kind, detectors, args.with_audio)}
            out.write(json.dumps(row) + "\n")
            out.flush()
            n_new += 1
            if n_new % 20 == 0:
                rate = (time.time() - t0) / n_new
                print(f"  {i + 1}/{len(files)}  {rate:.1f} s/file", flush=True)
    print(f"measured {n_new} new files in {time.time() - t0:.0f} s -> {args.out}")


def _print_table(table: dict) -> None:
    for detector, constraints in table["detectors"].items():
        print(detector)
        for k, e in constraints.items():
            p0, p1 = e["provisional"]
            print(f"  {k:34} start {e['start']:>10.4g}  full {e['full']:>10.4g}   "
                  f"(provisional {p0:.4g} -> {p1:.4g}; real median {e['median']:.4g}, n={e['n']})")
    for detector, kept in table["kept_provisional"].items():
        print(f"{detector}: too few real measurements, provisional kept for {', '.join(kept)}")


def cmd_fit(args) -> None:
    rows = [r for r in read_rows(args.measurements) if r["label"] == "real"]
    if not rows:
        sys.exit("No real clips in the measurements")
    fit_rows = [r for r in rows if not in_holdout(r["group"], args.holdout)]
    held = [r for r in rows if in_holdout(r["group"], args.holdout)]
    provisional = provisional_priors()
    sources = sorted({r["source"] for r in rows if r.get("source")})
    print(f"{len(rows)} real clips from {len({r['group'] for r in rows})} groups; "
          f"fitting on {len(fit_rows)}, holding out {len(held)}")

    # held-out false alarms: priors fitted without the held-out groups
    train_table = fit_priors(measurements_by_detector(fit_rows), provisional, args.alpha, sources)
    report = {"alpha": args.alpha, "n_real": len(rows), "n_holdout": len(held), "holdout_false_alarms": {}}
    print(f"\nheld-out real clips (target: any violation on about {args.alpha:.0%} per detector)")
    for detector in DETECTORS:
        learned = false_alarm_summary(score_rows(held, detector, train_table)[0])
        before = false_alarm_summary(score_rows(held, detector, {"format": train_table["format"]})[0])
        report["holdout_false_alarms"][detector] = {"learned": learned, "provisional": before}
        if learned["n"]:
            print(f"  {detector:15} learned: {learned['any_violation']:.1%} any, {learned['score_ge_0.5']:.1%} >= 0.5"
                  f"   provisional: {before['any_violation']:.1%} any, {before['score_ge_0.5']:.1%} >= 0.5"
                  f"   (n={learned['n']})")

    table = fit_priors(measurements_by_detector(rows), provisional, args.alpha, sources)  # final: all real clips
    table["holdout_false_alarms"] = report["holdout_false_alarms"]
    print()
    _print_table(table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=1))
    print(f"\npriors -> {args.out}")


def cmd_eval(args) -> None:
    rows = read_rows(args.measurements)
    real = [r for r in rows if r["label"] == "real"]
    fake = [r for r in rows if r["label"] == "fake"]
    print(f"{len(real)} real, {len(fake)} fake clips")
    provisional = provisional_priors()
    results = {"priors": str(args.priors) if args.priors else "default", "detectors": {}, "constraints": {}}
    per_clip = {}  # row id -> max over detectors (physics-max fusion)
    for detector in DETECTORS:
        s_real, i_real = score_rows(real, detector, args.priors)
        s_fake, i_fake = score_rows(fake, detector, args.priors)
        for s, i, tag in ((s_real, i_real, "r"), (s_fake, i_fake, "f")):
            for score, j in zip(s, i):
                per_clip[(tag, j)] = max(per_clip.get((tag, j), 0.0), score)
        if len(s_real) and len(s_fake):
            summary = binary_summary(s_fake, s_real, threshold=None)
            summary["real_false_alarms"] = false_alarm_summary(s_real)
            summary["fake_flagged"] = round(float(np.mean(s_fake > 0)), 4)
            results["detectors"][detector] = summary
            print(f"{detector:15} AUC {summary['auc']:.3f}  AP {summary['ap']:.3f}  "
                  f"real flagged {summary['real_false_alarms']['any_violation']:.1%}  "
                  f"fake flagged {summary['fake_flagged']:.1%}  (n={len(s_real)}+{len(s_fake)})")
        # per constraint: the raw measurement as a score, on clips where it applies
        vals_r = measurements_by_detector(real).get(detector, {})
        vals_f = measurements_by_detector(fake).get(detector, {})
        for k, prior in provisional[detector].items():
            r, f = oriented(vals_r.get(k, []), prior), oriented(vals_f.get(k, []), prior)
            r, f = r[np.isfinite(r)], f[np.isfinite(f)]
            if len(r) and len(f):
                results["constraints"][f"{detector}/{k}"] = binary_summary(f, r, threshold=None)
    fused_r = [v for (t, _), v in per_clip.items() if t == "r"]
    fused_f = [v for (t, _), v in per_clip.items() if t == "f"]
    if fused_r and fused_f:
        results["physics_max"] = binary_summary(fused_f, fused_r, threshold=None)
        print(f"{'physics-max':15} AUC {results['physics_max']['auc']:.3f}  (max over the detectors that ran)")
    print("\nper constraint (raw value as the score):")
    for k, s in results["constraints"].items():
        print(f"  {k:50} AUC {s['auc']:.3f}  (n={s['n_real']}+{s['n_fake']})")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1))
        print(f"\nresults -> {args.out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    m = sub.add_parser("measure", help="run the physics detectors over a folder of media")
    m.add_argument("--media", type=Path, required=True)
    m.add_argument("--kind", choices=["audio", "video"], required=True)
    m.add_argument("--with-audio", action="store_true", help="also measure the sound track of videos (ffmpeg)")
    m.add_argument("--label", choices=["real", "fake"], default="real")
    m.add_argument("--labels-csv", type=Path, help="per-file labels instead of --label")
    m.add_argument("--file-col")
    m.add_argument("--label-col")
    m.add_argument("--source", default="", help="dataset tag stored with each clip")
    m.add_argument("--detectors", nargs="+", choices=list(DETECTORS))
    m.add_argument("--max-files", type=int)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", type=Path, required=True)
    m.set_defaults(func=cmd_measure)

    f = sub.add_parser("fit", help="fit priors on the real clips")
    f.add_argument("--measurements", type=Path, nargs="+", required=True)
    f.add_argument("--alpha", type=float, default=0.01, help="target false-alarm rate per detector")
    f.add_argument("--holdout", type=float, default=0.2, help="fraction of groups held out for the report")
    f.add_argument("--out", type=Path, required=True)
    f.set_defaults(func=cmd_fit)

    e = sub.add_parser("eval", help="real vs fake AUC from measurements")
    e.add_argument("--measurements", type=Path, nargs="+", required=True)
    e.add_argument("--priors", type=Path, help="priors table (default: packaged or provisional)")
    e.add_argument("--out", type=Path)
    e.set_defaults(func=cmd_eval)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
