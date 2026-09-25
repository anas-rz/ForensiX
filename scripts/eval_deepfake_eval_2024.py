"""Evaluate ForensiX detectors on Deepfake-Eval-2024 (in-the-wild deepfakes, 2024).

Dataset: https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024
(CC BY-SA 4.0, evaluation only; request access on Hugging Face first).
Download it where you run the evaluation (e.g. Colab), then:

    huggingface-cli download nuriachandra/Deepfake-Eval-2024 --repo-type dataset \\
        --local-dir /content/de2024
    python scripts/eval_deepfake_eval_2024.py --data /content/de2024 \\
        --modality video --device cuda --max-per-class 250 --out results/de2024_video.json

Each modality has a folder of media (``<modality>-data/``) and a metadata CSV
(``<modality>-metadata-publish*.csv``). The file and label columns are
detected automatically and printed; override them with ``--file-col`` and
``--label-col`` if detection picks the wrong ones. Files a model can't
decode are skipped and counted, not scored.
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np

import forensix
from forensix.metrics import binary_summary

FAKE_WORDS = {"fake", "deepfake", "manipulated", "synthetic", "1", "true"}
REAL_WORDS = {"real", "authentic", "genuine", "0", "false"}


def read_metadata(data: Path, modality: str) -> list[dict]:
    candidates = sorted(data.glob(f"{modality}-metadata-publish*.csv"), key=lambda p: len(p.name))
    if not candidates:
        raise SystemExit(f"No {modality}-metadata-publish*.csv in {data}")
    with open(candidates[0], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"{candidates[0].name}: {len(rows)} rows, columns: {list(rows[0])}")
    return rows


def detect_columns(rows: list[dict], media_names: set[str]) -> tuple[str, str]:
    """File column: most values name a file in the media folder.
    Label column: all values are real/fake words."""
    stems = {Path(n).stem for n in media_names}
    file_col = label_col = None
    best = 0
    for col in rows[0]:
        values = [str(r[col]).strip() for r in rows]
        hits = sum(Path(v).name in media_names or Path(v).stem in stems for v in values if v)
        if hits > best:
            file_col, best = col, hits
        words = {v.lower() for v in values if v}
        if label_col is None and words and words <= FAKE_WORDS | REAL_WORDS and words & FAKE_WORDS and words & REAL_WORDS:
            label_col = col
    if file_col is None or label_col is None:
        raise SystemExit(f"Couldn't detect columns (file={file_col}, label={label_col}); "
                         "pass --file-col and --label-col")
    return file_col, label_col


def resolve(media_dir: Path, value: str, by_stem: dict[str, Path]) -> Path | None:
    path = media_dir / Path(value).name
    return path if path.exists() else by_stem.get(Path(value).stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="local copy of the dataset")
    parser.add_argument("--modality", choices=["video", "audio", "image"], required=True)
    parser.add_argument("--models", nargs="+", help="default: every model of the modality's family")
    parser.add_argument("--max-per-class", type=int, help="balanced random subset per class")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--file-col")
    parser.add_argument("--label-col")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, help="write results JSON here")
    parser.add_argument("--scores-out", type=Path, help="write per-file scores CSV here")
    args = parser.parse_args()

    media_dir = args.data / f"{args.modality}-data"
    media = [p for p in media_dir.iterdir() if p.is_file()]
    by_stem = {p.stem: p for p in media}
    rows = read_metadata(args.data, args.modality)
    file_col, label_col = detect_columns(rows, {p.name for p in media})
    if args.file_col:
        file_col = args.file_col
    if args.label_col:
        label_col = args.label_col
    print(f"file column: {file_col!r}, label column: {label_col!r}")

    items, missing = [], 0
    for r in rows:
        path = resolve(media_dir, str(r[file_col]).strip(), by_stem)
        label = str(r[label_col]).strip().lower()
        if path is None:
            missing += 1
        elif label in FAKE_WORDS | REAL_WORDS:
            items.append((path, label in FAKE_WORDS))
    rng = random.Random(args.seed)
    if args.max_per_class:
        fakes = sorted(i for i in items if i[1])
        reals = sorted(i for i in items if not i[1])
        items = (rng.sample(fakes, min(args.max_per_class, len(fakes)))
                 + rng.sample(reals, min(args.max_per_class, len(reals))))
    n_fake = sum(is_fake for _, is_fake in items)
    print(f"{len(items)} files ({n_fake} fake, {len(items) - n_fake} real); {missing} rows without a file")

    models = args.models or forensix.list_models(args.modality)
    results = {"dataset": "Deepfake-Eval-2024", "modality": args.modality,
               "n_fake": n_fake, "n_real": len(items) - n_fake, "seed": args.seed, "models": {}}
    all_scores = {}
    for name in models:
        detector = forensix.load(name, device=args.device)
        start = time.perf_counter()
        scores, failed = [], 0
        for k, (path, _) in enumerate(items, 1):
            try:
                scores.append(detector.predict(path).fake_prob)
            except Exception as e:  # undecodable or corrupt media
                scores.append(np.nan)
                failed += 1
                print(f"\n  {name}: skipped {path.name}: {e}")
            print(f"\r  {name}: {k}/{len(items)}", end="", flush=True)
        print()
        scores = np.array(scores)
        all_scores[name] = scores
        ok = ~np.isnan(scores)
        labels = np.array([is_fake for _, is_fake in items])
        if (ok & labels).any() and (ok & ~labels).any():
            summary = binary_summary(scores[ok & labels], scores[ok & ~labels], detector.threshold)
        else:
            summary = {"error": "no scored files for one of the classes"}
        summary.update(failed=failed, seconds=round(time.perf_counter() - start, 1))
        results["models"][name] = summary
        if "auc" in summary:
            print(f"  {name}: AUC {summary['auc']:.3f}  AP {summary['ap']:.3f}  "
                  f"bal.acc {summary['balanced_acc']:.3f}  (failed {failed})")
        else:
            print(f"  {name}: {summary['error']} (failed {failed})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"Wrote {args.out}")
    if args.scores_out:
        args.scores_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.scores_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "label", *all_scores])
            for i, (path, is_fake) in enumerate(items):
                writer.writerow([path.name, "fake" if is_fake else "real",
                                 *(f"{s[i]:.6f}" for s in all_scores.values())])


if __name__ == "__main__":
    main()
