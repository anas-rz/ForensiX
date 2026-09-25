"""Evaluate identity-based (reference-based) detectors.

Folder layout (any data you have the rights to use):

    DATA/references/<person>/*            reference photos or videos, used to enroll
    DATA/queries/<person>/real/*          genuine media of the person
    DATA/queries/<person>/fake/*          deepfakes of the person (optional)

For each person, the detector is enrolled from the references. Three query
groups are then scored against the person:
- real: their own real queries (should pass);
- fake: deepfakes of them (should be flagged);
- impostor: other people's real queries, claimed to be them (should be flagged).

It reports AUC, AP and EER of real vs fake and real vs impostor, and the
accuracy at the model's threshold.

    python scripts/eval_identity.py --data /content/id_eval --models id-face id-motion \
        --weights id-motion=id_motion.safetensors --out results/identity.json

(id-motion enrolls from videos only: give each person reference videos.)

VoxCeleb2 test (CC BY 4.0) gives real and impostor queries without any fakes:
put a few clips per speaker under references/ and the rest under queries/<id>/real/.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

import forensix
from forensix.metrics import binary_summary, compute_eer

MEDIA = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v",
         ".wav", ".flac", ".mp3", ".m4a"}


def files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in MEDIA) if folder.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=["id-face"])
    ap.add_argument("--impostors-per-person", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--weights", nargs="+", default=[], metavar="NAME=PATH",
                    help="checkpoints for models you trained yourself, e.g. id-motion=id_motion.safetensors")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    people = sorted(p.name for p in (args.data / "references").iterdir() if p.is_dir())
    real_queries = {p: files(args.data / "queries" / p / "real") for p in people}
    fake_queries = {p: files(args.data / "queries" / p / "fake") for p in people}
    print(f"{len(people)} people; {sum(map(len, real_queries.values()))} real and "
          f"{sum(map(len, fake_queries.values()))} fake queries")
    rng = random.Random(args.seed)
    results = {}
    for name in args.models:
        weights = dict(w.split("=", 1) for w in args.weights)
        det = forensix.load(name, device=args.device, weights_path=weights.get(name))
        scores = {"real": [], "fake": [], "impostor": []}
        skipped = 0
        for person in people:
            try:
                det.enroll(person, files(args.data / "references" / person))
            except ValueError as e:
                print(f"  can't enroll {person}: {e}")
                continue
        for person in people:
            if person not in det.gallery:
                continue
            others = [q for p in people if p != person for q in real_queries[p]]
            impostors = rng.sample(others, min(args.impostors_per_person, len(others)))
            for group, queries in (("real", real_queries[person]), ("fake", fake_queries[person]), ("impostor", impostors)):
                for q in queries:
                    try:
                        scores[group].append(det.predict(q, identity=person).fake_prob)
                    except ValueError:
                        skipped += 1
        r = {"n_people": len(det.gallery), "skipped": skipped, "threshold": det.threshold}
        for group in ("fake", "impostor"):
            if scores[group] and scores["real"]:
                s = binary_summary(scores[group], scores["real"], threshold=det.threshold)
                s["eer"] = round(compute_eer(np.array(scores[group]), np.array(scores["real"]))[0], 4)
                r[f"real_vs_{group}"] = s
                print(f"{name:10} real vs {group:8} AUC {s['auc']:.3f}  EER {s['eer']:.3f}  "
                      f"balanced acc {s.get('balanced_acc', float('nan')):.3f}  (n={s['n_real']}+{s['n_fake']})")
        results[name] = r
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1))
        print(f"results -> {args.out}")


if __name__ == "__main__":
    main()
