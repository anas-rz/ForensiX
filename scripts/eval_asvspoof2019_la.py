"""Evaluate audio detectors on a reproducible subset of ASVspoof 2019 LA eval.

Dataset: ASVspoof 2019 LA (ODC-By 1.0, https://datashare.ed.ac.uk/handle/10283/3336).
Only LA.zip is needed; point the script at its protocol file and eval flac folder:

    python scripts/eval_asvspoof2019_la.py \\
        --protocol LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \\
        --audio-dir LA/ASVspoof2019_LA_eval/flac \\
        --models aasist aasist-l --out results/asvspoof2019_la_subset.json

Subset: ``--n-bonafide`` bona fide trials plus ``--n-per-attack`` trials from each
of the 13 eval attacks (A07-A19), sampled with a fixed seed. Reports pooled EER
and per-attack EER (each attack vs. the same bona fide set).
"""

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import forensix
from forensix.metrics import compute_eer


def read_protocol(path: Path) -> list[tuple[str, str]]:
    """Return ``(utt_id, attack)`` pairs; attack is ``"bonafide"`` for real speech."""
    trials = []
    for line in path.read_text().splitlines():
        # e.g. "LA_0039 LA_E_2834763 - A11 spoof" / "LA_0069 LA_E_1665632 - - bonafide"
        _, utt_id, _, attack, key = line.split()
        trials.append((utt_id, "bonafide" if key == "bonafide" else attack))
    return trials


def select_subset(
    trials: list[tuple[str, str]], n_bonafide: int, n_per_attack: int, seed: int
) -> list[tuple[str, str]]:
    by_attack = defaultdict(list)
    for utt_id, attack in trials:
        by_attack[attack].append(utt_id)
    rng = random.Random(seed)
    subset = []
    for attack in sorted(by_attack):
        ids = sorted(by_attack[attack])
        n = n_bonafide if attack == "bonafide" else n_per_attack
        subset += [(u, attack) for u in rng.sample(ids, min(n, len(ids)))]
    return subset


def evaluate(detector, subset, audio_dir: Path, batch_size: int) -> dict:
    paths = [audio_dir / f"{utt_id}.flac" for utt_id, _ in subset]
    start = time.perf_counter()
    scores = []
    for i in range(0, len(paths), batch_size):
        preds = detector.predict_batch(paths[i : i + batch_size], batch_size=batch_size)
        scores += [p.fake_prob for p in preds]
        print(f"\r  {detector.name}: {len(scores)}/{len(paths)}", end="", flush=True)
    print()
    elapsed = time.perf_counter() - start

    scores = np.array(scores)
    attacks = np.array([a for _, a in subset])
    real = scores[attacks == "bonafide"]
    eer, thr = compute_eer(scores[attacks != "bonafide"], real)
    per_attack = {
        a: round(100 * compute_eer(scores[attacks == a], real)[0], 2)
        for a in sorted(set(attacks) - {"bonafide"})
    }
    acc = float(np.mean((scores >= detector.threshold) == (attacks != "bonafide")))
    return {
        "eer_percent": round(100 * eer, 2),
        "eer_threshold": thr,
        "accuracy_at_default_threshold": round(acc, 4),
        "per_attack_eer_percent": per_attack,
        "seconds": round(elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=forensix.list_models("audio"))
    parser.add_argument("--n-bonafide", type=int, default=1000)
    parser.add_argument("--n-per-attack", type=int, default=77)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, help="write results JSON here")
    parser.add_argument("--save-subset", type=Path, help="write selected utt ids here")
    args = parser.parse_args()

    subset = select_subset(
        read_protocol(args.protocol), args.n_bonafide, args.n_per_attack, args.seed
    )
    counts = defaultdict(int)
    for _, attack in subset:
        counts[attack] += 1
    print(f"Subset: {len(subset)} trials " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if args.save_subset:
        args.save_subset.write_text("".join(f"{u} {a}\n" for u, a in subset))

    results = {
        "dataset": "ASVspoof 2019 LA eval (ODC-By 1.0)",
        "subset": {
            "n_bonafide": args.n_bonafide,
            "n_per_attack": args.n_per_attack,
            "seed": args.seed,
            "counts": dict(sorted(counts.items())),
        },
        "models": {},
    }
    for name in args.models:
        detector = forensix.load(name, device=args.device)
        results["models"][name] = evaluate(detector, subset, args.audio_dir, args.batch_size)
        r = results["models"][name]
        print(f"  {name}: EER {r['eer_percent']}%  per-attack {r['per_attack_eer_percent']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
