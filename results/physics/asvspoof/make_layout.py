"""Lay out ASVspoof 2019 LA for scripts/calibrate_physics.py (symlinks only).

- fit:  train + dev bona fide, as <out>/fit_<shard>/<speaker>/<utt>.flac, so the
        calibration script's group (first folder) is the speaker and the
        held-out split is by speaker. Speakers are spread over --shards folders
        so several measure processes can run in parallel.
- eval: the eval subset of scripts/eval_asvspoof2019_la.py (select_subset,
        1000 bona fide + 77 per attack, seed 0), as <out>/eval_<shard>/<speaker>/<utt>.flac,
        plus <out>/eval_labels.csv (file,label,attack) for --labels-csv.

Usage: python asvspoof_layout.py LA_DIR OUT_DIR REPO_DIR [--shards 6]
"""
import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("la", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("repo", type=Path)
ap.add_argument("--shards", type=int, default=6)
args = ap.parse_args()

spec = importlib.util.spec_from_file_location("eval_la", args.repo / "scripts/eval_asvspoof2019_la.py")
sys.path.insert(0, str(args.repo))
eval_la = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_la)

proto = args.la / "ASVspoof2019_LA_cm_protocols"


def rows(name):
    for line in (proto / name).read_text().splitlines():
        spk, utt, _, attack, key = line.split()
        yield spk, utt, attack, key


def link(dst_root, shard, spk, src):
    d = args.out / f"{dst_root}_{shard}" / spk
    d.mkdir(parents=True, exist_ok=True)
    dst = d / src.name
    if not dst.exists():
        os.symlink(src, dst)


# fit: train + dev bona fide, speakers round-robin over shards
fit = []
for split, name in (("train", "ASVspoof2019.LA.cm.train.trn.txt"), ("dev", "ASVspoof2019.LA.cm.dev.trl.txt")):
    fit += [(spk, args.la / f"ASVspoof2019_LA_{split}/flac/{utt}.flac") for spk, utt, _, key in rows(name)
            if key == "bonafide"]
speakers = sorted({s for s, _ in fit})
shard_of = {s: i % args.shards for i, s in enumerate(speakers)}
for spk, src in fit:
    assert src.exists(), src
    link("fit", shard_of[spk], spk, src)
print(f"fit: {len(fit)} bona fide clips, {len(speakers)} speakers, {args.shards} shards")

# eval: the same subset as scripts/eval_asvspoof2019_la.py
eval_rows = list(rows("ASVspoof2019.LA.cm.eval.trl.txt"))
speaker = {utt: spk for spk, utt, _, _ in eval_rows}
trials = eval_la.read_protocol(proto / "ASVspoof2019.LA.cm.eval.trl.txt")
subset = eval_la.select_subset(trials, 1000, 77, 0)
with open(args.out / "eval_labels.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "label", "attack"])
    for i, (utt, attack) in enumerate(subset):
        src = args.la / f"ASVspoof2019_LA_eval/flac/{utt}.flac"
        assert src.exists(), src
        link("eval", i % args.shards, speaker[utt], src)
        w.writerow([src.name, "bonafide" if attack == "bonafide" else "spoof", attack])
n_bona = sum(a == "bonafide" for _, a in subset)
print(f"eval: {len(subset)} trials ({n_bona} bona fide, {len(subset) - n_bona} spoof), "
      f"{len({speaker[u] for u, _ in subset})} speakers")
