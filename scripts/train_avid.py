"""Train avid (voice-face identity) on real talking-face videos. Meant for Colab.

1. extract: per clip, time-aligned voice (ECAPA-TDNN) and face (DINOv2) segment
   embeddings of the speaking face track. Resumable.

       python scripts/train_avid.py extract --media /content/voxceleb2/mp4 \\
           --out /content/avid_cache --max-files 20000 --device cuda

2. train: speakers are split three ways (by id): heads (contrastive voice-face
   space), scorer (calibration against swaps and splices), and a report split
   (AUC per negative kind, 1:2 voice-to-face matching accuracy).

       python scripts/train_avid.py train --cache /content/avid_cache --out avid.safetensors --device cuda

Use it:

    det = forensix.load("avid", weights_path="avid.safetensors")

Data: real videos only, many speakers (the heads learn from identity pairs, so
the number of speakers matters more than hours). VoxCeleb2 (CC BY 4.0; the
copyright of the videos stays with their owners): the test set has 118
speakers, which is enough to check the pipeline; train on (a part of) the dev
set, which has 5,994. A clip's group is its first folder under ``--media``
(the speaker id in VoxCeleb2).
"""

import argparse
import json
import random
import time
import zlib
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def split_of(group: str, fractions=(0.7, 0.15)) -> str:
    u = (zlib.crc32(group.encode()) % 10000) / 10000
    return "heads" if u < fractions[0] else "scorer" if u < fractions[0] + fractions[1] else "report"


def cmd_extract(args) -> None:
    from forensix.audiovisual.avid import Encoders, segment_features
    from forensix.audiovisual.avid.train import save_clip
    from forensix.audiovisual.io import read_av

    encoders = Encoders(args.device)
    files = sorted(p for p in args.media.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
    random.Random(args.seed).shuffle(files)
    files = files[: args.max_files] if args.max_files else files
    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    t0, done = time.time(), 0
    for p in files:
        rel = p.relative_to(args.media).as_posix()
        key = rel.replace("/", "__").rsplit(".", 1)[0] + ".npz"
        if key in index:
            continue
        try:
            frames, audio = read_av(p)
            n = min(len(frames), len(audio) // 640, int(args.max_seconds * 25))
            tracks = segment_features(frames[:n], audio[: n * 640] / 32768.0, encoders)
        except Exception as e:  # noqa: BLE001  (keep going over a large dataset)
            print(f"  skip {rel}: {type(e).__name__}: {e}")
            index[key] = "none"
            continue
        if not tracks:
            index[key] = "none"
            continue
        t = max(tracks, key=lambda t: t["n_frames"])  # the speaking track (one speaker per clip)
        save_clip(args.out / key, t["voice"], t["face"])
        index[key] = {"file": rel, "group": rel.split("/")[0] if "/" in rel else rel}
        done += 1
        if done % 50 == 0:
            index_path.write_text(json.dumps(index))
            print(f"  {done} clips, {(time.time() - t0) / done:.1f} s/clip", flush=True)
    index_path.write_text(json.dumps(index))
    print(f"extracted {done} clips -> {args.out}")


def cmd_train(args) -> None:
    from forensix.audiovisual.avid.train import evaluate, fit_scorer, load_clip, save, train_heads

    index = json.loads((args.cache / "index.json").read_text())
    splits = {"heads": [], "scorer": [], "report": []}
    for key, e in index.items():
        if isinstance(e, dict) and (args.cache / key).exists():
            splits[split_of(e["group"])].append(load_clip(args.cache / key, e["group"]))
    n_spk = {k: len({c.group for c in v}) for k, v in splits.items()}
    print(f"clips (speakers): heads {len(splits['heads'])} ({n_spk['heads']}), scorer {len(splits['scorer'])} "
          f"({n_spk['scorer']}), report {len(splits['report'])} ({n_spk['report']})")
    if min(n_spk.values()) < 5:
        raise SystemExit("Need at least 5 speakers in every split; add data")
    heads = train_heads(splits["heads"], steps=args.steps, batch=args.batch, lr=args.lr, device=args.device,
                        seed=args.seed, log=lambda m: print(m, flush=True))
    scorer = fit_scorer(heads, splits["scorer"], args.device, args.seed)
    report = evaluate(heads, scorer, splits["report"], args.device)
    print("\nreport split (unseen speakers)")
    print(f"  1:2 voice-to-face matching accuracy {report['matching_1_of_2']:.3f} (chance 0.5)")
    print(f"  real mean probability {report['real_mean_prob']:.3f}")
    for kind, r in report.items():
        if isinstance(r, dict):
            print(f"  {kind:13} AUC {r['auc']:.3f}  balanced acc {r['balanced_acc']:.3f}")
    heads.cpu(); scorer.cpu()
    save(args.out, heads, scorer, {"clips": {k: len(v) for k, v in splits.items()}, "speakers": n_spk,
                                   "steps": args.steps, "report": report})
    print(f"\ncheckpoint -> {args.out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract", help="cache voice and face segment embeddings of real videos")
    e.add_argument("--media", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--max-files", type=int)
    e.add_argument("--max-seconds", type=float, default=20.0)
    e.add_argument("--device", default="cpu")
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(func=cmd_extract)
    t = sub.add_parser("train", help="train heads, fit the scorer, report")
    t.add_argument("--cache", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--steps", type=int, default=5000)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--device", default="cpu")
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=cmd_train)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
