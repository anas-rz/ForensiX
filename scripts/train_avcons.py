"""Train avcons (audio-visual consistency) on real talking-face videos. Meant for Colab.

1. extract: SyncNet lip and audio embeddings of the speaking face track (plus the
   audio at 0.95x and 1.05x speed, for drift negatives), re-aligned to the clip's
   own sync; clips with unclear sync (SyncNet confidence under 3, e.g. off-screen
   speech) are dropped. Cached per clip; resumable.

       python scripts/train_avcons.py extract --media /content/voxceleb2_test/mp4 \\
           --out /content/avcons_cache --max-files 5000 --device cuda

2. train: the transformer on examples made from the cached real clips, then
   AUC per manipulation on held-out speakers, and the checkpoint.

       python scripts/train_avcons.py train --cache /content/avcons_cache \\
           --out avcons.safetensors --steps 20000 --device cuda

Use it:

    det = forensix.load("avcons", weights_path="avcons.safetensors")
    det.predict("clip.mp4")

Only real videos: VoxCeleb2 test (CC BY 4.0; the copyright of the videos stays
with their owners) is a good start. VoxCeleb2 dev is larger. A clip's group
(the unit of the held-out split) is its first folder under ``--media`` (the
speaker id in VoxCeleb2).
"""

import argparse
import json
import random
import time
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def cmd_extract(args) -> None:
    import forensix
    from forensix.audiovisual.avcons import SPEEDS, align_to_sync, track_embeddings
    from forensix.audiovisual.avcons.train import save_clip
    from forensix.audiovisual.io import read_av
    from forensix.audiovisual.syncnet import AUD_FACT

    syncnet = forensix.load("syncnet", device=args.device)
    files = sorted(p for p in args.media.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
    random.Random(args.seed).shuffle(files)
    files = files[: args.max_files] if args.max_files else files
    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    t0, done = time.time(), 0
    for p in files:
        rel = p.relative_to(args.media).as_posix()
        name = rel.replace("/", "__").rsplit(".", 1)[0]
        if any(k.startswith(name + "__t") for k in index) or index.get(name) == "none":
            continue
        try:
            frames, audio = read_av(p)
            n = min(len(frames), len(audio) // AUD_FACT, int(args.max_seconds * 25))
            tracks = track_embeddings(frames[:n], audio[: n * AUD_FACT], syncnet, SPEEDS)
        except Exception as e:  # noqa: BLE001  (keep going over a large dataset)
            print(f"  skip {rel}: {type(e).__name__}: {e}")
            index[name] = "none"
            continue
        # the speaking track: the longest (VoxCeleb2 clips hold one speaker), re-aligned
        # to its own sync; clips with unclear sync are dropped (as LatentSync does)
        tracks = sorted(tracks, key=lambda t: -len(t["v"]))[:1]
        aligned = align_to_sync(tracks[0]["v"], tracks[0]["a"], tracks[0]["a_speed"]) if tracks else None
        if aligned is None or len(aligned[0]) < 50:
            index[name] = "none"
            continue
        v, a, a_speed, offset, conf = aligned
        key = f"{name}__t0.npz"
        save_clip(args.out / key, v, a, a_speed)
        index[key] = {"file": rel, "group": rel.split("/")[0] if "/" in rel else rel,
                      "offset_frames": offset, "sync_confidence": round(conf, 3)}
        done += 1
        if done % 25 == 0:
            index_path.write_text(json.dumps(index))
            print(f"  {done} clips, {(time.time() - t0) / done:.1f} s/clip", flush=True)
    index_path.write_text(json.dumps(index))
    print(f"extracted {done} clips -> {args.out}")


def cmd_train(args) -> None:
    from forensix.audiovisual.avcons.train import evaluate, load_clip, save, train
    from forensix.physics.calibration import in_holdout

    index = json.loads((args.cache / "index.json").read_text())
    entries = {k: v for k, v in index.items() if isinstance(v, dict) and (args.cache / k).exists()}
    fit = [load_clip(args.cache / k, v["group"]) for k, v in entries.items() if not in_holdout(v["group"], args.holdout)]
    held = [load_clip(args.cache / k, v["group"]) for k, v in entries.items() if in_holdout(v["group"], args.holdout)]
    print(f"{len(entries)} clips: training on {len(fit)}, evaluating on {len(held)} (held-out speakers)")
    if len(held) < 10:
        raise SystemExit("Need at least 10 held-out clips; add data or raise --holdout")
    model, history = train(fit, steps=args.steps, batch=args.batch, window_s=args.window_s, lr=args.lr,
                           device=args.device, seed=args.seed, dim=args.dim, layers=args.layers,
                           log=lambda m: print(m, flush=True))
    results = evaluate(model, held, window_s=args.window_s, device=args.device)
    print("\nheld-out speakers: real vs each manipulation")
    print(f"  real mean probability {results['real_mean_prob']:.3f}")
    for kind, r in results.items():
        if isinstance(r, dict):
            extra = f"  window AUC {r['window_auc']:.3f}" if "window_auc" in r else ""
            print(f"  {kind:14} AUC {r['auc']:.3f}  balanced acc {r['balanced_acc']:.3f}{extra}")
    model.cpu()
    save(args.out, model, {"n_train": len(fit), "n_eval": len(held), "steps": args.steps,
                           "heldout": results, "loss_history": history})
    print(f"\ncheckpoint -> {args.out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract", help="cache SyncNet embeddings of real videos")
    e.add_argument("--media", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--max-files", type=int)
    e.add_argument("--max-seconds", type=float, default=20.0)
    e.add_argument("--device", default="cpu")
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(func=cmd_extract)
    t = sub.add_parser("train", help="train and evaluate")
    t.add_argument("--cache", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--window-s", type=float, default=5.0)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--dim", type=int, default=256)
    t.add_argument("--layers", type=int, default=4)
    t.add_argument("--holdout", type=float, default=0.1)
    t.add_argument("--device", default="cpu")
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=cmd_train)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
