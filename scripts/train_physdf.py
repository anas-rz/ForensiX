"""Train the physdf graph prior on real talking-face videos (with sound). Meant for Colab.

1. extract: graph states (face, pulse, lighting, voice) per clip, cached as .npz
   (resumable; the slow part, CPU-bound: MediaPipe and the ICT lighting fit).

       python scripts/train_physdf.py extract --media /content/voxceleb2_test/mp4 \\
           --out /content/physdf_cache --max-files 3000 --workers 8

2. train: masked modelling on the cached real clips, calibration on held-out
   speakers, and the checkpoint (safetensors, with the feature normalization
   and the real residual distributions inside).

       python scripts/train_physdf.py train --cache /content/physdf_cache \\
           --out physdf_graph.safetensors --steps 20000 --device cuda

Use it:

    det = forensix.load("physdf", weights_path="physdf_graph.safetensors")
    pred = det.predict("clip.mp4")
    print(pred.details["report"]["markdown"])

Only real media: the prior must not learn any generator. Suggested data:
VoxCeleb2 test (CC BY 4.0; the copyright of the videos stays with their owners),
interviews with sound. Audio-only clips (e.g. LibriSpeech, CC BY 4.0) can be
added with ``--kind audio``: they train the voice part. The group of a clip
(held-out split unit) is its first folder under ``--media`` (the speaker).
"""

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


from forensix.physics.calibration import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, in_holdout


def _extract_one(job: tuple[str, str, str, float]) -> str:
    path, out, kind, max_seconds = job
    from forensix.audio.io import load_audio
    from forensix.physics.graph.features import build_sequence

    try:
        frames = fps = audio = None
        if kind == "video":
            from forensix.physics.io import read_video

            frames, fps = read_video(path, max_seconds=max_seconds)
        try:
            audio = load_audio(path, 16000)[: int(max_seconds * 16000)]
        except ValueError:
            if kind == "audio":
                raise
        seq = build_sequence(frames, fps, None, audio, 16000)
        if len(seq) < 50:
            return f"skip {path}: under 2 s"
        seq.save(out)
        return "ok"
    except Exception as e:  # noqa: BLE001  (keep going over a large dataset)
        return f"error {path}: {type(e).__name__}: {e}"


def cmd_extract(args) -> None:
    extensions = VIDEO_EXTENSIONS if args.kind == "video" else AUDIO_EXTENSIONS
    files = sorted(p for p in args.media.rglob("*") if p.suffix.lower() in extensions)
    random.Random(args.seed).shuffle(files)
    files = files[: args.max_files] if args.max_files else files
    args.out.mkdir(parents=True, exist_ok=True)
    jobs, index = [], {}
    for p in files:
        rel = p.relative_to(args.media).as_posix()
        name = rel.replace("/", "__").rsplit(".", 1)[0] + ".npz"
        index[name] = {"file": rel, "group": rel.split("/")[0] if "/" in rel else rel, "kind": args.kind}
        if not (args.out / name).exists():
            jobs.append((str(p), str(args.out / name), args.kind, args.max_seconds))
    index_path = args.out / "index.json"
    old = json.loads(index_path.read_text()) if index_path.exists() else {}
    index_path.write_text(json.dumps({**old, **index}, indent=0))
    print(f"{len(files)} files, {len(jobs)} to extract")
    t0, done = time.time(), 0
    with ProcessPoolExecutor(args.workers) as pool:
        for status in pool.map(_extract_one, jobs, chunksize=1):
            done += 1
            if status != "ok":
                print(" ", status)
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)}  {(time.time() - t0) / done:.1f} s/file", flush=True)
    print(f"extracted {done} clips in {time.time() - t0:.0f} s -> {args.out}")


def cmd_train(args) -> None:
    from forensix.physics.graph import train as G
    from forensix.physics.graph.features import GraphSequence

    index = json.loads((args.cache / "index.json").read_text())
    names = sorted(n for n in index if (args.cache / n).exists())
    fit_names = [n for n in names if not in_holdout(index[n]["group"], args.holdout)]
    held_names = [n for n in names if in_holdout(index[n]["group"], args.holdout)]
    print(f"{len(names)} clips from {len({index[n]['group'] for n in names})} groups: "
          f"training on {len(fit_names)}, calibrating on {len(held_names)}")
    if len(held_names) < 20:
        raise SystemExit("Need at least 20 held-out clips to calibrate; add data or raise --holdout")
    train_seqs = [GraphSequence.load(args.cache / n) for n in fit_names]
    held_seqs = [GraphSequence.load(args.cache / n) for n in held_names]
    model, norm, history = G.train(train_seqs, steps=args.steps, window_s=args.window_s, batch=args.batch,
                                   lr=args.lr, dim=args.dim, layers=args.layers, device=args.device,
                                   seed=args.seed, log=lambda m: print(m, flush=True))
    calibration = G.calibrate(model, norm, held_seqs, args.device)
    print("\nheld-out real residuals (clip summary mean ± std):")
    for cond, groups in calibration.items():
        print(f" {cond}")
        for g, c in groups.items():
            print(f"  {g:28} {c['clip_mean']:.3f} ± {c['clip_std']:.3f}  (n={c['n_clips']})")
    info = {"n_train": len(train_seqs), "n_calibration": len(held_seqs), "steps": args.steps,
            "sources": sorted({index[n]["file"].split("/")[0] for n in names})[:20], "loss_history": history}
    model.cpu()
    G.save(args.out, model, norm, calibration, info)
    print(f"\ncheckpoint -> {args.out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract", help="cache graph states of real clips")
    e.add_argument("--media", type=Path, required=True)
    e.add_argument("--kind", choices=["video", "audio"], default="video")
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--max-files", type=int)
    e.add_argument("--max-seconds", type=float, default=20.0)
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(func=cmd_extract)
    t = sub.add_parser("train", help="train and calibrate the graph prior")
    t.add_argument("--cache", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--window-s", type=float, default=4.0)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--dim", type=int, default=64)
    t.add_argument("--layers", type=int, default=6)
    t.add_argument("--holdout", type=float, default=0.1)
    t.add_argument("--device", default="cpu")
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=cmd_train)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
