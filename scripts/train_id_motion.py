"""Train id-motion (behavioural identity) on real talking-face videos. Meant for Colab.

1. extract: MediaPipe motion features (head pose, 52 blendshapes) per clip at
   25 fps, cached. CPU-bound; parallel; resumable.

       python scripts/train_id_motion.py extract --media /content/voxceleb2/mp4 \\
           --out /content/idmotion_cache --max-files 30000 --workers 8

2. train: the temporal ID network (supervised contrastive, positives from
   different videos of the same person), then verification on held-out people
   (EER; its threshold becomes the model's decision threshold).

       python scripts/train_id_motion.py train --cache /content/idmotion_cache \\
           --out id_motion.safetensors --steps 20000 --device cuda

Use it:

    det = forensix.load("id-motion", weights_path="id_motion.safetensors")
    det.enroll("alice", ["alice_interview1.mp4", "alice_interview2.mp4"])
    det.predict("new.mp4", identity="alice")

Data: real videos, many people, several videos per person. VoxCeleb2 (CC BY
4.0; the copyright of the videos stays with their owners) is laid out as
<person>/<video>/<clip>.mp4; the dev set has 5,994 people. Aim for 1,000+ people
with 3+ videos each. The person and video come from the first two folders
under --media.
"""

import argparse
import json
import random
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _extract_one(job):
    path, out, max_seconds = job
    try:
        from forensix.identity.motion.features import motion_features
        from forensix.identity.motion.train import save_clip
        from forensix.physics.face.tracker import track_face
        from forensix.physics.io import read_video

        frames, fps = read_video(path, max_seconds=max_seconds)
        track = track_face(frames, fps)
        if track.valid.mean() < 0.8:
            return "skip: face tracked in under 80% of frames"
        feats, valid = motion_features(track)
        if len(feats) < 110:
            return "skip: under 4.4 s"
        save_clip(out, feats, valid)
        return "ok"
    except Exception as e:  # noqa: BLE001  (keep going over a large dataset)
        return f"error: {type(e).__name__}: {e}"


def cmd_extract(args) -> None:
    files = sorted(p for p in args.media.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
    random.Random(args.seed).shuffle(files)
    files = files[: args.max_files] if args.max_files else files
    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    jobs, keys = [], []
    for p in files:
        rel = p.relative_to(args.media)
        key = rel.as_posix().replace("/", "__").rsplit(".", 1)[0] + ".npz"
        if key in index:
            continue
        parts = rel.parts
        index[key] = {"file": rel.as_posix(), "person": parts[0], "video": "/".join(parts[:2]) if len(parts) > 2 else parts[0]}
        jobs.append((str(p), str(args.out / key), args.max_seconds)); keys.append(key)
    print(f"{len(files)} files, {len(jobs)} to extract")
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (key, status) in enumerate(zip(keys, pool.map(_extract_one, jobs, chunksize=4)), 1):
            if status != "ok":
                index[key]["status"] = status
            if i % 200 == 0:
                index_path.write_text(json.dumps(index))
                print(f"  {i}/{len(jobs)}  {(time.time() - t0) / i:.2f} s/clip", flush=True)
    index_path.write_text(json.dumps(index))
    print(f"done in {time.time() - t0:.0f} s -> {args.out}")


def cmd_train(args) -> None:
    from forensix.identity.motion.train import evaluate, load_clip, save, train

    index = json.loads((args.cache / "index.json").read_text())
    ok = {k: v for k, v in index.items() if "status" not in v and (args.cache / k).exists()}
    held = lambda person: (zlib.crc32(person.encode()) % 10000) / 10000 < args.holdout  # noqa: E731
    fit = [load_clip(args.cache / k, v["person"], v["video"]) for k, v in ok.items() if not held(v["person"])]
    test = [load_clip(args.cache / k, v["person"], v["video"]) for k, v in ok.items() if held(v["person"])]
    print(f"train: {len(fit)} clips, {len({c.person for c in fit})} people; "
          f"held out: {len(test)} clips, {len({c.person for c in test})} people")
    enc = train(fit, steps=args.steps, people_per_batch=args.people_per_batch, lr=args.lr, device=args.device,
                seed=args.seed, log=lambda m: print(m, flush=True))
    report = evaluate(enc, test, device=args.device)
    print(f"\nheld-out verification: EER {report['eer']:.3f}  AUC {report['auc']:.3f}  "
          f"(genuine sim {report['genuine_mean']:.3f}, impostor sim {report['impostor_mean']:.3f}, "
          f"{report['n_people']} people); threshold {report['threshold']:.3f}")
    enc.cpu()
    save(args.out, enc, report["threshold"], {"n_train_clips": len(fit), "n_people": len({c.person for c in fit}),
                                               "steps": args.steps, "heldout": report})
    print(f"checkpoint -> {args.out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract", help="cache motion features of real videos")
    e.add_argument("--media", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--max-files", type=int)
    e.add_argument("--max-seconds", type=float, default=20.0)
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(func=cmd_extract)
    t = sub.add_parser("train", help="train and evaluate")
    t.add_argument("--cache", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--people-per-batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--holdout", type=float, default=0.1)
    t.add_argument("--device", default="cpu")
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=cmd_train)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
