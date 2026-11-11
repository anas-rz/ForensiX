"""Evaluate video and image detectors on a folder of real and fake videos.

Works with any dataset laid out as two folders of videos, e.g. SDFVD
(https://data.mendeley.com/datasets/bcmkfgct2s/1, CC BY 4.0, 53 real + 53
face-swapped clips):

    python scripts/eval_video_folder.py --real SDFVD/videos_real --fake SDFVD/videos_fake \\
        --dataset SDFVD --out results/sdfvd.json --scores-out results/sdfvd_scores.csv

Video detectors score each video with their own frame sampling and face
cropping. Image detectors score ``--frames`` evenly spaced full frames, and the
video score is the mean frame score. With ``--face-crop`` they score a square
crop around the largest face in each frame instead (MTCNN; 1.3x the face box,
at least 256 px, never resized, so each model's own preprocessing is unchanged). Results are written after every model, and
models already in ``--out`` are skipped, so an interrupted run can be resumed.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import forensix
from forensix.metrics import binary_summary, compute_eer
from forensix.video.io import read_frames

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def list_videos(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS)


def score_video_model(detector, items) -> list[float]:
    scores = []
    for k, (path, _) in enumerate(items, 1):
        try:
            scores.append(detector.predict(path).fake_prob)
        except Exception as e:  # undecodable media or no face found
            scores.append(np.nan)
            print(f"\n  skipped {path.name}: {e}")
        print(f"\r  {detector.name}: {k}/{len(items)}", end="", flush=True)
    print()
    return scores


def face_crop(frame: np.ndarray, face_detector, min_side: int = 256, scale: float = 1.3) -> np.ndarray | None:
    """Square crop centred on the largest face: ``scale`` times the box, at least
    ``min_side`` px, clipped to the frame and not resized. None if no face."""
    from PIL import Image

    boxes, _ = face_detector.detect(Image.fromarray(frame), landmarks=False)
    if boxes is None or len(boxes) == 0:
        return None
    x0, y0, x1, y1 = boxes[int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))]
    h, w = frame.shape[:2]
    side = int(min(max(scale * max(x1 - x0, y1 - y0), min_side), h, w))
    left = int(np.clip(round((x0 + x1 - side) / 2), 0, w - side))
    top = int(np.clip(round((y0 + y1 - side) / 2), 0, h - side))
    return frame[top : top + side, left : left + side]


def score_image_model(detector, items, num_frames: int, face_detector=None) -> list[float]:
    scores = []
    for k, (path, _) in enumerate(items, 1):
        try:
            frames, _ = read_frames(path, num_frames)
            if face_detector is not None:
                frames = [c for c in (face_crop(f, face_detector) for f in frames) if c is not None]
                if not frames:
                    raise ValueError("no face found")
            preds = detector.predict_batch(list(frames), batch_size=num_frames)
            scores.append(float(np.mean([p.fake_prob for p in preds])))
        except Exception as e:
            scores.append(np.nan)
            print(f"\n  skipped {path.name}: {e}")
        print(f"\r  {detector.name}: {k}/{len(items)}", end="", flush=True)
    print()
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--real", type=Path, required=True, help="folder of real videos")
    parser.add_argument("--fake", type=Path, required=True, help="folder of fake videos")
    parser.add_argument("--dataset", default="video-folder", help="name stored in the results")
    parser.add_argument("--models", nargs="+",
                        default=forensix.list_models("video") + forensix.list_models("image"))
    parser.add_argument("--frames", type=int, default=8, help="frames per video for image detectors")
    parser.add_argument("--face-crop", action="store_true",
                        help="image detectors score a crop around the largest face (MTCNN) instead of the full frame")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--weights", nargs="+", default=[], metavar="NAME=PATH",
                        help="checkpoints to use instead of the pretrained ones, e.g. dfdet-clip=dfdet-clip-ft.safetensors")
    parser.add_argument("--out", type=Path, required=True, help="results JSON (also used to resume)")
    parser.add_argument("--scores-out", type=Path, help="write per-video scores CSV here")
    args = parser.parse_args()

    items = [(p, False) for p in list_videos(args.real)] + [(p, True) for p in list_videos(args.fake)]
    labels = np.array([is_fake for _, is_fake in items])
    n_fake = int(labels.sum())
    print(f"{len(items)} videos ({n_fake} fake, {len(items) - n_fake} real)")

    image_input = "face crop" if args.face_crop else "full frame"
    if args.out.exists():
        results = json.loads(args.out.read_text())
        if results.get("image_input", "full frame") != image_input:
            parser.error(f"{args.out} holds {results.get('image_input', 'full frame')} results; use another --out")
    else:
        results = {"dataset": args.dataset, "n_fake": n_fake, "n_real": len(items) - n_fake,
                   "image_frames_per_video": args.frames, "image_input": image_input, "models": {}, "scores": {}}
    face_detector = None
    if args.face_crop:
        from forensix.video.mtcnn.mtcnn import MTCNN

        face_detector = MTCNN(device=args.device)
    weights = dict(w.split("=", 1) for w in args.weights)
    for name in args.models:
        if name in results["models"]:
            print(f"  {name}: already done, skipping")
            continue
        detector = forensix.load(name, device=args.device, weights_path=weights.get(name))
        family = forensix.get_spec(name).family
        start = time.perf_counter()
        if family == "image":
            scores = score_image_model(detector, items, args.frames, face_detector)
        else:
            scores = score_video_model(detector, items)
        scores = np.array(scores)
        ok = ~np.isnan(scores)
        summary = binary_summary(scores[ok & labels], scores[ok & ~labels], detector.threshold)
        summary["eer"] = round(compute_eer(scores[ok & labels], scores[ok & ~labels])[0], 4)
        summary.update(family=family, failed=int((~ok).sum()),
                       seconds=round(time.perf_counter() - start, 1))
        results["models"][name] = summary
        results["scores"][name] = [None if np.isnan(s) else float(s) for s in scores]  # full precision: rounding ties tiny scores
        print(f"  {name}: AUC {summary['auc']:.3f}  AP {summary['ap']:.3f}  EER {summary['eer']:.3f}  "
              f"bal.acc {summary['balanced_acc']:.3f}  (failed {summary['failed']})")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n")
        del detector

    if args.scores_out:
        args.scores_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.scores_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "label", *results["scores"]])
            for i, (path, is_fake) in enumerate(items):
                writer.writerow([path.name, "fake" if is_fake else "real",
                                 *("" if s[i] is None else s[i] for s in results["scores"].values())])


if __name__ == "__main__":
    main()
