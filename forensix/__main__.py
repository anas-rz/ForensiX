"""Score files from the command line: ``python -m forensix FILE [FILE ...]``.

Runs the server's default models for each file's media type (an image detector, a video
detector, voice and lip-sync models for video with sound) plus the provenance checks, and prints
one line per model. ``--models`` picks others; ``--json`` prints machine-readable results.
"""

import argparse
import json
import sys
from pathlib import Path

import forensix
from forensix.server.config import DEFAULT_MODELS
from forensix.server.media import probe


def score(path: Path, models: list[str] | None, device: str) -> dict:
    info = probe(path)
    chosen = models or DEFAULT_MODELS[info.modality]
    results = {}
    for name in chosen:
        try:
            pred = forensix.load(name, device=device).predict(str(path))
        except Exception as e:  # e.g. no face in a video, or a model that needs trained weights
            results[name] = {"error": f"{type(e).__name__}: {e}"}
            continue
        if pred.score_type == "provenance":
            results[name] = {"verdict": pred.details.get("verdict"), "summary": pred.details.get("summary")}
        else:
            results[name] = {"score": round(pred.fake_prob, 4), "threshold": pred.threshold, "label": pred.label}
    return {"file": str(path), "media": info.modality, "results": results}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m forensix", description="Score images, video or audio "
                                     "with ForensiX detectors and provenance checks.")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--models", help="comma-separated model names (default: by media type)")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or cuda:N")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = parser.parse_args(argv)
    models = [m.strip() for m in args.models.split(",")] if args.models else None
    reports = []
    for path in args.files:
        if not path.is_file():
            parser.error(f"{path} is not a file")
        report = score(path, models, args.device)
        reports.append(report)
        if args.json:
            continue
        print(f"\n{path.name} ({report['media']})")
        for name, r in report["results"].items():
            if "error" in r:
                print(f"  {name:<16} could not score: {r['error']}")
            elif "verdict" in r:
                print(f"  {name:<16} {r['verdict']:<22} {r['summary']}")
            else:
                threshold = "" if r["threshold"] is None else f"  (threshold {r['threshold']})"
                print(f"  {name:<16} {(r['label'] or '').upper():<6} {r['score']:.3f}{threshold}")
    if args.json:
        json.dump(reports if len(reports) > 1 else reports[0], sys.stdout, indent=2)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
