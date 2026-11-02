"""Build a fairness audit set (real faces with demographic labels) from FairFace.

FairFace (Kärkkäinen & Joo, WACV 2021; https://github.com/joojs/fairface) is CC BY 4.0:
real face photos annotated with age band, gender and race. Download the validation
Parquet file from the Hugging Face mirror (pinned revision), then:

    curl -L -o fairface_val.parquet "https://huggingface.co/datasets/HuggingFaceM4/FairFace/resolve/54d573cdb8b5af490ba8da9da2799628f6e5c496/1.25/validation-00000-of-00001-09e3e67bb00ab4ec.parquet"
    python scripts/build_fairface_set.py --parquet fairface_val.parquet --out fairface_audit --per-cell 30
    (cd fairface_audit/.. && zip -qr fairface_audit.zip fairface_audit)   # upload in the web manager

The output is a ForensiX dataset (manifest.csv with label=real, age, gender and
attr_race as annotated by FairFace) sampled evenly over race x gender cells. Its images
are real, so a fairness report on it measures false alarms. FairFace has no skin-tone
labels: let the fairness job estimate them (ITA), or add your own annotations.
Needs pyarrow (pip install pyarrow).
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parquet", type=Path, required=True, help="FairFace validation Parquet file")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-cell", type=int, default=30, help="images per race x gender cell")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit("This script needs pyarrow: pip install pyarrow") from e

    table = pq.read_table(args.parquet)
    names = json.loads(pq.read_schema(args.parquet).metadata[b"huggingface"])["info"]["features"]
    label = {k: names[k]["names"] for k in ("age", "gender", "race")}
    cells: dict[tuple, list[int]] = defaultdict(list)
    for i, (race, gender) in enumerate(zip(table.column("race").to_pylist(), table.column("gender").to_pylist())):
        cells[(race, gender)].append(i)
    rng = random.Random(args.seed)
    chosen = sorted(i for idx in cells.values() for i in rng.sample(idx, min(args.per_cell, len(idx))))

    media = args.out / "media"
    media.mkdir(parents=True, exist_ok=True)
    rows = table.take(chosen).to_pylist()
    with open(args.out / "manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label", "group", "age", "gender", "attr_race"])
        for i, row in zip(chosen, rows):
            name = f"media/fairface_val_{i:05d}.jpg"
            (args.out / name).write_bytes(row["image"]["bytes"])  # original bytes, not re-encoded
            writer.writerow([name, "real", f"ff{i}", label["age"][row["age"]], label["gender"][row["gender"]].lower(),
                             label["race"][row["race"]]])
    (args.out / "ATTRIBUTION.md").write_text(
        "Images and labels: FairFace (Kärkkäinen & Joo, 2021), https://github.com/joojs/fairface, licensed CC BY 4.0.\n"
        f"Sampled from the validation split (padding 1.25), {args.per_cell} per race x gender cell, seed {args.seed}.\n"
        "Race labels are FairFace's annotations and are reported as such; they are not skin tone.\n")
    print(f"{len(chosen)} images -> {args.out}")


if __name__ == "__main__":
    main()
