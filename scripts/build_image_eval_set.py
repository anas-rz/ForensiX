"""Build ForensiX's license-clean real-vs-generated image evaluation set.

Real: photos from the Open Images V7 validation split whose listed license is
CC BY 2.0 (attribution for each one is written to ATTRIBUTION.csv).
Fake: images from generators whose licenses allow commercial use of outputs,
prompted with the Localized Narratives caption (CC BY 4.0) of a real photo in
the set and generated at that photo's aspect ratio, so content and shape match.

Run on a GPU (e.g. Google Colab). The script resumes where it stopped:

    pip install -e ".[eval]"
    huggingface-cli login        # FLUX.1-schnell needs its terms accepted once
    python scripts/build_image_eval_set.py --out /content/forensix_image_eval

Output:
    real/<id>.jpg                     original Open Images bytes (never re-encoded)
    fake_png/<generator>/<id>.png     generator output, lossless
    fake_jpeg/<generator>/<id>.jpg    same, JPEG at the reals' median quality
    manifest.csv, ATTRIBUTION.csv, info.json
"""

import argparse
import csv
import io
import json
import math
import random
import re
import statistics
import time
import urllib.request
from pathlib import Path

from PIL import Image

OPEN_IMAGES_CSV = (
    "https://storage.googleapis.com/openimages/2018_04/validation/"
    "validation-images-with-rotation.csv"
)
NARRATIVES_JSONL = (
    "https://storage.googleapis.com/localized-narratives/annotations/"
    "open_images_validation_localized_narratives.jsonl"
)
IMAGE_URL = "https://s3.amazonaws.com/open-images-dataset/validation/{}.jpg"
CC_BY_2 = "https://creativecommons.org/licenses/by/2.0/"
_IMAGE_ID = re.compile(rb'"image_id":\s*"([0-9a-f]{16})"')

# Generators whose licenses allow commercial use of generated images.
GENERATORS = {
    "sdxl": dict(
        repo="stabilityai/stable-diffusion-xl-base-1.0",
        revision="462165984030d82259a11f4367a4eed129e94a7b",
        license="CreativeML OpenRAIL++-M",
        area=1024 * 1024,
        kwargs=dict(num_inference_steps=30, guidance_scale=5.0),
    ),
    "pixart-sigma": dict(
        repo="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        revision="e102b3591cc82e97071b8b4cb90d834d0c487207",
        license="CreativeML OpenRAIL++-M",
        area=1024 * 1024,
        kwargs=dict(num_inference_steps=20, guidance_scale=4.5),
    ),
    "kandinsky-2.2": dict(
        repo="kandinsky-community/kandinsky-2-2-decoder",
        revision="9ae140d347fed8ce6e8bb3005dcc1f48543bb8e3",
        license="Apache-2.0",
        area=768 * 768,
        kwargs=dict(num_inference_steps=50, guidance_scale=4.0),
    ),
    "flux-schnell": dict(
        repo="black-forest-labs/FLUX.1-schnell",
        revision="741f7c3ce8b383c54771c7003378a50191e9efe9",
        license="Apache-2.0",
        area=1024 * 1024,
        kwargs=dict(num_inference_steps=4, guidance_scale=0.0, max_sequence_length=256),
    ),
}


def _open(url: str, retries: int = 3):
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(url, timeout=60)
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)


def read_cc_by_images() -> dict[str, dict]:
    """Open Images validation metadata for images listed as CC BY 2.0."""
    text = _open(OPEN_IMAGES_CSV).read().decode("utf-8")
    return {
        row["ImageID"]: row for row in csv.DictReader(io.StringIO(text)) if row["License"] == CC_BY_2
    }


def read_captions(wanted: set[str]) -> dict[str, str]:
    """First Localized Narratives caption for each wanted image (streams ~1.1 GB)."""
    captions = {}
    with _open(NARRATIVES_JSONL) as f:
        for line in f:
            # cheap prefilter before parsing the large per-line JSON
            m = _IMAGE_ID.search(line)
            if not m or m[1].decode() not in wanted or m[1].decode() in captions:
                continue
            record = json.loads(line)
            if record["image_id"] in wanted:
                captions.setdefault(record["image_id"], " ".join(record["caption"].split()))
    return captions


def generation_size(width: int, height: int, area: int) -> tuple[int, int]:
    """Size with the real photo's aspect ratio (clamped to 1:2..2:1), about
    ``area`` pixels, both sides multiples of 64."""
    aspect = min(max(width / height, 0.5), 2.0)
    w = math.sqrt(area * aspect)
    h = area / w
    return max(64, round(w / 64) * 64), max(64, round(h / 64) * 64)


def _jpeg_quality_tables() -> dict[int, tuple]:
    tables = {}
    for q in range(1, 101):
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, format="JPEG", quality=q)
        tables[q] = tuple(Image.open(buf).quantization[0])
    return tables


def estimate_jpeg_quality(img: Image.Image, tables: dict[int, tuple]) -> int | None:
    """IJG quality whose luminance table is closest to the image's."""
    if not getattr(img, "quantization", None):
        return None
    table = img.quantization[0]
    return min(tables, key=lambda q: sum(abs(a - b) for a, b in zip(tables[q], table)))


def select_reals(out: Path, n_real: int, seed: int) -> list[dict]:
    """Sample and download real photos; cached in real/selection.json."""
    cache = out / "real" / "selection.json"
    if cache.exists():
        return json.loads(cache.read_text())
    (out / "real").mkdir(parents=True, exist_ok=True)
    print("Reading Open Images metadata ...")
    meta = read_cc_by_images()
    print(f"  {len(meta)} CC BY 2.0 validation images. Reading captions ...")
    captions = read_captions(set(meta))
    candidates = sorted(set(meta) & set(captions))
    random.Random(seed).shuffle(candidates)

    selected = []
    for image_id in candidates:
        if len(selected) == n_real:
            break
        path = out / "real" / f"{image_id}.jpg"
        try:
            if not path.exists():
                path.write_bytes(_open(IMAGE_URL.format(image_id)).read())
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                width, height = img.size
        except Exception as e:  # unavailable or corrupt: take the next candidate
            print(f"  skip {image_id}: {e}")
            path.unlink(missing_ok=True)
            continue
        m = meta[image_id]
        selected.append(
            dict(
                image_id=image_id,
                caption=captions[image_id],
                width=width,
                height=height,
                author=m["Author"],
                author_url=m["AuthorProfileURL"],
                title=m["Title"],
                landing_url=m["OriginalLandingURL"],
                license=m["License"],
            )
        )
        print(f"\r  downloaded {len(selected)}/{n_real}", end="", flush=True)
    print()
    cache.write_text(json.dumps(selected, indent=1))
    return selected


def load_pipeline(name: str):
    import torch
    from diffusers import AutoPipelineForText2Image

    g = GENERATORS[name]
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if name == "flux-schnell" and dtype == torch.float16:
        print("  warning: this GPU has no bfloat16; FLUX in float16 may produce black images")
    pipe = AutoPipelineForText2Image.from_pretrained(
        g["repo"], revision=g["revision"], torch_dtype=dtype
    )
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    if total_gb < 24:
        pipe.enable_model_cpu_offload()  # T5 text encoders don't fit alongside on 16 GB
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate(out: Path, name: str, jobs: list[dict]) -> None:
    import torch

    g = GENERATORS[name]
    folder = out / "fake_png" / name
    folder.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (folder / f"{j['image_id']}.png").exists()]
    print(f"{name}: {len(jobs) - len(todo)}/{len(jobs)} done")
    if not todo:
        return
    pipe = load_pipeline(name)
    for k, job in enumerate(todo, 1):
        width, height = generation_size(job["width"], job["height"], g["area"])
        seed = int(job["image_id"][:8], 16)
        for attempt in range(3):
            generator = torch.Generator("cpu").manual_seed(seed + attempt)
            image = pipe(
                prompt=job["caption"], width=width, height=height, generator=generator, **g["kwargs"]
            ).images[0]
            # a flat image means the pipeline produced NaNs / was filtered: retry
            if max(hi - lo for lo, hi in image.getextrema()) > 10:
                break
        image.save(folder / f"{job['image_id']}.png")
        print(f"\r  {name}: {len(jobs) - len(todo) + k}/{len(jobs)}", end="", flush=True)
    print()
    del pipe
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-real", type=int, default=1000)
    parser.add_argument("--n-per-generator", type=int, default=250)
    parser.add_argument("--generators", nargs="*", default=list(GENERATORS), choices=list(GENERATORS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    reals = select_reals(args.out, args.n_real, args.seed)
    # prompts: real photo i goes to generator i mod G, so each caption is used once
    names = list(GENERATORS)
    jobs = {n: [r for i, r in enumerate(reals) if i % len(names) == k][: args.n_per_generator]
            for k, n in enumerate(names)}
    for name in args.generators:
        generate(args.out, name, jobs[name])

    # JPEG copies of fakes at the reals' median quality (removes the PNG-vs-JPEG shortcut)
    tables = _jpeg_quality_tables()
    qualities = []
    for r in reals:
        with Image.open(args.out / "real" / f"{r['image_id']}.jpg") as img:
            q = estimate_jpeg_quality(img, tables)
        if q is not None:
            qualities.append(q)
    jpeg_quality = int(statistics.median(qualities))

    rows = [dict(path=f"real/{r['image_id']}.jpg", label="real", condition="real",
                 generator="", image_id=r["image_id"], prompt="") for r in reals]
    for name in names:
        for job in jobs[name]:
            png = args.out / "fake_png" / name / f"{job['image_id']}.png"
            if not png.exists():
                continue
            jpg = args.out / "fake_jpeg" / name / f"{job['image_id']}.jpg"
            if not jpg.exists():
                jpg.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(png) as img:
                    img.convert("RGB").save(jpg, format="JPEG", quality=jpeg_quality)
            for condition, path in (("png", png), ("jpeg", jpg)):
                rows.append(dict(path=str(path.relative_to(args.out)), label="fake",
                                 condition=condition, generator=name,
                                 image_id=job["image_id"], prompt=job["caption"]))

    with open(args.out / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(args.out / "ATTRIBUTION.csv", "w", newline="") as f:
        fields = ["image_id", "title", "author", "author_url", "landing_url", "license"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reals)
    counts = {n: sum(1 for r in rows if r["generator"] == n and r["condition"] == "png") for n in names}
    info = dict(
        real=dict(source="Open Images V7 validation", license=CC_BY_2, count=len(reals),
                  median_jpeg_quality=jpeg_quality),
        captions=dict(source="Localized Narratives (Open Images validation)", license="CC BY 4.0"),
        generators={n: dict(GENERATORS[n], count=counts[n]) for n in names},
        seed=args.seed,
    )
    (args.out / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"Done: {len(reals)} real, {counts} fake; fake JPEG quality {jpeg_quality}")


if __name__ == "__main__":
    main()
