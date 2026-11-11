"""Download a small test kit of real and fake media to try ForensiX on (about 10 MB).

    python scripts/fetch_samples.py                  # into ./forensix-samples
    python -m forensix forensix-samples/images/fake/dalle3_angry_man.jpg

Every file is public domain, CC0 or MIT/Apache test material, so it's fine for commercial
evaluation. Sources and licenses are written to SOURCES.md next to the files.
"""

import argparse
import time
import urllib.error
import urllib.request
from pathlib import Path

USER_AGENT = "ForensiX-samples/1.0 (https://github.com/anas-rz/ForensiX)"  # Wikimedia asks for one
COMMONS = "https://upload.wikimedia.org/wikipedia/commons/"
C2PA = "https://raw.githubusercontent.com/contentauth/c2pa-rs/main/sdk/tests/fixtures/"
SAMPLES = [  # destination, URL, what it is, license
    ("images/fake/dalle3_angry_man.jpg", COMMONS + "8/8b/An_angry_man_with_an_empty_plate%2C_hungry.jpg",
     "AI-generated (DALL-E 3)", "Public domain"),
    ("images/fake/dalle3_astronaut.jpg", COMMONS + "f/f3/Astronaut_walking_on_the_surface_of_Mercury.jpg",
     "AI-generated (DALL-E 3)", "Public domain"),
    ("images/fake/sd_burned_city.jpg", COMMONS + "9/9e/Burned_city.jpg", "AI-generated (Stable Diffusion)", "CC0"),
    ("images/real/nasa_crew.jpg", COMMONS + "e/e4/Expedition_58_crew_portrait.jpg", "Real photo (NASA)",
     "Public domain"),
    ("images/real/nasa_aldrin.jpg", COMMONS + "a/a7/Buzz_Aldrin_%28S69-31743%29.jpg", "Real photo (NASA, 1969)",
     "Public domain"),
    ("audio/real/armstrong_small_step.ogg", COMMONS + "d/dd/Armstrong_Small_Step.ogg",
     "Real speech (NASA, 1969); old, band-limited audio that many voice detectors find hard", "Public domain"),
    ("provenance/c2pa_valid.jpg", C2PA + "CA.jpg", "C2PA Content Credentials (test certificate)",
     "MIT OR Apache-2.0"),
    ("provenance/c2pa_ai_declared.jpg", C2PA + "C.jpg", "C2PA credentials declaring AI processing",
     "MIT OR Apache-2.0"),
    ("provenance/c2pa_tampered.jpg", C2PA + "XCA.jpg", "C2PA credentials, image edited after signing",
     "MIT OR Apache-2.0"),
]


def fetch(url: str, dest: Path) -> None:
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT})) as r:
                dest.write_bytes(r.read())
            return
        except urllib.error.HTTPError as e:   # Wikimedia rate-limits bursts with 429
            if e.code != 429 or attempt == 5:
                raise
            time.sleep(int(e.headers.get("Retry-After") or 0) or 5 * 2 ** attempt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("forensix-samples"))
    args = parser.parse_args()
    rows = ["# ForensiX test kit", "", "| File | What it is | License | Source |", "|---|---|---|---|"]
    for rel, url, what, license_ in SAMPLES:
        dest = args.out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            fetch(url, dest)
            time.sleep(1)
        rows.append(f"| `{rel}` | {what} | {license_} | {url} |")
        print(f"  {rel}")
    (args.out / "SOURCES.md").write_text("\n".join(rows) + "\n")
    print(f"\n{len(SAMPLES)} files in {args.out}/. Try:\n  python -m forensix {args.out}/images/fake/dalle3_angry_man.jpg "
          f"{args.out}/provenance/c2pa_tampered.jpg")


if __name__ == "__main__":
    main()
