"""Download pretrained weights into the ForensiX cache ($FORENSIX_CACHE or ~/.cache/forensix).

    python scripts/download_weights.py              # all models
    python scripts/download_weights.py aasist       # selected models
    python scripts/download_weights.py --family audio
"""

import argparse

import forensix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="*", help="model names (default: all)")
    parser.add_argument("--family", help="only models of this family")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    names = args.models or forensix.list_models(args.family)
    for name in names:
        path = forensix.download(name, force=args.force)
        print(f"{name}: {path or 'no pretrained weights'}")


if __name__ == "__main__":
    main()
