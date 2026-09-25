"""Train a teeny tiny model specialized for one individual person on CPU.

Examples:
    # 1. Train an image detector for a person using their real photos:
    python scripts/train_person_model.py --person "armstrong" \\
        --real tests/data/face_public_domain.jpg \\
        --fake tests/data/other_face_public_domain.jpg \\
        --modality image --out armstrong_detector.safetensors

    # 2. Train an audio detector for a person from voice recordings:
    python scripts/train_person_model.py --person "alice" \\
        --real voice_sample.wav --modality audio --out alice_audio.safetensors

    # 3. Test the trained detector on a new file:
    python scripts/train_person_model.py --load armstrong_detector.safetensors \\
        --test tests/data/face2_public_domain.jpg
"""

import argparse
import sys
from pathlib import Path

import forensix
from forensix.specialized import PersonDetector, train_person_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a teeny tiny deepfake detector specialized for one person.")
    parser.add_argument("--person", type=str, help="Name or identifier of the individual to specialize for.")
    parser.add_argument("--real", nargs="+", help="Paths to genuine media files (photos, audio, or videos).")
    parser.add_argument("--fake", "--negative", nargs="+", default=None, help="Optional paths to negative/impostor media.")
    parser.add_argument(
        "--modality",
        type=str,
        default="auto",
        choices=["auto", "image", "audio", "video", "audiovisual"],
        help="Media modality (default: auto).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output .safetensors path to save the trained model.")
    parser.add_argument("--no-peft", dest="peft", action="store_false", default=True, help="Disable PEFT / LoRA.")
    parser.add_argument("--lora-rank", type=int, default=4, help="LoRA rank for PEFT.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs on CPU.")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda).")
    parser.add_argument("--load", type=Path, default=None, help="Load an existing detector checkpoint.")
    parser.add_argument("--test", type=str, default=None, help="Test media path to predict on after training/loading.")

    args = parser.parse_args()

    if args.load is not None:
        print(f"Loading detector from {args.load}...")
        detector = PersonDetector.load(args.load, device=args.device)
    else:
        if not args.person or not args.real:
            parser.error("--person and --real are required when training a new model.")

        print(f"Training specialized model for {args.person!r} on {args.device}...")
        detector = train_person_model(
            person_name=args.person,
            real_media=args.real,
            negative_media=args.fake,
            modality=args.modality,
            use_peft=args.peft,
            lora_rank=args.lora_rank,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            verbose=True,
        )

        if args.out is not None:
            saved_p = detector.save(args.out)
            print(f"Model saved to {saved_p}")

    if args.test is not None:
        print(f"\nEvaluating on {args.test}...")
        pred = detector.predict(args.test)
        print(f"Result: {pred.label.upper()} (fake_prob: {pred.fake_prob:.4f}, threshold: {pred.threshold:.4f})")
        print(f"Evidence details: {pred.details}")


if __name__ == "__main__":
    main()
