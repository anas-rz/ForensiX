"""The ForensiX labelled-dataset format, used to calibrate and evaluate detectors on your own data.

A dataset is a folder, or a ZIP archive of one, in either layout:

1. **Manifest** (recommended): ``manifest.csv`` at the root, one row per media file::

       file,label,group,identity
       media/clip001.mp4,real,interview-7,
       media/clip001_swap.mp4,fake,interview-7,
       photos/a.jpg,real,,alice

   - ``file``: path relative to the dataset root (required).
   - ``label``: ``real`` or ``fake`` (also accepted: bonafide/bona-fide/genuine/
     authentic/0 and spoof/deepfake/synthetic/manipulated/1) (required).
   - ``group``: files that belong together (a real clip and fakes made from it,
     one recording session). Cross-validation keeps a group in one fold, so the
     report isn't flattered by near-duplicates. Optional; default: each file alone.
   - ``identity``: the enrolled person the media claims to show, for identity
     models. Optional.

2. **Folders**: no manifest; every file under ``real/`` is real and every file
   under ``fake/`` is fake (subfolders allowed). Each file is its own group.

Files other than images, video and audio are listed as problems, not errors.
"""

import csv
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

REAL_LABELS = {"real", "bonafide", "bona-fide", "bona fide", "genuine", "authentic", "0"}
FAKE_LABELS = {"fake", "spoof", "deepfake", "synthetic", "manipulated", "1"}
MEDIA_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v",
    ".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus",
}


@dataclass(frozen=True)
class Item:
    path: Path
    label: str  # "real" or "fake"
    group: str
    identity: str | None = None

    @property
    def is_fake(self) -> bool:
        return self.label == "fake"


@dataclass
class Dataset:
    root: Path
    items: list[Item]
    layout: str  # "manifest" or "folders"
    problems: list[str] = field(default_factory=list)

    @property
    def n_real(self) -> int:
        return sum(not i.is_fake for i in self.items)

    @property
    def n_fake(self) -> int:
        return sum(i.is_fake for i in self.items)

    def summary(self) -> dict:
        return {"layout": self.layout, "n_items": len(self.items), "n_real": self.n_real, "n_fake": self.n_fake,
                "n_groups": len({i.group for i in self.items}),
                "with_identity": sum(i.identity is not None for i in self.items),
                "problems": self.problems[:50], "n_problems": len(self.problems)}


def normalize_label(value: str) -> str | None:
    v = str(value).strip().lower()
    return "real" if v in REAL_LABELS else "fake" if v in FAKE_LABELS else None


def read_dataset(root: str | Path) -> Dataset:
    """Read a dataset folder (see the module docstring). Raises ValueError if it has
    no usable items; per-row issues are collected in ``Dataset.problems``."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"{root} is not a folder (extract ZIP archives with extract_dataset)")
    manifest = root / "manifest.csv"
    dataset = _read_manifest(root, manifest) if manifest.exists() else _read_folders(root)
    if not dataset.items:
        raise ValueError("The dataset has no usable items. " + "; ".join(dataset.problems[:5]))
    return dataset


def _read_manifest(root: Path, manifest: Path) -> Dataset:
    items, problems, seen = [], [], set()
    with open(manifest, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = {c.strip().lower() for c in reader.fieldnames or []}
        if not {"file", "label"} <= columns:
            raise ValueError(f"manifest.csv needs 'file' and 'label' columns, found {sorted(columns)}")
        for n, row in enumerate(reader, start=2):
            row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            rel, label = row.get("file", ""), normalize_label(row.get("label", ""))
            path = (root / rel).resolve()
            if not rel:
                problems.append(f"line {n}: empty file")
            elif label is None:
                problems.append(f"line {n}: unknown label {row.get('label')!r} for {rel}")
            elif not path.is_relative_to(root):
                problems.append(f"line {n}: {rel} is outside the dataset")
            elif not path.is_file():
                problems.append(f"line {n}: {rel} not found")
            elif path in seen:
                problems.append(f"line {n}: {rel} listed twice")
            else:
                seen.add(path)
                items.append(Item(path, label, row.get("group") or rel, row.get("identity") or None))
    return Dataset(root, items, "manifest", problems)


def _read_folders(root: Path) -> Dataset:
    items, problems = [], []
    for label in ("real", "fake"):
        folder = root / label
        if not folder.is_dir():
            problems.append(f"no manifest.csv and no {label}/ folder")
            continue
        for p in sorted(folder.rglob("*")):
            if not p.is_file() or p.name.startswith("."):
                continue
            if p.suffix.lower() not in MEDIA_SUFFIXES:
                problems.append(f"{p.relative_to(root)}: not an image, video or audio file")
                continue
            items.append(Item(p, label, str(p.relative_to(root))))
    return Dataset(root, items, "folders", problems)


def extract_dataset(archive: str | Path | io.IOBase, dest: str | Path, max_bytes: int = 50 * 2**30,
                    max_files: int = 200_000) -> Path:
    """Safely extract a dataset ZIP into ``dest`` and return the dataset root (the
    folder holding ``manifest.csv`` or ``real/`` and ``fake/``, which may be one
    level down). Rejects paths escaping ``dest`` and archives over the limits."""
    dest = Path(dest).resolve()
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if len(members) > max_files:
            raise ValueError(f"Archive has {len(members)} files; the limit is {max_files}")
        if sum(m.file_size for m in members) > max_bytes:
            raise ValueError(f"Archive expands to more than {max_bytes / 2**30:.1f} GB")
        for m in members:
            name = PurePosixPath(m.filename)
            target = (dest / name).resolve()
            if name.is_absolute() or ".." in name.parts or not target.is_relative_to(dest):
                raise ValueError(f"Unsafe path in archive: {m.filename}")
            if "__MACOSX" in name.parts:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(target, "wb") as out:
                written = 0
                while chunk := src.read(1 << 20):
                    written += len(chunk)
                    if written > m.file_size:  # a lying header: refuse rather than trust it
                        raise ValueError(f"{m.filename} is larger than its archive header says")
                    out.write(chunk)
    return find_root(dest)


def find_root(folder: Path) -> Path:
    """``folder``, or its only subfolder when the archive wrapped everything in one."""
    def is_root(p: Path) -> bool:
        return (p / "manifest.csv").exists() or ((p / "real").is_dir() and (p / "fake").is_dir())

    if is_root(folder):
        return folder
    subdirs = [p for p in folder.iterdir() if p.is_dir() and p.name != "__MACOSX"]
    if len(subdirs) == 1 and is_root(subdirs[0]):
        return subdirs[0]
    return folder
