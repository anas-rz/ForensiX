"""Uploaded media: hashing, finding out what it contains, and bounding its size."""

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

IMAGE_FORMATS = {"JPEG": ".jpg", "MPO": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp", "TIFF": ".tif"}
# extensions the library's readers recognize; others get a canonical one
KNOWN_SUFFIXES = {
    "image": {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"},
    "video": {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"},
    "audio": {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"},
}


@dataclass(frozen=True)
class MediaInfo:
    kind: str  # "image", "video" or "audio"
    has_audio: bool = False
    duration_s: float | None = None
    height: int | None = None
    image_format: str | None = None

    @property
    def modality(self) -> str:
        """Key of ``ServerConfig.default_models``."""
        return "video+audio" if self.kind == "video" and self.has_audio else self.kind

    def suffix(self, original: str) -> str:
        """``original`` if it fits the content, else a canonical extension. Some readers
        (identity, physdf, person models) pick their decoding path by extension."""
        original = original.lower()
        if original in KNOWN_SUFFIXES[self.kind]:
            return original
        if self.kind == "image":
            return IMAGE_FORMATS[self.image_format]
        return ".mp4" if self.kind == "video" else ".wav"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(path: Path) -> MediaInfo:
    """What ``path`` contains, from its content rather than its extension."""
    try:
        with Image.open(path) as img:
            if img.format in IMAGE_FORMATS and not getattr(img, "is_animated", False):
                img.verify()
                return MediaInfo("image", image_format=img.format)
    except Exception:
        pass
    if shutil.which("ffprobe"):
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "stream=codec_type,height,disposition:format=duration", "-of", "json", str(path)],
                capture_output=True, text=True, timeout=30, check=True).stdout
        except (subprocess.SubprocessError, OSError):
            out = ""
        if out:
            info = json.loads(out)
            # cover art in audio files shows up as a video stream with attached_pic
            video = [s for s in info.get("streams", []) if s.get("codec_type") == "video"
                     and not s.get("disposition", {}).get("attached_pic")]
            audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
            duration = float(info.get("format", {}).get("duration") or 0) or None
            if video:
                return MediaInfo("video", audio, duration, video[0].get("height"))
            if audio:
                return MediaInfo("audio", False, duration)
    try:  # no ffprobe: audio libsndfile can read
        import soundfile as sf

        meta = sf.info(str(path))
        return MediaInfo("audio", False, meta.duration)
    except Exception:
        pass
    raise ValueError("Unsupported or unreadable media")


def bound_video(path: Path, info: MediaInfo, max_seconds: float, max_height: int, out_dir: Path) -> tuple[Path, dict | None]:
    """Re-encode a video that's longer or larger than the limits; return it unchanged otherwise.
    Some models read every frame of a video into memory, so this caps memory per request."""
    too_long = info.duration_s is not None and info.duration_s > max_seconds
    too_big = info.height is not None and info.height > max_height
    if not (too_long or too_big):
        return path, None
    dst = out_dir / "bounded.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(path), "-t", str(max_seconds),
         "-vf", f"scale=-2:'min({max_height},ih)'", "-c:v", "libx264", "-crf", "18",
         "-c:a", "aac", "-b:a", "192k", str(dst)],
        check=True, timeout=600, capture_output=True)
    return dst, {"trimmed_to_s": max_seconds if too_long else None,
                 "scaled_to_height": max_height if too_big else None}


AUDIO_PHYSICS = {"phys-voice", "phys-room"}
AUDIO_IDENTITY = {"id-voice"}
VIDEO_PHYSICS = {"phys-bloodflow", "phys-motion", "phys-lighting"}


def applicable(name: str, family: str, info: MediaInfo) -> str | None:
    """None if ``name`` can score this media, else why not."""
    kind, sound = info.kind, info.kind == "audio" or info.has_audio
    if family == "image":
        ok = kind == "image"
    elif family == "video" or name in VIDEO_PHYSICS:
        ok = kind == "video"
    elif family == "audio" or name in AUDIO_PHYSICS or name in AUDIO_IDENTITY:
        ok = sound
    elif family == "audiovisual":
        ok = kind == "video" and info.has_audio
    elif name == "physdf":
        ok = kind != "image"
    elif name == "id-motion":
        ok = kind == "video"
    elif family == "identity":
        ok = kind in {"image", "video"}
    else:
        ok = True
    return None if ok else f"{name} ({family}) doesn't score {info.modality} media"
