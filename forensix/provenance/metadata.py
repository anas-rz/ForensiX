"""Metadata and re-encoding evidence: what a file says about its own history.

Images: EXIF, XMP and PNG text chunks. Generator parameters (Automatic1111's
``parameters``, ComfyUI's ``prompt``/``workflow``, NovelAI, InvokeAI), IPTC digital
source types that declare AI, generator names in software fields, editing software,
an EXIF thumbnail that no longer matches the image, camera metadata, JPEG quality.
Video and audio: container and stream tags read by ffprobe: generator names,
FFmpeg re-encoding, camera/phone tags, audio and video tracks of different lengths,
variable frame rate, speech-synthesis-like audio formats.

Metadata is easy to strip and easy to forge, so it can't prove authenticity (this
check never says ``real``). Generator traces are strong evidence when present.
Verdicts: ``ai_generated`` (label ``fake``), ``edited``, ``re_encoded``,
``capture_metadata``, ``no_metadata``, ``no_findings``.
"""

import io
import re
from pathlib import Path

import numpy as np

from forensix.provenance.evidence import Finding, ffprobe, media_kind, provenance_prediction
from forensix.registry import ModelSpec, register

# names of generative tools, matched as whole words in software/creator/encoder-like fields
GENERATORS = [
    "stable diffusion", "stablediffusion", "sdxl", "automatic1111", "comfyui", "invokeai", "novelai", "fooocus",
    "midjourney", "dall-e", "dall·e", "dalle", "openai", "adobe firefly", "firefly", "imagen", "google gemini",
    "leonardo.ai", "ideogram", "flux.1", "black forest labs", "runway", "runwayml", "pika", "kling", "sora", "veo",
    "luma", "dream machine", "hailuo", "minimax", "dreamstudio", "nightcafe", "craiyon", "bing image creator",
    "generative fill", "elevenlabs", "play.ht", "resemble.ai", "murf", "speechify", "suno", "udio", "heygen",
    "synthesia", "d-id", "deepfacelab", "faceswap", "roop", "facefusion",
]
GENERATOR_RE = re.compile(r"(?<![\w.])(" + "|".join(re.escape(g) for g in GENERATORS) + r")(?![\w])", re.I)
# "generated with Sora", "created by Midjourney": a statement of origin rather than a mention
MADE_WITH_RE = re.compile(r"\b(generated|created|made|produced|rendered|synthesi[sz]ed)\s+(with|by|using|in|on)\s+"
                          r"(?:[\w.\-]+\s+){0,3}?(" + "|".join(re.escape(g) for g in GENERATORS) + r")(?![\w])", re.I)
AI_SOURCE_RE = re.compile(r"(trainedAlgorithmicMedia|compositeWithTrainedAlgorithmicMedia|compositeSynthetic)", re.I)
EDITORS_RE = re.compile(r"(photoshop|lightroom|gimp|snapseed|picsart|facetune|canva|affinity|pixelmator|"
                        r"capture one|luminar|after effects|premiere|final cut|davinci resolve|capcut|imovie|"
                        r"inshot|kinemaster|audacity|adobe audition)", re.I)
# fields where a tool name means "made or processed with"
TOOL_KEYS = re.compile(r"(software|creator|creatortool|encoder|encoded_by|handler_name|producer|tool|"
                       r"generator|application|make|model|vendor)", re.I)
SD_BUCKETS = {(512, 512), (768, 768), (1024, 1024), (512, 768), (768, 512), (832, 1216), (1216, 832),
              (896, 1152), (1152, 896), (768, 1344), (1344, 768), (640, 1536), (1536, 640), (1024, 1792),
              (1792, 1024), (1024, 1536), (1536, 1024)}


class MetadataChecker:
    name = "metadata"

    def predict(self, media) -> "Prediction":  # noqa: F821
        kind = media_kind(media)
        if kind == "image":
            findings, extra = image_findings(Path(media))
        elif kind in ("video", "audio"):
            findings, extra = av_findings(Path(media), kind)
        else:
            return provenance_prediction(self.name, "not_checked", None, "Unrecognized media.", [])
        codes = {f.code for f in findings}
        ai = [f for f in findings if f.severity == "strong"]
        if ai:
            return provenance_prediction(self.name, "ai_generated", "fake",
                                         "The metadata carries traces of a generative tool: " + ai[0].message + ".",
                                         findings, **extra)
        if any(f.severity == "moderate" for f in findings):
            moderate = [f for f in findings if f.severity == "moderate"]
            return provenance_prediction(self.name, "edited", None,
                                         "The metadata shows signs of editing: " + moderate[0].message + ".",
                                         findings, **extra)
        if "meta.reencoded" in codes:
            return provenance_prediction(self.name, "re_encoded", None,
                                         "Re-encoded by software after capture; original metadata is gone.",
                                         findings, **extra)
        if "meta.camera" in codes:
            return provenance_prediction(self.name, "capture_metadata", None,
                                         "Camera or phone metadata is present (supporting, but easy to forge).",
                                         findings, **extra)
        if "meta.none" in codes:
            return provenance_prediction(self.name, "no_metadata", None,
                                         "No metadata: stripped, as most platforms and generators do.",
                                         findings, **extra)
        return provenance_prediction(self.name, "no_findings", None, "Nothing notable in the metadata.",
                                     findings, **extra)

    def predict_batch(self, inputs) -> list:
        return [self.predict(x) for x in inputs]


# -- images ----------------------------------------------------------------------------------

def image_findings(path: Path) -> tuple[list[Finding], dict]:
    from PIL import ExifTags, Image

    findings: list[Finding] = []
    with Image.open(path) as img:
        fmt, size, info = img.format, img.size, dict(img.info)
        exif = img.getexif()
        tags = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}
        try:
            tags.update({ExifTags.TAGS.get(k, str(k)): v for k, v in exif.get_ifd(ExifTags.IFD.Exif).items()})
        except Exception:
            pass
        quality = jpeg_quality(img) if fmt == "JPEG" else None
        thumb_corr = exif_thumbnail_match(img, info.get("exif")) if fmt == "JPEG" else None
    text = {k: v for k, v in info.items() if isinstance(v, (str, bytes)) and k not in ("exif", "icc_profile")}
    text = {k: (v.decode("utf-8", "replace") if isinstance(v, bytes) else v) for k, v in text.items()}
    xmp = text.pop("XML:com.adobe.xmp", None) or text.pop("xmp", None) or _xmp_packet(path)
    strings = {k: str(v) for k, v in tags.items() if isinstance(v, (str, bytes))}

    # generator parameters left by Stable Diffusion front ends
    params = text.get("parameters", "")
    if params and re.search(r"\bSteps:\s*\d+", params) and re.search(r"\b(Sampler|CFG scale|Seed):", params):
        findings.append(Finding("meta.sd_parameters", "strong",
                                "Stable Diffusion generation parameters (Automatic1111-style) in the image",
                                {"parameters": params[:300]}))
    for key in ("prompt", "workflow"):
        if '"class_type"' in text.get(key, "") or '"nodes"' in text.get(key, ""):
            findings.append(Finding("meta.comfyui", "strong", f"ComfyUI {key} embedded in the image",
                                    {key: text[key][:300]}))
            break
    for key in ("invokeai_metadata", "sd-metadata", "dream", "Dream"):
        if key in text:
            findings.append(Finding("meta.generator_chunk", "strong", f"Generator metadata chunk '{key}'",
                                    {key: text[key][:300]}))
    if xmp and (m := AI_SOURCE_RE.search(xmp)):
        findings.append(Finding("meta.iptc_ai", "strong",
                                f"IPTC digital source type declares AI-generated content ({m.group(1)})"))
    tool_fields = {**{k: v for k, v in strings.items() if TOOL_KEYS.search(k)},
                   **{k: v for k, v in text.items() if TOOL_KEYS.search(k)}}
    if xmp:
        tool_fields.update({m.group(1): m.group(2) for m in
                            re.finditer(r"(xmp:CreatorTool|tiff:Software)\s*=\s*\"([^\"]+)\"|<(?:xmp:CreatorTool|tiff:Software)>([^<]+)<", xmp)
                            if m.group(1)})
    findings += _tool_findings(tool_fields)
    for key, value in text.items():
        if key in ("parameters", "prompt", "workflow") or not GENERATOR_RE.search(value or ""):
            continue
        strong = MADE_WITH_RE.search(value)
        findings.append(Finding("meta.generator_text", "strong" if strong else "moderate",
                                f"The '{key}' field says it was {strong.group(0)}" if strong
                                else f"A generative tool is named in the '{key}' text field",
                                {"field": key, "value": value[:200]}))
    make, model = strings.get("Make"), strings.get("Model")
    if make or model:
        findings.append(Finding("meta.camera", "info", f"Camera metadata: {' '.join(x for x in (make, model) if x)}",
                                {"make": make, "model": model, "taken": strings.get("DateTimeOriginal")}))
    if thumb_corr is not None and thumb_corr < 0.6:
        findings.append(Finding("meta.thumbnail_mismatch", "moderate",
                                "The embedded EXIF thumbnail doesn't match the image (edited after capture?)",
                                {"correlation": round(thumb_corr, 3)}))
    if not tags and not text and not xmp:
        findings.append(Finding("meta.none", "info", "No EXIF, XMP or text metadata"))
    if fmt == "PNG" and not (make or model) and tuple(size) in SD_BUCKETS:
        findings.append(Finding("meta.generator_size", "info",
                                f"{size[0]}x{size[1]} PNG without camera metadata: a common generator output size"))
    extra = {"format": fmt, "size": list(size), "jpeg_quality": quality,
             "exif": {k: v[:200] for k, v in strings.items()} if strings else {},
             "text_chunks": sorted(text), "xmp": bool(xmp)}
    return findings, extra


def _tool_findings(fields: dict) -> list[Finding]:
    out = []
    for key, value in fields.items():
        value = str(value)
        if m := GENERATOR_RE.search(value):
            out.append(Finding("meta.generator_tool", "strong", f"{key} names a generative tool: {value[:80]}",
                               {"field": key, "value": value[:200], "match": m.group(1)}))
        elif m := EDITORS_RE.search(value):
            out.append(Finding("meta.editor", "moderate", f"Edited with {m.group(1)} ({key}: {value[:60]})",
                               {"field": key, "value": value[:200]}))
    return out


def _xmp_packet(path: Path) -> str | None:
    """The XMP packet, found by scanning the first 2 MB (works for JPEG, PNG, WebP, TIFF)."""
    head = path.read_bytes()[: 2 << 20]
    start = head.find(b"<x:xmpmeta")
    if start < 0:
        return None
    end = head.find(b"</x:xmpmeta>", start)
    return head[start: end + 12].decode("utf-8", "replace") if end > 0 else None


def jpeg_quality(img) -> int | None:
    """libjpeg-equivalent quality from the luminance quantization table (as ImageMagick does)."""
    q = getattr(img, "quantization", None)
    if not q or 0 not in q:
        return None
    std = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
                    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], float)
    table = np.array(q[0], float)
    scale = np.mean(table * 100 / std)
    quality = (200 - scale) / 2 if scale <= 100 else 5000 / scale
    return int(round(min(100, max(1, quality))))


def exif_thumbnail_match(img, exif_bytes) -> float | None:
    """Correlation between the EXIF thumbnail and the image downscaled to its size."""
    from PIL import ExifTags, Image

    if not exif_bytes:
        return None
    try:
        ifd1 = img.getexif().get_ifd(ExifTags.IFD.IFD1)
        offset, length = ifd1.get(0x0201), ifd1.get(0x0202)
        if not offset or not length:
            return None
        tiff = exif_bytes[6:] if exif_bytes[:6] == b"Exif\x00\x00" else exif_bytes
        with Image.open(io.BytesIO(tiff[offset:offset + length])) as thumb:
            t = np.asarray(thumb.convert("L"), float)
        main = np.asarray(img.convert("L").resize((t.shape[1], t.shape[0])), float)
        if t.std() < 1 or main.std() < 1:
            return None
        return float(np.corrcoef(t.ravel(), main.ravel())[0, 1])
    except Exception:
        return None


# -- video and audio ------------------------------------------------------------------------

def av_findings(path: Path, kind: str) -> tuple[list[Finding], dict]:
    info = ffprobe(path)
    fmt, streams = info.get("format", {}), info.get("streams", [])
    findings: list[Finding] = []
    all_tags = {f"format:{k}": v for k, v in (fmt.get("tags") or {}).items()}
    for s in streams:
        all_tags.update({f"{s.get('codec_type')}{s.get('index')}:{k}": v for k, v in (s.get("tags") or {}).items()})
    tool_fields = {k: v for k, v in all_tags.items() if TOOL_KEYS.search(k.split(":", 1)[1])}
    findings += _tool_findings(tool_fields)
    for key, value in all_tags.items():
        if key in tool_fields:
            continue
        value = str(value)
        if AI_SOURCE_RE.search(value) or (GENERATOR_RE.search(value) and
                                          re.search(r"(comment|description|title|synopsis|keywords)", key, re.I)):
            made = MADE_WITH_RE.search(value)
            findings.append(Finding("meta.generator_text", "strong" if AI_SOURCE_RE.search(value) or made else "moderate",
                                    f"{key} says it was {made.group(0)}" if made
                                    else f"{key} mentions a generative tool or AI source",
                                    {"field": key, "value": value[:200]}))
    encoder = str((fmt.get("tags") or {}).get("encoder", ""))
    camera = {k: v for k, v in all_tags.items()
              if re.search(r"(com\.apple\.quicktime\.(make|model|location)|com\.android\.(version|manufacturer|model)|"
                           r"location|make$|model$)", k.split(":", 1)[1], re.I)}
    if camera:
        findings.append(Finding("meta.camera", "info", "Camera or phone tags present", {"tags": camera}))
    if encoder.startswith("Lavf") or any(str(v).startswith("Lavc") for v in all_tags.values()):
        findings.append(Finding("meta.reencoded", "info", f"Written by FFmpeg ({encoder or 'Lavc'})",
                                {"encoder": encoder}))
    if not all_tags.get("format:creation_time") and kind == "video":
        findings.append(Finding("meta.no_creation_time", "info", "No creation time in the container"))
    durations = {s.get("codec_type"): float(s["duration"]) for s in streams if s.get("duration") not in (None, "N/A")}
    if "video" in durations and "audio" in durations and abs(durations["video"] - durations["audio"]) > 0.5:
        findings.append(Finding("meta.track_length_mismatch", "moderate",
                                f"Audio and video tracks differ in length ({durations['audio']:.2f} s vs "
                                f"{durations['video']:.2f} s): edited or muxed separately",
                                {"audio_s": durations["audio"], "video_s": durations["video"]}))
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video and _fps(video.get("r_frame_rate")) and _fps(video.get("avg_frame_rate")):
        r, avg = _fps(video["r_frame_rate"]), _fps(video["avg_frame_rate"])
        if abs(r - avg) / max(r, avg) > 0.05:
            findings.append(Finding("meta.variable_frame_rate", "info",
                                    f"Variable frame rate ({avg:.2f} average vs {r:.2f} nominal fps): phone camera, "
                                    "screen recording or editing app", {"avg_fps": avg, "nominal_fps": r}))
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if kind == "audio" and audio and not all_tags and int(audio.get("sample_rate") or 0) in (16000, 22050, 24000) \
            and int(audio.get("channels") or 0) == 1:
        findings.append(Finding("meta.tts_format", "info",
                                f"Mono {int(audio['sample_rate']) // 1000} kHz audio with no tags, a format typical of "
                                "speech synthesis (and of many recorders)"))
    if not all_tags:
        findings.append(Finding("meta.none", "info", "No container or stream tags"))
    extra = {"container": fmt.get("format_name"), "encoder": encoder or None,
             "codecs": [s.get("codec_name") for s in streams], "tags": {k: str(v)[:200] for k, v in all_tags.items()},
             "duration_s": float(fmt["duration"]) if fmt.get("duration") not in (None, "N/A") else None}
    return findings, extra


def _fps(rate: str | None) -> float | None:
    try:
        num, den = (rate or "").split("/")
        return float(num) / float(den) if float(den) else None
    except ValueError:
        return None


register(
    ModelSpec(
        name="metadata",
        family="provenance",
        builder=lambda weights_path, device="cpu": MetadataChecker(),
        weights=None,
        license="ForensiX (algorithmic). No model weights.",
        source="",
        paper="",
        description="Reads what the file says about its own history: generator parameters and AI source types, "
        "generator and editor names, EXIF thumbnail mismatch, camera tags, FFmpeg re-encoding, audio/video "
        "track mismatch. Can flag AI generation; never vouches for authenticity.",
    )
)
