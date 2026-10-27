"""Invisible watermarks that open image generators embed (the "dwtDct" method).

The Stable Diffusion reference code and Hugging Face diffusers embed a fixed
message with invisible-watermark's dwtDct method (MIT):

- SD 1.x (CompVis ``txt2img.py``): the bytes ``"StableDiffusionV1"``;
- SD 2.x (Stability ``txt2img.py``): the bytes ``"SDV2"``;
- SDXL and Stable Video Diffusion (Stability ``generative-models``, diffusers
  ``StableDiffusionXLWatermarker``): a 48-bit message.

All three embed in OpenCV BGR channel order. Decoding is a vectorized version of the
vendored reference (``imwatermark/maxdct.py``), checked equal to it bit for bit.
A message counts as found when at least 90% of its bits match; by chance that's
below 1e-6 for any of these lengths.

The watermark survives JPEG and colour changes, but not resizing, cropping,
screenshots or most video re-encoding. Finding it is strong evidence of
generation; not finding it says nothing.
"""

import numpy as np

from forensix.provenance.evidence import Finding, media_kind, provenance_prediction
from forensix.registry import ModelSpec, register

SDXL_BITS = [int(b) for b in bin(0b101100111110110010010000011110111011000110011110)[2:]]
KNOWN_MESSAGES = {
    "Stable Diffusion 1.x": np.unpackbits(np.frombuffer(b"StableDiffusionV1", np.uint8)),
    "Stable Diffusion 2.x": np.unpackbits(np.frombuffer(b"SDV2", np.uint8)),
    "SDXL / Stable Video Diffusion": np.array(SDXL_BITS, np.uint8),
}
FOUND_AT = 0.9
BLOCK, SCALES = 4, (0, 36, 36)  # invisible-watermark's dwtDct defaults


def block_bits(bgr: np.ndarray) -> list[np.ndarray]:
    """Per-block bit votes of the dwtDct decoder, in the reference's block order, for
    each watermarked channel (only U with the default scales). Grouping them by ``index % length`` and averaging
    gives the reference ``EmbedMaxDct(wmLen=length).decode`` for any message length."""
    import cv2
    import pywt

    rows, cols = bgr.shape[:2]
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    votes = []
    for channel in range(2):  # as the reference: only channels 0 and 1, so only U (Y has scale 0)
        scale = SCALES[channel]
        if scale <= 0:
            continue
        ca, _ = pywt.dwt2(yuv[: rows // 4 * 4, : cols // 4 * 4, channel], "haar")
        br, bc = ca.shape[0] // BLOCK, ca.shape[1] // BLOCK
        blocks = ca[: br * BLOCK, : bc * BLOCK].reshape(br, BLOCK, bc, BLOCK).swapaxes(1, 2).reshape(-1, BLOCK * BLOCK)
        # the largest-magnitude coefficient, skipping the first (DC-like) one
        pos = np.argmax(np.abs(blocks[:, 1:]), axis=1) + 1
        val = np.abs(blocks[np.arange(len(blocks)), pos])
        votes.append(((val % scale) > 0.5 * scale).astype(np.float64))
    return votes


def decode(bgr: np.ndarray, length: int, votes: list[np.ndarray] | None = None) -> np.ndarray:
    """The decoded ``length``-bit message (as the reference: mean vote per bit > 127/255)."""
    votes = block_bits(bgr) if votes is None else votes
    sums, counts = np.zeros(length), np.zeros(length)
    for v in votes:
        idx = np.arange(len(v)) % length
        sums += np.bincount(idx, weights=v, minlength=length)
        counts += np.bincount(idx, minlength=length)
    return (sums / np.maximum(counts, 1) * 255 > 127).astype(np.uint8)


def match(bgr: np.ndarray) -> dict[str, float]:
    """Bit accuracy of each known generator message."""
    votes = block_bits(bgr)
    return {name: float(np.mean(decode(bgr, len(bits), votes) == bits)) for name, bits in KNOWN_MESSAGES.items()}


class ImageWatermarkChecker:
    name = "watermark-image"
    frames = 8

    def predict(self, media) -> "Prediction":  # noqa: F821
        import cv2

        kind = media_kind(media) if not isinstance(media, np.ndarray) else "image"
        if kind == "image":
            if isinstance(media, np.ndarray):
                bgr = np.ascontiguousarray(media[:, :, ::-1])  # RGB arrays, as elsewhere in ForensiX
            else:
                from forensix.image.io import load_image

                bgr = np.ascontiguousarray(np.array(load_image(media))[:, :, ::-1])
            per_frame = [match(bgr)]
        elif kind == "video":
            from forensix.video.io import read_frames

            frames, _ = read_frames(media, self.frames)
            per_frame = [match(np.ascontiguousarray(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))) for f in frames]
        else:
            return provenance_prediction(self.name, "not_checked", None, "Not an image or video.", [])
        accuracy = {name: float(np.median([m[name] for m in per_frame])) for name in KNOWN_MESSAGES}
        best = max(accuracy, key=accuracy.get)
        extra = {"bit_accuracy": {k: round(v, 4) for k, v in accuracy.items()}, "frames_checked": len(per_frame)}
        if accuracy[best] >= FOUND_AT:
            finding = Finding("watermark.sd_dwtdct", "strong",
                              f"Invisible watermark of {best} found ({accuracy[best]:.0%} of bits match)",
                              {"generator": best, "bit_accuracy": round(accuracy[best], 4)})
            return provenance_prediction(self.name, "watermark_found", "fake",
                                         f"The {best} invisible watermark is present: generated by that model.",
                                         [finding], generator=best, **extra)
        return provenance_prediction(self.name, "no_watermark", None,
                                     "No known generator watermark found (it doesn't survive resizing, cropping "
                                     "or most re-encoding, so absence proves nothing).", [], **extra)

    def predict_batch(self, inputs) -> list:
        return [self.predict(x) for x in inputs]


register(
    ModelSpec(
        name="watermark-image",
        family="provenance",
        builder=lambda weights_path, device="cpu": ImageWatermarkChecker(),
        weights=None,
        license="MIT (invisible-watermark dwtDct method, vendored). No model weights.",
        source="https://github.com/ShieldMnt/invisible-watermark",
        paper="",
        description="Finds the invisible watermarks Stable Diffusion 1.x, 2.x, SDXL and Stable Video Diffusion "
        "embed by default (dwtDct). Strong evidence when found; removed by resizing and most re-encoding.",
    )
)
