"""AudioSeal: Meta's localized audio watermark (Roman et al., ICML 2024).

Code and weights are MIT (https://github.com/facebookresearch/audioseal); the
detector's architecture comes from the ``audioseal`` package's own model card, and
its checkpoint is fetched through the ForensiX weights cache (pinned revision,
SHA-256 checked) and loaded without unpickling code. Install with
``pip install -e ".[provenance]"``.

The detector scores every audio sample (16 kHz) for the watermark and decodes a
16-bit message. As upstream, the clip's detection probability is the share of
samples above 0.5; ``details["timeline"]`` gives it per second, so partly
watermarked audio (e.g. a generated segment spliced into a real recording) shows
where. AudioSeal is used by Meta's generative audio tools and anyone can embed
it; finding it is strong evidence the audio went through such a generator, and
not finding it says nothing.
"""

import pickle
import types

import numpy as np
import torch

from forensix.hub import WeightFile
from forensix.provenance.evidence import Finding, provenance_prediction
from forensix.registry import ModelSpec, register

REVISION = "3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba"
DETECTOR = WeightFile(
    url=f"https://huggingface.co/facebook/audioseal/resolve/{REVISION}/detector_base.pth",
    sha256="8a78e8a83584113523e161fc599fcab10fd0e94c04d2eb9d2fa1e9ec91ab69d9",
    filename="audioseal/detector_base.pth",
)
SAMPLE_RATE = 16000
FOUND_AT = 0.5        # upstream's detection threshold on the share of watermarked samples
MAX_SECONDS = 600     # scored in chunks of CHUNK_SECONDS; longer audio is truncated
CHUNK_SECONDS = 60


# Everything the checkpoint's pickle references (listed statically with pickletools): tensors, plain
# containers, and the training config it carries as omegaconf objects. Data only, no callables.
_ALLOWED_GLOBALS = {
    ("collections", "OrderedDict"), ("collections", "defaultdict"), ("builtins", "dict"), ("builtins", "list"),
    ("builtins", "int"), ("typing", "Any"), ("torch._utils", "_rebuild_tensor_v2"),
    ("omegaconf.dictconfig", "DictConfig"), ("omegaconf.listconfig", "ListConfig"),
    ("omegaconf.base", "ContainerMetadata"), ("omegaconf.base", "Metadata"), ("omegaconf.nodes", "AnyNode"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        module = {"__builtin__": "builtins"}.get(module, module)
        if (module, name) == ("builtins", "long"):  # Python 2's int, as the checkpoint names it
            name = "int"
        if (module, name) not in _ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"{module}.{name} isn't allowed in an AudioSeal checkpoint")
        return super().find_class(module, name)


_restricted_pickle = types.SimpleNamespace(Unpickler=_RestrictedUnpickler, __name__="forensix_restricted_pickle",
                                           load=lambda *a, **k: (_ for _ in ()).throw(pickle.UnpicklingError(
                                               "legacy checkpoint format isn't supported")))


def load_checkpoint(path) -> dict:
    """The checkpoint's model weights. The file also pickles its training config as omegaconf
    objects, which torch's weights_only loader can't rebuild, so it's read with an unpickler
    that resolves only the classes above (tensor storages are rebuilt by torch itself)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_restricted_pickle)
    if "best_state" in ckpt:
        return ckpt["best_state"]["model"]
    return ckpt.get("model", ckpt)


def build_detector(weights_path, device: str = "cpu"):
    try:
        from audioseal.builder import AudioSealDetectorConfig, create_detector
        from audioseal.loader import AudioSeal, convert_state_dict_for_scriptable_model, load_local_model_config
        from audioseal.loader import load_state_dict as load_into
        from omegaconf import OmegaConf
    except ImportError as e:
        raise ImportError('watermark-audio needs the audioseal package: pip install -e ".[provenance]"') from e
    card = OmegaConf.to_container(load_local_model_config("audioseal_detector_16bits"))
    card.pop("checkpoint")
    config = AudioSeal.parse_config(card, AudioSealDetectorConfig, nbits=None)
    model = create_detector(config, device="cpu")
    load_into(model, convert_state_dict_for_scriptable_model(load_checkpoint(weights_path)))
    return model.to(device).eval()


class AudioWatermarkChecker:
    name = "watermark-audio"

    def __init__(self, detector, device: str = "cpu"):
        self.detector, self.device = detector, device

    @torch.inference_mode()
    def scores(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-sample watermark probability and per-bit message probability (16 kHz mono)."""
        probs, bit_logits, n = [], [], CHUNK_SECONDS * SAMPLE_RATE
        for start in range(0, min(len(audio), MAX_SECONDS * SAMPLE_RATE), n):
            x = torch.from_numpy(np.ascontiguousarray(audio[start:start + n], np.float32))[None, None].to(self.device)
            result, message = self.detector(x, sample_rate=SAMPLE_RATE)
            probs.append(result[0, 1].cpu().numpy())
            bit_logits.append((message[0].cpu().numpy(), x.shape[-1]))
        weights = np.array([w for _, w in bit_logits], float)
        bits = np.average(np.stack([b for b, _ in bit_logits]), axis=0, weights=weights)
        return np.concatenate(probs), bits

    def predict(self, media, sample_rate: int | None = None) -> "Prediction":  # noqa: F821
        from forensix.audio.io import load_audio

        audio = load_audio(media, SAMPLE_RATE, sample_rate)
        if len(audio) < SAMPLE_RATE // 4:
            return provenance_prediction(self.name, "not_checked", None, "Audio too short to check.", [])
        per_sample, bits = self.scores(audio)
        detect = float(np.mean(per_sample > 0.5))
        seconds = len(per_sample) // SAMPLE_RATE
        timeline = [round(float(np.mean(per_sample[s * SAMPLE_RATE:(s + 1) * SAMPLE_RATE] > 0.5)), 3)
                    for s in range(max(1, seconds))]
        message = "".join(str(int(b > 0.5)) for b in bits)
        extra = {"detection": round(detect, 4), "message_bits": message, "message_hex": f"{int(message, 2):04x}",
                 "timeline": timeline, "seconds_checked": round(len(per_sample) / SAMPLE_RATE, 2)}
        marked = [i for i, v in enumerate(timeline) if v > 0.5]
        if detect >= FOUND_AT:
            where = "" if detect >= 0.9 or not marked else f", seconds {marked[0]}–{marked[-1] + 1}"
            finding = Finding("watermark.audioseal", "strong",
                              f"AudioSeal watermark detected in {detect:.0%} of the audio{where} "
                              f"(message {extra['message_hex']})",
                              {"detection": round(detect, 4), "message_hex": extra["message_hex"], "seconds": marked})
            return provenance_prediction(self.name, "watermark_found", "fake",
                                         "An AudioSeal watermark is present: the audio went through a watermarking "
                                         "generator.", [finding], **extra)
        if marked:
            finding = Finding("watermark.audioseal_partial", "moderate",
                              f"AudioSeal watermark in part of the audio ({len(marked)} of {len(timeline)} s)",
                              {"seconds": marked})
            return provenance_prediction(self.name, "partial_watermark", None,
                                         "Part of the audio carries an AudioSeal watermark: a watermarked segment "
                                         "may have been spliced in.", [finding], **extra)
        return provenance_prediction(self.name, "no_watermark", None,
                                     "No AudioSeal watermark found (absence proves nothing: most generators "
                                     "don't embed it).", [], **extra)

    def predict_batch(self, inputs, sample_rate: int | None = None) -> list:
        return [self.predict(x, sample_rate) for x in inputs]


register(
    ModelSpec(
        name="watermark-audio",
        family="provenance",
        builder=lambda weights_path, device="cpu": AudioWatermarkChecker(build_detector(weights_path, device), device),
        weights=DETECTOR,
        license="MIT (code and weights, facebookresearch/audioseal; weights MIT since 2024-04-02)",
        source=f"https://huggingface.co/facebook/audioseal/tree/{REVISION}",
        paper="https://arxiv.org/abs/2401.17264",
        description="Detects Meta's AudioSeal watermark and decodes its 16-bit message, per second for "
        "localization. Audio files or a video's sound track.",
    )
)
