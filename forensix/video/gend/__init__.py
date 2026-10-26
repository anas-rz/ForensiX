"""GenD (Yermakov et al., WACV 2026): LayerNorm-tuned foundation-model encoders
(CLIP ViT-L/14, Perception Encoder L) for generalizable face-deepfake detection.

Weights: https://huggingface.co/yermandy/GenD_CLIP_L_14 and GenD_PE_L (MIT). Code reference:
https://github.com/yermandy/GenD (MIT, see LICENSE in this directory).

Upstream preprocesses faces with InsightFace's SCRFD detector, whose weights
are non-commercial. ForensiX uses MTCNN (MIT) for the 5 landmarks instead and
keeps GenD's alignment (1.3x context, 256x256), so face crops can differ
slightly from the ones GenD was trained on.
"""

from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn

from forensix.hub import WeightFile
from forensix.image.clip import clip_vit_l14
from forensix.registry import ModelSpec, register
from forensix.video.detector import VideoDetector, VideoInput
from forensix.video.faces import align_face, largest_face_landmarks
from forensix.video.io import read_frames
from forensix.video.mtcnn import MTCNN_WEIGHTS

FRAMES_PER_VIDEO = 32
DETECT_MAX_SIDE = 640  # detect on a downscaled frame, like SCRFD's 640 input


def _subset(state: dict, prefix: str) -> dict:
    return {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}


class GenD(nn.Module):
    """Upstream src/hf/modeling_gend.GenD with head=LinearNorm and either the
    CLIP ViT-L/14 or the Perception Encoder (PE-Core L) backbone."""

    def __init__(self, state: dict, backbone: str = "clip"):
        super().__init__()
        self.kind = backbone
        if backbone == "clip":
            self.backbone = clip_vit_l14(_subset(state, "feature_extractor.")).vision_model
        elif backbone == "pe":
            import timm  # imported lazily

            # upstream PerceptionEncoder: timm Eva with the classifier head removed
            self.backbone = timm.create_model(
                "vit_pe_core_large_patch14_336", pretrained=False, dynamic_img_size=True
            )
            self.backbone.head = nn.Identity()
            self.backbone.load_state_dict(_subset(state, "feature_extractor.backbone."))
        else:
            raise ValueError(f"Unknown backbone {backbone!r}")
        self.linear = nn.Linear(1024, 2)
        self.linear.load_state_dict(_subset(state, "model.linear."))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.kind == "clip":
            features = self.backbone(pixel_values).pooler_output
        else:
            features = self.backbone(pixel_values)
        return self.linear(nn.functional.normalize(features, p=2, dim=1))

    def preprocess(self):
        """Upstream encoder preprocessing: callable(list of HxWx3 uint8) -> tensor."""
        if self.kind == "clip":
            from transformers import CLIPImageProcessor

            processor = CLIPImageProcessor()  # OpenAI CLIP: resize 224 bicubic, crop, normalize
            return lambda faces: processor(images=faces, return_tensors="pt")["pixel_values"]
        import timm
        from PIL import Image

        config = timm.data.resolve_model_data_config(self.backbone)
        config["input_size"] = (3, 224, 224)  # as upstream
        transform = timm.data.create_transform(**config, is_training=False)
        return lambda faces: torch.stack([transform(Image.fromarray(f)) for f in faces])


def face_landmarks(frame: np.ndarray, face_detector) -> np.ndarray | None:
    """5 landmarks of the largest face, detected on the frame downscaled to DETECT_MAX_SIDE."""
    import cv2

    factor = min(1.0, DETECT_MAX_SIDE / max(frame.shape[:2]))
    small = frame if factor == 1.0 else cv2.resize(
        frame, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    landmarks = largest_face_landmarks(small, face_detector)
    return None if landmarks is None else landmarks / factor


def aligned_faces(frames: np.ndarray, face_detector) -> list[np.ndarray]:
    """Aligned 256x256 crops of the largest face in each frame that has one."""
    faces = []
    for frame in frames:
        landmarks = face_landmarks(frame, face_detector)
        if landmarks is not None:
            faces.append(align_face(frame, landmarks))
    return faces


class GenDDetector(VideoDetector):
    def __init__(self, name: str, network: GenD, device: str = "cpu"):
        super().__init__(name)
        from forensix.video.mtcnn.mtcnn import MTCNN

        self.device = torch.device(device)
        self.network = network.to(self.device).eval()
        self.preprocess = network.preprocess()
        self.face_detector = MTCNN(device=self.device)

    def _landmarks(self, frame: np.ndarray) -> np.ndarray | None:
        return face_landmarks(frame, self.face_detector)

    @torch.inference_mode()
    def face_fake_probs(self, faces: list[np.ndarray]) -> np.ndarray:
        """Fake probability for aligned 256x256 RGB face crops."""
        pixels = self.preprocess(faces)
        return self.network(pixels.to(self.device)).softmax(dim=-1)[:, 1].cpu().numpy()

    def _score(self, video: VideoInput) -> tuple[float, list[float]]:
        frames, _ = read_frames(video, FRAMES_PER_VIDEO)
        faces = aligned_faces(frames, self.face_detector)
        if not faces:
            return 0.5, []
        probs = np.concatenate([self.face_fake_probs(faces[i : i + 8]) for i in range(0, len(faces), 8)])
        return float(probs.mean()), probs.tolist()


def _builder(name: str, backbone: str, load=load_file):
    def build(weights_path: Path, device: str = "cpu") -> GenDDetector:
        return GenDDetector(name, GenD(load(weights_path), backbone), device=device)

    return build


def load_dfdet_checkpoint(path: Path) -> dict:
    """Predecessor model's Lightning checkpoint -> GenD-CLIP state dict.

    Its CLIP LayerNorms are PEFT LN-tuning wrappers: the tuned copy
    (``ln_tuning_layers.default``) replaces the frozen ``base_layer``.
    """
    from forensix.hub import torch_load

    state = {}
    for key, value in torch_load(path)["state_dict"].items():
        if ".base_layer." in key:
            continue
        key = key.replace("feature_extractor.base_model.model.", "feature_extractor.")
        state[key.replace(".ln_tuning_layers.default.", ".")] = value
    return state


_COMMON = dict(
    family="video",
    extra_weights=tuple(MTCNN_WEIGHTS.values()),
    source="https://github.com/yermandy/GenD/tree/387a42266dd385fbe8f7626c5d3aa03eea3bfaab",
    paper="https://arxiv.org/abs/2508.06248",
)

register(
    ModelSpec(
        name="gend",
        builder=_builder("gend", "clip"),
        weights=WeightFile(
            url="https://huggingface.co/yermandy/GenD_CLIP_L_14/resolve/"
            "891ce014a0308386c4d7d25b3dcf436a22db5504/model.safetensors",
            sha256="d76f0bdfd74a29fe1b1c1b84a80ac92486993e426878e8c7a3944281fbb96833",
            filename="gend-clip-l14.safetensors",
        ),
        license="MIT (code, weights; fine-tuned from OpenAI CLIP ViT-L/14, MIT, whose model "
        "card lists deployed use as out of scope). Face detector: MTCNN (MIT)",
        description="LayerNorm-tuned CLIP ViT-L/14 with L2-normalized features and a linear "
        "head, trained on FaceForensics++ for cross-dataset generalization. Scores the "
        "largest face in 32 frames; video score is the mean.",
        **_COMMON,
    )
)

register(
    ModelSpec(
        name="gend-pe",
        builder=_builder("gend-pe", "pe"),
        weights=WeightFile(
            url="https://huggingface.co/yermandy/GenD_PE_L/resolve/"
            "4fd5a38cf071af2640d314110826ed5cd25cab88/model.safetensors",
            sha256="e2367f8db3a6b76f8d5cc5857f1ed0ea0993b33c9e1f428d3b06e5ca95de2213",
            filename="gend-pe-l.safetensors",
        ),
        license="MIT (code, weights; fine-tuned from Meta Perception Encoder PE-Core L, "
        "Apache-2.0). Face detector: MTCNN (MIT)",
        description="GenD on Meta's Perception Encoder (PE-Core L/14): LayerNorm-tuned, "
        "L2-normalized features, linear head, trained on FaceForensics++. Same face "
        "pipeline as `gend`.",
        **_COMMON,
    )
)

register(
    ModelSpec(
        name="dfdet-clip",
        builder=_builder("dfdet-clip", "clip", load=load_dfdet_checkpoint),
        weights=WeightFile(
            url="https://huggingface.co/yermandy/deepfake-detection/resolve/"
            "9a6857ec642deb57373c5437be803a199468b8c6/model.ckpt",
            sha256="57a0d00f2f5b4046afd2c344ff9877a35e8889e075916cf816796c54816c9955",
            filename="dfdet-clip.ckpt",
        ),
        license="MIT (code, weights; fine-tuned from OpenAI CLIP ViT-L/14, MIT, whose model "
        "card lists deployed use as out of scope). Face detector: MTCNN (MIT)",
        description="GenD's predecessor ('Unlocking the Hidden Potential of CLIP in "
        "Generalizable Deepfake Detection'): LN-tuned CLIP ViT-L/14 trained on FaceForensics++ "
        "with label smoothing and uniformity/alignment losses on the hypersphere. Same face "
        "pipeline as `gend`.",
        family="video",
        extra_weights=tuple(MTCNN_WEIGHTS.values()),
        source="https://github.com/yermandy/deepfake-detection/tree/0b872f7903e72d2af8c5fff262d2fc2ca99bb44f",
        paper="https://arxiv.org/abs/2503.19683",
    )
)
