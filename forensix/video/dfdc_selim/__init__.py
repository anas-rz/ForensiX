"""DFDC winner (Selim Seferbekov, Deepfake Detection Challenge 2020, 1st place).

Per video: 32 evenly spaced frames -> MTCNN faces -> 7 EfficientNet-B7 (Noisy
Student init) classifiers -> "confident strategy" per model -> mean of models.
Adapted from https://github.com/selimsef/dfdc_deepfake_challenge
(kernel_utils.py, training/zoo/classifiers.py; MIT, see LICENSE in this
directory). Differences: runs in fp32 (upstream: fp16 on GPU), and videos with
fewer than 32 frames use each frame once (upstream drops frames after the
first repeated index).
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import Normalize

from forensix.hub import WeightFile, fetch, torch_load
from forensix.registry import ModelSpec, register
from forensix.video.detector import VideoDetector, VideoInput
from forensix.video.io import read_frames
from forensix.video.mtcnn import MTCNN_WEIGHTS

_RELEASE = "https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/0.0.1/"
_CHECKPOINTS = {
    "final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36": "9db77ab9318863e2f8ab287c8eb83c2232584b82dc2fb41f1d614ddd7900cccb",
    "final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19": "e2f0066a8ff036add2f5b418d2eba1051c2d2788f84168f00a85207f3b666e04",
    "final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_29": "9fe4311d551a9cbaa1475dbcd6b15fccb31a1e17a8d63069e90a8899aef03b5c",
    "final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31": "00dd6cd9466cddfd7da7c333d8fecae81593307827093deaf4d7cdf704bc8bfa",
    "final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_37": "d4a2bb4aefde59e045463ad469ed9660765beb086f65a09f9478ab060d403c7c",
    "final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_40": "c1d326a6651011390efe602e42cb8228232d6d18a15c404cf85d7233482f6d6d",
    "final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23": "849036dc211387420412feb68c7451e98a948072b94e213e1104e2f2bf7791ad",
}
WEIGHTS = [WeightFile(_RELEASE + name, sha, f"dfdc-selim/{name}") for name, sha in _CHECKPOINTS.items()]

FRAMES_PER_VIDEO = 32
INPUT_SIZE = 380
MAX_FACES = 4 * FRAMES_PER_VIDEO - 1  # upstream batch of 128 keeps at most 127 faces
_normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class DeepFakeClassifier(nn.Module):
    """Upstream training/zoo/classifiers.DeepFakeClassifier (tf_efficientnet_b7_ns)."""

    def __init__(self):
        super().__init__()
        import timm  # imported lazily

        self.encoder = timm.create_model("tf_efficientnet_b7", pretrained=False)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.0)
        self.fc = nn.Linear(2560, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avg_pool(self.encoder.forward_features(x)).flatten(1)
        return self.fc(self.dropout(x))


def _load_checkpoint(path: Path) -> dict:
    # checkpoints hold {"epoch", "state_dict", "bce_best": numpy float64}
    state = torch_load(path)["state_dict"]
    return {k.removeprefix("module."): v for k, v in state.items()}


def isotropically_resize(img: np.ndarray, size: int) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    scale = size / max(w, h)
    new_w, new_h = (size, h * scale) if w > h else (w * scale, size)
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(img, (int(new_w), int(new_h)), interpolation=interpolation)


def put_to_center(img: np.ndarray, size: int) -> np.ndarray:
    img = img[:size, :size]
    out = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    out[top : top + img.shape[0], left : left + img.shape[1]] = img
    return out


def extract_faces(frames: np.ndarray, detector) -> list[np.ndarray]:
    """MTCNN on half-resolution frames; crops padded by 1/3 of the box size."""
    faces = []
    for frame in frames:
        img = Image.fromarray(frame)
        img = img.resize(size=[s // 2 for s in img.size])
        boxes, _ = detector.detect(img, landmarks=False)
        if boxes is None:
            continue
        for box in boxes:
            xmin, ymin, xmax, ymax = [int(b * 2) for b in box]
            p_h, p_w = (ymax - ymin) // 3, (xmax - xmin) // 3
            faces.append(frame[max(ymin - p_h, 0) : ymax + p_h, max(xmin - p_w, 0) : xmax + p_w])
    return faces


def confident_strategy(pred: np.ndarray, t: float = 0.8) -> float:
    """Upstream frame aggregation: trust a confident majority, else the mean."""
    sz = len(pred)
    fakes = np.count_nonzero(pred > t)
    if fakes > sz // 2.5 and fakes > 11:
        return float(np.mean(pred[pred > t]))
    if np.count_nonzero(pred < 0.2) > 0.9 * sz:
        return float(np.mean(pred[pred < 0.2]))
    return float(np.mean(pred))


class DFDCSelimDetector(VideoDetector):
    def __init__(self, models: list[nn.Module], device: str = "cpu", chunk_size: int = 8):
        super().__init__("dfdc-selim")
        from forensix.video.mtcnn.mtcnn import MTCNN

        self.device = torch.device(device)
        self.models = [m.to(self.device).eval() for m in models]
        self.face_detector = MTCNN(margin=0, thresholds=[0.7, 0.8, 0.8], device=self.device)
        self.chunk_size = chunk_size

    @torch.inference_mode()
    def _score(self, video: VideoInput) -> tuple[float, list[float]]:
        frames, _ = read_frames(video, FRAMES_PER_VIDEO)
        faces = extract_faces(frames, self.face_detector)[:MAX_FACES]
        if not faces:
            return 0.5, []  # upstream's answer when no face is found
        x = np.stack([put_to_center(isotropically_resize(f, INPUT_SIZE), INPUT_SIZE) for f in faces])
        x = _normalize(torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0)
        per_model = []
        for model in self.models:
            probs = torch.cat(
                [model(chunk.to(self.device)).sigmoid().flatten().cpu() for chunk in x.split(self.chunk_size)]
            ).numpy()
            per_model.append(probs)
        video_prob = float(np.mean([confident_strategy(p) for p in per_model]))
        return video_prob, np.mean(per_model, axis=0).tolist()


def build(weights_path: Path, device: str = "cpu") -> DFDCSelimDetector:
    models = []
    for weights in WEIGHTS:
        model = DeepFakeClassifier()
        model.load_state_dict(_load_checkpoint(fetch(weights)))
        models.append(model)
    return DFDCSelimDetector(models, device=device)


register(
    ModelSpec(
        name="dfdc-selim",
        family="video",
        builder=build,
        weights=WEIGHTS[0],
        extra_weights=(*WEIGHTS[1:], *MTCNN_WEIGHTS.values()),
        license="MIT (code, weights). Backbone init: timm tf_efficientnet_b7 Noisy "
        "Student (Apache-2.0). Face detector: MTCNN via facenet-pytorch (MIT)",
        source="https://github.com/selimsef/dfdc_deepfake_challenge/tree/89c6290490bac96b29193a4061b3db9dd3933e36",
        paper="https://arxiv.org/abs/2006.07397",  # the challenge paper; the solution has none
        description="1st place, Deepfake Detection Challenge (Kaggle, 2020): ensemble of 7 "
        "EfficientNet-B7 face classifiers over 32 frames per video (1.8 GB of weights). "
        "Face-swap / face-manipulation detector; returns 0.5 if no face is found.",
    )
)
