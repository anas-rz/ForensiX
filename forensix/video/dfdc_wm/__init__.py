"""DFDC 2nd place (Team WM: Hanqing Zhao, Hao Cui, Wenbo Zhou, 2020).

Per video: every 10th frame -> RetinaFace (largest face) -> ArcFace-aligned
320x320 crop -> Xception (0.2) + WS-DAN/Xception (0.7) + WS-DAN/EfficientNet-B3
(0.1) -> mean over up to 25 faces. Adapted from
https://github.com/cuihaoleo/kaggle-dfdc (submission.py; MIT, see LICENSE in
this directory). Runs in fp32 on any device (upstream: CUDA).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from forensix.hub import WeightFile, fetch, torch_load
from forensix.registry import ModelSpec, register
from forensix.video.detector import VideoDetector, VideoInput

_DRIVE = "https://drive.usercontent.google.com/download?export=download&confirm=t&id="
WEIGHTS = {
    "xception": WeightFile(_DRIVE + "1y21OH5_SKgoHyH7ZeICAC71Zhk5mtPUO",
                           "651ca679c582501b474cfb1664b7f8021160392c5c13f338c53c01eb11711570", "dfdc-wm/xception-hg-2.pth"),
    "wsdan_x": WeightFile(_DRIVE + "1wlrnoxHmgzgItAwxRAfS0q5qArn-4B6g",
                          "8164b6f3671ad9275f5992cd45d5a9e3055696437b74c77fd8daccd56e99c4f4", "dfdc-wm/ckpt_x.pth"),
    "wsdan_e": WeightFile(_DRIVE + "1OrHdRwpgBMi9lZJSsZu66bHWq5a-fI3W",
                          "74f44789b3118389a11c87659f0c19caa3c1fdafc0e41ab4593d86a72531d847", "dfdc-wm/ckpt_e.pth"),
    "retinaface": WeightFile(_DRIVE + "1lEdFpC4_6qch3YtmSuGB0vPxQOmebsor",
                             "ee23d8552ec244cd5efdeb1b0ae00f95a89cf0be806f7665e2b26703cff7a296",
                             "dfdc-wm/RetinaFace-Resnet50-fixed.pth"),
}

FRAME_SKIP = 9  # grab 9 frames, then decode 1
FACE_LIMIT = 25
FACE_SIZE = 320
_WSDAN_MEAN = torch.tensor([0.4479, 0.3744, 0.3473]).view(1, 3, 1, 1)
_WSDAN_STD = torch.tensor([0.2537, 0.2502, 0.2424]).view(1, 3, 1, 1)


def read_faces(video: VideoInput, detector) -> list[np.ndarray]:
    """Largest face in every 10th frame, aligned to 320x320 RGB, up to 25 faces."""
    import cv2

    from forensix.video.retinaface import norm_crop

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Can't read {video}")
    faces = []
    try:
        while len(faces) < FACE_LIMIT:
            for _ in range(FRAME_SKIP):
                capture.grab()
            ok, frame = capture.read()
            if not ok:
                break
            boxes, landmarks = detector.detect(frame)
            if boxes.shape[0] == 0:
                continue
            areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            best = int(areas.argmax())
            points = landmarks[best].numpy().reshape(5, 2).astype(int)  # upstream truncates to int
            faces.append(norm_crop(frame, points, FACE_SIZE)[:, :, ::-1].copy())  # BGR -> RGB
    finally:
        capture.release()
    return faces


class DFDCWMDetector(VideoDetector):
    def __init__(self, xception, wsdan_x, wsdan_e, face_detector, device="cpu", chunk_size: int = 8):
        super().__init__("dfdc-wm")
        self.device = torch.device(device)
        self.xception = xception.to(self.device).eval()
        self.wsdan_x = wsdan_x.to(self.device).eval()
        self.wsdan_e = wsdan_e.to(self.device).eval()
        self.face_detector = face_detector
        self.chunk_size = chunk_size

    @torch.inference_mode()
    def face_fake_probs(self, faces: list[np.ndarray]) -> np.ndarray:
        """Ensemble fake probability for aligned 320x320 RGB faces."""
        out = []
        for i in range(0, len(faces), self.chunk_size):
            batch = torch.from_numpy(np.stack(faces[i : i + self.chunk_size])).permute(0, 3, 1, 2)
            batch = (batch.float() / 255.0).to(self.device)  # torchvision ToTensor
            i1 = F.interpolate(batch, size=299, mode="bilinear").sub_(0.5).mul_(2.0)
            o1 = self.xception(i1).softmax(-1)[:, 1]
            i2 = (batch - _WSDAN_MEAN.to(self.device)) / _WSDAN_STD.to(self.device)
            o2 = self.wsdan_x(i2)[0].softmax(-1)[:, 1]
            i3 = F.interpolate(i2, size=300, mode="bilinear")
            o3 = self.wsdan_e(i3)[0].softmax(-1)[:, 1]
            out.append((0.2 * o1 + 0.7 * o2 + 0.1 * o3).cpu())
        return torch.cat(out).numpy()

    def _score(self, video: VideoInput) -> tuple[float, list[float]]:
        faces = read_faces(video, self.face_detector)
        if not faces:
            return 0.5, []  # upstream default score
        probs = self.face_fake_probs(faces)
        return float(probs.mean()), probs.tolist()


def build(weights_path: Path, device: str = "cpu") -> DFDCWMDetector:
    from forensix.video.dfdc_wm.wsdan import WSDAN
    from forensix.video.dfdc_wm.xception import xception
    from forensix.video.retinaface import RetinaFaceDetector

    model1 = xception(num_classes=2, pretrained=False)
    model1.load_state_dict(torch_load(weights_path)["state_dict"])
    model2 = WSDAN(num_classes=2, M=8, net="xception", pretrained=False)
    model2.load_state_dict(torch_load(fetch(WEIGHTS["wsdan_x"]))["state_dict"])
    model3 = WSDAN(num_classes=2, M=8, net="efficientnet", pretrained=False)
    model3.load_state_dict(torch_load(fetch(WEIGHTS["wsdan_e"]))["state_dict"])
    detector = RetinaFaceDetector(fetch(WEIGHTS["retinaface"]), device=device)
    return DFDCWMDetector(model1, model2, model3, detector, device=device)


register(
    ModelSpec(
        name="dfdc-wm",
        family="video",
        builder=build,
        weights=WEIGHTS["xception"],
        extra_weights=(WEIGHTS["wsdan_x"], WEIGHTS["wsdan_e"], WEIGHTS["retinaface"]),
        license="MIT (code, weights; Team WM). Xception port BSD-3-Clause (Cadene), WS-DAN MIT, "
        "EfficientNet-PyTorch (vendored) Apache-2.0, RetinaFace MIT (biubug6); weights hosted on the "
        "authors' Google Drive",
        source="https://github.com/cuihaoleo/kaggle-dfdc/tree/91dae24a31caf6a3ca273e2b5d7337b9fe6f52d5",
        paper="https://arxiv.org/abs/2006.07397",  # the challenge paper; the solution has a model summary PDF
        description="2nd place, Deepfake Detection Challenge (Kaggle, 2020): WS-DAN attention "
        "models on Xception and EfficientNet-B3 plus a plain Xception, on RetinaFace-aligned "
        "faces from every 10th frame (up to 25). Returns 0.5 if no face is found.",
    )
)
