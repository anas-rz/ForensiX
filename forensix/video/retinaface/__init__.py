"""RetinaFace (ResNet-50) face detector from biubug6/Pytorch_Retinaface (MIT).

``RetinaFaceDetector`` follows the DFDC 2nd-place (Team WM) wrapper in
face_utils.py (MIT): BGR input with mean subtraction, confidence threshold 0.8,
and by default no NMS (callers pick the largest box).
"""

from pathlib import Path

import numpy as np
import torch


class RetinaFaceDetector:
    def __init__(
        self,
        weights_path: Path,
        device: str | torch.device = "cpu",
        confidence_threshold: float = 0.8,
        nms_threshold: float | None = None,
    ):
        from forensix.hub import torch_load
        from forensix.video.retinaface.config import cfg_re50
        from forensix.video.retinaface.retinaface import RetinaFace

        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold  # None: keep all boxes (Team WM's wrapper)
        self.cfg = dict(cfg_re50, pretrain=False)
        self.net = RetinaFace(cfg=self.cfg, phase="test")
        self.net.load_state_dict(torch_load(weights_path))
        self.net.to(self.device).eval()
        self._priors = {}

    def _decode_params(self, height: int, width: int):
        from forensix.video.retinaface.prior_box import PriorBox

        key = (height, width)
        if key not in self._priors:
            priors = PriorBox(self.cfg, image_size=(height, width)).forward()
            self._priors[key] = (priors, torch.Tensor([width, height] * 2), torch.Tensor([width, height] * 5))
        return self._priors[key]

    @torch.inference_mode()
    def detect(self, img_bgr: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Boxes ``(n, 4)`` and landmarks ``(n, 10)`` above the threshold, in pixels."""
        from forensix.video.retinaface.box_utils import decode, decode_landm

        priors, scale, scale1 = self._decode_params(*img_bgr.shape[:2])
        x = np.float32(img_bgr) - (104, 117, 123)
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device, dtype=torch.float32)
        loc, conf, landms = self.net(x)
        variance = self.cfg["variance"]
        boxes = decode(loc.squeeze(0).cpu(), priors, variance) * scale
        scores = conf.squeeze(0).cpu()[:, 1]
        landms = decode_landm(landms.squeeze(0).cpu(), priors, variance) * scale1
        keep = scores > self.confidence_threshold
        boxes, landms, scores = boxes[keep], landms[keep], scores[keep]
        if self.nms_threshold is not None and len(boxes):
            from torchvision.ops import nms

            keep = nms(boxes, scores, self.nms_threshold)
            boxes, landms = boxes[keep], landms[keep]
        return boxes, landms


# ArcFace 5-point template for a 320x320 crop (Team WM's norm_crop)
ARCFACE_320 = np.array(
    [[122.5, 141.25], [197.5, 141.25], [160.0, 178.75], [137.5, 225.25], [182.5, 225.25]], dtype=np.float64
)


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity (rotation, uniform scale, translation) mapping
    ``src`` points onto ``dst`` (Umeyama, 1991). Returns a 2x3 matrix."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    cov = dst_c.T @ src_c / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[1] = -1
    rotation = u @ np.diag(d) @ vt
    scale = (s * d).sum() / src_c.var(0).sum()
    matrix = np.zeros((2, 3))
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = dst_mean - scale * rotation @ src_mean
    return matrix


def norm_crop(img: np.ndarray, landmarks: np.ndarray, size: int = 320) -> np.ndarray:
    """Warp a face onto the ArcFace template (Team WM's norm_crop, size 320)."""
    import cv2

    matrix = similarity_transform(landmarks, ARCFACE_320 * size / 320)
    return cv2.warpAffine(img, matrix, (size, size), borderValue=0.0)
