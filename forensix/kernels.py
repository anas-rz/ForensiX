"""GPU preprocessing kernels for high-throughput inference.

``to_normalized_chw`` turns a batch of uint8 RGB images (B, H, W, 3), already
cropped on the CPU, into the float32 (B, 3, H, W) tensor that torchvision's
``ToTensor()`` + ``Normalize(mean, std)`` produce, in one pass on the GPU. Moving
uint8 to the GPU and converting there copies 4x fewer bytes than float32 and
takes the conversion off the CPU workers.

The Triton kernel uses IEEE round-to-nearest division (``div_rn``) in the same
order as torchvision (``x / 255``, ``- mean``, ``/ std``), so its output is
bit-identical to the torchvision path; ``tests/test_batch.py`` checks this.
Without Triton or CUDA, the same arithmetic runs as PyTorch ops.

``center_crop_u8`` is torchvision's ``CenterCrop`` (offsets, rounding and zero
padding) on a uint8 HWC array, so crop + kernel == the PIL/torchvision path.
"""

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs
    triton = None


def center_crop_u8(img: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    """torchvision ``CenterCrop(size)`` on an (H, W, 3) uint8 array (pads with 0)."""
    ch, cw = (size, size) if isinstance(size, int) else size
    h, w = img.shape[:2]
    if cw > w or ch > h:
        left, top = max(cw - w, 0) // 2, max(ch - h, 0) // 2
        right, bottom = max(cw - w + 1, 0) // 2, max(ch - h + 1, 0) // 2
        img = np.pad(img, ((top, bottom), (left, right), (0, 0)))
        h, w = img.shape[:2]
        if (h, w) == (ch, cw):
            return img
    top, left = int(round((h - ch) / 2.0)), int(round((w - cw) / 2.0))
    return np.ascontiguousarray(img[top : top + ch, left : left + cw])


if triton is not None:

    @triton.jit
    def _u8_hwc_to_chw_kernel(src, dst, mean, std, hw, BLOCK: tl.constexpr):
        b = tl.program_id(1)
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offs < hw
        for c in tl.static_range(3):
            x = tl.load(src + b * hw * 3 + offs * 3 + c, mask=valid).to(tl.float32)
            x = tl.math.div_rn(x, 255.0)
            x = tl.math.div_rn(x - tl.load(mean + c), tl.load(std + c))
            tl.store(dst + b * 3 * hw + c * hw + offs, x, mask=valid)


def _torch_impl(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    x = x.permute(0, 3, 1, 2).to(torch.float32).div(255)
    return x.sub(mean.view(1, 3, 1, 1)).div(std.view(1, 3, 1, 1))


def to_normalized_chw(images: torch.Tensor, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)) -> torch.Tensor:
    """(B, H, W, 3) uint8 -> (B, 3, H, W) float32 ``(x / 255 - mean) / std``."""
    if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("Expected a (B, H, W, 3) uint8 tensor")
    mean_t = torch.tensor(mean, dtype=torch.float32, device=images.device)
    std_t = torch.tensor(std, dtype=torch.float32, device=images.device)
    if triton is None or images.device.type != "cuda":
        return _torch_impl(images, mean_t, std_t)
    images = images.contiguous()
    b, h, w, _ = images.shape
    out = torch.empty((b, 3, h, w), dtype=torch.float32, device=images.device)
    block = 1024
    _u8_hwc_to_chw_kernel[(triton.cdiv(h * w, block), b)](images, out, mean_t, std_t, h * w, BLOCK=block)
    return out
