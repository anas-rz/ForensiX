"""High-throughput batch inference: CPU workers decode, the GPU scores big batches.

``BatchRunner`` runs one detector over many inputs:

- **CPU workers** (``num_workers`` processes) decode and preprocess inputs in
  parallel, ahead of the GPU.
- **Cross-input batching**: the units of many inputs (frames, audio windows,
  faces) are packed into GPU batches of ``batch_size``, then their scores are
  put back together per input exactly as the detector does (mean over frames,
  windows or faces).
- **Pinned, asynchronous copies**; for crop-and-normalize image models, uint8
  crops go to the GPU and ``forensix.kernels`` normalizes them there (4x
  fewer bytes than float32, bit-identical to torchvision).
- **Precision**: ``"fp32"`` (default; matches ``predict`` up to batch-size
  dependent rounding, ~1e-6), or ``"bf16"`` / ``"fp16"`` autocast (faster on
  tensor-core GPUs; check the score drift on your data with
  ``scripts/batch_infer.py --compare-fp32``).

Supported directly: image detectors (on image files, or on videos as the mean
over ``frames`` evenly spaced frames, like ``scripts/eval_video_folder.py``),
audio detectors, and the GenD face pipeline (``gend``, ``gend-pe``,
``dfdet-clip``; ``fast_faces=True`` batches MTCNN over a video's frames, with a
small score drift).
Any other detector runs through ``predict`` one input at a time (on GPU), or
in a process pool for CPU detectors (physics), via ``parallel_predict``.
Scale out across GPUs by sharding inputs, one process per GPU
(``scripts/batch_infer.py --gpus 0,1,...``).

    runner = BatchRunner(forensix.load("univfd", device="cuda"), batch_size=64, precision="bf16")
    for path, pred, error in runner.run(paths):
        ...
"""

import contextlib
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch

from forensix.types import Prediction

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
_PRECISIONS = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def _fusable(transform) -> tuple[int, tuple, tuple] | None:
    """(crop, mean, std) if ``transform`` is CenterCrop -> ToTensor [-> Normalize]."""
    from torchvision import transforms as T

    steps = getattr(transform, "transforms", None)
    if not steps or not 2 <= len(steps) <= 3:
        return None
    crop, to_tensor, *norm = steps
    if not (isinstance(crop, T.CenterCrop) and isinstance(to_tensor, T.ToTensor)):
        return None
    if norm and not isinstance(norm[0], T.Normalize):
        return None
    size = crop.size if isinstance(crop.size, int) else tuple(crop.size)
    mean, std = (tuple(norm[0].mean), tuple(norm[0].std)) if norm else ((0.0,) * 3, (1.0,) * 3)
    return size, mean, std


class _Adapter:
    """How the runner splits a detector's work. ``prepare`` runs in CPU workers
    (no CUDA); ``expand``, ``forward`` and ``reduce`` run in the main process."""

    empty_score: float | None = None  # score of an input with no units (None: error)

    def prepare(self, item):
        raise NotImplementedError

    def expand(self, payload) -> list:
        return list(payload)

    def forward(self, units: list) -> np.ndarray:
        raise NotImplementedError

    def reduce(self, probs: np.ndarray) -> tuple[float, tuple]:
        return float(probs.mean()), ()


class _ImageAdapter(_Adapter):
    def __init__(self, det, frames: int):
        self.det, self.frames = det, frames
        self.fused = _fusable(det.transform)

    def _images(self, item) -> list[np.ndarray]:
        from forensix.image.io import load_image

        if Path(str(item)).suffix.lower() in VIDEO_EXTS:
            from forensix.video.io import read_frames

            return list(read_frames(item, self.frames)[0])
        return [np.asarray(load_image(item))]

    def prepare(self, item):
        from PIL import Image

        from forensix.kernels import center_crop_u8

        images = self._images(item)
        if self.fused:
            return [center_crop_u8(im, self.fused[0]) for im in images]
        return [self.det.transform(Image.fromarray(im)).numpy() for im in images]

    def forward(self, units):
        from forensix.kernels import to_normalized_chw

        batch = torch.from_numpy(np.stack(units))
        if self.det.device.type == "cuda":
            batch = batch.pin_memory().to(self.det.device, non_blocking=True)
        if self.fused:
            batch = to_normalized_chw(batch, *self.fused[1:])
        return self.det._fake_probs(batch).float().cpu().numpy()

    def reduce(self, probs):
        # an image file has one unit; a video is the mean over its frames
        return float(probs.mean()), (tuple(float(p) for p in probs) if len(probs) > 1 else ())


class _AudioAdapter(_Adapter):
    def __init__(self, det):
        self.det = det

    def prepare(self, item):
        from forensix.audio.detector import split_windows
        from forensix.audio.io import load_audio

        return list(split_windows(load_audio(item, self.det.sample_rate), self.det.window, self.det.pad_short))

    def forward(self, units):
        batch = torch.from_numpy(np.stack(units))
        if self.det.device.type == "cuda":
            batch = batch.pin_memory().to(self.det.device, non_blocking=True)
        return self.det._fake_probs(batch).float().cpu().numpy()

    def reduce(self, probs):
        return float(probs.mean()), tuple(float(p) for p in probs)


class _GenDAdapter(_Adapter):
    """GenD face pipeline (``gend``, ``gend-pe``, ``dfdet-clip``): workers decode,
    the main process finds and aligns faces with GenDDetector's own per-frame
    MTCNN, and faces from many videos are classified together. Per-frame MTCNN is
    many small GPU kernels, so on one GPU it is the bottleneck (more processes
    don't help: one GPU time-slices them). ``fast_faces`` runs MTCNN on all of a
    video's frames in one batch: ~2x faster, but MTCNN's batched path moves
    landmarks by up to a few pixels, so scores drift by ~1e-2."""

    empty_score = 0.5  # GenDDetector: no face found

    def __init__(self, det, fast_faces: bool = False):
        from forensix.video import gend

        self.det, self.gend, self.fast_faces = det, gend, fast_faces

    def prepare(self, item):
        from forensix.video.io import read_frames

        return read_frames(item, self.gend.FRAMES_PER_VIDEO)[0]

    def expand(self, frames):
        import cv2
        from PIL import Image

        from forensix.video.faces import align_face

        if not self.fast_faces:
            faces = self.gend.aligned_faces(frames, self.det.face_detector)
        else:
            factor = min(1.0, self.gend.DETECT_MAX_SIDE / max(frames.shape[1:3]))
            small = [Image.fromarray(f if factor == 1.0 else cv2.resize(f, None, fx=factor, fy=factor,
                                                                         interpolation=cv2.INTER_AREA)) for f in frames]
            boxes, _, landmarks = self.det.face_detector.detect(small, landmarks=True)
            faces = []
            for frame, b, lm in zip(frames, boxes, landmarks):
                if b is not None and len(b):
                    largest = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
                    faces.append(align_face(frame, lm[largest] / factor))
        return [self.det.preprocess([f])[0].numpy() for f in faces]

    def forward(self, units):
        pixels = torch.from_numpy(np.stack(units)).pin_memory().to(self.det.device, non_blocking=True)
        return self.det.network(pixels).float().softmax(dim=-1)[:, 1].cpu().numpy()

    def reduce(self, probs):
        return float(probs.mean()), tuple(float(p) for p in probs)


def adapter_for(detector, frames: int = 8, fast_faces: bool = False) -> _Adapter | None:
    """The batching adapter for ``detector``, or None if it must run through ``predict``."""
    from forensix.audio.detector import AudioDetector
    from forensix.image.detector import ImageDetector
    from forensix.video.gend import GenDDetector

    if isinstance(detector, ImageDetector):
        return _ImageAdapter(detector, frames)
    if isinstance(detector, AudioDetector):
        return _AudioAdapter(detector)
    if isinstance(detector, GenDDetector):
        return _GenDAdapter(detector, fast_faces)
    return None


class _Prepared(torch.utils.data.Dataset):
    def __init__(self, items, prepare):
        self.items, self.prepare = items, prepare

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        try:
            return i, self.prepare(self.items[i]), None
        except Exception as e:  # noqa: BLE001  (undecodable media: report, keep going)
            return i, None, f"{type(e).__name__}: {e}"


def _identity(x):
    return x


class BatchRunner:
    """Score many inputs with one detector (see the module docstring)."""

    def __init__(self, detector, batch_size: int = 64, num_workers: int = 4, precision: str = "fp32",
                 frames: int = 8, prefetch: int = 4, fast_faces: bool = False):
        if precision not in _PRECISIONS:
            raise ValueError(f"precision must be one of {list(_PRECISIONS)}")
        self.detector, self.batch_size, self.num_workers = detector, batch_size, num_workers
        self.precision, self.prefetch = precision, prefetch
        self.adapter = adapter_for(detector, frames, fast_faces)

    def _autocast(self):
        dtype = _PRECISIONS[self.precision]
        device = getattr(self.detector, "device", torch.device("cpu"))
        if dtype is None or device.type != "cuda":
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=dtype)

    def _prediction(self, score: float, segments: tuple) -> Prediction:
        return Prediction(model=self.detector.name, fake_prob=score,
                          threshold=getattr(self.detector, "threshold", 0.5), segment_fake_probs=segments)

    def run(self, inputs: Sequence) -> Iterator[tuple[object, Prediction | None, str | None]]:
        """Yield ``(input, prediction, error)`` as inputs finish (not in input order)."""
        inputs = list(inputs)
        if self.adapter is None:
            for item in inputs:
                try:
                    yield item, self.detector.predict(item), None
                except Exception as e:  # noqa: BLE001
                    yield item, None, f"{type(e).__name__}: {e}"
            return
        loader = torch.utils.data.DataLoader(
            _Prepared(inputs, self.adapter.prepare), batch_size=None, shuffle=False,
            num_workers=self.num_workers, collate_fn=_identity,
            prefetch_factor=self.prefetch if self.num_workers else None,
            persistent_workers=False)
        units, owners, need, probs = [], [], {}, {}

        def flush(n):
            batch, who = units[:n], owners[:n]
            del units[:n], owners[:n]
            with torch.inference_mode(), self._autocast():
                out = self.adapter.forward(batch)
            done = []
            for i, p in zip(who, out):
                probs[i].append(float(p))
                if len(probs[i]) == need[i]:
                    done.append(i)
            return done

        def finished(i):
            score, segments = self.adapter.reduce(np.array(probs.pop(i)))
            del need[i]
            return inputs[i], self._prediction(score, segments), None

        for i, payload, error in loader:
            if error is not None:
                yield inputs[i], None, error
                continue
            try:
                with torch.inference_mode():
                    item_units = self.adapter.expand(payload)
            except Exception as e:  # noqa: BLE001
                yield inputs[i], None, f"{type(e).__name__}: {e}"
                continue
            if not item_units:
                if self.adapter.empty_score is None:
                    yield inputs[i], None, "nothing to score"
                else:
                    yield inputs[i], self._prediction(self.adapter.empty_score, ()), None
                continue
            need[i], probs[i] = len(item_units), []
            units += item_units
            owners += [i] * len(item_units)
            while len(units) >= self.batch_size:
                for j in flush(self.batch_size):
                    yield finished(j)
        while units:
            for j in flush(min(len(units), self.batch_size)):
                yield finished(j)


def _pool_init(name, device):
    global _POOL_DETECTOR
    import forensix

    _POOL_DETECTOR = forensix.load(name, device=device)


def _pool_predict(item):
    try:
        return item, _POOL_DETECTOR.predict(item), None
    except Exception as e:  # noqa: BLE001
        return item, None, f"{type(e).__name__}: {e}"


def parallel_predict(name: str, inputs: Sequence, processes: int, device: str = "cpu"):
    """``predict`` over ``inputs`` in a process pool, one detector per process.
    For CPU-bound detectors without a batching adapter (e.g. the physics family)."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")  # each process builds its own detector (and CUDA context)
    with ctx.Pool(processes, initializer=_pool_init, initargs=(name, device)) as pool:
        yield from pool.imap_unordered(_pool_predict, inputs)
