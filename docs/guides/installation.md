# Installation

## Requirements

| Requirement | Notes |
|---|---|
| Python ≥ 3.10 | Tested on 3.13 |
| PyTorch ≥ 2.0 | CPU works for everything; a CUDA GPU is strongly recommended for video, audiovisual and XLS-R audio models |
| ffmpeg on `PATH` | Required for audiovisual models, and for audio in formats libsndfile can't read (m4a, aac, audio inside video files) |
| Disk | About 18 GB for every model's weights; much less if you load only some |
| RAM | 8 GB is enough for one model at a time; audiovisual models need more on long or high-resolution videos ([input limits](../deployment.md#input-limits)) |

## Install

```bash
git clone https://github.com/anas-rz/ForensiX.git
cd ForensiX
pip install -e ".[dev]"            # dev adds pytest
```

On a GPU machine, install the CUDA build of PyTorch first, following
[pytorch.org](https://pytorch.org/get-started/locally/), so pip doesn't pull in a CPU-only build.

ffmpeg:

```bash
sudo apt-get install ffmpeg        # Debian/Ubuntu
brew install ffmpeg                # macOS
```

The `eval` extra (`pip install -e ".[eval]"`) is only needed to build the image evaluation set
with diffusion models.

## Check it works

```python
import torch, forensix

print(torch.cuda.is_available())
print(forensix.list_models())

pred = forensix.load("aasist").predict("tests/data/speech_cc0.flac")
print(pred)            # a small model; downloads 1.3 MB on first use
```

Run the test suite with `pytest tests`. Some tests download model weights on their first run,
and tests that need ffmpeg are skipped without it.

## The weights cache

Weights download on first use into `~/.cache/forensix/weights/`, and their SHA-256 hashes are
checked. A `.verified` stamp next to each file lets later loads skip rehashing it while the file
is unchanged (`FORENSIX_VERIFY_HASH=always` forces a full check). To put the cache somewhere else
(a bigger disk, a shared volume, Google Drive):

```bash
export FORENSIX_CACHE=/data/forensix_cache
```

Pre-fetch weights instead of downloading on first use:

```bash
python scripts/download_weights.py                    # every model
python scripts/download_weights.py gend-pe syncnet    # some models
python scripts/download_weights.py --family video     # one family
```

## Offline and air-gapped machines

Some models fetch helper files the first time they *run*, not when they load: face detectors
and landmark models (RetinaFace, YuNet, MediaPipe), and the ICT-FaceKit meshes used by
`phys-lighting`. All of them go through the same verified cache, so the reliable way to fill a
cache for an offline machine is to load *and run* every model you need once, on a connected
machine:

```python
# warm_cache.py: run on a connected machine with FORENSIX_CACHE set
import forensix

AUDIO = "tests/data/speech_cc0.flac"
IMAGE = "tests/data/face_public_domain.jpg"
VIDEO = "sample_talking_head.mp4"     # any short video of a face, with sound

needs_your_weights = {"avcons", "avid", "id-motion", "person-detector"}
for name in forensix.list_models():
    if name in needs_your_weights:
        continue
    family = forensix.info(name).family
    detector = forensix.load(name)
    if family == "identity":
        detector.enroll("warmup", [IMAGE])
        detector.predict(IMAGE, identity="warmup")
    elif family == "audio" or name in {"phys-voice", "phys-room"}:
        detector.predict(AUDIO)
    elif family == "image":
        detector.predict(IMAGE)
    else:
        detector.predict(VIDEO)
    print("ok", name)
```

Then copy the whole `$FORENSIX_CACHE` directory to the offline machine and set
`FORENSIX_CACHE` there. Before you depend on it, test with networking disabled. For models you
train, ship the checkpoint file and pass `weights_path=`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `RuntimeError: Checksum mismatch` | The download was corrupted, or the upstream file changed. Delete the file and retry. If it persists, the upstream host changed the file; open an issue rather than editing the hash. |
| `ValueError: Can't decode ... ffmpeg isn't installed` | Install ffmpeg. |
| `ValueError: ... needs both a video and an audio stream` | Audiovisual models need a video with sound. |
| `person-detector requires a trained checkpoint` (and similar for `avcons`, `avid`, `id-motion`) | These models ship without weights. Train them ([Training](training.md)) and pass `weights_path=`. |
| A video scores exactly 0.5 | No face was found. Check `pred.segment_fake_probs` (empty when no face was found). |
| OpenCV prints `setPreferableTarget Targets are not supported` | Harmless warning from OpenCV 5's DNN module (used by `id-face`). |
| MediaPipe / TensorFlow Lite log lines | Harmless; set `GLOG_minloglevel=2` to hide them. |
| CUDA out of memory | Load one large model at a time, `del` it before the next, and reduce `batch_size` for image and audio models. |
