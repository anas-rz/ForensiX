# Concepts

ForensiX has four building blocks: a **registry** of models, a **loader** that builds detectors
with verified weights, **detectors** that share one calling convention per input type, and the
**`Prediction`** every detector returns.

```text
forensix.load("gend-pe", device="cuda")
   │
   ├─ registry.get_spec("gend-pe")          → ModelSpec (family, license, weights URL + SHA-256, builder)
   ├─ hub.fetch(spec.weights)                → ~/.cache/forensix/weights/gend-pe-l.safetensors (hash-checked)
   └─ spec.builder(path, device="cuda")      → detector object
                                                 │
detector.predict("clip.mp4")  ───────────────────┘→ Prediction(model, fake_prob, threshold, segment_fake_probs,
                                                               score_type, details)
```

## Registry and `ModelSpec`

Each model module calls `register(ModelSpec(...))` when `forensix` is imported. A spec holds:

| Field | Meaning |
|---|---|
| `name` | The string you pass to `forensix.load`, e.g. `"gend-pe"` |
| `family` | `audio`, `image`, `video`, `audiovisual`, `identity`, `physics` or `specialized` |
| `builder` | `(weights_path, device) -> detector` |
| `weights` | A `WeightFile(url, sha256, filename)`, or `None` for models that need no weights or that you train yourself |
| `extra_weights` | Shared files the builder fetches itself, e.g. a CLIP backbone used by several models |
| `license`, `source`, `paper`, `description` | Provenance. Read `license` before you ship a model ([Licensing](guides/licensing.md)) |

```python
import forensix
forensix.list_models()            # every registered name
forensix.list_models("video")     # one family
forensix.info("gend-pe").license  # provenance for one model
```

## Weights and the cache

`forensix.load(name)` calls `forensix.download(name)`, which fetches every file in the spec into
the cache and checks its SHA-256 hash. A file whose hash doesn't match is deleted and downloaded
again, and a download whose hash doesn't match raises an error. Nothing unverified is loaded.

- **Location:** `$FORENSIX_CACHE` if set, otherwise `~/.cache/forensix`. Weights live under
  `weights/`.
- **Size:** about 18 GB for every model. The large ones are the CLIP/PE-based models
  (1.2–1.6 GB each), the XLS-R audio models (1.2 GB each), `stablesyncnet` (1.5 GB) and
  `dfdc-selim` (1.8 GB).
- **Hashing cost:** hashing a 1.6 GB file takes a few seconds. After a successful check,
  `<file>.verified` records the hash, size and modification time, and later loads trust an
  unchanged file without rehashing it. A changed size or mtime triggers a full check. Set
  `FORENSIX_VERIFY_HASH=always` to rehash on every load.
- **Load once:** building a model still takes seconds (library imports, initialization), so
  load once per process and reuse the detector.
- **Your own checkpoints:** `forensix.load(name, weights_path="my.safetensors")` skips the
  download and hash check for the model's own weights. Shared backbones are still fetched.

Checkpoints are read with `torch.load(weights_only=True)` or safetensors, so loading weights
never runs pickled code.

## Detectors by input type

Detectors of one input type share a calling convention:

| Family | Input to `predict` | Base class | Batch call |
|---|---|---|---|
| audio | file path (any format libsndfile or ffmpeg reads, including video containers), or a waveform array plus `sample_rate=` | `forensix.audio.detector.AudioDetector` | `predict_batch(inputs, sample_rate=None, batch_size=16)` |
| image | file path, PIL image, or `(H, W, 3)` uint8 RGB array | `forensix.image.detector.ImageDetector` | `predict_batch(images, batch_size=16)` |
| video | video file path | `forensix.video.detector.VideoDetector` | `predict_batch(videos)` (a loop) |
| audiovisual | video file path with an audio track (needs ffmpeg) | none | `predict_batch(videos)` (a loop) |
| physics | video path, audio path, or both (`physdf`) | `forensix.physics.base` | `predict_batch(inputs)` |
| identity | media path plus `identity=` (after `enroll`) | none | none |
| specialized | media path or array | `PersonDetector` | none |

Only the audio and image detectors truly batch on the GPU. Video, audiovisual and physics
`predict_batch` calls are loops over `predict`, so for throughput, run several worker processes
([Deployment](deployment.md)).

### What each type does with your media

- **Audio:** converted to mono and resampled to 16 kHz, then cut into the model's window
  (4.04 s for AASIST-family models). Short clips are repeat-padded. Long clips are cut into
  consecutive windows, with the last one aligned to the end. The clip score is the mean window
  score.
- **Image:** never resized or re-encoded before the model's own upstream preprocessing,
  because the traces these detectors look for don't survive resampling. Several models score
  only a central crop (224 or 256 px) of a large image.
- **Video:** each model uses its upstream frame sampling and face pipeline: typically 32 evenly
  spaced frames, face detection (MTCNN or RetinaFace), alignment and per-face scores. Frames are
  decoded sequentially, so time grows with video length, but memory doesn't.
- **Audiovisual:** the whole video is decoded at 25 fps and full resolution into memory, plus
  16 kHz audio. Face tracks are built syncnet_python-style. **Memory grows with resolution ×
  duration**: downscale and trim long inputs first ([Deployment](deployment.md#input-limits)).
- **Physics:** every frame of the first 30 s, downscaled to a 640 px long side, because
  physiological signals need every frame.

## `Prediction`

Every detector returns a frozen `forensix.Prediction`:

| Field | Meaning |
|---|---|
| `model` | Model name |
| `fake_prob` | The score. **Higher always means more likely fake.** |
| `threshold` | The model's default decision threshold, or `None` when the score has no natural one |
| `score_type` | `"probability"` (a softmax/sigmoid output in [0, 1]) or `"anomaly"` (distance from real behaviour; may be unbounded or negative) |
| `segment_fake_probs` | Per-segment scores: audio windows, faces, frames or AV windows, depending on the model |
| `details` | Structured evidence: per-model measurements, timelines, reports |
| `is_fake`, `label` | `fake_prob >= threshold` and `"fake"`/`"real"`; `None` when `threshold` is `None` |

Two rules follow, and they cause most integration mistakes:

1. **Scores are not comparable across models.** A `fake_prob` of 0.4 from `rine` and 0.4 from
   `gend` mean different things. Compare models by AUC/EER, and choose thresholds per model on
   data like yours ([Interpreting scores](guides/interpreting-scores.md)).
2. **The default threshold is often wrong for your data.** On the SDFVD benchmark, `rine` scored
   every video below 0.5 (it called all fakes real) despite an AUC of 0.745.

## Families and their evidence

| Family | Question it answers | Needs |
|---|---|---|
| audio | Is this speech synthetic or converted? | Speech |
| image | Is this image generated by a GAN or diffusion model? | A whole generated image (not a face swap in a real photo) |
| video | Is this face manipulated (swap, reenactment)? | A visible face |
| audiovisual | Do the voice and the lips belong together? | A talking face with its audio track |
| physics | Does the media obey physics (pulse, head motion, lighting, voice, room)? | A few seconds of face or speech |
| identity | Is this really person X? | Reference media of X |
| specialized | Is this real media of person X? (a tiny model trained only for X) | Real media of X to train on |

No single family covers every attack. A face swap keeps in-sync lips and real physics in the
untouched parts of the frame, while a voice clone has no face to check. See
[Choosing detectors](guides/choosing-detectors.md).
