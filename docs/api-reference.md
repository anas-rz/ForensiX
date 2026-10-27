# API reference

This covers the public Python API. Anything not listed here is internal and may change.

- [Top level: `forensix`](#top-level-forensix)
- [`Prediction`](#prediction)
- [Detector interfaces](#detector-interfaces)
- [Identity detectors](#identity-detectors)
- [Physics detectors](#physics-detectors)
- [Person-specialized models: `forensix.specialized`](#person-specialized-models-forensixspecialized)
- [Metrics: `forensix.metrics`](#metrics-forensixmetrics)
- [Media helpers](#media-helpers)
- [Registry and weights](#registry-and-weights)
- [Command-line scripts](#command-line-scripts)

## Top level: `forensix`

```python
import forensix
```

| Function | Returns | Description |
|---|---|---|
| `load(name, device="cpu", weights_path=None)` | detector | Builds a detector, downloading and hash-checking its weights first. `device` is `"cpu"`, `"cuda"` or `"cuda:N"`. `weights_path` uses a local checkpoint instead of the registered one (required for models you train: `avcons`, `avid`, `id-motion`, `person-detector`; optional for `physdf`). |
| `list_models(family=None)` | `list[str]` | Registered model names, sorted, optionally filtered by family. |
| `info(name)` | `ModelSpec` | Family, license, source, paper, description and weight file of a model. |
| `download(name, force=False)` | `Path \| None` | Fetches the model's weights and shared backbones into the cache, without building the model. `None` for models without registered weights. |
| `cache_dir()` | `Path` | `$FORENSIX_CACHE` or `~/.cache/forensix`. |
| `forensix.specialized` (alias `forensix.person`) | module | [Person-specialized models](#person-specialized-models-forensixspecialized). |

Unknown names raise `KeyError` with the list of available models.

## `Prediction`

`forensix.Prediction` is a frozen dataclass (`forensix/types.py`).

| Attribute | Type | Description |
|---|---|---|
| `model` | `str` | The detector's name. |
| `fake_prob` | `float` | Score; higher means more likely fake. Not comparable across models. |
| `threshold` | `float \| None` | Default decision threshold; `None` for scores without one. |
| `segment_fake_probs` | `tuple[float, ...]` | Per-segment scores (audio windows, faces, frames or AV windows). May be empty. |
| `score_type` | `str` | `"probability"` or `"anomaly"`. |
| `details` | `dict` | Model-specific evidence. Excluded from equality comparison. |
| `is_fake` | `bool \| None` | `fake_prob >= threshold`, or `None` without a threshold. |
| `label` | `str \| None` | `"fake"`, `"real"` or `None`. |

Models and their score conventions:

| Models | `score_type` | `threshold` | `segment_fake_probs` | Notable `details` |
|---|---|---|---|---|
| audio models | probability | 0.5 | one per audio window | none |
| image models | probability | 0.5 | none | none |
| `gend`, `gend-pe`, `dfdet-clip`, `dfdc-selim`, `dfdc-wm` | probability | 0.5 | one per scored face | none |
| `avad` | anomaly | `None` | none | none |
| `syncnet` | anomaly (`-confidence`) | `None` | none | `offset_frames`, `confidence`, `min_dist`, `mean_dist_by_offset`, `tracks` |
| `stablesyncnet` | probability (1 − sync probability) | 0.5 | none | offset, sync probability per offset and per window |
| `avcons` | probability | 0.5 | one per AV window | inconsistent segments (s), per-track results |
| `avid` | probability | 0.5 | none | features, per-segment voice–face match, tracks |
| `id-face` | anomaly (1 − cosine similarity) | 0.637 | one per sampled frame | `similarity`, `frame_similarity`, `frame_time_s`, `other_enrolled_similarity` |
| `id-motion` | anomaly (1 − similarity) | EER threshold from training | one per 4 s window | `window_similarity`, `window_start_s` |
| `phys-*` | anomaly in [0, 1] | `None` | none | `constraints` (measured value, expected range, violation), `timeline`, per-detector summaries |
| `physdf` | anomaly in [0, 1] | `None` | none | `constraints`, `localization`, `report`, `detectors`, `modalities`, `status` |
| `person-detector` | probability | from training (default 0.5) | per frame (video) | `target_person`, `modality` |

## Detector interfaces

### Audio detectors

```python
det = forensix.load("nes2net-x-itw", device="cuda")
det.predict(audio, sample_rate=None) -> Prediction
det.predict_batch(inputs, sample_rate=None, batch_size=16) -> list[Prediction]
det.sample_rate   # 16000
det.window        # window length in samples (e.g. 64600 for AASIST)
det.threshold     # 0.5
```

`audio` is a path or a 1-D (or `(n, channels)`) float array. With an array, `sample_rate` is
required. Paths libsndfile can't read (m4a, aac, mp4 and other video containers) are decoded
with ffmpeg if it's on `PATH`.

### Image detectors

```python
det = forensix.load("commfor", device="cuda")
det.predict(image) -> Prediction
det.predict_batch(images, batch_size=16) -> list[Prediction]
det.transform     # PIL image -> tensor, the model's upstream eval preprocessing
```

`image` is a path, a PIL image or an `(H, W, 3)` uint8 RGB array. Arrays of another dtype or
shape raise `ValueError`.

### Video detectors

```python
det = forensix.load("gend-pe", device="cuda")
det.predict(video_path) -> Prediction
det.predict_batch(video_paths) -> list[Prediction]   # loops over predict
```

A video with no detectable face scores exactly 0.5 for these models. Check
`len(pred.segment_fake_probs) == 0` to tell "no face" from a genuine 0.5.

### Audiovisual detectors

```python
det = forensix.load("syncnet", device="cuda")
det.predict(video_path) -> Prediction     # needs a video stream and an audio stream
```

Raises `ValueError` if the file lacks either stream. Needs `ffmpeg` on `PATH`.

## Identity detectors

`id-face` (`forensix.identity.face.FaceVerifier`) and `id-motion`
(`forensix.identity.motion.MotionVerifier`) compare media with enrolled people.

| Method | `id-face` | `id-motion` |
|---|---|---|
| `enroll(name, media)` | photos and/or videos (paths or RGB arrays); returns `{"name", "n_embeddings"}` | videos only; returns `{"name", "n_windows"}` |
| `predict(media, identity=name)` | `Prediction` | `Prediction` |
| `verify(media, identity)` | the `details` dict, without building a `Prediction` | same |
| `identify(media)` | `[(name, similarity), ...]`, best first | not available |
| `save_gallery(path)` / `load_gallery(path)` | `.npz` of enrolled embeddings | same; galleries are tagged with the model and can't be mixed |
| `gallery.names()`, `gallery.remove(name)` | manage enrolled people | same |

`verify` raises `ValueError("No face found")` (or no tracked 4 s window, for `id-motion`) when
the media has nothing to compare.

## Physics detectors

```python
det = forensix.load("phys-voice")           # also phys-room (audio); phys-bloodflow, phys-motion, phys-lighting (video)
det.predict(path) -> Prediction             # audio detectors also take an array plus sample_rate=
det.load_priors("learned_priors.json")      # or a dict; None = packaged table, if any, else provisional ranges
det.analyze(frames, fps) / det.analyze(x, sr) -> dict   # the same evidence, from decoded media

physdf = forensix.load("physdf", weights_path="physdf_graph.safetensors")   # weights_path optional
physdf.predict(media, audio_path=None) -> Prediction    # video (uses its sound), audio, or video + separate audio
```

Report helpers in `forensix.physics.report`:

| Function | Description |
|---|---|
| `build_report(result)` | Builds the report dict from a `physdf` result (already done for you in `details["report"]`). |
| `to_markdown(report)` | The report as Markdown (also in `details["report"]["markdown"]`). |
| `verify_report(report, result)` | Re-derives every claim, the score and the verdict from the evidence; returns a list of problems (empty means it checks out). |
| `render_figures(result, out_dir)` | Draws the residual timeline; needs matplotlib. Returns the written paths. |

## Person-specialized models: `forensix.specialized`

| Function / class | Description |
|---|---|
| `train_person_model(person_name, real_media, negative_media=None, modality="auto", base_model=None, use_peft=True, lora_rank=4, epochs=30, lr=2e-3, device="cpu", synthetic_perturbations_per_sample=6, verbose=False)` | Trains a ~20k-parameter model for one person and returns a `PersonDetector`. Without negatives, it synthesizes manipulated and impostor-like samples. |
| `pretrain_base_model(real_media, fake_media, modality="image", epochs=25, lr=2e-3, hidden_dim=64, embed_dim=64, device="cpu")` | Pretrains a base `TinyPersonModel` on general real vs fake media, to fine-tune per person with LoRA. |
| `PersonDetector.predict(media)` | Scores media for the enrolled person. |
| `PersonDetector.save(path)` / `PersonDetector.load(path, device="cpu")` | safetensors checkpoint. `forensix.load("person-detector", weights_path=path)` also works. |
| `extract_features(media, modality)` and per-modality `extract_*_features` | The feature extractors, for custom training loops. |

## Metrics: `forensix.metrics`

Fakes are the positive class, and higher scores mean more likely fake.

| Function | Returns |
|---|---|
| `roc_auc(fake_scores, real_scores)` | ROC-AUC, ties counted as ½ |
| `average_precision(fake_scores, real_scores)` | AP, same definition as scikit-learn |
| `compute_eer(fake_scores, real_scores)` | `(eer, threshold)`, ASVspoof-compatible, including tie-breaking |
| `binary_summary(fake_scores, real_scores, threshold=0.5)` | dict with `auc`, `ap`, `n_fake`, `n_real`, and (if `threshold` is not `None`) `balanced_acc`, `real_acc`, `fake_acc` |

## Media helpers

| Function | Description |
|---|---|
| `forensix.audio.io.load_audio(audio, target_sr, sample_rate=None)` | Mono float32 waveform at `target_sr`; ffmpeg fallback |
| `forensix.image.io.load_image(image)` | RGB PIL image, without resizing |
| `forensix.video.io.read_frames(path, num_frames)` | `(frames, indices)`: evenly spaced RGB frames, decoded sequentially |
| `forensix.audiovisual.io.read_av(path, fps=25, sample_rate=16000)` | `(frames, audio)` via ffmpeg; full resolution, whole file in memory |
| `forensix.physics.io.read_video(path, max_seconds=30.0, max_side=640)` | Every frame of the first 30 s, downscaled |
| `forensix.identity.media.sample_frames(path, fps=2.0, max_frames=120)` | Frames at a fixed rate; an image gives one frame |

## Registry and weights

| Name | Description |
|---|---|
| `forensix.registry.ModelSpec` | Model metadata ([Concepts](concepts.md#registry-and-modelspec)) |
| `forensix.registry.register(spec)` | Adds a model; raises `ValueError` on a duplicate name |
| `forensix.registry.get_spec(name)` | The `ModelSpec`, or `KeyError` |
| `forensix.hub.WeightFile(url, sha256, filename)` | A downloadable, hash-pinned file |
| `forensix.hub.fetch(weights, force=False)` | Local path to a verified file, downloading it if needed |
| `forensix.hub.torch_load(path)` | `torch.load(weights_only=True)` that also allows the numpy types some upstream checkpoints contain |

## HTTP server: `forensix.server`

| Name | Description |
|---|---|
| `python -m forensix.server [--config F] [--device D] [--preload M ...] [--port P] [--api-key K]` | Run the server ([HTTP server](server.md)) |
| `forensix.server.ServerConfig` | Settings dataclass; `ServerConfig.load(path, **overrides)` layers JSON file, `FORENSIX_SERVER_*` environment variables and overrides |
| `forensix.server.create_app(config)` | The FastAPI app, e.g. to mount in your own ASGI server |

## Command-line scripts

All scripts are in `scripts/` and print `--help`.

| Script | Purpose |
|---|---|
| `download_weights.py [models...] [--family F] [--force]` | Pre-fetch weights into the cache |
| `eval_video_folder.py --real DIR --fake DIR --out JSON` | Benchmark video and image detectors on two folders of videos (resumable) |
| `eval_image.py --data DIR` | Image detectors on the ForensiX image eval set |
| `build_image_eval_set.py --out DIR` | Build the license-clean image eval set (GPU) |
| `eval_asvspoof2019_la.py --protocol F --audio-dir DIR` | Audio detectors on a fixed ASVspoof 2019 LA subset |
| `eval_deepfake_eval_2024.py --data DIR --modality M` | Any family on Deepfake-Eval-2024 |
| `eval_identity.py --data DIR --models ...` | Identity detectors: real vs fake vs impostor |
| `calibrate_physics.py measure / fit / eval` | Learned physics priors |
| `train_avcons.py`, `train_avid.py`, `train_id_motion.py`, `train_physdf.py` | `extract` then `train` ([Training](guides/training.md)) |
| `train_person_model.py --person NAME --real FILES [--fake FILES] --out F` | Tiny per-person model; `--load F --test FILE` to score |
