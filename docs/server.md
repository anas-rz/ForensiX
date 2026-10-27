# HTTP server

`forensix.server` serves every ForensiX detector over HTTP: images, video, audio, lip sync,
physics and identity, with any combination of models per request. Models stay in memory between
requests, so after the first load (or a preload at startup) a request pays only for inference.

This page is for developers who call the API and for operators who run it. For running it at
scale behind a queue, see [Deployment at scale](deployment.md).

## Quick start

```bash
pip install -e ".[server]"
python -m forensix.server --device cuda --preload dfdet-clip commfor nes2net-x-itw syncnet --api-key "$KEY"
```

```bash
curl -H "X-API-Key: $KEY" -F file=@photo.jpg http://localhost:8000/v1/detect
```

```json
{
  "id": "364cdd572d63440187b5cd489f40c7cf",
  "filename": "photo.jpg",
  "sha256": "d91631d7923a2b224406e0dc7496f97bb699399d16c9a80adde2b3b9c2ca06ef",
  "media": {"kind": "image", "has_audio": false, "duration_s": null, "bounded": null},
  "results": {
    "commfor": {
      "status": "ok", "family": "image", "score": 0.0022, "score_type": "probability",
      "threshold": 0.5, "label": "real", "n_segments": 0, "cached": false, "seconds": 0.183
    }
  },
  "seconds": 0.197
}
```

Interactive API docs (OpenAPI) are at `http://localhost:8000/docs`.

## Endpoints

| Method and path | Auth | Purpose |
|---|---|---|
| `GET /health` | none | Liveness: the process is up |
| `GET /ready` | none | Readiness: preloaded models are in memory (503 while loading) |
| `POST /v1/detect` | user | Score one file |
| `POST /v1/detect/batch` | user | Score several files concurrently |
| `GET /v1/models[?family=video]` | user | Every registered model: family, license, loaded, available, defaults |
| `GET /v1/models/{name}` | user | One model |
| `POST /v1/models/{name}/load` | admin | Load and warm up a model now |
| `DELETE /v1/models/{name}` | admin | Unload a model |
| `GET /v1/identities?model=id-face` | user | Enrolled people |
| `POST /v1/identities/{identity}?model=id-face` | admin | Enroll a person from photos and/or videos (`files`) |
| `DELETE /v1/identities/{identity}?model=id-face` | admin | Remove a person |
| `GET /v1/stats` | user | Per-model load time, calls, errors, cache hits, latency, batch size |

### `POST /v1/detect`

Multipart form:

| Field | Default | Meaning |
|---|---|---|
| `file` | required | An image, video or audio file. The type is detected from the content, not the extension. |
| `models` | by media type | Comma-separated model names, e.g. `gend-pe,syncnet`. |
| `identity` | none | The enrolled person to check against, for identity models (`id-face`, `id-motion`). |
| `details` | `false` | Include each model's evidence (`Prediction.details`): sync offsets, physics constraints, reports, similarities. |
| `segments` | `false` | Include per-segment scores: audio windows, faces, frames. |

Without `models`, the server runs the defaults for the media type:

| Media | Default models |
|---|---|
| image | `commfor` |
| video without sound | `dfdet-clip` |
| video with sound | `dfdet-clip`, `nes2net-x-itw`, `syncnet` |
| audio | `nes2net-x-itw` |

Change them with `default_models` in the config.

### Per-model results

Every requested model gets an entry in `results`, with a `status`:

| `status` | Meaning | Other fields |
|---|---|---|
| `ok` | Scored | `score`, `score_type`, `threshold`, `label`, `n_segments`, `cached`, `seconds`, optional `segments` and `details` |
| `no_face` | A video model found no face. Its score is the meaningless 0.5, and `label` is `null` | as `ok` |
| `not_applicable` | The model can't score this media (an image model on audio), or an identity model got no `identity` | `reason` |
| `unavailable` | The model needs weights you train (`avcons`, `avid`, `id-motion`, `person-detector`) and none are configured | `reason` |
| `error` | The model failed on this file (undecodable stream, no face track for AV models, ...) | `error` |

One model failing never fails the request. Request-level errors: `401` wrong or missing key,
`404` unknown model name, `400` empty file, `413` file too large, `415` not an image, video or
audio file.

**Reading scores.** `score` is the model's `fake_prob`: higher means more likely fake. Scores
are **not comparable across models**. `threshold` is the configured override or the model's
default, and `label` applies it. Anomaly-type models (`syncnet`, `avad`, physics) have
`threshold: null` and `label: null` until you configure a threshold for them. See
[Interpreting scores](guides/interpreting-scores.md); defaults are often wrong for a new domain.

### Batch

```bash
curl -H "X-API-Key: $KEY" -F files=@a.jpg -F files=@b.jpg -F files=@c.jpg \
     -F models=commfor http://localhost:8000/v1/detect/batch
```

Returns `{"results": [<one /v1/detect response per file>]}`. A file that fails as a whole
(unsupported type, too large) gets `{"filename", "error", "status_code"}` in its place. The files
run concurrently, so image and audio models score them in shared batches.

### Identities

```bash
# enroll (admin key): photos and/or videos
curl -H "X-API-Key: $ADMIN" -F files=@alice1.jpg -F files=@alice_interview.mp4 \
     http://localhost:8000/v1/identities/alice
# check media against alice, together with a face-manipulation model
curl -H "X-API-Key: $KEY" -F file=@new.mp4 -F identity=alice -F models=id-face,dfdet-clip \
     http://localhost:8000/v1/detect
```

Galleries persist to `gallery_dir` (one `.npz` per identity model) and are reloaded at startup.
Without `gallery_dir`, they live in memory only. Galleries hold face embeddings, which are
biometric data: protect the directory accordingly ([Licensing](guides/licensing.md#biometric-and-privacy-law)).

## Python client

```python
import requests

API, KEY = "http://localhost:8000", "..."

def detect(path, models=None, **form):
    with open(path, "rb") as f:
        r = requests.post(f"{API}/v1/detect", headers={"X-API-Key": KEY}, files={"file": f},
                          data={"models": ",".join(models) if models else None, **form},
                          timeout=600)   # video on many models can take minutes on CPU
    r.raise_for_status()
    return r.json()

result = detect("interview.mp4", models=["gend-pe", "syncnet"], details="true")
for name, res in result["results"].items():
    print(name, res["status"], res.get("score"), res.get("label"))
```

## Configuration

Settings are layered: defaults, then a JSON file (`--config`), then `FORENSIX_SERVER_<FIELD>`
environment variables, then command-line flags.

| Field | Default | Meaning |
|---|---|---|
| `device` | `"cpu"` | `cpu`, `cuda` or `cuda:N` |
| `preload` | `[]` | Models loaded and warmed up at startup; `/ready` turns 200 when they're in memory |
| `warmup` | `true` | Run one synthetic prediction after loading image and audio models |
| `max_loaded` | `8` | Models in memory at once; the least recently used one is unloaded beyond this |
| `weights` | `{}` | Checkpoints for models you trained: `{"avcons": "/models/avcons.safetensors"}` |
| `thresholds` | `{}` | Decision thresholds replacing model defaults: `{"dfdet-clip": 0.71, "syncnet": -2.0}` |
| `default_models` | see above | Models per media type: keys `image`, `video`, `video+audio`, `audio` |
| `max_batch` | `16` | Largest dynamic batch for image and audio models (1 disables batching) |
| `batch_wait_ms` | `5` | How long a request waits for others to join its batch |
| `result_cache_size` | `10000` | Cached results per (file hash, model); 0 disables |
| `max_upload_mb` | `500` | Larger uploads get 413 |
| `max_video_seconds` | `60` | Longer videos are trimmed to this before scoring |
| `max_video_height` | `720` | Taller videos are downscaled to this before scoring |
| `gallery_dir` | none | Where identity galleries are saved |
| `api_key` | none | Required for user endpoints when set (`X-API-Key` header or `Authorization: Bearer`) |
| `admin_key` | none | Required for load/unload and enrollment when set; falls back to `api_key` |
| `tmp_dir` | system temp | Where uploads are written while being scored |

Example `server.json`:

```json
{
  "device": "cuda",
  "preload": ["dfdet-clip", "commfor", "nes2net-x-itw", "syncnet", "id-face"],
  "thresholds": {"dfdet-clip": 0.71, "nes2net-x-itw": 0.83, "syncnet": -2.0},
  "weights": {"avcons": "/models/avcons.safetensors"},
  "gallery_dir": "/data/galleries",
  "max_loaded": 6
}
```

```bash
FORENSIX_SERVER_API_KEY=... FORENSIX_SERVER_PRELOAD=commfor,dfdet-clip python -m forensix.server --config server.json
```

Lists in environment variables can be comma-separated or JSON; dicts are JSON.

## Performance

### Where time goes

Measured on CPU (this repo's dev machine):

| Step | Cost | What the server does about it |
|---|---|---|
| `import forensix` and model libraries (`transformers`) | 3–9 s, once per process | Paid at startup, before `/ready` |
| Verifying weight hashes | 4.5 s per 1.6 GB file, every load | Verified files get a stamp (hash, size, mtime), so unchanged files aren't rehashed: `dfdet-clip` + `clipdet` weights went from 7.0 s to 0.0 s on reload |
| Building the model | 0.1–10 s | Done once; models stay loaded (`preload`, `max_loaded`) |
| First inference (CUDA kernels, lazy imports) | varies | `warmup` runs one synthetic prediction for image and audio models |
| Inference | `commfor` image 0.18 s; `aasist` 10 s clip 1.2 s; `dfdc-wm` 8 s video ~50 s; `syncnet` ~80 s (CPU) | GPU, batching, parallel models |
| Same file again | 1 ms | Result cache keyed by file hash and model |

Video and audiovisual models are slow on CPU. Use a GPU for them; see
[Choosing detectors](guides/choosing-detectors.md#cost) for measured GPU times.

### Concurrency model

- **One process per device.** Models load once and requests share them through threads, so
  memory isn't duplicated. Don't run `uvicorn --workers N`: each worker would load every model
  again. To use several GPUs, run one server per GPU (`CUDA_VISIBLE_DEVICES=0`, `=1`, ...)
  behind a load balancer.
- **Per-model locks.** Detectors aren't designed for concurrent calls, so each model scores one
  request (or one batch) at a time. Different models run in parallel. A video-with-sound
  request runs its video, audio and sync models concurrently.
- **Dynamic batching.** Concurrent requests to the same image or audio model are grouped into one
  forward pass (up to `max_batch`, waiting at most `batch_wait_ms`). With 16 images sent at once
  on CPU, a batch took 3.1 s vs 3.9 s one by one; the gain is larger on GPU.
  `/v1/stats` shows `mean_batch_size`.
- **Bounded inputs.** Videos over `max_video_seconds` or `max_video_height` are re-encoded first,
  because audiovisual models hold every frame in memory. `media.bounded` reports it.

### Tuning

- Preload the models you serve, and set `max_loaded` to fit them all in GPU memory.
- Set `OMP_NUM_THREADS` to the CPU cores you want the server to use.
- Raise `batch_wait_ms` (e.g. 20) when throughput matters more than single-request latency.
- Put frequently re-uploaded content behind the result cache, or a shared cache keyed by `sha256`.

## Security

- **Keys.** Set `api_key`, and a separate `admin_key` for loading models and changing
  galleries. Terminate TLS at a reverse proxy (nginx, a cloud load balancer); the server speaks
  plain HTTP.
- **Untrusted media.** Uploads are decoded by ffmpeg, OpenCV and PIL. Run the server as a
  non-root user, in a container with no network egress (bake weights into the image), with
  memory limits, and keep images patched ([Deployment: Security](deployment.md#security)).
- **Limits.** `max_upload_mb`, `max_video_seconds` and `max_video_height` bound the work per
  request. Put a request timeout and rate limit in the proxy.
- **Integrity.** Weights are hash-verified (stamps are invalidated when a file's size or mtime
  changes; set `FORENSIX_VERIFY_HASH=always` to rehash on every load), and checkpoints load
  without unpickling code.

## Limits

- **Requests are synchronous.** A long video on several heavy models can take minutes on CPU.
  Set client timeouts accordingly. For sustained heavy traffic, put a queue in front of workers
  ([Deployment at scale](deployment.md)).
- **No per-model timeout.** A decoder that hangs on a malformed file holds a thread. Use the
  proxy's request timeout, and restart the process if `/health` stops answering.
- **Long media is trimmed** to `max_video_seconds`. Score long recordings in segments on the
  client side if the whole file matters.
- **Video models' "no face" score.** A 0.5 with no faces is reported as `no_face`, not as a
  result.
