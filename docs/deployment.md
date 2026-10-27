# Deployment at scale

ForensiX is a Python library, not a service. This guide shows how to build a service around it
that scores large volumes of media reliably, cheaply and auditably. It is written for the
engineers who will design and run that service.

The worker code below was run end to end with real models. The Dockerfile, queue, Kubernetes and
autoscaling sections are patterns to adapt. They have not been built or run as part of ForensiX's
tests, so build and check them in your environment.

- [Architecture](#architecture)
- [Input limits](#input-limits)
- [Workers](#workers)
- [Packaging](#packaging)
- [Sizing and scaling](#sizing-and-scaling)
- [Decisions, thresholds and review](#decisions-thresholds-and-review)
- [Monitoring](#monitoring)
- [Security](#security)
- [Audit and reproducibility](#audit-and-reproducibility)
- [Launch checklist](#launch-checklist)

## Architecture

```text
                ┌──────────────┐
 uploads ──────▶│  API / intake │── store original (object storage), compute SHA-256
                └──────┬───────┘
                       │ job {media_uri, sha256, priority}
                       ▼
                ┌──────────────┐    cache hit (same sha256 + same model versions)
                │  job queue    │───────────────────────────────────────────┐
                └──────┬───────┘                                            │
        ┌──────────────┼──────────────────┬─────────────────┐               │
        ▼              ▼                  ▼                 ▼               │
 ┌────────────┐ ┌─────────────┐  ┌───────────────┐ ┌──────────────┐         │
 │ probe +    │ │ face-video  │  │ audio pool    │ │ AV pool      │         │
 │ normalize  │ │ pool (GPU)  │  │ (GPU)         │ │ (GPU, RAM)   │         │
 │ (CPU)      │ │ gend-pe ... │  │ nes2net-x-itw │ │ syncnet ...  │         │
 └────────────┘ └──────┬──────┘  └──────┬────────┘ └──────┬───────┘         │
                       └─────────────────┼─────────────────┘                 │
                                         ▼                                   │
                                ┌─────────────────┐                          │
                                │ results store    │◀────────────────────────┘
                                │ (scores+versions)│
                                └────────┬────────┘
                                         ▼
                                ┌─────────────────┐     ┌──────────────┐
                                │ decision layer   │────▶│ human review │
                                │ thresholds, rules│     │ queue        │
                                └─────────────────┘     └──────────────┘
```

Design principles:

- **Asynchronous.** Video models take seconds per clip on a GPU and minutes on a CPU. Accept a
  job, return an ID, and deliver results by callback or polling. Never run inference inside a
  web request.
- **Pools by model group.** Each pool is a set of identical worker processes with models loaded
  once at startup. Pools scale independently: audio-only traffic shouldn't wait behind video.
- **Normalize first.** One cheap CPU step probes the media and caps resolution and duration
  before any GPU work ([Input limits](#input-limits)).
- **Scores, then decisions.** Workers produce raw scores with full provenance. A separate
  decision layer applies thresholds and rules from versioned configuration, so you can change a
  threshold without rescoring anything.
- **Idempotent by content.** Key results by `(input SHA-256, model, weights hash)`. Re-uploads
  and retries cost nothing.

## Input limits

Some code paths read whole files into memory. Bound every input before it reaches a model.

| Path | Reads | Memory and time | Limit to apply |
|---|---|---|---|
| Video models (`gend-pe` etc.) | 32 evenly spaced frames, decoded sequentially | Memory small; time grows with length | Fine for long files, but see coverage below |
| Audiovisual models | **Every frame at 25 fps, full resolution, all in RAM**, plus audio | About 155 MB per second at 1080p, so a 1-minute 1080p clip is about 9 GB | Downscale to ≤ 720p and cut to ≤ 60 s segments |
| Physics models | Every frame of the first 30 s, downscaled to 640 px | Bounded (about 1 GB at 30 fps) | Content after 30 s is ignored: segment long files |
| Audio models | The whole file, at 16 kHz mono | Small (≈ 4 MB per minute) | Segment very long recordings for latency |
| `id-face` | 2 frames per second, up to 120 | Bounded | Content after 60 s is ignored: segment |

**Coverage on long videos.** Video models spread 32 frames over the whole file. On a
10-minute video that's one frame every 19 s, so a short manipulated segment can be missed.
Split long media into 30–60 s segments, score each segment, and keep per-segment results, then
flag on the most suspicious segment.

A normalization step with ffmpeg (used by the worker below):

```bash
ffmpeg -nostdin -v error -y -i input -t 60 -vf "scale=-2:'min(720,ih)'" \
       -c:v libx264 -crf 18 -c:a aac -b:a 192k normalized.mp4
```

Re-encoding changes pixels. The detectors were tested on compressed media, and CRF 18 is close
to visually lossless, but measure your thresholds on media that went through the same
normalization. For segments, use `-ss START -t 60` per segment.

## Workers

A worker is a long-running process that loads its models once and then scores jobs one at a
time. This one probes the input, normalizes video, runs only the models that apply (no audio
track means no audio or AV models), and returns a record with provenance:

```python
"""ForensiX scoring worker: one process, models loaded once, one job at a time."""

import hashlib
import importlib.metadata
import json
import subprocess
import tempfile
import time
from pathlib import Path

import forensix

MAX_SECONDS = 60          # longer media is split into segments before it reaches this worker
MAX_HEIGHT = 720


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(path) -> dict:
    """Stream types and duration, via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type:format=duration",
         "-of", "json", str(path)], capture_output=True, text=True, timeout=30, check=True).stdout
    info = json.loads(out)
    kinds = {s["codec_type"] for s in info.get("streams", [])}
    return {"video": "video" in kinds, "audio": "audio" in kinds,
            "duration_s": float(info.get("format", {}).get("duration", 0) or 0)}


def normalize(src, workdir) -> Path:
    """Cap duration and resolution so memory per job is bounded."""
    dst = Path(workdir) / "input.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-t", str(MAX_SECONDS),
         "-vf", f"scale=-2:'min({MAX_HEIGHT},ih)'", "-c:v", "libx264", "-crf", "18",
         "-c:a", "aac", "-b:a", "192k", str(dst)],
        check=True, timeout=300)
    return dst


class Worker:
    def __init__(self, video_models, audio_models, av_models, device="cuda"):
        self.models = {}
        for group, names in (("video", video_models), ("audio", audio_models), ("av", av_models)):
            for name in names:
                self.models[name] = (group, forensix.load(name, device=device))
        self.versions = {"forensix": importlib.metadata.version("forensix")}
        for name in self.models:
            w = forensix.info(name).weights
            self.versions[name] = w.sha256[:16] if w else "no-weights"

    def run(self, media_path) -> dict:
        record = {"input_sha256": sha256_file(media_path), "versions": self.versions, "results": {}}
        info = probe(media_path)
        record["input"] = info
        with tempfile.TemporaryDirectory() as tmp:
            path = normalize(media_path, tmp) if info["video"] else Path(media_path)
            for name, (group, detector) in self.models.items():
                if (group == "video" and not info["video"]) or (group == "audio" and not info["audio"]) \
                        or (group == "av" and not (info["video"] and info["audio"])):
                    record["results"][name] = {"status": "not_applicable"}
                    continue
                start = time.perf_counter()
                try:
                    p = detector.predict(path)
                    result = {"status": "ok", "score": p.fake_prob, "score_type": p.score_type,
                              "default_threshold": p.threshold, "n_segments": len(p.segment_fake_probs)}
                    if group == "video" and not p.segment_fake_probs:
                        result["status"] = "no_face"
                except Exception as e:          # decode failures, no face track, ...
                    result = {"status": "error", "error": repr(e)[:500]}
                result["seconds"] = round(time.perf_counter() - start, 2)
                record["results"][name] = result
        return record
```

For a video with sound, run on CPU with `dfdc-wm`, `aasist` and `syncnet`:

```json
{
 "input_sha256": "…",
 "versions": {"forensix": "0.1.0", "dfdc-wm": "651ca679c582501b", "aasist": "51d2d9cf0738172f", "syncnet": "961e8696f888fce4"},
 "results": {
  "dfdc-wm": {"status": "ok", "score": 0.032, "score_type": "probability", "default_threshold": 0.5, "n_segments": 24, "seconds": 49.4},
  "aasist":  {"status": "ok", "score": 0.109, "score_type": "probability", "default_threshold": 0.5, "n_segments": 2, "seconds": 1.0},
  "syncnet": {"status": "ok", "score": -0.535, "score_type": "anomaly", "default_threshold": null, "n_segments": 0, "seconds": 76.9}
 },
 "input": {"video": true, "audio": true, "duration_s": 8.1}
}
```

For an audio file, the video and AV models report `"not_applicable"`.

Connect the worker to your queue (SQS, Pub/Sub, Kafka, Redis, Celery) with a loop: take a job,
download the media to local disk, call `worker.run`, write the record, acknowledge the job.

**Rules for workers:**

- **Load once.** `forensix.load` hashes every weight file, which takes seconds for 1.6 GB
  files. Never load per request.
- **Processes, not threads.** Detectors keep state (networks, face detectors) and aren't
  designed for concurrent calls. Run one detector instance per process, and scale with
  processes.
- **Timeouts per job.** Run each job under a hard timeout: a watchdog that kills and restarts
  the worker process. A Python-level timeout can't interrupt a stuck decoder.
- **Limit CPU threads.** Set `OMP_NUM_THREADS` / `torch.set_num_threads` to the worker's
  share of the cores. Otherwise several workers fight over every core.
- **Retry errors, not results.** Retry on infrastructure errors (OOM, killed worker). A
  `"no_face"` or decode error is a result: store it, don't retry it forever.
- **Spawn, don't fork, with CUDA.** If one parent process starts the workers, use
  `multiprocessing.get_context("spawn")`.

## Packaging

Bake code, dependencies and weights into one image, so a new node starts without downloading
anything and the running service never needs the network. This is a starting template, not yet
built in ForensiX's CI. Check it with a CPU build first
(`--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu --build-arg MODELS="aasist dfdc-wm"`)
and `docker run --network none`.

```dockerfile
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# CUDA build of PyTorch; use https://download.pytorch.org/whl/cpu for CPU-only images
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir torch torchvision --index-url ${TORCH_INDEX}

WORKDIR /app
COPY pyproject.toml README.md ./
COPY forensix ./forensix
RUN pip install --no-cache-dir .

# Bake the weights into the image: no downloads at startup, no network needed at runtime.
# Loading each model also fetches the helper files its builder needs (face detectors etc.).
ENV FORENSIX_CACHE=/opt/forensix_cache
ARG MODELS="gend-pe dfdet-clip nes2net-x-itw syncnet"
RUN python -c "import forensix, sys; [forensix.load(m) for m in sys.argv[1:]]" ${MODELS} \
 && chmod -R a+rX ${FORENSIX_CACHE}

COPY worker.py ./
RUN useradd --create-home --uid 10001 app
USER app
ENV HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
CMD ["python", "worker.py"]
```

- **One image per pool** (`--build-arg MODELS=...`) keeps images small. The four default models
  add about 6 GB of weights.
- **Match the CUDA build to your nodes.** `cu124` above is an example; pick the PyTorch wheel
  index that matches your GPU driver.
- **Lazy helper files.** `phys-lighting` and `physdf` download the ICT-FaceKit meshes on their
  first *prediction*, not at load. For those, add a build step that runs one prediction on a
  short sample video, then verify the image runs with networking disabled
  (`docker run --network none`).
- **Your trained checkpoints** (`avcons`, `avid`, `id-motion`, `physdf`, `person-detector`):
  `COPY` them into the image, or mount them read-only, and pass `weights_path=`.

## Sizing and scaling

### Per-model cost

Measured on one NVIDIA L4 (24 GB) with 2–6 s clips (SDFVD). Most of these were measured with
2–4 processes sharing the GPU, so treat them as upper bounds:

| Model | Seconds per clip | Weights |
|---|---|---|
| image models (8 frames per video) | 0.06–0.5 (`rine-ldm` 2–2.7) | 5 MB – 1.6 GB |
| `dfdc-wm` | ~1 | 312 MB |
| `dfdet-clip` | ~2.7 | 1.2 GB + 1.6 GB CLIP |
| `gend-pe`, `gend` | ~7 (shared GPU) | 1.2 GB each |
| `dfdc-selim` | ~7 (shared GPU) | 1.8 GB |

Audio, audiovisual and physics models haven't been timed on a GPU yet. Measure them on your
own media before sizing those pools.

### Workers per GPU

Video and audiovisual models spend much of each job on the CPU (decoding, face detection,
tracking), so one process leaves the GPU idle.

1. Start with 3 workers per 24 GB GPU and about 3 CPU cores per worker.
2. Raise the count while GPU utilization rises and GPU memory fits; stop when per-job latency
   grows without throughput gains.
3. Each worker holds its own copy of the model in GPU memory, which sets the ceiling.

### Autoscaling

- **Scale each pool on its own queue depth** (or age of the oldest job), not on CPU or GPU
  utilization. On Kubernetes, KEDA does this for most queue systems.
- **Keep a minimum of one warm worker per pool.** Cold starts include pulling a multi-GB image
  and hashing the weights, which can take minutes.
- **Use separate queues per priority.** User-facing checks shouldn't wait behind a bulk
  backfill.
- **Route by content.** Use `probe` to send audio-only files only to the audio pool.
  A cheap face check (any face in a few frames?) avoids sending faceless videos to face models.

### Cutting cost

- **Cache by content hash.** Viral media gets uploaded many times.
- **Cascade.** Run a fast model on everything (e.g. `dfdet-clip`), and send only uncertain
  cases to the slower ones. Measure the cascade's detection rate as a whole: the first stage's
  misses are never seen again.
- **Don't run models that can't help.** Image detectors on face-swap video added little on
  SDFVD (AUC 0.56–0.75).

## Decisions, thresholds and review

Keep decision logic out of the workers, in versioned configuration:

```yaml
# decision-config v7 (2026-10-26), thresholds set at 1% FPR on 12,000 real uploads (platform X)
rules:
  face_manipulation:
    model: dfdet-clip
    threshold: 0.71
  voice_clone:
    model: nes2net-x-itw
    threshold: 0.83
  lip_sync_mismatch:
    model: syncnet
    threshold: -2.0        # anomaly score = -confidence; flag when confidence < 2
action:
  any_rule_fires: human_review
  none_fire: pass
```

- **Set thresholds on real traffic** from the deployment's own sources, at the false-alarm rate
  your review team can absorb ([Interpreting scores](guides/interpreting-scores.md#choosing-a-threshold)).
  The defaults are wrong for many domains: on SDFVD, `rine` called every video real at 0.5.
- **Version the config** and store its version with every decision, so any decision can be
  explained later.
- **Send uncertain or consequential cases to people,** with the evidence: which check fired,
  its score and threshold, segment timelines, `syncnet` offset and confidence, the `physdf`
  report. The `stablesyncnet` weights license forbids fully automated decisions that adversely
  affect someone's legal rights ([Licensing](guides/licensing.md)).
- **Feed reviewer verdicts back** as labelled data for re-checking thresholds.

## Monitoring

| Signal | Why | Alert when |
|---|---|---|
| Score distribution per model (daily quantiles) | Drift: new cameras, codecs or generators shift scores | The 99th percentile on presumed-real traffic moves past the threshold |
| Flag rate per rule | Thresholds going stale; attacks | Outside the expected band |
| `no_face` / `not_applicable` / `error` rates | Pipeline breakage, new formats | Sudden change |
| Latency per pool (p50, p95) and queue age | Capacity | SLA risk |
| GPU utilization and memory per node | Sizing | Persistently below 50% (too few workers) or OOM kills |
| Reviewer agreement with flags | Real precision | Precision falls |

Regularly audit a random sample of unflagged media too. Monitoring only flagged items shows
precision, never what you miss.

## Security

The service decodes untrusted files with ffmpeg and OpenCV, two large C codebases with a
history of decoder vulnerabilities. Treat every upload as hostile:

- **Sandbox the workers.** Run as non-root (the Dockerfile does), with a read-only root
  filesystem, a writable temp directory only, and no network egress
  (weights are baked in, `HF_HUB_OFFLINE=1`).
- **Limit resources:** CPU and memory limits per container, file-size caps at intake, a
  duration cap in normalization, and job timeouts.
- **Patch the decoders.** Rebuild images regularly to pick up ffmpeg and OpenCV security
  fixes.
- **Never trust file extensions.** Probe the streams.
- **Weights are integrity-checked.** `forensix.load` verifies SHA-256 hashes, and checkpoints
  load with `weights_only=True` or safetensors, so a tampered checkpoint can't execute code.
  Keep the cache read-only at runtime.
- **Protect the gallery and scores.** Identity galleries are biometric data, and stored scores
  reveal which media was flagged. Encrypt them at rest, restrict access, and set retention.

## Audit and reproducibility

Store with every result:
- the input SHA-256 and, for segments, their start and end;
- the ForensiX version and git commit (set it in the image, e.g. as an environment variable at
  build time);
- the model name and weights hash (`forensix.info(name).weights.sha256`), or the checkpoint
  hash for models you trained;
- the normalization parameters;
- the decision-config version, the decision, and any reviewer action.

With these, any past decision can be re-derived exactly: same input, same code, same weights,
same thresholds. Reprocessing after a model upgrade becomes a query: everything scored with the
old weights hash.

## Launch checklist

- [ ] Models chosen per threat ([Choosing detectors](guides/choosing-detectors.md)) and
      evaluated on your own media ([Evaluation](guides/evaluation.md)).
- [ ] Thresholds set on real traffic at a chosen false-alarm rate; config versioned.
- [ ] Inputs normalized: resolution, duration, segmentation for long media.
- [ ] Image built with baked weights; verified with `--network none`.
- [ ] Workers: models loaded once, processes not threads, job timeouts, CPU threads limited.
- [ ] Pools sized by measurement; autoscaling on queue depth; warm minimums.
- [ ] Results keyed by content hash, with full provenance.
- [ ] Human review for consequential decisions; reviewer feedback captured.
- [ ] Monitoring on score drift, flag rates, error rates, latency; random audits of unflagged
      media.
- [ ] Sandboxed decoding, no egress, patched images.
- [ ] License review done ([Licensing](guides/licensing.md)), including `‡` training-data
      terms and `stablesyncnet` restrictions if used.
