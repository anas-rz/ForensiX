# Recipes

Copy-paste solutions to common tasks. Each recipe was run on CPU against the repo's test
fixtures or the SDFVD videos, and the outputs shown are real. Swap `device="cpu"` for
`device="cuda"` on a GPU.

| # | Recipe | Models |
|---|---|---|
| 1 | [Scan a folder of images into a CSV](#1-scan-a-folder-of-images-into-a-csv) | image |
| 2 | [Check a face video](#2-check-a-face-video) | video |
| 3 | [Find which part of a long recording is fake](#3-find-which-part-of-a-long-recording-is-fake) | audio |
| 4 | [Check a talking-head video across modalities](#4-check-a-talking-head-video-across-modalities) | video, audio, audiovisual, physics |
| 5 | [Pick a threshold for a target false-alarm rate](#5-pick-a-threshold-for-a-target-false-alarm-rate) | any |
| 6 | [Combine several detectors](#6-combine-several-detectors) | any |
| 7 | [Verify that media shows a specific person](#7-verify-that-media-shows-a-specific-person) | identity |
| 8 | [Train a tiny detector for one person](#8-train-a-tiny-detector-for-one-person) | specialized |
| 9 | [Produce a physics forensic report](#9-produce-a-physics-forensic-report) | physics |
| 10 | [Score a large folder of videos in parallel](#10-score-a-large-folder-of-videos-in-parallel) | video |
| 11 | [Benchmark detectors on your own videos](#11-benchmark-detectors-on-your-own-videos) | video, image |

## 1. Scan a folder of images into a CSV

Image detectors batch on the GPU, so pass lists to `predict_batch`. Load one model at a time
and free it before the next one: three CLIP-sized models at once need several GB.

```python
import csv
from pathlib import Path

import forensix

paths = sorted(p for p in Path("photos").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
models = ["commfor", "clipdet", "univfd"]

scores = {}
for name in models:
    detector = forensix.load(name, device="cpu")
    scores[name] = [p.fake_prob for p in detector.predict_batch(paths, batch_size=16)]
    del detector

with open("image_scores.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", *models])
    for i, p in enumerate(paths):
        w.writerow([p.name, *(f"{scores[m][i]:.4f}" for m in models)])
```

On the three real public-domain photos in `tests/data`:

```text
file,commfor,clipdet,univfd
face2_public_domain.jpg,0.0123,0.2115,0.0001
face_public_domain.jpg,0.0022,0.1946,0.0000
other_face_public_domain.jpg,0.0001,0.2191,0.0000
```

**Notes.**
- Image detectors look for whole generated images (GAN, diffusion). They are weak on face
  swaps inside real photos and video frames ([SDFVD benchmark](../benchmark.md): AUC 0.56–0.75).
- Don't resize or re-compress images before scoring. The detectors rely on traces that
  resampling destroys.
- `commfor` is the strongest choice for modern diffusion images (trained on ~4,800 generators).

## 2. Check a face video

The face-based CLIP/PE detectors are the strongest video models in ForensiX (SDFVD AUC
0.92–0.96). Each returns a video score and one score per scored face.

```python
import forensix

detector = forensix.load("dfdet-clip", device="cuda")
pred = detector.predict("clip.mp4")

if not pred.segment_fake_probs:
    print("no face found: the 0.5 score means nothing")
else:
    print(pred.fake_prob, pred.label, f"{len(pred.segment_fake_probs)} faces scored")
```

On SDFVD's face-swapped `vs1.mp4`: `0.8128 fake 32 faces scored` (about 2 minutes on CPU,
about 3 s on an L4).

**Notes.**
- `gend-pe` ranked first on SDFVD (AUC 0.956), but at the default 0.5 threshold it flagged 24%
  of real videos. `dfdet-clip` is better balanced at 0.5 (89% of reals and 76% of fakes right).
  See [recipe 5](#5-pick-a-threshold-for-a-target-false-alarm-rate) to set your own threshold.
- Every video model returns exactly 0.5 when it finds no face. Always check
  `segment_fake_probs` before trusting the score.

## 3. Find which part of a long recording is fake

Audio detectors cut long clips into windows and return one score per window. This maps the
window scores back to timestamps. The windows follow `split_windows`: consecutive, with the last
one aligned to the end of the clip.

```python
import forensix
from forensix.audio.io import load_audio

path = "call_recording.wav"          # any audio format, or a video file (needs ffmpeg)
detector = forensix.load("aasist")
pred = detector.predict(path)

sr, window = detector.sample_rate, detector.window
n = len(load_audio(path, sr))
starts = list(range(0, max(n - window, 0) + 1, window))
if len(pred.segment_fake_probs) > len(starts):
    starts.append(n - window)        # the last window is aligned to the end
for s, p in zip(starts, pred.segment_fake_probs):
    flag = "  <-- suspicious" if p >= detector.threshold else ""
    print(f"{s / sr:6.2f}s - {(s + window) / sr:6.2f}s  fake_prob={p:.3f}{flag}")
print("clip:", pred.fake_prob, pred.label)
```

On the 10 s real CC0 speech fixture:

```text
  0.00s -   4.04s  fake_prob=0.120
  4.04s -   8.07s  fake_prob=0.050
  5.96s -  10.00s  fake_prob=0.002
clip: 0.05715656653046608 real
```

**Notes.**
- For real-world, compressed audio (phone calls, social media), use `nes2net-x-itw` or
  `xlsr-aasist-df` rather than `aasist`; they were built for out-of-domain audio. They need a GPU
  for reasonable speed (about 1.3 s per 4 s window on CPU).
- `nes2net-x-itw` scores whole utterances up to 30 s, so its segments are 30 s chunks, not 4 s
  windows.

## 4. Check a talking-head video across modalities

Different attacks leave different evidence. A face swap shows up in the face models; a cloned
voice in the audio models; a dubbed or lip-synced video in the sync models; physics checks look
for violations of how real faces and voices behave. Run one model per question and read the
answers side by side. Don't average them ([recipe 6](#6-combine-several-detectors) explains why).

```python
import forensix

video = "interview.mp4"
checks = {
    "face (gend-pe)": forensix.load("gend-pe", device="cuda"),
    "voice (nes2net-x-itw)": forensix.load("nes2net-x-itw", device="cuda"),   # reads the video's audio track
    "lip sync (syncnet)": forensix.load("syncnet", device="cuda"),
}
preds = {}
for question, detector in checks.items():
    try:
        p = preds[question] = detector.predict(video)
    except ValueError as e:              # e.g. no audio track, no face
        print(f"{question:24} not applicable: {e}")
        continue
    if p.score_type == "anomaly":
        print(f"{question:24} anomaly score {p.fake_prob:.3f} (no default threshold)")
    else:
        print(f"{question:24} {p.fake_prob:.3f} -> {p.label}")

if "lip sync (syncnet)" in preds:
    sync = preds["lip sync (syncnet)"]
    print("AV offset", sync.details["offset_frames"], "frames, confidence", sync.details["confidence"])
```

SyncNet's confidence is about 10 on a real, in-sync talking head (upstream's example clip). On
an SDFVD clip with unrelated CC0 speech dubbed over it, it measured 0.54, with the best offset
pinned at the edge of the search range (15 frames). That's what a mismatched voice looks like.

**Notes.**
- The audiovisual models load the whole video into memory at full resolution. Downscale to
  720p and trim to about a minute before scoring long videos
  ([Deployment](deployment.md#input-limits)).
- Face swaps keep the original, in-sync mouth, so lip-sync models pass them. Lip-sync
  generators keep the face, so face models may pass them. That's why you need both.

## 5. Pick a threshold for a target false-alarm rate

Default thresholds are rarely right for your data. If you have scores for real media like the
media you'll check, set the threshold so that only a chosen share of real media is flagged:

```python
import numpy as np

def threshold_at_fpr(real_scores, target_fpr):
    """Smallest threshold that flags at most target_fpr of the real scores."""
    return float(np.quantile(real_scores, 1 - target_fpr, method="higher")) + 1e-9
```

Applied to the SDFVD per-video scores in `results/sdfvd_scores.csv`, with a 5% target:

```python
import csv
rows = list(csv.DictReader(open("results/sdfvd_scores.csv")))
is_fake = np.array([r["label"] == "fake" for r in rows])
for model in ["dfdet-clip", "rine"]:
    s = np.array([float(r[model]) for r in rows])
    t = threshold_at_fpr(s[~is_fake], 0.05)
    print(model, round(t, 4), "FPR", np.mean(s[~is_fake] >= t).round(3), "TPR", np.mean(s[is_fake] >= t).round(3))
```

```text
dfdet-clip 0.6307 FPR 0.038 TPR 0.604
rine 0.0063 FPR 0.038 TPR 0.396
```

`rine` needs a threshold of 0.0063, not 0.5: its default threshold called every SDFVD video
real.

**Notes.**
- Fit the threshold on one set of real media and check it on another. Thresholds fitted and
  measured on the same data look better than they are.
- With 53 real videos, as here, the measured false-alarm rate is only accurate to about ±5
  points. For a 1% target you need thousands of real samples.
- Anomaly scores (`avad`, `syncnet`, `physdf`) work the same way: they have no default
  threshold, so this recipe is how you get one.

## 6. Combine several detectors

Scores from different models live on different scales, so never average raw `fake_prob`s.
Normalize each model against its own real-media distribution first, then combine. This version
converts each score to a z-score relative to real media, using 5-fold cross-validation so the
normalization never sees the media it scores:

```python
import numpy as np

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def fuse(scores, is_fake, n_folds=5, seed=0):
    """scores: (n_media, n_models) fake_probs. Returns one fused score per media item."""
    folds = np.random.default_rng(seed).permutation(len(scores)) % n_folds
    x, fused = logit(scores), np.zeros(len(scores))
    for k in range(n_folds):
        train, test = folds != k, folds == k
        real = x[train & ~is_fake]
        fused[test] = ((x[test] - real.mean(0)) / real.std(0)).mean(1)
    return fused
```

On SDFVD, fusing `gend-pe`, `gend` and `dfdet-clip` gave AUC 0.942, **below** `gend-pe` alone
(0.956). The three models share a CLIP-style backbone and training set, so they make the same
mistakes, and averaging dilutes the best one.

**Notes.**
- Fusion helps when models catch *different* things, e.g. a face model plus a voice model
  plus a sync model on talking heads. It rarely helps among near-identical models.
- Measure the fused score on held-out data, exactly as you would a single model.
- For models that answer different questions ([recipe 4](#4-check-a-talking-head-video-across-modalities)),
  an OR over per-model thresholds ("flag if any check fires") is easier to explain than a fused
  number. Set each threshold with recipe 5 so the combined false-alarm rate stays where you
  want it.

## 7. Verify that media shows a specific person

`id-face` enrolls people from photos or videos and checks new media against them. It catches
impersonation and poor face swaps. A good face swap of the person is built to look like them
and passes, so pair it with a face-manipulation model ([recipe 2](#2-check-a-face-video)).

```python
import forensix

det = forensix.load("id-face")
det.enroll("armstrong", ["tests/data/face_public_domain.jpg"])       # photos and/or videos
det.enroll("aldrin", ["tests/data/other_face_public_domain.jpg"])
det.save_gallery("people.npz")                                        # det.load_gallery("people.npz") later

pred = det.predict("tests/data/face2_public_domain.jpg", identity="armstrong")
print(pred.label, pred.fake_prob, pred.details["similarity"], pred.details["other_enrolled_similarity"])
print(det.identify("tests/data/face2_public_domain.jpg"))
```

```text
real 0.3878 0.6122 {'aldrin': 0.0583}
[('armstrong', 0.6121812462806702), ('aldrin', 0.058341484516859055)]
```

The Aldrin photo checked against Armstrong comes back `fake`. For videos,
`details["frame_similarity"]` and `details["frame_time_s"]` show when the face stops being the
person.

More in [Identity and person models](guides/identity.md).

## 8. Train a tiny detector for one person

A `person-detector` is a ~23k-parameter model trained on CPU in seconds from one person's real
media. Negatives are optional: without them, manipulated and impostor-like samples are
synthesized.

```python
from forensix.specialized import PersonDetector, train_person_model

det = train_person_model(
    "armstrong",
    real_media=["tests/data/face_public_domain.jpg"],
    negative_media=["tests/data/other_face_public_domain.jpg"],    # optional
    modality="image",                                              # image, audio, video, audiovisual or auto
)
det.save("armstrong.safetensors")

det = PersonDetector.load("armstrong.safetensors")    # or forensix.load("person-detector", weights_path=...)
print(det)
for f in ["face2_public_domain.jpg", "other_face_public_domain.jpg"]:
    p = det.predict(f"tests/data/{f}")
    print(f, p.label, p.fake_prob)
```

```text
PersonDetector(person='armstrong', modality='image', params=23,301, threshold=0.500)
face2_public_domain.jpg real 0.1861
other_face_public_domain.jpg fake 0.9999
```

The same from the command line:

```bash
python scripts/train_person_model.py --person armstrong --real tests/data/face_public_domain.jpg \
    --fake tests/data/other_face_public_domain.jpg --modality image --out armstrong.safetensors
python scripts/train_person_model.py --load armstrong.safetensors --test tests/data/face2_public_domain.jpg
```

**Notes.** One training photo is a demo, not a deployment. Use dozens of varied real samples,
and hold some back to measure false alarms ([Identity guide](guides/identity.md)).

## 9. Produce a physics forensic report

`physdf` runs every physics check and writes a report from the measured evidence. The
single-constraint detectors (`phys-voice`, `phys-room`, `phys-bloodflow`, `phys-motion`,
`phys-lighting`) expose the same evidence per constraint.

```python
import forensix

pred = forensix.load("phys-voice").predict("tests/data/speech_cc0.flac")
print(pred.fake_prob, pred.score_type)                          # 0.0 anomaly
print(pred.details["constraints"]["f0_jump_rate_per_s"])
# {'expected_range_start': 2.0, 'worse': 'higher', 'prior': 'provisional', 'measured': 0.2584, 'violation': 0.0}

det = forensix.load("physdf")                                   # add weights_path= once you've trained the graph prior
pred = det.predict("interview.mp4")                             # a video (uses its sound), or an audio file
print(pred.details["report"]["markdown"])
pred.details["localization"]                                    # where and when constraints were violated
```

**Read this before relying on physics scores.** Until learned priors are fitted on real media,
the expected ranges are provisional and hand-set. In testing, a real SDFVD clip that was looped
three times and dubbed with unrelated speech scored 1.00 (report confidence: low), driven by
blood-flow ranges. The looping itself may break the pulse signal, but provisional video ranges
are known to fire on real video. Treat physics output as leads for a human reviewer until you
have run `scripts/calibrate_physics.py` on real media like yours ([Physics guide](guides/physics.md)).

## 10. Score a large folder of videos in parallel

Video detectors spend much of their time decoding and finding faces on the CPU, so one process
leaves a GPU idle. Run several worker processes, each with its own copy of the model:

```python
import csv
import multiprocessing as mp
import sys
from pathlib import Path

MODEL, DEVICE, WORKERS = "dfdet-clip", "cuda", 3

_detector = None


def init_worker():
    global _detector
    import torch
    torch.set_num_threads(2)          # leave CPU cores for the other workers
    import forensix
    _detector = forensix.load(MODEL, device=DEVICE)


def score(path):
    try:
        pred = _detector.predict(path)
        return str(path), pred.fake_prob, len(pred.segment_fake_probs), ""
    except Exception as e:            # keep going on corrupt files; record why
        return str(path), None, 0, repr(e)


if __name__ == "__main__":
    folder, out = Path(sys.argv[1]), sys.argv[2]
    videos = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"})
    ctx = mp.get_context("spawn")     # required for CUDA in child processes
    with ctx.Pool(WORKERS, initializer=init_worker) as pool, open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "fake_prob", "n_faces", "error"])
        for row in pool.imap_unordered(score, videos):
            w.writerow(row)
```

Tested with `dfdc-wm`, 2 CPU workers, on 4 SDFVD videos.

**Notes.**
- Each worker holds a full copy of the model in GPU memory, so size `WORKERS` by GPU memory.
  The SDFVD benchmark ran 2–4 processes on one 24 GB L4.
- For a service rather than a one-off batch, see [Deployment at scale](deployment.md).

## 11. Benchmark detectors on your own videos

Put real videos in one folder and fake ones in another, then:

```bash
python scripts/eval_video_folder.py --device cuda \
    --real my_data/real --fake my_data/fake --dataset my-data \
    --models gend-pe gend dfdet-clip dfdc-selim dfdc-wm commfor \
    --out results/my_data.json --scores-out results/my_data_scores.csv
```

It reports AUC, AP, EER and accuracy at each model's default threshold, saves after every
model, and skips finished models when rerun. Image detectors score 8 evenly spaced frames per
video (`--frames`). See [Evaluating on your data](guides/evaluation.md) for how to build a test
set that measures what you think it measures.
