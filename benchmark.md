# ForensiX benchmark: SDFVD (face-swap videos)

Every ForensiX image and video detector was scored on the same 106 videos. Raw results are in
`results/sdfvd.json` (metrics and per-video scores), `results/sdfvd_scores.csv` (one row per video)
and `results/sdfvd.log`.

## Dataset

- **Name:** SDFVD.
- **Link:** <https://data.mendeley.com/datasets/bcmkfgct2s/1>
- **License:** CC BY 4.0.
- **Size:** 106 videos. 53 are real (`SDFVD/videos_real/v1.mp4` … `v53.mp4`) and 53 are fake
  (`SDFVD/videos_fake/vs1.mp4` … `vs53.mp4`). The archive is 182 MB, with sha256
  `734a101fdfd89d5b20ac69f510d6a7cf3b588a145c1993ba2488322ae50d95a5`.
- **Real videos:** short stock clips from Pexels, 2.1–5.8 s long (median 3.3 s), at 30 fps.
  Resolutions are 1280x720 (26), 1280x674 (11), 378x720 (8) and 404x720 (8).
- **Fake videos:** a face swap of each real clip, made with Remaker AI. The dataset uses one
  face-swap tool for all fakes.
- **Format-matched pairs:** each fake `vsN` comes from real `vN` and has the same resolution,
  frame rate, frame count and codec. So a detector can't separate the classes by file format,
  resolution or length; it has to respond to the manipulation itself.

## Setup

- **Hardware:** NVIDIA L4 (23 GB, driver 580.82.07) on Google Colab Pro.
- **Date:** 2026-10-26.
- **Code:** ForensiX commit `fa0381d85d9a272697d89e4ca03f609d6a377d89`. Scored with
  `scripts/eval_video_folder.py` as it is at that commit; no code changes were needed.
- **Software:** Python 3.13.16, torch 2.11.0+cu130, OpenCV 5.0.0, numpy 2.1.3, timm 1.0.30,
  transformers 5.20.2.
- **Image detectors:** each video gets 8 evenly spaced full frames. The frames are not
  face-cropped. Each frame goes through the model's own preprocessing, and the video score is
  the mean of the 8 frame scores. Most of these models score only a central crop of the frame
  without resizing it (`univfd`, `rine`: 224x224; `safe`: 256x256). On a 1280x720 frame, that crop
  covers only a small part of the image and may miss some or all of the face.
- **Video detectors:** each uses its own upstream frame sampling and face pipeline.
  - `gend`, `gend-pe`, `dfdet-clip`: MTCNN, largest face in 32 frames, 5-point alignment.
  - `dfdc-selim`: MTCNN, all faces in 32 frames, 7 EfficientNet-B7 models.
  - `dfdc-wm`: RetinaFace, largest face in every 10th frame, up to 25 faces.
- **Threshold:** every model's default threshold is 0.5 on `fake_prob`. Balanced, real and fake
  accuracy are measured at that threshold.
- **No-face fallback:** these video models return exactly 0.5 when no face is found. No video got
  that score from any model, so every video model found a face in every video.

## Results

Sorted by AUC. Fakes are the positive class.

| Model | Family | AUC [95% CI] | AP | EER | Bal. acc @0.5 | Real acc | Fake acc | Failed | Seconds |
|---|---|---|---|---|---|---|---|---|---|
| `gend-pe` | video | 0.956 [0.918, 0.984] | 0.959 | 0.132 | 0.859 | 0.755 | 0.962 | 0 | 714.0 |
| `gend` | video | 0.919 [0.863, 0.962] | 0.916 | 0.170 | 0.811 | 0.679 | 0.943 | 0 | 767.4 |
| `dfdet-clip` | video | 0.916 [0.860, 0.962] | 0.913 | 0.151 | 0.821 | 0.887 | 0.755 | 0 | 285.4 |
| `dfdc-selim` | video | 0.761 [0.670, 0.846] | 0.756 | 0.321 | 0.566 | 0.943 | 0.189 | 0 | 745.5 |
| `rine` | image | 0.745 [0.620, 0.812] | 0.774 | 0.340 | 0.500 | 1.000 | 0.000 | 0 | 24.1 |
| `univfd` | image | 0.719 [0.618, 0.811] | 0.751 | 0.302 | 0.509 | 1.000 | 0.019 | 0 | 25.7 |
| `clipdet` | image | 0.712 [0.608, 0.809] | 0.697 | 0.321 | 0.641 | 0.604 | 0.679 | 0 | 48.5 |
| `dfdc-wm` | video | 0.699 [0.597, 0.791] | 0.718 | 0.396 | 0.594 | 0.924 | 0.264 | 0 | 106.3 |
| `commfor` | image | 0.696 [0.592, 0.792] | 0.722 | 0.340 | 0.557 | 0.981 | 0.132 | 0 | 21.9 |
| `safe` | image | 0.693 [0.588, 0.792] | 0.723 | 0.377 | 0.500 | 1.000 | 0.000 | 0 | 10.0 |
| `commfor-224` | image | 0.642 [0.534, 0.743] | 0.630 | 0.396 | 0.509 | 0.981 | 0.038 | 0 | 13.9 |
| `rine-ldm` | image | 0.561 [0.452, 0.667] | 0.566 | 0.472 | 0.528 | 0.849 | 0.207 | 0 | 289.6 |

- **AUC 95% CI:** a bootstrap interval, with 5,000 resamples of real and fake videos drawn
  separately and seed 0. It is not in `results/sdfvd.json`.
- **AP:** average precision.
- **EER:** equal error rate.
- **Failed:** videos the script could not score (decode error or exception). These are left out
  of the metrics. No model failed on any video.
- **Seconds:** wall-clock time to score all 106 videos. It excludes model loading and weight
  downloads. **Only `univfd` and `rine` were timed alone on the GPU.** The other models ran as
  2–4 processes sharing the L4 and the 12 CPU cores, so their times are inflated. Four image
  models were also scored alone by accident: they took 1.2–1.5x less time alone, with identical
  scores (`rine-ldm` 200 s, `commfor` 16.5 s, `commfor-224` 11.5 s, `safe` 6.5 s). Use the column
  for rough relative cost only.

## Analysis

- **Only the three face-based CLIP/PE detectors work well here.**
  - `gend-pe`, `gend` and `dfdet-clip` reach AUC 0.92–0.96 and EER 0.13–0.17, and they are the
    only models above 0.8 balanced accuracy at the default threshold.
  - All three were trained on FaceForensics++ with aligned face crops, which matches this task.
  - Their AUC differences are within noise on this dataset. Paired bootstrap 95% CIs:
    `gend-pe` − `gend` +0.037 [−0.008, +0.087], `gend-pe` − `dfdet-clip` +0.040 [−0.008, +0.092],
    `gend` − `dfdet-clip` +0.003 [−0.028, +0.035].
  - `dfdet-clip` beats the next model, `dfdc-selim`, by +0.154 AUC [+0.058, +0.258].
- **The two DFDC winners are weak here.** `dfdc-selim` (AUC 0.761) and `dfdc-wm` (0.699) are
  face-based too, but they were trained on DFDC fakes, not on this tool's output. Neither is clearly better than the
  image detectors: for example, `dfdc-selim` − `rine` is +0.016 AUC [−0.092, +0.181].
- **Most models are miscalibrated at the default threshold.** They rank fakes above reals better
  than chance, but nearly all their scores fall below 0.5, so almost everything is called real:
  - `rine` and `safe` score every video below 0.5. They get 100% of reals and 0% of fakes right,
    giving exactly 0.500 balanced accuracy despite AUC 0.745 and 0.693. Their highest fake scores
    are 0.384 and 0.276.
  - `univfd` (AUC 0.719) catches 1.9% of fakes; `commfor-224` catches 3.8% and `commfor` 13.2%.
  - `dfdc-selim` catches 18.9% of fakes and `dfdc-wm` 26.4%, while getting 92–94% of reals right.
  - These models need a threshold chosen on in-domain data before their binary decisions are
    usable. AUC and EER are the fairer comparison for them.
- **`gend` and `gend-pe` lean the other way.** They catch 94–96% of fakes but get only 68% and
  76% of reals right, so they will raise false alarms on real faces at 0.5. `dfdet-clip` is more
  balanced at 0.5: it gets 89% of reals and 76% of fakes right.
- **`clipdet` is the best calibrated image detector** at 0.641 balanced accuracy. Its AUC (0.712)
  is in line with the other image detectors.
- **The image detectors are expected to be weak here.**
  - `univfd`, `rine`, `rine-ldm`, `commfor`, `commfor-224`, `safe` and `clipdet` were trained to
    flag whole images from GANs and diffusion models (ProGAN, Latent/Stable Diffusion and similar).
  - In a face swap, the background, body and most of the frame are untouched real camera footage,
    and only the face region is generated and blended in. That region is a small part of a
    1280x720 frame.
  - Several of these models score only a central crop without resizing, so they may not see the
    face at all.
  - Their AUCs of 0.56–0.75 show some weak signal. All of their 95% CIs reach down to between
    0.45 and 0.62, and `rine-ldm`'s includes 0.5. They should not be used as face-swap detectors
    based on this run.

## Caveats

- **Small sample.** With 53 real and 53 fake videos, the 95% CI on AUC is about ±0.03–0.05 for
  the top models and about ±0.1 for models near AUC 0.7 (see the table). Differences of a few AUC
  points are not meaningful, and the ranking from `dfdc-selim` down to `commfor` is not reliable.
  Accuracy at a threshold has binomial error of up to about ±0.13 per class (95%, n=53).
- **One face-swap tool.** All fakes come from Remaker AI on short Pexels clips.
  These results say nothing about other face-swap tools, reenactment, lip-sync, fully generated
  videos, or heavily compressed (social-media) footage.
- **Face detector substitution.** Upstream GenD uses the SCRFD face detector. ForensiX uses MTCNN
  for `gend`, `gend-pe` and `dfdet-clip` (see `forensix/video/gend/__init__.py`), so their face
  crops can differ slightly from upstream.
- **Image detectors on video.** Image detectors were applied to whole frames as described above.
  A face-crop protocol could give different numbers; this run does not test that.
- **Timing.** Times other than `univfd` and `rine` were measured with other models sharing the
  GPU (see the Results notes).
- **Run history.** The run was split across processes to use the whole GPU:
  - `univfd` and `rine` come from a sequential run of the command below.
  - The other ten models come from parallel runs of the same script with `--models` subsets and
    their own `--out` files, merged into `results/sdfvd.json`.
  - Then the command below was rerun; it skipped all 12 models and wrote the CSV.
  - Splitting doesn't change scores. `rine` was scored twice, and `rine-ldm`, `commfor`,
    `commfor-224` and `safe` were scored both alone and in parallel; per-video scores were
    identical every time. `results/sdfvd.log` has the full history.

## Reproduce

```bash
# from the repo root, with ForensiX installed (pip install -e .) and a CUDA GPU
mkdir -p /content/sdfvd && cd /content/sdfvd
curl -L -o sdfvd.zip "https://data.mendeley.com/public-files/datasets/bcmkfgct2s/files/a9ef71a3-b625-4682-a21c-f0579e85cdd4/file_downloaded"
echo "734a101fdfd89d5b20ac69f510d6a7cf3b588a145c1993ba2488322ae50d95a5  sdfvd.zip" | sha256sum -c
unzip -q sdfvd.zip && cd -

export FORENSIX_CACHE=/path/to/forensix_cache   # optional; about 6 GB of weights
python scripts/eval_video_folder.py --device cuda \
  --real /content/sdfvd/SDFVD/videos_real --fake /content/sdfvd/SDFVD/videos_fake \
  --dataset SDFVD \
  --models univfd rine rine-ldm commfor commfor-224 safe clipdet gend gend-pe dfdet-clip dfdc-wm dfdc-selim \
  --out results/sdfvd.json --scores-out results/sdfvd_scores.csv
```

The script resumes: models already in `--out` are skipped. To use more of a large GPU, run
several copies with disjoint `--models` lists and separate `--out` files, then merge their
`models` and `scores` entries. The metrics are unchanged.
