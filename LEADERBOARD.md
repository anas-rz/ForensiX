# ForensiX leaderboard

Only benchmarks whose license allows commercial use. Every number is recomputed from the result files in results/ by forensix/leaderboard.py; nothing is typed in by hand.

- **AUC 95% CI:** stratified bootstrap, 2000 resamples, seed 0 (Hanley-McNeil when per-item scores weren't saved).
- **Rates at a fixed threshold** (0.5 on `fake_prob`; physics: any violation): Wilson score interval, 95%. False alarm = real media flagged; detection = fakes caught.
- **Tied:** the model's interval overlaps the leader's, so the data can't separate them.
- Regenerate with `python scripts/build_leaderboard.py`.

## SDFVD face swaps

Face-swap video (whole frames for image detectors). License: CC BY 4.0. [Dataset](https://data.mendeley.com/datasets/bcmkfgct2s/1).

53 real Pexels clips and 53 Remaker AI face swaps of them, format-matched pairs (same resolution, frame rate, codec).

Hardware: NVIDIA L4 (Google Colab Pro) · date: 2026-10-26 · commit: `16649a37fd31`

| # | Model | AUC [95% CI] | EER | False alarms | Detection | s/item | Tied with #1 |
|---|---|---|---|---|---|---|---|
| 1 | `gend-pe` | 0.956 [0.919, 0.983] | 13.2% | 24.5% [14.9%, 37.6%] | 96.2% [87.2%, 99.0%] | 6.736 | — |
| 2 | `gend` | 0.919 [0.864, 0.961] | 17.0% | 32.1% [21.1%, 45.5%] | 94.3% [84.6%, 98.1%] | 7.24 | yes |
| 3 | `dfdet-clip` | 0.916 [0.861, 0.963] | 15.1% | 11.3% [5.3%, 22.6%] | 75.5% [62.4%, 85.1%] | 2.692 | yes |
| 4 | `dfdc-selim` | 0.761 [0.666, 0.844] | 32.1% | 5.7% [1.9%, 15.4%] | 18.9% [10.6%, 31.4%] | 7.033 | no |
| 5 | `rine`¹ | 0.745 [0.619, 0.812] | 34.0% | 0.0% [0.0%, 6.8%] | 0.0% [0.0%, 6.8%] | 0.227 | no |
| 6 | `univfd` | 0.719 [0.613, 0.811] | 30.2% | 0.0% [0.0%, 6.8%] | 1.9% [0.3%, 9.9%] | 0.242 | no |
| 7 | `clipdet` | 0.712 [0.608, 0.803] | 32.1% | 39.6% [27.6%, 53.1%] | 67.9% [54.5%, 78.9%] | 0.458 | no |
| 8 | `dfdc-wm` | 0.699 [0.598, 0.796] | 39.6% | 7.5% [3.0%, 17.9%] | 26.4% [16.4%, 39.6%] | 1.003 | no |
| 9 | `commfor` | 0.697 [0.595, 0.798] | 34.0% | 1.9% [0.3%, 9.9%] | 13.2% [6.6%, 24.8%] | 0.207 | no |
| 10 | `safe` | 0.693 [0.593, 0.790] | 37.7% | 0.0% [0.0%, 6.8%] | 0.0% [0.0%, 6.8%] | 0.094 | no |
| 11 | `commfor-224` | 0.642 [0.531, 0.748] | 39.6% | 1.9% [0.3%, 9.9%] | 3.8% [1.0%, 12.8%] | 0.131 | no |
| 12 | `rine-ldm` | 0.561 [0.448, 0.673] | 47.2% | 15.1% [7.8%, 27.1%] | 20.8% [12.0%, 33.5%] | 2.732 | no |

¹ `rine`: Saved per-item scores were rounded to 6 decimals, which ties this model's tiny scores; AUC and EER are the run's full-precision values (recomputed from saved scores: AUC 0.719). The interval comes from the rounded scores, so it's approximate.

Speed: Seconds per video on the L4. Only univfd and rine were timed alone; the other models shared the GPU with 1-3 other processes, so their times are inflated (upper bounds).

- 106 videos from one face-swap tool: differences of a few AUC points aren't meaningful (see the confidence intervals).
- Image detectors score 8 whole frames per video; they were built for whole generated images, not face swaps.

Reproduce:

```bash
python scripts/eval_video_folder.py --device cuda --real SDFVD/videos_real --fake SDFVD/videos_fake --dataset SDFVD --out results/sdfvd.json --scores-out results/sdfvd_scores.csv
```

## SDFVD face swaps, face crops

Face-swap video, image detectors on face crops. License: CC BY 4.0. [Dataset](https://data.mendeley.com/datasets/bcmkfgct2s/1).

The same 106 videos; image detectors score the largest face (MTCNN crop) in 8 frames per video.

Hardware: NVIDIA L4 (Google Colab Pro) · date: 2026-10-26 · commit: `not recorded`

| # | Model | AUC [95% CI] | EER | False alarms | Detection | s/item | Tied with #1 |
|---|---|---|---|---|---|---|---|
| 1 | `commfor` | 0.877 [0.810, 0.934] | 22.6% | 0.0% [0.0%, 6.8%] | 35.9% [24.3%, 49.3%] | 1.43 | — |
| 2 | `commfor-224` | 0.848 [0.767, 0.915] | 22.6% | 3.8% [1.0%, 12.8%] | 30.2% [19.5%, 43.5%] | 0.74 | yes |
| 3 | `rine`¹ | 0.844 [0.770, 0.916] | 18.9% | 0.0% [0.0%, 6.8%] | 1.9% [0.3%, 9.9%] | 1.423 | yes |
| 4 | `univfd` | 0.818 [0.734, 0.898] | 18.9% | 0.0% [0.0%, 6.8%] | 9.4% [4.1%, 20.2%] | 2.133 | yes |
| 5 | `safe` | 0.722 [0.617, 0.812] | 35.9% | 0.0% [0.0%, 6.8%] | 0.0% [0.0%, 6.8%] | 0.477 | yes |
| 6 | `rine-ldm` | 0.694 [0.593, 0.793] | 39.6% | 39.6% [27.6%, 53.1%] | 56.6% [43.3%, 69.0%] | 3.232 | no |
| 7 | `clipdet` | 0.678 [0.578, 0.776] | 37.7% | 71.7% [58.4%, 82.0%] | 98.1% [90.1%, 99.7%] | 2.022 | no |

¹ `rine`: Saved per-item scores were rounded to 6 decimals, which ties this model's tiny scores; AUC and EER are the run's full-precision values (recomputed from saved scores: AUC 0.847). The interval comes from the rounded scores, so it's approximate.

Speed: Seconds per video with 3 processes sharing one L4 (upper bounds).

- Same dataset caveats as SDFVD.

Reproduce:

```bash
python scripts/eval_video_folder.py --device cuda --face-crop --real SDFVD/videos_real --fake SDFVD/videos_fake --dataset SDFVD --models univfd rine rine-ldm commfor commfor-224 safe clipdet --out results/sdfvd_facecrop.json --scores-out results/sdfvd_facecrop_scores.csv
```

## ASVspoof 2019 LA (subset)

Spoofed and synthetic speech. License: ODC-By 1.0. [Dataset](https://datashare.ed.ac.uk/handle/10283/3336).

Fixed subset of the 2019 LA evaluation set: 1,000 bona fide trials and 77 from each of the 13 attacks A07-A19 (2,001 trials, seed 0).

Hardware: not recorded · date: not recorded · commit: `not recorded`

| # | Model | EER [≈95% CI] | Accuracy @0.5 | s/item |
|---|---|---|---|---|
| 1 | `xlsr-aasist-df` | 0.0% [0.0%, 0.2%] | 100.0% | 0.0292 |
| 2 | `nes2net-x-itw` | 0.1% [0.0%, 0.2%] | 96.0% | 0.0417 |
| 3 | `nes2net-x` | 0.2% [0.0%, 0.4%] | 99.8% | 0.029 |
| 4 | `xlsr-aasist` | 0.4% [0.1%, 0.7%] | 99.8% | 0.029 |
| 5 | `aasist-l` | 1.4% [0.9%, 1.9%] | 97.5% | 0.0133 |
| 6 | `aasist` | 1.6% [1.1%, 2.1%] | 96.8% | 0.0171 |
| 7 | `rawgat-st` | 3.1% [2.3%, 3.9%] | 98.0% | 0.0246 |

Speed: Seconds per trial, hardware not recorded.

- Per-trial scores weren't saved, so there's no AUC or false-alarm rate here: the EER interval is a binomial approximation. Rerun with scores saved for the full set of metrics.
- Older, in-domain attacks; the In-the-Wild-tuned models (nes2net-x-itw, xlsr-aasist-df) are the better choice for real-world audio.

Reproduce:

```bash
python scripts/eval_asvspoof2019_la.py --protocol LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt --audio-dir LA/ASVspoof2019_LA_eval/flac --device cuda --out results/asvspoof2019_la_subset.json
```

## SDFVD, physics detectors

Face-swap video, physics constraints (provisional priors). License: CC BY 4.0. [Dataset](https://data.mendeley.com/datasets/bcmkfgct2s/1).

Physics detectors on the SDFVD clips they could measure (47 real, 47 fake).

Hardware: not recorded · date: not recorded · commit: `not recorded`

| # | Model | AUC [95% CI] | EER | False alarms | Detection | s/item | Tied with #1 |
|---|---|---|---|---|---|---|---|
| 1 | `phys-lighting` | 0.481 [0.364, 0.598] | — | 100.0% [92.4%, 100.0%] | 97.9% [88.9%, 99.6%] | — | — |
| 2 | `phys-motion` | 0.446 [0.329, 0.562] | — | 42.5% [29.5%, 56.7%] | 31.9% [20.4%, 46.2%] | — | yes |

- Provisional (hand-set) ranges: no packaged video reference yet, since fitting one needs real video separate from this evaluation set.
- Cross-validated reference scoring on SDFVD itself fixes the false alarms but stays near chance: face swaps keep the source video's motion and lighting (physics_benchmark.md).

Reproduce:

```bash
python scripts/calibrate_physics.py eval --measurements results/physics/sdfvd_real.jsonl results/physics/sdfvd_fake.jsonl
```

## ASVspoof 2019 LA, physics detectors

Spoofed speech, voice and room physics (reference scoring). License: ODC-By 1.0. [Dataset](https://datashare.ed.ac.uk/handle/10283/3336).

phys-voice with the packaged reference: real-media distributions fitted on real ASVspoof 2019 LA train/dev speech plus LibriSpeech dev-clean (no fakes), combined with Fisher's method and calibrated out of fold. phys-room has no packaged reference and uses its provisional ranges. Evaluated on eval-set speakers never used in fitting.

Hardware: not recorded · date: not recorded · commit: `not recorded`

| # | Model | AUC [95% CI] | EER | False alarms | Detection | s/item | Tied with #1 |
|---|---|---|---|---|---|---|---|
| 1 | `phys-voice` | 0.679 [0.638, 0.719] | — | 0.0% [0.0%, 1.3%] | 4.1% [2.5%, 6.6%] | — | — |
| 2 | `phys-room` | 0.492 [0.354, 0.630] | — | 15.4% [6.2%, 33.5%] | 12.0% [5.6%, 23.8%] | — | no |

- phys-voice measures clips with at least 1.5 s of voiced speech (294 real + 369 fake of the 2,001); phys-room needs 6 s (76 clips).
- Fitting a reference on your own real media does better in-domain: fitted on ASVspoof train/dev speech alone, phys-voice reaches AUC 0.741 here (physics_benchmark.md).

Reproduce:

```bash
python scripts/calibrate_physics.py fit-reference --detectors phys-voice --measurements results/physics/asvspoof/fit_*.jsonl results/physics/librispeech/dev_clean.jsonl --out forensix/physics/reference.json && python scripts/calibrate_physics.py eval --measurements results/physics/asvspoof/eval_*.jsonl --out results/physics/asvspoof_eval_reference.json
```

## Deepfake-Eval-2024

Real-world deepfakes collected in 2024 (video, audio, image). License: CC BY-SA 4.0 (evaluation only, gated). [Dataset](https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024).

**Not run yet.** Reproduce:

```bash
python scripts/eval_deepfake_eval_2024.py --data /content/de2024 --modality video --device cuda --max-per-class 250 --out results/de2024_video.json --scores-out results/de2024_video_scores.csv
```

## ForensiX image set

Generated vs real images (SDXL, PixArt-Σ, Kandinsky 2.2, FLUX.1-schnell). License: Open Images CC BY 2.0 photos; generators whose outputs allow commercial use.

**Not run yet.** Reproduce:

```bash
python scripts/build_image_eval_set.py --out /content/forensix_image_eval && python scripts/eval_image.py --data /content/forensix_image_eval --device cuda --out results/image_eval.json --scores-out results/image_scores.csv
```

## Speed

Items per second scoring SDFVD (video models) or ASVspoof (audio models): the plain per-file loop, and the batch runner in fp32 and bf16.

| Model | Items | Per-file loop | Batch runner fp32 | Batch runner bf16 | bf16 max score change |
|---|---|---|---|---|---|
| `aasist` | 400 | 67.28/s | 67.85/s | 77.99/s | 0.0397 |
| `xlsr-aasist` | 400 | 33.91/s | 34.36/s | 82.84/s | 0.0082 |
| `commfor` | 106 | 6.55/s | 13.19/s | 18.94/s | 0.0604 |
| `univfd` | 106 | 4.31/s | 5.25/s | 16.19/s | 0.0187 |
| `gend` | 40 | 0.58/s | 0.55/s | 0.77/s | 0.0110 |

## Not included

- **LAV-DF** (CC BY-NC 4.0): Non-commercial license. Results exist in results/lavdf/ but aren't on this leaderboard.
