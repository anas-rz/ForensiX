# Audio-visual detectors: LAV-DF

This report evaluates the three audio-visual detectors with released weights (`syncnet`, `stablesyncnet`, `avad`) on LAV-DF. It also calibrates them on one split and checks the calibration on another. `avcons` and `avid` aren't included, because they have no released weights: you train them on your own real videos.

Raw results:
- `results/lavdf/lavdf_test.json`, `results/lavdf/lavdf_dev.json` (metrics);
- `results/lavdf/{test,dev}/*.jsonl.gz` (per-clip and per-window scores, gzipped);
- `results/lavdf/lavdf_{test,dev}_scores.csv` (one row per clip);
- `results/lavdf/calibration_lavdf_dev.json` and `calibrations/lavdf/` (calibrations).

## Summary

- **The two lip-sync models detect LAV-DF fakes moderately well.**
  - Clip AUC: `stablesyncnet` 0.785 [0.753, 0.819], `syncnet` 0.722 [0.688, 0.757].
  - They catch edits to the audio, the video and both at similar rates, which fits a sync check: any word edit breaks lip-sync.
- **Score a clip by its most suspicious window, not the mean.**
  - LAV-DF edits are short (median 0.66 s in a 7 s clip), and the mean dilutes them.
  - This rule was chosen on the dev split and then tested on test. It raises test AUC to 0.853 for `stablesyncnet` (+0.068 [+0.043, +0.093], paired) and 0.833 for `syncnet` (+0.110 [+0.078, +0.143]).
  - With this rule, the two models' AUCs are within noise of each other (difference +0.017 [−0.013, +0.048]).
- **The windows roughly locate the edit.** Windows that overlap a fake segment score above windows of real clips with AUC 0.83–0.85.
- **`avad` is near chance** (0.563 [0.525, 0.601]).
  - Following upstream, it scores only the first 50 frame-steps (about 2.2 s) of the face track. Most LAV-DF edits come later, and it can't see them (AUC 0.50–0.55 on those).
  - Even on edits inside that window, it reaches only about 0.6, and about 0.5 on audio-only edits.
- **Calibration transferred from dev to test for `stablesyncnet` but not for `syncnet`.**
  - `stablesyncnet` with the worst-window score: 2.5% false alarms at a 5% target, ECE 0.03.
  - `syncnet` flagged 10.7–12.0% of test reals, against the 5% target. Its real scores shift with clip length, and that length mix differs between splits.
  - `scripts/recalibrate.py check` flags exactly these two tables for recalibration.

## Dataset

- **Name:** LAV-DF, "Localized Audio Visual DeepFake" (Cai et al., DICTA 2022).
- **Link:** <https://huggingface.co/datasets/ControlNet/LAV-DF>
- **License:** CC BY-NC 4.0, non-commercial research and education only. It's built from VoxCeleb2, whose terms also apply. Access is gated on Hugging Face.
- **Archive:** `LAV-DF.tar`, 25,607,198,720 bytes, sha256 `2abc0646cf1469f0bebec60a792fec8b6d670cd9e6133105139110c4316b20a4`. Downloaded on 2026-10-26 to `/content/lavdf`, outside the repo.
- **Content:** 136,304 VoxCeleb2 talking-head clips. Each fake replaces one or two words, with content-driven edits generated in the visual track, the audio track or both. Fake segments last 0.2–1.1 s (5th–95th percentile; median 0.66 s). Test clips last 4.4–16.8 s (median 7.1 s).
- **Subsets** (sampled with `scripts/eval_lavdf.py`, seed 0, equal numbers per class):

| Split | Real | Audio edited | Video edited | Both edited | Used for |
|---|---|---|---|---|---|
| test | 300 | 300 | 300 | 300 | all reported results |
| dev | 200 | 200 | 200 | 200 | choosing the clip score; fitting calibrations |

## Setup

- **Hardware:** NVIDIA L4 (23 GB), 12 CPU cores, Google Colab Pro.
- **Date:** 2026-10-26.
- **Code:** ForensiX `124205e` plus the uncommitted changes listed under "Code changes" below.
- **Runtime:** 9 processes (3 per model) sharing the GPU, at about 7 s per clip per process, so about 1.3 clips/s in total. A single process takes about 4.5 s per clip.
- **Scores** (higher = more likely fake):

| Model | Clip score | Windows |
|---|---|---|
| `syncnet` | −LSE-C confidence | per 25 fps frame (median-filtered confidence) |
| `stablesyncnet` | 1 − mean sync probability | non-overlapping 0.64 s windows |
| `avad` | anomaly score (KL divergence) | none reported |

- **Skipped clips:**
  - `stablesyncnet` refuses clips whose face is tracked in under 90% of frames. It skipped 87 of 1,200 test clips (7%: 23 real, 24 audio, 21 video, 19 both) and 64 of 800 dev clips.
  - Its numbers cover only the clips it scored. In deployment, a skipped clip gets no answer.
  - The other two models scored every clip.

## Results (test, 1,200 clips)

| Model | Clip AUC [95% CI] | AP | EER | Audio edited | Video edited | Both edited | Worst-window clip AUC | Window AUC |
|---|---|---|---|---|---|---|---|---|
| `stablesyncnet` | 0.785 [0.753, 0.819] | 0.899 | 0.285 | 0.807 [0.771, 0.842] | 0.784 [0.744, 0.823] | 0.766 [0.726, 0.804] | **0.853** [0.828, 0.878] | 0.834 |
| `syncnet` | 0.722 [0.688, 0.757] | 0.872 | 0.343 | 0.751 [0.714, 0.788] | 0.704 [0.663, 0.745] | 0.711 [0.671, 0.751] | **0.833** [0.810, 0.856] | 0.847 |
| `avad` | 0.563 [0.525, 0.601] | 0.772 | 0.463 | 0.518 [0.472, 0.563] | 0.578 [0.533, 0.623] | 0.593 [0.547, 0.641] | – | – |

- **CIs:** stratified bootstrap with 2,000 resamples (reals and fakes drawn separately), seed 0.
- **AP:** 900 of the 1,200 clips are fakes, so a random detector scores AP 0.75. Compare APs with that baseline.
- **Per-modality columns:** each kind of fake is compared against the same 300 reals.
- **Window AUC:** windows that overlap a fake segment, against all windows of real clips. Windows from the same clip are correlated, so no CI is given.

**Choosing the clip score.** The clip score was chosen on dev before looking at the test result for that choice:

| Model | Dev: mean | Dev: worst window | Test: mean | Test: worst window | Test gain, paired [95% CI] |
|---|---|---|---|---|---|
| `syncnet` | 0.760 | 0.870 | 0.722 | 0.833 | +0.110 [+0.078, +0.143] |
| `stablesyncnet` | 0.810 | 0.851 | 0.786 | 0.853 | +0.068 [+0.043, +0.093] |

The detectors' own `fake_prob` is still the mean, so parity with upstream is unchanged. The worst-window score is computed from the per-window details that both detectors already return.

**`avad` and its 2.2 s limit.** Its 50-step limit is upstream's own (`detect.py` `test2`, `max_len=50`), and ForensiX matches upstream on this (see the README's Verification table).

| Test fakes | n | AUC vs reals |
|---|---|---|
| edit starts in the first 2.16 s | 351 | 0.592 |
| … audio edited only | 113 | 0.505 |
| … video edited | 122 | 0.627 |
| … both edited | 116 | 0.639 |
| edits only after 2.16 s | 549 | 0.545 |

## Calibration (fit on dev, evaluated on test)

**Fitting.** `scripts/recalibrate.py fit` fitted one table per score on the dev clips, with α = 5% (`calibrations/lavdf/`).
- `syncnet`'s scores and `avad`'s anomaly scores aren't probabilities, so they use Platt scaling on the raw score. `recalibrate.py` detects this automatically when scores fall outside [0, 1]; this support was added for this evaluation.
- The test split was used only for the evaluation below.

| Score | Test false alarms (target 5%) | Test detection | Bal. acc, Platt | ECE, Platt | Real coverage | Fake coverage | Single-label sets |
|---|---|---|---|---|---|---|---|
| `stablesyncnet` worst window | **2.5%** | 36.5% | 0.723 | 0.030 | 97.5% | 93.5% | 45.0% |
| `stablesyncnet` | 7.2% | 27.0% | 0.677 | 0.048 | 92.8% | 94.1% | 36.8% |
| `syncnet` worst window | 10.7% | 65.8% | 0.656 | 0.050 | 89.3% | 95.1% | 62.6% |
| `syncnet` | 12.0% | 32.6% | 0.587 | 0.047 | 88.0% | 97.1% | 34.7% |
| `avad` | 4.0% | 4.4% | 0.500 | 0.001 | 96.0% | 95.3% | 9.7% |

**`syncnet`'s threshold doesn't carry over to test.**
- The conformal guarantee holds only if new real clips are exchangeable with the calibration reals. For `syncnet` they aren't.
- Its real-clip scores differ between dev and test (KS p = 0.015). The score depends on clip length (Spearman ρ = −0.36), and the test reals are slightly shorter (median 5.9 s vs 6.4 s).
- `recalibrate.py check` on the test scores flags `syncnet` and `syncnet-max-window` for recalibration (exit status 3). The `stablesyncnet` and `avad` tables pass.
- In deployment, calibrate `syncnet` on clips with the same length mix as the traffic, or run `check` on a labelled sample regularly.

**Other points.**
- **Platt probabilities are well calibrated on test** (ECE 0.03–0.05), even where the conformal threshold isn't.
- **`stablesyncnet`'s own 0.5 threshold** gives 0.53 balanced accuracy on dev (it calls almost every clip real). Platt scaling raises that to 0.68 on test.
- **`avad`'s calibration is "correct" but useless:** its probabilities stay near 0.5 and it detects almost nothing.

## Code changes

- **`forensix/calibration.py`:** a `score_type` field (`"probability"` or `"raw"`) for calibrating scores that aren't probabilities. Existing tables load as `"probability"` and behave as before. Added test: `tests/test_calibration.py::test_raw_scores_are_calibrated_without_logit`.
- **`scripts/recalibrate.py`:** picks the score type per model automatically, or with `--raw-scores`. Raw-score models get no "raw" probability metrics in the CV report.
- **`forensix/audiovisual/stablesyncnet`:** the detail `window_start_s` gives each window's start time. The window-selection rule moved into `window_starts()`, used by both scoring and the details, so scores are unchanged (`tests/test_audiovisual.py` passes).
- **`scripts/eval_lavdf.py` (new):** sampling, resumable sharded scoring, the report and calibration checks.

## Caveats

- **LAV-DF is one kind of fake.** Edits are short word replacements in VoxCeleb2 interviews. These results say nothing about face swaps, full-clip lip-sync dubbing, or other generators.
- **License.** LAV-DF is non-commercial. These numbers can be shared for research, but the data can't be used to build a commercial product.
- **Model training data.** `stablesyncnet` was trained on VoxCeleb2, the source of LAV-DF's real clips, so it may have seen some of these speakers or videos. `syncnet` (BBC television) and `avad` (LRS2) weren't trained on it.
- **`stablesyncnet`'s skips** (7% of clips) aren't counted as errors in its AUC.
- **Sample size.** 300 clips per class gives AUC CIs about ±0.035 wide. Per-modality differences within a model aren't significant.

## Reproduce

```bash
# accept the terms at https://huggingface.co/datasets/ControlNet/LAV-DF, then: hf auth login
hf download ControlNet/LAV-DF LAV-DF.tar --repo-type dataset --local-dir /content/lavdf
echo "2abc0646cf1469f0bebec60a792fec8b6d670cd9e6133105139110c4316b20a4  /content/lavdf/LAV-DF.tar" | sha256sum -c
mkdir -p /content/lavdf/extracted && tar -xf /content/lavdf/LAV-DF.tar -C /content/lavdf/extracted
D=/content/lavdf/extracted/LAV-DF
# score (run shards in parallel: --shard 0/3, 1/3, 2/3 per model)
python scripts/eval_lavdf.py score --data $D --split test --per-class 300 --models syncnet stablesyncnet avad --out-dir results/lavdf
python scripts/eval_lavdf.py score --data $D --split dev --per-class 200 --models syncnet stablesyncnet avad --out-dir results/lavdf
python scripts/eval_lavdf.py report --data $D --split dev --out-dir results/lavdf
# calibrate on dev, evaluate on test, check for drift
python scripts/recalibrate.py fit --scores results/lavdf/lavdf_dev_scores.csv --alpha 0.05 --source lavdf-dev \
    --out-dir calibrations/lavdf --report results/lavdf/calibration_lavdf_dev.json
python scripts/eval_lavdf.py report --data $D --split test --out-dir results/lavdf --calibration-dir calibrations/lavdf
python scripts/recalibrate.py check --calibration-dir calibrations/lavdf --scores results/lavdf/lavdf_test_scores.csv
```
