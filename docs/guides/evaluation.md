# Evaluating on your data

Published accuracy rarely survives a change of data. Before trusting a detector, measure it on
media like yours. This guide covers the evaluation scripts, the metrics, and how to build a test
set that measures what you think it measures.

## Metrics

All scripts use `forensix.metrics`, with fakes as the positive class:

| Metric | Answers | Notes |
|---|---|---|
| **AUC** (ROC-AUC) | How often does a random fake score above a random real? | Threshold-free. 0.5 is chance. The main number for comparing models. |
| **AP** (average precision) | Precision averaged over recall levels | Sensitive to class balance |
| **EER** | The error rate where false alarms equal misses | Threshold-free; standard in audio anti-spoofing (ASVspoof-compatible implementation) |
| **Balanced accuracy** | Mean of real and fake accuracy at the model's default threshold | Shows whether the default threshold suits your data |
| **Real acc / fake acc** | Share of reals called real / fakes called fake at the threshold | 1 − real acc is the false-alarm rate |

```python
from forensix.metrics import binary_summary, compute_eer
binary_summary(fake_scores, real_scores, threshold=0.5)
# {'auc': ..., 'ap': ..., 'balanced_acc': ..., 'real_acc': ..., 'fake_acc': ..., 'n_fake': ..., 'n_real': ...}
eer, eer_threshold = compute_eer(fake_scores, real_scores)
```

## Evaluation scripts

| Script | Data | Models |
|---|---|---|
| `eval_video_folder.py` | Any two folders of videos (real, fake) | video, image (on frames) |
| `eval_deepfake_eval_2024.py` | [Deepfake-Eval-2024](https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024) (in-the-wild, gated, CC BY-SA 4.0) | any family; `--modality video/audio/image` |
| `eval_asvspoof2019_la.py` | ASVspoof 2019 LA eval (ODC-By) | audio; pooled and per-attack EER |
| `build_image_eval_set.py` + `eval_image.py` | ForensiX's license-clean image set (Open Images + 4 permissive generators) | image; per generator, PNG and JPEG conditions |
| `eval_identity.py` | `references/<person>/`, `queries/<person>/{real,fake}/` | identity; real vs fake and real vs impostor |
| `calibrate_physics.py eval` | Stored physics measurements | physics; per detector and per constraint |

All of them write a JSON of metrics, and most also write a per-file scores CSV, which you need
for thresholds and fusion. The folder script is the most general:

```bash
python scripts/eval_video_folder.py --device cuda \
    --real data/real --fake data/fake --dataset my-set \
    --models gend-pe dfdet-clip commfor \
    --out results/my_set.json --scores-out results/my_set_scores.csv
```

It saves after each model and skips models already in `--out`, so a dropped session resumes
where it stopped. To use a big GPU fully, run several copies with disjoint `--models` lists and
separate `--out` files, then merge them; the scores don't change.

## Building a fair test set

Most surprising benchmark results come from the test set, not the model.

- **Match formats between classes.** If all fakes are PNG and all reals JPEG, a detector can
  score well by spotting PNG. SDFVD pairs each fake with its source clip at the same
  resolution, frame rate and codec; the ForensiX image set re-encodes fakes at the reals' JPEG
  quality. Check codec, resolution, duration, bitrate and container per class before trusting
  any number.
- **Match content.** Fakes of talking heads against reals of landscapes measure scene
  recognition. Use reals and fakes of the same kind of content, ideally the same sources.
- **Keep sources apart.** Don't let the same person, video or recording session appear in
  both data used to choose thresholds and data used to report results.
- **Cover several generators.** One face-swap tool says little about the next one. Report per
  generator when you can.
- **Process like production.** If production media is re-encoded by a platform, re-encode the
  test set the same way.
- **Check licenses.** For commercial work, prefer test sets with commercially usable licenses:
  SDFVD (CC BY 4.0), ASVspoof 2019 (ODC-By), Deepfake-Eval-2024 (CC BY-SA 4.0, evaluation only),
  or the ForensiX image set. FaceForensics++, Celeb-DF, DFDC and FakeAVCeleb have research-only
  terms.

## How much data you need

Confidence intervals shrink slowly. On SDFVD (53 + 53 videos), the 95% interval on AUC was about
±0.04 for the best models and about ±0.1 for models near AUC 0.7, so differences of a few
points meant nothing. Rough guide:

| Per class | AUC 95% CI at AUC 0.9 (Hanley–McNeil) | What it can tell apart |
|---|---|---|
| 50 | ±0.06 | Good vs. useless models |
| 250 | ±0.03 | Models 5+ AUC points apart |
| 1,000 | ±0.014 | Models 2–3 AUC points apart |

Bootstrap your own intervals: resample reals and fakes separately, with replacement, and
recompute the metric a few thousand times. To compare two models, bootstrap the *difference* on
the same resamples (a paired bootstrap), as `benchmark.md` does.

## Reporting

Write results up like [benchmark.md](../../benchmark.md):
- the dataset, its license and how the fakes were made;
- the hardware, date, ForensiX commit and software versions;
- AUC with confidence intervals, EER, and accuracy at the threshold;
- the exact reproduce command;
- caveats: sample size, generators covered, protocol choices.

Keep the raw per-file scores. They let you recompute any metric later without rerunning the
models.
