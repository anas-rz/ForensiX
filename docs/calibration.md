# Calibrating models on your data

A detector's raw score isn't a probability you can act on. On SDFVD, `rine` called every video
real at its default threshold, while `gend` flagged a third of real videos. Calibration fixes the
*meaning* of the score for your data, without retraining the model:

- **Calibrated probability:** "0.92" means about 92% of media scoring like this were fake in your
  labelled data (Platt scaling).
- **A false-alarm guarantee:** the decision threshold flags at most `alpha` (e.g. 5%) of real
  media like yours (split-conformal threshold, fitted on your real items only).
- **A verdict with confidence:** `fake`, `real`, `uncertain` (the score fits both classes) or
  `out_of_distribution` (the score fits neither), from conformal p-values.

Calibration doesn't make a model better at telling real from fake: AUC is unchanged. If a model
can't separate your data (the report's AUC is near 0.5), you need a different model, not a
calibration.

The workflow:
1. Put your labelled media in the [standard dataset format](#the-dataset-format).
2. Calibrate, from Python ([`forensix.calibrate`](#python)) or over HTTP ([the server](#http-api)).
3. Read the report: AUC, out-of-fold false-alarm and detection rates, warnings.
4. Predict: calibrated scores and a confidence block on every prediction.
5. Check for drift on newly labelled data, and recalibrate when it fails.

## The dataset format

A dataset is a ZIP archive (or a folder) in one of two layouts.

**Manifest** (recommended): a `manifest.csv` at the root, with one row per file:

```text
my_dataset/
├── manifest.csv
└── media/
    ├── interview7.mp4
    ├── interview7_swap.mp4
    └── ...
```

```csv
file,label,group,identity
media/interview7.mp4,real,interview-7,
media/interview7_swap.mp4,fake,interview-7,
photos/alice_01.jpg,real,,alice
```

| Column | Required | Meaning |
|---|---|---|
| `file` | yes | Path relative to the dataset root |
| `label` | yes | `real` or `fake`. Also accepted: `bonafide`, `genuine`, `authentic`, `0` / `spoof`, `deepfake`, `synthetic`, `manipulated`, `1` |
| `group` | no | Files that belong together: a real clip and fakes made from it, or one recording session. Cross-validation keeps a group in one fold, so near-duplicates can't flatter the report. Default: each file alone |
| `identity` | no | The enrolled person the media claims to be, for identity models |
| `skin_tone`, `age`, `gender`, `attr_<name>` | no | Demographic attributes for [fairness reports](fairness.md) |

**Folders:** no manifest. Every file under `real/` is real and every file under `fake/` is fake
(subfolders allowed).

```text
my_dataset/
├── real/...
└── fake/...
```

Zip it with the folder at the top level or one level down; both work:

```bash
cd parent_dir && zip -r my_dataset.zip my_dataset
```

### What makes a good calibration set

- **Media like what you'll score:** same sources, platforms, compression and content. A
  calibration only holds for media exchangeable with it.
- **Enough real items.** The conformal threshold needs at least `1/alpha − 1` real items
  (19 for 5%, 99 for 1%), or it never flags; the report warns you. A few hundred real items give
  stable rates.
- **Fakes from the generators you expect.** The false-alarm guarantee needs no fakes at all, but
  the probabilities and the detection rate do.
- **Groups for related files,** e.g. SDFVD's `v12.mp4` and its face swap `vs12.mp4`.
- **Separate data for checking.** Keep newly labelled media for drift checks rather than adding it
  to the calibration set straight away.

Uploads are validated: unknown labels, missing files, duplicates and files outside the dataset
are listed as `problems` and skipped; a dataset with no usable items is rejected.

If a model ranks your real and fake media poorly (a low AUC), calibration can't fix that:
[fine-tune it](guides/finetuning.md) on your data first, then calibrate the fine-tuned model.

## Python

```python
import forensix

cal, report = forensix.calibrate("dfdet-clip", dataset="my_dataset.zip", device="cuda", alpha=0.05)
print(report["auc"], report["cv"]["conformal"], report["warnings"], report["skipped"])
cal.save("calibrations/dfdet-clip.json")

det = forensix.load("dfdet-clip", device="cuda", calibration="calibrations/dfdet-clip.json")
pred = det.predict("new.mp4")
pred.fake_prob                     # calibrated probability
pred.label                         # decision at the conformal threshold (≤ alpha false alarms)
pred.details["prediction_set"]     # ["fake"], ["real"], ["real", "fake"] or []
pred.details["p_real"], pred.details["p_fake"], pred.details["raw_score"]
```

`forensix.calibrate` also takes `real_media=[...]` and `fake_media=[...]` lists instead of a
dataset, `folds` and `seed` for cross-validation, and `score_type="raw"` for scores that aren't
probabilities (detected automatically for anomaly models such as `syncnet`). For scores you've
already computed (e.g. `scripts/eval_video_folder.py --scores-out`), use
`forensix.calibration.calibrate_scores(model, scores, is_fake, groups)` or
`scripts/recalibrate.py`.

To build on an existing calibration with newly labelled scores of the same model:

```python
from forensix.calibration import Calibration, update_calibration

base = Calibration.load("calibrations/dfdet-clip.json")
new, report = update_calibration(base, new_fake_scores, new_real_scores, window=2000)
new.save("calibrations/dfdet-clip.json")    # report: n_new_*, n_carried_*, AUC, CV, warnings
```

`window` caps the scores kept per class: all new scores stay, and older ones are subsampled.
Without it, every old score is kept. The HTTP API below does the same with item-level
bookkeeping: deduplication by file and lineage.

## HTTP API

All calibration data lives under the server's `calibration_dir`. Set it, or datasets and
calibrations are lost on restart. Changing calibrations needs the admin key.

**1. Upload the dataset:**

```bash
curl -H "X-API-Key: $ADMIN" -F file=@my_dataset.zip -F name=platform-x-oct \
     http://localhost:8000/v1/datasets
```

```json
{"id": "9249deec143c", "name": "sdfvd25", "layout": "manifest", "n_items": 50, "n_real": 25,
 "n_fake": 25, "n_groups": 25, "with_identity": 0, "problems": [], "n_problems": 0}
```

**2. Start a calibration job** for one or more models:

```bash
curl -H "X-API-Key: $ADMIN" -H "content-type: application/json" \
     -d '{"mode": "full", "datasets": ["9249deec143c"], "models": ["dfdc-wm", "dfdet-clip"], "alpha": 0.05}' \
     http://localhost:8000/v1/calibrations
```

There are three modes ([Calibration modes](#calibration-modes) has the details):

| `mode` | Fits on | Use it for |
|---|---|---|
| `full` (default) | The given datasets, from scratch | A first calibration; a clean rebuild on everything you have |
| `incremental` | The base version's items plus the given datasets | Adding newly labelled data on top of previous calibrations |
| `refit` | The base version's items only; nothing is scored | Changing `alpha` or `folds` |

| Field | Default | Meaning |
|---|---|---|
| `mode` | `"full"` | `full`, `incremental` or `refit` |
| `datasets` | required for `full` and `incremental` | Dataset ids (`dataset` takes one id) |
| `models` | required | Models to calibrate (fusion names work too) |
| `base_version` | the active version, else the latest | `incremental`/`refit`: the version to build on |
| `window` | none (keep everything) | `incremental`: max items per class. All new items stay; older ones are randomly subsampled |
| `only_if_drift` | `false` | `incremental`: refit only if the new data fails the drift check against the base version |
| `level` | `0.01` | Significance level of that drift check |
| `alpha` | the base's, else `0.05` | Target false-alarm rate on real media |
| `folds` | `5` | Cross-validation folds for the report |
| `seed` | `0` | Seed for folds and window subsampling |
| `activate` | `true` | Use the new version for predictions as soon as it's fitted |
| `score_types` | auto | `{"model": "raw"}` for scores that aren't probabilities |

The job scores every item with each model's raw (uncalibrated) output. Image and audio models
batch the items, and videos are bounded like uploads. Scores are cached per dataset and model
configuration, so a dataset is scored once per model however many calibrations use it. The job
then fits, saves a new version and (by default) activates it. It returns `202` with a job id.

**3. Poll the job:**

```bash
curl -H "X-API-Key: $KEY" http://localhost:8000/v1/jobs/19a78650a8ef
```

While running, `progress` shows `{"model": "dfdc-wm", "scored": 12, "total": 50}`. When done,
`results` has one report per model:

Real output: `dfdc-wm` calibrated on 25 real and 25 face-swapped SDFVD clips (pairs grouped), on CPU:

```json
{
 "status": "done", "version": 1, "activated": true, "model": "dfdc-wm",
 "n_real": 25, "n_fake": 25, "n_skipped": 0, "score_type": "probability", "alpha": 0.05,
 "auc": 0.6656, "ap": 0.6891,
 "cv": {
  "raw":       {"threshold": 0.5, "balanced_acc": 0.58, "false_alarm_rate": 0.12, "detection_rate": 0.28, "ece": 0.3214},
  "platt":     {"balanced_acc": 0.60, "false_alarm_rate": 0.40, "detection_rate": 0.60, "ece": 0.0652},
  "conformal": {"alpha": 0.05, "balanced_acc": 0.56, "false_alarm_rate": 0.04, "detection_rate": 0.16},
  "prediction_sets": {"coverage_real": 0.96, "coverage_fake": 0.96, "single_label": 0.2, "both_labels": 0.8, "empty": 0.0}
 },
 "conformal_threshold": 0.7326,
 "warnings": ["fewer than 50 items in a class: probabilities and rates are rough"]
}
```

How to read it:
- **Probabilities:** calibration error (ECE) fell from 0.32 to 0.07.
- **False alarms:** the conformal threshold held them at 4% against a 5% target.
- **Ranking:** AUC 0.67 says `dfdc-wm` separates this data poorly. That's why 80% of the
  prediction sets hold both labels: calibration made the model honest about its uncertainty, but
  didn't make it better. For this data, calibrate a stronger model (`gend-pe` reached AUC 0.96 on
  SDFVD).

| Report field | Meaning |
|---|---|
| `version`, `activated` | The saved version and whether it's now in use |
| `mode`, `parent`, `lineage` | How it was built: the mode, the version it built on, and every dataset in its history |
| `n_new`, `n_carried`, `n_dropped_old` | Items from this job's datasets, items carried over from the base version, and old items left out by `window` or replaced by a newer copy of the same file |
| `n_real`, `n_fake`, `n_skipped`, `skipped` | Items scored, and items the model couldn't score (wrong media type, no face, unreadable) |
| `auc`, `ap` | How well the model separates your data. Calibration doesn't change these |
| `cv.raw` | Out-of-fold results at the model's default threshold |
| `cv.platt` | Out-of-fold results deciding at calibrated probability ≥ 0.5, plus Brier score, log loss and ECE |
| `cv.conformal` | Out-of-fold results at the conformal threshold: `false_alarm_rate` should be near `alpha` |
| `cv.prediction_sets` | Coverage of the prediction sets, and how often they're single, double or empty |
| `conformal_threshold` | Raw score above which media is flagged |
| `warnings` | Too few real items for `alpha`, small classes, low AUC |

**4. Manage versions.** Every fit is a new version; nothing is overwritten.

```bash
curl -H "X-API-Key: $KEY" http://localhost:8000/v1/calibrations/dfdc-wm                 # versions, active one
curl -H "X-API-Key: $KEY" http://localhost:8000/v1/calibrations/dfdc-wm/versions/1      # table + report
curl -X PUT -H "X-API-Key: $ADMIN" -H "content-type: application/json" \
     -d '{"version": 1}' http://localhost:8000/v1/calibrations/dfdc-wm/active           # roll back
curl -X PUT -H "X-API-Key: $ADMIN" -H "content-type: application/json" \
     -d '{"version": null}' http://localhost:8000/v1/calibrations/dfdc-wm/active        # raw scores again
```

A version's `table` is the standard calibration JSON. Save it to a file and it works with
`forensix.load(name, calibration=path)`, so a calibration fitted on the server can be used in
batch jobs too.

**5. Check for drift** when you have newly labelled media. Upload it as another dataset, then:

```bash
curl -H "X-API-Key: $ADMIN" -H "content-type: application/json" \
     -d '{"dataset": "<new dataset id>"}' http://localhost:8000/v1/calibrations/check
```

The job scores the new data and tests it against each active calibration: false alarms above
`alpha`, shifted real or fake scores, a dropped detection rate. `recalibrate: true` means at least
one test failed. Then calibrate on data that includes the new media. If the new data's AUC has
dropped, the model itself no longer works on it, and recalibrating won't help.

## Calibration modes

Every job creates a new version. Each version stores the exact items it was fitted on (file
hash, label, group, raw score, source dataset), so later versions can build on it.

**Full.** Fits from scratch on the given datasets: one, or several at once (`"datasets": [a, b]`).
A file that appears in two datasets is used once; a file labelled both real and fake is left out,
with a warning. Use it for the first calibration of a model, and periodically to rebuild on
exactly the data you trust.

**Incremental.** Starts from a base version: `base_version`, else the active version, else the
latest. It combines the base's items with the new datasets' items:

- **Duplicates:** a file the base already had (same SHA-256) takes its new label and score, so
  re-uploading corrected labels fixes them rather than doubling the file.
- **`window`:** keeps at most that many items per class: all new items, plus a random subset of
  the old. The calibration then follows changing traffic instead of being dominated by old data.
  Without `window`, the calibration set only grows.
- **`only_if_drift`:** checks the new data against the base version first. If none of the drift
  tests fail, no version is created and the result is `"status": "skipped"`, with the drift
  report. Use this for scheduled updates: upload each week's labelled media and recalibrate only
  when it's needed.
- **Settings:** `alpha` and the score type default to the base version's.

```bash
# weekly: add the newest labelled media, keep the 2,000 most recent items per class,
# and only create a new version if the old one no longer fits
curl -H "X-API-Key: $ADMIN" -H "content-type: application/json" \
     -d '{"mode": "incremental", "datasets": ["<this week>"], "models": ["dfdet-clip"],
          "window": 2000, "only_if_drift": true}' \
     http://localhost:8000/v1/calibrations
```

**Refit.** Re-fits the base version's items with new settings, such as a stricter `alpha`
(`{"mode": "refit", "models": ["dfdet-clip"], "alpha": 0.01}`). Nothing is scored, so it takes
seconds.

**Lineage.** `GET /v1/calibrations/{model}` lists every version with its `mode`, `parent` and
datasets. For example (illustrative values):

```json
{"version": 7, "mode": "incremental", "parent": 6, "datasets": ["oct", "nov", "dec"],
 "n_real": 1840, "n_fake": 1020, "n_new": 412, "n_carried": 2448, "auc": 0.93, "alpha": 0.05}
```

**Same model only.** Raw scores are only combined when they come from the same model
configuration: the same registered weights or checkpoint file, and the same load options. Each
version records this as `fingerprint`. If you change a model's checkpoint or options, an
incremental or refit job on an older version fails with a clear error. Run a `full` calibration
instead; the datasets are rescored with the new configuration.

## Predictions with confidence

With a calibration in use, every `/v1/detect` result for that model reads:

Real output for two SDFVD clips that weren't in the calibration set (`dfdc-wm`, version 1 above):

```jsonc
// v30.mp4 (real)
{"score": 0.531, "threshold": 0.669, "label": "real", "calibration": "v1",
 "confidence": {"fake_probability": 0.531, "verdict": "uncertain", "prediction_set": ["real", "fake"],
                "p_real": 0.423, "p_fake": 0.5, "alpha": 0.05, "calibration": "v1"}}
// vs30.mp4 (its face swap)
{"score": 0.707, "threshold": 0.669, "label": "fake", "calibration": "v1",
 "confidence": {"fake_probability": 0.707, "verdict": "fake", "prediction_set": ["fake"],
                "p_real": 0.038, "p_fake": 0.962, "alpha": 0.05, "calibration": "v1"}}
```

| Field | Meaning |
|---|---|
| `score` | Calibrated fake probability, at the calibration set's fake rate |
| `threshold`, `label` | Decision at the conformal threshold: at most `alpha` of real media like the calibration set is labelled fake |
| `calibration` | Version in use (`"v3"`, or `"config"` for a table from the server config) |
| `confidence.verdict` | `fake` or `real`: the score fits only that class. `uncertain`: it fits both (the model can't tell). `out_of_distribution`: it fits neither (unlike the calibration data, so check it by hand) |
| `confidence.p_real`, `confidence.p_fake` | Conformal p-values: how typical the score is among real and among fake calibration items |
| `confidence.prediction_set` | The classes with p-value above `alpha`. Each true class is included with probability at least 1 − `alpha` |

`label` and `verdict` answer different questions. `label` is the decision with a false-alarm
guarantee; `verdict` says how sure the evidence is. A good triage rule: act automatically only
on `verdict: fake`, send `uncertain` and `out_of_distribution` to a person, and pass `real`.

These guarantees hold for media exchangeable with the calibration data. When your traffic
changes, run a drift check.

**Next step:** check that false alarms are even across groups with a [fairness report](fairness.md).
