# Fine-tuning on your data

This guide takes a pretrained detector from your labelled data to a service on AWS, Google Cloud
or Azure. It covers fine-tuning, measuring the result, calibrating it, packaging it and deploying
it. It's for engineers whose media differ from what the public models were trained on: a new
generator, a camera pipeline, a language, a compression chain.

To train the models that ship *without* weights (`avcons`, `avid`, `id-motion`, `physdf`,
`person-detector`), see [Training models](training.md) instead.

## Fine-tune or calibrate?

| Problem | Fix | Needs |
|---|---|---|
| Scores rank real and fake well on your data, but thresholds or probabilities are off | [Calibrate](../calibration.md) | Hundreds of labelled items; no training |
| The model misses a generator or manipulation it has never seen, or flags your real media | **Fine-tune**, then calibrate | Hundreds to thousands of labelled items per class, a GPU |

Measure first (step 2): if the AUC on your data is already high, calibration alone is enough.

## The whole path

1. **Data:** three labelled sets in the [ForensiX dataset format](../calibration.md#the-dataset-format):
   one to train on, one to calibrate, and one to test.
2. **Baseline:** score the test set with the pretrained model.
3. **Fine-tune:** `python -m forensix.finetuning MODEL --dataset train.zip --out MODEL-ft.safetensors`.
4. **Evaluate:** score the test set again with the checkpoint, and compare.
5. **Calibrate** the fine-tuned model on the calibration set.
6. **Serve it locally** with the checkpoint and calibration.
7. **Package** both into an image on top of the published one.
8. **Deploy** that image to AWS, Google Cloud or Azure with the usual one command or one click.

## What can be fine-tuned

| Family | Models | What a training sample is |
|---|---|---|
| Image | `commfor`, `commfor-224`, `univfd`, `rine`, `rine-ldm`, `safe`, `clipdet` | An image, with light augmentation (flips, mild JPEG recompression) |
| Audio | `aasist`, `aasist-l`, `rawgat-st`, `nes2net-x`, `nes2net-x-itw`, `xlsr-aasist`, `xlsr-aasist-df` | A random window of a clip, with random gain |
| Video | `gend`, `gend-pe`, `dfdet-clip` | An aligned face crop. Crops are extracted once per video and cached; a video's score is the mean over its faces, as in production |

Other video, audiovisual, physics and identity models aren't supported by the fine-tuner. Most
of them are calibrated rather than trained, or have their own training scripts.

**Modes** (`--mode`) set how much of the network changes:

| Mode | Trains | Use when | Default learning rate |
|---|---|---|---|
| `head` | The final classifier layer | A few hundred items per class; a quick, safe first try | `1e-3` |
| **`norm`** (default) | Normalization layers' scales and shifts, plus the head ("LN-tuning", as GenD and `dfdet-clip` were trained) | Hundreds to a few thousand items per class | `1e-4` |
| `full` | Every parameter | Many thousands of items per class and a large GPU | `1e-5` |

`head` and `norm` checkpoints store only the tensors they changed, from kilobytes to a few
megabytes. They load on top of the pretrained weights, so the base model's license and weights
still apply.

## 1. Data

Use the same format as calibration: a ZIP or folder with `real/` and `fake/` folders, or a
`manifest.csv` with `file,label,group` ([format](../calibration.md#the-dataset-format)).

- **Three separate sets.** Fine-tuning holds out part of its data to choose the best epoch,
  which makes that part optimistic. Calibrate on a second set and report results on a third that
  neither step has seen.
- **Groups.** Put a real clip and the fakes made from it, or one person's or one session's media,
  in the same `group`. The fine-tuner splits by group, so near-duplicates can't leak between
  training and held-out data. Split your three sets by group too.
- **Match production.** Use the same sources, resolutions, compression and platforms as the
  media you'll score. Get real media from the same channels as the fakes, or the model learns
  the channel instead of the manipulation.
- **Balance.** Classes are reweighted automatically, but each class still needs variety: many
  generators and many sources.
- **Licenses.** Your training data's license carries into the checkpoint's use. For commercial
  use, train only on data you're allowed to use commercially (see
  [Licensing and compliance](licensing.md)).

## 2. Baseline

Score the test set with the pretrained model:

```bash
python scripts/eval_video_folder.py --real test/real --fake test/fake --models dfdet-clip \
  --device cuda --out results/test_base.json
```

For images and audio, `forensix.calibrate("commfor", dataset="test.zip")` reports the AUC, or use
`scripts/eval_image.py`. If the AUC is already high, calibrate instead of fine-tuning.

## 3. Fine-tune

```bash
python -m forensix.finetuning dfdet-clip --dataset train.zip --out models/dfdet-clip-ft.safetensors \
  --mode norm --epochs 5 --device cuda --cache-dir face_cache/
```

```python
import forensix

det, report = forensix.finetune("commfor", dataset="train.zip", out="models/commfor-ft.safetensors",
                                mode="norm", epochs=5, device="cuda")
report["base_val_auc"], report["val_auc"], report["best_epoch"], report["warnings"]
```

| Option | Default | Meaning |
|---|---|---|
| `--dataset` or `--real` / `--fake` | | A dataset ZIP or folder, or two folders of files |
| `--mode` | `norm` | `head`, `norm` or `full` (above) |
| `--epochs` | 5 | Passes over the training data. The best epoch on held-out data is kept |
| `--lr` | by mode | Learning rate (AdamW) |
| `--batch-size` | 16 | Lower it if the GPU runs out of memory |
| `--val-fraction` | 0.2 | Share held out (by group) to score each epoch |
| `--max-per-class` | all | Cap for a quick trial run |
| `--frames` | 16 | Video: frames sampled per video for face crops |
| `--cache-dir` | temporary | Video: keep face crops so reruns skip extraction (put it on persistent storage) |
| `--no-augment` | | Turn augmentation off |

**What it does:**
- It first scores the held-out part with the base model, then again after every epoch, and
  keeps the best epoch.
- If no epoch beats the base model, the checkpoint keeps the base weights and the report says so.
- Next to the checkpoint it writes `MODEL-ft.json`, with the AUC before and after, every
  epoch's loss and held-out AUC, the item counts and warnings.

**Where to run it:**

| Where | How |
|---|---|
| Google Colab | `pip install -e .` in a GPU runtime, as in the [Colab guide](colab.md). Keep `--cache-dir` and `--out` on Drive |
| Your GPU machine | `pip install -e .`, then the command above |
| A cloud GPU VM, from the published image | `docker run --gpus all -v $PWD:/work -w /work ghcr.io/anas-rz/forensix:gpu python -m forensix.finetuning commfor --dataset train.zip --out models/commfor-ft.safetensors` on an AWS `g6.xlarge`, a GCP `g2-standard-8` or an Azure `NC4as_T4_v3` VM with NVIDIA drivers (each cloud's deep learning VM images include them) |

The image runs as user 10001, so the mounted folder must be writable by it
(`chmod a+w models face_cache`).

## 4. Evaluate

Score the same test set with the checkpoint, writing to a separate results file:

```bash
python scripts/eval_video_folder.py --real test/real --fake test/fake --models dfdet-clip \
  --weights dfdet-clip=models/dfdet-clip-ft.safetensors --device cuda --out results/test_ft.json
```

Compare the AUC, the false-alarm rate on real media and the detection rate with the baseline.
Also check what the model must keep doing well, such as public benchmarks
([Evaluating on your data](evaluation.md)): fine-tuning on one generator can cost accuracy on
others. If you have demographic labels, run a [fairness report](../fairness.md) on real faces
before and after.

## 5. Calibrate

Fit the calibration on the calibration set, loading the fine-tuned model:

```python
cal, report = forensix.calibrate("dfdet-clip", dataset="calibration.zip", device="cuda",
                                 weights_path="models/dfdet-clip-ft.safetensors")
cal.save("models/dfdet-clip-ft.calibration.json")
```

A calibration belongs to one set of weights. Recalibrate whenever you fine-tune again. Once
deployed, you can instead calibrate in the web manager; the server fingerprints the weights, so
a calibration fitted to other weights is refused.

## 6. Serve it locally

```json
{
  "device": "cuda",
  "preload": ["dfdet-clip", "commfor"],
  "weights": {"dfdet-clip": "models/dfdet-clip-ft.safetensors"},
  "calibrations": {"dfdet-clip": "models/dfdet-clip-ft.calibration.json"}
}
```

```bash
python -m forensix.server --config server.json        # or FORENSIX_SERVER_WEIGHTS='{"dfdet-clip": "..."}'
```

In Python, use `forensix.load("dfdet-clip", weights_path=..., calibration=...)`.

## 7. Package

[`deploy/Dockerfile.finetuned`](../../deploy/Dockerfile.finetuned) adds your `models/` folder to
a published image and points the server at it:

```bash
docker build -f deploy/Dockerfile.finetuned -t forensix:gpu-ft \
  --build-arg BASE=ghcr.io/anas-rz/forensix:gpu-bundled \
  --build-arg WEIGHTS='{"dfdet-clip": "/opt/forensix/models/dfdet-clip-ft.safetensors"}' \
  --build-arg CALIBRATIONS='{"dfdet-clip": "/opt/forensix/models/dfdet-clip-ft.calibration.json"}' \
  .                     # the folder that contains models/
```

Use a `*-bundled` base (`cpu-bundled` for CPU targets), because the checkpoint loads on top of
the pretrained weights baked into it.

## 8. Deploy

Push the image to your cloud's registry, then deploy it with the same one command or one-click
workflow as the stock image, passing the image:

| Cloud | Push | Deploy |
|---|---|---|
| AWS | `aws ecr create-repository --repository-name forensix`; `aws ecr get-login-password \| docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.REGION.amazonaws.com`; tag and push `ACCOUNT.dkr.ecr.REGION.amazonaws.com/forensix:gpu-ft` | `deploy/aws/deploy.sh ecs-gpu --image ACCOUNT.dkr.ecr.REGION.amazonaws.com/forensix:gpu-ft` ([AWS](../aws.md)) |
| Google Cloud | `gcloud artifacts repositories create forensix --repository-format docker --location REGION`; `gcloud auth configure-docker REGION-docker.pkg.dev`; tag and push `REGION-docker.pkg.dev/PROJECT/forensix/forensix:gpu-ft` | `deploy/gcp/deploy.sh cloudrun-gpu --image REGION-docker.pkg.dev/PROJECT/forensix/forensix:gpu-ft` ([Google Cloud](../gcp.md)) |
| Azure | `az acr create -g RG -n REGISTRY --sku Basic`; `az acr login -n REGISTRY`; tag and push `REGISTRY.azurecr.io/forensix:gpu-ft` | `deploy/azure/deploy.sh aca-gpu --image REGISTRY.azurecr.io/forensix:gpu-ft`, after giving the apps or cluster pull access ([Azure](../azure.md)) |

The GitHub Actions workflows (`deploy-aws`, `deploy-gcp`, `deploy-azure`) take the same image in
their `image` input. Redeploying with a new image rolls the services over to it. The shared data
(datasets, calibration versions made in the web manager) stays.

## Retraining over time

- **Detect drift:** when new labelled media arrives, run a drift check in the web manager
  (**Calibration → Check for drift**) or with `scripts/recalibrate.py`.
- **Recalibrate first:** if the ranking still holds, recalibrating is enough.
- **Fine-tune again:** if the AUC on new data drops, fine-tune again on the old and new training
  data together. Give the checkpoint a new name, repeat steps 4–8, and keep each run's
  `MODEL-ft.json` report with the image tag it went into.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| "No epoch beat the base model" | The base model already does well on this data (calibrate instead), the training data is too small or too different from the held-out part, or the learning rate is off. Try `head` mode, more data, or `--lr` 3× lower or higher |
| Held-out AUC rises, then falls | Overfitting. Fewer epochs (the best one is kept anyway), `head` instead of `norm`, or more data |
| CUDA out of memory | Lower `--batch-size`, or use `head`/`norm` instead of `full` |
| Video: "no face in ..." | The face detector found no face in the sampled frames. Those videos are skipped; check that they show a face |
| The served model behaves like the base model | The server didn't get the checkpoint: `GET /v1/server` lists the models with custom weights under `trained_weights`; also check that the path exists inside the container |
| "fine-tunes X, not Y" | The checkpoint belongs to a different model; checkpoints only load into the model they were trained from |
