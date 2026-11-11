# Parameter-efficient fine-tuning (PEFT)

This guide is for engineers who want to adapt the large detectors (CLIP- and ViT-based image and
video models, wav2vec-style audio models) to their own media without training hundreds of
millions of parameters. It uses LoRA from Hugging Face's [PEFT](https://github.com/huggingface/peft)
library (`pip install peft`, Apache-2.0), and goes from development to a deployed service.

It builds on [Fine-tuning on your data](finetuning.md), which covers the data, evaluation,
calibration and deployment steps shared by every fine-tuning mode. This page covers what's
specific to LoRA.

## What LoRA does here

LoRA (low-rank adaptation) leaves each pretrained weight matrix `W` frozen and learns a small
correction `B·A` beside it, where `A` and `B` have rank `r` (4–16). The layer then computes with
`W + (alpha / r)·B·A`. ForensiX adds these adapters to every Linear layer except the classifier
head, and trains them together with the head.

| | `norm` (default) | **`lora`** | `full` |
|---|---|---|---|
| What changes | Normalization scales and shifts, plus the head | Low-rank corrections to every Linear layer, plus the head | Everything |
| Trained parameters, `dfdet-clip` (CLIP ViT-L) | ~0.1M | **1.8M at rank 4** | ~300M |
| Can learn | Shifts in the features' statistics | New feature directions: unseen generators, new artifacts | Anything, given enough data |
| Data needed per class | Hundreds | Hundreds to thousands | Many thousands |
| GPU memory | Low | Low: frozen weights need no gradients or optimizer state | High |
| Checkpoint, `dfdet-clip` | Under 1 MB | **7.1 MB at rank 4** | 1.2 GB |
| Serving cost | Same as the base | **Same as the base**: adapters are merged into the weights | Same as the base |

Use LoRA when `norm` stops improving and you have more data, or when the model has to pick up a
kind of manipulation the pretrained features don't separate. It's the closest thing to full
fine-tuning that fits on a Colab T4.

## Development

### 1. Install

```bash
pip install peft                     # or: pip install -e ".[finetune]"
```

PEFT is only needed to **train**. Loading and serving a LoRA checkpoint doesn't import it, so
deployment images stay as they are.

### 2. Data and baseline

Use the same three labelled sets (train, calibrate, test) and the same baseline as for any
fine-tuning: [Data](finetuning.md#1-data) and [Baseline](finetuning.md#2-baseline).

### 3. Train

```bash
python -m forensix.finetuning dfdet-clip --dataset train.zip --mode lora \
  --lora-rank 8 --lora-alpha 16 --epochs 5 --device cuda \
  --cache-dir face_cache/ --out models/dfdet-clip-lora.safetensors
```

```python
import forensix

det, report = forensix.finetune("univfd", dataset="train.zip", mode="lora", lora_rank=8, lora_alpha=16,
                                epochs=5, device="cuda", out="models/univfd-lora.safetensors")
report["base_val_auc"], report["val_auc"], report["lora"]   # {'rank': 8, 'alpha': 16, 'scaling': 2.0, 'n_targets': ...}
```

| Option | Default | Meaning |
|---|---|---|
| `--lora-rank` | 8 | Rank `r` of each adapter. Higher learns more and needs more data; 4–16 is the useful range |
| `--lora-alpha` | 16 | Scale numerator: corrections are multiplied by `alpha / r`. Keep `alpha ≈ 2r` and tune the learning rate instead |
| `--lora-dropout` | 0.05 | Dropout on the adapter input, against overfitting |
| `--lora-targets` | every Linear but the head | Regex of module names to adapt (below) |
| `--lr` | `2e-4` | Learning rate; `1e-4`–`5e-4` works for LoRA |

All other options (`--epochs`, `--batch-size`, `--val-fraction`, `--frames`, `--cache-dir`) work
as in [Fine-tune](finetuning.md#3-fine-tune). The held-out split is scored before training and
after every epoch. The best epoch is kept, and if no epoch beats the base model the checkpoint
leaves the base weights unchanged.

**Choosing targets.** To see a model's layer names:

```python
det = forensix.load("dfdet-clip")
[n for n, m in det.network.named_modules() if m.__class__.__name__ == "Linear"][:6]
# ['backbone.encoder.layers.0.self_attn.k_proj', '...v_proj', '...q_proj', '...out_proj', '...mlp.fc1', '...mlp.fc2']
```

| Goal | `--lora-targets` |
|---|---|
| Every Linear layer (default): most capacity | *(omit)* |
| Attention only, a common LoRA setting with fewer parameters | `'(q_proj\|v_proj)$'` |
| Only the last layers of a 24-layer ViT, for small datasets | `'layers\.(1[6-9]\|2[0-3])\.'` |

The CNN models (`safe`, `aasist` and others) have few Linear layers. For them, the default also
adapts the Conv2d layers. With `--lora-targets`, any matching Linear or Conv2d layer gets an
adapter.

**Hyperparameters that usually work:**
- Rank 8, alpha 16, learning rate `2e-4`, 3–10 epochs.
- If the held-out AUC climbs and then falls, use fewer epochs, rank 4, more dropout or more data.
- If it never climbs, raise the learning rate (`5e-4`) or the rank, or check that the training
  and held-out data come from comparable sources.

### 4. How the checkpoint works

1. PEFT injects the adapters (`peft.inject_adapter_in_model`), and only the adapters and the head
   are trained.
2. After the best epoch is restored, each adapter is **merged** into its layer
   (`W ← W + (alpha / r)·B·A`) and PEFT's wrapper is removed. The returned detector is a plain
   model running at the base model's speed.
3. The checkpoint stores the head and, for each adapted layer:
   - `A` and `B` for Linear layers, which is what keeps it small;
   - the merged correction for convolutions.

   The metadata records the rank, alpha and scaling.
4. `forensix.load(model, weights_path=checkpoint)` builds the pretrained model and adds the
   corrections with plain tensor math. **No PEFT is needed at load time**; the test suite checks
   this by loading with PEFT blocked.

```python
from forensix.finetuning import checkpoint_info
checkpoint_info("models/dfdet-clip-lora.safetensors")["lora"]   # {'rank': 8, 'alpha': 16, 'dropout': 0.05, 'scaling': 2.0}
```

### 5. Evaluate and calibrate

Same as any checkpoint. Evaluate on the test set with `--weights`, then calibrate the LoRA model
on the calibration set:

```bash
python scripts/eval_video_folder.py --real test/real --fake test/fake --models dfdet-clip \
  --weights dfdet-clip=models/dfdet-clip-lora.safetensors --device cuda --out results/test_lora.json
```

```python
cal, report = forensix.calibrate("dfdet-clip", dataset="calibration.zip", device="cuda",
                                 weights_path="models/dfdet-clip-lora.safetensors")
cal.save("models/dfdet-clip-lora.calibration.json")
```

Compare against the base model and, if you trained one, the `norm` checkpoint. LoRA is only
worth its larger checkpoint if it beats `norm` on the test set.

## Training on cloud GPUs

The published `gpu` image has ForensiX and PyTorch but not PEFT. Add it when you start the
container:

```bash
docker run --gpus all -e HOME=/tmp -v $PWD:/work -w /work ghcr.io/anas-rz/forensix:gpu \
  sh -c "pip install --user -q peft && python -m forensix.finetuning dfdet-clip --dataset train.zip \
         --mode lora --cache-dir face_cache --out models/dfdet-clip-lora.safetensors"
```

Run it on any GPU VM with NVIDIA drivers: AWS `g6.xlarge`, Google Cloud `g2-standard-8`, or Azure
`NC4as_T4_v3`. In Colab, `pip install peft` in a GPU runtime. Keep `--cache-dir` and `--out` on
persistent storage (Drive, a mounted bucket or a disk), because face-crop extraction is the slow
part and is reused across runs.

## Deployment

A LoRA checkpoint deploys exactly like any other fine-tune checkpoint, and the serving image
doesn't need PEFT:

1. **Serve locally:** add `"weights": {"dfdet-clip": "models/dfdet-clip-lora.safetensors"}` and the
   calibration to the server config ([details](finetuning.md#6-serve-it-locally)).
2. **Package:** build [`deploy/Dockerfile.finetuned`](../../deploy/Dockerfile.finetuned) on a
   `*-bundled` base. The adapters are applied on top of the pretrained weights baked into it.

    ```bash
    docker build -f deploy/Dockerfile.finetuned -t REGISTRY/forensix:gpu-lora \
      --build-arg BASE=ghcr.io/anas-rz/forensix:gpu-bundled \
      --build-arg WEIGHTS='{"dfdet-clip": "/opt/forensix/models/dfdet-clip-lora.safetensors"}' \
      --build-arg CALIBRATIONS='{"dfdet-clip": "/opt/forensix/models/dfdet-clip-lora.calibration.json"}' .
    docker push REGISTRY/forensix:gpu-lora
    ```

3. **Deploy** with your cloud's one command or one-click workflow, passing the image:

    | Cloud | Command |
    |---|---|
    | AWS | `deploy/aws/deploy.sh ecs-gpu --image ACCOUNT.dkr.ecr.REGION.amazonaws.com/forensix:gpu-lora` ([AWS](../aws.md)) |
    | Google Cloud | `deploy/gcp/deploy.sh cloudrun-gpu --image REGION-docker.pkg.dev/PROJECT/forensix/forensix:gpu-lora` ([Google Cloud](../gcp.md)) |
    | Azure | `deploy/azure/deploy.sh aca-gpu --image REGISTRY.azurecr.io/forensix:gpu-lora` ([Azure](../azure.md)) |

    [Deploy](finetuning.md#8-deploy) has the registry setup for each cloud.

The merge happens once at load, so latency and memory are the same as the base model's. When
you retrain, build a new image tag and redeploy, then [recalibrate](finetuning.md#5-calibrate):
the server refuses a calibration fitted to other weights.

## Beyond LoRA

PEFT has other methods (IA³, LoHa, LoKr, DoRA and more). `forensix.finetuning` supports LoRA,
because LoRA merges cleanly into the weights and keeps checkpoints small and PEFT-free at serving
time. To experiment with another method:
1. Inject it into `det.network` with PEFT yourself and train with your own loop.
2. Merge it (for example with the method's merge on each layer).
3. Save with `forensix.finetuning.save(det, path, "full", report)`. That gives a full checkpoint,
   which loads anywhere like the others.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `mode='lora' needs PEFT` | `pip install peft` in the training environment (not needed for serving) |
| "no Linear or Conv2d layers match" | The `--lora-targets` regex matches no layer names; list them as shown above |
| The held-out AUC drops after the first epoch | Overfitting: rank 4, `--lora-dropout 0.1`, fewer epochs, or more data. The best epoch is kept regardless |
| CUDA out of memory | Lower `--batch-size`, target attention only (`'(q_proj\|v_proj)$'`), or a lower rank. Training on CUDA already uses mixed precision |
| The loaded model scores like the base model | The run kept the base weights (see the report's `warnings`), or the server isn't using the checkpoint (`trained_weights` in `GET /v1/server`) |

**Next step:** once it's deployed, watch it in the [web manager's dashboard](../ui.md#the-overview-dashboard).
