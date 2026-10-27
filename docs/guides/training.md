# Training models

Most ForensiX models ship with pretrained weights. Five ship without, because no commercially
usable weights exist and ForensiX trains its own:

| Model | Learns | From | Training script | Where to run |
|---|---|---|---|---|
| `avcons` | Whether audio and lips belong together | Real talking-face videos only | `train_avcons.py` | Colab GPU |
| `avid` | Whether the voice belongs to the face | Real talking-face videos, many speakers | `train_avid.py` | Colab GPU |
| `id-motion` | A person's motion and expression style | Real videos, several per person | `train_id_motion.py` | Colab (CPU extract, GPU train) |
| `physdf` graph prior | How face, pulse, light and voice relate | Real talking-face videos | `train_physdf.py` | Colab (CPU extract, GPU train) |
| physics priors | Real ranges of each physics constraint | Real media | `calibrate_physics.py` | Anywhere |
| `person-detector` | One person | That person's media | `train_person_model.py` or Python | Your laptop (CPU, seconds) |

All of them train on **real media only**, with negatives synthesized from it. They learn what
real looks like rather than one generator's fingerprints, and they avoid deepfake datasets,
most of which have non-commercial terms.

## Data

VoxCeleb2 (CC BY 4.0) fits most of these models: talking faces with sound, from thousands of
speakers. Video copyright stays with the YouTube owners.

| Split | Speakers | Use for |
|---|---|---|
| VoxCeleb2 dev | 5,994 | `avid`, `id-motion` (the number of people matters more than hours) |
| VoxCeleb2 test | 118 | `avcons`, `physdf`, physics priors |

Lay it out as `<person>/<video>/<clip>.mp4` (VoxCeleb's own layout). The scripts split train and
held-out sets by person, so held-out results measure unseen people.

## The two-step pattern

Every training script has two subcommands:

1. **`extract`** runs the slow, frozen parts (face tracking, SyncNet, ECAPA, MediaPipe) once
   per clip and caches compact features. It's resumable: rerun the same command after a
   disconnect.
2. **`train`** trains the small model on the cache, reports held-out metrics, and writes a
   `.safetensors` checkpoint.

Keep the cache on Google Drive or another persistent disk: extraction is the expensive part.

## avcons

```bash
python scripts/train_avcons.py extract --media /content/voxceleb2_test/mp4 \
    --out /content/drive/MyDrive/avcons_cache --max-files 5000 --device cuda
python scripts/train_avcons.py train --cache /content/drive/MyDrive/avcons_cache \
    --out avcons.safetensors --steps 20000 --device cuda
```

- **Extraction** drops clips with unclear sync (SyncNet confidence under 3, e.g. off-screen
  speech) and re-aligns the rest to their own sync.
- **Negatives:** constant offsets of 6–15 frames, 5% speed drift, another speaker's voice or
  face, and spliced segments.
- **`train` output:** AUC per manipulation on held-out speakers.

## avid

```bash
python scripts/train_avid.py extract --media /content/voxceleb2/mp4 \
    --out /content/drive/MyDrive/avid_cache --max-files 20000 --device cuda
python scripts/train_avid.py train --cache /content/drive/MyDrive/avid_cache \
    --out avid.safetensors --device cuda
```

- **Speaker split:** speakers are divided three ways: heads (contrastive voice–face space),
  scorer (calibrated against swaps and splices), and a report split.
- **`train` output:** AUC per negative kind, and 1:2 voice-to-face matching accuracy (about 80% is
  the published level for this task).

## id-motion

```bash
python scripts/train_id_motion.py extract --media /content/voxceleb2/mp4 \
    --out /content/drive/MyDrive/idmotion_cache --max-files 30000 --workers 8
python scripts/train_id_motion.py train --cache /content/drive/MyDrive/idmotion_cache \
    --out id_motion.safetensors --steps 20000 --device cuda
```

- **Extraction** is CPU-bound (MediaPipe), so raise `--workers` to the number of CPU cores.
- **`train` output:** the verification EER on held-out people. Its threshold becomes the model's
  decision threshold.

## physdf

```bash
python scripts/train_physdf.py extract --media /content/voxceleb2_test/mp4 \
    --out /content/drive/MyDrive/physdf_cache --max-files 3000 --workers 8
python scripts/train_physdf.py train --cache /content/drive/MyDrive/physdf_cache \
    --out physdf_graph.safetensors --steps 20000 --device cuda
```

- **Extraction** (MediaPipe plus the ICT lighting fit) is the slow, CPU-bound part.
- **Training** is masked modelling: node spans, whole nodes, the mouth or a whole modality are
  hidden, and the model predicts them from the rest.
- **Checkpoint:** stores the feature normalization and the real residual distributions, per input
  condition (audio+video, audio only, video only).

Fit the physics priors too ([Physics guide](physics.md#calibrate-before-you-rely-on-it)).

## Using your checkpoints

```python
forensix.load("avcons", weights_path="avcons.safetensors")
forensix.load("avid", weights_path="avid.safetensors")
forensix.load("id-motion", weights_path="id_motion.safetensors")
forensix.load("physdf", weights_path="physdf_graph.safetensors")
```

Evaluation scripts take them with `--weights NAME=PATH`:

```bash
python scripts/eval_deepfake_eval_2024.py --data /content/de2024 --modality video \
    --models avcons syncnet avad --weights avcons=avcons.safetensors --device cuda \
    --max-per-class 250 --out results/de2024_av.json
```

## Before you train

- **Run a smoke test.** Before a long run, run with `--max-files 50 --steps 200` to check the
  data layout and the full pipeline in a few minutes.
- **Check the held-out report** that `train` prints. If held-out AUC or EER is far worse than
  training, look for too few speakers or leaks between splits.
- **Record what you trained on** (dataset, split, file count, steps, commit) next to the
  checkpoint. You'll need it for audits and for licensing questions.
- **Shipping trained weights:** a checkpoint you train is yours, but check the training data's
  terms. VoxCeleb2's CC BY 4.0 requires attribution ([Licensing](licensing.md)).
