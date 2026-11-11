# ForensiX documentation

ForensiX is a PyTorch model zoo for deepfake detection and media forensics. Every detector,
whether it checks audio, images, video, lip sync, physics or identity, sits behind the same small
Python API: `forensix.load(name).predict(media)` returns a `Prediction`.

These pages are for engineers who use ForensiX in their own systems. The
[top-level README](../README.md) is the model reference: per-model papers, licenses,
preprocessing details and parity checks against upstream code.

## Start here

| If you want to… | Read |
|---|---|
| Try everything in the browser, with no install: detect, calibrate, evaluate, web manager | [Colab notebook](https://colab.research.google.com/github/anas-rz/ForensiX/blob/main/notebooks/ForensiX_Get_Started.ipynb) |
| Go from a fresh checkout to the running server and web manager in ten minutes | [Quickstart](getting-started.md) |
| Install ForensiX and score your first file | [Installation](guides/installation.md) |
| Understand the building blocks (registry, `Prediction`, score types, families) | [Concepts](concepts.md) |
| Look up a function, class or field | [API reference](api-reference.md) |
| Compare models on commercially usable benchmarks | [Leaderboard](../LEADERBOARD.md) |
| Copy working code for a common task | [Recipes](recipes.md) |
| Call the detectors over HTTP | [HTTP server](server.md) |
| Manage, test and calibrate models in the browser | [Web manager](ui.md) |
| Calibrate models on your own labelled data, and get predictions with confidence | [Calibration](calibration.md) |
| Check C2PA credentials, generator watermarks and metadata | [Provenance](provenance.md) |
| Report false-alarm rates by skin tone, age and gender | [Fairness reports](fairness.md) |
| Run it in Docker, on Kubernetes or a managed cloud | [Docker, Kubernetes and cloud](cloud.md) |
| Deploy on AWS (ECS or EKS) with one command or one click | [Deploying on AWS](aws.md) |
| Run ForensiX as a service over large volumes of media | [Deployment at scale](deployment.md) |

## Guides

| Guide | What it covers |
|---|---|
| [Installation](guides/installation.md) | Python, ffmpeg, GPU, the weights cache, offline machines |
| [Choosing detectors](guides/choosing-detectors.md) | Which detector catches which manipulation; measured results |
| [Interpreting scores](guides/interpreting-scores.md) | `fake_prob`, thresholds, anomaly scores, calibration, localization, `details` |
| [Evaluating on your data](guides/evaluation.md) | The eval scripts, metrics, building a fair test set, reading benchmark numbers |
| [Identity and person models](guides/identity.md) | Enrolling people, verifying media against them, galleries, tiny per-person models |
| [Physics forensics](guides/physics.md) | The physics detectors, forensic reports, learned priors |
| [Training models](guides/training.md) | Training the models that ship without weights (avcons, avid, id-motion, physdf) |
| [Working in Google Colab](guides/colab.md) | GPU choice, Drive, GitHub, Claude Code in Colab |
| [Extending ForensiX](guides/extending.md) | Adding a model or a family, tests, parity checks |
| [Licensing and compliance](guides/licensing.md) | What the licenses allow, and what to check before shipping |

## Status at a glance

| Family | Models | State |
|---|---|---|
| audio | `aasist`, `aasist-l`, `rawgat-st`, `xlsr-aasist`, `xlsr-aasist-df`, `nes2net-x`, `nes2net-x-itw` | pretrained, parity-checked against upstream |
| image | `univfd`, `rine`, `rine-ldm`, `commfor`, `commfor-224`, `safe`, `clipdet` | pretrained, parity-checked |
| video | `gend`, `gend-pe`, `dfdet-clip`, `dfdc-selim`, `dfdc-wm` | pretrained, parity-checked, [benchmarked on SDFVD](../benchmark.md) |
| audiovisual | `avad`, `syncnet`, `stablesyncnet` | pretrained, parity-checked |
| audiovisual | `avcons`, `avid` | ForensiX models; you train them ([Training](guides/training.md)) |
| identity | `id-face` | pretrained (SFace) |
| identity | `id-motion` | ForensiX model; you train it |
| physics | `phys-bloodflow`, `phys-motion`, `phys-lighting`, `phys-voice`, `phys-room` | algorithmic; ranges provisional until learned priors are fitted |
| physics | `physdf` | runs without a graph prior; train the prior for the full system |
| specialized | `person-detector` | tiny per-person model you train in seconds |
| provenance | `c2pa`, `watermark-image`, `watermark-audio`, `metadata` | evidence in the file: Content Credentials, SD/SDXL and AudioSeal watermarks, metadata ([Provenance](provenance.md)) |
| multimodal, graph-based | none yet | planned |

`forensix.list_models()` always gives the current list.
