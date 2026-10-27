# Choosing detectors

No detector catches every manipulation. Choose by **what was manipulated** and **what
evidence is left in the media**, then confirm on data like yours.

## By threat

| Threat | What changed | Detectors that can see it | Detectors that can't |
|---|---|---|---|
| Face swap in a video | The face region; the rest is real footage | `gend-pe`, `gend`, `dfdet-clip` (best measured), `dfdc-selim`, `dfdc-wm`; `id-face` / `id-motion` if you have references; `person-detector` | Lip-sync models (the mouth stays in sync); image detectors (only a small part of the frame is generated) |
| Fully generated image (diffusion, GAN) | Every pixel | `commfor` (broadest training), `rine-ldm` (diffusion), `clipdet`, `univfd`, `rine`, `safe` | Face models, unless the image is a face |
| Cloned or converted voice | The speech | `nes2net-x-itw`, `xlsr-aasist-df` (real-world audio), `xlsr-aasist`, `aasist`, `rawgat-st`; `phys-voice`, `phys-room` | Face and image models |
| Dubbed video, voice replaced | The audio track | `syncnet`, `stablesyncnet`, `avcons`, `avad`, `avid` (voice doesn't fit the face); the audio models | Face models |
| Lip-sync generation (Wav2Lip-style) | The mouth region, driven by new audio | `avad`, `avcons`, face models; `syncnet` and `stablesyncnet` only if the sync is imperfect | Generators trained against a SyncNet can pass the sync models |
| Voice swap + lip-sync | Audio and mouth | `avid` (identity mismatch), audio models, `id-face` with references | Sync models (the sync is good) |
| Impersonation of a known person | The identity | `id-face`, `id-motion`, `person-detector` | Detectors without references can't know who it should be |
| Spliced or edited recording | A segment | Window/segment scores of audio models, `phys-room` (room or microphone change), `avcons` localization, `physdf` localization | Clip-level scores alone |

## Measured results

### Face-swap video: SDFVD

Full write-up: [benchmark.md](../../benchmark.md). 53 real and 53 face-swapped clips (one
face-swap tool), on an L4 GPU.

| Model | AUC | EER | Balanced acc. at 0.5 | Comment |
|---|---|---|---|---|
| `gend-pe` | 0.956 | 0.132 | 0.859 | Best ranking; flags 24% of reals at 0.5 |
| `gend` | 0.919 | 0.170 | 0.811 | Flags 32% of reals at 0.5 |
| `dfdet-clip` | 0.916 | 0.151 | 0.821 | Best balanced at 0.5; fastest of the three |
| `dfdc-selim` | 0.761 | 0.321 | 0.566 | Catches 19% of fakes at 0.5 |
| `rine` (image, whole frames) | 0.745 | 0.340 | 0.500 | Calls everything real at 0.5 |
| `dfdc-wm` | 0.699 | 0.396 | 0.594 | |
| other image models | 0.56–0.72 | | | Not face-swap detectors |

The top three are within statistical noise of each other on 106 videos. The gap between them
and everything else is real.

### Other families

Audio, image, audiovisual and identity models have published results (see the
[README Models tables](../../README.md#models)), and their implementations are checked against
upstream code, but they haven't yet been benchmarked by ForensiX on held-out data. The
evaluation scripts are ready: [Evaluating on your data](evaluation.md).

## Practical defaults

| Situation | Start with |
|---|---|
| Social-media video of a face | `dfdet-clip` (balanced at 0.5) or `gend-pe` (best ranking, set your own threshold) |
| Video call or interview with sound | `gend-pe` + `nes2net-x-itw` + `syncnet` ([recipe 4](../recipes.md#4-check-a-talking-head-video-across-modalities)) |
| Phone call or voice note | `nes2net-x-itw` |
| Image upload | `commfor` |
| Protecting specific people | `id-face` with enrolled references, plus a face-manipulation model |
| Explaining a decision to a non-expert | `physdf` report and `syncnet` offset/confidence, as supporting evidence (with calibrated priors) |

## Cost

Measured on an NVIDIA L4 during the SDFVD run. Times are per video (2–6 s clips), and most were
measured with other models sharing the GPU, so they are upper bounds:

| Model | Seconds per video | Weights on disk |
|---|---|---|
| image models (8 frames) | 0.06–0.5 (`rine-ldm` 2–2.7) | 5 MB – 1.6 GB (CLIP backbones are shared) |
| `dfdc-wm` | ~1 | 312 MB |
| `dfdet-clip` | ~2.7 | 1.2 GB + CLIP |
| `gend-pe`, `gend` | ~7 (shared GPU) | 1.2 GB each |
| `dfdc-selim` | ~7 (shared GPU) | 1.8 GB |

On CPU, expect around 1–2 minutes per short video for the CLIP-based video models and about
6 s for `dfdc-wm`.
