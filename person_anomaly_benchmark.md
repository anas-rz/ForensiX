# Per-person detectors trained on real media only

This report asks one question: given only *real* media of one person, which small model best tells
their real media from deepfakes of them? It compares the options on held-out people, then measures
the method ForensiX ships, `person-anomaly`, and the [browser studio](docs/guides/studio.md) that
uses it.

## Summary

- **What works:** the distance from the person's average embedding under a frozen, pretrained
  identity encoder: SFace for faces, ECAPA-TDNN for voices. Fitting it means averaging the
  person's embeddings, which takes under a second.
- **What doesn't:** small convolutional autoencoders (105k–234k parameters) trained on the person's
  face crops or voice spectrograms scored near chance, whether trained from scratch or fine-tuned
  from a base trained on other people's real media. Adding one to the identity score lowered it.
  The previous `person-detector` scored below the identity encoders on both modalities.
- **Thresholds:** a threshold set from the person's own media flagged a third of their held-out real
  frames. The encoders' published population thresholds flagged 0–2% of real media, including
  real speech from a different recording session.
- **Shipped as `person-anomaly`** (AUC [95% CI]; with the default thresholds):

| Data | AUC | Real flagged | Fakes caught | Fakes caught at 5% false alarms |
|---|---|---|---|---|
| SDFVD face swaps, per video | **0.989** [0.963, 1.0] | 1.9% (1 of 53) | 77.4% | 100% |
| SDFVD face swaps, per frame | **0.992** [0.98, 1.0] | 1.1% | 76.5% | 100% |
| ASVspoof 2019 LA voice spoofs, per clip | **0.914** [0.898, 0.929] | 0.0% | 19.2% | 65.5% |
| LibriSpeech, real speech from another session | – | 0.0% | – | – |

## Data

All three datasets are under commercially usable terms:

| Data | License | Used for |
|---|---|---|
| [SDFVD](https://data.mendeley.com/datasets/bcmkfgct2s/1) | CC BY 4.0 | 53 people. Each real clip `vN` (2–4 s) has a face swap `vsN` made from it |
| ASVspoof 2019 LA eval, one shard of the [`SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA`](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA) mirror (`test-00000-of-00009.parquet`) | ODC-By 1.0 | Speakers with at least 8 bona fide clips; spoofs targeting them from 13 attacks (A07–A19) |
| [LibriSpeech](https://www.openslr.org/12) dev-clean | CC BY 4.0 | Real only: one speaker's speech across two recording sessions (book chapters) |
| [FairFace](https://github.com/joojs/fairface) validation (3,000 faces) | CC BY 4.0 | Base model for the fine-tuned face autoencoder only |

## Protocol

**People are held out.** Each person's detector sees only their own enrollment media; nothing is
trained across the test people.

- **SDFVD:**
    - **Enrollment:** the frames of the first half of the person's real clip.
    - **Held-out real:** the frames of the last 40% of the clip, after a gap.
    - **Fakes:** every frame of the face swap made from that clip.
    - **Per video:** the median over frames.
- **ASVspoof:**
    - **Enrollment:** the first half of a speaker's bona fide clips (at most 10).
    - **Test:** the rest of their bona fide clips, and up to 40 spoofs targeting them (seed 0). That
      gives 42 speakers with spoofs in the shard: 292 real and 1,680 spoofed clips.
- **LibriSpeech:**
    - **Speakers:** every other speaker with two chapters.
    - **Enrollment:** 10 utterances of their larger chapter.
    - **Test:** up to 10 more utterances from that chapter (same session) and up to 20 from the other
      chapter (another session). That's 18 speakers: 176 same-session and 332 other-session
      utterances.
- **Confidence intervals:** 95%, bootstrapping people.

**Pre-registration.** The comparison was fixed before any results were seen: the methods, the
calibration and the metrics. Two settings changed before any evaluation finished, both to cut run
time on a busy machine:
- **Training steps:** the autoencoders' step counts were lowered.
- **LibriSpeech scope:** the shipped benchmark's session test was capped at the counts above.

## Which method? The comparison

Every method was fitted on the person's real enrollment media only, then put through the same
calibration: z-scores against out-of-fold scores of the person's enrollment media. The methods
were:

- **`person-detector`:** the previous ForensiX model. It's an MLP of about 23k parameters, trained
  against synthetic negatives (random vectors and noise in feature space).
- **Hand-crafted kNN:** k-nearest neighbours on the hand-crafted features `person-detector` uses:
  - faces: SFace plus 64 frequency and texture statistics;
  - voice: 128 spectral statistics.
- **Identity:** cosine distance to the person's mean SFace or ECAPA embedding.
- **Autoencoder from scratch:** a tiny convolutional autoencoder trained per person:
  - faces: 64×64 crops, 234k parameters;
  - voice: 64×32 log-mel patches, 105k parameters, scored by the worst 20% of patches.
- **Autoencoder, frozen base:** the same network trained once on other people's real media
  (FairFace faces; LibriSpeech speakers not used in testing), then used unchanged.
- **Autoencoder, fine-tuned:** the base, fine-tuned per person.
- **Identity + autoencoder:** the sum of the two z-scores.

**Faces, SDFVD per video** (AUC [95% CI]; fakes caught when 5% of real videos are flagged):

| Method | AUC | Caught at 5% false alarms |
|---|---|---|
| Identity (SFace) | **0.977** [0.956, 0.992] | 85% |
| Hand-crafted kNN | 0.971 [0.947, 0.991] | 87% |
| Identity + fine-tuned autoencoder | 0.948 [0.912, 0.979] | 79% |
| `person-detector` | 0.923 [0.879, 0.961] | 68% |
| Autoencoder, fine-tuned | 0.559 [0.491, 0.626] | 9% |
| Autoencoder, from scratch | 0.543 [0.474, 0.607] | 8% |
| Autoencoder, frozen base | 0.444 [0.359, 0.519] | 6% |

Per frame, the order is the same: identity 0.976, kNN 0.970, `person-detector` 0.909, and the
autoencoders 0.43–0.57.

**Voice, ASVspoof per clip:**

| Method | AUC | Caught at 5% false alarms |
|---|---|---|
| Identity (ECAPA) | **0.905** [0.889, 0.920] | 64% |
| Identity + fine-tuned autoencoder | 0.830 [0.798, 0.861] | 48% |
| Hand-crafted kNN | 0.810 [0.783, 0.839] | 40% |
| `person-detector`, its own probability | 0.660 [0.631, 0.694] | 26% (none at its 0.5 threshold) |
| Autoencoder, from scratch | 0.517 [0.478, 0.557] | 8% |
| Autoencoder, fine-tuned | 0.445 [0.409, 0.487] | 3% |
| Autoencoder, frozen base | 0.404 [0.361, 0.451] | 3% |

**Why the autoencoders fail.**
- **Faces:** a face swap's pixels are realistic face pixels. A small autoencoder reconstructs a
  plausible swapped face about as well as the real one. The identity encoder was trained on
  millions of faces to tell people apart, and the swap puts a *different person's* face on the
  clip.
- **Voice:** at the level of 64-band spectrogram patches, synthetic speech is close to real speech.
  The speaker encoder sees whose voice it is, which is what the spoofs get only partly right.

**Why thresholds come from the population.** A threshold calibrated on the person's own media was
far too tight:
- **SFace:** the 2.33-z cutoff flagged 33% of held-out real frames.
- **ECAPA:** at a threshold set to flag 5% of ASVspoof real clips, it flagged 31% of LibriSpeech
  speech from another session.

A few seconds of one person's media don't show how much they really vary. The encoders' published
thresholds, set on many people, held up (next section).

## The shipped method: `person-anomaly`

`forensix.identity.anomaly.PersonAnomaly` works as follows:
- **The profile:** the person's mean embedding.
- **The score:** a clip's similarity is the median, over its frames or speech segments, of the
  cosine to that mean.
- **The decision:** flagged below SFace 0.363 (OpenCV Zoo, on LFW) or ECAPA 0.25 (SpeechBrain).
- **Faces:** detected on the frame scaled to fit 640×640, which also finds large faces that
  full-resolution detection misses. This is why the per-frame AUC here is a little higher than the
  identity row above.

**Voice threshold trade-off** (ASVspoof real and spoofed clips; LibriSpeech real speech):

| Similarity threshold | ASVspoof real flagged | Spoofs caught | LibriSpeech flagged, same session | LibriSpeech flagged, other session |
|---|---|---|---|---|
| **0.25 (default)** | 0.0% | 19.2% | 0.0% | 0.0% |
| 0.30 | 0.0% | 24.0% | 0.0% | 0.0% |
| 0.40 | 0.0% | 34.0% | 0.0% | 0.3% |
| 0.50 | 1.4% | 45.1% | 0.0% | 2.1% |
| 0.60 | 4.8% | 64.2% | 1.1% | 15.4% |

These thresholds were picked after seeing the data, so read them as a trade-off curve, not a tuned
setting. The default stays at SpeechBrain's 0.25. Choose yours on labelled media of your own.

**By attack** (ASVspoof; the share caught at the default threshold, then at 5% false alarms):

| Attack | Default threshold | 5% false alarms |
|---|---|---|
| A09, A17, A18 | 54–79% | 100% |
| A19 | 23% | 98% |
| A08, A13 | 1–2% | 87–90% |
| A07, A12, A14 | 0–1% | 51–60% |
| A10, A11, A15 | 0% | 25–31% |
| A16 | 0% | 12% |

The attacks that get through are the ones that reproduce the target speaker's voice well, which is
what this check looks at. Pair it with an audio deepfake detector, which looks at how the speech
was made.

## Browser studio

The [browser studio](docs/guides/studio.md) runs the same encoders with onnxruntime-web and saves
the same model file. On test files, browser and Python embeddings matched:
- **Voice:** cosine 1.000.
- **Faces:** 0.97–0.99, from the browser's image resizing.

Every verdict was the same. `scripts/check_studio.py` repeats the comparison on any files.

## Limits

- **Small, and same-session for faces.** SDFVD has 53 people, and its enrollment and test frames
  come from the same 2–4 s clip. Real footage from another day, camera or lighting will be less
  similar to the profile, so expect more false alarms than shown here.
- **Old voice attacks.** ASVspoof 2019's attacks date from 2019. Current voice clones reproduce
  speaker identity better, so expect fewer of them to be caught at the default threshold.
- **One shard.** ASVspoof is one of nine shards of the eval set, about 11% of it.
- **Good fakes are built to pass.** A face swap or clone that fully reaches the target's identity is
  designed to pass an identity check. This detector catches the ones that don't, and impostors.
- **Encoder training data.** SFace was trained on face datasets with research-only terms
  (see `id-face`). ECAPA was trained on VoxCeleb.

## Reproduce

```bash
# shipped method (forensix.identity.anomaly)
python scripts/benchmark_person_anomaly.py faces --sdfvd SDFVD/ --out results/person_anomaly/sdfvd.json
python scripts/benchmark_person_anomaly.py voice --asvspoof-parquet test-00000-of-00009.parquet \
    --asvspoof-protocol ASVspoof2019.LA.cm.eval.trl.txt --librispeech LibriSpeech/dev-clean \
    --out results/person_anomaly/asvspoof.json
# method comparison (research code)
cd scripts/experiments/person_anomaly_methods   # see its README.md
```

The results are in `results/person_anomaly/`:
- **shipped method:** `sdfvd.json` and `asvspoof.json`;
- **method comparison:** `methods_sdfvd.json` and `methods_asvspoof_librispeech.json`.
