# Physics detectors: SDFVD and ASVspoof 2019 LA

This report evaluates the physics-based detectors (`phys-*`). These are hand-built checks on physical constraints, with no trained classifier. They ran on SDFVD (video) and a subset of ASVspoof 2019 LA (audio). It also tests whether priors fitted on real data improve them. For the learned detectors, see [benchmark.md](benchmark.md).

## Summary

- **SDFVD: no physics detector beats chance.**
  - `phys-motion` AUC 0.45 and `phys-lighting` 0.48.
  - `phys-bloodflow` never applies: every clip is under its 6 s minimum.
  - With the default priors, `phys-lighting` flags 100% of the real clips, so it isn't usable on this footage.
  - Face swaps keep the source video's head motion and scene lighting. The constraints these detectors check are mostly preserved by construction.
- **ASVspoof 2019 LA: near chance with the default priors.**
  - `phys-voice` AUC 0.49 and `phys-room` 0.49.
  - With priors fitted on ASVspoof's own bona fide train/dev speech, `phys-voice` rises to AUC 0.62. It flags 26% of fakes, with 1.4% of reals violating any constraint.
  - Coverage is low: `phys-voice` measures 33% of trials (it needs 1.5 s of voiced speech) and `phys-room` measures 4% (it needs 6 s of audio).
- **The fitted priors are not shipped as the package default.**
  - They fail the package's own tests: a real 10 s LibriVox recording scores 1.0.
  - The reason: some ranges learned from ~3 s ASVspoof clips don't transfer to longer speech (details below).
  - The default stays the provisional, hand-set priors. This is a decision for the maintainer; options are listed at the end.
- **Camera-consistency feature.** One experimental feature, fixed in advance on a dev split of SDFVD, reached AUC 0.66 [0.50, 0.81] on the held-out test pairs. It is not added to the package (see below).

## Data

| Set | Content | Used for |
|---|---|---|
| SDFVD | 53 real Pexels clips + 53 Remaker AI face swaps of them, 2.1–5.8 s, 30 fps (CC BY 4.0) | video detectors; all 106 clips |
| ASVspoof 2019 LA train + dev, bona fide | 5,128 clips, 40 speakers | fitting priors (held-out speakers for false alarms) |
| ASVspoof 2019 LA eval subset | 1,000 bona fide + 77 per attack (13 attacks), seed 0 = 2,001 trials; the same subset as `scripts/eval_asvspoof2019_la.py` and the audio rows of the README | audio evaluation |

The ASVspoof eval clips were never used for fitting. The eval speakers are disjoint from the train and dev speakers.

## Video: SDFVD (default priors)

| Detector | Applies to | AUC | AP | Reals with any violation | Reals with score ≥ 0.5 | Fakes with any violation |
|---|---|---|---|---|---|---|
| `phys-motion` | 47 + 47 | 0.446 | 0.479 | 42.6% | 10.6% | 31.9% |
| `phys-lighting` | 47 + 47 | 0.481 | 0.489 | 100% | 70.2% | 97.9% |
| `phys-bloodflow` | 0 | – | – | – | – | – |
| max over detectors | 47 + 47 | 0.487 | 0.495 | | | |

- Six of the 53 pairs have a face tracked in under 80% of frames, so they are skipped.
- `phys-bloodflow` skips every clip (it needs at least 6 s).

Per constraint (raw measurement as the score, oriented so that higher means more suspicious; independent of the priors):

| Constraint | n | AUC |
|---|---|---|
| `phys-motion/rotation_hf_energy_ratio` | 47 + 47 | 0.620 |
| `phys-motion/p99_angular_acceleration_deg_s2` | 47 + 47 | 0.556 |
| `phys-motion/p99_angular_velocity_deg_s` | 47 + 47 | 0.514 |
| `phys-motion/rigid_deformation` | 47 + 47 | 0.422 |
| `phys-motion/jaw_lip_coupling` | 4 + 4 | 0.438 |
| `phys-lighting/face_scene_coupling` | 17 + 17 | 0.561 |
| `phys-lighting/light_direction_jitter_deg` | 47 + 47 | 0.510 |
| `phys-lighting/shading_residual` | 47 + 47 | 0.508 |

With 47 pairs, a single AUC has a 95% interval about ±0.11 wide. So only `rotation_hf_energy_ratio` is even borderline, and it was not tested on held-out data.

**`phys-lighting` flags every real clip.** SDFVD's real Pexels clips fall outside its provisional ranges; the cause wasn't investigated. Fitting priors needs real video with labelled speakers or groups, and SDFVD's 53 reals are also the evaluation set, so no video priors were fitted here.

## Audio: ASVspoof 2019 LA subset

| Detector | Priors | Applies to (real + fake) | AUC | AP | Reals with any violation | Reals with score ≥ 0.5 | Fakes with any violation |
|---|---|---|---|---|---|---|---|
| `phys-voice` | default (provisional) | 294 + 369 | 0.488 | 0.553 | 5.4% | 0.3% | 3.0% |
| `phys-voice` | fitted on ASVspoof train/dev | 294 + 369 | **0.624** | 0.664 | 1.4% | 0.3% | 26.0% |
| `phys-room` | either | 26 + 50 | 0.492 | 0.699 | 15.4% | 0% | 12.0% |

For comparison, the learned audio detectors reach 0.0–3.1% EER on the same subset (see the README).

Per-constraint AUCs on the trials where `phys-voice` applies:

| Constraint | AUC |
|---|---|
| `longest_pause_free_s` | 0.789 |
| `hnr_db` | 0.695 |
| `jitter_percent` | 0.610 |
| `vtl_out_of_range_cm` | 0.500 |
| `f0_jump_rate_per_s` | 0.460 |
| `vtl_variation` (276 + 205) | 0.446 |

**`longest_pause_free_s` is not a duration artifact.** On these trials fakes are slightly *shorter* than reals (duration AUC 0.42). Within duration bins the constraint still separates well:

| Duration | AUC |
|---|---|
| 3–4 s | 0.88 |
| 4–6 s | 0.86 |
| 6–20 s | 0.92 |

The 2019 LA attacks rarely insert pauses. Normalizing by clip length (`longest_pause_free_s / duration`) gives AUC 0.88. That number was found by looking at this eval set, so treat it as a hypothesis for a future held-out test, not a result. It may not transfer to other generators.

**Fitting the priors.**
- Command: `scripts/calibrate_physics.py fit`, α = 0.01, at least 100 real clips per constraint.
- Held-out speakers (`phys-voice`, n = 236 clips): reals with any violation fell from 5.1% to 4.2%, and reals with score ≥ 0.5 from 1.7% to 0.8%.
- The fitted values (`results/physics/learned_priors_asvspoof.json`):

| Constraint | Provisional (start, full) | Fitted start | Real median |
|---|---|---|---|
| `jitter_percent` | 0.25, 0.1 | 0.92 | 2.06 |
| `hnr_db` | 35, 45 | 17.0 | 9.7 |
| `longest_pause_free_s` | 12, 25 | 4.22 | 2.4 |
| `f0_jump_rate_per_s` | 2, 6 | 2.70 | 0 |
| `vtl_variation` | 0.12, 0.3 | 0.20 | 0.04 |
| `digital_silence_fraction` (`phys-room`, n = 106) | 0.2, 0.8 | 0.0 | 0 |

Constraints with fewer than 100 real measurements kept their provisional values. That covers all video constraints and three of the four `phys-room` constraints.

## Why the fitted priors are not the package default

The fitted table was installed as `forensix/physics/learned_priors.json` and the physics test suite was run. Three tests failed, and the file was removed:

1. **`longest_pause_free_s` starts at 4.22 s.**
   - That is the 99.8th percentile of real ASVspoof clips, whose median length is 3.3 s (95th percentile 5.3 s). Clips that short can't contain a long pause-free stretch, so the fitted range reflects clip length rather than speech.
   - The test suite's real 10 s LibriVox recording has an 8.24 s pause-free stretch. It scores 1.0 (certain fake).
2. **`digital_silence_fraction` starts at 0.**
   - None of the 106 real ASVspoof clips long enough to measure had any digital silence, so any digital silence at all becomes a violation.
   - This was fitted on only 106 clips.

The gate did its job: priors fitted on one corpus encode that corpus's recording conditions. The package therefore keeps the provisional priors. The fitted table stays in `results/` for anyone scoring ASVspoof-like audio:

```python
det = forensix.load("phys-voice")
det.load_priors("results/physics/learned_priors_asvspoof.json")
```

## Experiment: camera consistency on SDFVD

**Idea.** A face swap pastes a generated face into a real frame. The face region and its surroundings should then carry different sensor noise.

**Protocol.**
- Fix a dev/test split of the 53 pairs before computing anything: seed 0, 27 dev pairs and 26 test pairs ([split.json](results/physics/camera_experiment/split.json)).
- Explore features on dev only.
- Choose one feature and its direction.
- Evaluate it once on test.

**Features explored on dev.** All compare the inner face skin with a ring around the face, using the existing face tracker and skin mask. AUCs below use "higher = fake".

| Feature | Dev AUC (26 pairs) |
|---|---|
| median noise ratio | 0.50 |
| median sharpness ratio | 0.54 |
| luminance flicker ratio | 0.49 |
| temporal noise ratio | 0.47 |
| **`noise_ratio_spread`** (std over frames of the log noise ratio) | **0.34**, i.e. 0.66 with lower = fake |
| temporal noise ratio spread | 0.33 |

**Chosen before test.** `noise_ratio_spread`, with lower = fake. In a swap, the face's noise tracks the surroundings less from frame to frame.

**Test result** (24 pairs; 2 pairs have too few tracked frames):

- AUC **0.660**, 95% CI [0.498, 0.812] from a stratified bootstrap, or [0.556, 0.771] when resampling pairs.
- The fake has lower spread than its own real source in 71% of pairs. Wilcoxon signed-rank p = 0.0065 (two-sided).

**Why it isn't in the package.**
- The effect is real on this held-out data, but small.
- The physical story is weak. A swapped and re-encoded face could plausibly show *more* variation as easily as less.
- Re-encoding by the swap service is a confound that could produce the same effect without any compositing.
- One dataset and one generator are not enough evidence for a shipped detector.

Code and per-clip features are in [results/physics/camera_experiment/](results/physics/camera_experiment/).

## Fixes made during this evaluation

- **`forensix/physics/room.py`.**
  - Bug: when an analysis chunk reached the file's last samples, the speech mask was shorter than the chunk, and `phys-room` raised `IndexError`. It hit 7 of the ASVspoof clips.
  - Fix: pad the mask (one line).
  - Test: added a regression test, `test_room_detector_chunk_at_end_of_file`.
- **Not fixed:** `phys-voice` raised "attempt to get argmax of an empty sequence" on 1 of the 5,128 fitting clips. That clip is excluded from the fit; the eval set was unaffected.

## Caveats

- SDFVD is small (47 usable pairs) and has a single generator. ASVspoof 2019 LA's attacks are older systems. Neither represents current deepfakes.
- Every physics result here is close to chance, apart from `phys-voice` with in-domain priors. Use the physics detectors as interpretable side evidence, not as a primary detector.
- No priors were fitted for video, because no separate labelled real video was available.

## Reproduce

These commands use the files in this repo. The data downloads are described in [benchmark.md](benchmark.md); ASVspoof 2019 LA is `LA.zip` from the Edinburgh DataShare (sha256 `208a7e4e…5e4ec`).

```bash
export FORENSIX_CACHE=/content/drive/MyDrive/forensix_cache
# SDFVD
for lab in real fake; do
  python scripts/calibrate_physics.py measure --media /content/sdfvd/SDFVD/videos_$lab --kind video \
      --label $lab --source sdfvd --out results/physics/sdfvd_$lab.jsonl
done
python scripts/calibrate_physics.py eval --measurements results/physics/sdfvd_{real,fake}.jsonl \
    --out results/physics/sdfvd_eval_provisional.json
# ASVspoof 2019 LA: symlink layout (fit = train+dev bona fide by speaker, eval = the README subset)
python results/physics/asvspoof/make_layout.py /content/asvspoof/LA /content/asvspoof/layout . --shards 6
for i in 0 1 2 3 4 5; do   # one process per shard; can run in parallel
  python scripts/calibrate_physics.py measure --media /content/asvspoof/layout/fit_$i --kind audio --label real \
      --source asvspoof2019-la-train-dev --out results/physics/asvspoof/fit_$i.jsonl
  python scripts/calibrate_physics.py measure --media /content/asvspoof/layout/eval_$i --kind audio \
      --labels-csv /content/asvspoof/layout/eval_labels.csv --source asvspoof2019-la-eval \
      --out results/physics/asvspoof/eval_$i.jsonl
done
python scripts/calibrate_physics.py fit --measurements results/physics/asvspoof/fit_*.jsonl --alpha 0.01 \
    --out results/physics/learned_priors_asvspoof.json
python scripts/calibrate_physics.py eval --measurements results/physics/asvspoof/eval_*.jsonl \
    --out results/physics/asvspoof_eval_provisional.json
python scripts/calibrate_physics.py eval --measurements results/physics/asvspoof/eval_*.jsonl \
    --priors results/physics/learned_priors_asvspoof.json --out results/physics/asvspoof_eval_learned.json
# camera experiment
python results/physics/camera_experiment/camera_features.py . /content/sdfvd/SDFVD \
    results/physics/camera_experiment/split.json out_test.json test
```
