# Interpreting scores

A detector's output is evidence, not a verdict. This guide explains what each part of a
`Prediction` means, and how to turn scores into decisions you can defend.

## The three kinds of score

| `score_type` | `threshold` | Models | Meaning |
|---|---|---|---|
| `"probability"` | a number (usually 0.5) | audio, image, video, `stablesyncnet`, `avcons`, `avid`, `person-detector` | The model's softmax or sigmoid output. In [0, 1], but **not calibrated**: 0.8 doesn't mean "80% chance of fake". |
| `"anomaly"` | `None` | `avad`, `syncnet`, `phys-*`, `physdf` | Distance from how real media behaves. There's no natural cut-off: choose one on your own real data. `syncnet`'s score is `-confidence`, so it's usually negative. |
| `"anomaly"` | a number | `id-face` (0.637), `id-motion` | 1 − similarity to the enrolled person. The threshold comes from face verification (LFW) or training, not from deepfake data. |

In every case, **higher means more likely fake**, so ranking metrics (AUC, EER) work the same
way for all of them.

## Why the default threshold often fails

A model's default threshold of 0.5 fits the data it was trained on. On other data, the score
distributions shift. On the SDFVD face-swap benchmark:

| Model | AUC | Real correct at 0.5 | Fake correct at 0.5 | What happened |
|---|---|---|---|---|
| `rine` | 0.745 | 100% | 0% | Every score below 0.5; highest fake 0.384 |
| `dfdc-selim` | 0.761 | 94% | 19% | Scores too low on this face-swap tool |
| `gend` | 0.919 | 68% | 94% | Scores too high on real faces |
| `dfdet-clip` | 0.916 | 89% | 76% | Reasonably balanced |

A model can rank well (good AUC) and still make bad yes/no decisions at its default threshold.
Treat the default as a placeholder.

## Choosing a threshold

1. **Collect real media like the media you'll check:** same sources, platforms, compression,
   resolution, languages. Real data is much easier to get than fakes, and it's what sets the
   false-alarm rate.
2. **Decide the false-alarm rate you can afford:** for example 1% of real uploads sent to
   human review.
3. **Set the threshold at that quantile** of the real scores
   ([recipe 5](../recipes.md#5-pick-a-threshold-for-a-target-false-alarm-rate)).
4. **Measure the detection rate on fakes,** if you have any, at that threshold. Measure it on
   media not used to set the threshold.
5. **Re-check periodically.** New cameras, codecs and platforms shift real-media scores too.

Sample size matters. To confirm a 1% false-alarm rate to within about ±0.5 points, you need
roughly 1,500 real samples; to within ±0.2 points, about 10,000.

## Segments and localization

`segment_fake_probs` breaks a score down:

| Models | One segment per | Use |
|---|---|---|
| audio (most) | 4.04 s window ([recipe 3](../recipes.md#3-find-which-part-of-a-long-recording-is-fake)) | Find the fake part of a long recording |
| `nes2net-x-itw` | 30 s chunk | Coarse localization |
| video models | scored face crop | Spot one bad face among good ones; empty means no face found |
| `avcons` | AV window | When audio and face stop matching |
| `id-face` | sampled frame (2 per second) | When the face stops being the person |
| `id-motion` | 4 s window | Same, for behaviour |
| `person-detector` (video) | frame | Same, for the per-person model |

Physics detectors put their timelines in `details["timeline"]`, and `physdf` puts segments in
`details["localization"]["segments"]`, each with a start and end in seconds, the constraint and
the face or voice node involved.

Clip-level scores average over segments, so a short manipulation in a long clip can be diluted
below the threshold. If short edits matter, also flag on `max(segment_fake_probs)`, with its own
threshold chosen the same way.

## `details`

`details` holds structured evidence, and its keys depend on the model ([API reference](../api-reference.md#prediction)).
The most useful ones:

- **`syncnet`:** `offset_frames` (best audio shift; real clips are usually within ±3 frames,
  and an offset at ±15 means no good alignment was found), `confidence` (about 10 on a clean real
  clip, under 1 when audio and lips don't match), `mean_dist_by_offset` (the full curve).
- **`id-face`:** `similarity`, `frame_similarity`, `frame_time_s`, `frames_without_face`, and
  `other_enrolled_similarity`, which tells you when the face looks more like someone else in
  your gallery.
- **Physics detectors:** `constraints[name]` with `measured`, `expected_range_start`, `worse`
  (whether higher or lower is suspicious), `violation` (0–1) and `prior` (`provisional` or
  `learned`).
- **`physdf`:** `report` (verdict, findings, counter-evidence, and Markdown), `localization`,
  `status` (which checks ran).

## Reporting results to people

- Report the score **with** the model name and threshold, never a bare "87% fake".
- Say what the model checks: "the face region shows manipulation artefacts" (`gend-pe`) is a
  different claim from "the voice doesn't match the lip movements" (`syncnet`).
- A low score is not proof of authenticity. It means this model found nothing. Models miss
  manipulations they weren't trained for.
- Keep a human in the loop for consequential decisions. The `stablesyncnet` weights license
  forbids fully automated decisions that affect someone's legal rights ([Licensing](licensing.md)).
