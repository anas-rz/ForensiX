# Physics forensics

Learned detectors recognize the fingerprints of generators they were trained on. Physics
detectors work the other way round: they measure physically meaningful quantities (pulse, head
motion, lighting, voice production, room acoustics), compare them with how real faces and
voices behave, and say which physical constraint was broken. The output is evidence a person can
check, not just a number.

The [README physics section](../../README.md#physics-based-detection-physdf) describes every
measurement and its verification. This guide covers using them.

## The detectors

| Detector | Input | Constraint | Needs |
|---|---|---|---|
| `phys-bloodflow` | video | One heart drives the pulse in all facial skin | ≥ 6 s of a steady, visible face |
| `phys-motion` | video | The head is a rigid body with inertia; the jaw is a hinge | ≥ 2 s of a tracked face at ≥ 20 fps |
| `phys-lighting` | video | One light field lights the face, and it follows the scene | ≥ 2 s of a tracked face |
| `phys-voice` | audio | Voice production stays within anatomical limits | ≥ 1.5 s of voiced speech |
| `phys-room` | audio | One room and one microphone | ≥ 6 s of speech with pauses |
| `physdf` | video, audio, or both | All of the above, plus a learned graph prior across face, pulse, light and voice | Whatever the parts need |

All of them return `score_type="anomaly"` with no default threshold. The score is the violation
(0–1) of the **most-violated** constraint, because one clearly broken physical law is evidence
on its own. Constraints that don't apply to a clip (for example jaw coupling when the mouth never
moves) are listed in `details` but left out of the score.

## Reading the evidence

```python
pred = forensix.load("phys-room").predict("voice_note.wav")
for name, c in pred.details["constraints"].items():
    print(name, c["measured"], c.get("expected_range_start"), c["worse"], c["violation"], c["prior"])
```

Each constraint reports:
- `measured`: the value in this clip;
- `expected_range_start` and `worse`: where real media stops, and which direction is suspicious;
- `violation`: 0 inside the real range, rising to 1 at the end of the range;
- `p_real` (reference scoring): the share of real media at least this extreme;
- `prior`: `reference` (a real-media distribution), `learned` (a range fitted on real media) or
  `provisional` (hand-set from the literature).

`details["scoring"]` says how the score was computed (next section).

`details["timeline"]` gives per-frame or per-stretch values for localizing the problem in time.

## The physdf report

```python
det = forensix.load("physdf", weights_path="physdf_graph.safetensors")   # weights_path optional
pred = det.predict("interview.mp4")       # its sound track is used if present
report = pred.details["report"]
print(report["markdown"])
```

The report gives a verdict and confidence, then:
- **Findings:** one claim per violated constraint, naming the physical law, the measured value,
  where the real range ends, whether that range is calibrated, and where and when.
- **Counter-evidence:** the checks that passed.
- **What couldn't be checked,** and why.

`forensix.physics.report.verify_report(report, pred.details)` re-derives every claim, the
score and the verdict from the evidence. It returns an empty list when everything checks out.
The report abstains ("inconclusive") when fewer than 5 checks apply or the strongest violation
is weak, and says "low confidence" when only provisional ranges are violated.

Without a trained graph prior, `physdf` runs the five detectors and reports on them. With the
prior, it adds cross-system residuals: the mouth predicted from the speech, the speech from the
mouth, each skin region's pulse from the others, and so on ([Training](training.md#physdf)).

## How the score is computed

**Reference scoring** (`details["scoring"] == "reference"`) compares each measurement with its
distribution over *real* media:

1. Each constraint becomes a tail probability `p_real`: how much real media is at least this
   extreme, in the direction the physics says is suspicious (too clean a voice, too long without
   a breath).
2. Fisher's method adds the constraints up, so several weak signals count together.
3. The result is calibrated on real media: the **score is the share of real clips that are less
   unusual**. A score of 0.99 means "more unusual than 99% of real media". The detector's
   `threshold` is 0.99, so on media like the reference, about 1% of real clips are flagged.

No fakes are used, so a reference can't overfit one generator. It does encode the recording
conditions of the real media behind it.

**Range scoring** (`"provisional"`) is the fallback for detectors without a reference: each
constraint has a hand-set range, and the score is the largest violation. Most clips fall inside
the ranges and score exactly 0, so range scores rank clips poorly and have no threshold.

| Detector | Packaged default | Measured result |
|---|---|---|
| `phys-voice` | **Reference**: real ASVspoof 2019 LA train/dev speech plus LibriSpeech dev-clean | AUC 0.679 on held-out ASVspoof speakers (range scoring: 0.488), with 0.0% false alarms at 0.99 |
| `phys-room` | Ranges | A general reference wasn't reliable: room acoustics vary too much between recording setups |
| `phys-bloodflow`, `phys-motion`, `phys-lighting` | Ranges | No real-video corpus to fit a default yet |

The [physics benchmark](../reference/benchmark-physics.md) has the details.

## Fit a reference on your own media (recommended)

A reference fitted on real media *like yours* does better than any general default. Fitted on
ASVspoof's own real training speech, `phys-voice` reaches AUC 0.741 on that corpus, against 0.679
for the general default. It's also the only way to get calibrated video physics. You need real
media only, ideally a few hundred clips from many speakers or sources:

```bash
# 1. measure every constraint on real media like yours (resumable JSONL);
#    the first folder under --media is the group (speaker, channel), the unit of every split
python scripts/calibrate_physics.py measure --media /data/real --kind video --with-audio \
    --label real --source my-real --out m/real.jsonl
# 2. fit the reference; it reports false alarms on held-out groups at 0.99 and 0.95
python scripts/calibrate_physics.py fit-reference --measurements m/real.jsonl --out reference.json
# 3. optional, if you have labelled fakes: real vs fake AUC per detector and per constraint
python scripts/calibrate_physics.py measure --media /data/fakes --kind video --with-audio --label fake --out m/fake.jsonl
python scripts/calibrate_physics.py eval --measurements m/real.jsonl m/fake.jsonl --priors reference.json
```

Use it:

```python
det = forensix.load("phys-bloodflow")
det.load_priors("reference.json")
```

To make it the default everywhere, replace `forensix/physics/reference.json`, which ships as
package data.

A constraint needs at least 100 real measurements to get a distribution; the rest keep their
provisional range for the evidence but don't count towards the score. A detector needs 50 real
clips across at least 5 groups to be calibrated.

Older **learned priors** (`fit`, ranges fitted on real media) still work with `load_priors`, and
an explicit priors table forces range scoring.

Suggested real data with commercially usable licenses: LibriSpeech (CC BY 4.0) for audio and
VoxCeleb2 test (CC BY 4.0; video copyright stays with the owners) for video with sound.

## Limits

- **rPPG needs a steady face:** heavy compression, makeup, low light and fast motion weaken the
  pulse signal.
- **Weak cues:** noise-colour changes and spliced rooms score low even when present, because blind
  RT60 from a few decays is noisy.
- **Pitch continuity:** at this tracker quality, it can't separate fast genuine pitch moves from
  an octave-jumping synthetic voice.
- **Face swaps keep physics:** a swap keeps the source video's head motion and lighting, so
  `phys-motion` and `phys-lighting` stay near chance on face swaps even with a reference. Blood
  flow needs at least 6 s of a steady face: on 2–6 s compressed clips, the pulse isn't measurable
  even in real faces.
- **Recording conditions:** a reference describes the real media it was fitted on. Phone calls,
  noisy rooms or heavy compression need their own reference.
- **No attribution:** the priors know real media only, so the system says *that* physics was
  violated, not *which* generator did it.
