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
- `prior`: `provisional` (hand-set from the literature) or `learned` (fitted on real media).

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

## Calibrate before you rely on it

The provisional ranges are hand-set, and the video ranges haven't been checked against a large
set of real videos. In testing, provisional blood-flow and lighting ranges fired on real
footage. Until you fit learned priors on real media like yours, treat physics output as leads
for a human reviewer, not as a detector score.

Learned priors set each constraint's range from real media only, so a real clip trips any one
constraint of a detector with probability about α (default 1%):

```bash
# 1. measure every constraint on real media like yours (resumable JSONL)
python scripts/calibrate_physics.py measure --media /data/real_videos --kind video --with-audio \
    --label real --source my-real --out m/real.jsonl
# 2. fit priors on real clips; reports held-out false-alarm rates, learned vs provisional
python scripts/calibrate_physics.py fit --measurements m/real.jsonl --out learned_priors.json
# 3. optional: real vs fake AUC per detector and per constraint
python scripts/calibrate_physics.py measure --media /data/fakes --kind video --with-audio --label fake --out m/fake.jsonl
python scripts/calibrate_physics.py eval --measurements m/real.jsonl m/fake.jsonl --priors learned_priors.json
```

Use them:

```python
det = forensix.load("phys-bloodflow")
det.load_priors("learned_priors.json")
```

To make them the default for every detector, ship the file as
`forensix/physics/learned_priors.json` (the package already declares it as package data). A
constraint with fewer than 100 applicable real clips keeps its provisional range;
`details["constraints"][k]["prior"]` tells you which applied.

Suggested real data with commercially usable licenses: LibriSpeech (CC BY 4.0) for audio and
VoxCeleb2 test (CC BY 4.0; video copyright stays with the owners) for video with sound.

## Limits

- **rPPG needs a steady face:** heavy compression, makeup, low light and fast motion weaken the
  pulse signal.
- **Weak cues:** noise-colour changes and spliced rooms score low even when present, because blind
  RT60 from a few decays is noisy.
- **Pitch continuity:** at this tracker quality, it can't separate fast genuine pitch moves from
  an octave-jumping synthetic voice.
- **No attribution:** the priors know real media only, so the system says *that* physics was
  violated, not *which* generator did it.
