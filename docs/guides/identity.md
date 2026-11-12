# Identity and person models

Most detectors judge media on its own. Identity-based detectors also know who the media claims
to show, and compare it with reference media of that person. This fits person-of-interest
protection: executives, public figures, or customers who enrolled for verification. It needs
nothing about how a fake was made.

ForensiX has these tools for it:

| Tool | Question | Enroll from | Needs training? |
|---|---|---|---|
| `person-anomaly` | Is this media consistent with this one person's real face or voice? | Their real photos, videos or recordings | No: enrolling takes seconds, in Python or [in the browser](studio.md) |
| `id-face` | Does the face look like the person? | Photos and/or videos | No (SFace weights) |
| `id-voice` | Is the voice the person's? | Recordings, or videos' sound | No (ECAPA weights) |
| `id-motion` | Does the person move and emote like the person? | Videos only | Yes, once, on many people (Colab) |
| `person-detector` | Is this real media of this one person? | Their real media (any modality) | Yes, per person, seconds on CPU. Superseded by `person-anomaly`, see below |

`person-anomaly` is the one to start with when all you have is real media of the person: it's what
measured best when trained on real media only and tested on fakes
([benchmark](../../person_anomaly_benchmark.md)), and the [browser studio](studio.md) builds one
without installing anything.

## What each one catches

| Attack | `person-anomaly`, `id-face`, `id-voice` | `id-motion` | `person-detector` |
|---|---|---|---|
| A different person claiming to be X | Yes | Yes | Yes |
| Poor face swap or voice clone of X | Often | Yes (face) | Often |
| Good face swap of X, with an actor's motion | **No**: built to look like X | Yes: the motion is the actor's | Depends on the training data |
| Good voice clone of X | **No**: built to sound like X | n/a | Rarely |
| Reenactment of X driven by someone else | No | Yes | Depends |
| Real X in unusual conditions (lighting, age, microphone) | May false-alarm | May false-alarm | May false-alarm |

A good face swap is designed to pass face recognition. Use `id-face` together with a
face-manipulation detector (`gend-pe`, `dfdet-clip`), and with `id-motion` where you have
videos of the person.

## `id-face`

```python
import forensix

det = forensix.load("id-face")                        # CPU is fine
det.enroll("alice", ["alice_passport.jpg", "alice_interview.mp4"])
det.save_gallery("people.npz")

pred = det.predict("new_video.mp4", identity="alice")
pred.is_fake                                  # True when similarity < 0.363 (fake_prob > 0.637)
pred.details["similarity"]                    # median over frames of the best-matching face
pred.details["frame_similarity"]              # per sampled frame (2 per second, up to 120 frames)
pred.details["frames_without_face"]
pred.details["other_enrolled_similarity"]     # {"bob": 0.72} means it looks more like bob

det.identify("unknown.jpg")                   # [("alice", 0.61), ("bob", 0.12)]
```

How it works:
- **Detection and embedding:** YuNet finds faces and 5 landmarks, OpenCV aligns each face to
  112×112, and SFace embeds it. The 0.363 cosine threshold is OpenCV Zoo's, which gives 99.40% on
  LFW.
- **Enrollment:** each frame contributes its largest face. In videos, frames whose face
  disagrees with the rest (another person, a bad detection) are dropped.
- **Verification:** in each frame, the face that looks most like the person is kept (the mean of
  its top-3 similarities to the references). The clip score is the median over frames, which
  resists a few bad frames, while the per-frame timeline shows a face switching mid-video.

Enrollment tips:
- Use 5–20 references covering different angles, lighting, ages and expressions. One passport
  photo works but is brittle.
- Enroll from video when you can: it gives many frames, filtered for consistency.
- Re-enroll as people age or change appearance.
- Galleries hold embeddings, not images, but they are still biometric data. Store and delete them
  under the rules that apply to you (GDPR, BIPA and similar).

## `id-motion`

```python
det = forensix.load("id-motion", weights_path="id_motion.safetensors")   # train it first
det.enroll("alice", ["alice_interview1.mp4", "alice_interview2.mp4"])
pred = det.predict("new.mp4", identity="alice")
pred.details["window_similarity"], pred.details["window_start_s"]       # per 4 s window
```

- **Features:** head rotation, translation and their velocities, plus MediaPipe's 52
  expression blendshapes, at 25 fps. These describe behaviour, not the face's shape.
- **Model:** a temporal network embeds every 4 s window. It is trained contrastively on real
  videos of many people, with each person's positives taken from *different* videos, so it
  learns the person rather than the recording.
- **Data needs:** a few seconds of natural, talking behaviour in both the references and the
  query. Several reference videos in different settings work best. Short clips, or a person who
  barely moves, give weak evidence.
- **Training:** on VoxCeleb2 dev on Colab ([Training](training.md#id-motion)). The decision
  threshold is the equal-error point measured on held-out people.

## `person-anomaly`

A per-person detector built from the person's real media only. The person is described by a frozen,
pretrained identity encoder; enrolling stores their mean embedding and how much their own media
varies. Nothing is trained, so it takes seconds on a CPU and needs no fakes.

```python
from forensix.identity.anomaly import fit_person

alice = fit_person("alice", ["alice_1.mp4", "alice_2.jpg", "alice_3.mp4"])   # kind="auto": face for images/videos
alice.save("alice.face.json")
voice = fit_person("alice", ["call1.wav", "podcast.mp3"], kind="voice")

det = forensix.load("person-anomaly", weights_path="alice.face.json")
pred = det.predict("new.mp4")
pred.is_fake                          # similarity < 0.363 (faces) or < 0.25 (voices)
pred.details["similarity"]            # median over frames (2 per second, up to 60) or 3 s speech segments
pred.details["item_similarity"]       # per frame or segment: where it stops looking or sounding like them
pred.details["person_percentile"]     # share of their own held-out media that was less similar
```

How it works:
- **Faces:** YuNet finds the face on the frame scaled to fit 640×640 (this also finds large faces
  that detection at full resolution misses), OpenCV's 5-point alignment crops it to 112×112, and
  SFace embeds it.
- **Voice:** ECAPA-TDNN embeds the whole clip up to 6 s, or 3 s segments every 1.5 s, skipping
  near-silent ones (as `id-voice`).
- **Decision:** `fake_prob = 1 − similarity` to the person's mean embedding, flagged below the
  encoder's published population threshold (SFace 0.363 from OpenCV Zoo; ECAPA 0.25 from
  SpeechBrain). The person's own media is too narrow a sample to set a threshold on: calibrated
  that way, a third of held-out real frames were flagged in the benchmark. It still says how
  typical a clip is for them (`person_percentile`), from leave-one-file-out similarities.
- **Files:** a small JSON (`forensix-person-anomaly/1`), the same format the
  [browser studio](studio.md) saves, so a person enrolled in either place works in the other.

Measured on held-out people ([benchmark](../../person_anomaly_benchmark.md)), with the default
thresholds:

| Data | AUC | Real flagged | Fakes caught |
|---|---|---|---|
| SDFVD face swaps, per video | 0.99 | 2% (1 of 53) | 77% |
| ASVspoof 2019 LA voice spoofs, per clip | 0.91 | 0% | 19% |

The voice threshold is conservative. At a similarity threshold of 0.5 instead of 0.25, 1.4% of
real clips were flagged and 45% of spoofs were caught. Choose it on your own data
([Calibration](../calibration.md)).

Real speech from a different recording session than the enrollment is less similar:
- **at 0.25:** none of it was flagged;
- **at 0.5:** 2.1% was flagged;
- **at 0.6:** 15% was flagged, against 1% from the same session.

Enroll from more than one session.

## `person-detector`

!!! note "Superseded by `person-anomaly`"
    Trained on one person's real media only and tested on fakes, `person-detector` scored AUC 0.91
    on SDFVD face swaps and 0.66 on ASVspoof voice spoofs, where its default 0.5 threshold caught
    none ([benchmark](../../person_anomaly_benchmark.md)). Its synthetic negatives are random
    vectors in feature space, which a classifier separates from real media without learning what
    real fakes look like. It stays for existing checkpoints.

A tiny model (~23k parameters) trained for one person, on CPU, in seconds. Without negatives,
it synthesizes manipulated and impostor-like samples from the real ones.

```python
from forensix.specialized import train_person_model

det = train_person_model("alice", real_media=alice_files, negative_media=other_people_files,
                         modality="video", epochs=30)
det.save("alice.safetensors")
forensix.load("person-detector", weights_path="alice.safetensors").predict("new.mp4")
```

- `modality` is `image`, `audio`, `video`, `audiovisual` or `auto`; one model covers one
  modality.
- `use_peft=True` (default) adapts a base model with LoRA. `pretrain_base_model` builds that base
  from general real/fake data; without it, a fresh base is used.
- Real negatives (other people, known fakes of the person) beat synthetic ones. Add them when you
  have them.
- Hold out some of the person's real media and measure the false-alarm rate on it before
  relying on the model. A few training samples give an overconfident model.

## Evaluating identity detectors

`scripts/eval_identity.py` enrolls each person from `references/<person>/`, then scores their
real queries (should pass), their fakes (should be flagged) and other people's real queries
claimed as them (impostors):

```text
DATA/references/<person>/*        reference photos or videos
DATA/queries/<person>/real/*      genuine media of the person
DATA/queries/<person>/fake/*      deepfakes of the person (optional)
```

```bash
python scripts/eval_identity.py --data /content/id_eval --models id-face id-motion \
    --weights id-motion=id_motion.safetensors --out results/identity.json
```

There's no commercially usable deepfake benchmark with per-person references. VoxCeleb2 test
(CC BY 4.0) alone already measures the impostor rate and the false-alarm rate on real media.
