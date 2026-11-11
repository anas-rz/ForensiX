# Licensing and compliance

The ForensiX framework itself (library, server, web console, deployment templates, scripts) is
MIT-licensed ([LICENSE](../../LICENSE)). The models are licensed separately, as below.

ForensiX includes only code and weights whose licenses allow commercial use. That doesn't mean
every deployment is automatically fine. This page lists what to check. It's engineering
guidance, not legal advice: have counsel review a commercial deployment.

## Three layers

| Layer | What governs it | Where to look |
|---|---|---|
| Code | The license in each model's directory | `forensix/<family>/<model>/LICENSE*` |
| Weights | The weights' own license, or the repo's when the authors give none | `forensix.info(name).license`; README Models tables |
| Training data | The dataset's terms, which may or may not carry over to the weights | README footnote ‡ |

```python
for name in forensix.list_models():
    print(name, "|", forensix.info(name).license)
```

## Marks in the Models tables

| Mark | Meaning | What to do |
|---|---|---|
| `*` (e.g. `MIT*`) | The weights are linked from a permissively licensed repo but hosted elsewhere, with no separate license. Applies to `xlsr-aasist`, `xlsr-aasist-df`, `avad`, `syncnet`. | Generally treated as covered by the repo license; flag it in your license review. |
| `†` | Built on OpenAI CLIP ViT-L/14 (MIT). OpenAI's model card lists "any deployed use case" as out of scope. That's an intended-use statement, not a license term. | Note it in your risk assessment. |
| `‡` | Trained on data with research-only or non-commercial terms: FaceForensics++, DFDC, ForenSynths, LRS2, VGGFace2 and others. Whether those terms carry over to trained weights is legally unsettled. | The main open question for commercial use. Get a legal opinion; prefer models you train yourself on permissive data where it matters. |
| `§` | `stablesyncnet`: CreativeML Open RAIL++-M | Use-based restrictions apply; see below. |

## `stablesyncnet` and use-based restrictions

`stablesyncnet`'s weights are under OpenRAIL++-M. Commercial use is allowed, but the license's
use restrictions (paragraph 5 and Attachment A) apply, and you must pass them on to your users.
Two matter for forensics. The weights may not be used:
- for fully automated decision-making that adversely affects someone's legal rights;
- to generate information for administration of justice, law enforcement, immigration or asylum
  processes.

In those settings, use `syncnet` (MIT), which performs the same check. The full text is in
`forensix/audiovisual/stablesyncnet/LICENSE-WEIGHTS-OpenRAIL++-M.txt`.

## Models you train

`avcons`, `avid`, `id-motion`, `physdf` and `person-detector` are ForensiX code (no third-party
restrictions beyond their frozen components), trained by you. The weights' terms then depend on
your training data:
- **VoxCeleb2, LibriSpeech:** CC BY 4.0. Attribution required; for VoxCeleb2, video copyright
  stays with the owners.
- **Your own customers' media:** your contracts and privacy law decide.
- **Frozen components:** SyncNet (MIT*), ECAPA-TDNN (Apache-2.0), DINOv2 (Apache-2.0),
  MediaPipe (Apache-2.0), ICT-FaceKit (MIT).

## Provenance components

| Component | License |
|---|---|
| `c2pa-python` (C2PA SDK) | MIT OR Apache-2.0 |
| invisible-watermark dwtDct code (vendored in `forensix/provenance/imwatermark/`) | MIT |
| AudioSeal code and detector weights | MIT (weights MIT since 2024-04-02) |

## Evaluation datasets

Evaluating on a dataset is a use of it too. ForensiX's benchmarks use commercially usable data:

| Dataset | License |
|---|---|
| SDFVD | CC BY 4.0 |
| ASVspoof 2019 LA | ODC-By 1.0 |
| Deepfake-Eval-2024 | CC BY-SA 4.0, evaluation only, gated |
| ForensiX image eval set | Open Images CC BY 2.0 photos + generators whose outputs allow commercial use |

FaceForensics++, Celeb-DF, DFDC, FakeAVCeleb, DeepSpeak and most other deepfake datasets have
non-commercial or research-only terms.

## Biometric and privacy law

Identity features process biometric data:
- **`id-face`:** face embeddings.
- **`id-motion`:** behavioural embeddings.
- **`avid` and the audio models:** voice.

Consent, retention limits and deletion rights may apply (GDPR, BIPA, CCPA and similar).
Galleries (`.npz`) store embeddings, not images, but embeddings are still biometric data.

## Before shipping

- [ ] List the exact models in production and their `forensix.info(name).license`.
- [ ] Review the `‡` training-data question for each, with counsel.
- [ ] If `stablesyncnet` is used: no automated adverse legal decisions, no justice or immigration
  use, and the restrictions passed on to users.
- [ ] Attribution for CC BY data you trained or evaluated on.
- [ ] Biometric consent and retention for identity features.
- [ ] A human review step for consequential decisions.
