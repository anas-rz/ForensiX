# Extending ForensiX

ForensiX adds models one at a time, and every model goes through the same steps:

1. Find it and check its licenses.
2. Implement it and verify parity with upstream.
3. Register it in the Python API.
4. Evaluate it.

This guide shows how, using `univfd` (`forensix/image/univfd/__init__.py`) as the worked
example.

## 1. Check licenses first

Do this before writing any code. ForensiX ships only components that can be used commercially:

- **Code:** the upstream repo's license, plus any code it borrows. SAFE's network was rewritten
  because upstream copied unlicensed NPR code.
- **Weights:** their own license if they have one. Weights linked from a licensed repo but hosted
  elsewhere with no separate license are accepted, marked with an asterisk (`MIT*`), and
  explained in the README.
- **Inherited licenses:** the backbone the weights were fine-tuned from, and preprocessing models
  the pipeline needs. SPAI was rejected because its weights derive from a non-commercial
  backbone. InsightFace and DeepfakeBench preprocessing were replaced with MTCNN or MediaPipe.
- **Reject** NC, ND, research-only or unlicensed components. Record why in the README, so
  nobody checks the same model twice.

## 2. Implement the network

Put the model in `forensix/<family>/<model>/`, with the upstream license file next to it:

```text
forensix/image/univfd/
├── __init__.py      # network, detector, build(), register()
└── LICENSE          # upstream license, verbatim
```

- **Vendor or rewrite:** copy upstream network code verbatim where its license allows, keeping
  parameter names so released checkpoints load unchanged. Rewrite from the paper where it
  doesn't.
- **Pin upstream:** pin the upstream commit (`_COMMIT`) and the weight file's SHA-256.
- **No new heavy dependencies:** if upstream needs fairseq, librosa, scikit-image or similar,
  reimplement the small part you need (see `forensix/audio/xlsr.py`).
- **Preprocessing must match upstream's evaluation pipeline:** resize or not, crop size,
  normalization, interpolation. Most parity bugs live here.

## 3. Wrap it in a detector

Subclass the base class for the input type and implement one method:

| Base class | Implement | Gets |
|---|---|---|
| `forensix.audio.detector.AudioDetector` | `_fake_probs(windows)` | `(batch, window)` float tensor at 16 kHz |
| `forensix.image.detector.ImageDetector` | `_fake_probs(batch)` | Batch of tensors from your `transform` |
| `forensix.video.detector.VideoDetector` | `_score(video_path)` | Returns `(video_prob, per_segment_probs)` |

```python
class UnivFDDetector(ImageDetector):
    def _fake_probs(self, batch: torch.Tensor) -> torch.Tensor:
        # upstream label convention: 1 = fake
        return self.network(batch).sigmoid().flatten()
```

Families without a base class (audiovisual, identity, physics) implement `predict(...)`
returning a `Prediction` directly. Follow the conventions:
- higher `fake_prob` means more fake;
- `threshold=None` and `score_type="anomaly"` for scores without a natural cut-off;
- evidence in `details`.

## 4. Register it

```python
def build(weights_path: Path, device: str = "cpu") -> UnivFDDetector:
    network = UnivFD(clip_vit_l14())
    network.fc.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return UnivFDDetector("univfd", network, TRANSFORM, device=device)


register(
    ModelSpec(
        name="univfd",
        family="image",
        builder=build,
        weights=WeightFile(url=..., sha256="4771...", filename="univfd-fc.pth"),
        extra_weights=(CLIP_VIT_L14,),            # shared backbone, fetched by download()
        license="MIT (code, linear head). Backbone: OpenAI CLIP ViT-L/14, MIT; ...",
        source=f"https://github.com/WisconsinAIVision/UniversalFakeDetect/tree/{_COMMIT}",
        paper="https://arxiv.org/abs/2302.10174",
        description="Linear probe on frozen CLIP ViT-L/14 image embeddings, ...",
    )
)
```

Then:
- import the module in the family's `__init__.py`, so registration happens on `import forensix`;
- add the license file to `[tool.setuptools.package-data]` in `pyproject.toml`;
- load checkpoints with `weights_only=True` (`forensix.hub.torch_load` allows the numpy types
  some upstream checkpoints carry) or safetensors. Never unpickle arbitrary objects;
- reuse shared backbones (`forensix/image/clip.py`, `forensix/audio/xlsr.py`, the face
  detectors in `forensix/video/`) instead of adding another copy.

## 5. Verify parity with upstream

Run upstream's own evaluation code and ForensiX's on the same inputs, and compare the outputs.
Every model in the [README Verification table](../../README.md#verification) has a row like:

| Model | Check | Result |
|---|---|---|
| `univfd` | ForensiX (HF CLIP) vs upstream (OpenAI `clip` package) on PNG/JPEG/grayscale images of 180×150 to 1024×768 | max fake_prob diff 1.1e-6 |

- Check preprocessing, network and end-to-end output separately, so a mismatch points to its
  cause.
- Use inputs that exercise the edge cases: small images, grayscale, odd sizes, short clips, no
  face.
- Explain any remaining difference (fp16 vs fp32, OpenCV versions, a replaced face detector).
- Use public-domain or CC0 test media (`tests/data/`).

## 6. Test

Add tests in `tests/test_<family>.py`:
- registration and `forensix.info` fields;
- output shape, range and label convention on a tiny or toy input;
- edge cases: an empty or short input, no face, corrupt files;
- for trainable models, a synthetic-data test that training actually learns (`tests/training/`).

Keep tests fast: build toy networks, or mark tests that need large weights.

## 7. Document and evaluate

- Add a row to the family's table in the [README](../../README.md#models), with the license
  marks (`*`, `†`, `‡`) and published results, plus a model paragraph: paper, code, what
  changed from upstream and why, speed.
- Run the family's evaluation script and record the result ([Evaluation](evaluation.md)).
- Update [Choosing detectors](choosing-detectors.md) if the model changes the recommendation.

## Adding a family

A new family (the planned `multimodal` and graph-based families) needs:
- a package `forensix/<family>/` imported in `forensix/__init__.py`;
- a base class if its models share an input type and batching logic;
- a family name added to the comment in `forensix/registry.py`;
- an evaluation path: an existing script with a new `--modality`, or a new script;
- a row in the [status table](../README.md#status-at-a-glance).
