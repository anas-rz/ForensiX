# Provenance: credentials, watermarks and metadata

Detectors judge how media *looks*. Provenance checks read what the file *carries*: signed
Content Credentials, invisible watermarks embedded by generators, and the metadata tools leave
behind. They're cheap, they explain themselves, and when they find something it's stronger
evidence than any classifier score. A signed "AI-generated" declaration or a generator's
watermark is close to proof. Their silence, though, means nothing: most files carry no
provenance at all, and the evidence is easy to strip.

| Check | Reads | Can conclude | License |
|---|---|---|---|
| `c2pa` | C2PA Content Credentials (images, video, audio) | **tampered** after signing, **AI-generated** (declared), **verified capture** (with a trust list) | MIT OR Apache-2.0 (`c2pa-python`) |
| `watermark-image` | The invisible watermark of Stable Diffusion 1.x, 2.x, SDXL and Stable Video Diffusion (images, video frames) | **watermark found**, naming the generator | MIT (vendored dwtDct from `invisible-watermark`) |
| `watermark-audio` | Meta's AudioSeal watermark (audio, or a video's sound) | **watermark found**, its 16-bit message, and where in time | MIT code and weights |
| `metadata` | EXIF, XMP, PNG text chunks; container and stream tags | **AI-generated** (generator traces), **edited**, re-encoded, camera metadata | ForensiX |

```bash
pip install -e ".[provenance]"     # c2pa-python and audioseal; the server extra includes them
```

## Python

```python
import forensix

report = forensix.provenance.check("upload.jpg")      # every applicable check, combined
report["verdict"], report["label"], report["summary"]
# ('ai_generated', 'fake', 'Content Credentials from OpenAI declare AI-generated content (trainedAlgorithmicMedia).')
for f in report["findings"]:
    print(f["check"], f["severity"], f["message"])

pred = forensix.load("c2pa", trust_anchors="c2pa-trust-list.pem").predict("photo.jpg")
pred.details["verdict"], pred.details["signer"], pred.details["validation_state"]
```

Each check returns a `Prediction` with `score_type="provenance"` and no threshold. The evidence
is in `details`:

| Field | Meaning |
|---|---|
| `verdict` | What the evidence says (tables below) |
| `label` | `"fake"`, `"real"` or `None`. `None` is the common case: no decisive evidence either way |
| `summary` | One sentence for people |
| `findings` | Each piece of evidence: `code`, `severity` (`strong` decides on its own, `moderate` needs corroboration, `info` is context), `message`, `evidence` |

`fake_prob` is 1.0 for `fake`, 0.0 for `real` and 0.5 otherwise, only so provenance can sit next
to detector scores; read `details`. `check()` leads with the most decisive verdict across the
checks (tampering, then declared AI or a watermark, then verified capture) and keeps every finding.

## C2PA Content Credentials

| Verdict | Label | When |
|---|---|---|
| `tampered` | fake | The credentials are signed, but the content's hash no longer matches: it changed after signing |
| `ai_generated` | fake | The manifest or any ingredient declares AI content (IPTC `trainedAlgorithmicMedia`, `compositeWithTrainedAlgorithmicMedia`, `compositeSynthetic`) |
| `verified_capture` | real | A **trusted** signer declares a capture (camera, film) and no AI appears anywhere in the history |
| `ai_enhanced` | — | Declared algorithmic enhancement |
| `credentials_untrusted` | — | Intact and signed, but the signer isn't on your trust list |
| `trusted_credentials` | — | Trusted, but declares neither AI nor capture |
| `invalid_credentials` | — | The credentials don't validate (bad signature, broken manifest) |
| `no_credentials` | — | Nothing to verify (most files) |

**Trust.** A signature proves who signed, not that they're honest: anyone can sign "this is a
camera photo" with their own certificate. Without a trust list every signer is untrusted, so
capture claims are never verified. Pass `trust_anchors`, a PEM bundle of the root certificates
you trust (for example the C2PA conformance trust list, or your own camera fleet's CA):

```python
forensix.load("c2pa", trust_anchors="/etc/forensix/c2pa-trust.pem")
```

On the server: `"options": {"c2pa": {"trust_anchors": "/etc/forensix/c2pa-trust.pem"}}`, or the
`FORENSIX_C2PA_TRUST_ANCHORS` environment variable. AI declarations count whoever signed them.
Nobody gains by falsely labelling their own media as AI, and they never vouch for authenticity.

C2PA credentials come from some cameras and phones (e.g. recent Leica, Sony, Google Pixel and
Samsung Galaxy models), Adobe tools including Firefly, OpenAI's image generation, and other
members of the Content Authenticity Initiative. Most social platforms still strip them on upload.

## Generator watermarks

**Images** (`watermark-image`): the Stable Diffusion reference code and Hugging Face diffusers
embed a fixed message by default, with invisible-watermark's dwtDct method:

| Generator | Message | Source |
|---|---|---|
| Stable Diffusion 1.x | the bytes `StableDiffusionV1` (136 bits) | CompVis `scripts/txt2img.py` |
| Stable Diffusion 2.x | the bytes `SDV2` (32 bits) | Stability `scripts/txt2img.py` |
| SDXL, Stable Video Diffusion | a 48-bit message | Stability `generative-models`; diffusers `StableDiffusionXLWatermarker` |

The decoder is a vectorized version of the vendored reference, equal to it bit for bit, and runs
in about 60 ms on a 1024×1024 image. A message counts as found at ≥ 90% matching bits; by chance
that's below one in a million. Videos are checked on 8 frames (median).

Measured robustness, on an SDXL-watermarked 1024×1024 image:

| After | Found? |
|---|---|
| PNG, lossless WebP | yes (100% of bits) |
| JPEG quality 90–95 with full-resolution colour (4:4:4) | yes |
| JPEG with standard 4:2:0 colour, any quality; JPEG quality ≤ 80 | **no** |
| Resizing to half | **no** |

The watermark sits in the colour (U) channel, which ordinary JPEG subsamples. So the check finds
files close to the generator's output (PNG downloads, many web UIs), not media that went through
a social platform. Many hosted services and forks turn the watermark off.

**Audio** (`watermark-audio`): AudioSeal (Roman et al., ICML 2024) marks every sample. The
detector gives the share of watermarked audio, a per-second timeline (a watermarked segment
spliced into a real recording shows exactly where), and the 16-bit message. Checked here:
- **Parity:** identical outputs to upstream `AudioSeal.load_detector`.
- **Detection:** watermarked speech 1.0, clean 0.001.
- **Compression:** the watermark survives MP3 at 128 kbit/s.

It's embedded by Meta's audio generation research tools and by anyone who adopts it; most voice
cloning services don't use it.

Weights load from a pinned Hugging Face revision with SHA-256 checks. The checkpoint pickles its
training config, so it's read with an unpickler that accepts only the specific data classes
inside and refuses anything else (tested with a malicious checkpoint).

## Metadata

| Finding | Severity | Examples |
|---|---|---|
| Generator parameters | strong | Automatic1111 `parameters` chunk (steps, sampler, seed); ComfyUI `prompt`/`workflow`; InvokeAI, NovelAI chunks |
| IPTC digital source type declares AI | strong | XMP `Iptc4xmpExt:DigitalSourceType = trainedAlgorithmicMedia` |
| A generative tool in a software or encoder field | strong | EXIF `Software: Midjourney v6`; an MP4 `encoder` naming a video generator |
| A statement of origin in a comment | strong | "Generated with Runway Gen-3", "created by Midjourney" |
| A tool mentioned in free text | moderate | a description that names a generator in passing |
| Editing software | moderate | Photoshop, Lightroom, GIMP, CapCut, Premiere |
| EXIF thumbnail doesn't match the image | moderate | edited after capture without updating the thumbnail |
| Audio and video tracks of different lengths | moderate | tracks edited or muxed separately (dubbing) |
| Re-encoded by FFmpeg; no creation time; variable frame rate | info | most downloaded or processed video |
| Camera or phone metadata | info | supporting, but easy to forge |
| No metadata; a typical generator output size; a TTS-like audio format | info | weak context only |

Verdicts: `ai_generated` (label `fake`), `edited`, `re_encoded`, `capture_metadata`,
`no_metadata`, `no_findings`. Metadata never says `real`.

## On the server

The default models run all provenance checks that apply next to the detectors (the server
config `default_models` lists them):

| Media | Provenance checks by default |
|---|---|
| image | `c2pa`, `watermark-image`, `metadata` |
| video | `c2pa`, `watermark-image`, `metadata` |
| video with sound | `c2pa`, `watermark-audio`, `metadata` |
| audio | `c2pa`, `watermark-audio`, `metadata` |

They read the **file as uploaded**, even when the server re-encodes a long or large video for
the detectors, because re-encoding would destroy credentials and metadata. Each result carries
`provenance: {verdict, summary, findings}`, and its `label` is the check's own. On the dashboard,
provenance has its own tile and chart; it isn't mixed into the detector outcomes. Provenance
checks aren't calibrated, since there's no score to calibrate.

Without the `provenance` extra, `c2pa` and `watermark-audio` report `unavailable` (with the
install command) while everything else runs.

**Next step:** pair provenance with calibrated detector scores: [Calibration](calibration.md).
