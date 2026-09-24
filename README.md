# ForensiX

A PyTorch model zoo for **deepfake detection and multimodal media forensics**: audio, video, audiovisual, multimodal, physics-based, image, identity-based and graph-based detectors behind one Python API.

All bundled models and weights are under **commercially usable licenses**.

## Install

```bash
pip install -e ".[dev]"
python scripts/download_weights.py          # optional: pre-fetch all weights
```

Weights are cached in `~/.cache/forensix` (override with `FORENSIX_CACHE`) and checked against a SHA-256 hash. `forensix.load()` downloads them automatically on first use.

## Usage

```python
import forensix

forensix.list_models()                 # ['aasist', 'aasist-l', 'nes2net-x', 'nes2net-x-itw', 'rawgat-st', ...]
forensix.list_models("audio")
forensix.info("aasist")                # license, source, paper, description

detector = forensix.load("aasist")     # device="cuda" for GPU
pred = detector.predict("clip.flac")   # any format libsndfile reads (wav, flac, ogg, mp3)
pred.fake_prob, pred.label             # 0.97, 'fake'
pred.segment_fake_probs                # per-4s-window scores for long clips

detector.predict(waveform, sample_rate=44100)       # numpy array input
detector.predict_batch(["a.wav", "b.wav"])

forensix.load("aasist", weights_path="my_finetuned.pth")  # your own checkpoint
```

Audio is converted to mono and resampled to 16 kHz. Clips longer than the model window (4.04 s) are split into windows and the window scores are averaged. `fake_prob` is the softmax output and is **not calibrated**, so pick thresholds per model.

## Models

| Family | Name | License | Params | Trained on | Published EER | Subset EER, upstream scores | Subset EER, ForensiX |
|---|---|---|---|---|---|---|---|
| audio | `aasist` | MIT | 297K | ASVspoof 2019 LA | 0.83% (2019 LA eval) | n/a | pending |
| audio | `aasist-l` | MIT | 85K | ASVspoof 2019 LA | 0.99% (2019 LA eval) | n/a | pending |
| audio | `rawgat-st` | MIT | 441K | ASVspoof 2019 LA | 1.06% (2019 LA eval) | 1.30% | pending |
| audio | `xlsr-aasist` | MIT* | 316M | ASVspoof 2019 LA | 0.82% (2021 LA eval) | n/a | pending |
| audio | `xlsr-aasist-df` | MIT* | 316M | ASVspoof 2019 LA | 2.85% (2021 DF eval) | n/a | pending |
| audio | `nes2net-x` | Apache-2.0 | 316M (512K back end) | ASVspoof 2019 LA | 1.49% (2021 DF), 1.88% (2021 LA) | n/a | pending |
| audio | `nes2net-x-itw` | Apache-2.0 | 316M (512K back end) | ASVspoof 2019 LA | 5.15% (In-the-Wild) | n/a | pending |

\* The code is MIT. The weights are released by the authors through a link in the MIT repo's README, but they are hosted outside the repo and have no separate license of their own.

"Subset EER, upstream scores" is computed from the score files the authors released, restricted to our evaluation subset (see [Evaluation](#evaluation)). It's the number our own run should reproduce. With 1,000 bona fide trials the subset EER moves in 0.1-point steps.

- **AASIST**: Jung et al., *AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks*, ICASSP 2022 ([paper](https://arxiv.org/abs/2110.01200), [code](https://github.com/clovaai/aasist)). The network code is vendored verbatim and its outputs match the upstream evaluation pipeline exactly. The upstream MIT license is in [forensix/audio/aasist/LICENSE](forensix/audio/aasist/LICENSE).
- **RawGAT-ST**: Tak et al., *End-to-End Spectro-Temporal Graph Attention Networks for Speaker Verification Anti-Spoofing and Speech Deepfake Detection*, ASVspoof 2021 Workshop ([paper](https://arxiv.org/abs/2107.12710), [code](https://github.com/eurecom-asp/RawGAT-ST-antispoofing)). This is the element-wise-multiplication fusion variant (`RawGAT_ST_mul`), the paper's best model. The network code is vendored verbatim and its outputs match upstream exactly. The license is in [forensix/audio/rawgat_st/LICENSE](forensix/audio/rawgat_st/LICENSE).
- **XLS-R + AASIST**: Tak et al., *Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation*, Odyssey 2022 ([paper](https://arxiv.org/abs/2202.12233), [code](https://github.com/TakHemlata/SSL_Anti-spoofing)). A wav2vec 2.0 XLS-R 300M front end (Apache-2.0) fine-tuned jointly with an AASIST back end, with RawBoost augmentation. `xlsr-aasist` is the LA-track checkpoint and `xlsr-aasist-df` the DF-track one; the DF one is the better choice for real-world, compressed deepfakes. Upstream depends on fairseq. ForensiX runs the front end with Hugging Face `transformers` and converts the checkpoint on load; the back end is vendored verbatim. Its outputs match the fairseq implementation (see [Verification](#verification)). The weights (1.2 GB each) are hosted on the authors' Google Drive, and inference takes about 1.3 s per 4 s window on CPU, so use a GPU. The license is in [forensix/audio/xlsr_aasist/LICENSE](forensix/audio/xlsr_aasist/LICENSE).
- **Nes2Net-X**: Liu et al., *Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-spoofing*, IEEE T-IFS 2025 ([paper](https://arxiv.org/abs/2504.05657), [code](https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW)). A 512K-parameter nested Res2Net back end on the same XLS-R front end as XLS-R + AASIST (the shared HF port lives in [forensix/audio/xlsr.py](forensix/audio/xlsr.py)). `nes2net-x` is the ASVspoof 2021 checkpoint (an average of 5 epochs) and scores 4 s windows. `nes2net-x-itw` is trained with validation augmentation for out-of-domain audio and scores whole utterances; clips over 30 s are split into 30 s chunks. It is the best choice for real-world audio. The upstream README releases the code and models under Apache-2.0 and welcomes commercial use. The repo has no LICENSE file, so ForensiX ships the Apache-2.0 text in [forensix/audio/nes2net_x/LICENSE](forensix/audio/nes2net_x/LICENSE), with our changes noted in the file header.

## Evaluation

Each model is evaluated on a fixed, reproducible subset of a benchmark with a commercially usable license.

**Audio: ASVspoof 2019 LA eval** ([ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/), [download](https://datashare.ed.ac.uk/handle/10283/3336)). The subset is 1,000 bona fide trials plus 77 trials from each of the 13 attacks A07–A19 (2,001 total, seed 0). The script reports pooled EER and per-attack EER.

```bash
python scripts/eval_asvspoof2019_la.py \
    --protocol LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \
    --audio-dir LA/ASVspoof2019_LA_eval/flac \
    --models aasist aasist-l rawgat-st xlsr-aasist xlsr-aasist-df nes2net-x nes2net-x-itw --device cuda --out results/asvspoof2019_la_subset.json
```

The dataset is not downloaded by this repo. Run the script wherever the data lives (e.g. Google Colab).

## Verification

Every model is checked against its upstream implementation before it's added.

| Model | Check | Result |
|---|---|---|
| `aasist`, `aasist-l` | ForensiX vs upstream `main.py` eval pipeline (padding, scoring), random inputs of 0.5–4 s | identical (diff 0) |
| `rawgat-st` | same as above; upstream released scores give 1.06% EER on full 2019 LA eval | identical (diff 0); matches paper |
| `xlsr-aasist`, `xlsr-aasist-df` | HF `transformers` front end vs upstream fairseq front end (fairseq at the commit bundled upstream), both checkpoints | max logit diff 4e-6 |
| `xlsr-aasist`, `xlsr-aasist-df` | upstream released scores + 2021 keys | 0.82% (2021 LA), 2.85% (2021 DF): matches paper |
| `nes2net-x`, `nes2net-x-itw` | HF front end vs upstream fairseq, both checkpoints, 4 s batches and whole utterances of 0.5–6 s | max logit diff 3e-5 |
| `nes2net-x` | upstream released scores + 2021 DF keys | 1.49% (2021 DF): matches paper |
| `nes2net-x-itw` | In-the-Wild EER | not checked: needs the dataset's labels |

## Structure

```text
forensix/
├── __init__.py        # load / download / list_models / info
├── registry.py        # ModelSpec + registry
├── hub.py             # weight download, cache, checksum
├── metrics.py         # EER (ASVspoof-compatible)
├── types.py           # Prediction
└── audio/
    ├── io.py          # loading, mono, resampling
    ├── detector.py    # AudioDetector base (windowing, batching)
    ├── xlsr.py        # shared XLS-R front end (HF) + fairseq checkpoint converter
    ├── aasist/        # AASIST, AASIST-L
    ├── nes2net_x/     # Nes2Net-X (2021, In-the-Wild)
    ├── rawgat_st/     # RawGAT-ST
    └── xlsr_aasist/   # XLS-R + AASIST (LA, DF)
scripts/
├── download_weights.py
└── eval_asvspoof2019_la.py
```

## Adding a model

1. Put the network in `forensix/<family>/<model>/` and keep the upstream license file next to it.
2. Wrap it in a detector (e.g. subclass `AudioDetector` and implement `_fake_probs`).
3. `register(ModelSpec(...))` with the weight URL and SHA-256, license, source and paper.
4. Add tests and a row in the Models table, then run the family's eval script.

## License

Each model's code and weights carry their upstream license, listed in the Models table and in the model's directory.
