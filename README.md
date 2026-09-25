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

Image detectors work the same way:

```python
detector = forensix.load("univfd")
detector.predict("photo.png").fake_prob     # path, PIL image or (H, W, 3) uint8 array
detector.predict_batch(["a.jpg", "b.png"])
```

Images are never resized or re-encoded before the model's own preprocessing, because the traces these detectors look for don't survive resampling or compression.

Video detectors take a file path:

```python
detector = forensix.load("dfdc-selim")
pred = detector.predict("clip.mp4")
pred.fake_prob, pred.segment_fake_probs    # video score, per-face scores
```

Audio-visual detectors take a video with an audio track (requires ffmpeg):

```python
pred = forensix.load("avad").predict("talking_head.mp4")
pred.fake_prob, pred.score_type     # 0.14, "anomaly" (higher = more likely fake)
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

**Image**

| Family | Name | License | Params | Trained on | Published result | ForensiX eval |
|---|---|---|---|---|---|---|
| image | `univfd` | MIT† | 304M (769 trainable) | ProGAN vs real (ForenSynths) | 93.4 mAP, 81.4% mean accuracy over 19 test sets (ProGAN + 18 unseen GAN, diffusion and other generators) | pending |
| image | `rine` | Apache-2.0† | 310M (6.3M trainable) | ProGAN, 4 classes (ForenSynths) | 98.8 mAP, 91.5% mean accuracy over 20 test sets (ProGAN + 19 unseen) | pending |
| image | `rine-ldm` | Apache-2.0† | 314M (10.5M trainable) | Latent Diffusion (DMID training set) | Synthbuster: 90.5 mAP, 87.5% mean accuracy over 9 generators (SDXL 99.8 AP; DALL-E 3 only 32.5 AP) | pending |
| image | `commfor` | MIT | 22M | Community Forensics: 2.7M images from ~4,800 generators (mostly SD derivatives) | 0.986 mAP, 92.3% mean accuracy over 6 benchmarks (Synthbuster 0.974 mAP) | pending |
| image | `commfor-224` | MIT | 22M | same, 224 input | 0.966 mAP, 88.9% mean accuracy over 6 benchmarks | pending |
| image | `safe` | Apache-2.0 | 1.4M | ProGAN (ForenSynths), from scratch | 96.7% ACC / 99.3 AP averaged over 33 test subsets from 26 generators; FLUX 99.3 / 99.9, PixArt 99.6 / 100 (authors' DiTFake) | pending |
| image | `clipdet` | Apache-2.0 | 303M (1,025 trainable) | 10k COCO real + 10k latent-diffusion, augmented | 90.0 mean AUC over 18 generators (GAN, diffusion, commercial tools); 85.2 on cropped/resized/compressed images | pending |

**Video**

| Family | Name | License | Params | Trained on | Published result | ForensiX eval |
|---|---|---|---|---|---|---|
| video | `dfdc-selim` | MIT‡ | 7 × 66M (ensemble) + MTCNN | DFDC training set | 1st place, Deepfake Detection Challenge: private-test log loss 0.4279; 65.18% accuracy | pending |
| video | `gend` | MIT†‡ | 303M | FaceForensics++ | 91.2% mean video-level AUROC over 14 cross-dataset benchmarks (CDFv2 94.6, DFDC 86.4; mean of 5 training seeds) | pending |
| video | `gend-pe` | MIT‡ | 316M | FaceForensics++ | 91.4% mean video-level AUROC over 14 cross-dataset benchmarks (CDFv2 95.0, DFDC 82.2; mean of 5 training seeds) | pending |
| video | `dfdet-clip` | MIT†‡ | 303M | FaceForensics++ | video-level AUROC: CDFv2 96.6, DFD 98.0, DFDC 87.2, FFIW 91.5, DeepSpeak v1 92.0 | pending |
| video | `dfdc-wm` | MIT‡ | 4 nets, 81M total (incl. RetinaFace) | DFDC training set | 2nd place, Deepfake Detection Challenge: private-test log loss 0.4284 |  pending |

**Audiovisual**

| Family | Name | License | Params | Trained on | Published result | ForensiX eval |
|---|---|---|---|---|---|---|
| audiovisual | `avad` | MIT*‡ | 41.6M + RetinaFace | real videos only (LRS2), self-supervised | FakeAVCeleb: 94.5 AUC averaged over the 4 fake-video categories; 71.6 AUC on fake audio with real video | pending |

`avad` returns an **anomaly score**, not a probability: `Prediction.score_type == "anomaly"`, and `threshold`, `is_fake` and `label` are `None`. Choose a threshold on your own data. AUC and AP are threshold-free.

**Physics (physdf)**: detectors that learn the physics of real faces and voices and flag violations. See [Physics-based detection](#physics-based-detection-physdf).

| Family | Name | License | Needs | Status |
|---|---|---|---|---|
| physics | `phys-bloodflow` | ForensiX + MediaPipe (Apache-2.0) | ≥ 6 s of a continuously visible face | signals verified on synthetic video; provisional priors |
| physics | `phys-motion` | ForensiX + MediaPipe (Apache-2.0) | ≥ 2 s of a tracked face at ≥ 20 fps (for the smoothness check) | signals verified on synthetic data; provisional priors |
| physics | `phys-lighting` | ForensiX + MediaPipe (Apache-2.0) + ICT-FaceKit (MIT) | ≥ 2 s of a tracked face | signals verified on synthetic relighting; provisional priors |
| physics | `phys-voice` | ForensiX (no model weights) | ≥ 1.5 s of voiced speech | estimators verified on synthetic speech; priors anchored on real speech; no false alarms on 10 real excerpts |
| physics | `phys-room` | ForensiX (no model weights) | ≥ 6 s of speech with pauses | verified on synthetic rooms and microphones; priors anchored on real speech; no false alarms on 10 real excerpts |

‡ Training data: several image and video models were trained on data with non-commercial terms. UnivFD, RINE and SAFE used ForenSynths (CC BY-NC-SA 4.0), `clipdet` used COCO (mixed Flickr licenses), and Community Forensics used ImageNet, LAION and others. `avad` was trained on LRS2 (BBC, research-only terms). The video models were trained on DFDC (`dfdc-selim`, `dfdc-wm`) and FaceForensics++ (`gend`, `gend-pe`, `dfdet-clip`), both of which have research-only terms of use. ForensiX lists the licenses of the code and weights, which are what govern redistribution. Whether a training set's non-commercial terms carry over to the trained weights is legally unsettled; check with counsel for commercial deployment.

† For UnivFD the code and linear head are MIT, for RINE the code and trainable weights are Apache-2.0, and the video models `gend` and `dfdet-clip` are MIT and fine-tuned from the same CLIP. Their backbone is OpenAI CLIP ViT-L/14 (MIT license). OpenAI's model card lists "any deployed use case" as out of scope, which is an intended-use statement rather than a license term.

\* The code is MIT. The weights (for `avad`, its synchronization model) are released by the authors through a link in the MIT repo's README, but they are hosted outside the repo and have no separate license of their own.

"Subset EER, upstream scores" is computed from the score files the authors released, restricted to our evaluation subset (see [Evaluation](#evaluation)). It's the number our own run should reproduce. With 1,000 bona fide trials the subset EER moves in 0.1-point steps.

- **AASIST**: Jung et al., *AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks*, ICASSP 2022 ([paper](https://arxiv.org/abs/2110.01200), [code](https://github.com/clovaai/aasist)). The network code is vendored verbatim and its outputs match the upstream evaluation pipeline exactly. The upstream MIT license is in [forensix/audio/aasist/LICENSE](forensix/audio/aasist/LICENSE).
- **RawGAT-ST**: Tak et al., *End-to-End Spectro-Temporal Graph Attention Networks for Speaker Verification Anti-Spoofing and Speech Deepfake Detection*, ASVspoof 2021 Workshop ([paper](https://arxiv.org/abs/2107.12710), [code](https://github.com/eurecom-asp/RawGAT-ST-antispoofing)). This is the element-wise-multiplication fusion variant (`RawGAT_ST_mul`), the paper's best model. The network code is vendored verbatim and its outputs match upstream exactly. The license is in [forensix/audio/rawgat_st/LICENSE](forensix/audio/rawgat_st/LICENSE).
- **XLS-R + AASIST**: Tak et al., *Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation*, Odyssey 2022 ([paper](https://arxiv.org/abs/2202.12233), [code](https://github.com/TakHemlata/SSL_Anti-spoofing)). A wav2vec 2.0 XLS-R 300M front end (Apache-2.0) fine-tuned jointly with an AASIST back end, with RawBoost augmentation. `xlsr-aasist` is the LA-track checkpoint and `xlsr-aasist-df` the DF-track one; the DF one is the better choice for real-world, compressed deepfakes. Upstream depends on fairseq. ForensiX runs the front end with Hugging Face `transformers` and converts the checkpoint on load; the back end is vendored verbatim. Its outputs match the fairseq implementation (see [Verification](#verification)). The weights (1.2 GB each) are hosted on the authors' Google Drive, and inference takes about 1.3 s per 4 s window on CPU, so use a GPU. The license is in [forensix/audio/xlsr_aasist/LICENSE](forensix/audio/xlsr_aasist/LICENSE).
- **Nes2Net-X**: Liu et al., *Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-spoofing*, IEEE T-IFS 2025 ([paper](https://arxiv.org/abs/2504.05657), [code](https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW)). A 512K-parameter nested Res2Net back end on the same XLS-R front end as XLS-R + AASIST (the shared HF port lives in [forensix/audio/xlsr.py](forensix/audio/xlsr.py)). `nes2net-x` is the ASVspoof 2021 checkpoint (an average of 5 epochs) and scores 4 s windows. `nes2net-x-itw` is trained with validation augmentation for out-of-domain audio and scores whole utterances; clips over 30 s are split into 30 s chunks. It is the best choice for real-world audio. The upstream README releases the code and models under Apache-2.0 and welcomes commercial use. The repo has no LICENSE file, so ForensiX ships the Apache-2.0 text in [forensix/audio/nes2net_x/LICENSE](forensix/audio/nes2net_x/LICENSE), with our changes noted in the file header.
- **UnivFD**: Ojha et al., *Towards Universal Fake Image Detectors that Generalize Across Generative Models*, CVPR 2023 ([paper](https://arxiv.org/abs/2302.10174), [code](https://github.com/WisconsinAIVision/UniversalFakeDetect)). A linear classifier on frozen CLIP ViT-L/14 image embeddings. Following upstream evaluation, it scores the central 224×224 crop without resizing; smaller images are zero-padded. The backbone is loaded from Hugging Face `openai/clip-vit-large-patch14` at a pinned revision and shared with other CLIP-based detectors ([forensix/image/clip.py](forensix/image/clip.py)).
- **RINE**: Koutlis & Papadopoulos, *Leveraging Representations from Intermediate Encoder-blocks for Synthetic Image Detection*, ECCV 2024 ([paper](https://arxiv.org/abs/2402.19091), [code](https://github.com/mever-team/rine)). It combines the CLS tokens of all 24 CLIP ViT-L/14 blocks with learned per-block importance weights, then an MLP head. `rine` is the 4-class ProGAN checkpoint and scores the central 224×224 crop. `rine-ldm` is trained on latent-diffusion images and is the better choice for modern diffusion output. It averages logits over ten 224×224 crops, as upstream does on Synthbuster; images under 224 px are zero-padded, where upstream would raise an error. The trainable modules keep upstream's names, so the released checkpoints load unchanged. The block features come from hooks on the shared HF CLIP backbone. Upstream ran CLIP in fp16 on GPU, while ForensiX runs fp32, so scores can differ very slightly from the paper's.
- **Community Forensics**: Park & Owens, *Community Forensics: Using Thousands of Generators to Train Fake Image Detectors*, CVPR 2025 ([paper](https://arxiv.org/abs/2411.04125), [code](https://github.com/JeongsooP/Community-Forensics)). A plain ViT-S/16, initialized from timm's AugReg ImageNet-21k weights (Apache-2.0) and trained on images from about 4,800 generators. Unlike the CLIP-based models, it follows upstream in resizing the short side (to 440, or 256 for `commfor-224`) before center-cropping 384 (or 224). Weights come from the authors' Hugging Face repos (MIT) at pinned revisions. Its training set covers a wide range of Stable Diffusion derivatives, so SDXL-family images in the ForensiX eval set are likely in-distribution for it.
- **SAFE**: Li et al., *Improving Synthetic Image Detection Towards Generalization: An Image Transformation Perspective*, KDD 2025 ([paper](https://arxiv.org/abs/2408.06741), [code](https://github.com/Ouxiang-Li/SAFE)). A 1.4M-parameter network (the first two stages of a ResNet-50) applied to the diagonal (HH) band of a one-level bior1.3 wavelet transform. It scores the central 256×256 crop without resizing or normalization. Upstream's `models/resnet.py` is about 80% shared with NPR's code, and NPR has no license. So ForensiX does not copy it: the network is written from torchvision's ResNet blocks (BSD-3), and the wavelet step is a small implementation on PyWavelets filters. The Apache-2.0 weights were trained from scratch, and they load unchanged into this implementation.
- **CLIP-based detector (GRIP)**: Cozzolino et al., *Raising the Bar of AI-generated Image Detection with CLIP*, CVPR-W 2024 ([paper](https://arxiv.org/abs/2312.00195), [code](https://github.com/grip-unina/ClipBased-SyntheticImageDetection)). A linear classifier on L2-normalized, pre-projection OpenCLIP ViT-L/14 features, using the CommonPool.XL checkpoint (MIT). It is the released `clipdet_latent10k_plus`, the paper's "10k+" variant. Its output is a log-likelihood ratio (positive means synthetic), and `fake_prob` is its sigmoid. It follows upstream in a bicubic resize of the short side to 224 before center-cropping. The backbone is run with `open_clip` (MIT), exactly as upstream does, and loaded at a pinned revision.
- **DFDC winner**: Selim Seferbekov's 1st-place solution to the [Deepfake Detection Challenge](https://ai.meta.com/blog/deepfake-detection-challenge-results-an-open-initiative-to-advance-ai/) ([code](https://github.com/selimsef/dfdc_deepfake_challenge)).
  - **Pipeline:** 32 evenly spaced frames. MTCNN finds faces on half-resolution frames, and each face is cropped with a ⅓ margin, resized to 380 and padded. Seven EfficientNet-B7 classifiers score every face. Each model's face scores go through upstream's "confident strategy", and the seven results are averaged.
  - **No face:** a video with no detectable face scores 0.5.
  - **Differences from upstream:** it runs in fp32 (upstream: fp16 on GPU). Videos with fewer than 32 frames use each frame once, where upstream drops every frame after the first repeated index and then errors to 0.5.
  - **MTCNN:** vendored from facenet-pytorch (MIT) because that package pins old torch versions. Its weights descend from the original MTCNN release (MIT).
  - **Speed:** about 2 minutes per video on CPU, so use a GPU.
- **GenD**: Yermakov et al., *Deepfake Detection that Generalizes Across Benchmarks*, WACV 2026 ([paper](https://arxiv.org/abs/2508.06248), [code](https://github.com/yermandy/GenD), [weights](https://huggingface.co/yermandy/GenD_CLIP_L_14)).
  - **Model:** CLIP ViT-L/14 with only its LayerNorm parameters fine-tuned. The features are L2-normalized and passed to a linear head.
  - **Pipeline:** the largest face in each of 32 evenly spaced frames is aligned with GenD's 5-point method (1.3× context, 256×256), then gets standard CLIP preprocessing. The video score is the mean of the per-face scores.
  - **Face detector:** upstream uses InsightFace's SCRFD detector, whose weights are non-commercial. ForensiX takes the 5 landmarks from MTCNN (MIT) instead, on frames downscaled to 640 px like SCRFD's input. So crops can differ slightly from GenD's training data.
  - **Speed:** about 25 s per video on CPU.
  - **`gend-pe`:** the same method on Meta's Perception Encoder (PE-Core L/14, Apache-2.0) through timm, with timm's eval transform (bicubic 224, mean/std 0.5), as upstream.
  - **`dfdet-clip`:** GenD's predecessor, *Unlocking the Hidden Potential of CLIP in Generalizable Deepfake Detection* ([paper](https://arxiv.org/abs/2503.19683), [code](https://github.com/yermandy/deepfake-detection), MIT), with the same face pipeline. Its Lightning checkpoint stores PEFT LayerNorm-tuning wrappers, and ForensiX folds the tuned LayerNorms into a plain CLIP. The checkpoint loads with `weights_only=True`.
  - **Upstream preprocessing is non-commercial:** DeepfakeBench (CC BY-NC) for `dfdet-clip` and InsightFace for GenD. MTCNN plus GenD's alignment replaces both.
- **AVAD**: Feng, Chen & Owens, *Self-Supervised Video Forensics by Audio-Visual Anomaly Detection*, CVPR 2023 ([paper](https://arxiv.org/abs/2301.01767), [code](https://github.com/cfeng16/audio-visual-forensics)).
  - **Method:** an audio-visual synchronization network trained only on real talking-head videos scores 31 audio offsets (±15 frames) for each 5-frame window. An autoregressive model of those offset distributions then measures how unusual the video's sync pattern is (KL divergence).
  - **Scoring:** as upstream's `detect.py` with `lam=0` and `max_len=50`. The optional PCA feature branch is omitted because it needs a pickled scikit-learn object.
  - **Face tracks:** upstream expects face tracks pre-cropped by `syncnet_python`. ForensiX builds them itself: 25 fps, RetinaFace detection with NMS, then syncnet_python's IoU tracking, median-smoothed crops and 0.4 crop scale (MIT, [forensix/audiovisual/facetrack.py](forensix/audiovisual/facetrack.py)). There is no scene-cut detection and no JPEG/XVID re-encoding.
  - **Track selection:** the longest track is scored. If none meets upstream's minimum (100 frames, 100 px face), the longest track of at least 5 frames is used.
  - **Rewritten parts:** the visual encoder's upstream code comes from an unlicensed repo, so it is written independently with the same parameter names. The log-mel front end is also reimplemented, without librosa, and without `torch.stft(return_complex=False)`, which current PyTorch has removed.
  - **Speed:** the visual embedding is computed once per window instead of 31 times; the math is identical.
- **DFDC 2nd place (Team WM)**: Zhao, Cui & Zhou ([code](https://github.com/cuihaoleo/kaggle-dfdc); results in the [challenge paper](https://arxiv.org/abs/2006.07397), Table 2).
  - **Pipeline:** every 10th frame, up to 25 faces. RetinaFace-ResNet50 (biubug6, MIT) finds the largest face, keeping upstream's lack of NMS. Each face gets an ArcFace-style 5-point alignment to 320×320, then an ensemble scores it: Xception (0.2), WS-DAN/Xception (0.7) and WS-DAN/EfficientNet-B3 (0.1). The video score is the mean.
  - **Vendored code:** the model files are vendored with their licenses (MIT, BSD-3 for the Xception port). EfficientNet-PyTorch is vendored at the exact 2020 commit WM trained with: later PyPI releases pad convolutions differently, and they change the model's outputs by up to 0.86.
  - **No scikit-image:** the alignment's similarity fit (Umeyama) is implemented directly instead of depending on it.
  - **Speed:** about 6 s per video on CPU.

## Physics-based detection (physdf)

Most detectors learn the fingerprints of particular generators. physdf instead measures physically meaningful quantities in the media, compares them with how real faces and voices behave, and reports which physical constraint was broken. Each constraint family is its own detector, and `physdf` will fuse them. Everything is built from permissively licensed parts:
- **Face geometry:** MediaPipe FaceLandmarker (Apache-2.0) and the ICT-FaceKit morphable model (MIT). FLAME is research-only, so it isn't used.
- **Signal processing:** written from the papers, instead of GPL or unlicensed toolkits.

| Detector | Physical constraint | Status |
|---|---|---|
| `phys-bloodflow` | one heart drives the pulse in all facial skin: plausible heart rate, consistent across regions, coherent signals | done (provisional priors) |
| `phys-motion` | the head is a rigid body with inertia; the jaw is a hinge | done (provisional priors) |
| `phys-lighting` | one light field: the face's lighting is stable and follows the scene | done (provisional priors) |
| `phys-voice` | voice production within anatomical limits: pitch continuity, jitter, noise, vocal-tract length, breathing | done (provisional priors) |
| `phys-room` | one room and one microphone: stable reverberation, noise floor, microphone colouration and noise spectrum; never digitally silent | done (provisional priors) |
| learned priors | expected ranges of every constraint fitted on real media only (`scripts/calibrate_physics.py`) | done (fit on Colab) |
| `physdf` | graph-network priors learned from real media only, audio-visual articulatory coupling, localization, forensic report | planned |

Physics detectors return an anomaly score in [0, 1] (`score_type="anomaly"`). It is the violation of the **most-violated constraint**, because one clearly broken physical law is evidence on its own. `Prediction.details` holds the evidence for every constraint: the measured value, where the expected range ends, and the violation. Constraints that don't apply to a clip are listed but left out of the score, e.g. jaw coupling when the mouth never moves.

```python
pred = forensix.load("phys-bloodflow").predict("clip.mp4")
pred.fake_prob                      # 0.0 (nothing violated)
pred.details["heart_rate_bpm"]      # 72.1
pred.details["constraints"]         # {"mean_coherence": {"measured": 0.986, ...}, ...}
```

**`phys-bloodflow`.**
- **Pipeline:** rPPG pulse signals from 5 skin regions on the MediaPipe mesh (forehead, cheeks, nose, chin) via POS (Wang et al., 2017), with heart rate and SNR from the 42 to 180 bpm band.
- **What it checks:** that the regions' pulses are coherent (mean correlation), that their heart rates agree, and that a pulse is detectable at all.
- **Priors:** the ranges are provisional, literature-informed values, labelled as such in `details`, until [learned priors](#learned-priors) are fitted.
- **Tested on:** synthetic video of a public-domain portrait. A coherent 72 bpm pulse scores 0.00 (72.1 bpm measured), while no pulse and different heart rates on the two halves of the face both score 1.00.
- **Limits:** rPPG needs about 6 s or more of a steady, visible face, and heavy compression weakens the signal.

**`phys-motion`.** It runs on MediaPipe head pose and landmarks, and checks:
- **Smoothness:** the share of head-rotation velocity power above 8 Hz. A head has inertia, and per-frame generation adds jitter.
- **Kinematic limits:** 99th-percentile angular velocity and acceleration.
- **Skull rigidity:** the 90th-percentile Procrustes residual of rigid landmarks (forehead midline, nose bridge, eye corners) against their consensus shape, relative to the eye distance.
- **Jaw-lip coupling:** wide mouth openings need the chin to drop.

`details["timeline"]` gives per-frame angular speed and deformation, for localizing the manipulation in time. On synthetic motion of the portrait, rigid motion scores 0.00. Per-frame rotation jitter scores 0.89 (67% of rotation energy above 8 Hz, versus 7% for smooth motion), and a 10% periodic stretch of the upper face scores 0.22 (deformation 0.016, versus a tracker noise floor of 0.006).

**`phys-lighting`.** Each frame, the ICT-FaceKit face (MIT, 9,409 vertices) is placed on the video:
- **Expression:** MediaPipe's ARKit blendshapes drive ICT's 53 expression shapes.
- **Placement:** a 3D similarity fit from 14 ICT landmark vertices onto MediaPipe's 3D landmarks, with about a 2 px residual on the test portrait.
- **Lighting:** a 9-coefficient spherical-harmonic model (Ramamoorthi & Hanrahan, 2001) is fitted to skin luminance at the camera-facing vertices. Eyes, brows and lips are excluded.

It then checks three constraints:
- **Light-direction stability:** p90 frame-to-frame change of the dominant light direction.
- **Shading plausibility:** the fit residual.
- **Face-scene coupling:** the correlation of the face's ambient light with the brightness of the frame border. Checked only when the scene's brightness actually changes.

On synthetic relighting of the portrait, which uses its fitted normals:
- **Consistent:** a global brightness change scores 0.00 (coupling 0.996).
- **Jitter:** a per-frame random light direction on the face scores 1.00 (98° jitter).
- **Decoupled:** only the background brightening and dimming scores 1.00 (coupling −0.77).

ICT's 54 source meshes (about 140 MB, pinned and hash-checked) are downloaded once and cached as a 2.5 MB file. Identity shapes aren't fitted yet.

**`phys-voice`.** It uses source-filter speech measurements written from the papers (Praat is GPL):
- **F0:** YIN.
- **Jitter and shimmer:** pitch-synchronous marks with sub-sample refinement.
- **HNR:** autocorrelation (Boersma).
- **Formants:** LPC at 10 kHz.
- **Vocal-tract length:** Fitch's formant dispersion.
- **Pauses:** an energy-based voice-activity detector.

Checks: pitch continuity (jumps over 4 semitones within 10 ms, per voiced second, after octave-error correction), too-periodic phonation (jitter), implausibly clean phonation (HNR), vocal-tract length within human anatomy (10 to 21 cm) and stable over time, and breathing (the longest pause-free stretch).

**Verification.** On source-filter speech synthesized with known physics, the estimators recover F0 within 3%, formants within 5% and tract length within 1 cm (17 and 14 cm tracts). They flag a too-periodic voice (0.71), a 28 cm tract (1.00) and 20 s without a breath (0.62).

**Real speech.** The priors are anchored just above values measured on real read speech: two CC0 LibriVox readers, ten 20 s excerpts. All ten score 0.00, and a 10 s excerpt is a test fixture. That check showed the first version of the pitch-continuity check gave false alarms on real speech (0.2 to 1.0), because of YIN octave errors at voicing edges. With octave-error correction, real speech measures 0 to 1.6 jumps per second. The check is still weak: at this tracker quality it can't separate fast genuine pitch moves from a synthetic octave-jumping voice.

**`phys-room`.** A recording happens in one room, through one microphone. It measures the recording environment from speech pauses and decays:
- **Reverberation:** blind RT60 from free decays at speech offsets (after Löllmann et al., 2010). The slow end of the decay times in consecutive stretches of the file should agree.
- **Noise floor:** the quietest level of each pause. The detector compares the quietest pause before and after every point, because breaths in pauses only raise their level.
- **Microphone self-noise:** real microphones and preamps are never digitally silent (below −90 dBFS). Synthetic or gated speech often is.
- **Channel:** the level-normalized long-term speech spectrum of each 3 s chunk (microphone and codec colouration) keeps its shape.
- **Noise spectrum:** the spectral shape of the background noise in quiet pauses doesn't switch between two regimes.

`details["room"]` gives the median RT60 and the noise floor, and `details["timeline"]` gives RT60 per stretch and the floor of each pause, for localization.

**Verification.** On synthetic speech in synthetic rooms:

| Case | Score | Measured |
|---|---|---|
| one room (RT60 0.5 s) | 0.00 | RT60 0.47 to 0.57 s |
| noise floor jumps −70 → −45 dBFS | 1.00 | 24.7 dB |
| digitally silent pauses | 1.00 | 100% of pause frames |
| microphone change (band-pass) mid-file | 1.00 | 11.3 dB |
| rooms spliced (RT60 0.15 s and 1.0 s) | 0.23 | RT60 variation 0.45 |
| white → brown noise at the same level | 0.25 | 8.2 dB |

**Real speech.** The priors are anchored above values measured on the same ten 20 s LibriVox excerpts used for `phys-voice`. All ten score 0.00, and the 10 s fixture is a test. The first version gave false alarms on real speech:
- **Noise floor:** measured per 3 s window, it read speech level in windows that were nearly all speech (jumps of 8 to 33 dB). It now uses pauses only.
- **Noise spectrum:** real pauses hold breaths, so only pauses at the floor are used.

**Limits.** Two cues are weak:
- **Noise colour:** real read speech (64 kbps MP3) reaches 6 dB of noise-spectrum change, close to the 8 dB of a white-to-brown switch.
- **Spliced rooms:** reverberation splices score only 0.23, because blind RT60 from 3 to 4 decays per stretch is noisy.

Phase-3 priors learned from real recordings should tighten both.

### Learned priors

The provisional ranges are hand-set, and the physics measurements haven't been checked on real video yet. Learned priors replace each constraint's range with one measured on **real media only**, so no generator's artefacts are learned:
- **Start of the range:** the real-data quantile 1 − α/K of the constraint, where K is the detector's number of constraints. A real clip then trips any one constraint of a detector with probability of about α (default 1%).
- **End of the range (violation 1):** two more excursions (start − median) beyond the start. It is never shorter than a quarter of the provisional range, and never past a provisional physical limit (e.g. jitter 0.1%) unless real data already reaches it.
- **Too little data:** a constraint with fewer than 100 applicable real clips keeps its provisional range.

`details["constraints"][k]["prior"]` says which kind of range each constraint used.

```python
det = forensix.load("phys-room")
det.load_priors("learned_priors.json")   # or ship it as forensix/physics/learned_priors.json (the default)
```

`scripts/calibrate_physics.py` has three steps:
1. **`measure`:** run the detectors once per clip and store the raw constraint values (JSONL, resumable). Video is tracked once, and its sound track goes to the audio detectors.
2. **`fit`:** fit the priors on real clips. It reports the false-alarm rate on held-out speakers (learned vs provisional) and writes the table.
3. **`eval`:** real-vs-fake AUC per detector, per constraint and for the max over detectors. Scores are recomputed from the stored values, so refitting doesn't mean re-measuring.

Suggested real data (commercially usable):
- **Audio:** LibriSpeech (CC BY 4.0).
- **Video with sound:** VoxCeleb2 test (CC BY 4.0; video copyright stays with the owners).

Calibrate on media like the media you'll check. Evaluate with real clips that weren't used for fitting, e.g. the real half of Deepfake-Eval-2024.

```bash
python scripts/calibrate_physics.py measure --media /content/LibriSpeech/test-clean --kind audio \
    --label real --source librispeech-test-clean --max-files 2000 --out m/librispeech.jsonl
python scripts/calibrate_physics.py measure --media /content/voxceleb2_test/mp4 --kind video --with-audio \
    --label real --source voxceleb2-test --max-files 1000 --out m/voxceleb2.jsonl
python scripts/calibrate_physics.py fit --measurements m/librispeech.jsonl m/voxceleb2.jsonl --out learned_priors.json
python scripts/calibrate_physics.py measure --media /content/de2024/video-data --kind video --with-audio \
    --labels-csv /content/de2024/video-metadata-publish.csv --source de2024 --out m/de2024.jsonl
python scripts/calibrate_physics.py eval --measurements m/de2024.jsonl --priors learned_priors.json --out results/physics_de2024.json
```

**Checked here** on synthetic audio: 120 "real" clips in varied rooms, and 40 with physical violations. On held-out speakers, learned priors cut phys-voice false alarms from 37.5% (provisional) to 6.2%. Constraints with too few applicable clips kept their provisional ranges. A smoke test on a slowly rotating photo encoded as MP4 also showed why real video is needed: the provisional bloodflow and lighting ranges flagged it.

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

**Image: ForensiX image eval set (self-built, commercially usable).** No existing real-vs-generated benchmark we found has a commercially usable license: most use real photos from ImageNet, FFHQ or LAION, or carry NC terms. So ForensiX builds its own set:

- **Real:** 1,000 Open Images V7 validation photos listed as [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/). They are stored as the original JPEG bytes, and credits go in `ATTRIBUTION.csv`.
- **Fake:** 250 images from each of four generators whose licenses allow commercial use of outputs: SDXL and PixArt-Σ (OpenRAIL++-M), Kandinsky 2.2 and FLUX.1-schnell (Apache-2.0). Revisions are pinned.
- **Matched content:** each fake is prompted with the [Localized Narratives](https://google.github.io/localized-narratives/) caption (CC BY 4.0) of a real photo in the set and generated at that photo's aspect ratio.
- **Format bias:** the reals are JPEG and the fakes PNG, and detectors can learn "not JPEG means fake". So every fake also gets a JPEG copy at the reals' median quality (estimated from their quantization tables). The `jpeg` condition is the headline number, and `png` is reported alongside.

None of these generators is in the training data of the current image models, except where noted in the Models table.

```bash
# on a GPU runtime (e.g. Colab); resumable if the session drops
pip install -e ".[eval]"
huggingface-cli login     # FLUX.1-schnell requires accepting its terms once
python scripts/build_image_eval_set.py --out /content/forensix_image_eval
python scripts/eval_image.py --data /content/forensix_image_eval --models univfd rine rine-ldm commfor commfor-224 safe clipdet \
    --device cuda --out results/image_eval.json --scores-out results/image_scores.csv
```

The eval script reports ROC-AUC, AP and balanced accuracy at the default threshold, pooled and per generator, for both conditions. On GPUs with less than 24 GB (e.g. a Colab T4), pipelines run with CPU offload. FLUX.1-schnell is the slowest there, and a T4 has no bfloat16, which FLUX may need. Use `--generators` to build the set in parts, or build FLUX on a larger GPU.

**Real-world data: Deepfake-Eval-2024** ([dataset](https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024), [paper](https://arxiv.org/abs/2503.02857)). It holds 2024 in-the-wild deepfakes and real media collected from social media and fact-checking reports: 44 h of video, 56.5 h of audio and 1,975 images. It is CC BY-SA 4.0, for evaluation only, and gated, so you request access on Hugging Face. This is the benchmark for video models, and a real-world check for the audio and image models alongside the benchmarks above.

```bash
huggingface-cli download nuriachandra/Deepfake-Eval-2024 --repo-type dataset --local-dir /content/de2024
python scripts/eval_deepfake_eval_2024.py --data /content/de2024 --modality video \
    --device cuda --max-per-class 250 --out results/de2024_video.json
# likewise --modality audio / --modality image
```

The script finds the metadata CSV's file and label columns by itself and prints them; `--file-col` and `--label-col` override that. Media a model can't decode is skipped and counted. Audio formats libsndfile can't read (e.g. m4a) are decoded with ffmpeg.

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
| `univfd` | ForensiX (HF CLIP) vs upstream (OpenAI `clip` package, official ViT-L-14.pt) on PNG/JPEG/grayscale images of 180×150 to 1024×768 | max fake_prob diff 1.1e-6 |
| `rine`, `rine-ldm` | same images, upstream `src/models.py` + checkpoint loader + test transforms (CPU fp32) | max fake_prob diff 1.5e-6 / 4.2e-7 |
| `commfor`, `commfor-224` | same images + upstream's 5 DALL-E 2 samples, upstream HF inference path (`CommForImageProcessor` + `ViTClassifier.from_pretrained`) | identical (diff 0) |
| `safe` | ForensiX reimplementation vs upstream `models/resnet.py` + eval transform; DWT step vs `pytorch_wavelets` and PyWavelets `dwt2` | max fake_prob diff 2.6e-8; DWT diff ≤ 1.2e-7 (fp32), 1e-12 (fp64) |
| `clipdet` | same images, upstream `create_architecture` + `load_weights` + `main.py` transform | max fake_prob diff 1.8e-7 |
| `dfdc-selim` | upstream `kernel_utils.py` + `DeepFakeClassifier` (CUDA/fp16 swapped for CPU/fp32 only) on test clips made from a public-domain NASA portrait | face clip: diff 1.4e-8 over 32 faces; no-face clip: both 0.5 |
| `gend` | upstream `modeling_gend.GenD` + `detector.align_face` + `CLIPProcessor` on faces from a public-domain portrait: alignment, preprocessing and model checked separately | identical (diff 0) for all three |
| `gend-pe` | same check against upstream `modeling_gend.GenD` (timm Perception Encoder) | identical (diff 0) for all three |
| `dfdet-clip` | upstream `DeepfakeDetectionModel` with its PEFT LN-tuning wrappers and `get_preprocessing()`, on the same aligned faces | identical (diff 0) |
| `avad` | upstream `detect.test2` + models (torch 1.13, librosa 0.9.2) fed the same 60-frame face track and audio; visual encoder, mel filterbank and log-mel checked separately | anomaly score diff 4.5e-8; encoder diff 0; mel filterbank 1.9e-9; log-mel 4.8e-7. Upstream took 17.5 min on CPU for this track |
| `dfdc-wm` | upstream `submission.py` loop (`FaceDetector`, scikit-image `norm_crop`, WS-DAN, Xception; CUDA swapped for CPU) on the test clips | landmarks identical; similarity matrix diff 6e-6; model ensemble on identical crops diff 2e-6. End to end 4e-3, entirely from OpenCV 5.0 vs 4.8 `warpAffine` rounding (≤3/255 per pixel) |

## Structure

```text
forensix/
├── __init__.py        # load / download / list_models / info
├── registry.py        # ModelSpec + registry
├── hub.py             # weight download, cache, checksum
├── metrics.py         # EER (ASVspoof-compatible), ROC-AUC, AP
├── types.py           # Prediction
├── audio/
│   ├── io.py          # loading, mono, resampling
│   ├── detector.py    # AudioDetector base (windowing, batching)
│   ├── xlsr.py        # shared XLS-R front end (HF) + fairseq checkpoint converter
│   ├── aasist/        # AASIST, AASIST-L
│   ├── nes2net_x/     # Nes2Net-X (2021, In-the-Wild)
│   ├── rawgat_st/     # RawGAT-ST
│   └── xlsr_aasist/   # XLS-R + AASIST (LA, DF)
├── image/
│   ├── io.py          # image loading (RGB, no resizing)
│   ├── detector.py    # ImageDetector base (batching)
│   ├── clip.py        # shared OpenAI CLIP ViT-L/14 backbone (HF)
│   ├── clipdet/       # GRIP CLIP-based detector (OpenCLIP)
│   ├── commfor/       # Community Forensics (384, 224)
│   ├── rine/          # RINE (ProGAN 4-class, LDM)
│   ├── safe/          # SAFE (reimplemented network + wavelet front end)
│   └── univfd/        # UnivFD
├── video/
│   ├── io.py          # evenly spaced frame reading (OpenCV)
│   ├── detector.py    # VideoDetector base
│   ├── faces.py       # 5-point face alignment (GenD template)
│   ├── mtcnn/         # MTCNN face detector (vendored, MIT)
│   ├── retinaface/    # RetinaFace-ResNet50 detector (vendored, MIT) + ArcFace alignment
│   ├── dfdc_selim/    # DFDC winner (7 x EfficientNet-B7)
│   ├── dfdc_wm/       # DFDC 2nd place (WS-DAN + Xception)
│   └── gend/          # GenD (CLIP, Perception Encoder) + its predecessor dfdet-clip
├── audiovisual/
│   ├── io.py          # 25 fps frames + 16 kHz audio via ffmpeg
│   ├── facetrack.py   # face tracks and talking-face crops (syncnet_python-style)
│   └── avad/          # AVAD (self-supervised A/V anomaly detection)
└── physics/           # physdf
    ├── io.py          # full-frame-rate video reading
    ├── face/          # MediaPipe tracker, skin regions, ICT-FaceKit model + fitting
    ├── signals/       # rPPG (POS, CHROM); speech: YIN, jitter, HNR, LPC formants, VTL; room: pause VAD, band spectra, blind RT60
    ├── base.py        # constraint scoring and evidence
    ├── priors.py      # learned priors: fitting and loading
    ├── calibration.py # measuring constraints on media; scoring stored measurements
    ├── bloodflow.py   # phys-bloodflow
    ├── motion.py      # phys-motion
    ├── lighting.py    # phys-lighting
    ├── voice.py       # phys-voice
    └── room.py        # phys-room
scripts/
├── download_weights.py
├── eval_asvspoof2019_la.py
├── eval_deepfake_eval_2024.py
├── calibrate_physics.py
├── build_image_eval_set.py
└── eval_image.py
```

## Adding a model

1. Put the network in `forensix/<family>/<model>/` and keep the upstream license file next to it.
2. Wrap it in a detector (e.g. subclass `AudioDetector` and implement `_fake_probs`).
3. `register(ModelSpec(...))` with the weight URL and SHA-256, license, source and paper.
4. Add tests and a row in the Models table, then run the family's eval script.

## License

Each model's code and weights carry their upstream license, listed in the Models table and in the model's directory.
