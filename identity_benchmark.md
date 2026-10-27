# Identity-based detection: voice (ASVspoof 2019 LA)

This report covers identity-based (reference-based) detection on ASVspoof 2019 LA:
- **`id-voice`:** enrolls a person from reference recordings, then checks whether new audio is their voice.
- **Fusion with a countermeasure (CM):** `id-voice` is combined with `aasist`, an audio deepfake detector that checks how the audio was made. This setting is spoofing-aware speaker verification (SASV): accept only the real person, speaking for real.

Results: `results/sasv_asvspoof2019_la.json`. Face and video identity (`id-face`, `id-motion`) aren't covered yet; that needs DeepSpeak v2 (access requested, see the end).

## Summary

| System (eval) | SV-EER | SPF-EER | SASV-EER |
|---|---|---|---|
| `id-voice` alone | **0.76%** | 28.10% | 21.27% |
| organisers' baseline ASV (for reference) | 2.46% | 45.88% | 34.49% |
| `aasist` alone (CM) | 50.30% | **1.38%** | 24.43% |
| **`id-voice` + `aasist`, fused** | 1.60% | 2.05% | **1.92%** |

All three columns are EERs with genuine trials as positives:
- **SV-EER:** against other speakers.
- **SPF-EER:** against spoofs.
- **SASV-EER:** against both.

- **`id-voice` tells people apart well.** It separates the target speaker from other speakers at 0.76% EER, better than the organisers' baseline ASV (2.46%).
- **Voice clones fool it.** At its equal-error threshold, `id-voice` accepts 66% of spoofed trials, which are TTS and voice conversion of the target speaker. With SpeechBrain's default threshold it accepts 81%. Speaker verification can't tell a good clone from the person.
- **`aasist` has the opposite weakness.** It rejects spoofs but has no notion of identity, so it accepts every other real speaker (SV-EER 50%).
- **Fused, they cover each other: SASV-EER 1.92%.** The fusion is a logistic regression on [voice cosine, CM logit] with 3 parameters. It was fitted on the dev split, whose attacks (A01–A06) are all different from eval's (A07–A19), so the eval result is on unseen attacks.

## Data and protocol

- **Dataset:** ASVspoof 2019 LA (ODC-By 1.0; `LA.zip`, sha256 `208a7e4e…5e4ec`), with the dataset's own ASV protocol files: `ASVspoof2019.LA.asv.{dev,eval}.{female,male}.trn.txt` for enrollment and `…gi.trl.txt` for trials.
- **Eval:** 48 target speakers, each enrolled from 11 or 19 utterances (696 in total).
  - 5,370 genuine trials (`target`).
  - 33,327 trials by other speakers (`nontarget`).
  - 63,882 spoofs of the target speakers: 13 attacks, 4,914 each.
- **Dev:** used only to fit the fusion and to measure thresholds.
- **Scoring:**
  - **Voice:** cosine of each trial's embedding to the mean of the speaker's enrollment embeddings.
  - **CM:** `aasist` `fake_prob` from `scripts/batch_infer.py`, with ForensiX's default averaging over all 4.04 s windows.
  - All 96,081 files were scored, with no failures.
- **Run:** NVIDIA L4, 2026-10-26. Embedding took about 30 min for 97k files; the CM took about 20 min (58 files/s).

## `id-voice` alone

| Threshold (cosine) | Genuine rejected | Other speakers accepted | Spoofs accepted |
|---|---|---|---|
| SpeechBrain default, 0.25 | 0.02% | 18.2% | 81.3% |
| equal-error threshold on dev, 0.417 | 0.63% | 0.98% | 66.2% |

The default comes from SpeechBrain's `SpeakerRecognition.verify_batch`. It's too lenient for clips this short (2–5 s). The dev-measured threshold carries over to eval: about 1% errors in each direction. ForensiX keeps SpeechBrain's default; for short clips, pass your own threshold or calibrate it (see the README's Calibration section).

**Spoofs accepted by attack**, at the SASV equal-error threshold:

| Attack | A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `id-voice` | 43% | 10% | 0% | 73% | 71% | 31% | 5% | 45% | 63% | 80% | 0% | 0% | 0.1% |
| fused | 1.1% | 0.04% | 0% | **19.0%** | 0.1% | 4.7% | 0% | 0% | 2.3% | 2.8% | 0% | 0.04% | 0% |

- **The worst attacks for voice identity alone:** A10, A11, A16 and A15 sound like the target speaker 63–80% of the time.
- **A10 is the fusion's weak spot:** 19% of its spoofs are still accepted. The dataset's README lists it as a TTS system with a neural waveform model.
- **A17–A19 (voice conversion)** don't reach the target's voice at all.

## Fusion

The fitted weights are cosine 26.79, CM logit −0.727, bias −19.06. To accept a trial, a high voice similarity has to outweigh a CM that says "fake". At the SASV equal-error threshold, the fused system:
- rejects 1.9% of genuine trials;
- accepts 1.2% of other speakers;
- accepts 2.3% of spoofs.

## CM parity check: `aasist` on the full 2019 LA eval

Scoring every eval file allowed a check of `aasist` against its published EER on the full 71,237-trial CM protocol:

| Scoring of clips longer than 4.04 s | CM EER |
|---|---|
| upstream: first 4.04 s (shorter clips repeat-padded) | **0.82%** (published 0.83%) |
| ForensiX default: mean over all 4.04 s windows | 1.58% |

The port matches upstream. ForensiX's documented averaging of long clips roughly doubles the in-domain EER, as it does for `rawgat-st` (README, Models). This report's CM and fusion results use the ForensiX default. Changing that default is a separate decision; see the README note.

## Caveats

- **Old attacks.** ASVspoof 2019 LA's attacks date from 2019 or earlier. Current zero-shot voice cloning (e.g. ElevenLabs) is likely to sound more like the target, and to be harder for `aasist`, which was trained on this dataset's train split, attacks A01–A06. The fused result is in-domain for the CM.
- **Clean, matched audio.** Enrollment and trials come from the same corpus (studio read speech). Expect higher error rates across channels, such as a phone call against an interview.
- **`id-voice`'s embedding model** was trained on VoxCeleb, which doesn't include these speakers.

## Reproduce

```bash
L=/content/asvspoof/LA; S=/content/sasv
for s in dev eval; do
  python scripts/eval_sasv.py embed --la $L --split $s --out $S/${s}_emb.npz
  python scripts/eval_sasv.py trial-list --la $L --split $s --out $S/${s}_files.txt
  python scripts/batch_infer.py --model aasist --inputs $S/${s}_files.txt --out $S/${s}_aasist.jsonl
done
python scripts/eval_sasv.py report --la $L --emb-dev $S/dev_emb.npz --emb-eval $S/eval_emb.npz \
    --cm aasist --cm-dev $S/dev_aasist.jsonl --cm-eval $S/eval_aasist.jsonl \
    --baseline-asv $L/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt \
    --out results/sasv_asvspoof2019_la.json
```

## Next: face and video identity

`id-face` and `id-motion` need real recordings of each person plus deepfakes made of that person. DeepSpeak v2 has both (consenting participants, face swaps and voice clones of them), and access has been requested on Hugging Face. With it, the plan is:
- `id-face` against face swaps (expected to pass them: a swap is built to look like the person);
- `id-motion` and `avid` against them (expected to catch them);
- `id-voice` + CM against current commercial voice clones.
