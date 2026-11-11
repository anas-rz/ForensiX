# Evaluating a deepfake detector: a buyer's checklist

This checklist is for teams choosing a deepfake detection product or vendor: trust and safety,
fraud, identity verification, newsrooms, compliance and procurement. It's vendor-neutral. Ask
every vendor these questions, including us, and keep the answers for your risk file.

Each question says what a good answer looks like. The last column says where ForensiX answers it,
so you can check our answers the same way.

## Licensing and data

| Ask | A good answer | ForensiX |
|---|---|---|
| Can we use the models commercially? Under which licenses? | A license per model and per weight file, in writing. "Proprietary" alone isn't an answer for open models underneath | [Models](../reference/models.md), [Licensing](licensing.md) |
| What were the models trained on, and was that data licensed for this use? | Named datasets and their terms. Many public deepfake datasets are research-only | Listed per model; models without commercially usable weights are left out |
| Do you keep or train on our media? | No, or only with a contract that says so. Self-hosting removes the question | Self-hosted: media stays on your infrastructure |

## Accuracy you can verify

| Ask | A good answer | ForensiX |
|---|---|---|
| How accurate is it *on media like ours*? | An evaluation on your data, or a way to run one yourself. Vendor benchmarks alone don't transfer | [Evaluating on your data](evaluation.md) |
| What are the confidence intervals on the published numbers? | Intervals, dataset sizes and the exact test sets. A single accuracy figure hides how small most test sets are | [Leaderboard](../reference/leaderboard.md): 95% intervals, sizes, reproduce commands |
| What happens with a generator it has never seen? | Results on held-out generators, plus a way to adapt to new ones | Public benchmarks today (a multi-generator image set is pending); adapt with [fine-tuning](finetuning.md) |
| What is the false-alarm rate at the threshold you'll actually run? | A rate at a stated threshold, and a way to set the threshold for your tolerance | [Calibration](../calibration.md): a threshold for your false-alarm target |
| How does it behave after compression, resizing or re-recording? | Results on platform-processed media, since that is what you'll score | Evaluate on your own pipeline's output; [interpreting scores](interpreting-scores.md) |

## Decisions you can defend

| Ask | A good answer | ForensiX |
|---|---|---|
| Does a score mean a probability? | Calibrated probabilities on data like yours, or a clear statement that scores aren't probabilities | [Calibration](../calibration.md) |
| What does it do when it can't tell? | An explicit "uncertain" outcome rather than a forced yes or no | Prediction sets: real, fake, or both |
| What evidence comes with a result? | More than a number: which frames, which segment, which signal | Per-segment scores, evidence `details`, [provenance](../provenance.md) findings |
| Does it check provenance (C2PA, watermarks)? | Yes, and it reports what was found and what wasn't | [Provenance](../provenance.md) |

## Fairness and compliance

| Ask | A good answer | ForensiX |
|---|---|---|
| Do false alarms differ by skin tone, age or gender? | A report per group with intervals, on data you can inspect, and a way to rerun it on your population | [Fairness reports](../fairness.md) |
| Which regulations does this help with, and which does it not? | A specific, modest answer. Be wary of "fully compliant" claims for any regulation | Tools for transparency and audit evidence; compliance stays your organization's responsibility |

## Operating it

| Ask | A good answer | ForensiX |
|---|---|---|
| Where does it run? Can it run in our cloud or on-premises? | Your choice of environment, including air-gapped | Local, Kubernetes, [AWS](../aws.md), [Google Cloud](../gcp.md), [Azure](../azure.md) |
| What are latency and cost per item, on what hardware? | Measured numbers with the hardware named | Seconds per item on the [leaderboard](../reference/leaderboard.md); measure on your hardware |
| How do we know when accuracy drifts? | Monitoring plus a drift check on newly labelled data | [Web manager](../ui.md) dashboard, drift checks |
| Who reviews borderline cases? | A workflow for human review; detectors should support people, not replace them | Uncertain outcomes and evidence for reviewers |

## Red flags

- **"99% accurate"** with no dataset, no interval and no threshold.
- **No way to test on your own data** before you buy.
- **Unclear training-data licenses**, or none at all.
- **"Compliant with [regulation]"** as a product feature.
- **A yes or no with no evidence**, and no way to say "uncertain".
