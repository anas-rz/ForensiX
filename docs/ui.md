# Web manager

The ForensiX server includes a web manager. Use it to test files, load and unload models,
calibrate them on your data, and watch how models perform. Open it at
`http://<server>:8000/ui/`; the server root redirects there.

```bash
pip install -e ".[server]"
python -m forensix.server --config server.json      # then open http://localhost:8000/ui/
```

It's a static page served by the server (`forensix/server/static/`). It has no build step and
loads nothing from the internet, so it also works on air-gapped deployments. Everything it shows
comes from the [HTTP API](server.md), so anything you do in the UI can be scripted.

## Pages

| Page | What you can do |
|---|---|
| **Leaderboard** | Every model on commercially usable benchmarks ([below](#the-leaderboard)): AUC with 95% intervals as a forest plot, EER, false-alarm and detection rates at a fixed threshold, speed; run details, caveats and a copyable reproduce command; pending benchmarks; a throughput table |
| **Overview** | A live dashboard of this server ([below](#the-overview-dashboard)): requests, outcomes, latency, cache and resource use over the last 15 minutes to 24 hours, for all models or one; loaded models and recent jobs. Refreshes every 5 seconds |
| **Models** | Every registered detector, with search and filters by family and loaded state: whether it's loaded, needs trained weights, or has a calibration in use, and which media types it's a default for. Load or unload it |
| **Model** (click a model) | Description, license and paper; runtime statistics; every calibration version (mode, what it was built on, datasets, counts, AUC) with activate, roll back and turn off; the performance of the selected version; benchmark results |
| **Test** | Drop an image, video or audio file, pick models (or use the server's defaults for the media type), optionally an identity, then score it. Each model shows its score on a meter with the threshold, the decision, the confidence verdict when calibrated, per-segment scores, and optionally the full evidence |
| **Calibration** | Upload labelled datasets (with progress) and see their validation; start full, incremental or refit calibrations; start drift checks; follow jobs live, with progress and per-model reports |
| **Fairness** | Start a [fairness report](fairness.md) on a dataset with demographic columns; per model and attribute, each group's false-alarm rate with its interval against the overall rate, disparity verdicts and warnings; download as Markdown or JSON |
| **Settings** | API and admin keys (kept in this browser only), theme (system, light, dark), and the server's configuration |

## The overview dashboard

Filters sit in one row at the top and apply to everything below them: a **time range** (15 min,
1 hour, 6 hours, 24 hours) and a **model** (all, or one). They're remembered in the browser, and
a link can carry them, e.g. `/ui/#/overview?window=900&model=dfdc-wm`.

| Panel | Shows |
|---|---|
| Status strip | Ready, device, uptime, models in memory, calibrated models, version |
| Tiles | Requests, flagged fake (share of decided), uncertain, error rate, latency p50 (with p95), cache hit rate; most with a trend sparkline |
| Requests over time | Stacked columns per time bucket, by outcome: real, fake, uncertain / no decision, error |
| Latency | p50 and p95 per bucket, for requests computed by a model (cache hits excluded). Pick a model for its own latency; "all models" mixes fast and slow ones |
| Outcomes by model | Each model's share of real, fake, uncertain and error |
| Requests by model, Media types | Volume per model and per media type (hidden when one model is selected) |
| Score distribution | With one model selected: its live scores (calibrated probabilities when calibrated) and its threshold |
| Provenance evidence | Verdicts of the provenance checks (C2PA, watermarks, metadata), with a tile counting files that carry AI evidence. Kept apart from the detector outcomes |
| Memory, CPU, GPU memory | Resident memory, CPU (100% = one core) and, on GPU servers, allocated and reserved GPU memory, sampled every 10 s |
| Models in memory, Recent jobs | As before, with unload buttons |

**Outcomes** come from the calibrated verdict when a model is calibrated, otherwise from its
label at the threshold. *Uncertain / no decision* covers uncertain and out-of-distribution
verdicts, anomaly scores without a threshold, and videos without a face.

**Data** comes from `GET /v1/telemetry?window=…&model=…`. It's held in memory: the last 50,000
model runs and 24 hours of resource samples. It starts empty when the server starts, and covers
this one instance. For a fleet, or for history beyond a restart, use the Prometheus metrics at
`/metrics`.

Every chart has a tooltip on hover, and the main ones have a **Table view**.

## The leaderboard

One tab per benchmark, plus **Speed**. Links can open a tab: `/ui/#/leaderboard?b=asvspoof2019-la`.

- **Forest plot:** each model's AUC (dot) and 95% interval (line). The shaded band is the
  leader's interval: a model whose line reaches into it is **tied** with the leader, because the
  data can't separate them. The line at 0.5 is chance. Benchmarks without saved per-item scores
  show EER instead (lower is better).
- **Table:** rank, AUC [95% CI], EER, false alarms (real media flagged) and detections (fakes
  caught) at the fixed threshold with their intervals, seconds per item, and "vs #1".
  Footnotes flag corrected values and timings measured on a shared GPU.
- **Run details:** dataset, license, archive hash, hardware, date, commit and result files,
  then caveats and the exact command to reproduce.
- **Not run yet / not included:** benchmarks waiting to be run (with their commands), and
  benchmarks left out because of their license.

The same data is in [LEADERBOARD.md](../LEADERBOARD.md) and at `GET /v1/leaderboard`
([Evaluation: the leaderboard](guides/evaluation.md#the-leaderboard)).

## Reading model performance

A model's page shows the performance of a calibration version on its own calibration data,
measured out of fold (each item scored by a calibration fitted without it):

- **AUC:** how well the model separates real from fake on your data. Calibration doesn't
  change it; a low AUC means you need a different model.
- **False alarms:** the share of real items flagged at the conformal threshold. It should be
  close to the target `alpha`.
- **Fakes caught:** the detection rate at that threshold.
- **Calibration error (ECE):** raw → calibrated. Lower means the probabilities can be taken at
  face value.
- **Certain verdicts:** how often the verdict is `fake` or `real` rather than `uncertain`.
- **Score distribution:** raw scores of real and fake calibration items, with the decision
  threshold. Overlap is where the model can't tell them apart. **Table view** shows the counts.
- **Out-of-fold decisions:** balanced accuracy, false alarms, detection rate, ECE and Brier score
  for three rules: the model's default threshold, calibrated probability ≥ 0.5, and the conformal
  threshold.

**Benchmarks** lists results found in the server's `benchmarks_dir`: any JSON with a `models`
dictionary of metrics, as written by `scripts/eval_*.py`. By default that's the repo's
`results/` folder, so the SDFVD and ASVspoof results appear on each model's page.

## Access

The page itself has no secrets, so it loads without a key. Every call it makes uses the keys
you enter in **Settings**: the API key for reading and testing, and the admin key for loading
models, uploading datasets and changing calibrations. They're kept in the browser's
`localStorage` and sent as the `X-API-Key` header. Use HTTPS through a reverse proxy when the
server isn't on localhost. Turn the UI off with `"ui": false` in the server config.
