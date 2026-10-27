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
| **Overview** | Server status, device, uptime, models in memory, requests, errors, cache hits, calibrated models; a table of loaded models (load time, calls, errors, mean time, calibration in use) with unload buttons; mean time per request per model; recent calibration jobs. Refreshes every 5 seconds |
| **Models** | Every registered detector, with search and filters by family and loaded state: whether it's loaded, needs trained weights, or has a calibration in use, and which media types it's a default for. Load or unload it |
| **Model** (click a model) | Description, license and paper; runtime statistics; every calibration version (mode, what it was built on, datasets, counts, AUC) with activate, roll back and turn off; the performance of the selected version; benchmark results |
| **Test** | Drop an image, video or audio file, pick models (or use the server's defaults for the media type), optionally an identity, then score it. Each model shows its score on a meter with the threshold, the decision, the confidence verdict when calibrated, per-segment scores, and optionally the full evidence |
| **Calibration** | Upload labelled datasets (with progress) and see their validation; start full, incremental or refit calibrations; start drift checks; follow jobs live, with progress and per-model reports |
| **Settings** | API and admin keys (kept in this browser only), theme (system, light, dark), and the server's configuration |

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
