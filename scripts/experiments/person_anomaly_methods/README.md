# Per-person method comparison (experiment)

The experiment behind [person_anomaly_benchmark.md](../../../person_anomaly_benchmark.md): which
small, real-only method separates a person's real media from fakes of them. It compares the
original `person-detector`, nearest neighbours on its features, identity-encoder distance (SFace,
ECAPA) and tiny convolutional autoencoders trained per person, from scratch or fine-tuned from a
base trained on other people's real media. Every method goes through the same cross-fitted
calibration. The method was fixed before any results were seen.

```bash
export PA_DATA=/path/to/data/ PA_WORK=/path/to/work/
python feats_audio.py asvspoof && python feats_audio.py libri   # ASVspoof shard + LibriSpeech dev-clean
python feats_face.py sdfvd && python feats_face.py fairface      # SDFVD + 3,000 FairFace validation faces
python evaluate.py face && python evaluate.py audio              # prints the tables; writes results_*.json
```

`PA_DATA` holds `asvspoof_hf/` (the `SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA` shard
`data/test-00000-of-00009.parquet` and its `protocols/`), `LibriSpeech/dev-clean/`, `SDFVD/` and
`fairface_val.parquet`. The results from this run are in
`results/person_anomaly/methods_*.json`. These scripts are research code. The shipped
detector is `forensix.identity.anomaly`, measured by `scripts/benchmark_person_anomaly.py`.
