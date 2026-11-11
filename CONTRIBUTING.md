# Contributing to ForensiX

Thanks for helping. ForensiX exists so that teams fighting harmful synthetic media have detectors
they can measure, trust and legally ship. Every contribution should keep it that way.

## Ways to help

- **Report a miss or a false alarm.** Open an issue with the model, its score, and, if you're
  allowed to share it, the file. Say where the media came from (camera, platform, generator).
  Reports like these are how the benchmarks grow.
- **Add a model.** Follow [Extending ForensiX](docs/guides/extending.md). Check the license first:
  code *and* weights must allow commercial use, and the training data's terms must be known.
- **Add a benchmark.** It needs a license that allows commercial use, a fixed and documented
  subset, and an evaluation script that writes per-item scores, so the
  [leaderboard](LEADERBOARD.md) can recompute the numbers.
- **Improve the docs.** Fix what confused you; the next reader will thank you.

## Ground rules

- **Licenses:** no non-commercial (NC) or no-derivatives (ND) models, weights or datasets. Name the
  license of anything you add.
- **Numbers:** report results with dataset sizes and intervals, from a command others can rerun.
  No hand-typed metrics.
- **Ports:** a ported model must match its upstream implementation's outputs (a parity check in
  the tests or the README's Verification table).
- **Tests:** `pytest tests` passes. Tests that download weights are fine; tests that need large
  datasets or GPUs should skip cleanly without them.
- **Docs:** user-facing changes come with documentation, and `mkdocs build --strict` passes.

## Development setup

```bash
git clone https://github.com/anas-rz/ForensiX.git && cd ForensiX
pip install -e ".[dev,server,docs]"
pytest tests
mkdocs serve -a 127.0.0.1:8001
```

## Pull requests

Keep them focused: one model, one fix or one guide. Describe what changed and how you checked it.
For larger changes, open an issue first so the approach can be agreed before the work.
