import json
from pathlib import Path

import numpy as np
import pytest

from forensix import leaderboard as lb

REPO = Path(__file__).resolve().parents[1]


def test_interval_helpers():
    lo, hi = lb.wilson(0, 53)
    assert lo == 0 and hi == pytest.approx(0.0676, abs=1e-3)
    assert lb.wilson(53, 53)[1] == pytest.approx(1.0)
    lo, hi = lb.hanley_mcneil(0.9, 50, 50)
    assert lo < 0.9 < hi and hi - lo == pytest.approx(0.126, abs=0.01)
    rng = np.random.default_rng(1)
    fake, real = rng.normal(1, 1, 60), rng.normal(0, 1, 60)
    a = lb.bootstrap_auc(fake, real)
    assert a == lb.bootstrap_auc(fake, real)                     # seeded: reproducible
    assert a[0] < np.mean(fake[:, None] > real[None, :]) < a[1]


def test_repo_leaderboard_matches_the_benchmark_results():
    board = lb.build(REPO / "results")
    by_id = {b["id"]: b for b in board["benchmarks"]}
    sdfvd = by_id["sdfvd"]
    assert sdfvd["status"] == "done" and len(sdfvd["rows"]) == 12
    top = sdfvd["rows"][0]
    assert (top["model"], top["rank"], top["auc"]) == ("gend-pe", 1, 0.9559)
    assert top["false_alarm_rate"]["k"] == 13 and top["false_alarm_rate"]["n"] == 53     # 1 - 0.755 real acc
    rine = next(r for r in sdfvd["rows"] if r["model"] == "rine")
    assert rine["auc"] == 0.7447 and "rounded" in rine["note"]   # the run's full-precision AUC, not the rounded scores'
    assert [r["model"] for r in sdfvd["rows"] if r["tied_with_leader"]][:3] == ["gend-pe", "gend", "dfdet-clip"]
    asv = by_id["asvspoof2019-la"]
    assert [r["eer"] for r in asv["rows"]] == sorted(r["eer"] for r in asv["rows"]) and asv["rows"][0]["auc"] is None
    assert {b["id"] for b in board["benchmarks"] if b["status"] == "pending"} == {"deepfake-eval-2024", "forensix-image-set"}
    assert board["excluded"][0]["name"] == "LAV-DF"
    assert board["throughput"]["rows"]


def test_published_leaderboard_is_up_to_date():
    """LEADERBOARD.md must be regenerated (scripts/build_leaderboard.py) whenever results change."""
    expected = lb.to_markdown(lb.build(REPO / "results"))
    assert (REPO / "LEADERBOARD.md").read_text() == expected


def test_generic_scores_benchmark(tmp_path):
    rng = np.random.default_rng(0)
    labels = ["real"] * 40 + ["fake"] * 40
    good = np.r_[rng.uniform(0, 0.4, 40), rng.uniform(0.6, 1, 40)]
    weak = rng.uniform(0, 1, 80)
    with open(tmp_path / "s.csv", "w") as f:
        f.write("file,label,good,weak\n")
        for i, (lab, g, w) in enumerate(zip(labels, good, weak)):
            f.write(f"{i}.mp4,{lab},{float(g)!r},{float(w)!r}\n")
    (tmp_path / "r.json").write_text(json.dumps({"models": {"good": {"family": "video", "seconds": 8.0},
                                                           "weak": {"family": "video", "seconds": 80.0}}}))
    (tmp_path / "leaderboard.json").write_text(json.dumps({"title": "t", "policy": "p", "benchmarks": [
        {"id": "toy", "name": "Toy", "task": "x", "license": "CC0", "results": "r.json", "scores": "s.csv",
         "reproduce": "run it"}]}))
    rows = lb.build(tmp_path)["benchmarks"][0]["rows"]
    assert [r["model"] for r in rows] == ["good", "weak"] and rows[0]["auc"] == 1.0
    assert rows[0]["false_alarm_rate"]["value"] == 0 and rows[0]["detection_rate"]["value"] == 1
    assert rows[1]["tied_with_leader"] is False and rows[0]["seconds_per_item"] == 0.1
