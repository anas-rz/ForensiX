import numpy as np
import pytest
from PIL import Image

from forensix.fairness import estimate_skin_tone, group_of, ita_terciles, report, to_markdown, wilson


def test_grouping():
    assert [group_of("age", a) for a in (5, 17, 18, 44.9, 45, 61)] == ["0-17", "0-17", "18-29", "30-44", "45-59", "60+"]
    assert group_of("age", "20-29") == "20-29" and group_of("age", "more than 70") == "70+"
    assert [group_of("skin_tone", s) for s in ("1", "MST 5", "10", "II", "vi")] == \
        ["Monk 1-3", "Monk 4-7", "Monk 8-10", "Fitzpatrick I-II", "Fitzpatrick V-VI"]
    assert group_of("gender", "Female") == "female" and group_of("gender", "") is None and group_of("age", None) is None


def test_wilson_interval():
    lo, hi = wilson(0, 100)
    assert lo == 0 and hi == pytest.approx(0.037, abs=1e-3)
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi and hi - lo == pytest.approx(0.19, abs=0.01)


def items(rng, rates, n=200, attribute="gender"):
    """Real items whose chance of scoring above 0.5 depends on their group."""
    out = []
    for group, rate in rates.items():
        for _ in range(n):
            score = rng.uniform(0.5, 1) if rng.random() < rate else rng.uniform(0, 0.5)
            out.append({"label": "real", "attributes": {attribute: group}, "scores": {"m": score}})
    return out


def test_a_biased_model_shows_a_significant_disparity():
    rng = np.random.default_rng(0)
    rep = report(items(rng, {"female": 0.20, "male": 0.05}), {"m": {"threshold": 0.5, "threshold_source": "test"}},
                 attributes=("gender",))
    d = rep["models"]["m"]["by_attribute"]["gender"]["disparity"]
    assert (d["highest"], d["lowest"], d["significant"]) == ("female", "male", True)
    assert d["difference"] == pytest.approx(0.15, abs=0.05) and d["ratio"] > 2
    rows = {g["group"]: g for g in rep["models"]["m"]["by_attribute"]["gender"]["groups"]}
    assert rows["female"]["false_alarm_rate"]["ci"][0] < rows["female"]["false_alarm_rate"]["value"]


def test_a_fair_model_does_not():
    rng = np.random.default_rng(1)
    rep = report(items(rng, {"female": 0.10, "male": 0.10}), {"m": {"threshold": 0.5}}, attributes=("gender",))
    assert rep["models"]["m"]["by_attribute"]["gender"]["disparity"]["significant"] is False


def test_small_groups_are_reported_but_not_compared():
    rng = np.random.default_rng(2)
    data = items(rng, {"18-29": 0.1, "30-44": 0.1}, n=100, attribute="age") + \
        items(rng, {"60+": 0.9}, n=10, attribute="age")
    rep = report(data, {"m": {"threshold": 0.5}}, attributes=("age",))
    a = rep["models"]["m"]["by_attribute"]["age"]
    assert [g["group"] for g in a["groups"]] == ["18-29", "30-44", "60+"]
    assert not a["groups"][2]["reliable"] and a["disparity"]["highest"] != "60+"
    assert any("60+" in w for w in rep["models"]["m"]["warnings"])
    assert "60+ ⚠" in to_markdown(rep)


def test_models_without_a_threshold_are_flagged():
    rep = report([{"label": "real", "attributes": {"gender": "f"}, "scores": {"m": 1.0}}], {"m": {"threshold": None}})
    assert "calibrate" in rep["models"]["m"]["warnings"][0]


def test_skin_tone_estimate_and_terciles():
    rgb = np.array(Image.open("tests/data/face_public_domain.jpg").convert("RGB"))
    est = estimate_skin_tone(rgb)
    assert est is None or (isinstance(est["ita"], float) and est["category"])
    assert estimate_skin_tone(np.repeat(np.array(Image.fromarray(rgb).convert("L"))[:, :, None], 3, axis=2)) is None  # grayscale
    assert estimate_skin_tone(np.full((200, 200, 3), 128, np.uint8)) is None                                           # no face
    groups = ita_terciles({"a": 40.0, "b": 20.0, "c": -10.0, "d": None, "e": 35.0, "f": 0.0})
    assert groups["a"] == "lighter third" and groups["c"] == "darker third" and "d" not in groups
