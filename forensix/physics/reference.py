"""Reference scoring: physics detectors calibrated on real media.

A hand-set range turns a measurement into "inside or outside"; most clips land inside and score
exactly 0, so the score can't rank them. Reference scoring keeps the measurement's information:

1. For each constraint, a **reference distribution** of the measurement over real media (stored as
   a quantile table, no raw data). A clip's measurement becomes a one-sided tail probability
   ``p``: the share of real media at least as extreme, in the direction the physics says is
   suspicious (fixed by the detector's provisional range, never learned from fakes).
2. The constraints are combined with **Fisher's method**, ``S = sum(-ln p)`` over the constraints
   that apply, so several weak signals add up.
3. ``S`` is **calibrated on real clips scored out of fold**: the groups (speakers, channels) are
   split into folds, each clip is scored against references built without its group, and the
   detector's score is the share of those real clips with a lower ``S``. A score of 0.99 means "more unusual than 99%
   of real media", so thresholding at 0.99 gives about 1% false alarms on media like the
   reference. The detector's decision threshold is set to that value.

Per-constraint **violations** (used by the forensic report and ``physdf``) stay in [0, 1]: 0 until
a measurement is rarer than the detector's false-alarm budget split over its constraints
(Bonferroni), rising to 1 at the rarest value seen in the reference.

No fakes are used anywhere, so a reference can't learn one generator's quirks. It does encode the
recording conditions of the real media it was fitted on: fit it on media like the media you'll
check (``scripts/calibrate_physics.py fit-reference``).
"""

import json
import math
import random
from datetime import date
from pathlib import Path

import numpy as np

FORMAT = "forensix-physics-reference/1"
PACKAGED = Path(__file__).with_name("reference.json")
QUANTILES = 401          # grid points per stored distribution
MIN_REAL = 100           # real measurements a constraint needs to get a reference
MIN_CALIBRATION = 50     # held-out real clips needed to calibrate a detector
THRESHOLD = 0.99         # decision threshold: about 1% false alarms on reference-like media
ALPHA = 0.01             # false-alarm budget per detector for the per-constraint violations


def load(table=None) -> dict | None:
    """A reference table from a path or dict; None loads the packaged one, if any."""
    if table is None:
        return json.loads(PACKAGED.read_text()) if PACKAGED.exists() else None
    table = table if isinstance(table, dict) else json.loads(Path(table).read_text())
    return table if table.get("format") == FORMAT else None


def is_reference(table) -> bool:
    if table is None:
        return False
    if isinstance(table, dict):
        return table.get("format") == FORMAT
    try:
        return json.loads(Path(table).read_text()).get("format") == FORMAT
    except (OSError, ValueError):
        return False


def tail_p(x: float, quantiles: list[float], n: int) -> float:
    """P(X >= x) under a distribution stored as an evenly spaced quantile grid, floored at 1/(n+1)
    (the reference can't show anything rarer than its own sample)."""
    q = np.asarray(quantiles, dtype=np.float64)
    m, floor = len(q), 1.0 / (n + 1)
    i = int(np.searchsorted(q, x, side="left"))   # grid points strictly below x
    if i == 0:
        return 1.0
    if i == m:
        return floor
    lo, hi = q[i - 1], q[i]
    frac = (x - lo) / (hi - lo) if hi > lo else 1.0
    return max(1.0 - ((i - 1) + frac) / (m - 1), floor)


def _grid(values) -> list[float]:
    return [round(float(v), 6) for v in np.quantile(np.asarray(values, dtype=np.float64), np.linspace(0, 1, QUANTILES))]


def _sign(prior: tuple) -> float:
    start, full = prior[0], prior[1]
    return 1.0 if full > start else -1.0


def _finite(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def fisher_statistic(measured: dict, entry: dict) -> tuple[float | None, dict[str, float]]:
    """``(S, {constraint: p})`` over the constraints of ``entry`` that apply to ``measured``."""
    ps = {}
    for k, c in entry["constraints"].items():
        v = measured.get(k)
        if _finite(v):
            ps[k] = tail_p(c["sign"] * v, c["quantiles"], c["n"])
    if not ps:
        return None, ps
    return float(sum(-math.log(p) for p in ps.values())), ps


def score(measured: dict, entry: dict, provisional: dict) -> tuple[float, dict]:
    """``(score, evidence)`` for one clip; same shape as ``base.constraint_report``.

    Constraints without a reference keep their provisional range for the evidence, but only
    referenced constraints count towards the score."""
    from forensix.physics.base import violation

    stat, ps = fisher_statistic(measured, entry)
    cal = entry["calibration"]
    value = 0.0 if stat is None else 1.0 - tail_p(stat, cal["quantiles"], cal["n"])
    k_ref = max(len(entry["constraints"]), 1)
    evidence = {}
    for k, prior in provisional.items():
        v = measured.get(k)
        c = entry["constraints"].get(k)
        if c is None:   # no reference: report against the provisional range, outside the score
            start, full = prior[0], prior[1]
            e = {"expected_range_start": start, "worse": "higher" if full > start else "lower",
                 "prior": "provisional", "in_score": False}
            if _finite(v):
                e.update(measured=round(float(v), 4), violation=round(violation(v, start, full), 3))
            else:
                e.update(measured=None, violation=None, applicable=False)
            evidence[k] = e
            continue
        # the value real media stay below (or above) at this detector's false-alarm budget
        q_start = np.quantile(np.asarray(c["quantiles"]), 1.0 - ALPHA / k_ref)
        e = {"expected_range_start": round(float(c["sign"] * q_start), 4),
             "worse": "higher" if c["sign"] > 0 else "lower", "prior": "reference", "in_score": True}
        if k in ps:
            p = ps[k]
            lo, hi = -math.log(ALPHA / k_ref), -math.log(1.0 / (c["n"] + 1))
            ramp = float(np.clip((-math.log(p) - lo) / (hi - lo), 0.0, 1.0)) if hi > lo else float(p <= ALPHA / k_ref)
            e.update(measured=round(float(v), 4), p_real=round(p, 5), violation=round(ramp, 3))
        else:
            e.update(measured=None, violation=None, applicable=False)
        evidence[k] = e
    return round(value, 4), evidence


def _constraints(measured: list[dict], prior_table: dict, min_real: int) -> dict:
    out = {}
    for k, prior in prior_table.items():
        s = _sign(prior)
        vals = [s * m[k] for m in measured if _finite(m.get(k))]
        if len(vals) >= min_real:
            out[k] = {"sign": s, "n": len(vals), "median": round(float(s * np.median(vals)), 6), "quantiles": _grid(vals)}
    return out


def fit(rows: list[dict], provisional: dict[str, dict[str, tuple]], folds: int = 5, seed: int = 0,
        sources: list[str] | None = None) -> dict:
    """A reference table from real clips only.

    ``rows`` are measurement records (``scripts/calibrate_physics.py measure``): ``group`` and
    ``detectors[name]["measured"]``. The constraint distributions use every clip. The combined
    statistic is calibrated by cross-fitting over ``folds`` folds of groups: each clip is scored
    against distributions built without its group, so no clip calibrates its own score and every
    group contributes to the calibration.
    """
    table = {"format": FORMAT, "created": date.today().isoformat(), "sources": sources or [],
             "method": "fisher", "threshold": THRESHOLD, "alpha": ALPHA, "detectors": {}, "skipped": {}}
    groups = sorted({r["group"] for r in rows})
    random.Random(seed).shuffle(groups)
    fold_of = {g: i % folds for i, g in enumerate(groups)}
    for name, prior_table in provisional.items():
        measured = [(fold_of[r["group"]], m) for r in rows if (m := r.get("detectors", {}).get(name, {}).get("measured"))]
        constraints = _constraints([m for _, m in measured], prior_table, MIN_REAL)
        if not constraints:
            table["skipped"][name] = f"fewer than {MIN_REAL} real measurements for every constraint"
            continue
        stats = []
        for f in range(folds):
            others = _constraints([m for g, m in measured if g != f], {k: prior_table[k] for k in constraints}, 1)
            if not others:
                continue
            stats += [s for s in (fisher_statistic(m, {"constraints": others})[0] for g, m in measured if g == f)
                      if s is not None]
        if len(stats) < MIN_CALIBRATION or len(groups) < folds:
            table["skipped"][name] = (f"only {len(stats)} real clips in {len(groups)} groups to calibrate "
                                      f"(need {MIN_CALIBRATION} clips and {folds} groups)")
            continue
        table["detectors"][name] = {"constraints": constraints, "n_fit": len(measured),
                                    "calibration": {"n": len(stats), "folds": folds, "quantiles": _grid(stats)},
                                    "kept_provisional": sorted(set(prior_table) - set(constraints))}
    return table
