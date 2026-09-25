"""Learned priors (physdf phase 3): expected ranges fitted on real media only.

A detector's constraint ``k`` has a prior ``(start, full)``: no violation up to
``start``, rising to 1 at ``full``. The provisional ranges in each detector are
hand-set; ``fit_priors`` replaces them with ranges measured on real recordings:

- ``start`` is the real-data quantile ``1 - alpha / K`` of the constraint (K =
  the detector's number of constraints), so a real clip trips any one
  constraint with probability about ``alpha`` (Bonferroni over the max);
- ``full`` lies 2 more "excursions" beyond ``start``, where the excursion is
  ``start - median``, but never closer than a quarter of the provisional span,
  and not beyond the provisional ``full`` unless real data already reaches it.

The direction (higher or lower is worse) comes from the provisional prior. No
fakes are used, so the priors don't learn any particular generator.

Learned priors are a small JSON table. The package uses
``forensix/physics/learned_priors.json`` when it exists; otherwise it uses
the provisional ranges. ``detector.load_priors(path)`` loads another table.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np

FORMAT = "forensix-physics-priors/1"
PACKAGED = Path(__file__).with_name("learned_priors.json")
MIN_REAL = 100  # fewer finite real measurements: keep the provisional range


def _load_table(priors) -> dict | None:
    if priors is None:
        return json.loads(PACKAGED.read_text()) if PACKAGED.exists() else None
    table = priors if isinstance(priors, dict) else json.loads(Path(priors).read_text())
    if table.get("format") != FORMAT:
        raise ValueError(f"Not a physics priors table (expected format {FORMAT!r})")
    return table


def resolve(detector: str, provisional: dict[str, tuple[float, float]], priors=None) -> dict[str, tuple]:
    """``{constraint: (start, full, source)}``: learned where the table covers the
    constraint, provisional otherwise."""
    out = {k: (start, full, "provisional") for k, (start, full) in provisional.items()}
    table = _load_table(priors)
    for k, entry in ((table or {}).get("detectors", {}).get(detector, {})).items():
        if k not in provisional:
            continue  # a constraint this version no longer has
        start, full = entry["start"], entry["full"]
        p_start, p_full = provisional[k]
        if np.sign(full - start) != np.sign(p_full - p_start):
            raise ValueError(f"{detector}/{k}: learned prior points the wrong way")
        out[k] = (float(start), float(full), "learned")
    return out


def fit_constraint(values, provisional: tuple[float, float], quantile: float) -> dict | None:
    """Prior for one constraint from real measurements (None if too few)."""
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) < MIN_REAL:
        return None
    p_start, p_full = provisional
    sign = 1.0 if p_full > p_start else -1.0  # orient so that higher is worse
    s = sign * v
    start, median = np.quantile(s, quantile), np.median(s)
    min_span = 0.25 * abs(p_full - p_start)
    full = start + max(2.0 * (start - median), min_span)
    if sign * p_full > start + min_span:
        # the provisional full is a physical limit (e.g. jitter 0.1%); don't ramp past it
        full = min(full, sign * p_full)
    return {
        "start": round(float(sign * start), 6),
        "full": round(float(sign * full), 6),
        "n": int(len(v)),
        "quantile": round(quantile, 6),
        "median": round(float(np.median(v)), 6),
        "provisional": [p_start, p_full],
    }


def fit_priors(measurements: dict[str, dict[str, list]], provisional: dict[str, dict[str, tuple]],
               alpha: float = 0.01, sources: list[str] | None = None) -> dict:
    """Priors table from real measurements.

    ``measurements[detector][constraint]`` holds one value per real clip (None
    or NaN where not applicable); ``provisional[detector]`` holds the
    detector's provisional priors. ``alpha`` is the target false-alarm rate per
    detector on real media.
    """
    table = {"format": FORMAT, "created": date.today().isoformat(), "alpha": alpha,
             "sources": sources or [], "detectors": {}, "kept_provisional": {}}
    for detector, constraints in provisional.items():
        q = 1.0 - alpha / len(constraints)
        fitted, kept = {}, []
        for k, prior in constraints.items():
            entry = fit_constraint(measurements.get(detector, {}).get(k, []), prior, q)
            if entry is None:
                kept.append(k)
            else:
                fitted[k] = entry
        if fitted:
            table["detectors"][detector] = fitted
        if kept:
            table["kept_provisional"][detector] = kept
    return table


def provisional_priors() -> dict[str, dict[str, tuple[float, float]]]:
    """Provisional priors of every registered physics detector."""
    from forensix.physics import DETECTORS

    return {name: dict(cls.provisional_priors) for name, cls in DETECTORS.items()}
