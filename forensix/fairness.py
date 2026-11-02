"""Fairness reporting: error rates by demographic group, for face and identity models.

The question buyers and regulators ask is whether a detector wrongly flags real
media of some groups more than others. Given scored media with demographic
attributes, ``report`` computes for each model and attribute (skin tone, age,
gender, or any other):

- the **false-alarm rate** per group: real media flagged at the model's operating
  threshold (for identity models: genuine people flagged as not themselves), with a
  Wilson 95% interval; when fakes are present, the **detection rate** per group too;
- **disparity**: the highest and lowest group rates, their ratio and difference, and a
  two-sided Fisher exact test between those two groups. A disparity counts as
  significant at p < 0.05; groups with fewer than ``min_group`` real items are
  reported but left out of the comparison, since their rates are too uncertain.

Attributes are binned so groups are large enough to measure: Monk skin tone 1-10 into
1-3 / 4-7 / 8-10, Fitzpatrick I-VI into I-II / III-IV / V-VI, ages into bands.
``estimate_skin_tone`` measures the Individual Typology Angle (ITA) of facial skin when
no skin-tone labels exist; it's a lighting-sensitive estimate, marked as such.
"""

import math
import re

import numpy as np

ATTRIBUTES = ("skin_tone", "age", "gender")
AGE_BANDS = ((0, "0-17"), (18, "18-29"), (30, "30-44"), (45, "45-59"), (60, "60+"))  # lower bounds, contiguous
# Del Bino & Bernerd (2013) ITA categories, lightest first
ITA_BANDS = ((55, "very light"), (41, "light"), (28, "intermediate"), (10, "tan"), (-30, "brown"), (-999, "dark"))
ITA_ORDER = [name for _, name in ITA_BANDS]


# -- grouping -----------------------------------------------------------------------------------

def group_of(attribute: str, value) -> str | None:
    """The reporting group of an attribute value (binned), or None when missing."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "unknown", "na", "n/a", "?"}:
        return None
    if attribute == "skin_tone":
        roman = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6}
        if text.lower() in roman:                              # Fitzpatrick I-VI
            n = roman[text.lower()]
            return "Fitzpatrick " + ("I-II" if n <= 2 else "III-IV" if n <= 4 else "V-VI")
        m = re.fullmatch(r"(?:mst|monk)?\s*(\d+)", text.lower())
        if m and 1 <= int(m.group(1)) <= 10:                   # Monk Skin Tone 1-10
            n = int(m.group(1))
            return "Monk " + ("1-3" if n <= 3 else "4-7" if n <= 7 else "8-10")
        return text
    if attribute == "age":
        m = re.fullmatch(r"(\d+(?:\.\d+)?)", text)
        if m:  # an exact age: banded
            years = float(m.group(1))
            return [name for lo, name in AGE_BANDS if years >= lo][-1]
        if m := re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", text):   # already a band (e.g. "20-29"): kept as given
            return f"{int(m.group(1))}-{int(m.group(2))}"
        if m := re.search(r"(?:more than|over|above)\s*(\d+)|(\d+)\s*\+", text.lower()):
            return f"{m.group(1) or m.group(2)}+"
        return text
    if attribute == "gender":
        return text.lower()
    return text


def _order(attribute: str, groups: list[str]) -> list[str]:
    if attribute == "age":  # by the band's lower bound
        return sorted(groups, key=lambda g: (int(m.group(1)) if (m := re.match(r"(\d+)", g)) else 999, g))
    if attribute == "skin_tone":
        known = list(TERCILES) + ITA_ORDER + ["Monk 1-3", "Monk 4-7", "Monk 8-10", "Fitzpatrick I-II", "Fitzpatrick III-IV", "Fitzpatrick V-VI"]
        return sorted(groups, key=lambda g: (known.index(g) if g in known else 99, g))
    return sorted(groups)


# -- statistics ---------------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def fisher_p(k1: int, n1: int, k2: int, n2: int) -> float:
    from scipy.stats import fisher_exact

    return float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"value": round(k / n, 4) if n else None, "k": int(k), "n": int(n), "ci": [round(lo, 4), round(hi, 4)]}


# -- report -------------------------------------------------------------------------------------

def report(items: list[dict], models: dict[str, dict], attributes=ATTRIBUTES, min_group: int = 30) -> dict:
    """``items``: ``{"label": "real"|"fake", "attributes": {...}, "scores": {model: score}}``.
    ``models``: ``{model: {"threshold": float, "threshold_source": str, "kind": "detector"|"identity"}}``.
    A score at or above the threshold is flagged (the convention of every ForensiX detector)."""
    out = {"min_group": min_group, "n_items": len(items), "attributes": list(attributes), "models": {}}
    for model, meta in models.items():
        thr = meta.get("threshold")
        entry = {**meta, "by_attribute": {}, "warnings": []}
        scored = [(it, it["scores"].get(model)) for it in items]
        scored = [(it, s) for it, s in scored if isinstance(s, (int, float)) and math.isfinite(s)]
        if thr is None or not isinstance(thr, (int, float)) or not math.isfinite(thr):
            entry["warnings"].append("No operating threshold (an anomaly score without calibration): calibrate the "
                                     "model to report false-alarm rates.")
            out["models"][model] = entry
            continue
        real = [s for it, s in scored if it["label"] == "real"]
        fake = [s for it, s in scored if it["label"] == "fake"]
        entry["overall"] = {"false_alarm_rate": _rate(sum(s >= thr for s in real), len(real)),
                            "detection_rate": _rate(sum(s >= thr for s in fake), len(fake)) if fake else None}
        for attr in attributes:
            groups: dict[str, dict] = {}
            for it, s in scored:
                g = group_of(attr, (it.get("attributes") or {}).get(attr))
                if g is None:
                    continue
                d = groups.setdefault(g, {"real": [], "fake": []})
                d[it["label"]].append(s)
            if not groups:
                continue
            rows = []
            for g in _order(attr, list(groups)):
                r, f = groups[g]["real"], groups[g]["fake"]
                row = {"group": g, "n_real": len(r), "n_fake": len(f),
                       "false_alarm_rate": _rate(sum(s >= thr for s in r), len(r)),
                       "detection_rate": _rate(sum(s >= thr for s in f), len(f)) if f else None,
                       "reliable": len(r) >= min_group}
                if r and f:
                    from forensix.metrics import roc_auc

                    row["auc"] = round(roc_auc(np.array(f), np.array(r)), 4)
                rows.append(row)
            entry["by_attribute"][attr] = {"groups": rows, "disparity": _disparity(rows, min_group),
                                           "estimated": attr == "skin_tone" and any(g["group"] in ITA_ORDER + list(TERCILES) for g in rows)}
            small = [r["group"] for r in rows if not r["reliable"]]
            if small:
                entry["warnings"].append(f"{attr}: {', '.join(small)} {'has' if len(small) == 1 else 'have'} fewer than "
                                         f"{min_group} real items; reported but left out of the disparity comparison.")
        out["models"][model] = entry
    return out


def _disparity(rows: list[dict], min_group: int) -> dict | None:
    usable = [r for r in rows if r["reliable"] and r["false_alarm_rate"]["n"]]
    if len(usable) < 2:
        return None
    hi = max(usable, key=lambda r: r["false_alarm_rate"]["value"])
    lo = min(usable, key=lambda r: r["false_alarm_rate"]["value"])
    h, l = hi["false_alarm_rate"], lo["false_alarm_rate"]
    p = fisher_p(h["k"], h["n"], l["k"], l["n"])
    return {"highest": hi["group"], "lowest": lo["group"], "highest_rate": h["value"], "lowest_rate": l["value"],
            "difference": round(h["value"] - l["value"], 4),
            "ratio": round(h["value"] / l["value"], 3) if l["value"] else None,  # undefined when the lowest is 0
            "p_value": round(p, 6), "significant": p < 0.05, "groups_compared": len(usable)}


# -- skin tone estimate ---------------------------------------------------------------------------

TERCILES = ("lighter third", "middle third", "darker third")


def ita_terciles(itas: dict) -> dict:
    """Group estimated ITA values into the dataset's lighter, middle and darker thirds.

    Del Bino's absolute ITA categories assume calibrated colorimetry; ordinary photos are
    darker and vary in exposure, so absolute categories mislabel them (light skin often
    measures near 0°). The ordering of ITA holds up, so relative groups are the robust
    choice for comparing error rates. ``itas``: {key: ita}; returns {key: group}."""
    values = np.array([v for v in itas.values() if v is not None])
    if len(values) < 3:
        return {}
    lo, hi = np.percentile(values, [100 / 3, 200 / 3])
    return {k: (TERCILES[0] if v >= hi else TERCILES[2] if v < lo else TERCILES[1])
            for k, v in itas.items() if v is not None}


def ita_category(ita: float) -> str:
    return next(name for lo, name in ITA_BANDS if ita > lo)


def estimate_skin_tone(image_rgb: np.ndarray, detector=None) -> dict | None:
    """Individual Typology Angle (ITA) of the largest face's skin, and its Del Bino category.

    Samples both cheeks and the forehead, keeps pixels whose colour is plausibly skin
    (CIELAB a* and b* positive, no shadows or highlights) and takes their median. Returns
    None when the estimate would be meaningless: no face, a grayscale photo, a profile view
    (eyes too close together), or too few skin pixels. ``detector`` is an OpenCV
    FaceDetectorYN (YuNet), created on first use when None. Lighting and white balance
    shift ITA, so the category is an estimate, not a self-reported skin tone."""
    import cv2

    if detector is None:
        detector = _yunet()
    h, w = image_rgb.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if faces is None or not len(faces):
        return None
    f = max(faces, key=lambda r: r[2] * r[3])
    x, y, fw, fh = (float(v) for v in f[:4])
    eye_r, eye_l, nose = f[4:6], f[6:8], f[8:10]
    box = image_rgb[max(0, int(y)):int(y + fh), max(0, int(x)):int(x + fw)].astype(int)
    if box.size == 0 or np.mean(np.abs(box[..., 0] - box[..., 1]) + np.abs(box[..., 1] - box[..., 2])) < 6:
        return None                                            # grayscale: no colour to measure
    if np.hypot(*(eye_l - eye_r)) < 0.25 * fw:
        return None                                            # profile: cheeks not visible
    half = max(3, int(0.08 * fw))
    eye_mid = (eye_r + eye_l) / 2
    centres = [((eye_r[0] + nose[0]) / 2, nose[1]), ((eye_l[0] + nose[0]) / 2, nose[1]),
               (eye_mid[0], eye_mid[1] - 0.12 * fh)]           # both cheeks, lower forehead
    patches = []
    for cx, cy in centres:
        x0, x1, y0, y1 = int(cx - half), int(cx + half), int(cy - half), int(cy + half)
        if 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h:
            patches.append(image_rgb[y0:y1, x0:x1].reshape(-1, 3))
    if not patches:
        return None
    pixels = np.concatenate(patches).astype(np.uint8)[:, None, :]
    lab = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)
    L, a, b = lab[:, 0] * 100 / 255, lab[:, 1] - 128, lab[:, 2] - 128   # OpenCV 8-bit Lab -> CIELAB
    skin = (L > 15) & (L < 95) & (a > 2) & (a < 40) & (b > 3) & (b < 45)
    if skin.sum() < max(20, 0.3 * len(L)):
        return None                                            # mostly not skin (hair, glasses, background)
    ita = math.degrees(math.atan((np.median(L[skin]) - 50) / np.median(b[skin])))
    return {"ita": round(ita, 1), "category": ita_category(ita)}


def _yunet():
    import cv2

    from forensix.hub import fetch
    from forensix.identity.face import YUNET

    return cv2.FaceDetectorYN.create(str(fetch(YUNET)), "", (320, 320), 0.8, 0.3, 50)


# -- markdown -------------------------------------------------------------------------------------

def to_markdown(rep: dict, title: str = "Fairness report") -> str:
    def pct(x):
        return "—" if x is None else f"{x * 100:.1f}%"

    def rate(r):
        return "—" if not r or r.get("value") is None else f"{pct(r['value'])} [{pct(r['ci'][0])}, {pct(r['ci'][1])}] (n={r['n']})"

    lines = [f"# {title}", "",
             f"{rep['n_items']} items. False alarms = real media flagged at each model's operating threshold "
             f"(identity models: genuine people flagged). 95% Wilson intervals. Disparity compares the highest and lowest "
             f"groups with at least {rep['min_group']} real items (two-sided Fisher exact test).", ""]
    for model, m in rep["models"].items():
        lines += [f"## `{model}`", ""]
        if "overall" in m:
            lines += [f"Threshold {m.get('threshold')} ({m.get('threshold_source', '—')}). Overall false alarms: "
                      f"{rate(m['overall']['false_alarm_rate'])}.", ""]
        for attr, a in m["by_attribute"].items():
            label = attr.replace("_", " ") + (" (estimated from photos: ITA thirds of this dataset)" if a.get("estimated") else "")
            lines += [f"### {label}", "", "| Group | Real | False alarms | Fake | Detection |", "|---|---|---|---|---|"]
            for g in a["groups"]:
                lines.append(f"| {g['group']}{'' if g['reliable'] else ' ⚠'} | {g['n_real']} | {rate(g['false_alarm_rate'])} | "
                             f"{g['n_fake']} | {rate(g['detection_rate'])} |")
            d = a["disparity"]
            if d:
                ratio = f"{d['ratio']:.2f}×" if d["ratio"] is not None and math.isfinite(d["ratio"]) else "no ratio (lowest is 0%)"
                lines += ["", f"Highest {d['highest']} ({pct(d['highest_rate'])}) vs lowest {d['lowest']} "
                          f"({pct(d['lowest_rate'])}): {ratio}, {d['difference'] * 100:+.1f} points, p = {d['p_value']:.3g} — "
                          + ("**significant**" if d["significant"] else "not significant") + "."]
            lines += [""]
        lines += [f"- {w}" for w in m.get("warnings", [])] + [""]
    return "\n".join(lines)
