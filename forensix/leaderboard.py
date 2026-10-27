"""The ForensiX leaderboard: every model on commercially usable benchmarks, recomputed from raw results.

``results/leaderboard.json`` lists the benchmarks (license, dataset, hardware, commit,
reproduce command, caveats) and points at their result files. This module computes,
per benchmark and model:

- **AUC** with a 95% confidence interval: a stratified bootstrap (reals and fakes
  resampled separately, 2,000 times, seed 0) when per-item scores were saved, else
  the Hanley-McNeil normal approximation;
- **EER**;
- **false-alarm rate** (real media flagged) and **detection rate** (fakes caught) at a
  fixed threshold: 0.5 on ``fake_prob`` for detectors, "any violation" for physics,
  each with a Wilson 95% interval;
- **speed**: seconds per item on the recorded hardware;
- whether the model is **statistically tied** with the top model (overlapping AUC
  intervals).

Nothing is typed in by hand: change a result file and the leaderboard follows.
``scripts/build_leaderboard.py`` writes LEADERBOARD.md; the server serves the same
data at ``/v1/leaderboard`` and in the web manager.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np

from forensix.metrics import compute_eer, roc_auc

THRESHOLD = 0.5
BOOTSTRAP, SEED = 2000, 0


def default_root() -> Path:
    return Path(__file__).resolve().parents[1] / "results"


def build(root: str | Path | None = None) -> dict:
    """The full leaderboard from ``<root>/leaderboard.json``."""
    root = Path(root) if root else default_root()
    manifest = json.loads((root / "leaderboard.json").read_text())
    benchmarks = []
    for spec in manifest["benchmarks"]:
        entry = {k: v for k, v in spec.items() if k not in ("results", "scores", "format")}
        if spec.get("status") == "pending" or not (root / spec.get("results", "")).is_file():
            entry.update(status="pending", rows=[])
        else:
            fmt = spec.get("format", "scores")
            rows = {"scores": _from_scores, "asvspoof": _from_asvspoof, "physics": _from_physics}[fmt](root, spec)
            _rank(rows, by="eer" if fmt == "asvspoof" else "auc")
            entry.update(status="done", rows=rows, files=[f for f in (spec.get("results"), spec.get("scores")) if f])
        benchmarks.append(entry)
    out = {"title": manifest.get("title"), "policy": manifest.get("policy"), "benchmarks": benchmarks,
           "excluded": manifest.get("excluded", []), "threshold": THRESHOLD,
           "method": {"auc_ci": f"stratified bootstrap, {BOOTSTRAP} resamples, seed {SEED} (Hanley-McNeil when "
                                "per-item scores weren't saved)", "rate_ci": "Wilson score interval, 95%"}}
    if tp := manifest.get("throughput"):
        out["throughput"] = {**{k: v for k, v in tp.items() if k != "file"}, "rows": _throughput(root / tp["file"])}
    return out


# -- per-format readers ---------------------------------------------------------------------

def _from_scores(root: Path, spec: dict) -> list[dict]:
    """A results JSON (models: summaries) plus a per-item scores CSV (file, label, <model>...)."""
    summary = json.loads((root / spec["results"]).read_text())
    with open(root / spec["scores"], newline="") as f:
        table = list(csv.DictReader(f))
    is_fake = np.array([r["label"] == "fake" for r in table])
    rows = []
    for model, s in summary["models"].items():
        scores = np.array([float(r[model]) if r.get(model) not in (None, "") else np.nan for r in table])
        ok = np.isfinite(scores)
        fake, real = scores[ok & is_fake], scores[ok & ~is_fake]
        auc, eer = roc_auc(fake, real), compute_eer(fake, real)[0]
        lo, hi = bootstrap_auc(fake, real)
        note = None
        # older runs saved scores rounded to 6 decimals; for models with tiny scores that creates ties
        # and lowers the recomputed AUC. The run's own summary was computed at full precision.
        if isinstance(s.get("auc"), (int, float)) and abs(s["auc"] - auc) > 5e-4:
            note = (f"Saved per-item scores were rounded to 6 decimals, which ties this model's tiny scores; "
                    f"AUC and EER are the run's full-precision values (recomputed from saved scores: AUC {auc:.3f}). "
                    "The interval comes from the rounded scores, so it's approximate.")
            auc, eer = s["auc"], s.get("eer", eer)
        n = int(ok.sum())
        seconds = s.get("seconds")
        rows.append({
            "model": model, "family": s.get("family"), "n_real": int(real.size), "n_fake": int(fake.size),
            "auc": round(auc, 4), "auc_ci": [round(lo, 4), round(hi, 4)], "auc_ci_method": "bootstrap",
            "note": note,
            "eer": round(eer, 4),
            "threshold": THRESHOLD,
            "false_alarm_rate": _rate(int((real >= THRESHOLD).sum()), real.size),
            "detection_rate": _rate(int((fake >= THRESHOLD).sum()), fake.size),
            "seconds_per_item": round(seconds / n, 3) if seconds and n else None,
            "timed_alone": model in spec.get("timed_alone", []) if "timed_alone" in spec else None,
            "failed": s.get("failed", 0),
        })
    return rows


def _from_asvspoof(root: Path, spec: dict) -> list[dict]:
    """EER and accuracy summaries only (no per-trial scores were saved)."""
    data = json.loads((root / spec["results"]).read_text())
    n_real, n_fake = spec["n_real"], spec["n_fake"]
    rows = []
    for model, s in data["models"].items():
        eer = s["eer_percent"] / 100
        rows.append({
            "model": model, "family": "audio", "n_real": n_real, "n_fake": n_fake,
            "auc": None, "auc_ci": None, "eer": round(eer, 4), "eer_ci": _eer_ci(eer, n_real, n_fake),
            "accuracy": s.get("accuracy_at_default_threshold"), "threshold": THRESHOLD,
            "false_alarm_rate": None, "detection_rate": None,
            "seconds_per_item": round(s["seconds"] / (n_real + n_fake), 4) if s.get("seconds") else None,
            "per_attack_eer": s.get("per_attack_eer_percent"),
        })
    return rows


def _from_physics(root: Path, spec: dict) -> list[dict]:
    data = json.loads((root / spec["results"]).read_text())
    rows = []
    for model, s in data["detectors"].items():
        n_real, n_fake = s["n_real"], s["n_fake"]
        lo, hi = hanley_mcneil(s["auc"], n_fake, n_real)
        fa = s["real_false_alarms"]
        rows.append({
            "model": model, "family": "physics", "n_real": n_real, "n_fake": n_fake,
            "auc": s["auc"], "auc_ci": [round(lo, 4), round(hi, 4)], "auc_ci_method": "Hanley-McNeil",
            "eer": None, "threshold": "any violation",
            "false_alarm_rate": _rate(round(fa["any_violation"] * fa["n"]), fa["n"]),
            "detection_rate": _rate(round(s["fake_flagged"] * n_fake), n_fake),
            "seconds_per_item": None,
        })
    return rows


def _throughput(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        runners = {k: r[k] for k in ("baseline", "runner_fp32", "runner_bf16") if k in r}
        rows.append({"model": r["model"], "n": r["n"], "gpu": r.get("gpu"),
                     **{f"{k}_items_per_s": v.get("items_per_s") for k, v in runners.items()},
                     "bf16_max_score_change": (runners.get("runner_bf16") or {}).get("max_abs_diff_vs_baseline"),
                     "bf16_auc": (runners.get("runner_bf16") or {}).get("auc"),
                     "baseline_auc": (runners.get("baseline") or {}).get("auc")})
    return sorted(rows, key=lambda r: -(r.get("runner_fp32_items_per_s") or r.get("baseline_items_per_s") or 0))


# -- statistics -----------------------------------------------------------------------------

def bootstrap_auc(fake: np.ndarray, real: np.ndarray, n: int = BOOTSTRAP, seed: int = SEED) -> tuple[float, float]:
    """95% percentile interval of AUC, resampling fakes and reals separately."""
    rng = np.random.default_rng(seed)
    fake, real = np.asarray(fake, float), np.asarray(real, float)
    # AUC = P(fake > real) + 0.5 P(tie), computed on the pairwise comparison matrix
    cmp = (fake[:, None] > real[None, :]) + 0.5 * (fake[:, None] == real[None, :])
    aucs = np.empty(n)
    for k in range(n):
        i = rng.integers(0, fake.size, fake.size)
        j = rng.integers(0, real.size, real.size)
        aucs[k] = cmp[np.ix_(i, j)].mean()
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def hanley_mcneil(auc: float, n_fake: int, n_real: int) -> tuple[float, float]:
    """95% normal-approximation interval of AUC (Hanley & McNeil, 1982)."""
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n_fake - 1) * (q1 - auc * auc) + (n_real - 1) * (q2 - auc * auc)) / (n_fake * n_real)
    se = math.sqrt(max(var, 0.0))
    return max(0.0, auc - 1.96 * se), min(1.0, auc + 1.96 * se)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson(int(k), int(n))
    return {"value": round(k / n, 4) if n else None, "k": int(k), "n": int(n), "ci": [round(lo, 4), round(hi, 4)]}


def _eer_ci(eer: float, n_real: int, n_fake: int) -> list[float]:
    """Approximate interval: EER averages a false-alarm and a miss rate at one threshold."""
    se = math.sqrt(eer * (1 - eer) * (1 / n_real + 1 / n_fake) / 4) if 0 < eer < 1 else 0.0
    lo, hi = wilson(round(eer * (n_real + n_fake)), n_real + n_fake) if eer == 0 else (eer - 1.96 * se, eer + 1.96 * se)
    return [round(max(0.0, lo), 4), round(min(1.0, hi), 4)]


def _rank(rows: list[dict], by: str) -> None:
    """Sort best first, number the ranks, and mark models whose interval overlaps the leader's."""
    key = (lambda r: -(r["auc"] if r["auc"] is not None else -1)) if by == "auc" else (lambda r: r["eer"])
    rows.sort(key=key)
    lead = rows[0] if rows else None
    ci = "auc_ci" if by == "auc" else "eer_ci"
    for i, r in enumerate(rows, 1):
        r["rank"] = i
        if lead and r is not lead and r.get(ci) and lead.get(ci):
            # tied: the intervals overlap (higher AUC / lower EER is better)
            r["tied_with_leader"] = (r[ci][1] >= lead[ci][0]) if by == "auc" else (r[ci][0] <= lead[ci][1])
        else:
            r["tied_with_leader"] = None if r is not lead else True


# -- markdown ---------------------------------------------------------------------------------

def to_markdown(board: dict) -> str:
    """LEADERBOARD.md: the same numbers, for publishing."""
    def pct(x):
        return "—" if x is None else f"{x * 100:.1f}%"

    def rate(r):
        return "—" if not r or r.get("value") is None else f"{pct(r['value'])} [{pct(r['ci'][0])}, {pct(r['ci'][1])}]"

    lines = [f"# {board['title']}", "", board["policy"], "",
             f"- **AUC 95% CI:** {board['method']['auc_ci']}.",
             f"- **Rates at a fixed threshold** ({board['threshold']} on `fake_prob`; physics: any violation): "
             f"{board['method']['rate_ci']}. False alarm = real media flagged; detection = fakes caught.",
             "- **Tied:** the model's interval overlaps the leader's, so the data can't separate them.",
             "- Regenerate with `python scripts/build_leaderboard.py`.", ""]
    for b in board["benchmarks"]:
        lines += [f"## {b['name']}", "", f"{b['task']}. License: {b['license']}."
                  + (f" [Dataset]({b['dataset_url']})." if b.get("dataset_url") else ""), ""]
        if b["status"] == "pending":
            lines += ["**Not run yet.** Reproduce:", "", "```bash", b["reproduce"], "```", ""]
            continue
        if b.get("description"):
            lines += [b["description"], ""]
        lines += [f"Hardware: {b.get('hardware') or 'not recorded'} · date: {b.get('date') or 'not recorded'} · "
                  f"commit: `{(b.get('commit') or 'not recorded')[:12]}`", ""]
        if any(r.get("auc") is not None for r in b["rows"]):
            lines += ["| # | Model | AUC [95% CI] | EER | False alarms | Detection | s/item | Tied with #1 |",
                      "|---|---|---|---|---|---|---|---|"]
            for r in b["rows"]:
                mark = "¹" if r.get("note") else ""
                lines.append(f"| {r['rank']} | `{r['model']}`{mark} | {r['auc']:.3f} [{r['auc_ci'][0]:.3f}, {r['auc_ci'][1]:.3f}] | "
                             f"{pct(r.get('eer'))} | {rate(r['false_alarm_rate'])} | {rate(r['detection_rate'])} | "
                             f"{r['seconds_per_item'] if r.get('seconds_per_item') is not None else '—'} | "
                             f"{'—' if r['rank'] == 1 else ('yes' if r.get('tied_with_leader') else 'no')} |")
        else:
            lines += ["| # | Model | EER [≈95% CI] | Accuracy @0.5 | s/item |", "|---|---|---|---|---|"]
            for r in b["rows"]:
                lines.append(f"| {r['rank']} | `{r['model']}` | {pct(r['eer'])} [{pct(r['eer_ci'][0])}, {pct(r['eer_ci'][1])}] | "
                             f"{pct(r.get('accuracy'))} | {r['seconds_per_item'] or '—'} |")
        lines += [""]
        notes = [r for r in b["rows"] if r.get("note")]
        if notes:
            lines += [f"¹ {', '.join('`' + r['model'] + '`' for r in notes)}: {notes[0]['note']}", ""]
        if b.get("timing"):
            lines += [f"Speed: {b['timing']}", ""]
        lines += [f"- {c}" for c in b.get("caveats", [])] + ["", "Reproduce:", "", "```bash", b["reproduce"], "```", ""]
    if tp := board.get("throughput"):
        lines += ["## Speed", "", tp["description"], "", "| Model | Items | Per-file loop | Batch runner fp32 | Batch runner bf16 | bf16 max score change |",
                  "|---|---|---|---|---|---|"]
        for r in tp["rows"]:
            lines.append(f"| `{r['model']}` | {r['n']} | {r.get('baseline_items_per_s') or '—'}/s | "
                         f"{r.get('runner_fp32_items_per_s') or '—'}/s | {r.get('runner_bf16_items_per_s') or '—'}/s | "
                         f"{r['bf16_max_score_change']:.4f} |" if r.get("bf16_max_score_change") is not None else
                         f"| `{r['model']}` | {r['n']} | {r.get('baseline_items_per_s') or '—'}/s | — | — | — |")
        lines += [""]
    if board.get("excluded"):
        lines += ["## Not included", ""] + [f"- **{e['name']}** ({e['license']}): {e['reason']}" for e in board["excluded"]] + [""]
    return "\n".join(lines)
