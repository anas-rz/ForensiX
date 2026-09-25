"""Forensic report for physdf results.

Every statement comes from the structured evidence: each claim names the
physical law, the measured value, where the real range ends, whether that
range was calibrated on real media, and where in time and which part of the
face or voice it happened. The report lists counter-evidence (checks that
passed) and what couldn't be checked. It abstains when the evidence is thin or
weak. ``verify_report`` re-derives every claim and the verdict from the result;
a report that fails verification is marked inconclusive.

Generator attribution is not attempted: the priors are trained on real media
only, so nothing here knows what particular generators look like.
"""

import numpy as np

# constraint -> (physical law, unit of the measurement)
LAWS: dict[str, tuple[str, str]] = {
    "mean_coherence": ("one heart drives every facial skin region, so their pulse signals move together", "mean correlation"),
    "hr_spread_bpm": ("every skin region shows the same heart rate", "bpm spread"),
    "median_snr_db": ("a pulse is visible in real facial skin", "dB pulse SNR"),
    "rotation_hf_energy_ratio": ("a head has inertia, so its rotation is smooth", "share of rotation energy above 8 Hz"),
    "p99_angular_velocity_deg_s": ("head rotation stays within biomechanical speed limits", "deg/s"),
    "p99_angular_acceleration_deg_s2": ("head rotation stays within biomechanical acceleration limits", "deg/s²"),
    "rigid_deformation": ("the skull is rigid", "× eye distance"),
    "jaw_lip_coupling": ("the jaw is a hinge: opening the mouth drops the chin", "correlation"),
    "light_direction_jitter_deg": ("the scene light changes direction slowly", "deg between frames"),
    "shading_residual": ("the skin is shaded by one smooth light field", "relative residual"),
    "face_scene_coupling": ("the face's light follows the scene's brightness", "correlation"),
    "f0_jump_rate_per_s": ("vocal folds change pitch continuously", "jumps/s"),
    "jitter_percent": ("vocal-fold vibration is never perfectly periodic", "% jitter"),
    "hnr_db": ("real phonation carries turbulence noise", "dB HNR"),
    "vtl_out_of_range_cm": ("the vocal tract has a human length (10–21 cm)", "cm outside"),
    "vtl_variation": ("one speaker's vocal-tract length is stable", "relative variation"),
    "longest_pause_free_s": ("a speaker has to breathe", "s without a pause"),
    "rt60_variation": ("one room: the reverberation stays the same", "coefficient of variation"),
    "noise_floor_jump_db": ("one microphone: the noise floor doesn't jump", "dB"),
    "digital_silence_fraction": ("real microphones are never digitally silent", "fraction of pause frames"),
    "channel_variation_db": ("one microphone: the channel colouring is stable", "dB"),
    "noise_spectrum_change_db": ("the background noise keeps its spectrum", "dB"),
    "graph_skull_residual": ("head pose and skull shape evolve as real heads do, given the rest of the face", "σ above real"),
    "graph_muscle_residual": ("jaw, lips and upper face move together as real facial muscles do", "σ above real"),
    "graph_bloodflow_residual": ("each skin region's pulse follows the others (one heart)", "σ above real"),
    "graph_lighting_residual": ("the face's lighting evolves with head motion as in real video", "σ above real"),
    "graph_voice_residual": ("pitch, vocal-tract resonances and breathing fit each other as in real speech", "σ above real"),
    "graph_av_mouth_from_audio": ("the mouth moves as the speech sound requires", "σ above real"),
    "graph_av_audio_from_video": ("the speech sound matches the visible mouth", "σ above real"),
}
MIN_APPLICABLE = 5  # fewer applicable checks: abstain
STRONG = 0.5


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    return f"{v:.0f}" if 1000 <= abs(v) < 1e5 else f"{v:.3g}"


def _claims(result: dict) -> list[dict]:
    claims = []
    segments = result.get("localization", {}).get("segments", [])
    for key, e in result["constraints"].items():
        if not e.get("violation"):
            continue
        detector, k = key.split("/", 1)
        law, unit = LAWS.get(k, (k, ""))
        claims.append({
            "constraint": key, "law": law, "unit": unit, "measured": e["measured"],
            "expected_range_start": e["expected_range_start"], "worse": e.get("worse", "higher"),
            "violation": e["violation"],
            "prior": "calibrated on real media" if (e.get("prior") == "learned" or detector == "physdf") else "provisional",
            "real_rarity": e.get("real_rarity"),
            "where": [s for s in segments if s["constraint"] == k][:5],
        })
    return sorted(claims, key=lambda c: -c["violation"])


def _verdict(result: dict, claims: list[dict]) -> tuple[str, str, list[str]]:
    applicable = [e for e in result["constraints"].values() if e.get("violation") is not None]
    strong = [c for c in claims if c["violation"] >= STRONG]
    reasons = []
    if len(applicable) < MIN_APPLICABLE:
        reasons.append(f"only {len(applicable)} physical checks applied to this clip (need {MIN_APPLICABLE})")
    if claims and not strong:
        reasons.append(f"the strongest violation is weak ({claims[0]['violation']:.2f} < {STRONG})")
    if strong:
        calibrated = [c for c in strong if c["prior"] != "provisional"]
        systems = {c["constraint"].split("/")[0] for c in strong}
        if calibrated and (max(c["violation"] for c in calibrated) >= 0.8 or len(systems) >= 2):
            confidence = "high"
        elif calibrated:
            confidence = "medium"
        else:
            confidence = "low"
            reasons.append("the violated ranges are provisional (not yet calibrated on real media)")
        return "physical violations found", confidence, reasons
    if reasons:
        return "inconclusive", "none", reasons
    return "no physical violation found", "medium" if len(applicable) >= 2 * MIN_APPLICABLE else "low", reasons


def build_report(result: dict) -> dict:
    claims = _claims(result)
    verdict, confidence, reasons = _verdict(result, claims)
    counter = []
    not_applicable = []
    for key, e in result["constraints"].items():
        law = LAWS.get(key.split("/", 1)[1], (key, ""))[0]
        if e.get("violation") == 0:
            counter.append({"constraint": key, "law": law, "measured": e["measured"],
                            "expected_range_start": e["expected_range_start"]})
        elif e.get("violation") is None:
            not_applicable.append({"constraint": key, "law": law})
    report = {
        "verdict": verdict,
        "confidence": confidence,
        "score": result["score"],
        "claims": claims,
        "counter_evidence": counter,
        "not_applicable": not_applicable,
        "abstain_reasons": reasons,
        "status": result.get("status", {}),
        "attribution": "not attempted: physdf models real media only",
    }
    report["markdown"] = to_markdown(report)
    failures = verify_report(report, result)
    report["verification"] = {"passed": not failures, "failures": failures}
    if failures:
        report["verdict"], report["confidence"] = "inconclusive", "none"
        report["abstain_reasons"] = reasons + ["the report failed its own verification"]
        report["markdown"] = to_markdown(report)
    return report


def to_markdown(report: dict) -> str:
    lines = ["# physdf forensic report", "",
             f"**Verdict:** {report['verdict']} (confidence: {report['confidence']}; anomaly score {report['score']:.2f})", ""]
    if report["abstain_reasons"]:
        lines += ["**Caveats:** " + "; ".join(report["abstain_reasons"]) + ".", ""]
    if report["claims"]:
        lines += ["## Findings", ""]
        for i, c in enumerate(report["claims"], 1):
            text = (f"{i}. **{c['law'][0].upper() + c['law'][1:]}.** Measured {_fmt(c['measured'])} {c['unit']}; "
                    f"real media stay {'below' if c['worse'] == 'higher' else 'above'} {_fmt(c['expected_range_start'])} (violation {c['violation']:.2f}, "
                    f"range {c['prior']}; `{c['constraint']}`).")
            if c["real_rarity"] is not None:
                text += f" Beyond {c['real_rarity']:.1%} of real clips."
            if c["where"]:
                text += " Where: " + ", ".join(
                    f"{s['start_s']:.0f}–{s['end_s']:.0f} s ({s['modality']}, strongest at `{s['node']}`)" for s in c["where"]) + "."
            lines.append(text)
        lines.append("")
    if report["counter_evidence"]:
        lines += ["## Counter-evidence (checks that passed)", ""]
        lines += [f"- {c['law'][0].upper() + c['law'][1:]}: {_fmt(c['measured'])} (`{c['constraint']}`)"
                  for c in report["counter_evidence"]]
        lines.append("")
    if report["not_applicable"]:
        lines += ["## Not checked", "", ", ".join(f"`{c['constraint']}`" for c in report["not_applicable"]), ""]
    skipped = {k: v for k, v in report["status"].items() if v != "ran"}
    if skipped:
        lines += ["## Not run", ""] + [f"- `{k}`: {v}" for k, v in skipped.items()] + [""]
    lines += ["## Method", "",
              "Each check compares a physically meaningful measurement with the range of real media. The score is "
              "the most-violated check. Ranges marked *provisional* are hand-set and not yet calibrated on real "
              f"media. Generator attribution: {report['attribution']}."]
    return "\n".join(lines)


def verify_report(report: dict, result: dict) -> list[str]:
    """Re-derive the report from the result; return every mismatch."""
    failures = []
    constraints = result["constraints"]
    for c in report["claims"]:
        e = constraints.get(c["constraint"])
        if e is None:
            failures.append(f"claim on unknown constraint {c['constraint']}")
            continue
        if e.get("violation") is None or e["violation"] <= 0:
            failures.append(f"claim on {c['constraint']}, which isn't violated")
        if c["measured"] != e["measured"] or c["violation"] != e["violation"]:
            failures.append(f"claim on {c['constraint']} doesn't match the measurement")
        if f"Measured {_fmt(c['measured'])}" not in report["markdown"]:
            failures.append(f"claim on {c['constraint']} missing from the text")
    violated = {k for k, e in constraints.items() if e.get("violation")}
    if violated != {c["constraint"] for c in report["claims"]}:
        failures.append("the claims don't cover exactly the violated constraints")
    for c in report["counter_evidence"]:
        if constraints.get(c["constraint"], {}).get("violation") != 0:
            failures.append(f"counter-evidence {c['constraint']} is not a passed check")
    if abs(report["score"] - max([e["violation"] for e in constraints.values() if e.get("violation") is not None] or [0.0])) > 1e-9:
        failures.append("score is not the largest violation")
    expected = _verdict(result, _claims(result))
    if report["verdict"] != expected[0]:
        failures.append(f"verdict {report['verdict']!r} doesn't follow from the evidence ({expected[0]!r})")
    return failures


def render_figures(result: dict, out_dir) -> list:
    """Timeline figure of the graph residuals (needs matplotlib, not a dependency)."""
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loc = result.get("localization", {})
    timeline = loc.get("timeline_z", {})
    if not timeline:
        return []
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for group, z in timeline.items():
        t = np.arange(len(z)) * loc["window_s"]
        ax.plot(t, [np.nan if v is None else v for v in z], label=group.replace("graph_", ""))
    for s in loc.get("segments", []):
        ax.axvspan(s["start_s"], s["end_s"], color="red", alpha=0.1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("residual (σ above real windows)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = out / "physdf_timeline.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return [path]
