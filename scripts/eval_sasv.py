"""Spoofing-aware speaker verification on ASVspoof 2019 LA: id-voice, a
countermeasure (CM, an audio deepfake detector), and their fusion.

Uses the dataset's official ASV protocols: each target speaker is enrolled from
the listed enrollment utterances, then every trial is scored against the claimed
speaker. Trials are genuine (``target``), other speakers (``nontarget``) or
spoofs of the target speaker (TTS / voice conversion, attacks A07-A19 on eval).

Metrics (SASV challenge, Jung et al. 2022), EER for genuine vs:
- SV-EER: nontarget; SPF-EER: spoof; SASV-EER: nontarget + spoof.

1. embed: id-voice embeddings of every enrollment and trial file (one .npz per split).

       python scripts/eval_sasv.py embed --la /content/asvspoof/LA --split eval --out /content/sasv/eval_emb.npz

2. CM scores of the trial files, e.g. with scripts/batch_infer.py
   (``trial_list`` writes the file list):

       python scripts/eval_sasv.py trial-list --la /content/asvspoof/LA --split eval --out /content/sasv/eval_files.txt
       python scripts/batch_infer.py --model aasist --inputs /content/sasv/eval_files.txt --out /content/sasv/eval_aasist.jsonl

3. report: the fusion (logistic regression on [voice cosine, CM logit]) is
   fitted on dev and applied to eval.

       python scripts/eval_sasv.py report --la /content/asvspoof/LA --emb-dev dev_emb.npz --emb-eval eval_emb.npz \\
           --cm aasist --cm-dev dev_aasist.jsonl --cm-eval eval_aasist.jsonl --out results/sasv_asvspoof2019_la.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

SPLIT_DIR = {"dev": "ASVspoof2019_LA_dev", "eval": "ASVspoof2019_LA_eval"}


def protocol(la: Path, split: str):
    """(enrollment {speaker: [utt]}, trials [(speaker, utt, attack, kind)])."""
    d = la / "ASVspoof2019_LA_asv_protocols"
    enroll = {}
    for gender in ("female", "male"):
        for line in (d / f"ASVspoof2019.LA.asv.{split}.{gender}.trn.txt").read_text().split("\n"):
            if line.strip():
                spk, utts = line.split()
                enroll[spk] = utts.split(",")
    trials = [tuple(line.split()) for line in (d / f"ASVspoof2019.LA.asv.{split}.gi.trl.txt").read_text().split("\n")
              if line.strip()]
    return enroll, trials


def audio_path(la: Path, split: str, utt: str) -> Path:
    return la / SPLIT_DIR[split] / "flac" / f"{utt}.flac"


def cmd_trial_list(args) -> None:
    _, trials = protocol(args.la, args.split)
    files = sorted({t[1] for t in trials})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(str(audio_path(args.la, args.split, u)) for u in files) + "\n")
    print(f"{len(files)} trial files -> {args.out}")


def cmd_embed(args) -> None:
    import torch

    import forensix
    from forensix.identity.voice import segments

    enroll, trials = protocol(args.la, args.split)
    utts = sorted({u for us in enroll.values() for u in us} | {t[1] for t in trials})
    det = forensix.load("id-voice", device=args.device)

    class Audio(torch.utils.data.Dataset):  # decode in worker processes
        def __len__(self):
            return len(utts)

        def __getitem__(self, i):
            return segments(det.load(audio_path(args.la, args.split, utts[i])))[0]

    loader = torch.utils.data.DataLoader(Audio(), batch_size=None, num_workers=args.workers)
    emb = np.zeros((len(utts), 192), np.float32)
    for i, segs in enumerate(loader):
        e = det.embed(segs)
        emb[i] = (e / np.linalg.norm(e, axis=1, keepdims=True)).mean(0)  # one embedding per file
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(utts)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, utts=np.array(utts), emb=emb)
    print(f"{len(utts)} embeddings -> {args.out}")


def voice_scores(la: Path, split: str, emb_path: Path):
    """Cosine of each trial to its speaker's mean enrollment embedding, plus the trial table."""
    z = np.load(emb_path)
    index = {u: i for i, u in enumerate(z["utts"])}
    emb = z["emb"] / np.linalg.norm(z["emb"], axis=1, keepdims=True)
    enroll, trials = protocol(la, split)
    centroid = {}
    for spk, us in enroll.items():
        c = emb[[index[u] for u in us]].mean(0)
        centroid[spk] = c / np.linalg.norm(c)
    cos = np.array([emb[index[utt]] @ centroid[spk] for spk, utt, _, _ in trials])
    return cos, trials


def cm_scores(path: Path, trials) -> np.ndarray:
    """CM fake probability of each trial's file (from a batch_infer.py JSONL)."""
    by_utt = {}
    for line in path.read_text().split("\n"):
        if line.strip():
            r = json.loads(line)
            by_utt[Path(r["input"]).stem] = r.get("fake_prob", np.nan)
    return np.array([by_utt.get(utt, np.nan) for _, utt, _, _ in trials], np.float64)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def sasv_metrics(score, kinds, attacks) -> dict:
    """EERs with ``score`` = higher means accept (the genuine speaker)."""
    from forensix.metrics import compute_eer

    target = score[kinds == "target"]
    eer = lambda neg: round(100 * compute_eer(target, neg)[0], 3)  # noqa: E731  (genuine as positives)
    out = {"sv_eer": eer(score[kinds == "nontarget"]), "spf_eer": eer(score[kinds == "spoof"]),
           "sasv_eer": eer(score[kinds != "target"])}
    _, thr = compute_eer(target, score[kinds != "target"])
    out["at_sasv_eer_threshold"] = {
        "genuine_rejected": round(float(np.mean(target < thr)), 4),
        "other_speakers_accepted": round(float(np.mean(score[kinds == "nontarget"] >= thr)), 4),
        "spoofs_accepted": round(float(np.mean(score[kinds == "spoof"] >= thr)), 4),
        "spoofs_accepted_by_attack": {a: round(float(np.mean(score[(kinds == "spoof") & (attacks == a)] >= thr)), 4)
                                      for a in sorted(set(attacks[kinds == "spoof"]))},
    }
    return out


def fit_fusion(x: np.ndarray, genuine: np.ndarray) -> np.ndarray:
    """Logistic regression weights (w..., b), genuine vs everything else, balanced classes."""
    from scipy.optimize import minimize

    X = np.c_[x, np.ones(len(x))]
    w_class = np.where(genuine, 0.5 / genuine.mean(), 0.5 / (1 - genuine.mean()))

    def loss(w):
        z = X @ w
        p = 1 / (1 + np.exp(-z))
        return np.sum(w_class * (np.logaddexp(0, z) - genuine * z)), X.T @ (w_class * (p - genuine))

    return minimize(loss, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def cmd_report(args) -> None:
    from forensix.identity.voice import COSINE_THRESHOLD

    data = {}
    for split, emb, cm in (("dev", args.emb_dev, args.cm_dev), ("eval", args.emb_eval, args.cm_eval)):
        cos, trials = voice_scores(args.la, split, emb)
        kinds = np.array([t[3] for t in trials])
        attacks = np.array([t[2] for t in trials])
        data[split] = {"cos": cos, "kinds": kinds, "attacks": attacks,
                       "cm": cm_scores(cm, trials) if cm else None}
    report = {"dataset": "ASVspoof 2019 LA (official ASV protocol)", "cm": args.cm, "systems": {}}
    e = data["eval"]
    report["n_trials"] = {k: int(np.sum(e["kinds"] == k)) for k in ("target", "nontarget", "spoof")}
    report["systems"]["id-voice"] = sasv_metrics(e["cos"], e["kinds"], e["attacks"])
    t = e["kinds"] == "target"
    report["systems"]["id-voice"]["at_default_threshold"] = {
        "cosine": COSINE_THRESHOLD,
        "genuine_rejected": round(float(np.mean(e["cos"][t] < COSINE_THRESHOLD)), 4),
        "other_speakers_accepted": round(float(np.mean(e["cos"][e["kinds"] == "nontarget"] >= COSINE_THRESHOLD)), 4),
        "spoofs_accepted": round(float(np.mean(e["cos"][e["kinds"] == "spoof"] >= COSINE_THRESHOLD)), 4)}
    if args.baseline_asv:  # the organisers' baseline ASV scores, for reference
        rows = [line.split() for line in args.baseline_asv.read_text().split("\n") if line.strip()]
        b_kind = np.array([r[1] for r in rows])
        b_attack = np.array(["-" if r[1] != "spoof" else r[0] for r in rows])
        report["systems"]["organisers_baseline_asv"] = sasv_metrics(np.array([float(r[2]) for r in rows]), b_kind, b_attack)
    if e["cm"] is not None:
        ok = {s: ~np.isnan(data[s]["cm"]) for s in data}
        for s in data:
            if (~ok[s]).any():
                print(f"{s}: {int((~ok[s]).sum())} trials without a CM score are left out of CM and fusion results")
        report["systems"][args.cm] = sasv_metrics(-logit(e["cm"][ok["eval"]]), e["kinds"][ok["eval"]], e["attacks"][ok["eval"]])
        d = data["dev"]
        x_dev = np.c_[d["cos"], logit(d["cm"])][ok["dev"]]
        w = fit_fusion(x_dev, d["kinds"][ok["dev"]] == "target")
        x_eval = np.c_[e["cos"], logit(e["cm"])][ok["eval"]]
        fused = np.c_[x_eval, np.ones(len(x_eval))] @ w
        name = f"id-voice + {args.cm}"
        report["systems"][name] = sasv_metrics(fused, e["kinds"][ok["eval"]], e["attacks"][ok["eval"]])
        report["systems"][name]["fusion_weights"] = {"cosine": round(float(w[0]), 4), "cm_logit": round(float(w[1]), 4),
                                                     "bias": round(float(w[2]), 4), "fitted_on": "dev"}
    for name, m in report["systems"].items():
        print(f"{name:28} SV-EER {m['sv_eer']:6.2f}%  SPF-EER {m['spf_eer']:6.2f}%  SASV-EER {m['sasv_eer']:6.2f}%")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"-> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("trial-list")
    p.add_argument("--la", type=Path, required=True)
    p.add_argument("--split", choices=["dev", "eval"], default="eval")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("embed")
    p.add_argument("--la", type=Path, required=True)
    p.add_argument("--split", choices=["dev", "eval"], default="eval")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("report")
    p.add_argument("--la", type=Path, required=True)
    p.add_argument("--emb-dev", type=Path, required=True)
    p.add_argument("--emb-eval", type=Path, required=True)
    p.add_argument("--cm", help="name of the CM model, for the report")
    p.add_argument("--cm-dev", type=Path)
    p.add_argument("--cm-eval", type=Path)
    p.add_argument("--baseline-asv", type=Path, help="ASVspoof2019.LA.asv.eval.gi.trl.scores.txt")
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    {"trial-list": cmd_trial_list, "embed": cmd_embed, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
