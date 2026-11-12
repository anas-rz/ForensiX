"""Benchmark person-anomaly (forensix.identity.anomaly): per-person detectors fitted on real media only.

Faces, SDFVD (53 real/face-swap pairs; vsN is a face swap made from vN):
    each person is enrolled on the frames of the first half of their real clip; the last 40% of the
    real clip (after a gap) are their held-out real frames, and every frame of the face swap made
    from that clip is a fake. All frames are used (the clips are 2-4 s).
Voice, ASVspoof 2019 LA eval (ODC-By), one parquet shard of the SpeechAntiSpoofingBenchmarks mirror:
    speakers with at least 8 bona fide clips; each is enrolled on the first half of their bona fide
    clips (at most 10), and tested on the rest and on up to 40 spoofed clips targeting them (seed 0).
Voice, LibriSpeech dev-clean (CC BY 4.0), real only: every other speaker (sorted by id) with two
    chapters of at least 12 and 1 utterances; enrolled on 10 utterances of their largest chapter,
    then scored on up to 10 more of it (same session) and up to 20 of their second chapter (another
    session): the false-alarm rate when the recording session changes.

Decisions use the encoders' published population thresholds (SFace 0.363, ECAPA 0.25); AUCs and
rates at 5% false alarms are also reported. 95% CIs bootstrap people.

    python scripts/benchmark_person_anomaly.py faces --sdfvd SDFVD/ --out results/person_anomaly/sdfvd.json
    python scripts/benchmark_person_anomaly.py voice --asvspoof-parquet test-00000-of-00009.parquet \\
        --asvspoof-protocol ASVspoof2019.LA.cm.eval.trl.txt --librispeech LibriSpeech/dev-clean \\
        --out results/person_anomaly/asvspoof.json
"""

import argparse
import glob
import io
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from forensix.identity.anomaly import THRESHOLDS, FaceEncoder, PersonAnomaly, VoiceEncoder


def boot_ci(y, s, groups, n=2000, seed=0):
    rng = np.random.default_rng(seed); G = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in G}; out = []
    for _ in range(n):
        p = np.concatenate([idx[g] for g in rng.choice(G, len(G))])
        if len(set(y[p])) == 2:
            out.append(roc_auc_score(y[p], s[p]))
    return [round(float(v), 3) for v in np.percentile(out, [2.5, 97.5])]


def summarize(y, sim, groups, thr):
    """Scores are similarities: lower = more anomalous."""
    y, sim, groups = map(np.asarray, (y, sim, groups))
    s = -sim
    t5 = np.quantile(sim[y == 0], 0.05)
    return {"n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()), "people": int(len(np.unique(groups))),
            "auc": round(float(roc_auc_score(y, s)), 3), "auc_ci95": boot_ci(y, s, groups),
            "false_alarms_at_threshold": round(float(np.mean(sim[y == 0] < thr)), 3),
            "fakes_caught_at_threshold": round(float(np.mean(sim[y == 1] < thr)), 3),
            "fakes_caught_at_5pct_false_alarms": round(float(np.mean(sim[y == 1] < t5)), 3)}


_ENC = None


def _embed_voice_file(path):
    """ECAPA segment embeddings of one file (in a worker process; one thread each)."""
    global _ENC
    import torch

    from forensix.audio.io import load_audio
    if _ENC is None:
        torch.set_num_threads(1); _ENC = VoiceEncoder()
    return _ENC.embed_media(load_audio(str(path), 16000))[0]


def cached(path, compute):
    if path and Path(path).exists():
        return np.load(path, allow_pickle=True).item()
    v = compute()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True); np.save(path, v, allow_pickle=True)
    return v


# ------------------------------------------------------------------ faces
def faces(args):
    import cv2

    def compute():
        enc, out = FaceEncoder(), {}
        for kind, pre in (("real", "v"), ("fake", "vs")):
            for f in sorted(glob.glob(f"{args.sdfvd}/videos_{kind}/*.mp4")):
                pid = int(Path(f).stem[len(pre):]); cap = cv2.VideoCapture(f); embs = []
                while True:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    e = enc.embed_frame(fr[:, :, ::-1].copy())
                    if e is not None:
                        embs.append(e)
                out[(kind, pid)] = np.stack(embs) if embs else None
                print(kind, pid, len(embs), flush=True)
        return out
    E = cached(args.cache and f"{args.cache}/sdfvd_sface.npy", compute)
    thr = THRESHOLDS["face"]
    fy, fs, fg, cy, cs, cg = [], [], [], [], [], []
    pids = sorted(p for k, p in E if k == "real" and E[("real", p)] is not None and E.get(("fake", p)) is not None
                  and len(E[("real", p)]) >= 20)
    for pid in pids:
        R, F = E[("real", pid)], E[("fake", pid)]; n = len(R)
        m = PersonAnomaly.from_embeddings(str(pid), "face", R[: n // 2])
        sr, sf = R[int(0.6 * n):] @ m.centroid, F @ m.centroid
        fy += [0] * len(sr) + [1] * len(sf); fs += list(sr) + list(sf); fg += [pid] * (len(sr) + len(sf))
        cy += [0, 1]; cs += [float(np.median(sr)), float(np.median(sf))]; cg += [pid, pid]
    res = {"dataset": "SDFVD", "protocol": "enroll on the first half of each real clip's frames; test on its last 40% "
           "and on the face swap made from it", "threshold_similarity": thr,
           "frames": summarize(fy, fs, fg, thr), "clips": summarize(cy, cs, cg, thr)}
    return res


# ------------------------------------------------------------------ voice
def voice(args):
    import pyarrow.parquet as pq
    import soundfile as sf

    def compute_asv():
        enc = VoiceEncoder()
        attack = {l.split()[1]: l.split()[3] for l in open(args.asvspoof_protocol)}
        t = pq.read_table(args.asvspoof_parquet).to_pandas()
        t["spk"] = [json.loads(n)["speaker_id"] for n in t.notes]
        rng, keep = random.Random(0), []
        for _, g in t.groupby("spk"):
            r = g[g.label == 0]
            if len(r) < 8:
                continue
            f = g[g.label == 1]; keep += list(r.index) + rng.sample(list(f.index), min(40, len(f)))
        out = {}
        for n, i in enumerate(keep):
            row = t.loc[i]; w, sr = sf.read(io.BytesIO(row.audio["bytes"]), dtype="float32")
            e, _, _ = enc.embed_media(w)
            uid = row.path.replace(".flac", "")
            out[uid] = {"spk": row.spk, "label": int(row.label), "attack": attack.get(uid, "-"), "emb": e}
            if n % 200 == 0:
                print("asvspoof", n, len(keep), flush=True)
        return out

    def ls_plan():
        """{speaker: (enroll, same-session, other-session) file lists}, following the protocol above."""
        plan = {}
        for spk in sorted(Path(args.librispeech).iterdir(), key=lambda p: p.name)[1::2]:
            ch = sorted((sorted(c.glob("*.flac")) for c in spk.iterdir() if c.is_dir()), key=len, reverse=True)
            if len(ch) >= 2 and len(ch[0]) >= 12:
                plan[spk.name] = (ch[0][:10], ch[0][10:20], ch[1][:20])
        return plan

    def compute_ls():
        from concurrent.futures import ProcessPoolExecutor
        files = [f for parts in ls_plan().values() for group in parts for f in group]
        out = {}
        with ProcessPoolExecutor(args.workers) as ex:
            for n, (f, e) in enumerate(zip(files, ex.map(_embed_voice_file, files, chunksize=4))):
                out[f.stem] = {"spk": f.parts[-3], "session": f.parts[-2], "emb": e}
                if n % 100 == 0:
                    print("librispeech", n, len(files), flush=True)
        return out

    thr = THRESHOLDS["voice"]
    asv = cached(args.cache and f"{args.cache}/asvspoof_ecapa.npy", compute_asv)
    by = {}
    for k, v in sorted(asv.items()):
        by.setdefault(v["spk"], []).append(v)
    y, s, g, att = [], [], [], []
    for spk, items in sorted(by.items()):
        real = [v for v in items if v["label"] == 0]; fake = [v for v in items if v["label"] == 1]
        if not fake:
            continue
        ne = min(10, (len(real) + 1) // 2)
        m = PersonAnomaly.from_embeddings(spk, "voice", np.concatenate([v["emb"] for v in real[:ne]]))
        for v, lab in [(v, 0) for v in real[ne:]] + [(v, 1) for v in fake]:
            y.append(lab); s.append(m.score_embeddings(v["emb"])["similarity"]); g.append(spk); att.append(v["attack"])
    y, s, att = np.array(y), np.array(s), np.array(att)
    t5 = np.quantile(s[y == 0], 0.05)
    res = {"dataset": "ASVspoof 2019 LA eval (one shard)", "threshold_similarity": thr, "asvspoof": summarize(y, s, g, thr),
           "asvspoof_by_attack": {a: {"n": int((att == a).sum()), "caught_at_threshold": round(float(np.mean(s[att == a] < thr)), 3),
                                      "caught_at_5pct_false_alarms": round(float(np.mean(s[att == a] < t5)), 3)}
                                  for a in sorted(set(att) - {"-"})},
           "asvspoof_threshold_curve": {f"{c:.2f}": {"false_alarms": round(float(np.mean(s[y == 0] < c)), 3),
                                                     "fakes_caught": round(float(np.mean(s[y == 1] < c)), 3)}
                                        for c in (0.25, 0.3, 0.4, 0.5, 0.6)}}
    if args.librispeech:
        ls = cached(args.cache and f"{args.cache}/librispeech_ecapa_plan.npy", compute_ls)
        same, cross, n_spk = [], [], 0
        for spk, (enroll, a, b) in ls_plan().items():
            n_spk += 1
            m = PersonAnomaly.from_embeddings(spk, "voice", np.concatenate([ls[f.stem]["emb"] for f in enroll]))
            same += [m.score_embeddings(ls[f.stem]["emb"])["similarity"] for f in a]
            cross += [m.score_embeddings(ls[f.stem]["emb"])["similarity"] for f in b]
        same, cross = np.array(same), np.array(cross)
        res["librispeech_sessions"] = {"speakers": n_spk, "n_same_session": len(same), "n_other_session": len(cross),
                                       **{f"{c:.2f}": {"false_alarms_same_session": round(float(np.mean(same < c)), 3),
                                                       "false_alarms_other_session": round(float(np.mean(cross < c)), 3)}
                                          for c in (0.25, 0.3, 0.4, 0.5, 0.6)}}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["faces", "voice"])
    ap.add_argument("--sdfvd"); ap.add_argument("--asvspoof-parquet"); ap.add_argument("--asvspoof-protocol")
    ap.add_argument("--librispeech"); ap.add_argument("--cache", help="folder to cache embeddings in")
    ap.add_argument("--workers", type=int, default=3, help="processes for the LibriSpeech embeddings")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = faces(args) if args.task == "faces" else voice(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
