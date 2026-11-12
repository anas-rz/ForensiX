"""Per-person, real-only anomaly detection: which small method works? (method fixed before results; see README.md)

Every method is fitted on one person's real enrollment media only. A shared wrapper turns raw scores
into z-scores against out-of-fold enrollment scores, so methods are compared on equal terms.
"""
import os
DATA = os.environ.get('PA_DATA', 'data/')   # downloaded datasets (see README.md)
WORK = os.environ.get('PA_WORK', 'work/')   # cached features and results
import sys, json, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
torch.set_num_threads(1)
import multiprocessing as mp
BASE_STEPS, SCRATCH_STEPS, FT_STEPS = 1500, 200, 100
W = WORK
SEED = 0

# ---------------------------------------------------------------- tiny autoencoders
class AE(nn.Module):
    def __init__(self, ch, chans, bottleneck=32):
        super().__init__()
        enc, dec, c = [], [], ch
        for o in chans:
            enc += [nn.Conv2d(c, o, 3, 2, 1), nn.GELU()]; c = o
        for o in list(reversed(chans[:-1])) + [ch]:
            dec += [nn.ConvTranspose2d(c, o, 4, 2, 1), nn.GELU()]; c = o
        dec[-1] = nn.Identity()
        self.enc, self.dec = nn.Sequential(*enc), nn.Sequential(*dec)
        self.chans = chans
    def forward(self, x):
        h = self.enc(x); s = h.shape
        z = self.fc2(F.gelu(self.fc1(h.flatten(1))))
        return self.dec(z.view(s))
    def build(self, x):  # lazily size the bottleneck
        with torch.no_grad(): s = self.enc(x[:1]).shape
        n = int(np.prod(s[1:])); self.fc1, self.fc2 = nn.Linear(n, 32), nn.Linear(32, n); return self

def audio_ae(): return AE(1, [16, 32, 32]).build(torch.zeros(1, 1, 64, 32))
def face_ae(): return AE(3, [16, 32, 64, 64]).build(torch.zeros(1, 3, 64, 64))

def train_ae(model, X, steps, lr):
    g = torch.Generator().manual_seed(SEED); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    X = torch.as_tensor(X); model.train()
    for _ in range(steps):
        b = X[torch.randint(len(X), (min(64, len(X)),), generator=g)]
        loss = F.mse_loss(model(b), b); opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()

@torch.no_grad()
def recon_err(model, X):
    X = torch.as_tensor(X); return torch.cat([((model(X[i:i + 512]) - X[i:i + 512]) ** 2).flatten(1).mean(1) for i in range(0, len(X), 512)]).numpy()

# ---------------------------------------------------------------- audio patches
MEL_MU = MEL_SD = None
def patches(mel, hop=16):
    m = (mel.astype(np.float32) - MEL_MU[:, None]) / MEL_SD[:, None]
    if m.shape[1] < 32: m = np.pad(m, ((0, 0), (0, 32 - m.shape[1])), mode='edge')
    starts = range(0, m.shape[1] - 32 + 1, hop)
    P = np.stack([m[:, s:s + 32] for s in starts])[:, None]
    energy = np.array([mel[:, s:s + 32].astype(np.float32).max(0).mean() for s in starts]) if mel.shape[1] >= 32 else np.zeros(len(P))
    return P[energy >= energy.max() - 3.0]   # skip near-silent patches (log-energy 3 nats ~ 13 dB below loudest)

def clip_score(errs):  # mean of the worst 20% of patches
    e = np.sort(errs)[::-1]; return float(e[:max(1, len(e) // 5)].mean())

# ---------------------------------------------------------------- methods: fit(items) -> scorer(items) -> raw scores
def centroid_method(key):
    def fit(tr):
        c = np.mean([x[key] / np.linalg.norm(x[key]) for x in tr], 0); c /= np.linalg.norm(c)
        return lambda te: np.array([1 - x[key] @ c / np.linalg.norm(x[key]) for x in te])
    return fit

def knn_method(key, k=3):
    def fit(tr):
        R = np.stack([x[key] / np.linalg.norm(x[key]) for x in tr]); kk = min(k, len(R))
        def sc(te):
            Q = np.stack([x[key] / np.linalg.norm(x[key]) for x in te]); D = np.linalg.norm(Q[:, None] - R[None], axis=2)
            return np.sort(D, 1)[:, :kk].mean(1)
        return sc
    return fit

def ae_method(kind, base=None, steps=300, lr=1e-3):
    def fit(tr):
        torch.manual_seed(SEED)
        m = (audio_ae() if kind == 'audio' else face_ae()) if base is None else __import__('copy').deepcopy(base)
        X = np.concatenate([x['P'] for x in tr]) if kind == 'audio' else np.stack([x['img'] for x in tr])
        train_ae(m, X, steps, lr)
        if kind == 'audio': return lambda te: np.array([clip_score(recon_err(m, x['P'])) for x in te])
        return lambda te: recon_err(m, np.stack([x['img'] for x in te]))
    return fit

def frozen_method(kind, base):
    def fit(tr):
        if kind == 'audio': return lambda te: np.array([clip_score(recon_err(base, x['P'])) for x in te])
        return lambda te: recon_err(base, np.stack([x['img'] for x in te]))
    return fit

def person_detector_method(modality):
    """The current framework's person-detector, trained on the cached features (same as its extractor)."""
    import forensix.specialized.train as T
    def fit(tr):
        orig = T.extract_features; T.extract_features = lambda m, mod: m
        try: det = T.train_person_model('p', [x['hand'] for x in tr], modality=modality)
        finally: T.extract_features = orig
        def sc(te):
            if not te: return np.array([])
            with torch.no_grad(): return det.model.score(torch.as_tensor(np.stack([x['hand'] for x in te])))[0].numpy()
        return sc
    return fit

# ---------------------------------------------------------------- calibration wrapper
def calibrated(fit, enroll, tests, folds):
    """z-scores of each test list against out-of-fold enrollment scores; folds = list of index arrays."""
    oof = np.concatenate([fit([enroll[i] for i in range(len(enroll)) if i not in set(f)])([enroll[i] for i in f]) for f in folds])
    final = fit(enroll); mu, sd = oof.mean(), oof.std() + 1e-8
    return [(final(t) - mu) / sd if len(t) else np.array([]) for t in tests], oof

def kfold(n, k):
    k = min(k, n); return [np.arange(n)[i::k] for i in range(k)]
def blocks(n, k):
    return [a for a in np.array_split(np.arange(n), k)]

# ---------------------------------------------------------------- metrics
def boot_auc(y, s, grp, n=1000):
    rng = np.random.default_rng(0); G = np.unique(grp); idx = {g: np.where(grp == g)[0] for g in G}; out = []
    for _ in range(n):
        p = np.concatenate([idx[g] for g in rng.choice(G, len(G))])
        if len(set(y[p])) == 2: out.append(roc_auc_score(y[p], s[p]))
    return np.percentile(out, [2.5, 97.5])

def summarize(name, y, s, grp, extra=''):
    auc = roc_auc_score(y, s); lo, hi = boot_auc(y, s, grp)
    thr5 = np.quantile(s[y == 0], 0.95)
    r = dict(auc=round(auc, 3), ci=[round(lo, 3), round(hi, 3)], tpr_at_5fa=round(float(np.mean(s[y == 1] > thr5)), 3),
             fa_z233=round(float(np.mean(s[y == 0] > 2.33)), 3), tpr_z233=round(float(np.mean(s[y == 1] > 2.33)), 3))
    print(f'  {name:16s} AUC {r["auc"]:.3f} [{lo:.3f},{hi:.3f}]  fakes caught @5%FA {r["tpr_at_5fa"]:.3f}  '
          f'z>2.33: FA {r["fa_z233"]:.3f} TPR {r["tpr_z233"]:.3f} {extra}', flush=True)
    return r

JOB = None
import os, pickle
CACHE = None
def run_job(x):
    p = f'{CACHE}/{x}.pkl' if CACHE else None
    if p and os.path.exists(p): return pickle.load(open(p, 'rb'))
    r = JOB(x)
    if p: pickle.dump(r, open(p, 'wb'))
    return r

# ================================================================ AUDIO
def run_audio():
    global MEL_MU, MEL_SD
    asv = np.load(W + 'audio_asvspoof.npy', allow_pickle=True).item()
    ls = np.load(W + 'audio_libri.npy', allow_pickle=True).item()
    ls_spk = sorted({v['spk'] for v in ls.values()}); base_spk = set(ls_spk[0::2]); test_spk = ls_spk[1::2]
    base_mels = [v['mel'] for v in ls.values() if v['spk'] in base_spk]
    allm = np.concatenate([m.astype(np.float32) for m in base_mels], 1); MEL_MU, MEL_SD = allm.mean(1), allm.std(1) + 1e-3
    for d in (asv, ls):
        for v in d.values(): v['P'] = patches(v['mel'])
    print('base: LibriSpeech speakers', len(base_spk), 'clips', len(base_mels), flush=True)
    torch.manual_seed(SEED)
    base = train_ae(audio_ae(), np.concatenate([patches(m) for m in base_mels]), BASE_STEPS, 1e-3)
    print('audio AE params', sum(p.numel() for p in base.parameters()), flush=True)
    methods = {
        'person-detector': person_detector_method('audio'),
        'hand-knn': knn_method('hand'),
        'ecapa-centroid': centroid_method('ecapa'),
        'ae-base(frozen)': frozen_method('audio', base),
        'ae-scratch': ae_method('audio', None, SCRATCH_STEPS, 1e-3),
        'ae-base-ft': ae_method('audio', base, FT_STEPS, 3e-4),
    }
    # ---- ASVspoof: per speaker, same session
    res = {m: dict(z=[], y=[], g=[], att=[]) for m in methods}
    raw_pd = dict(s=[], y=[], g=[])
    by = {}
    for k, v in asv.items(): by.setdefault(v['spk'], []).append((k, v))
    global JOB
    def job_asv(spk):
        items = by[spk]
        real = [v for _, v in sorted(items, key=lambda t: t[0]) if v['label'] == 0]; fake = [v for _, v in sorted(items, key=lambda t: t[0]) if v['label'] == 1]
        ne = min(10, len(real) // 2 + len(real) % 2); enroll, treal = real[:ne], real[ne:]
        out = {}
        for m, fit in methods.items():
            (zr, zf), _ = calibrated(fit, enroll, [treal, fake], kfold(ne, 3)); out[m] = (zr, zf)
        sc = methods['person-detector'](enroll); out['raw'] = (sc(treal), sc(fake))
        return spk, out, [f['attack'] for f in fake]
    JOB = job_asv
    global CACHE; CACHE = W + 'cache_asv'; os.makedirs(CACHE, exist_ok=True)
    with mp.get_context('fork').Pool(7) as pool:
        todo_asv = [k for k in sorted(by) if any(v['label'] == 1 for _, v in by[k])]
        print('speakers with fakes', len(todo_asv), 'of', len(by), flush=True)
        for n, (spk, o, atts) in enumerate(pool.imap_unordered(run_job, todo_asv)):
            for m in methods:
                zr, zf = o[m]; r = res[m]; r['z'] += list(zr) + list(zf); r['y'] += [0] * len(zr) + [1] * len(zf); r['g'] += [spk] * (len(zr) + len(zf))
                r['att'] += ['-'] * len(zr) + atts
            a, b = o['raw']; raw_pd['s'] += list(a) + list(b); raw_pd['y'] += [0] * len(a) + [1] * len(b); raw_pd['g'] += [spk] * (len(a) + len(b))
            print(n + 1, '/', len(by), spk, flush=True)
    out = {'asvspoof': {}}
    print(f'\nASVspoof 2019 LA eval, per speaker: {len(by)} speakers, {sum(1 for y in res["hand-knn"]["y"] if y == 0)} held-out real, {sum(res["hand-knn"]["y"])} fake')
    y = np.array(raw_pd['y']); s = np.array(raw_pd['s'])
    out['asvspoof']['person-detector(raw fake_prob)'] = summarize('person-det raw', y, s, np.array(raw_pd['g']), f'| FA at its 0.5 threshold {np.mean(s[y==0]>0.5):.3f}, TPR {np.mean(s[y==1]>0.5):.3f}')
    Z = {}
    for m, r in res.items():
        Z[m] = np.array(r['z']); out['asvspoof'][m] = summarize(m, np.array(r['y']), Z[m], np.array(r['g']))
    y, g, att = np.array(res['hand-knn']['y']), np.array(res['hand-knn']['g']), np.array(res['hand-knn']['att'])
    combo = Z['ecapa-centroid'] + Z['ae-base-ft']; out['asvspoof']['ecapa+ae-base-ft'] = summarize('ecapa+ae-ft', y, combo, g)
    Z['ecapa+ae-base-ft'] = combo
    # per attack, fakes caught at 5% FA
    print('  per attack, fakes caught at 5% FA:')
    out['asvspoof_by_attack'] = {}
    for m in ['ecapa-centroid', 'ae-base-ft', 'ecapa+ae-base-ft', 'hand-knn']:
        thr = np.quantile(Z[m][y == 0], 0.95)
        row = {a: round(float(np.mean(Z[m][(att == a)] > thr)), 2) for a in sorted(set(att) - {'-'})}
        out['asvspoof_by_attack'][m] = row; print(f'    {m:18s}', row)
    out['thr5'] = {m: float(np.quantile(Z[m][y == 0], 0.95)) for m in Z}
    # ---- LibriSpeech: cross-session false alarms (speakers not used for the base model)
    print('\nLibriSpeech dev-clean, cross-session real (held-out speakers)')
    lb = {}
    for k, v in ls.items():
        if v['spk'] in test_spk: lb.setdefault(v['spk'], {}).setdefault(v['session'], []).append((k, v))
    cs = {m: dict(same=[], cross=[]) for m in methods}
    n_spk = 0
    def job_ls(spk):
        sess = lb[spk]; s1, s2 = sorted(sess, key=lambda s: -len(sess[s]))[:2]
        a = [v for _, v in sorted(sess[s1], key=lambda t: t[0])]; b = [v for _, v in sorted(sess[s2], key=lambda t: t[0])]
        enroll, same = a[:10], a[10:]
        return {m: calibrated(fit, enroll, [same, b], kfold(10, 3))[0] for m, fit in methods.items()}
    JOB = job_ls
    CACHE = W + 'cache_ls'; os.makedirs(CACHE, exist_ok=True)
    todo = [spk for spk, sess in sorted(lb.items()) if len(sess) >= 2 and max(len(v) for v in sess.values()) >= 12]
    with mp.get_context('fork').Pool(7) as pool:
        for o in pool.imap_unordered(run_job, todo):
            n_spk += 1
            for m in methods: cs[m]['same'] += list(o[m][0]); cs[m]['cross'] += list(o[m][1])
            print('librispeech', n_spk, '/', len(todo), flush=True)
    out['librispeech'] = {}
    for m in list(methods) + ['ecapa+ae-base-ft']:
        if m == 'ecapa+ae-base-ft':
            same = np.array(cs['ecapa-centroid']['same']) + np.array(cs['ae-base-ft']['same']); cross = np.array(cs['ecapa-centroid']['cross']) + np.array(cs['ae-base-ft']['cross'])
        else: same, cross = np.array(cs[m]['same']), np.array(cs[m]['cross'])
        t = out['thr5'][m]
        r = dict(fa_same=round(float(np.mean(same > t)), 3), fa_cross=round(float(np.mean(cross > t)), 3), median_z_same=round(float(np.median(same)), 2), median_z_cross=round(float(np.median(cross)), 2))
        out['librispeech'][m] = r
        print(f'  {m:18s} false alarms at the ASVspoof 5%-FA threshold: same session {r["fa_same"]:.3f}, other session {r["fa_cross"]:.3f}  (median z {r["median_z_same"]} / {r["median_z_cross"]})')
    out['librispeech_n'] = dict(speakers=n_spk, same=len(cs['hand-knn']['same']), cross=len(cs['hand-knn']['cross']))
    print(' ', out['librispeech_n'])
    json.dump(out, open(W + 'results_audio.json', 'w'), indent=1)

# ================================================================ FACES (SDFVD)
def run_face():
    sd = np.load(W + 'face_sdfvd.npy', allow_pickle=True).item()
    ff = np.load(W + 'face_fairface.npy', allow_pickle=True).item()['crops']
    toimg = lambda c: (c.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
    torch.manual_seed(SEED)
    base = train_ae(face_ae(), toimg(ff), BASE_STEPS, 1e-3)
    print('face AE params', sum(p.numel() for p in base.parameters()), 'base images', len(ff), flush=True)
    methods = {
        'person-detector': person_detector_method('image'),
        'hand-knn': knn_method('hand'),
        'sface-centroid': centroid_method('sface'),
        'ae-base(frozen)': frozen_method('face', base),
        'ae-scratch': ae_method('face', None, SCRATCH_STEPS, 1e-3),
        'ae-base-ft': ae_method('face', base, FT_STEPS, 3e-4),
    }
    def items(d):
        im = toimg(d['crop']); return [dict(hand=d['hand'][i], sface=d['sface'][i], img=im[i]) for i in range(len(im))]
    res = {m: dict(z=[], y=[], g=[], clip=[]) for m in list(methods)}
    pids = sorted(p for k, p in sd if k == 'real' and sd[('real', p)] is not None and sd.get(('fake', p)) is not None and len(sd[('real', p)]['crop']) >= 20)
    global JOB
    def job_face(pid):
        R = items(sd[('real', pid)]); Fk = items(sd[('fake', pid)])
        n = len(R); enroll = R[: n // 2]; treal = R[int(n * 0.6):]   # 10% gap between enrollment and test frames
        return pid, {m: calibrated(fit, enroll, [treal, Fk], blocks(len(enroll), 3))[0] for m, fit in methods.items()}
    JOB = job_face
    with mp.get_context('fork').Pool(7) as pool:
        for n, (pid, o) in enumerate(pool.imap_unordered(run_job, pids)):
            for m in methods:
                zr, zf = o[m]; r = res[m]; r['z'] += list(zr) + list(zf); r['y'] += [0] * len(zr) + [1] * len(zf); r['g'] += [pid] * (len(zr) + len(zf))
                r['clip'] += [(pid, 0, float(np.mean(zr))), (pid, 1, float(np.mean(zf)))]
            print(n + 1, '/', len(pids), flush=True)
    print(f'\nSDFVD, per person (enroll on the first half of the real clip): {len(pids)} people, '
          f'{sum(1 for y in res["hand-knn"]["y"] if y == 0)} real test frames, {sum(res["hand-knn"]["y"])} fake frames')
    out = {'image_frames': {}, 'video_clips': {}}
    Z = {m: np.array(r['z']) for m, r in res.items()}; y = np.array(res['hand-knn']['y']); g = np.array(res['hand-knn']['g'])
    Z['sface+ae-base-ft'] = Z['sface-centroid'] + Z['ae-base-ft']
    print(' image (per frame):')
    for m in Z: out['image_frames'][m] = summarize(m, y, Z[m], g)
    print(' video (per clip, mean frame z):')
    for m in list(res) + ['sface+ae-base-ft']:
        if m == 'sface+ae-base-ft':
            c = [(a[0], a[1], a[2] + b[2]) for a, b in zip(res['sface-centroid']['clip'], res['ae-base-ft']['clip'])]
        else: c = res[m]['clip']
        out['video_clips'][m] = summarize(m, np.array([x[1] for x in c]), np.array([x[2] for x in c]), np.array([x[0] for x in c]))
    json.dump(out, open(W + 'results_face.json', 'w'), indent=1)

if __name__ == '__main__':
    {'audio': run_audio, 'face': run_face}[sys.argv[1]]()
