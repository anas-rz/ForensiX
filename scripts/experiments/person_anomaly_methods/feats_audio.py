"""Cache per-clip audio features: hand128 (person-detector), ECAPA (id-voice), log-mel (for autoencoders)."""
import os
DATA = os.environ.get('PA_DATA', 'data/')   # downloaded datasets (see README.md)
WORK = os.environ.get('PA_WORK', 'work/')   # cached features and results
import io, json, sys, glob, numpy as np, soundfile as sf, pyarrow.parquet as pq, random
import forensix
from forensix.specialized.features import extract_audio_features
from forensix.audio.io import load_audio

def mel_fb(sr=16000, n_fft=400, n_mels=64, fmin=20, fmax=7600):
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700); mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)); bins = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        l, c, r = pts[i:i + 3]
        fb[i] = np.clip(np.minimum((bins - l) / (c - l), (r - bins) / (r - c)), 0, None)
    return fb
FB = mel_fb()
def logmel(w):
    n_fft, hop = 400, 160
    w = np.pad(w, (n_fft // 2, n_fft // 2))
    fr = np.lib.stride_tricks.sliding_window_view(w, n_fft)[::hop] * np.hanning(n_fft)
    return np.log(FB @ (np.abs(np.fft.rfft(fr, axis=1)) ** 2).T + 1e-6).astype(np.float16)  # (64, T)

voice = forensix.load('id-voice')
def feats(w):
    e, _ = voice.embed_media(w)
    return dict(hand=extract_audio_features(w), ecapa=e.mean(0).astype(np.float32), mel=logmel(w))

out = {}
which = sys.argv[1]
if which == 'asvspoof':
    d = DATA + 'asvspoof_hf/'
    attack = {l.split()[1]: l.split()[3] for l in open(d + 'protocols/ASVspoof2019.LA.cm.eval.trl.txt')}
    t = pq.read_table(d + 'data/test-00000-of-00009.parquet').to_pandas()
    t['spk'] = [json.loads(n)['speaker_id'] for n in t.notes]
    rng = random.Random(0); keep = []
    for spk, g in t.groupby('spk'):
        r = g[g.label == 0]
        if len(r) < 8: continue
        f = g[g.label == 1]; keep += list(r.index) + rng.sample(list(f.index), min(40, len(f)))
    for n, i in enumerate(keep):
        row = t.loc[i]; w, sr = sf.read(io.BytesIO(row.audio['bytes']), dtype='float32'); assert sr == 16000
        uid = row.path.replace('.flac', '')
        out[uid] = dict(spk=row.spk, label=int(row.label), attack=attack.get(uid, '-'), session='vctk', **feats(w))
        if n % 200 == 0: print(n, len(keep), flush=True)
else:
    files = sorted(glob.glob(DATA + 'LibriSpeech/dev-clean/*/*/*.flac'))
    spks = sorted({'ls' + f.split('/')[-3] for f in files}); test_spk = set(spks[1::2])   # same split as evaluate.py
    for n, f in enumerate(files):
        p = f.split('/'); w = load_audio(f, 16000); spk = 'ls' + p[-3]
        fe = feats(w) if spk in test_spk else dict(mel=logmel(w))   # base-model speakers only need log-mel
        out[p[-1][:-5]] = dict(spk=spk, label=0, attack='-', session=p[-2], **fe)
        if n % 300 == 0: print(n, len(files), flush=True)
np.save(f'{WORK}audio_{which}.npy', out, allow_pickle=True)
print('done', len(out))
