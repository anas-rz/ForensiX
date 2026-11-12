"""Cache per-frame face features for SDFVD (all frames) and FairFace (base pretraining)."""
import os
DATA = os.environ.get('PA_DATA', 'data/')   # downloaded datasets (see README.md)
WORK = os.environ.get('PA_WORK', 'work/')   # cached features and results
import io, sys, glob, numpy as np, cv2, pyarrow.parquet as pq
from PIL import Image
from forensix.hub import fetch
from forensix.identity.face import SFACE, YUNET
from forensix.specialized.features import _extract_frequency_artifacts
rec = cv2.FaceRecognizerSF.create(str(fetch(SFACE)), ""); det = cv2.FaceDetectorYN.create(str(fetch(YUNET)), "", (320, 320), 0.8, 0.3, 5000)
def face(rgb):
    bgr = np.ascontiguousarray(rgb[:, :, ::-1]); det.setInputSize((bgr.shape[1], bgr.shape[0])); _, f = det.detect(bgr)
    if f is None or not len(f): return None
    f = f[int(np.argmax(f[:, 2] * f[:, 3]))]; al = rec.alignCrop(bgr, f[:-1])
    e = rec.feature(al).ravel(); e = e / np.linalg.norm(e)
    crop = al[:, :, ::-1]
    hand = np.concatenate([e, _extract_frequency_artifacts(crop)]).astype(np.float32)  # == extract_image_features
    return dict(sface=e.astype(np.float32), hand=hand, crop=cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA))
which = sys.argv[1]; out = {}
if which == 'sdfvd':
    root = DATA + 'SDFVD/'
    for kind, pre in [('real', 'v'), ('fake', 'vs')]:
        for f in sorted(glob.glob(root + f'videos_{kind}/*.mp4')):
            pid = int(f.split('/')[-1][len(pre):-4]); cap = cv2.VideoCapture(f); frames = []
            while True:
                ok, fr = cap.read()
                if not ok: break
                r = face(fr[:, :, ::-1].copy())
                if r: frames.append(r)
            out[(kind, pid)] = {k: np.stack([x[k] for x in frames]) for k in frames[0]} if frames else None
            print(kind, pid, len(frames), flush=True)
else:
    t = pq.read_table(DATA + 'fairface_val.parquet', columns=['image']).to_pandas()
    crops = []
    for b in t.image[:4000]:
        r = face(np.array(Image.open(io.BytesIO(b['bytes'])).convert('RGB').resize((224, 224))))
        if r: crops.append(r['crop'])
        if len(crops) >= 3000: break
    out['crops'] = np.stack(crops); print('fairface crops', len(crops))
np.save(f'{WORK}face_{which}.npy', out, allow_pickle=True)
