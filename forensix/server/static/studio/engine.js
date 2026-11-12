// ForensiX Studio engine: a per-person deepfake detector built in the browser from real media only.
//
// Nothing leaves the device. The pipeline mirrors forensix/identity/anomaly.py (person-anomaly), and
// models are saved in the same JSON format, so a person enrolled here loads in Python and back.
//
//   faces : YuNet detection (frame scaled to fit 640x640) -> OpenCV's 5-point similarity alignment
//           (112x112) -> SFace embedding (128-d); 2 frames per second, up to 60 per video
//   voice : 16 kHz mono -> SpeechBrain's 80-band log filterbank, sentence mean normalisation ->
//           ECAPA-TDNN embedding (192-d); whole clip up to 6 s, else 3 s segments every 1.5 s
//   model : the person's mean embedding; a clip's similarity is the median cosine of its frames or
//           segments; flagged below the encoder's population threshold (SFace 0.363, ECAPA 0.25).
//           Leave-one-file-out similarities of their own media say how typical a clip is for them.
//
// Why no network is trained from scratch: in tests, small autoencoders trained on one person's
// media scored near chance and made the identity score worse (person_anomaly_benchmark.md).

/* global ort */

export const FORMAT = "forensix-person-anomaly/1";
export const ENCODERS = { face: "sface-2021dec", voice: "ecapa-voxceleb-0f99f2d" };
export const THRESHOLDS = { face: 0.363, voice: 0.25 };
export const CDN = {
  ort: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.js",
  ortWasm: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/",
};
// Face models: the same pinned files forensix.identity.face downloads (OpenCV Zoo; Apache-2.0 / MIT).
// Voice model: ECAPA exported by forensix.identity.anomaly.export_ecapa_onnx, served next to this page.
export const WEIGHTS = {
  yunet: "https://huggingface.co/opencv/face_detection_yunet/resolve/3cc26e7f1014a5ee5d74a42acee58bafc9d0a310/face_detection_yunet_2023mar.onnx",
  sface: "https://huggingface.co/opencv/face_recognition_sface/resolve/3d7082438a6e4551e840c9b2bb60b71e8da4b524/face_recognition_sface_2021dec.onnx",
  ecapa: new URL("models/ecapa.onnx", import.meta.url).href,
};
const FPS = 2, MAX_FRAMES = 60, SR = 16000;

// ------------------------------------------------------------------ loading
const loaded = {};
export function loadScript(src) {
  if (!loaded[src]) {
    loaded[src] = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src; s.crossOrigin = "anonymous";
      s.onload = resolve; s.onerror = () => reject(new Error(`Couldn't load ${src}`));
      document.head.appendChild(s);
    });
  }
  return loaded[src];
}

async function fetchCached(url, onProgress) {
  let cache = null;
  try { cache = await caches.open("forensix-studio-v1"); } catch (e) { /* no Cache API */ }
  if (cache) { const hit = await cache.match(url); if (hit) return new Uint8Array(await hit.arrayBuffer()); }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Download failed (${res.status}): ${url}`);
  const total = +res.headers.get("content-length") || 0, reader = res.body.getReader(), chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read(); if (done) break;
    chunks.push(value); got += value.length; if (onProgress && total) onProgress(got / total);
  }
  const buf = new Uint8Array(got); let o = 0; for (const c of chunks) { buf.set(c, o); o += c.length; }
  if (cache) { try { await cache.put(url, new Response(buf)); } catch (e) { /* quota */ } }
  return buf;
}

/** "webgpu" when the browser has a usable GPU adapter, else "wasm". */
export async function bestDevice() {
  try { return navigator.gpu && (await navigator.gpu.requestAdapter()) ? "webgpu" : "wasm"; } catch (e) { return "wasm"; }
}

async function session(bytes, device) {
  await loadScript(CDN.ort);
  ort.env.wasm.wasmPaths = CDN.ortWasm;
  const providers = device === "webgpu" ? ["webgpu", "wasm"] : ["wasm"];
  try { return await ort.InferenceSession.create(bytes, { executionProviders: providers }); }
  catch (e) { if (device === "webgpu") return ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] }); throw e; }
}

// ------------------------------------------------------------------ media
const IMAGE_RE = /\.(jpe?g|png|bmp|webp|gif|tiff?)$/i, VIDEO_RE = /\.(mp4|mov|m4v|webm|mkv|avi)$/i, AUDIO_RE = /\.(wav|flac|mp3|m4a|aac|ogg|opus)$/i;
export function mediaKind(file) {
  const t = file.type || "";
  if (t.startsWith("image/") || IMAGE_RE.test(file.name)) return "image";
  if (t.startsWith("video/") || VIDEO_RE.test(file.name)) return "video";
  if (t.startsWith("audio/") || AUDIO_RE.test(file.name)) return "audio";
  return null;
}

/** Frames ({canvas, t}) of an image, or of a video at ``fps`` (at most ``maxFrames``). */
export async function framesFromFile(file, fps = FPS, maxFrames = MAX_FRAMES) {
  if (mediaKind(file) === "image") {
    const bmp = await createImageBitmap(file); const c = toCanvas(bmp, bmp.width, bmp.height); bmp.close?.();
    return [{ canvas: c, t: 0 }];
  }
  const url = URL.createObjectURL(file);
  try {
    const v = document.createElement("video"); v.muted = true; v.preload = "auto"; v.playsInline = true; v.src = url;
    await new Promise((res, rej) => {
      v.onloadedmetadata = res;
      v.onerror = () => rej(new Error(`your browser can't decode ${file.name} (${v.error?.message || "unsupported codec"}). ` +
        "Browsers play H.264, VP9 and AV1; convert older formats with: ffmpeg -i in.mp4 -c:v libx264 -crf 18 out.mp4"));
    });
    const frames = [], dur = isFinite(v.duration) ? v.duration : 0;
    for (let t = 0; t < dur && frames.length < maxFrames; t += 1 / fps) {
      await new Promise((res) => { v.onseeked = res; v.currentTime = Math.min(t, Math.max(0, dur - 1e-3)); });
      frames.push({ canvas: toCanvas(v, v.videoWidth, v.videoHeight), t: Math.round(t * 100) / 100 });
    }
    return frames;
  } finally { URL.revokeObjectURL(url); }
}
function toCanvas(src, w, h) { const c = new OffscreenCanvas(w, h); c.getContext("2d").drawImage(src, 0, 0, w, h); return c; }

/** 16 kHz mono samples of an audio file or a video's sound. */
export async function audioFromFile(file) {
  const buf = await file.arrayBuffer();
  // decode straight to 16 kHz (one resampling, as ffmpeg does for forensix.audio.io.load_audio), then average channels
  const Ctx = window.AudioContext || window.webkitAudioContext;
  let ctx; try { ctx = new Ctx({ sampleRate: SR }); } catch (e) { ctx = new Ctx(); }
  let decoded; try { decoded = await ctx.decodeAudioData(buf); } finally { ctx.close(); }
  if (decoded.sampleRate === SR) {
    const out = new Float32Array(decoded.length);
    for (let c = 0; c < decoded.numberOfChannels; c++) { const d = decoded.getChannelData(c); for (let i = 0; i < out.length; i++) out[i] += d[i] / decoded.numberOfChannels; }
    return out;
  }
  const n = Math.max(1, Math.round(decoded.duration * SR)), off = new OfflineAudioContext(1, n, SR);
  const src = off.createBufferSource(); src.buffer = decoded; src.connect(off.destination); src.start();
  return (await off.startRendering()).getChannelData(0).slice();
}

// ------------------------------------------------------------------ faces
const TEMPLATE = [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]];

export class FaceEncoder {
  static async create(device = "wasm", onProgress = () => {}) {
    const e = new FaceEncoder();
    e.det = await session(await fetchCached(WEIGHTS.yunet, (p) => onProgress("face detector", p)), "wasm");
    e.rec = await session(await fetchCached(WEIGHTS.sface, (p) => onProgress("face encoder", p)), device);
    return e;
  }

  /** Faces, largest first: {box, kps, score}; OpenCV FaceDetectorYN decoding on the frame scaled into 640x640. */
  async detect(canvas, scoreThreshold = 0.8, nmsThreshold = 0.3) {
    const S = 640, scale = Math.min(S / canvas.width, S / canvas.height);
    const c = new OffscreenCanvas(S, S), g = c.getContext("2d");
    g.fillStyle = "#000"; g.fillRect(0, 0, S, S); g.imageSmoothingQuality = "high";
    g.drawImage(canvas, 0, 0, Math.max(1, Math.round(canvas.width * scale)), Math.max(1, Math.round(canvas.height * scale)));
    const px = g.getImageData(0, 0, S, S).data, input = new Float32Array(3 * S * S);
    for (let i = 0; i < S * S; i++) { input[i] = px[4 * i + 2]; input[S * S + i] = px[4 * i + 1]; input[2 * S * S + i] = px[4 * i]; }   // BGR, CHW
    const out = await this.det.run({ input: new ort.Tensor("float32", input, [1, 3, S, S]) });
    const faces = [];
    for (const s of [8, 16, 32]) {
      const cols = S / s, cls = out[`cls_${s}`].data, obj = out[`obj_${s}`].data, bb = out[`bbox_${s}`].data, kp = out[`kps_${s}`].data;
      for (let i = 0; i < cols * cols; i++) {
        const r = Math.floor(i / cols), cc = i % cols;
        const score = Math.sqrt(Math.min(Math.max(cls[i], 0), 1) * Math.min(Math.max(obj[i], 0), 1));
        if (score < scoreThreshold) continue;
        const cx = (cc + bb[4 * i]) * s, cy = (r + bb[4 * i + 1]) * s, w = Math.exp(bb[4 * i + 2]) * s, h = Math.exp(bb[4 * i + 3]) * s;
        const kps = [0, 1, 2, 3, 4].map((n) => [(kp[10 * i + 2 * n] + cc) * s / scale, (kp[10 * i + 2 * n + 1] + r) * s / scale]);
        faces.push({ box: [(cx - w / 2) / scale, (cy - h / 2) / scale, w / scale, h / scale], kps, score });
      }
    }
    return nms(faces, nmsThreshold).sort((a, b) => b.box[2] * b.box[3] - a.box[2] * a.box[3]);
  }

  /** 112x112 aligned face (ImageData), as cv::FaceRecognizerSF::alignCrop. */
  align(canvas, face) {
    const [a, b, tx, ty] = similarity(face.kps, TEMPLATE);
    const c = new OffscreenCanvas(112, 112), g = c.getContext("2d");
    g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
    g.setTransform(a, b, -b, a, tx, ty); g.drawImage(canvas, 0, 0);
    return g.getImageData(0, 0, 112, 112);
  }

  /** Unit SFace embedding of an aligned face (RGB, 0..255, as OpenCV feeds it). */
  async embedAligned(aligned) {
    const n = 112 * 112, d = aligned.data, x = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) { x[i] = d[4 * i]; x[n + i] = d[4 * i + 1]; x[2 * n + i] = d[4 * i + 2]; }
    const out = await this.rec.run({ data: new ort.Tensor("float32", x, [1, 3, 112, 112]) });
    return unit(Float32Array.from(out.fc1.data));
  }

  /** {embeddings, times, n, thumbs} for a file's frames with a face. */
  async embedFile(file) {
    const frames = await framesFromFile(file), embeddings = [], times = [], thumbs = [];
    for (const f of frames) {
      const faces = await this.detect(f.canvas);
      if (!faces.length) continue;
      const al = this.align(f.canvas, faces[0]);
      embeddings.push(await this.embedAligned(al)); times.push(f.t); thumbs.push(al);
    }
    return { embeddings, times, n: frames.length, thumbs };
  }
}

function nms(faces, thr) {
  const sorted = [...faces].sort((a, b) => b.score - a.score), keep = [];
  const iou = (p, q) => {
    const x1 = Math.max(p[0], q[0]), y1 = Math.max(p[1], q[1]), x2 = Math.min(p[0] + p[2], q[0] + q[2]), y2 = Math.min(p[1] + p[3], q[1] + q[3]);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1); return inter / (p[2] * p[3] + q[2] * q[3] - inter + 1e-9);
  };
  for (const f of sorted) if (keep.every((k) => iou(k.box, f.box) <= thr)) keep.push(f);
  return keep;
}

/** Least-squares similarity transform src -> dst (Umeyama, no reflection): [a, b, tx, ty], x' = a x - b y + tx, y' = b x + a y + ty. */
export function similarity(src, dst) {
  const n = src.length, ms = [0, 0], md = [0, 0];
  for (let i = 0; i < n; i++) { ms[0] += src[i][0] / n; ms[1] += src[i][1] / n; md[0] += dst[i][0] / n; md[1] += dst[i][1] / n; }
  let sxx = 0, sxy = 0, vs = 0;
  for (let i = 0; i < n; i++) {
    const x = src[i][0] - ms[0], y = src[i][1] - ms[1], u = dst[i][0] - md[0], v = dst[i][1] - md[1];
    sxx += x * u + y * v; sxy += x * v - y * u; vs += x * x + y * y;
  }
  const a = sxx / vs, b = sxy / vs;
  return [a, b, md[0] - (a * ms[0] - b * ms[1]), md[1] - (b * ms[0] + a * ms[1])];
}

// ------------------------------------------------------------------ voice
const NFFT = 400, HOP = 160, NMELS = 80;
let FB = null, DFT = null, HAMMING = null;
function fbankInit() {   // speechbrain Fbank(n_mels=80): 0..8000 Hz triangular filters, as forensix.audiovisual.avid.ecapa.Fbank
  if (FB) return;
  const nb = NFFT / 2 + 1, toMel = (hz) => 2595 * Math.log10(1 + hz / 700);
  const m0 = toMel(0), m1 = toMel(SR / 2), hz = Array.from({ length: NMELS + 2 }, (_, i) => 700 * (10 ** ((m0 + (m1 - m0) * i / (NMELS + 1)) / 2595) - 1));
  FB = Array.from({ length: NMELS }, (_, m) => {
    const fc = hz[m + 1], band = hz[m + 1] - hz[m], row = new Float64Array(nb);
    for (let k = 0; k < nb; k++) { const f = (SR / 2) * k / (nb - 1), slope = (f - fc) / band; row[k] = Math.max(0, Math.min(slope + 1, 1 - slope)); }
    return row;
  });
  HAMMING = Float64Array.from({ length: NFFT }, (_, i) => 0.54 - 0.46 * Math.cos(2 * Math.PI * i / NFFT));   // torch.hamming_window (periodic)
  DFT = { cos: new Float64Array(nb * NFFT), sin: new Float64Array(nb * NFFT) };
  for (let k = 0; k < nb; k++) for (let n = 0; n < NFFT; n++) { DFT.cos[k * NFFT + n] = Math.cos(2 * Math.PI * k * n / NFFT); DFT.sin[k * NFFT + n] = Math.sin(2 * Math.PI * k * n / NFFT); }
}

/** ECAPA input: (frames x 80) log filterbank in dB, floored 80 dB below the max, mean-normalised over time. */
export function fbank(wave) {
  fbankInit();
  const pad = NFFT / 2, x = new Float64Array(wave.length + 2 * pad); x.set(wave, pad);   // centred, zero padding
  const T = Math.floor(wave.length / HOP) + 1, nb = NFFT / 2 + 1, out = new Float32Array(T * NMELS), fr = new Float64Array(NFFT), pw = new Float64Array(nb);
  let mx = -Infinity;
  for (let t = 0; t < T; t++) {
    for (let n = 0; n < NFFT; n++) fr[n] = x[t * HOP + n] * HAMMING[n];
    for (let k = 0; k < nb; k++) { let re = 0, im = 0; const o = k * NFFT; for (let n = 0; n < NFFT; n++) { re += fr[n] * DFT.cos[o + n]; im -= fr[n] * DFT.sin[o + n]; } pw[k] = re * re + im * im; }
    for (let m = 0; m < NMELS; m++) { let s = 0; const row = FB[m]; for (let k = 0; k < nb; k++) s += row[k] * pw[k]; const db = 10 * Math.log10(Math.max(s, 1e-10)); out[t * NMELS + m] = db; if (db > mx) mx = db; }
  }
  const floor = mx - 80, mean = new Float64Array(NMELS);
  for (let i = 0; i < out.length; i++) { if (out[i] < floor) out[i] = floor; mean[i % NMELS] += out[i] / T; }
  for (let i = 0; i < out.length; i++) out[i] -= mean[i % NMELS];
  return { data: out, frames: T };
}

/** The whole clip up to 6 s, else non-silent 3 s segments every 1.5 s (as forensix.identity.voice.segments). */
export function segments(wave) {
  if (wave.length <= 6 * SR) return [{ wave, start: 0 }];
  const seg = 3 * SR, hop = 1.5 * SR, segs = [];
  for (let s = 0; s + seg <= wave.length; s += hop) {
    let e = 0; for (let i = s; i < s + seg; i++) e += wave[i] * wave[i];
    segs.push({ wave: wave.subarray(s, s + seg), start: Math.round(s / SR * 100) / 100, level: 10 * Math.log10(e / seg + 1e-12) });
  }
  const top = Math.max(...segs.map((x) => x.level));
  return segs.filter((x) => x.level > top - 30);
}

export class VoiceEncoder {
  static async create(device = "wasm", onProgress = () => {}) {
    const e = new VoiceEncoder();
    e.net = await session(await fetchCached(WEIGHTS.ecapa, (p) => onProgress("voice encoder", p)), device);
    return e;
  }
  async embedWave(wave) {
    const f = fbank(wave);
    const out = await this.net.run({ feats: new ort.Tensor("float32", f.data, [1, f.frames, NMELS]) });
    return unit(Float32Array.from(out.embedding.data));
  }
  async embedFile(file) {
    const wave = await audioFromFile(file);
    if (wave.length < 0.5 * SR) return { embeddings: [], times: [], n: 0, seconds: wave.length / SR };
    const segs = segments(wave), embeddings = [];
    for (const s of segs) embeddings.push(await this.embedWave(s.wave));
    return { embeddings, times: segs.map((s) => s.start), n: segs.length, seconds: wave.length / SR };
  }
}

// ------------------------------------------------------------------ the person model (as forensix.identity.anomaly.PersonAnomaly)
export class PersonModel {
  constructor(kind, person) { this.kind = kind; this.person = person; this.thresholdSimilarity = THRESHOLDS[kind]; }

  /** Fit from unit embeddings; groups = source file per embedding. */
  static fit(kind, person, embeddings, groups, folds = 3) {
    if (embeddings.length < 2) throw new Error(`Need at least 2 usable samples (${kind === "face" ? "frames with a face" : "speech segments"}); got ${embeddings.length}`);
    const m = new PersonModel(kind, person), n = embeddings.length;
    let g = groups ? [...groups] : embeddings.map((_, i) => i);
    m.nFiles = new Set(g).size;
    if (m.nFiles < folds) g = embeddings.map((_, i) => Math.floor(i * folds / n));
    const oof = new Array(n);
    for (const f of new Set(g)) {
      const c = unit(mean(embeddings.filter((_, i) => g[i] !== f)));
      embeddings.forEach((e, i) => { if (g[i] === f) oof[i] = dot(e, c); });
    }
    m.centroid = unit(mean(embeddings));
    m.calibration = { oof_similarity_quantiles: Array.from({ length: 21 }, (_, i) => round(quantile(oof, i / 20), 5)), oof_n: n, folds: new Set(g).size };
    m.nEnroll = n;
    return m;
  }

  personPercentile(sim) {
    const q = this.calibration.oof_similarity_quantiles, k = q.length - 1;
    if (sim <= q[0]) return 0; if (sim >= q[k]) return 1;
    for (let i = 0; i < k; i++) if (sim <= q[i + 1]) return (i + (q[i + 1] > q[i] ? (sim - q[i]) / (q[i + 1] - q[i]) : 1)) / k;
    return 1;
  }

  /** {similarity (median), itemSimilarity, personPercentile, flagged}. */
  score(embeddings) {
    const sims = embeddings.map((e) => dot(e, this.centroid)), s = quantile(sims, 0.5);
    return { similarity: s, itemSimilarity: sims, personPercentile: this.personPercentile(s), flagged: s < this.thresholdSimilarity };
  }

  toJSON() {
    return { format: FORMAT, kind: this.kind, person: this.person, encoder: ENCODERS[this.kind], created: new Date().toISOString().slice(0, 10),
      trained_in: "browser", centroid: Array.from(this.centroid, (v) => round(v, 7)), threshold_similarity: this.thresholdSimilarity,
      calibration: this.calibration, n_enroll: this.nEnroll, n_files: this.nFiles };
  }

  static fromJSON(j) {
    if (j.format !== FORMAT) throw new Error(`Not a ForensiX person model (format ${j.format})`);
    if (j.encoder !== ENCODERS[j.kind]) throw new Error(`Made with encoder ${j.encoder}; this studio uses ${ENCODERS[j.kind]}`);
    const m = new PersonModel(j.kind, j.person);
    Object.assign(m, { centroid: Float32Array.from(j.centroid), thresholdSimilarity: j.threshold_similarity ?? THRESHOLDS[j.kind],
      calibration: j.calibration, nEnroll: j.n_enroll, nFiles: j.n_files });
    return m;
  }
}

// ------------------------------------------------------------------ metrics and helpers
export function auc(real, fake) {   // P(fake scores above real), ties count half
  let s = 0; for (const f of fake) for (const r of real) s += f > r ? 1 : f === r ? 0.5 : 0;
  return real.length && fake.length ? s / (real.length * fake.length) : NaN;
}
export function quantile(xs, q) { const s = [...xs].sort((a, b) => a - b); const p = (s.length - 1) * q, lo = Math.floor(p); return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (p - lo); }
function mean(vs) { const out = new Float64Array(vs[0].length); for (const v of vs) for (let i = 0; i < v.length; i++) out[i] += v[i] / vs.length; return out; }
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function unit(v) { const n = Math.sqrt(dot(v, v)) || 1; return Float32Array.from(v, (x) => x / n); }
function round(v, d) { const p = 10 ** d; return Math.round(v * p) / p; }
