// ForensiX Studio page controller. All work happens in this tab; see engine.js.
import * as E from "./engine.js";

const $ = (id) => document.getElementById(id);
const state = {
  kind: "face", device: "auto", encoders: {}, encoderDevice: null,
  real: { embeddings: [], groups: [], files: 0 }, model: null, tests: [], busy: false,
};

// ------------------------------------------------------------------ small UI helpers
function toast(msg, error = false) {
  const t = document.createElement("div"); t.className = "toast" + (error ? " error" : ""); t.textContent = msg;
  $("toasts").appendChild(t); setTimeout(() => t.remove(), error ? 9000 : 4500);
}
function segmented(id, onChange) {
  $(id).addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b || state.busy) return;
    pressed(id, b.dataset.v); onChange(b.dataset.v);
  });
}
function pressed(id, v) { $(id).querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x.dataset.v === v))); }
function dropZone(zoneId, inputId, folderId, onFiles) {
  const zone = $(zoneId), input = $(inputId);
  zone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  input.addEventListener("change", () => { onFiles([...input.files]); input.value = ""; });
  ["dragenter", "dragover"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => zone.addEventListener(ev, () => zone.classList.remove("over")));
  zone.addEventListener("drop", async (e) => { e.preventDefault(); onFiles(await filesFromDrop(e.dataTransfer)); });
  $(folderId).addEventListener("click", () => {
    const f = document.createElement("input"); f.type = "file"; f.multiple = true; f.webkitdirectory = true;
    f.onchange = () => onFiles([...f.files]); f.click();
  });
}
async function filesFromDrop(dt) {   // walks dropped folders, keeping each file's relative path
  const out = [], walk = async (entry, path) => {
    if (entry.isFile) {
      const f = await new Promise((res, rej) => entry.file(res, rej));
      Object.defineProperty(f, "relPath", { value: path + f.name }); out.push(f);
    } else if (entry.isDirectory) {
      const r = entry.createReader(); let batch;
      do { batch = await new Promise((res, rej) => r.readEntries(res, rej)); for (const c of batch) await walk(c, path + entry.name + "/"); } while (batch.length);
    }
  };
  const entries = [...dt.items].map((i) => i.webkitGetAsEntry && i.webkitGetAsEntry()).filter(Boolean);
  if (!entries.length) return [...dt.files];
  for (const en of entries) await walk(en, "");
  return out;
}
const pathOf = (f) => f.relPath || f.webkitRelativePath || f.name;
function labelOf(f) {
  const parts = pathOf(f).toLowerCase().split(/[/\\]/);
  for (const p of parts.slice(0, -1).reverse()) {
    if (/^(real|genuine|bona.?fide|authentic)$/.test(p)) return "real";
    if (/^(fake|spoof|deepfake|synthetic|generated)$/.test(p)) return "fake";
  }
  const n = parts[parts.length - 1];
  if (/(^|[^a-z])(fake|spoof|deepfake)([^a-z]|$)/.test(n)) return "fake";
  if (/(^|[^a-z])(real|genuine|bonafide)([^a-z]|$)/.test(n)) return "real";
  return null;
}
function usable(files) {
  return files.filter((f) => { const k = E.mediaKind(f); return state.kind === "face" ? k === "image" || k === "video" : k === "audio" || k === "video"; });
}
const fmt = (v, d = 2) => (v == null || Number.isNaN(v) ? "–" : v.toFixed(d));
const pct = (v) => (v == null || Number.isNaN(v) ? "–" : `${(100 * v).toFixed(1)}%`);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function tiles(el, list) {
  el.innerHTML = list.map(([label, value, note]) => `<div class="tile"><div class="label">${label}</div><div class="value">${value}</div>${note ? `<div class="note">${note}</div>` : ""}</div>`).join("");
  el.hidden = false;
}
function setBusy(b) {
  state.busy = b;
  $("train").disabled = b || state.real.embeddings.length < 2;
  document.querySelectorAll("#kind button, #device button").forEach((x) => { x.disabled = b; });
}

// ------------------------------------------------------------------ encoders
async function encoder(statusEl) {
  const device = state.device === "auto" ? await E.bestDevice() : state.device;
  if (state.encoderDevice !== device) { state.encoders = {}; state.encoderDevice = device; }
  if (!state.encoders[state.kind]) {
    const size = state.kind === "face" ? "about 39 MB" : "about 83 MB";
    statusEl.textContent = `Downloading the ${state.kind} models (${size}, once; then cached)…`;
    const progress = (what, p) => { statusEl.textContent = `Downloading the ${what}… ${Math.round(100 * p)}%`; };
    state.encoders[state.kind] = state.kind === "face" ? await E.FaceEncoder.create(device, progress) : await E.VoiceEncoder.create(device, progress);
  }
  return state.encoders[state.kind];
}
function deviceLabel() { return state.encoderDevice === "webgpu" ? "GPU (WebGPU)" : "CPU (WebAssembly)"; }

function thumb(aligned) {
  const c = document.createElement("canvas"); c.width = 112; c.height = 112; c.getContext("2d").putImageData(aligned, 0, 0); return c;
}

// ------------------------------------------------------------------ step 2: real media
async function addReal(files) {
  files = usable(files);
  if (!files.length) { toast(state.kind === "face" ? "Add photos or videos." : "Add audio files or videos with sound.", true); return; }
  setBusy(true);
  try {
    const enc = await encoder($("real-status"));
    let skipped = 0;
    for (const [i, f] of files.entries()) {
      $("real-status").textContent = `Reading ${i + 1}/${files.length}: ${f.name}`;
      try {
        const r = await enc.embedFile(f);
        if (!r.embeddings.length) { skipped++; continue; }
        const g = state.real.files++;
        r.embeddings.forEach((e) => { state.real.embeddings.push(e); state.real.groups.push(g); });
        if (state.kind === "face") r.thumbs.slice(0, 8).forEach((t) => { if ($("real-thumbs").childElementCount < 120) $("real-thumbs").appendChild(thumb(t)); });
        else { const d = document.createElement("div"); d.className = "clip"; d.textContent = `${r.seconds.toFixed(1)} s`; d.title = f.name; $("real-thumbs").appendChild(d); }
      } catch (e) { skipped++; console.warn(f.name, e); toast(`${f.name}: ${e.message || e}`, true); }
    }
    const unit = state.kind === "face" ? "face samples" : "speech segments";
    $("real-status").textContent = `${state.real.embeddings.length} ${unit} from ${state.real.files} file${state.real.files === 1 ? "" : "s"}` +
      (skipped ? ` · ${skipped} skipped (no ${state.kind === "face" ? "face" : "usable speech"})` : "");
    $("clear-real").disabled = !state.real.embeddings.length;
    $("step-real").classList.toggle("done", state.real.embeddings.length >= 2);
  } catch (e) { toast(String(e.message || e), true); } finally { setBusy(false); }
}

// ------------------------------------------------------------------ step 3: train
async function train() {
  if (state.real.embeddings.length < 2) { toast("Add real media of the person first.", true); return; }
  setBusy(true);
  try {
    const t0 = performance.now();
    const m = E.PersonModel.fit(state.kind, $("person").value.trim() || "person", state.real.embeddings, state.real.groups);
    state.model = m;
    const secs = (performance.now() - t0) / 1000;
    $("train-status").textContent = `Done in ${secs < 1 ? "under a second" : secs.toFixed(1) + " s"}.`;
    showModel();
    if (m.nFiles < 3) toast("Tip: real media from at least 3 different files (sessions) gives a more reliable profile.");
    if (state.tests.length) rescoreTests();
  } catch (e) { toast(`Training failed: ${e.message || e}`, true); } finally { setBusy(false); }
}
function showModel() {
  const m = state.model, q = m.calibration.oof_similarity_quantiles;
  tiles($("train-tiles"), [
    ["Samples", m.nEnroll, `${m.nFiles} file${m.nFiles === 1 ? "" : "s"}`],
    ["Their own media", `${fmt(q[1])}–${fmt(q[19])}`, "similarity, 5th–95th percentile (held out)"],
    ["Threshold", fmt(m.thresholdSimilarity, 3), `${m.kind === "face" ? "SFace" : "ECAPA"} population threshold`],
    ["Encoder ran on", state.encoderDevice === "webgpu" ? "GPU" : state.encoderDevice ? "CPU" : "–", state.encoderDevice === "webgpu" ? "WebGPU, on this device" : state.encoderDevice ? "WebAssembly, on this device" : "load media first"],
  ]);
  $("save").disabled = false; $("step-train").classList.add("done");
}

function save() {
  const j = state.model.toJSON();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(j)], { type: "application/json" }));
  a.download = `${(j.person || "person").replace(/[^\w.-]+/g, "_")}.${j.kind}.forensix.json`; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
async function load(file) {
  try {
    const m = E.PersonModel.fromJSON(JSON.parse(await file.text()));
    if (m.kind !== state.kind) { state.kind = m.kind; pressed("kind", m.kind); kindChanged(); }
    state.model = m; $("person").value = m.person || "";
    $("train-status").textContent = `Loaded ${file.name}.`; showModel();
    if (state.tests.length) rescoreTests();
  } catch (e) { toast(`Couldn't load the model: ${e.message || e}`, true); }
}

// ------------------------------------------------------------------ step 4: test
async function addTests(files) {
  if (!state.model) { toast("Train or load a model first.", true); return; }
  files = usable(files);
  if (!files.length) { toast("No files of the right type.", true); return; }
  setBusy(true);
  try {
    const enc = await encoder($("test-status"));
    for (const [i, f] of files.entries()) {
      $("test-status").textContent = `Scoring ${i + 1}/${files.length}: ${f.name}`;
      try {
        const r = await enc.embedFile(f);
        const note = state.kind === "face" ? (r.embeddings.length ? `${r.embeddings.length}/${r.n} frames with a face` : "no face found")
          : (r.embeddings.length ? `${r.seconds.toFixed(1)} s, ${r.embeddings.length} segment${r.embeddings.length > 1 ? "s" : ""}` : "no usable speech");
        state.tests.push({ name: pathOf(f), label: labelOf(f), embeddings: r.embeddings, note });
      } catch (e) { state.tests.push({ name: pathOf(f), label: labelOf(f), embeddings: [], note: `couldn't read: ${e.message || e}` }); }
    }
    rescoreTests();
    $("test-status").textContent = `${state.tests.length} files scored on ${deviceLabel()}.`;
    $("clear-test").disabled = false;
  } finally { setBusy(false); }
}

function rescoreTests() {
  for (const t of state.tests) {
    const s = t.embeddings.length ? state.model.score(t.embeddings) : null;
    t.sim = s ? s.similarity : null; t.pp = s ? s.personPercentile : null;
  }
  $("thr").value = state.model.thresholdSimilarity; renderTests();
}

function renderTests() {
  const thr = +$("thr").value; $("thr-val").textContent = fmt(thr, 3);
  $("test-results").hidden = false;
  const scored = state.tests.filter((t) => t.sim != null);
  const real = scored.filter((t) => t.label === "real").map((t) => t.sim), fake = scored.filter((t) => t.label === "fake").map((t) => t.sim);
  const list = [["Files", state.tests.length, `${state.tests.length - scored.length} not scored`]];
  const files = (n, what) => `${n} ${what} file${n === 1 ? "" : "s"}`;
  if (real.length) list.push(["Real flagged (false alarms)", pct(real.filter((v) => v < thr).length / real.length), files(real.length, "real")]);
  if (fake.length) list.push(["Fakes caught", pct(fake.filter((v) => v < thr).length / fake.length), files(fake.length, "fake")]);
  if (real.length && fake.length) {
    list.push(["AUC", fmt(E.auc(real.map((v) => -v), fake.map((v) => -v)), 3), "0.5 = chance, 1 = perfect"]);
    if (real.length >= 20) { const t5 = E.quantile(real, 0.05); list.push(["Fakes caught at 5% false alarms", pct(fake.filter((v) => v < t5).length / fake.length), "threshold set on these real files"]); }
  }
  tiles($("test-tiles"), list);
  histogram($("hist"), scored, thr);
  $("test-table").querySelector("tbody").innerHTML = state.tests.map((t) => {
    const v = t.sim == null ? "–" : t.sim < thr ? '<span class="verdict-fake">Not like them</span>' : '<span class="verdict-real">Like them</span>';
    const pp = t.pp == null ? "" : t.pp === 0 ? "less like them than any of their own media" : `more like them than ${Math.round(100 * t.pp)}% of their own media`;
    return `<tr><td>${esc(t.name)}</td><td>${t.label || '<span class="muted">–</span>'}</td><td class="num">${fmt(t.sim, 3)}</td><td>${v}</td><td class="muted">${esc([t.note, pp].filter(Boolean).join(" · "))}</td></tr>`;
  }).join("");
  $("step-test").classList.toggle("done", scored.length > 0);
}

function histogram(el, scored, thr) {
  if (!scored.length) { el.innerHTML = ""; return; }
  const W = 760, H = 150, pad = { l: 34, r: 10, t: 22, b: 26 }, lo = -0.2, hi = 1, span = hi - lo, nb = 36;
  const bin = (v) => Math.max(0, Math.min(nb - 1, Math.floor((v - lo) / span * nb)));
  const groups = { real: Array(nb).fill(0), fake: Array(nb).fill(0), other: Array(nb).fill(0) };
  for (const t of scored) groups[t.label || "other"][bin(t.sim)]++;
  const max = Math.max(1, ...Object.values(groups).flat()), bw = (W - pad.l - pad.r) / nb;
  const x = (v) => pad.l + (Math.max(lo, Math.min(hi, v)) - lo) / span * (W - pad.l - pad.r), y = (n) => H - pad.b - n / max * (H - pad.t - pad.b);
  const colors = { real: "var(--series-1)", fake: "var(--series-2)", other: "var(--muted)" };
  let bars = "";
  for (const [k, g] of Object.entries(groups)) g.forEach((n, i) => {
    if (n) bars += `<rect class="mark" x="${pad.l + i * bw + 1}" y="${y(n)}" width="${Math.max(1, bw - 2)}" height="${H - pad.b - y(n)}" fill="${colors[k]}" fill-opacity="0.75"><title>${k === "other" ? "unlabelled" : k}: ${n}</title></rect>`;
  });
  const present = Object.entries(groups).filter(([, g]) => g.some(Boolean)).map(([k]) => k);
  el.innerHTML = `<div class="legend">${present.map((k) => `<span><i class="swatch" style="background:${colors[k]}"></i>${k === "other" ? "unlabelled" : k}</span>`).join("")}<span><i class="line-key" style="border-color:var(--ink)"></i>threshold</span></div>
  <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Histogram of similarity to the person">
    <line class="baseline" x1="${pad.l}" x2="${W - pad.r}" y1="${H - pad.b}" y2="${H - pad.b}"/>${bars}
    <line x1="${x(thr)}" x2="${x(thr)}" y1="${pad.t - 6}" y2="${H - pad.b}" stroke="var(--ink)" stroke-width="2"/>
    <text class="axis-label" x="${pad.l}" y="${H - 8}">${lo}</text><text class="axis-label" x="${W - pad.r}" y="${H - 8}" text-anchor="end">${hi} · similarity to the person</text>
    <text class="axis-label" x="${x(thr) + 4}" y="${pad.t - 10}">threshold</text>
  </svg>`;
}

function exportCsv() {
  const thr = +$("thr").value;
  const rows = [["file", "label", "similarity", "flagged", "person_percentile"], ...state.tests.map((t) => [t.name, t.label || "", t.sim ?? "", t.sim == null ? "" : t.sim < thr, t.pp ?? ""])];
  const csv = rows.map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(",")).join("\n");
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" })); a.download = "forensix-studio-scores.csv"; a.click();
}

// ------------------------------------------------------------------ wiring
function resetReal() { state.real = { embeddings: [], groups: [], files: 0 }; $("real-thumbs").innerHTML = ""; $("real-status").textContent = ""; $("clear-real").disabled = true; $("train").disabled = true; }
function clearTests() { state.tests = []; $("test-results").hidden = true; $("test-status").textContent = ""; $("clear-test").disabled = true; }
function kindChanged() {
  const face = state.kind === "face";
  $("kind-note").textContent = face
    ? "Learns their face from photos and video frames (2 per second, up to 60 per video). Use 20 or more samples from several files, in different lighting and angles."
    : "Learns their voice from recordings or the sound of videos. Use a few minutes of speech from more than one recording session.";
  $("real-hint").textContent = face ? "Photos (JPG, PNG, WebP) or videos (MP4, WebM, MOV)" : "Audio (WAV, MP3, FLAC, M4A, OGG) or videos with sound";
  resetReal(); state.model = null; $("save").disabled = true; $("train-tiles").hidden = true; $("train-status").textContent = ""; clearTests();
}

segmented("kind", (v) => { state.kind = v; kindChanged(); });
segmented("device", (v) => { state.device = v; });
dropZone("drop-real", "files-real", "folder-real", addReal);
dropZone("drop-test", "files-test", "folder-test", addTests);
$("clear-real").addEventListener("click", resetReal);
$("clear-test").addEventListener("click", clearTests);
$("train").addEventListener("click", train);
$("save").addEventListener("click", save);
$("load-file").addEventListener("change", (e) => { if (e.target.files[0]) load(e.target.files[0]); e.target.value = ""; });
$("thr").addEventListener("input", renderTests);
$("thr-reset").addEventListener("click", () => { $("thr").value = state.model.thresholdSimilarity; renderTests(); });
$("export-csv").addEventListener("click", exportCsv);
if (location.protocol === "file:") toast("Open this page over http(s), for example from the ForensiX server at /ui/studio/ or the docs site.", true);
if (!location.pathname.includes("/ui/")) $("console-link").hidden = true;
kindChanged();

// test hook: lets automated tests drive the page with in-memory files
window.forensixStudio = { state, addReal, addTests, train, setKind: (k) => { pressed("kind", k); state.kind = k; kindChanged(); }, setDevice: (d) => { state.device = d; } };
