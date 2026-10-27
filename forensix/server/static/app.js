/* ForensiX manager: a dependency-free single-page UI over the ForensiX HTTP API.
 * Everything that comes from the server or from users goes into the DOM as text
 * (textContent / createTextNode), never as HTML. */
"use strict";

// -- DOM helpers ---------------------------------------------------------------------

function el(tag, props, ...children) {
  const node = document.createElement(tag);
  applyProps(node, props);
  appendAll(node, children);
  return node;
}

const SVG = "http://www.w3.org/2000/svg";
function sv(tag, attrs, ...children) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined) continue;
    if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  appendAll(node, children);
  return node;
}

function applyProps(node, props) {
  for (const [k, v] of Object.entries(props || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "style") Object.assign(node.style, v);
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (k in node && typeof v !== "string") node[k] = v;
    else node.setAttribute(k, v === true ? "" : v);
  }
}

function appendAll(node, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

/** node.replaceChildren that skips null/false children (the DOM method would print "null"). */
function put(node, ...children) {
  node.replaceChildren();
  appendAll(node, children);
  return node;
}

const safeUrl = (u) => (typeof u === "string" && /^https?:\/\//i.test(u) ? u : null);

// -- storage, formatting -------------------------------------------------------------

const store = {
  get(k, d = "") { try { return localStorage.getItem("fx." + k) ?? d; } catch (e) { return d; } },
  set(k, v) { try { v ? localStorage.setItem("fx." + k, v) : localStorage.removeItem("fx." + k); } catch (e) { /* private mode */ } },
};

const isNum = (x) => typeof x === "number" && Number.isFinite(x);
function num(x, digits = 3) { return isNum(x) ? (Math.abs(x) >= 1000 ? Math.round(x).toLocaleString() : +x.toFixed(digits) + "") : "—"; }
function pct(x, digits = 1) { return isNum(x) ? (x * 100).toFixed(digits) + "%" : "—"; }
function secs(x) {
  if (!isNum(x)) return "—";
  if (x < 1) return Math.round(x * 1000) + " ms";
  if (x < 120) return x.toFixed(x < 10 ? 2 : 1) + " s";
  if (x < 7200) return Math.round(x / 60) + " min";
  return (x / 3600).toFixed(1) + " h";
}
function compact(n) {
  if (!isNum(n)) return "—";
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(Math.round(n * 100) / 100);
}
function when(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return isNaN(d) ? iso : d.toLocaleString();
}

// -- API -----------------------------------------------------------------------------

function authHeaders(admin) {
  const key = admin ? (store.get("adminKey") || store.get("apiKey")) : store.get("apiKey");
  return key ? { "X-API-Key": key } : {};
}

async function api(path, { method = "GET", json, form, admin = false } = {}) {
  const headers = authHeaders(admin);
  let body;
  if (json !== undefined) { headers["content-type"] = "application/json"; body = JSON.stringify(json); }
  if (form) body = form;
  const r = await fetch(path, { method, headers, body });
  if (!r.ok) {
    let detail = r.statusText;
    try { const j = await r.json(); detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch (e) { /* not JSON */ }
    const err = new Error(`${r.status}: ${detail}`);
    err.status = r.status;
    throw err;
  }
  return r.json();
}

function uploadWithProgress(path, form, { admin = false, onProgress } = {}) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", path);
    for (const [k, v] of Object.entries(authHeaders(admin))) xhr.setRequestHeader(k, v);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch (e) { /* not JSON */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else reject(new Error(`${xhr.status}: ${data && data.detail ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : xhr.statusText}`));
    };
    xhr.onerror = () => reject(new Error("Network error"));
    xhr.send(form);
  });
}

function toast(message, kind = "") {
  const t = el("div", { class: "toast " + kind, text: message });
  document.getElementById("toasts").append(t);
  setTimeout(() => t.remove(), kind === "error" ? 8000 : 4000);
}

function guard(fn) {
  return async (...args) => {
    try { return await fn(...args); } catch (e) { toast(e.message, "error"); throw e; }
  };
}

// -- tooltip ---------------------------------------------------------------------------

const tip = () => document.getElementById("tooltip");
function showTip(evt, rows) {
  const t = tip();
  put(t, ...rows.map((r) => el("div", { class: "row", style: { gap: "8px", flexWrap: "nowrap" } },
    r.color ? el("span", { class: r.key === "line" ? "line-key" : "swatch", style: r.key === "line" ? { borderColor: r.color } : { background: r.color } }) : null,
    el("span", { class: "tt-value", text: r.value }),
    el("span", { class: "tt-label", text: r.label || "" }))));
  t.style.display = "block";
  const pad = 14, w = t.offsetWidth, h = t.offsetHeight;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
  if (y + h > window.innerHeight - 8) y = evt.clientY - h - pad;
  t.style.left = x + "px";
  t.style.top = y + "px";
}
function hideTip() { tip().style.display = "none"; }

// -- status ----------------------------------------------------------------------------

const STATUS = {
  good: { color: "var(--good)", icon: "✓" },
  warning: { color: "var(--warning)", icon: "?", cls: "warning" },
  serious: { color: "var(--serious)", icon: "!" },
  critical: { color: "var(--critical)", icon: "✕" },
};
function statusMark(kind, label) {
  const s = STATUS[kind];
  return el("span", { class: "verdict" },
    el("span", { class: "status-icon " + (s.cls || ""), style: { background: s.color }, "aria-hidden": "true", text: s.icon }),
    el("span", { text: label }));
}
const VERDICT = {
  fake: ["critical", "Fake"], real: ["good", "Real"],
  uncertain: ["warning", "Uncertain"], out_of_distribution: ["serious", "Out of distribution"],
};
function labelMark(label) {
  if (label === "fake") return statusMark("critical", "Fake");
  if (label === "real") return statusMark("good", "Real");
  return el("span", { class: "muted", text: "No decision" });
}

// -- charts ----------------------------------------------------------------------------

function niceTicks(max, count = 4, integer = false) {
  if (!(max > 0)) return [0, 1];
  const raw = max / count, mag = 10 ** Math.floor(Math.log10(raw));
  let step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  if (integer) step = Math.max(1, Math.ceil(step));
  const ticks = [];
  for (let v = 0; v <= max + step * 0.001; v += step) ticks.push(+v.toFixed(10));
  if (ticks[ticks.length - 1] < max) ticks.push(+(ticks[ticks.length - 1] + step).toFixed(10));
  return ticks;
}

// bar with a 4px rounded data end and a square baseline end
function columnPath(x, y, w, h) {
  const r = Math.min(4, w / 2, h);
  return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
}
function barPath(x, y, w, h) {
  const r = Math.min(4, h / 2, w);
  return `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h - r}Q${x + w},${y + h} ${x + w - r},${y + h}H${x}Z`;
}

function withTableView(chartNode, tableNode) {
  const wrap = el("div");
  let showTable = false;
  const btn = el("button", { class: "small view-toggle", type: "button", text: "Table view" });
  const holder = el("div", {}, chartNode);
  btn.onclick = () => {
    showTable = !showTable;
    put(holder, showTable ? tableNode : chartNode);
    btn.textContent = showTable ? "Chart view" : "Table view";
  };
  wrap.append(holder);
  return { node: wrap, toggle: btn };
}

function simpleTable(headers, rows, numeric = []) {
  return el("div", { class: "table-wrap" }, el("table", {},
    el("thead", {}, el("tr", {}, headers.map((h, i) => el("th", { class: numeric.includes(i) ? "num" : "", text: h })))),
    el("tbody", {}, rows.map((r) => el("tr", {}, r.map((c, i) => el("td", { class: numeric.includes(i) ? "num" : "" }, c)))))));
}

/** Real vs fake score distribution (two series) with the decision threshold. */
function scoreHistogram({ real, fake, threshold, scoreType }) {
  const all = real.concat(fake).filter(isNum);
  let lo = 0, hi = 1;
  if (scoreType === "raw" || all.some((s) => s < 0 || s > 1)) {
    lo = Math.min(...all); hi = Math.max(...all);
    if (hi === lo) { lo -= 0.5; hi += 0.5; }
  }
  const bins = 20, step = (hi - lo) / bins;
  const binOf = (s) => Math.min(bins - 1, Math.max(0, Math.floor((s - lo) / step)));
  const counts = { real: new Array(bins).fill(0), fake: new Array(bins).fill(0) };
  real.forEach((s) => isNum(s) && counts.real[binOf(s)]++);
  fake.forEach((s) => isNum(s) && counts.fake[binOf(s)]++);
  const W = 640, H = 230, m = { l: 36, r: 12, t: 22, b: 30 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const maxC = Math.max(1, ...counts.real, ...counts.fake);
  const yt = niceTicks(maxC, 4, true), yMax = yt[yt.length - 1];
  const xs = (v) => m.l + ((v - lo) / (hi - lo)) * pw;
  const ys = (c) => m.t + ph - (c / yMax) * ph;
  const band = pw / bins, gap = 2, bw = Math.min(24, (band - gap * 3) / 2);
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Score distribution of real and fake calibration items" });
  yt.forEach((t) => {
    svg.append(sv("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: ys(t), y2: ys(t) }));
    svg.append(sv("text", { class: "axis-label", x: m.l - 6, y: ys(t) + 4, "text-anchor": "end" }, compact(t)));
  });
  const fmtX = (v) => (Math.abs(hi - lo) >= 10 ? v.toFixed(0) : v.toFixed(2));
  for (let i = 0; i <= 4; i++) {
    const v = lo + ((hi - lo) * i) / 4;
    svg.append(sv("text", { class: "axis-label", x: xs(v), y: H - 10, "text-anchor": "middle" }, fmtX(v)));
  }
  const series = [["real", "var(--series-1)", "real"], ["fake", "var(--series-2)", "fake"]];
  for (let b = 0; b < bins; b++) {
    const x0 = m.l + b * band + (band - (2 * bw + gap)) / 2;
    series.forEach(([key, color, name], j) => {
      const c = counts[key][b];
      const x = x0 + j * (bw + gap);
      if (c > 0) svg.append(sv("path", { class: "mark", d: columnPath(x, ys(c), bw, ys(0) - ys(c)), fill: color }));
      const range = `score ${fmtX(lo + b * step)}–${fmtX(lo + (b + 1) * step)}`;
      svg.append(sv("rect", {
        x: x - gap / 2, y: m.t, width: bw + gap, height: ph, fill: "transparent",
        onpointermove: (e) => showTip(e, [{ value: `${c} ${name}`, label: range, color, key: "rect" }]),
        onpointerleave: hideTip,
      }));
    });
  }
  svg.append(sv("line", { class: "baseline", x1: m.l, x2: W - m.r, y1: ys(0), y2: ys(0) }));
  if (isNum(threshold) && threshold >= lo && threshold <= hi) {
    const tx = xs(threshold);
    svg.append(sv("line", { x1: tx, x2: tx, y1: m.t - 4, y2: ys(0), stroke: "var(--ink-2)", "stroke-width": 1.5 }));
    svg.append(sv("text", { class: "value-label", x: tx, y: m.t - 8, "text-anchor": "middle" }, `threshold ${fmtX(threshold)}`));
  }
  const legend = el("div", { class: "legend" },
    el("span", {}, el("span", { class: "swatch", style: { background: "var(--series-1)" } }), `Real (${real.length})`),
    el("span", {}, el("span", { class: "swatch", style: { background: "var(--series-2)" } }), `Fake (${fake.length})`));
  const chart = el("div", { class: "chart" }, legend, svg);
  const table = simpleTable(["Score range", "Real", "Fake"],
    counts.real.map((c, b) => [`${fmtX(lo + b * step)}–${fmtX(lo + (b + 1) * step)}`, String(c), String(counts.fake[b])]), [1, 2]);
  return withTableView(chart, table);
}

/** Horizontal bars, one series, as HTML rows (crisp text at any width).
 * items: [{label, value, display}] */
function hBars(items, { valueLabel = "" } = {}) {
  const max = Math.max(...items.map((d) => d.value).filter(isNum), 1e-9);
  const reserve = Math.max(48, ...items.map((d) => String(d.display).length * 7.2 + 12));  // px for the value label
  return el("div", { class: "hbars", role: "img", "aria-label": valueLabel }, items.map((d) => {
    const row = el("div", { class: "hbar-row", tabindex: "0" },
      el("span", { class: "hbar-label", text: d.label }),
      el("span", { class: "hbar-track" },
        el("span", { class: "hbar mark", style: { width: `calc((100% - ${reserve}px) * ${isNum(d.value) ? Math.max(0.005, d.value / max) : 0})` } }),
        el("span", { class: "hbar-value", text: d.display })));
    const tipRows = [{ value: d.display, label: `${d.label} · ${valueLabel}`, color: "var(--series-1)", key: "rect" }];
    row.addEventListener("pointermove", (e) => showTip(e, tipRows));
    row.addEventListener("pointerleave", hideTip);
    row.addEventListener("focus", () => { const b = row.getBoundingClientRect(); showTip({ clientX: b.left + b.width / 2, clientY: b.top }, tipRows); });
    row.addEventListener("blur", hideTip);
    return row;
  }));
}

/** One score on its scale, with the decision threshold. */
function meter(score, threshold, lo = 0, hi = 1) {
  const pos = (v) => Math.max(0, Math.min(100, ((v - lo) / (hi - lo)) * 100));
  return el("div", { class: "meter", role: "img", "aria-label": `score ${num(score)}${isNum(threshold) ? ", threshold " + num(threshold) : ""}` },
    el("div", { class: "fill", style: { width: pos(score) + "%" } }),
    isNum(threshold) ? el("div", { class: "tick", style: { left: `calc(${pos(threshold)}% - 1px)` } }) : null,
    isNum(threshold) ? el("div", { class: "tick-label", style: { left: pos(threshold) + "%" }, text: `threshold ${num(threshold)}` }) : null);
}

/** Per-segment scores as a line, with a crosshair tooltip. */
function sparkline(values, threshold, label = "segment") {
  const W = 640, H = 120, m = { l: 34, r: 10, t: 10, b: 22 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const finite = values.filter(isNum);
  let lo = Math.min(0, ...finite), hi = Math.max(1, ...finite);
  const xs = (i) => m.l + (values.length > 1 ? (i / (values.length - 1)) * pw : pw / 2);
  const ys = (v) => m.t + ph - ((v - lo) / (hi - lo)) * ph;
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Per-${label} scores` });
  [lo, (lo + hi) / 2, hi].forEach((t) => {
    svg.append(sv("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: ys(t), y2: ys(t) }));
    svg.append(sv("text", { class: "axis-label", x: m.l - 6, y: ys(t) + 4, "text-anchor": "end" }, num(t, 2)));
  });
  if (isNum(threshold)) {
    svg.append(sv("line", { x1: m.l, x2: W - m.r, y1: ys(threshold), y2: ys(threshold), stroke: "var(--ink-2)", "stroke-width": 1 }));
    svg.append(sv("text", { class: "axis-label", x: W - m.r, y: ys(threshold) - 4, "text-anchor": "end" }, "threshold"));
  }
  const d = values.map((v, i) => `${i ? "L" : "M"}${xs(i)},${ys(v)}`).join("");
  svg.append(sv("path", { d, fill: "none", stroke: "var(--series-1)", "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
  svg.append(sv("text", { class: "axis-label", x: m.l, y: H - 6 }, `${label} 1`));
  svg.append(sv("text", { class: "axis-label", x: W - m.r, y: H - 6, "text-anchor": "end" }, `${label} ${values.length}`));
  const cross = sv("line", { x1: 0, x2: 0, y1: m.t, y2: m.t + ph, stroke: "var(--axis)", "stroke-width": 1, visibility: "hidden" });
  const dotMark = sv("circle", { r: 4, fill: "var(--series-1)", stroke: "var(--surface)", "stroke-width": 2, visibility: "hidden" });
  svg.append(cross, dotMark);
  svg.append(sv("rect", {
    x: m.l, y: 0, width: pw, height: H, fill: "transparent",
    onpointermove: (e) => {
      const box = svg.getBoundingClientRect();
      const x = ((e.clientX - box.left) / box.width) * W;
      const i = Math.max(0, Math.min(values.length - 1, Math.round(((x - m.l) / pw) * (values.length - 1))));
      cross.setAttribute("x1", xs(i)); cross.setAttribute("x2", xs(i)); cross.setAttribute("visibility", "visible");
      dotMark.setAttribute("cx", xs(i)); dotMark.setAttribute("cy", ys(values[i])); dotMark.setAttribute("visibility", "visible");
      showTip(e, [{ value: num(values[i]), label: `${label} ${i + 1}`, color: "var(--series-1)", key: "line" }]);
    },
    onpointerleave: () => { hideTip(); cross.setAttribute("visibility", "hidden"); dotMark.setAttribute("visibility", "hidden"); },
  }));
  return el("div", { class: "chart" }, svg);
}

function tiles(list) {
  return el("div", { class: "tiles" }, list.map((t) => el("div", { class: "tile" },
    el("div", { class: "label", text: t.label }),
    el("div", { class: "value" }, t.value),
    t.note ? el("div", { class: "note", text: t.note }) : null,
    t.spark ? miniSpark(t.spark) : null)));
}

// -- router ----------------------------------------------------------------------------

let cleanups = [];
function every(ms, fn) { const id = setInterval(fn, ms); cleanups.push(() => clearInterval(id)); }
const page = () => document.getElementById("page");

const ROUTES = { overview: overviewPage, models: modelsPage, model: modelPage, test: testPage, calibration: calibrationPage, settings: settingsPage };

function route() {
  cleanups.forEach((f) => f());
  cleanups = [];
  hideTip();
  const parts = location.hash.replace(/^#\/?/, "").split("?")[0].split("/");
  const name = ROUTES[parts[0]] ? parts[0] : "overview";
  const param = parts.slice(1).map(decodeURIComponent).join("/");
  document.querySelectorAll(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === (name === "model" ? "models" : name)));
  const root = page();
  put(root, el("div", { class: "empty", text: "Loading…" }));
  Promise.resolve(ROUTES[name](root, param)).catch((e) => {
    put(root, errorState(e));
  });
}

function errorState(e) {
  return el("div", { class: "banner" },
    el("strong", { text: e.status === 401 ? "The server needs an API key. " : "Couldn't load this page. " }),
    e.status === 401 ? el("span", {}, "Add it in ", el("a", { href: "#/settings", text: "Settings" }), ".") : el("span", { text: e.message }));
}

function head(title, subtitle, ...actions) {
  return el("div", { class: "page-head" },
    el("div", {}, el("h1", { text: title }), subtitle ? el("p", { class: "sub", text: subtitle }) : null),
    el("div", { class: "row" }, actions));
}

async function refreshServerLine() {
  const line = document.getElementById("server-line");
  try {
    const s = await api("/v1/server");
    put(line, el("span", { class: "dot", style: { background: s.ready ? "var(--good)" : "var(--warning)", marginRight: "6px" } }),
      `${s.ready ? "Ready" : "Loading"} · ${s.device} · v${s.version}`);
  } catch (e) {
    put(line, el("span", { class: "dot", style: { background: "var(--critical)", marginRight: "6px" } }),
      e.status === 401 ? "API key needed" : "Server unreachable");
  }
}

// -- overview dashboard -------------------------------------------------------------------

const OUTCOMES = ["real", "fake", "uncertain", "error"];
const OUTCOME_STYLE = {
  real: { color: "var(--good)", label: "Real", status: "good" },
  fake: { color: "var(--critical)", label: "Fake", status: "critical" },
  uncertain: { color: "var(--warning)", label: "Uncertain / no decision", status: "warning" },
  error: { color: "var(--serious)", label: "Error", status: "serious" },
};
const WINDOWS = [[900, "15 min"], [3600, "1 hour"], [21600, "6 hours"], [86400, "24 hours"]];

function statusKey(kind) {
  const s = STATUS[kind];
  return el("span", { class: "status-icon small-icon " + (s.cls || ""), style: { background: s.color }, "aria-hidden": "true", text: s.icon });
}
function outcomeLegend() {
  return el("div", { class: "legend" }, OUTCOMES.map((o) => el("span", {}, statusKey(OUTCOME_STYLE[o].status), OUTCOME_STYLE[o].label)));
}
const clock = (t, withSeconds = false) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", ...(withSeconds ? { second: "2-digit" } : {}) });
const share = (a, b) => (b ? a / b : null);
const mem = (mb) => (!isNum(mb) ? "—" : mb >= 1000 ? `${(mb / 1024).toFixed(mb >= 10240 ? 0 : 1)} GB` : `${Math.round(mb)} MB`);

/** Stacked columns over time: one column per bucket, segments per outcome. */
function outcomeColumns(series, width) {
  const H = 210, m = { l: 38, r: 8, t: 10, b: 26 };
  const W = Math.max(280, width), pw = W - m.l - m.r, ph = H - m.t - m.b;
  const n = series.length, band = pw / n, bw = Math.max(2, Math.min(24, band - 2));
  const yt = niceTicks(Math.max(1, ...series.map((b) => b.n)), 4, true), yMax = yt[yt.length - 1];
  const ys = (v) => m.t + ph - (v / yMax) * ph;
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": "Requests over time by outcome" });
  yt.forEach((t) => {
    svg.append(sv("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: ys(t), y2: ys(t) }));
    svg.append(sv("text", { class: "axis-label", x: m.l - 6, y: ys(t) + 4, "text-anchor": "end" }, compact(t)));
  });
  const every = Math.max(1, Math.ceil(n / Math.max(2, Math.floor(pw / 70))));
  series.forEach((b, i) => {
    const x = m.l + i * band + (band - bw) / 2;
    if (i % every === 0) svg.append(sv("text", { class: "axis-label", x: x + bw / 2, y: H - 8, "text-anchor": "middle" }, clock(b.t)));
    const hover = sv("rect", { x: m.l + i * band, y: m.t, width: band, height: ph, fill: "var(--surface-2)", opacity: 0 });
    svg.append(hover);
    let base = ys(0);
    const present = OUTCOMES.filter((o) => b[o] > 0);
    present.forEach((o, j) => {
      const h = ys(0) - ys(b[o]);
      const top = base - h, isTop = j === present.length - 1;
      const gap = isTop ? 0 : Math.min(2, h / 2);  // 2px surface gap between stacked segments
      svg.append(isTop ? sv("path", { d: columnPath(x, top, bw, h), fill: OUTCOME_STYLE[o].color })
        : sv("rect", { x, y: top + gap, width: bw, height: Math.max(0.5, h - gap), fill: OUTCOME_STYLE[o].color }));
      base = top;
    });
    svg.append(sv("rect", {
      x: m.l + i * band, y: m.t, width: band, height: ph, fill: "transparent",
      onpointermove: (e) => { hover.setAttribute("opacity", 1); showTip(e, [
        { value: `${b.n} requests`, label: `${clock(b.t)}–${clock(b.t + (series[1] ? series[1].t - series[0].t : 60))}` },
        ...OUTCOMES.filter((o) => b[o]).map((o) => ({ value: String(b[o]), label: OUTCOME_STYLE[o].label, color: OUTCOME_STYLE[o].color, key: "rect" }))]); },
      onpointerleave: () => { hover.setAttribute("opacity", 0); hideTip(); },
    }));
  });
  // hover backgrounds sit under the bars
  svg.querySelectorAll("rect[opacity]").forEach((r) => svg.insertBefore(r, svg.querySelector(".gridline")));
  svg.append(sv("line", { class: "baseline", x1: m.l, x2: W - m.r, y1: ys(0), y2: ys(0) }));
  return el("div", { class: "chart" }, svg);
}

/** Lines over time on one axis. series: [{name, color, values}] aligned with xs (unix seconds). */
function timeLines(xs, series, width, { fmt = num, height = 180, floorZero = true } = {}) {
  const H = height, m = { l: 46, r: 12, t: 12, b: 26 };
  const W = Math.max(280, width), pw = W - m.l - m.r, ph = H - m.t - m.b;
  const all = series.flatMap((s) => s.values).filter(isNum);
  let lo = floorZero ? 0 : Math.min(...all), hi = Math.max(...all, floorZero ? 0 : -Infinity);
  if (!all.length) { lo = 0; hi = 1; }
  const yt = niceTicks(hi - lo || 1, 4).map((t) => t + lo), yMax = yt[yt.length - 1];
  const x0 = xs[0], x1 = xs[xs.length - 1] === x0 ? x0 + 1 : xs[xs.length - 1];
  const xsc = (t) => m.l + ((t - x0) / (x1 - x0)) * pw;
  const ysc = (v) => m.t + ph - ((v - lo) / (yMax - lo || 1)) * ph;
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": series.map((s) => s.name).join(", ") });
  yt.forEach((t) => {
    svg.append(sv("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: ysc(t), y2: ysc(t) }));
    svg.append(sv("text", { class: "axis-label", x: m.l - 6, y: ysc(t) + 4, "text-anchor": "end" }, fmt(t)));
  });
  const ticks = Math.max(2, Math.floor(pw / 90));
  for (let i = 0; i <= ticks; i++) {
    const t = x0 + ((x1 - x0) * i) / ticks;
    svg.append(sv("text", { class: "axis-label", x: xsc(t), y: H - 8, "text-anchor": i === 0 ? "start" : i === ticks ? "end" : "middle" }, clock(t)));
  }
  series.forEach((s) => {
    let d = "", pen = false;
    s.values.forEach((v, i) => {
      if (!isNum(v)) { pen = false; return; }
      d += `${pen ? "L" : "M"}${xsc(xs[i])},${ysc(v)}`;
      pen = true;
    });
    if (d) svg.append(sv("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    const last = s.values.map((v, i) => [v, i]).filter(([v]) => isNum(v)).pop();
    if (last) svg.append(sv("circle", { cx: xsc(xs[last[1]]), cy: ysc(last[0]), r: 4, fill: s.color, stroke: "var(--surface)", "stroke-width": 2 }));
  });
  const cross = sv("line", { y1: m.t, y2: m.t + ph, stroke: "var(--axis)", "stroke-width": 1, visibility: "hidden" });
  const dots = series.map((s) => sv("circle", { r: 4, fill: s.color, stroke: "var(--surface)", "stroke-width": 2, visibility: "hidden" }));
  svg.append(cross, ...dots);
  svg.append(sv("rect", {
    x: m.l, y: 0, width: pw, height: H, fill: "transparent",
    onpointermove: (e) => {
      const box = svg.getBoundingClientRect();
      const t = x0 + (((e.clientX - box.left) / box.width) * W - m.l) / pw * (x1 - x0);
      let i = 0;
      xs.forEach((x, k) => { if (Math.abs(x - t) < Math.abs(xs[i] - t)) i = k; });
      cross.setAttribute("x1", xsc(xs[i])); cross.setAttribute("x2", xsc(xs[i])); cross.setAttribute("visibility", "visible");
      series.forEach((s, k) => {
        const v = s.values[i];
        dots[k].setAttribute("visibility", isNum(v) ? "visible" : "hidden");
        if (isNum(v)) { dots[k].setAttribute("cx", xsc(xs[i])); dots[k].setAttribute("cy", ysc(v)); }
      });
      showTip(e, [{ value: clock(xs[i], true), label: "" },
        ...series.map((s) => ({ value: isNum(s.values[i]) ? fmt(s.values[i]) : "—", label: s.name, color: s.color, key: "line" }))]);
    },
    onpointerleave: () => { hideTip(); cross.setAttribute("visibility", "hidden"); dots.forEach((d) => d.setAttribute("visibility", "hidden")); },
  }));
  const legend = series.length > 1 ? el("div", { class: "legend" }, series.map((s) => el("span", {}, el("span", { class: "line-key", style: { borderColor: s.color } }), s.name))) : null;
  return el("div", { class: "chart" }, legend, svg);
}

/** Distribution of one model's live scores, with its threshold. */
function valueHistogram(values, threshold, width) {
  const H = 190, m = { l: 38, r: 10, t: 22, b: 26 };
  const W = Math.max(280, width), pw = W - m.l - m.r, ph = H - m.t - m.b;
  let lo = 0, hi = 1;
  if (values.some((v) => v < 0 || v > 1)) { lo = Math.min(...values); hi = Math.max(...values); if (hi === lo) { lo -= 0.5; hi += 0.5; } }
  const bins = 25, step = (hi - lo) / bins, counts = new Array(bins).fill(0);
  values.forEach((v) => counts[Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / step)))]++);
  const yt = niceTicks(Math.max(1, ...counts), 4, true), yMax = yt[yt.length - 1];
  const ys = (c) => m.t + ph - (c / yMax) * ph, xs = (v) => m.l + ((v - lo) / (hi - lo)) * pw;
  const band = pw / bins, bw = Math.min(24, band - 2);
  const fmtX = (v) => (Math.abs(hi - lo) >= 10 ? v.toFixed(0) : v.toFixed(2));
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": "Distribution of recent scores" });
  yt.forEach((t) => {
    svg.append(sv("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: ys(t), y2: ys(t) }));
    svg.append(sv("text", { class: "axis-label", x: m.l - 6, y: ys(t) + 4, "text-anchor": "end" }, compact(t)));
  });
  for (let i = 0; i <= 4; i++) svg.append(sv("text", { class: "axis-label", x: xs(lo + ((hi - lo) * i) / 4), y: H - 8, "text-anchor": "middle" }, fmtX(lo + ((hi - lo) * i) / 4)));
  counts.forEach((c, b) => {
    const x = m.l + b * band + (band - bw) / 2;
    if (c) svg.append(sv("path", { class: "mark", d: columnPath(x, ys(c), bw, ys(0) - ys(c)), fill: "var(--series-1)" }));
    svg.append(sv("rect", { x: m.l + b * band, y: m.t, width: band, height: ph, fill: "transparent",
      onpointermove: (e) => showTip(e, [{ value: `${c} requests`, label: `score ${fmtX(lo + b * step)}–${fmtX(lo + (b + 1) * step)}`, color: "var(--series-1)", key: "rect" }]),
      onpointerleave: hideTip }));
  });
  svg.append(sv("line", { class: "baseline", x1: m.l, x2: W - m.r, y1: ys(0), y2: ys(0) }));
  if (isNum(threshold) && threshold >= lo && threshold <= hi) {
    svg.append(sv("line", { x1: xs(threshold), x2: xs(threshold), y1: m.t - 4, y2: ys(0), stroke: "var(--ink-2)", "stroke-width": 1.5 }));
    svg.append(sv("text", { class: "value-label", x: xs(threshold), y: m.t - 8, "text-anchor": "middle" }, `threshold ${fmtX(threshold)}`));
  }
  return el("div", { class: "chart" }, svg);
}

/** Per-model outcome shares as 100% bars (HTML), with counts on hover. */
function outcomeShares(byModel) {
  return el("div", { class: "shares" }, Object.entries(byModel).map(([name, m]) => {
    const track = el("div", { class: "share-track" }, OUTCOMES.filter((o) => m[o] > 0).map((o) => {
      const seg = el("span", { class: "share-seg", tabindex: "0", style: { flexGrow: m[o], background: OUTCOME_STYLE[o].color } });
      const rows = [{ value: `${m[o]} (${pct(m[o] / m.n, 0)})`, label: `${OUTCOME_STYLE[o].label} · ${name}`, color: OUTCOME_STYLE[o].color, key: "rect" }];
      seg.addEventListener("pointermove", (e) => showTip(e, rows));
      seg.addEventListener("pointerleave", hideTip);
      return seg;
    }));
    return el("div", { class: "share-row" },
      el("a", { class: "share-label", href: `#/model/${encodeURIComponent(name)}`, text: name }), track,
      el("span", { class: "share-n", text: compact(m.n) }));
  }));
}

function miniSpark(values) {
  const v = values.map((x) => (isNum(x) ? x : null));
  const finite = v.filter(isNum);
  if (finite.length < 2) return null;
  const W = 120, H = 30, max = Math.max(...finite), min = Math.min(0, ...finite);
  const xs = (i) => 2 + (i / (v.length - 1)) * (W - 6), ys = (x) => H - 3 - ((x - min) / (max - min || 1)) * (H - 6);
  let d = "", pen = false;
  v.forEach((x, i) => { if (x === null) { pen = false; return; } d += `${pen ? "L" : "M"}${xs(i)},${ys(x)}`; pen = true; });
  const lastI = v.map((x, i) => [x, i]).filter(([x]) => x !== null).pop()[1];
  return sv("svg", { class: "spark", viewBox: `0 0 ${W} ${H}`, width: W, height: H, "aria-hidden": "true" },
    sv("path", { d, fill: "none", stroke: "var(--muted)", "stroke-width": 1.5, "stroke-linejoin": "round" }),
    sv("circle", { cx: xs(lastI), cy: ys(v[lastI]), r: 3, fill: "var(--accent)" }));
}

function card(title, ...extra) {
  const body = el("div", { class: "card-body" });
  const actions = el("div", { class: "row" });
  const node = el("div", { class: "card" }, el("div", { class: "card-head" }, el("h2", { text: title }), actions), ...extra, body);
  return { node, body, actions };
}

async function overviewPage(root) {
  const query = new URLSearchParams(location.hash.split("?")[1] || "");  // shareable: #/overview?window=900&model=x
  let windowS = +(query.get("window") || store.get("dash.window", "3600")) || 3600;
  let model = query.get("model") ?? store.get("dash.model", "");
  const seenModels = new Set();
  const winButtons = el("div", { class: "segmented", role: "group", "aria-label": "Time range" }, WINDOWS.map(([w, label]) => {
    const b = el("button", { type: "button", text: label, "aria-pressed": String(w === windowS) });
    b.onclick = () => { windowS = w; store.set("dash.window", String(w)); winButtons.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b))); refresh(); };
    return b;
  }));
  const modelSelect = el("select", { "aria-label": "Model" });
  modelSelect.onchange = () => { model = modelSelect.value; store.set("dash.model", model); refresh(); };
  const updated = el("span", { class: "small muted" });
  const statusStrip = el("div", { class: "status-strip" });
  const kpis = el("div");
  const traffic = card("Requests over time");
  const latency = card("Latency (computed requests)");
  const shares = card("Outcomes by model");
  const perModel = card("Requests by model");
  const media = card("Media types");
  const scores = card("Score distribution");
  const provenance = card("Provenance evidence");
  const memory = card("Memory");
  const cpu = card("CPU");
  const gpu = card("GPU memory");
  const loaded = el("div"), jobsHolder = el("div");
  const body = el("div", { class: "stack" },
    statusStrip, kpis, traffic.node,
    el("div", { class: "grid cols-2" }, latency.node, shares.node),
    el("div", { class: "grid cols-2" }, perModel.node, media.node),
    scores.node, provenance.node,
    el("div", { class: "grid cols-3" }, memory.node, cpu.node, gpu.node),
    loaded, jobsHolder);
  put(root,
    head("Overview", "Live activity of this server. Predictions and resource use since it started; refreshes every 5 seconds."),
    el("div", { class: "filter-row" }, winButtons, modelSelect, el("span", { class: "spacer" }), updated),
    body);

  let last = null;
  const draw = () => {
    if (!last) return;
    const { t, srv, stats, jobs, inUse } = last;
    const series = t.series, xs = series.map((b) => b.t), tot = t.totals;
    const decided = tot.real + tot.fake, computed = tot.n - tot.cached;
    // filter options: models seen in traffic or in memory
    Object.keys(stats.models).concat(Object.keys(t.by_model)).forEach((m) => seenModels.add(m));
    if (model) seenModels.add(model);
    const opts = ["", ...[...seenModels].sort()];
    if (modelSelect.options.length !== opts.length) {
      put(modelSelect, opts.map((m) => el("option", { value: m, text: m || "All models" })));
    }
    modelSelect.value = model;
    put(statusStrip,
      srv.ready ? statusMark("good", "Ready") : statusMark("warning", "Loading"),
      el("span", { class: "badge", text: srv.device }), el("span", { class: "badge", text: `up ${secs(srv.uptime_s)}` }),
      el("span", { class: "badge", text: `${Object.keys(stats.models).length}/${srv.max_loaded} models in memory` }),
      el("span", { class: "badge", text: `${Object.values(inUse).filter(Boolean).length} calibrated` }),
      el("span", { class: "badge", text: `v${srv.version}` }));
    put(kpis, tiles([
      { label: "Requests", value: compact(tot.n), note: `last ${WINDOWS.find((w) => w[0] === t.window)[1]}`, spark: series.map((b) => b.n) },
      { label: "Flagged fake", value: compact(tot.fake), note: decided ? `${pct(share(tot.fake, decided))} of decided` : "no decisions yet", spark: series.map((b) => b.fake) },
      { label: "Uncertain", value: compact(tot.uncertain), note: tot.n ? `${pct(share(tot.uncertain, tot.n))} of requests` : "—" },
      { label: "Error rate", value: tot.n ? pct(share(tot.error, tot.n)) : "—", note: `${compact(tot.error)} errors`, spark: series.map((b) => (b.n ? b.error / b.n : null)) },
      { label: "Latency p50", value: secs(tot.p50), note: `p95 ${secs(tot.p95)}`, spark: series.map((b) => b.p50) },
      { label: "Cache hit rate", value: tot.n ? pct(share(tot.cached, tot.n)) : "—", note: `${compact(tot.cached)} served from cache` },
      { label: "Provenance: evidence of fake", value: compact(t.provenance.fake), note: t.provenance.n ? `${compact(t.provenance.n)} checks · ${compact(t.provenance.real)} verified capture` : "no checks yet" },
    ]));
    provenance.node.style.display = t.provenance.n ? "" : "none";
    if (t.provenance.n) {
      const rows = Object.entries(t.provenance.by_verdict);
      put(provenance.body, hBars(rows.map(([v, n]) => ({ label: verdictName(v), value: n, display: compact(n) })), { valueLabel: "provenance checks" }));
      put(provenance.actions, el("span", { class: "small muted", text: Object.entries(t.provenance.by_check).map(([c, v]) => `${c} ${Object.values(v).reduce((a, b) => a + b, 0)}`).join(" · ") }));
    }
    const width = (c) => c.body.clientWidth || 600;
    const empty = (msg) => el("div", { class: "empty", text: msg });
    const noTraffic = tot.n === 0;

    const trafficTable = simpleTable(["Time", "Requests", "Real", "Fake", "Uncertain", "Error"],
      series.filter((b) => b.n).map((b) => [clock(b.t), String(b.n), String(b.real), String(b.fake), String(b.uncertain), String(b.error)]), [1, 2, 3, 4, 5]);
    const trafficView = withTableView(noTraffic ? empty("No requests in this period. Score a file in Test.") : el("div", {}, outcomeLegend(), outcomeColumns(series, width(traffic))), trafficTable);
    put(traffic.actions, trafficView.toggle); put(traffic.body, trafficView.node);

    put(latency.node.querySelector("h2"), model ? `Latency · ${model}` : "Latency · all models");
    put(latency.actions);
    perModel.node.parentElement.style.display = model ? "none" : "";  // one-bar charts add nothing
    const latView = withTableView(computed ? timeLines(xs, [
      { name: "p50", color: "var(--series-1)", values: series.map((b) => b.p50) },
      { name: "p95", color: "var(--series-2)", values: series.map((b) => b.p95) }], width(latency), { fmt: secs }) : empty("No computed requests in this period."),
    simpleTable(["Time", "p50", "p95"], series.filter((b) => isNum(b.p50)).map((b) => [clock(b.t), secs(b.p50), secs(b.p95)]), [1, 2]));
    put(latency.actions, latView.toggle); put(latency.body, latView.node);

    const models = Object.entries(t.by_model);
    const shareView = withTableView(models.length ? el("div", {}, outcomeLegend(), outcomeShares(t.by_model)) : empty("No requests in this period."),
      simpleTable(["Model", "Requests", "Real", "Fake", "Uncertain", "Error", "p50", "p95"],
        models.map(([n, m]) => [n, String(m.n), String(m.real), String(m.fake), String(m.uncertain), String(m.error), secs(m.p50), secs(m.p95)]), [1, 2, 3, 4, 5, 6, 7]));
    put(shares.actions, shareView.toggle); put(shares.body, shareView.node);

    put(perModel.body, models.length ? hBars(models.map(([n, m]) => ({ label: n, value: m.n, display: `${compact(m.n)} · p50 ${secs(m.p50)}` })), { valueLabel: "requests" })
      : empty("No requests in this period."));
    const mediaRows = Object.entries(t.by_media);
    put(media.body, mediaRows.length ? hBars(mediaRows.map(([k, v]) => ({ label: k, value: v, display: compact(v) })), { valueLabel: "model runs" })
      : empty("No requests in this period."));

    scores.node.style.display = model ? "" : "none";
    if (model) {
      const sc = t.scores || { values: [] };
      put(scores.actions, el("span", { class: "small muted", text: `${model} · ${sc.calibrated ? "calibrated probabilities" : "raw scores"} · ${sc.values.length} requests` }));
      put(scores.body, sc.values.length ? valueHistogram(sc.values, sc.threshold, width(scores)) : empty(`No ${model} requests in this period.`));
    }

    const sys = t.system, sxs = sys.map((s) => s.t);
    put(memory.body, sys.length > 1 ? timeLines(sxs, [{ name: "Resident memory", color: "var(--series-1)", values: sys.map((s) => s.rss_mb) }], width(memory), { fmt: mem, height: 150 }) : empty("Collecting…"));
    put(cpu.body, sys.length > 1 ? timeLines(sxs, [{ name: "CPU", color: "var(--series-1)", values: sys.map((s) => s.cpu_pct) }], width(cpu), { fmt: (v) => `${Math.round(v)}%`, height: 150 }) : empty("Collecting…"));
    put(memory.actions, sys.length ? el("span", { class: "small muted", text: `${mem(sys[sys.length - 1].rss_mb)} now` }) : null);
    put(cpu.actions, sys.length ? el("span", { class: "small muted", text: `${Math.round(sys[sys.length - 1].cpu_pct)}% now · 100% = 1 core` }) : null);
    const hasGpu = sys.some((s) => isNum(s.gpu_mb));
    gpu.node.style.display = hasGpu ? "" : "none";
    if (hasGpu) put(gpu.body, timeLines(sxs, [{ name: "Allocated", color: "var(--series-1)", values: sys.map((s) => s.gpu_mb) },
      { name: "Reserved", color: "var(--series-2)", values: sys.map((s) => s.gpu_reserved_mb) }], width(gpu), { fmt: mem, height: 150 }));

    put(loaded, loadedCard(Object.entries(stats.models), inUse, refresh));
    put(jobsHolder, jobsSummaryCard(jobs.jobs));
  };

  async function refresh() {
    body.classList.add("refreshing");
    try {
      const q = `/v1/telemetry?window=${windowS}${model ? "&model=" + encodeURIComponent(model) : ""}`;
      const [t, srv, stats, jobs, cals] = await Promise.all([
        api(q), api("/v1/server"), api("/v1/stats"),
        api("/v1/jobs").catch(() => ({ jobs: [] })), api("/v1/calibrations").catch(() => ({ calibrations: [] })),
      ]);
      last = { t, srv, stats, jobs, inUse: Object.fromEntries(cals.calibrations.map((c) => [c.model, c.in_use])) };
      draw();
      updated.textContent = `Updated ${clock(Date.now() / 1000, true)}`;
    } finally { body.classList.remove("refreshing"); }
  }
  await refresh();
  every(5000, () => refresh().catch(() => {}));
  let lastWidth = root.clientWidth, timer = null;
  const ro = new ResizeObserver(() => {
    if (root.clientWidth === lastWidth) return;
    lastWidth = root.clientWidth;
    clearTimeout(timer);
    timer = setTimeout(draw, 120);
  });
  ro.observe(root);
  cleanups.push(() => ro.disconnect());
}

function loadedCard(models, inUse, refresh) {
  const rows = models.map(([name, m]) => [
    el("a", { href: `#/model/${encodeURIComponent(name)}`, text: name }),
    m.family, secs(m.load_seconds), compact(m.calls), compact(m.errors), secs(m.mean_seconds),
    inUse[name] ? el("span", { class: "badge", text: inUse[name] }) : el("span", { class: "muted", text: "none" }),
    el("button", { class: "small", type: "button", text: "Unload", onclick: guard(async () => {
      await api(`/v1/models/${encodeURIComponent(name)}`, { method: "DELETE", admin: true });
      toast(`Unloaded ${name}`, "ok"); refresh();
    }) }),
  ]);
  return el("div", { class: "card" },
    el("div", { class: "card-head" }, el("h2", { text: "Models in memory" }), el("a", { href: "#/models", text: "All models" })),
    rows.length ? simpleTable(["Model", "Family", "Load", "Calls", "Errors", "Mean time", "Calibration", ""], rows, [2, 3, 4, 5])
      : el("div", { class: "empty", text: "No model loaded yet. Load one from Models, or score a file in Test." }));
}

function jobRow(j) {
  const kinds = { calibrate: "Calibration", check: "Drift check" };
  return [el("a", { href: `#/calibration/${j.id}`, text: j.id }), kinds[j.kind] || j.kind, jobStatus(j.status), when(j.created)];
}

function jobStatus(status) {
  if (status === "done") return statusMark("good", "Done");
  if (status === "failed") return statusMark("critical", "Failed");
  if (status === "running") return el("span", { class: "badge" }, el("span", { class: "dot", style: { background: "var(--accent)" } }), "Running");
  return el("span", { class: "badge", text: "Queued" });
}

function jobsSummaryCard(jobs) {
  const recent = jobs.slice(-5).reverse();
  return el("div", { class: "card" },
    el("div", { class: "card-head" }, el("h2", { text: "Recent calibration jobs" }), el("a", { href: "#/calibration", text: "Calibration" })),
    recent.length ? simpleTable(["Job", "Kind", "Status", "Created"], recent.map(jobRow))
      : el("div", { class: "empty", text: "No jobs since the server started." }));
}

// -- models ----------------------------------------------------------------------------

async function modelsPage(root) {
  const [models, cals] = await Promise.all([api("/v1/models"), api("/v1/calibrations").catch(() => ({ calibrations: [] }))]);
  const inUse = Object.fromEntries(cals.calibrations.map((c) => [c.model, c.in_use]));
  const families = [...new Set(models.models.map((m) => m.family))].sort();
  const search = el("input", { type: "search", placeholder: "Search models", "aria-label": "Search models" });
  const family = el("select", { "aria-label": "Family" }, el("option", { value: "", text: "All families" }), families.map((f) => el("option", { value: f, text: f })));
  const loadedOnly = el("input", { type: "checkbox" });
  const tableHolder = el("div");
  const render = () => {
    const q = search.value.trim().toLowerCase();
    const rows = models.models.filter((m) => (!q || m.name.includes(q) || (m.description || "").toLowerCase().includes(q))
      && (!family.value || m.family === family.value) && (!loadedOnly.checked || m.loaded));
    put(tableHolder, rows.length ? simpleTable(["Model", "Family", "State", "Calibration", "Default for", "License", ""],
      rows.map((m) => [
        el("a", { href: `#/model/${encodeURIComponent(m.name)}`, text: m.name }),
        m.family,
        m.loaded ? statusMark("good", "Loaded") : m.available ? el("span", { class: "muted", text: "Not loaded" }) : statusMark("serious", "Needs trained weights"),
        inUse[m.name] ? el("span", { class: "badge", text: inUse[m.name] }) : el("span", { class: "muted", text: "—" }),
        (m.default_for || []).join(", ") || el("span", { class: "muted", text: "—" }),
        el("span", { class: "small", title: m.license || "", text: (m.license || "").split(/[.;(]/)[0].slice(0, 48) }),
        loadButton(m, () => modelsPage(root)),
      ])) : el("div", { class: "empty", text: "No model matches." }));
  };
  [search, family, loadedOnly].forEach((c) => c.addEventListener("input", render));
  put(root, 
    head("Models", `${models.models.length} registered detectors. Load a model to keep it in memory for fast predictions.`),
    el("div", { class: "row", style: { marginBottom: "12px" } }, search, family, el("label", { class: "check" }, loadedOnly, "Loaded only")),
    el("div", { class: "card" }, tableHolder));
  render();
}

function loadButton(m, after) {
  if (!m.available) return el("span");
  if (m.loaded) {
    return el("button", { class: "small", type: "button", text: "Unload", onclick: guard(async () => {
      await api(`/v1/models/${encodeURIComponent(m.name)}`, { method: "DELETE", admin: true });
      toast(`Unloaded ${m.name}`, "ok"); after();
    }) });
  }
  const b = el("button", { class: "small", type: "button", text: "Load" });
  b.onclick = guard(async () => {
    b.disabled = true; b.textContent = "Loading…";
    try {
      const r = await api(`/v1/models/${encodeURIComponent(m.name)}/load`, { method: "POST", admin: true });
      toast(`Loaded ${m.name} in ${secs(r.load_seconds)}`, "ok"); after();
    } finally { b.disabled = false; b.textContent = "Load"; }
  });
  return b;
}

// -- model detail ----------------------------------------------------------------------

async function modelPage(root, name) {
  const [m, stats, cal, bench] = await Promise.all([
    api(`/v1/models/${encodeURIComponent(name)}`), api("/v1/stats"),
    api(`/v1/calibrations/${encodeURIComponent(name)}`).catch(() => null),
    api(`/v1/benchmarks?model=${encodeURIComponent(name)}`).catch(() => ({ benchmarks: [] })),
  ]);
  const s = stats.models[name];
  const paper = safeUrl(m.paper);
  const perfHolder = el("div");
  const info = el("div", { class: "card" },
    el("h2", { text: "About" }),
    m.description ? el("p", { style: { marginTop: 0 }, text: m.description }) : null,
    el("dl", { class: "kv" },
      el("dt", { text: "Family" }), el("dd", { text: m.family }),
      el("dt", { text: "License" }), el("dd", { text: m.license || "—" }),
      m.members ? [el("dt", { text: "Members" }), el("dd", { text: m.members.join(", ") })] : null,
      el("dt", { text: "Paper" }), el("dd", {}, paper ? el("a", { href: paper, target: "_blank", rel: "noopener noreferrer", text: paper }) : "—"),
      el("dt", { text: "Default for" }), el("dd", { text: (m.default_for || []).join(", ") || "—" }),
      el("dt", { text: "Threshold override" }), el("dd", { text: isNum(m.threshold_override) ? num(m.threshold_override) : "none" })));
  const runtime = el("div", { class: "card" },
    el("h2", { text: "Runtime" }),
    s ? tiles([
      { label: "Load time", value: secs(s.load_seconds) },
      { label: "Requests", value: compact(s.calls) },
      { label: "Errors", value: compact(s.errors) },
      { label: "Mean time", value: secs(s.mean_seconds) },
      { label: "Cache hits", value: compact(s.cache_hits) },
      { label: "Mean batch", value: isNum(s.mean_batch_size) ? num(s.mean_batch_size, 1) : "—" },
    ]) : el("div", { class: "empty", text: m.available ? "Not in memory. Load it to see runtime statistics." : "This model needs trained weights configured on the server." }));
  put(root, 
    head(name, null,
      m.loaded ? statusMark("good", "Loaded") : el("span", { class: "muted", text: "Not loaded" }),
      loadButton(m, () => modelPage(root, name)),
      el("a", { class: "btn", href: `#/test/${encodeURIComponent(name)}`, text: "Test this model" })),
    el("div", { class: "stack" },
      el("div", { class: "grid cols-2" }, info, runtime),
      calibrationCard(name, cal, perfHolder, root),
      perfHolder,
      benchmarksCard(bench.benchmarks)));
  const shown = cal && (cal.active_version || (cal.versions.length ? cal.versions[cal.versions.length - 1].version : null));
  if (shown) showPerformance(name, shown, perfHolder);
}

function calibrationCard(name, cal, perfHolder, root) {
  const versions = cal ? cal.versions : [];
  const card = el("div", { class: "card" },
    el("div", { class: "card-head" },
      el("h2", { text: "Calibration versions" }),
      el("div", { class: "row" },
        cal && cal.in_use ? el("span", { class: "badge", text: `In use: ${cal.in_use}` }) : el("span", { class: "muted", text: "Uncalibrated (raw scores)" }),
        cal && cal.active_version ? el("button", { class: "small", type: "button", text: "Use raw scores", onclick: guard(async () => {
          await api(`/v1/calibrations/${encodeURIComponent(name)}/active`, { method: "PUT", json: { version: null }, admin: true });
          toast(`${name}: calibration off`, "ok"); modelPage(root, name);
        }) }) : null,
        el("a", { href: `#/calibration?model=${encodeURIComponent(name)}`, text: "Calibrate" }))));
  if (!versions.length) {
    card.append(el("div", { class: "empty", text: "No calibration yet. Upload a labelled dataset and calibrate this model on it to get calibrated probabilities and confidence verdicts." }));
    return card;
  }
  card.append(simpleTable(["Version", "Created", "Mode", "Built on", "Datasets", "Real", "Fake", "AUC", "Alpha", ""],
    versions.slice().reverse().map((v) => [
      el("a", { href: "#", text: `v${v.version}`, onclick: (e) => { e.preventDefault(); showPerformance(name, v.version, perfHolder); } }),
      when(v.created), v.mode, v.parent ? `v${v.parent}` : "—", (v.datasets || []).join(", ") || "—",
      compact(v.n_real), compact(v.n_fake), num(v.auc), pct(v.alpha, 0),
      v.version === cal.active_version ? el("span", { class: "badge", text: "Active" })
        : el("button", { class: "small", type: "button", text: "Activate", onclick: guard(async () => {
          await api(`/v1/calibrations/${encodeURIComponent(name)}/active`, { method: "PUT", json: { version: v.version }, admin: true });
          toast(`${name}: v${v.version} active`, "ok"); modelPage(root, name);
        }) }),
    ]), [5, 6, 7, 8]));
  return card;
}

async function showPerformance(name, version, holder) {
  holder.classList.add("refreshing");
  try {
    const { table, report } = await api(`/v1/calibrations/${encodeURIComponent(name)}/versions/${version}`);
    const cv = report.cv || {};
    const conf = cv.conformal || {}, platt = cv.platt || {}, raw = cv.raw || {}, sets = cv.prediction_sets || {};
    const thr = isNum(table.threshold) ? table.threshold : null;
    const hist = scoreHistogram({ real: table.real_scores, fake: table.fake_scores, threshold: thr, scoreType: table.score_type });
    const download = el("button", { class: "small", type: "button", text: "Download table", onclick: () => {
      const a = el("a", { href: URL.createObjectURL(new Blob([JSON.stringify(table, null, 1)], { type: "application/json" })), download: `${name}-v${version}.json` });
      a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    } });
    const cvRows = [
      ["Default threshold (raw)", raw], ["Calibrated probability ≥ 0.5", platt], [`Conformal (alpha ${pct(conf.alpha, 0)})`, conf],
    ].filter(([, r]) => r && isNum(r.balanced_acc)).map(([label, r]) => [label, pct(r.balanced_acc), pct(r.false_alarm_rate), pct(r.detection_rate), num(r.ece), num(r.brier)]);
    put(holder, el("div", { class: "card" },
      el("div", { class: "card-head" },
        el("h2", { text: `Performance of v${version} on its calibration data` }),
        el("div", { class: "row" }, hist.toggle, download)),
      el("p", { class: "sub small", style: { marginTop: "-6px", marginBottom: "12px" },
        text: `${report.mode || "full"} calibration${report.parent ? " built on v" + report.parent : ""} · ${(report.lineage || []).map((x) => x.name || x.dataset).join(", ") || table.source || "—"} · created ${when(report.created || table.created)}` }),
      tiles([
        { label: "AUC", value: num(report.auc), note: "separation of real and fake" },
        { label: "False alarms", value: pct(conf.false_alarm_rate), note: `target ${pct(table.alpha, 0)}, out of fold` },
        { label: "Fakes caught", value: pct(conf.detection_rate), note: "at the conformal threshold" },
        { label: "Calibration error", value: num(platt.ece, 2), note: isNum(raw.ece) ? `ECE; ${num(raw.ece, 2)} uncalibrated` : "ECE" },
        { label: "Certain verdicts", value: pct(sets.single_label), note: `${pct(sets.both_labels)} uncertain` },
        { label: "Items", value: `${compact(report.n_real)} / ${compact(report.n_fake)}`, note: "real / fake" },
      ]),
      el("h3", { text: "Raw score distribution, with the decision threshold" }),
      hist.node,
      cvRows.length ? [el("h3", { text: "Out-of-fold decisions" }),
        simpleTable(["Decision rule", "Balanced acc.", "False alarms", "Fakes caught", "ECE", "Brier"], cvRows, [1, 2, 3, 4, 5])] : null,
      (report.warnings || []).length ? [el("h3", { text: "Warnings" }), el("ul", { class: "warn-list" }, report.warnings.map((w) => el("li", { text: w })))] : null));
  } catch (e) {
    put(holder, errorState(e));
  } finally {
    holder.classList.remove("refreshing");
  }
}

function benchmarksCard(rows) {
  const card = el("div", { class: "card" }, el("h2", { text: "Benchmarks" }));
  if (!rows.length) {
    card.append(el("div", { class: "empty", text: "No benchmark results for this model in the server's benchmarks folder." }));
    return card;
  }
  const withAuc = rows.filter((r) => isNum(r.metrics.auc));
  if (withAuc.length > 1) {
    card.append(el("h3", { text: "AUC by benchmark" }),
      hBars(withAuc.map((r) => ({ label: r.dataset.slice(0, 26), value: r.metrics.auc, display: num(r.metrics.auc) })), { valueLabel: "AUC" }));
  }
  card.append(simpleTable(["Benchmark", "Source", "AUC", "EER", "Balanced acc.", "Real / fake"],
    rows.map((r) => {
      const mt = r.metrics;
      const eer = isNum(mt.eer) ? pct(mt.eer) : isNum(mt.eer_percent) ? mt.eer_percent + "%" : "—";
      return [r.dataset, el("span", { class: "mono small", text: r.source }), num(mt.auc), eer,
        isNum(mt.balanced_acc) ? pct(mt.balanced_acc) : isNum(mt.accuracy_at_default_threshold) ? pct(mt.accuracy_at_default_threshold) : "—",
        isNum(mt.n_real) ? `${compact(mt.n_real)} / ${compact(mt.n_fake)}` : "—"];
    }), [2, 3, 4]));
  return card;
}

// -- test ------------------------------------------------------------------------------

async function testPage(root, preselect) {
  const models = (await api("/v1/models")).models.filter((m) => m.available);
  const chosen = new Set(preselect ? [preselect] : []);
  let file = null;
  const fileInput = el("input", { type: "file", style: { display: "none" }, accept: "image/*,video/*,audio/*" });
  const preview = el("div");
  const dropText = el("div", {}, el("strong", { text: "Drop an image, video or audio file" }), el("div", { class: "small", text: "or click to choose" }));
  const drop = el("div", { class: "drop", tabindex: "0", role: "button" }, dropText);
  const setFile = (f) => {
    file = f;
    put(dropText, el("strong", { text: f.name }), el("div", { class: "small", text: `${(f.size / 1e6).toFixed(2)} MB · click to change` }));
    const url = URL.createObjectURL(f);
    const kind = (f.type || "").split("/")[0];
    put(preview, kind === "image" ? el("img", { class: "preview", src: url, alt: "" })
      : kind === "video" ? el("video", { class: "preview", src: url, controls: true })
        : kind === "audio" ? el("audio", { src: url, controls: true, style: { width: "100%", marginTop: "12px" } }) : null);
  };
  drop.onclick = () => fileInput.click();
  drop.onkeydown = (e) => (e.key === "Enter" || e.key === " ") && fileInput.click();
  fileInput.onchange = () => fileInput.files[0] && setFile(fileInput.files[0]);
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); e.dataTransfer.files[0] && setFile(e.dataTransfer.files[0]); };

  const byFamily = {};
  models.forEach((m) => (byFamily[m.family] = byFamily[m.family] || []).push(m));
  const choiceGroups = Object.entries(byFamily).sort().map(([fam, list]) => el("div", { style: { marginBottom: "8px" } },
    el("div", { class: "small muted", text: fam }),
    el("div", { class: "choices" }, list.map((m) => {
      const box = el("input", { type: "checkbox", checked: chosen.has(m.name) });
      const chip = el("label", { class: "choice" + (chosen.has(m.name) ? " on" : "") }, box, m.name);
      box.onchange = () => { box.checked ? chosen.add(m.name) : chosen.delete(m.name); chip.classList.toggle("on", box.checked); };
      return chip;
    }))));
  const identity = el("input", { type: "text", placeholder: "for identity models" });
  const details = el("input", { type: "checkbox" });
  const segments = el("input", { type: "checkbox", checked: true });
  const results = el("div");
  const run = el("button", { class: "primary", type: "button", text: "Score file" });
  run.onclick = guard(async () => {
    if (!file) { toast("Choose a file first", "error"); return; }
    const form = new FormData();
    form.append("file", file);
    if (chosen.size) form.append("models", [...chosen].join(","));
    if (identity.value.trim()) form.append("identity", identity.value.trim());
    form.append("details", details.checked ? "true" : "false");
    form.append("segments", segments.checked ? "true" : "false");
    run.disabled = true; run.textContent = "Scoring…";
    results.classList.add("refreshing");
    try {
      const r = await api("/v1/detect", { method: "POST", form });
      put(results, renderDetection(r));
    } finally { run.disabled = false; run.textContent = "Score file"; results.classList.remove("refreshing"); }
  });
  put(root, 
    head("Test", "Score a file with any models. Without a selection, the server's defaults for the media type run."),
    el("div", { class: "grid cols-2" },
      el("div", { class: "card" }, el("h2", { text: "File" }), drop, fileInput, preview),
      el("div", { class: "card" },
        el("div", { class: "card-head" }, el("h2", { text: "Models" }), el("span", { class: "small muted", text: "none selected = defaults" })),
        choiceGroups,
        el("div", { class: "form-grid", style: { marginTop: "8px" } },
          el("label", { class: "field" }, "Identity", identity),
          el("label", { class: "check" }, segments, "Per-segment scores"),
          el("label", { class: "check" }, details, "Evidence (details)")),
        el("div", { class: "row", style: { marginTop: "14px" } }, run))),
    el("div", { style: { marginTop: "16px" } }, results));
}

function renderDetection(r) {
  const media = r.media;
  const summary = el("div", { class: "card", style: { marginBottom: "12px" } },
    el("div", { class: "row" },
      el("strong", { text: r.filename || "file" }),
      el("span", { class: "badge", text: media.kind + (media.has_audio ? " + audio" : "") }),
      isNum(media.duration_s) ? el("span", { class: "muted", text: secs(media.duration_s) }) : null,
      media.bounded ? el("span", { class: "badge", text: `bounded: ${media.bounded.trimmed_to_s ? "trimmed to " + media.bounded.trimmed_to_s + " s " : ""}${media.bounded.scaled_to_height ? "scaled to " + media.bounded.scaled_to_height + "p" : ""}` }) : null,
      el("span", { class: "spacer" }),
      el("span", { class: "muted small", text: `${secs(r.seconds)} total · sha256 ${r.sha256.slice(0, 12)}…` })));
  // the most decisive provenance evidence leads, since it outweighs any detector score
  const rank = { tampered: 6, ai_generated: 5, watermark_found: 5, verified_capture: 4 };
  const lead = Object.entries(r.results).filter(([, res]) => res.provenance && rank[res.provenance.verdict])
    .sort((a, b) => rank[b[1].provenance.verdict] - rank[a[1].provenance.verdict])[0];
  if (lead) {
    const [name, res] = lead;
    summary.append(el("div", { class: "row", style: { marginTop: "10px" } },
      res.label === "real" ? statusMark("good", verdictName(res.provenance.verdict)) : statusMark("critical", verdictName(res.provenance.verdict)),
      el("span", { class: "small", text: `${res.provenance.summary} (${name})` })));
  }
  const cards = Object.entries(r.results).map(([name, res]) => resultCard(name, res));
  return el("div", {}, summary, cards.length ? cards : el("div", { class: "empty", text: "No models ran for this media type." }));
}

const PROVENANCE_VERDICTS = {
  tampered: "Tampered after signing", ai_generated: "AI-generated", watermark_found: "Generator watermark found",
  verified_capture: "Verified capture", partial_watermark: "Partly watermarked", ai_enhanced: "AI-enhanced (declared)",
  credentials_untrusted: "Unverified credentials", trusted_credentials: "Verified credentials",
  invalid_credentials: "Invalid credentials", edited: "Signs of editing", re_encoded: "Re-encoded",
  capture_metadata: "Camera metadata", no_metadata: "No metadata", no_credentials: "No credentials",
  no_watermark: "No watermark", no_findings: "Nothing notable", not_checked: "Not checked",
};
const verdictName = (v) => PROVENANCE_VERDICTS[v] || String(v).replace(/_/g, " ");

/** Provenance evidence: verdict, one-line summary, findings by strength (no score meter: it's not a score). */
function provenanceCard(card, nameLink, res) {
  const p = res.provenance;
  const mark = res.label === "fake" ? statusMark("critical", verdictName(p.verdict))
    : res.label === "real" ? statusMark("good", verdictName(p.verdict))
      : el("span", { class: "badge", text: verdictName(p.verdict) });
  card.append(el("div", { class: "result-head" }, nameLink, mark, el("span", { class: "badge", text: "provenance" }),
    res.cached ? el("span", { class: "badge", text: "cached" }) : null,
    el("span", { class: "spacer" }), el("span", { class: "muted small", text: secs(res.seconds) })));
  card.append(el("div", { class: "small", text: p.summary }));
  if ((p.findings || []).length) {
    card.append(el("ul", { class: "findings" }, p.findings.map((f) => el("li", {},
      el("span", { class: "sev sev-" + f.severity, text: f.severity }), el("span", { text: f.message })))));
  }
  if (res.details && Object.keys(res.details).length) {
    card.append(el("details", {}, el("summary", { text: "Evidence (details)" }), el("pre", { text: JSON.stringify(res.details, null, 2) })));
  }
  return card;
}

function resultCard(name, res) {
  const card = el("div", { class: "result" });
  const nameLink = el("a", { class: "name", href: `#/model/${encodeURIComponent(name)}`, text: name });
  if (res.status !== "ok" && res.status !== "no_face") {
    const kinds = { not_applicable: ["warning", "Not applicable"], unavailable: ["serious", "Unavailable"], error: ["critical", "Error"] };
    const [kind, label] = kinds[res.status] || ["serious", res.status];
    card.append(el("div", { class: "result-head" }, nameLink, statusMark(kind, label)),
      el("div", { class: "small", text: res.reason || res.error || "" }));
    return card;
  }
  if (res.provenance) return provenanceCard(card, nameLink, res);
  const c = res.confidence;
  const isProb = res.score_type === "probability" || !!c;
  const headBits = [nameLink,
    res.status === "no_face" ? statusMark("serious", "No face found") : labelMark(res.label),
    c ? statusMark(...VERDICT[c.verdict]) : null,
    res.calibration ? el("span", { class: "badge", text: `calibrated ${res.calibration}` }) : el("span", { class: "badge", text: "raw score" }),
    res.cached ? el("span", { class: "badge", text: "cached" }) : null,
    el("span", { class: "spacer" }),
    el("span", { class: "muted small", text: secs(res.seconds) })];
  card.append(el("div", { class: "result-head" }, headBits));
  if (isProb && isNum(res.score)) card.append(meter(res.score, res.threshold, 0, 1));
  const kv = [["Score", `${num(res.score, 4)} (${res.score_type})`]];
  if (isNum(res.threshold)) kv.push(["Threshold", num(res.threshold, 4)]);
  else kv.push(["Threshold", "none: anomaly score, calibrate this model for a decision"]);
  if (c) {
    kv.push(["Fake probability", pct(c.fake_probability)], ["Prediction set", c.prediction_set.join(" + ") || "empty"],
      ["p-values", `real ${num(c.p_real)} · fake ${num(c.p_fake)} (alpha ${c.alpha})`]);
  }
  if (res.n_segments) kv.push(["Segments", String(res.n_segments)]);
  card.append(el("dl", { class: "kv" }, kv.map(([k, v]) => [el("dt", { text: k }), el("dd", { text: v })])));
  if (Array.isArray(res.segments) && res.segments.length > 1) {
    card.append(el("h3", { text: "Per-segment scores" }), sparkline(res.segments, isProb ? res.threshold : null));
  }
  if (res.details && Object.keys(res.details).length) {
    card.append(el("details", {}, el("summary", { text: "Evidence (details)" }), el("pre", { text: JSON.stringify(res.details, null, 2) })));
  }
  return card;
}

// -- calibration -----------------------------------------------------------------------

async function calibrationPage(root, param) {
  const query = new URLSearchParams(location.hash.split("?")[1] || "");
  const jobId = param || null;
  const [datasets, models, cals, srv] = await Promise.all([
    api("/v1/datasets"), api("/v1/models"), api("/v1/calibrations").catch(() => ({ calibrations: [] })), api("/v1/server"),
  ]);
  const calibrated = new Set(cals.calibrations.filter((c) => c.versions.length).map((c) => c.model));
  const jobsCard = el("div", { class: "card" });
  const reportHolder = el("div");
  put(root, 
    head("Calibration", "Upload labelled datasets, calibrate models on them, and check calibrations against new data."),
    srv.calibration_persistent ? null : el("div", { class: "banner" },
      el("strong", { text: "Not persistent. " }), "The server has no calibration_dir, so datasets and calibrations are lost when it restarts."),
    el("div", { class: "stack" },
      datasetsCard(datasets.datasets, () => calibrationPage(root, param), srv),
      el("div", { class: "grid cols-2" },
        newJobCard(datasets.datasets, models.models, calibrated, query.get("model"), (job) => { location.hash = `#/calibration/${job.id}`; }),
        checkCard(datasets.datasets, cals.calibrations, (job) => { location.hash = `#/calibration/${job.id}`; })),
      jobsCard, reportHolder));
  const refreshJobs = async () => {
    const { jobs } = await api("/v1/jobs");
    const list = jobs.slice().reverse();
    put(jobsCard, el("div", { class: "card-head" }, el("h2", { text: "Jobs" })),
      list.length ? simpleTable(["Job", "Kind", "Status", "Created"], list.map(jobRow)) : el("div", { class: "empty", text: "No jobs since the server started." }));
  };
  await refreshJobs();
  every(3000, () => refreshJobs().catch(() => {}));
  if (jobId) {
    const show = async () => {
      const job = await api(`/v1/jobs/${encodeURIComponent(jobId)}`);
      put(reportHolder, jobReport(job));
      return job.status === "done" || job.status === "failed";
    };
    if (!(await show())) {
      const id = setInterval(async () => { try { if (await show()) clearInterval(id); } catch (e) { clearInterval(id); } }, 2000);
      cleanups.push(() => clearInterval(id));
    }
    reportHolder.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function datasetsCard(datasets, refresh, srv) {
  const fileInput = el("input", { type: "file", accept: ".zip,application/zip" });
  const nameInput = el("input", { type: "text", placeholder: "Name (optional)" });
  const bar = el("div", { style: { width: "0%" } });
  const progress = el("div", { class: "progress", style: { display: "none" } }, bar);
  const go = el("button", { class: "primary", type: "button", text: "Upload dataset" });
  go.onclick = guard(async () => {
    const f = fileInput.files[0];
    if (!f) { toast("Choose a ZIP file", "error"); return; }
    const form = new FormData();
    form.append("file", f);
    if (nameInput.value.trim()) form.append("name", nameInput.value.trim());
    go.disabled = true; progress.style.display = "block";
    try {
      const info = await uploadWithProgress("/v1/datasets", form, { admin: true, onProgress: (p) => (bar.style.width = (p * 100).toFixed(0) + "%") });
      toast(`Dataset ${info.name}: ${info.n_real} real, ${info.n_fake} fake${info.n_problems ? `, ${info.n_problems} problems` : ""}`, "ok");
      refresh();
    } finally { go.disabled = false; progress.style.display = "none"; bar.style.width = "0%"; }
  });
  const rows = datasets.slice().reverse().map((d) => [
    el("strong", { text: d.name }), el("span", { class: "mono small", text: d.id }), when(d.uploaded),
    compact(d.n_items), compact(d.n_real), compact(d.n_fake), compact(d.n_groups),
    d.n_problems ? el("details", {}, el("summary", { text: `${d.n_problems} problems` }), el("ul", { class: "warn-list" }, d.problems.map((p) => el("li", { text: p }))))
      : el("span", { class: "muted", text: "none" }),
    el("button", { class: "small danger", type: "button", text: "Delete", onclick: guard(async () => {
      if (!confirm(`Delete dataset ${d.name}? Calibrations already fitted on it are kept.`)) return;
      await api(`/v1/datasets/${encodeURIComponent(d.id)}`, { method: "DELETE", admin: true });
      toast(`Deleted ${d.name}`, "ok"); refresh();
    }) }),
  ]);
  return el("div", { class: "card" },
    el("div", { class: "card-head" }, el("h2", { text: "Datasets" }),
      el("span", { class: "small muted", text: `ZIP with manifest.csv (file,label[,group][,identity]) or real/ and fake/ folders · up to ${compact(srv.limits.max_dataset_mb)} MB` })),
    el("div", { class: "row", style: { marginBottom: "12px" } }, fileInput, nameInput, go, progress),
    rows.length ? simpleTable(["Name", "Id", "Uploaded", "Items", "Real", "Fake", "Groups", "Problems", ""], rows, [3, 4, 5, 6])
      : el("div", { class: "empty", text: "No datasets yet." }));
}

function choiceList(items, selected, onChange) {
  return el("div", { class: "choices" }, items.map(({ value, label }) => {
    const box = el("input", { type: "checkbox", checked: selected.has(value) });
    const chip = el("label", { class: "choice" + (selected.has(value) ? " on" : "") }, box, label);
    box.onchange = () => { box.checked ? selected.add(value) : selected.delete(value); chip.classList.toggle("on", box.checked); onChange && onChange(); };
    return chip;
  }));
}

function newJobCard(datasets, models, calibrated, preselect, onSubmit) {
  const chosenData = new Set(), chosenModels = new Set(preselect ? [preselect] : []);
  const mode = el("select", {}, el("option", { value: "full", text: "Full: from scratch on the datasets" }),
    el("option", { value: "incremental", text: "Incremental: previous version + datasets" }),
    el("option", { value: "refit", text: "Refit: previous version, new settings" }));
  const alpha = el("input", { type: "number", step: "0.01", min: "0.001", max: "0.5", placeholder: "0.05" });
  const folds = el("input", { type: "number", min: "2", max: "20", value: "5" });
  const windowIn = el("input", { type: "number", min: "2", placeholder: "keep all" });
  const base = el("input", { type: "number", min: "1", placeholder: "active / latest" });
  const onlyDrift = el("input", { type: "checkbox" });
  const activate = el("input", { type: "checkbox", checked: true });
  const datasetPick = el("div", {}, el("div", { class: "small muted", text: "Datasets" }),
    datasets.length ? choiceList(datasets.map((d) => ({ value: d.id, label: `${d.name} (${d.n_real}/${d.n_fake})` })), chosenData)
      : el("div", { class: "small muted", text: "Upload a dataset first." }));
  const incOnly = [el("label", { class: "field" }, "Base version", base), el("label", { class: "field" }, "Window per class", windowIn),
    el("label", { class: "check" }, onlyDrift, "Only if drift")];
  const update = () => {
    const m = mode.value;
    datasetPick.style.display = m === "refit" ? "none" : "";
    incOnly.forEach((n) => (n.style.display = m === "full" ? "none" : ""));
    incOnly[1].style.display = m === "incremental" ? "" : "none";
    incOnly[2].style.display = m === "incremental" ? "" : "none";
  };
  mode.onchange = update;
  const modelItems = models.filter((m) => m.available).map((m) => ({ value: m.name, label: m.name + (calibrated.has(m.name) ? " ✓" : "") }));
  const go = el("button", { class: "primary", type: "button", text: "Start calibration" });
  go.onclick = guard(async () => {
    const body = { mode: mode.value, models: [...chosenModels], folds: +folds.value || 5, activate: activate.checked };
    if (mode.value !== "refit") body.datasets = [...chosenData];
    if (alpha.value) body.alpha = +alpha.value;
    if (mode.value !== "full" && base.value) body.base_version = +base.value;
    if (mode.value === "incremental") { if (windowIn.value) body.window = +windowIn.value; body.only_if_drift = onlyDrift.checked; }
    if (!body.models.length) { toast("Choose at least one model", "error"); return; }
    if (mode.value !== "refit" && !body.datasets.length) { toast("Choose at least one dataset", "error"); return; }
    const job = await api("/v1/calibrations", { method: "POST", json: body, admin: true });
    toast(`Started job ${job.id}`, "ok");
    onSubmit(job);
  });
  const card = el("div", { class: "card" },
    el("h2", { text: "Calibrate" }),
    el("div", { class: "stack", style: { gap: "12px" } },
      el("label", { class: "field" }, "Mode", mode),
      datasetPick,
      el("div", {}, el("div", { class: "small muted", text: "Models (✓ = has versions)" }), choiceList(modelItems, chosenModels)),
      el("div", { class: "form-grid" },
        el("label", { class: "field" }, "Alpha (false-alarm target)", alpha),
        el("label", { class: "field" }, "Folds", folds),
        incOnly,
        el("label", { class: "check" }, activate, "Activate when done")),
      el("div", { class: "row" }, go)));
  update();
  return card;
}

function checkCard(datasets, cals, onSubmit) {
  const chosenData = new Set();
  const active = cals.filter((c) => c.active_version);
  const chosenModels = new Set(active.map((c) => c.model));
  const go = el("button", { type: "button", text: "Start drift check" });
  go.onclick = guard(async () => {
    if (!chosenData.size) { toast("Choose at least one dataset", "error"); return; }
    if (!chosenModels.size) { toast("No model with an active calibration selected", "error"); return; }
    const job = await api("/v1/calibrations/check", { method: "POST", json: { datasets: [...chosenData], models: [...chosenModels] }, admin: true });
    toast(`Started job ${job.id}`, "ok");
    onSubmit(job);
  });
  return el("div", { class: "card" },
    el("h2", { text: "Check for drift" }),
    el("p", { class: "sub small", style: { marginTop: "-6px" }, text: "Score newly labelled data and test whether the active calibrations still hold." }),
    el("div", { class: "stack", style: { gap: "12px" } },
      el("div", {}, el("div", { class: "small muted", text: "New labelled datasets" }),
        datasets.length ? choiceList(datasets.map((d) => ({ value: d.id, label: d.name })), chosenData) : el("div", { class: "small muted", text: "Upload a dataset first." })),
      el("div", {}, el("div", { class: "small muted", text: "Calibrated models" }),
        active.length ? choiceList(active.map((c) => ({ value: c.model, label: `${c.model} (v${c.active_version})` })), chosenModels)
          : el("div", { class: "small muted", text: "No model has an active calibration yet." })),
      el("div", { class: "row" }, go)));
}

function jobReport(job) {
  const p = job.progress || {};
  const card = el("div", { class: "card" },
    el("div", { class: "card-head" },
      el("h2", { text: `${job.kind === "check" ? "Drift check" : "Calibration"} job ${job.id}` }),
      jobStatus(job.status)),
    el("p", { class: "sub small", style: { marginTop: "-6px" },
      text: `${job.params.mode ? job.params.mode + " · " : ""}models ${job.params.models.join(", ")} · created ${when(job.created)}${job.finished ? " · finished " + when(job.finished) : ""}` }));
  if (job.status === "running" && p.total) {
    const done = p.scored ?? p.done ?? 0;
    card.append(el("div", { class: "row" },
      el("span", { class: "small", text: `${p.stage === "preparing" ? "Preparing media" : "Scoring with " + p.model}: ${done} / ${p.total}` }),
      el("div", { class: "progress", style: { flex: "1" } }, el("div", { style: { width: ((done / p.total) * 100).toFixed(0) + "%" } }))));
  }
  if (job.error) card.append(el("div", { class: "banner", text: job.error }));
  for (const [model, r] of Object.entries(job.results || {})) {
    const box = el("div", { class: "result" });
    const status = r.status === "done" ? statusMark("good", "Done") : r.status === "skipped" ? statusMark("warning", "Skipped") : statusMark("critical", "Failed");
    box.append(el("div", { class: "result-head" }, el("a", { class: "name", href: `#/model/${encodeURIComponent(model)}`, text: model }), status,
      r.version ? el("span", { class: "badge", text: `v${r.version}${r.activated ? " · active" : ""}` }) : null));
    if (r.error || r.reason) box.append(el("div", { class: "small", text: r.error || r.reason }));
    const kv = [];
    if (job.kind === "check" && r.status === "done") {
      kv.push(["Needs recalibration", r.recalibrate ? "yes" : "no"], ["Against version", `v${r.version}`]);
      if (isNum(r.false_alarm_rate)) kv.push(["False alarms", `${pct(r.false_alarm_rate)} (alpha ${pct(r.alpha, 0)})`]);
      if (isNum(r.detection_rate)) kv.push(["Fakes caught", `${pct(r.detection_rate)} (was ${pct(r.calibration_detection_rate)})`]);
      if (isNum(r.auc)) kv.push(["AUC on new data", num(r.auc)]);
      Object.entries(r.tests || {}).forEach(([t, pv]) => kv.push([t.replace(/_/g, " "), `p = ${num(pv, 4)}`]));
    } else if (r.status === "done") {
      const conf = (r.cv || {}).conformal || {};
      kv.push(["Items", `${compact(r.n_real)} real / ${compact(r.n_fake)} fake (${compact(r.n_new)} new, ${compact(r.n_carried)} carried)`],
        ["AUC", num(r.auc)], ["False alarms (CV)", `${pct(conf.false_alarm_rate)} (alpha ${pct(r.alpha, 0)})`], ["Fakes caught (CV)", pct(conf.detection_rate)]);
    } else if (r.status === "skipped" && r.drift) {
      kv.push(["Drift tests", Object.entries(r.drift.tests || {}).map(([t, pv]) => `${t.replace(/_/g, " ")} p=${num(pv, 3)}`).join(" · ")]);
    }
    if (r.n_skipped) kv.push(["Items skipped", String(r.n_skipped)]);
    if (kv.length) box.append(el("dl", { class: "kv" }, kv.map(([k, v]) => [el("dt", { text: k }), el("dd", { text: v })])));
    if ((r.warnings || []).length) box.append(el("ul", { class: "warn-list" }, r.warnings.map((w) => el("li", { text: w }))));
    if ((r.skipped || []).length) {
      box.append(el("details", {}, el("summary", { text: `${r.n_skipped} skipped items` }),
        simpleTable(["File", "Reason"], r.skipped.map((s) => [s.file, s.reason || s.error || ""]))));
    }
    card.append(box);
  }
  return card;
}

// -- settings --------------------------------------------------------------------------

async function settingsPage(root) {
  const key = el("input", { type: "password", value: store.get("apiKey"), placeholder: "X-API-Key", autocomplete: "off" });
  const adminKey = el("input", { type: "password", value: store.get("adminKey"), placeholder: "same as API key if empty", autocomplete: "off" });
  const theme = el("select", {}, ["system", "light", "dark"].map((t) => el("option", { value: t, text: t[0].toUpperCase() + t.slice(1), selected: (store.get("theme") || "system") === t })));
  const info = el("div");
  const save = el("button", { class: "primary", type: "submit", text: "Save" });
  save.onclick = (e) => {
    e.preventDefault();
    store.set("apiKey", key.value.trim());
    store.set("adminKey", adminKey.value.trim());
    const t = theme.value === "system" ? "" : theme.value;
    store.set("theme", t);
    t ? (document.documentElement.dataset.theme = t) : delete document.documentElement.dataset.theme;
    toast("Saved", "ok");
    refreshServerLine();
    loadInfo();
  };
  const loadInfo = async () => {
    try {
      const s = await api("/v1/server");
      const rows = [
        ["Version", s.version], ["Device", s.device], ["Uptime", secs(s.uptime_s)], ["Models in memory (max)", String(s.max_loaded)],
        ["Preloaded", s.preload.join(", ") || "none"], ["Authentication", s.auth ? "API key required" : "open"],
        ["Calibration storage", s.calibration_persistent ? "persistent" : "temporary (lost on restart)"],
        ["Identity galleries", s.gallery_persistent ? "persistent" : "in memory"],
        ["Upload limit", `${compact(s.limits.max_upload_mb)} MB`], ["Video limits", `${s.limits.max_video_seconds} s, ${s.limits.max_video_height}p`],
        ["Fusions", s.fusions.join(", ") || "none"], ["Trained checkpoints", s.trained_weights.join(", ") || "none"],
        ...Object.entries(s.default_models).map(([k, v]) => [`Defaults: ${k}`, v.join(", ")]),
      ];
      put(info, el("dl", { class: "kv" }, rows.map(([k, v]) => [el("dt", { text: k }), el("dd", { text: v })])));
    } catch (e) { put(info, errorState(e)); }
  };
  put(root, head("Settings"),
    el("div", { class: "grid cols-2" },
      el("div", { class: "card" }, el("h2", { text: "Access" }),
        el("p", { class: "sub small", style: { marginTop: "-6px" }, text: "Keys are kept in this browser only (localStorage) and sent as the X-API-Key header." }),
        el("form", { class: "stack", style: { gap: "12px" }, onsubmit: (e) => { e.preventDefault(); save.click(); } },
          el("label", { class: "field" }, "API key", key),
          el("label", { class: "field" }, "Admin key (loading models, calibration)", adminKey),
          el("label", { class: "field" }, "Theme", theme),
          el("div", { class: "row" }, save))),
      el("div", { class: "card" }, el("h2", { text: "Server" }), info)));
  loadInfo();
}

// -- start -----------------------------------------------------------------------------

window.addEventListener("hashchange", route);
refreshServerLine();
setInterval(refreshServerLine, 15000);
route();
