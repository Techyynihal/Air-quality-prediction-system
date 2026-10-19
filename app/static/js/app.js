/* AeroSense dashboard — loads REAL metrics from the API; shows
   "Not evaluated yet" placeholders when result files don't exist. */
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const api = (p) => fetch(p).then((r) => r.json()).catch(() => null);
const NA = "Not evaluated yet";
const fmtPct = (x) => (x == null ? NA : (x * 100).toFixed(2) + "%");

document.addEventListener("DOMContentLoaded", () => {
  revealOnScroll();
  enableTilt($("#heroCard"));
  loadStatus();
  loadDataset();
  loadModel();
  loadComparison();
  loadEvaluation();
  renderAbout();
  wireDropzone();
});

/* ---- scroll reveal (IntersectionObserver, cheap) ---- */
function revealOnScroll() {
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => { if (e.isIntersecting) e.target.classList.add("in"); });
  }, { threshold: 0.12 });
  $$(".section, .hero-card").forEach((el) => io.observe(el));
}

/* ---- subtle 3D tilt on hover (transform only) ---- */
function enableTilt(el) {
  if (!el) return;
  el.addEventListener("mousemove", (e) => {
    const r = el.getBoundingClientRect();
    const rx = ((e.clientY - r.top) / r.height - 0.5) * -6;
    const ry = ((e.clientX - r.left) / r.width - 0.5) * 6;
    el.style.transform = `perspective(900px) rotateX(${rx}deg) rotateY(${ry}deg) translateY(-4px)`;
  });
  el.addEventListener("mouseleave", () => { el.style.transform = ""; });
}

/* ---- status ---- */
async function loadStatus() {
  const s = await api("/api/status");
  const txt = $("#statusText"), wrap = $("#modelStatus");
  if (!s) { txt.textContent = "API offline"; return; }
  if (s.model_ready) { wrap.classList.add("ready"); txt.textContent = s.evaluated ? "Model ready · evaluated" : "Model ready"; }
  else { txt.textContent = "Model not trained"; }
  $("#heroModel").textContent = s.selected_model || "—";
  $("#heroClasses").textContent = s.classes ? s.classes.length : "—";
}

/* ---- dataset ---- */
function statCard(label, value) {
  return `<div class="glass stat"><span class="metric-label">${label}</span><div class="v">${value}</div></div>`;
}
function distRow(name, value, pct) {
  return `<div class="dist-row"><span>${name}</span><div class="dist-track"><div class="dist-fill" style="width:${pct}%"></div></div><span>${value}</span></div>`;
}

async function loadDataset() {
  const d = await api("/api/dataset");
  const stats = $("#datasetStats"), wrap = $("#classDist");
  if (!d || !d.available) {
    stats.innerHTML = statCard("Dataset", "Audit not run yet");
    $("#classDistWrap").style.display = "none";
    return;
  }
  stats.innerHTML = [
    statCard("Total records", d.totals.records),
    statCard("Valid images", d.totals.valid_images),
    statCard("Classes", d.classes.n_classes),
    statCard("Imbalance ratio", d.classes.imbalance_ratio ?? "—"),
    statCard("Corrupted", d.totals.corrupted_or_unreadable),
  ].join("");
  const counts = d.classes.counts || {};
  const max = Math.max(1, ...Object.values(counts));
  wrap.innerHTML = Object.entries(counts)
    .map(([k, v]) => distRow(k, v, (v / max) * 100)).join("");
}

/* ---- final model ---- */
async function loadModel() {
  const m = await api("/api/model");
  const el = $("#modelStats");
  if (!m || !m.available) { el.innerHTML = statCard("Model", "Not selected yet"); return; }
  const meta = m.meta, met = m.metrics;
  $("#heroAcc").textContent = met ? fmtPct(met.accuracy) : NA;
  $("#heroF1").textContent = met ? met.macro_f1 : NA;
  el.innerHTML = [
    statCard("Selected model", meta.selected_model),
    statCard("Test accuracy", met ? fmtPct(met.accuracy) : NA),
    statCard("Macro F1", met ? met.macro_f1 : NA),
    statCard("Weighted F1", met ? met.weighted_f1 : NA),
    statCard("Parameters", meta.parameters ? meta.parameters.toLocaleString() : "—"),
    statCard("Model size", meta.model_size_mb ? meta.model_size_mb + " MB" : "—"),
    statCard("Input size", meta.image_size ? meta.image_size + "px" : "—"),
    statCard("Gen. gap", met ? met.generalization_gap_valF1_minus_testF1 : NA),
  ].join("");
}

/* ---- comparison table ---- */
async function loadComparison() {
  const c = await api("/api/experiments");
  const empty = $("#compEmpty"), table = $("#compTable");
  if (!c || !c.available || !c.rows.length) { table.style.display = "none"; return; }
  empty.style.display = "none";
  const cols = ["experiment", "model", "val_accuracy", "val_macro_f1",
    "val_loss", "parameters", "model_size_mb", "training_time_sec"];
  const head = ["Experiment", "Model", "Val Acc", "Val MacroF1", "Val Loss",
    "Params", "Size (MB)", "Time (s)"];
  $("thead", table).innerHTML = "<tr>" + head.map((h) => `<th>${h}</th>`).join("") + "</tr>";
  $("tbody", table).innerHTML = c.rows.map((r) => {
    const sel = r.model === c.selected ? " class='selected'" : "";
    return `<tr${sel}>` + cols.map((k) => {
      let v = r[k];
      if (k === "parameters" && v != null) v = Number(v).toLocaleString();
      return `<td>${v ?? "—"}</td>`;
    }).join("") + "</tr>";
  }).join("");
}

/* ---- evaluation + errors ---- */
async function loadEvaluation() {
  const e = await api("/api/evaluation");
  if (!e || !e.available) return;
  $("#cmEmpty").style.display = "none";
  const cm = $("#cmImg"); cm.src = e.plots.confusion_norm; cm.hidden = false;
  const err = $("#errImg"); err.src = e.plots.errors_grid; err.hidden = false;
  $("#errEmpty").style.display = "none";
  if (e.metrics && e.metrics.per_class) {
    $("#perClassEmpty").style.display = "none";
    const pc = e.metrics.per_class;
    $("#perClass").innerHTML = Object.entries(pc)
      .map(([k, v]) => distRow(k, v["f1-score"], v["f1-score"] * 100)).join("");
  }
}

/* ---- about (static project narrative) ---- */
function renderAbout() {
  const items = [
    ["Problem", "Classify the air-quality category of an outdoor scene from a single RGB image using deep learning."],
    ["Objectives", "<ul><li>Best genuine generalization</li><li>Strong accuracy/efficiency trade-off</li><li>Leakage-free, reproducible pipeline</li></ul>"],
    ["Dataset", "Air Pollution Image Dataset from India and Nepal. Structure, classes and counts are auto-discovered during the audit."],
    ["Methodology", "<ul><li>Audit → clean → dedup → leakage-safe split</li><li>Screen candidates → fine-tune best</li><li>One-shot test evaluation</li></ul>"],
    ["Architecture", "ImageNet backbone → GlobalAveragePooling → Dropout → Dense softmax. Two-stage transfer learning + fine-tuning."],
    ["Technologies", "Python, TensorFlow/Keras, scikit-learn, OpenCV/Pillow, Flask, HTML/CSS/JS."],
    ["Limitations", "A visual classifier does not measure pollutant concentrations; performance depends on scene similarity to training data."],
    ["Future scope", "<ul><li>Fuse image + sensor metadata</li><li>Calibrated confidence</li><li>On-device efficient deployment</li></ul>"],
  ];
  $("#aboutGrid").innerHTML = items.map(([h, b]) =>
    `<div class="glass about-item"><h4>${h}</h4><p>${b}</p></div>`).join("");
}

/* ---- upload + predict ---- */
let currentFile = null;
function wireDropzone() {
  const dz = $("#dropzone"), input = $("#fileInput");
  dz.addEventListener("click", (e) => { if (e.target.closest(".dz-actions")) return; input.click(); });
  input.addEventListener("change", () => input.files[0] && setFile(input.files[0]));
  ["dragover", "dragenter"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) setFile(f); });
  $("#resetBtn").addEventListener("click", resetPredict);
  $("#predictBtn").addEventListener("click", runPredict);
}

function setFile(file) {
  currentFile = file;
  const img = $("#preview");
  img.src = URL.createObjectURL(file);
  img.hidden = false;
  $("#dzInner").hidden = true;
  $("#dzActions").hidden = false;
}

function resetPredict() {
  currentFile = null;
  $("#fileInput").value = "";
  $("#preview").hidden = true;
  $("#dzInner").hidden = false;
  $("#dzActions").hidden = true;
  $("#resultBody").hidden = true;
  $("#resultEmpty").hidden = false;
}

async function runPredict() {
  if (!currentFile) return;
  $("#spinner").hidden = false;
  const fd = new FormData();
  fd.append("image", currentFile);
  let res;
  try {
    res = await fetch("/api/predict", { method: "POST", body: fd }).then((r) => r.json());
  } catch (_) { res = { error: "Request failed." }; }
  $("#spinner").hidden = true;
  if (!res || res.error) {
    $("#resultEmpty").hidden = false; $("#resultBody").hidden = true;
    $("#resultEmpty").innerHTML = `<p class="muted">${(res && res.error) || "Prediction failed."}<br>${(res && res.hint) || ""}</p>`;
    return;
  }
  renderPrediction(res);
}

function renderPrediction(r) {
  $("#resultEmpty").hidden = true;
  $("#resultBody").hidden = false;
  $("#predClass").textContent = r.predicted_class;
  $("#predDesc").textContent = r.description;
  const pct = Math.round(r.confidence * 100);
  const ring = $("#confRing"); ring.style.setProperty("--p", pct);
  $("#confText").textContent = pct + "%";
  const warn = $("#uncWarn");
  warn.hidden = !r.uncertain;
  if (r.uncertain) warn.textContent = "⚠ " + r.uncertainty_note;
  $("#probBars").innerHTML = r.top3.map((t) => {
    const p = Math.round(t.probability * 100);
    return `<div class="bar-row"><span>${t.class}</span><div class="bar-track"><div class="bar-fill" style="width:${p}%"></div></div><span>${p}%</span></div>`;
  }).join("");
  const gc = $("#gradcamWrap");
  if (r.gradcam_url) { gc.hidden = false; $("#gradcamImg").src = r.gradcam_url + "?t=" + Date.now(); }
  else gc.hidden = true;
}



