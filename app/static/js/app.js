
/* =========================================================
   AeroSense — Air Quality Image Classification Dashboard
   Loads real API metrics and supports image prediction.
   Prediction spinner remains hidden at all times.
   ========================================================= */

"use strict";

/* ---------- Utilities ---------- */

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) =>
  [...root.querySelectorAll(selector)];

const NA = "Not evaluated yet";

const api = async (path) => {
  try {
    const response = await fetch(path);
    if (!response.ok) return null;
    return await response.json();
  } catch (error) {
    console.error(`API request failed: ${path}`, error);
    return null;
  }
};

const fmtPct = (value) =>
  value == null ? NA : `${(Number(value) * 100).toFixed(2)}%`;

const escapeHTML = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  })[char]);

/* ---------- Spinner: always hidden ---------- */

function hidePredictionSpinner() {
  const spinner = $("#spinner");
  if (!spinner) return;

  spinner.hidden = true;
  spinner.style.setProperty("display", "none", "important");
  spinner.setAttribute("aria-hidden", "true");
}

/* ---------- Initialisation ---------- */

document.addEventListener("DOMContentLoaded", () => {
  hidePredictionSpinner();

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

/* ---------- Scroll reveal ---------- */

function revealOnScroll() {
  if (!("IntersectionObserver" in window)) {
    $$(".section, .hero-card").forEach((el) => el.classList.add("in"));
    return;
  }

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("in");
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 });

  $$(".section, .hero-card").forEach((el) => observer.observe(el));
}

/* ---------- Hero card tilt ---------- */

function enableTilt(element) {
  if (!element) return;

  element.addEventListener("mousemove", (event) => {
    const rect = element.getBoundingClientRect();
    if (!rect.width || !rect.height) return;

    const rx = ((event.clientY - rect.top) / rect.height - 0.5) * -6;
    const ry = ((event.clientX - rect.left) / rect.width - 0.5) * 6;

    element.style.transform =
      `perspective(900px) rotateX(${rx}deg) rotateY(${ry}deg) translateY(-4px)`;
  });

  element.addEventListener("mouseleave", () => {
    element.style.transform = "";
  });
}

/* ---------- Model status ---------- */
async function loadStatus() {
  const status = await api("/api/status");
  const text = $("#statusText");
  const wrapper = $("#modelStatus");

  if (!text || !wrapper) return;

  // Model file exists: show trained status.
  if (status && status.model_ready) {
    text.textContent = "Model Trained";
    wrapper.classList.add("ready");
  } else {
    text.textContent = "Model not trained";
    wrapper.classList.remove("ready");
  }

  const heroModel = $("#heroModel");
  const heroClasses = $("#heroClasses");

  if (heroModel) {
    heroModel.textContent = status?.selected_model || "—";
  }

  if (heroClasses) {
    heroClasses.textContent = Array.isArray(status?.classes)
      ? status.classes.length
      : "—";
  }
}


/* ---------- Dataset helpers ---------- */

function statCard(label, value) {
  return `
    <div class="glass stat">
      <span class="metric-label">${escapeHTML(label)}</span>
      <div class="v">${escapeHTML(value)}</div>
    </div>`;
}

function distRow(name, value, percentage) {
  const width = Math.max(0, Math.min(100, Number(percentage) || 0));

  return `
    <div class="dist-row">
      <span>${escapeHTML(name)}</span>
      <div class="dist-track">
        <div class="dist-fill" style="width:${width}%"></div>
      </div>
      <span>${escapeHTML(value)}</span>
    </div>`;
}

/* ---------- Dataset ---------- */

async function loadDataset() {
  const data = await api("/api/dataset");
  const stats = $("#datasetStats");
  const distribution = $("#classDist");
  const distributionWrap = $("#classDistWrap");

  if (!data || !data.available) {
    if (stats) {
      stats.innerHTML = statCard("Dataset", "Audit not run yet");
    }

    if (distributionWrap) distributionWrap.style.display = "none";
    return;
  }

  if (!stats || !distribution) return;

  const totals = data.totals || {};
  const classes = data.classes || {};

  stats.innerHTML = [
    statCard("Total records", totals.records ?? "—"),
    statCard("Valid images", totals.valid_images ?? "—"),
    statCard("Classes", classes.n_classes ?? "—"),
    statCard("Imbalance ratio", classes.imbalance_ratio ?? "—"),
    statCard("Corrupted", totals.corrupted_or_unreadable ?? "—")
  ].join("");

  if (distributionWrap) distributionWrap.style.display = "";

  const counts = classes.counts || {};
  const maxCount = Math.max(1, ...Object.values(counts).map(Number));

  distribution.innerHTML = Object.entries(counts)
    .map(([name, count]) =>
      distRow(name, count, (Number(count) / maxCount) * 100)
    )
    .join("");
}

/* ---------- Selected model and metrics ---------- */

async function loadModel() {
  const data = await api("/api/model");
  const stats = $("#modelStats");

  if (!stats) return;

  if (!data || !data.available) {
    stats.innerHTML = statCard("Model", "Not selected yet");
    return;
  }

  const meta = data.meta || {};
  const metrics = data.metrics;
  const heroAccuracy = $("#heroAcc");
  const heroF1 = $("#heroF1");

  if (heroAccuracy) {
    heroAccuracy.textContent = metrics
      ? fmtPct(metrics.accuracy)
      : NA;
  }

  if (heroF1) {
    heroF1.textContent = metrics?.macro_f1 ?? NA;
  }

  stats.innerHTML = [
    statCard("Selected model", meta.selected_model ?? "—"),
    statCard("Test accuracy", metrics ? fmtPct(metrics.accuracy) : NA),
    statCard("Macro F1", metrics?.macro_f1 ?? NA),
    statCard("Weighted F1", metrics?.weighted_f1 ?? NA),
    statCard(
      "Parameters",
      meta.parameters != null
        ? Number(meta.parameters).toLocaleString()
        : "—"
    ),
    statCard(
      "Model size",
      meta.model_size_mb != null ? `${meta.model_size_mb} MB` : "—"
    ),
    statCard(
      "Input size",
      meta.image_size != null ? `${meta.image_size}px` : "—"
    ),
    statCard(
      "Gen. gap",
      metrics?.generalization_gap_valF1_minus_testF1 ?? NA
    )
  ].join("");
}

/* ---------- Model comparison ---------- */

async function loadComparison() {
  const data = await api("/api/experiments");
  const empty = $("#compEmpty");
  const table = $("#compTable");

  if (!table) return;

  if (!data || !data.available ||
      !Array.isArray(data.rows) || !data.rows.length) {
    table.style.display = "none";
    if (empty) empty.style.display = "";
    return;
  }

  table.style.display = "";

  if (empty) empty.style.display = "none";

  const columns = [
    "experiment", "model", "val_accuracy", "val_macro_f1",
    "val_loss", "parameters", "model_size_mb", "training_time_sec"
  ];

  const headings = [
    "Experiment", "Model", "Val Acc", "Val MacroF1",
    "Val Loss", "Params", "Size (MB)", "Time (s)"
  ];

  const thead = $("thead", table);
  const tbody = $("tbody", table);

  if (!thead || !tbody) return;

  thead.innerHTML =
    "<tr>" +
    headings.map((heading) => `<th>${escapeHTML(heading)}</th>`).join("") +
    "</tr>";

  tbody.innerHTML = data.rows.map((row) => {
    const selected = row.model === data.selected ? "selected" : "";

    const cells = columns.map((key) => {
      let value = row[key];

      if (key === "parameters" && value != null) {
        value = Number(value).toLocaleString();
      }

      return `<td>${escapeHTML(value ?? "—")}</td>`;
    }).join("");

    return `<tr class="${selected}">${cells}</tr>`;
  }).join("");
}

/* ---------- Evaluation and error analysis ---------- */

async function loadEvaluation() {
  const data = await api("/api/evaluation");
  if (!data || !data.available) return;

  const plots = data.plots || {};

  const confusionEmpty = $("#cmEmpty");
  const confusionImage = $("#cmImg");
  const errorsImage = $("#errImg");
  const errorsEmpty = $("#errEmpty");

  if (confusionEmpty) confusionEmpty.style.display = "none";

  if (confusionImage && plots.confusion_norm) {
    confusionImage.src = plots.confusion_norm;
    confusionImage.hidden = false;
  }

  if (errorsImage && plots.errors_grid) {
    errorsImage.src = plots.errors_grid;
    errorsImage.hidden = false;
  }

  if (errorsEmpty) errorsEmpty.style.display = "none";

  const perClass = data.metrics?.per_class;
  if (!perClass) return;

  const perClassEmpty = $("#perClassEmpty");
  const perClassContainer = $("#perClass");

  if (perClassEmpty) perClassEmpty.style.display = "none";
  if (!perClassContainer) return;

  perClassContainer.innerHTML = Object.entries(perClass)
    .map(([name, values]) => {
      const f1 = Number(values["f1-score"]) || 0;
      return distRow(name, f1.toFixed(4), f1 * 100);
    })
    .join("");
}

/* ---------- About section ---------- */

function renderAbout() {
  const container = $("#aboutGrid");
  if (!container) return;

  const items = [
    [
      "Problem",
      "Classify the air-quality category of an outdoor scene from a single RGB image using deep learning."
    ],
    [
      "Objectives",
      "<ul><li>Best genuine generalization</li><li>Strong accuracy/efficiency trade-off</li><li>Leakage-free, reproducible pipeline</li></ul>"
    ],
    [
      "Dataset",
      "Air Pollution Image Dataset from India and Nepal. Structure, classes and counts are auto-discovered during the audit."
    ],
    [
      "Methodology",
      "<ul><li>Audit → clean → deduplicate → leakage-safe split</li><li>Screen candidates → fine-tune best</li><li>One-shot test evaluation</li></ul>"
    ],
    [
      "Architecture",
      "ImageNet backbone → GlobalAveragePooling → Dropout → Dense softmax. Two-stage transfer learning and fine-tuning."
    ],
    [
      "Technologies",
      "Python, TensorFlow/Keras, scikit-learn, OpenCV/Pillow, Flask, HTML, CSS and JavaScript."
    ],
    [
      "Limitations",
      "A visual classifier does not measure pollutant concentrations; performance depends on similarity to training data."
    ],
    [
      "Future scope",
      "<ul><li>Fuse image and sensor metadata</li><li>Calibrated confidence</li><li>Efficient on-device deployment</li></ul>"
    ]
  ];

  container.innerHTML = items.map(([heading, body]) =>
    `<div class="glass about-item">
      <h4>${heading}</h4>
      <p>${body}</p>
    </div>`
  ).join("");
}

/* ---------- Upload and prediction ---------- */

let currentFile = null;
let isPredicting = false;

const ALLOWED_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/bmp",
  "image/webp"
]);

const MAX_FILE_SIZE = 12 * 1024 * 1024;

function wireDropzone() {
  const dropzone = $("#dropzone");
  const input = $("#fileInput");
  const predictButton = $("#predictBtn");
  const resetButton = $("#resetBtn");

  if (!dropzone || !input || !predictButton || !resetButton) {
    console.error("Prediction UI elements are missing. Check index.html IDs.");
    return;
  }

  // Prevent a previously visible spinner from appearing.
  hidePredictionSpinner();

  dropzone.addEventListener("click", (event) => {
    if (event.target.closest(".dz-actions")) return;
    input.click();
  });

  input.addEventListener("change", () => {
    if (input.files && input.files[0]) {
      setFile(input.files[0]);
    }
  });

  ["dragover", "dragenter"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add("drag");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.remove("drag");
    });
  });

  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (file) setFile(file);
  });

  resetButton.addEventListener("click", resetPredict);
  predictButton.addEventListener("click", runPredict);
}

function setFile(file) {
  if (!file) return;

  if (!ALLOWED_TYPES.has(file.type)) {
    showPredictionMessage("Please select a JPG, PNG, BMP or WEBP image.");
    return;
  }

  if (file.size > MAX_FILE_SIZE) {
    showPredictionMessage("Image is too large. Maximum upload size is 12 MB.");
    return;
  }

  currentFile = file;

  const preview = $("#preview");
  const dropzoneInner = $("#dzInner");
  const dropzoneActions = $("#dzActions");
  const resultEmpty = $("#resultEmpty");
  const resultBody = $("#resultBody");

  if (preview) {
    if (preview.dataset.objectUrl) {
      URL.revokeObjectURL(preview.dataset.objectUrl);
    }

    const objectUrl = URL.createObjectURL(file);
    preview.dataset.objectUrl = objectUrl;
    preview.src = objectUrl;
    preview.hidden = false;
  }

  if (dropzoneInner) dropzoneInner.hidden = true;
  if (dropzoneActions) dropzoneActions.hidden = false;

  if (resultBody) resultBody.hidden = true;

  if (resultEmpty) {
    resultEmpty.hidden = false;
    resultEmpty.textContent = "Image ready. Press Predict to see the result.";
  }

  hidePredictionSpinner();
}

function showPredictionMessage(message) {
  const resultEmpty = $("#resultEmpty");
  const resultBody = $("#resultBody");

  if (resultBody) resultBody.hidden = true;

  if (resultEmpty) {
    resultEmpty.hidden = false;
    resultEmpty.textContent = message;
  }

  hidePredictionSpinner();
}

function resetPredict() {
  if (isPredicting) return;

  currentFile = null;

  const input = $("#fileInput");
  const preview = $("#preview");
  const dropzoneInner = $("#dzInner");
  const dropzoneActions = $("#dzActions");
  const resultBody = $("#resultBody");
  const resultEmpty = $("#resultEmpty");
  const predictButton = $("#predictBtn");

  if (input) input.value = "";

  if (preview) {
    if (preview.dataset.objectUrl) {
      URL.revokeObjectURL(preview.dataset.objectUrl);
      delete preview.dataset.objectUrl;
    }

    preview.removeAttribute("src");
    preview.hidden = true;
  }

  if (dropzoneInner) dropzoneInner.hidden = false;
  if (dropzoneActions) dropzoneActions.hidden = true;
  if (resultBody) resultBody.hidden = true;

  if (resultEmpty) {
    resultEmpty.hidden = false;
    resultEmpty.textContent =
      "Upload an image and press Predict to see results.";
  }

  if (predictButton) predictButton.disabled = false;

  hidePredictionSpinner();
}

/* ---------- Run prediction: spinner is never shown ---------- */

async function runPredict() {
  if (!currentFile || isPredicting) return;

  const predictButton = $("#predictBtn");
  const resultEmpty = $("#resultEmpty");
  const resultBody = $("#resultBody");

  isPredicting = true;
  hidePredictionSpinner();

  if (predictButton) predictButton.disabled = true;

  if (resultBody) resultBody.hidden = true;

  if (resultEmpty) {
    resultEmpty.hidden = false;
    resultEmpty.textContent = "Analyzing image… Please wait.";
  }

  try {
    const formData = new FormData();
    formData.append("image", currentFile);

    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData
    });

    let result;

    try {
      result = await response.json();
    } catch {
      throw new Error(
        `The server returned an invalid response (HTTP ${response.status}).`
      );
    }

    if (!response.ok || !result || result.error) {
      const message = result?.error || `Prediction failed (HTTP ${response.status}).`;
      const hint = result?.hint ? ` ${result.hint}` : "";
      throw new Error(message + hint);
    }

    renderPrediction(result);

  } catch (error) {
    console.error("Prediction request failed:", error);

    if (resultBody) resultBody.hidden = true;

    if (resultEmpty) {
      resultEmpty.hidden = false;
      resultEmpty.textContent =
        error.message || "Prediction failed. Please try again.";
    }

  } finally {
    hidePredictionSpinner();

    if (predictButton) predictButton.disabled = false;

    isPredicting = false;
  }
}

/* ---------- Render prediction result ---------- */

function renderPrediction(result) {
  if (
    !result ||
    !result.predicted_class ||
    result.confidence == null
  ) {
    throw new Error("The server response is missing prediction details.");
  }

  const resultEmpty = $("#resultEmpty");
  const resultBody = $("#resultBody");
  const predictedClass = $("#predClass");
  const description = $("#predDesc");
  const confidenceRing = $("#confRing");
  const confidenceText = $("#confText");
  const warning = $("#uncWarn");
  const probabilityBars = $("#probBars");
  const gradcamWrap = $("#gradcamWrap");
  const gradcamImage = $("#gradcamImg");

  if (resultEmpty) resultEmpty.hidden = true;
  if (resultBody) resultBody.hidden = false;

  if (predictedClass) {
    predictedClass.textContent = result.predicted_class;
  }

  if (description) {
    description.textContent = result.description || "";
  }

  const confidence = Math.max(
    0,
    Math.min(100, Math.round(Number(result.confidence) * 100))
  );

  if (confidenceRing) {
    confidenceRing.style.setProperty("--p", confidence);
  }

  if (confidenceText) {
    confidenceText.textContent = `${confidence}%`;
  }

  if (warning) {
    warning.hidden = !result.uncertain;

    if (result.uncertain) {
      warning.textContent =
        "⚠ " + (result.uncertainty_note || "Prediction is uncertain.");
    }
  }

  if (probabilityBars) {
    const topResults = Array.isArray(result.top3) ? result.top3 : [];

    probabilityBars.innerHTML = topResults.map((item) => {
      const probability = Math.max(
        0,
        Math.min(100, Math.round(Number(item.probability) * 100))
      );

      return `
        <div class="bar-row">
          <span>${escapeHTML(item.class || "Unknown")}</span>
          <div class="bar-track">
            <div class="bar-fill" style="width:${probability}%"></div>
          </div>
          <span>${probability}%</span>
        </div>`;
    }).join("");
  }

  if (gradcamWrap && gradcamImage) {
    if (result.gradcam_url) {
      gradcamWrap.hidden = false;
      gradcamImage.src =
        result.gradcam_url + "?t=" + Date.now();
    } else {
      gradcamWrap.hidden = true;
      gradcamImage.removeAttribute("src");
    }
  }

  hidePredictionSpinner();
}
