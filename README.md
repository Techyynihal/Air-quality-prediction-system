# Air Quality Prediction Using Deep Learning
### Air Quality **Image Classification** — high-accuracy, efficient, presentation-ready

A research-oriented B.Tech major project that classifies the **air-quality category
of an outdoor scene from a single RGB image** using transfer learning, with a
leakage-safe data pipeline, controlled experiments, explainability, and a premium
web dashboard.

> **Scientific scope.** This is an *image classifier*. It predicts a visual
> air-quality **category** (the dataset's ground-truth label). It does **not**
> physically measure PM2.5, PM10, NO₂, SO₂, CO or O₃ — a camera image alone
> cannot measure pollutant concentration. Any AQI metadata is used only to
> *verify* labels, never to fabricate them.

---

## Abstract
Outdoor scenes degrade visually as air quality worsens (haze, reduced contrast,
color shifts). This project learns that mapping with convolutional neural
networks and transfer learning, selecting an architecture that balances accuracy
and efficiency. The emphasis is on **genuine generalization**: strict leakage
prevention, a test set touched exactly once, and evidence-based model selection.

## Problem statement
Given one RGB image, predict its air-quality class as defined by the dataset's
ground truth, maximizing generalization (macro-averaged F1 across classes) while
keeping the model small and fast.

## Objectives
1. Best genuine generalization performance (not train/val-only accuracy).
2. Strong accuracy/efficiency trade-off (parameters, size, latency).
3. A fully reproducible, leakage-free, documented pipeline.

## Dataset
**Air Pollution Image Dataset from India and Nepal**
(<https://www.kaggle.com/datasets/adarshrouniyar/air-pollution-image-dataset-from-india-and-nepal>).
Place it under `dataset/` (see `dataset/README.txt`). The pipeline **auto-discovers**
structure and supports both folder-based labels and CSV/metadata labels. Nothing
about class names, counts, or resolution is assumed — everything is measured in
the audit stage.

**As measured on this machine** (real audit, see `results/audit/dataset_audit.json`).
The Kaggle download ships three redundant copies of the same images; the pipeline
resolves the single canonical root (`Country_wise_Dataset`, which also encodes
city/location) so nothing is counted more than once:

| Fact | Value |
| --- | --- |
| Canonical root | `.../Air Pollution Image Dataset/Country_wise_Dataset` |
| Images (valid / corrupted) | 12,240 / 0 |
| Classes | 6 |
| Resolution / mode | 224×224, RGB (JPEG) |
| Class imbalance ratio | 1.98 |

Per-class counts: `c_Unhealthy_for_Sensitive_Groups`=2863, `d_Unhealthy`=2622,
`e_Very_Unhealthy`=2194, `b_Moderate`=1573, `a_Good`=1541, `f_Severe`=1447.
Metadata (8 per-city CSVs, columns incl. `Location, Year, Month, Day, Hour, AQI,
PM2.5, PM10, O3, CO, SO2, NO2, AQI_Class`) is used to build leakage groups and to
verify labels — never to fabricate them.

## Methodology
```
env audit → discovery → dataset audit → label verification → cleaning →
duplicate/near-duplicate detection → leakage-safe split → EDA →
preprocessing → screening → select candidates → transfer learning →
fine-tuning → controlled experiments → final selection →
ONE-shot test evaluation → error analysis → Grad-CAM → prediction → UI
```

### Preprocessing
RGB decode → resize → **backbone-specific** `preprocess_input` (so training and
inference match exactly). Training augmentation is realistic and mild (horizontal
flip, small rotation/zoom/translation, mild contrast/brightness). Validation and
test use deterministic preprocessing only.

### Data-leakage prevention
Exact duplicates (MD5), near-duplicates (perceptual hash), and same
scene/location/session groups (from metadata) are merged into **leakage groups**
via union-find. Whole groups are assigned to a single split, so no group ever
straddles train/val/test. The split manifest verifies **zero group overlap**.

### Models evaluated
Tier 1 (efficient): MobileNetV3-Large, EfficientNet-B0, EfficientNet-B1.
Tier 2 (strong): ResNet50, ConvNeXt-Tiny, EfficientNetV2-S.
Tier 3 (academic baseline): VGG16 — included for comparison only; it is **not**
allowed to auto-win.

### Transfer learning & fine-tuning
Stage 1 freezes the ImageNet backbone and trains a compact head
(`GAP → Dropout → Dense`). Stage 2 unfreezes the top backbone layers (BatchNorm
kept frozen) and fine-tunes at a low learning rate. EarlyStopping,
ReduceLROnPlateau and ModelCheckpoint restore the **best validation** weights.

### Evaluation metrics
Accuracy, macro/weighted precision-recall-F1, per-class metrics, confusion matrix
(raw + normalized), and the val→test generalization gap. **Macro-F1** drives
selection because it weights minority classes equally.

## Results
Results are **not hard-coded**. After you run the pipeline they appear under
`results/` and are loaded live by the dashboard. Until then the UI shows
"Not evaluated yet". See `results/evaluation/final_metrics.json` and
`results/experiments/experiment_results.csv`.

### Current pipeline status
Data preparation has already been run against the real dataset and its genuine
artifacts are committed under `results/`:

- **Audit + EDA** — `results/audit/`, `results/eda/` (12,240 images, 6 classes).
- **Duplicates** — 129 exact-duplicate copies found (removed), near-duplicates
  reported but intentionally *not* merged into leakage groups (see below).
- **Leakage-safe split** — `results/splits/` — **train 8,434 / val 1,862 /
  test 1,815**, verified **zero group overlap** across all three splits
  (`split_manifest.json`, `"zero_leakage": true`).

**Training, test evaluation, error analysis and Grad-CAM are *Not evaluated yet*.**
They require TensorFlow + a GPU and must be run on your machine (see below); the
dashboard truthfully reports "Not evaluated yet" until you do.

### Run the model stages on your machine (Windows + NVIDIA GPU)
```bat
cd air-quality-deep-learning
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install "tensorflow[and-cuda]"     :: GPU build (skip for CPU-only)

:: split artifacts are already present; go straight to modelling
python run_all.py --stage screen        :: short screening of all candidates
python run_all.py --stage train         :: full fine-tuning + final selection
python run_all.py --stage evaluate       :: ONE-shot test evaluation
python run_all.py --stage explain        :: error analysis + Grad-CAM
python run_all.py --stage app            :: dashboard at http://127.0.0.1:5000
```
The split CSVs store a portable `rel_path`, so they resolve correctly on your
machine even though they were generated elsewhere.

## Error analysis & Grad-CAM
`results/error_analysis/` reports the most-confused class pairs, high-confidence
mistakes, and per-class error rates. `results/explainability/` contains Grad-CAM
overlays highlighting *possible visual factors* behind predictions.

## Web UI
A premium, dark, glassmorphism dashboard (`app/`) with drag-and-drop prediction,
top-3 probabilities, confidence + uncertainty warning, Grad-CAM, dataset insights,
model comparison, evaluation plots, and error analysis — all fed by real files.

---

## Project structure
```
air-quality-deep-learning/
├── dataset/                 # place the Kaggle dataset here (see README.txt)
├── src/                     # all pipeline modules
│   ├── config.py            # paths, seeds, hyper-parameters (single source of truth)
│   ├── utils.py             # logging, seeding, safe I/O, plotting
│   ├── env_audit.py         # Phase 1: OS/CPU/GPU/library audit
│   ├── dataset.py           # auto-discovery + unified records table
│   ├── dataset_analysis.py  # audit, label verification, EDA
│   ├── duplicates.py        # MD5 + perceptual-hash duplicate detection
│   ├── cleaning.py          # documented cleaning pipeline
│   ├── splits.py            # leakage-safe, group-aware, stratified split
│   ├── preprocessing.py     # tf.data pipelines + augmentation
│   ├── models.py            # backbone registry + head + fine-tune helpers
│   ├── screening.py         # short screening + final selection rule
│   ├── train.py             # two-stage transfer learning + experiment log
│   ├── evaluate.py          # ONE-shot test evaluation
│   ├── error_analysis.py    # error breakdown + representative errors
│   ├── explainability.py    # Grad-CAM
│   └── predict.py           # single-image inference
├── app/                     # Flask backend + premium 3D UI
├── results/                 # all generated artifacts (json/csv/png)
├── models/                  # saved models + selection metadata
├── notebooks/               # audit / EDA / training / evaluation notebooks
├── docs/viva_notes.md       # viva Q&A
├── run_all.py               # orchestrator
├── requirements.txt
├── README.md
└── walkthrough.md
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt   # for NVIDIA GPU: pip install "tensorflow[and-cuda]"
```

## Usage
```bash
# 1. Data prep: env audit, discovery, audit, label check, EDA, leakage-safe split
python run_all.py --stage audit

# 2. Short screening of all candidate architectures (frozen backbone, few epochs)
python run_all.py --stage screen

# 3. Full transfer learning + fine-tuning on the selected candidates, then select
python run_all.py --stage train

# 4. ONE-shot evaluation on the untouched test set
python run_all.py --stage evaluate

# 5. Error analysis + Grad-CAM
python run_all.py --stage explain

# …or run everything in order:
python run_all.py --stage all
```

### Prediction (single image)
```bash
python -m src.predict path/to/image.jpg --gradcam
```

### Launch the dashboard
```bash
python run_all.py --stage app        # http://127.0.0.1:5000
# or: cd app && python app.py
```

## Reproducibility
Fixed seed (`config.SEED = 42`) across Python/NumPy/TensorFlow; the split manifest,
class mapping, experiment log, best checkpoint, and final metrics are all saved.

## Limitations
- Image-based classification cannot measure pollutant concentrations.
- Predictions depend on visual similarity to the training distribution
  (geography, camera, weather, time of day).
- Softmax confidence is not a calibrated probability of correctness.

## Future scope
Fuse image features with sensor/AQI metadata; confidence calibration
(temperature scaling); quantization/pruning for on-device deployment;
larger and more geographically diverse data.

## Academic integrity
No metric in this repository is invented. Every number is computed from the real
data and saved to `results/`. The test set is used exactly once, after model
selection. Hard examples are never deleted to inflate scores.

## Security note
The dashboard is a **local** demo (binds to `127.0.0.1`, 12 MB upload cap, image
types only) with **no authentication**. Add auth, rate limiting, and hardened
file validation before exposing it to any network.
