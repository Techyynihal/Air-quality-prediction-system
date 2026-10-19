# Project Walkthrough — Air Quality Image Classification

This walkthrough explains the project end-to-end for a **B.Tech presentation,
viva, and report**, and — importantly — *why* each component exists.

---

## 1. The idea in one line
Teach a CNN to look at an outdoor photo and predict its **air-quality category**,
while being scientifically honest that an image classifies *visual conditions*
and does not *measure* pollutants.

## 2. Why deep learning / CNNs?
Air-quality visual cues (haze density, contrast loss, sky color, visibility of
distant objects) are subtle, spatial, and hard to hand-engineer. Convolutional
neural networks learn hierarchical spatial features automatically, which is
exactly what this problem needs.

## 3. Why transfer learning (and not training from scratch)?
Our dataset is far smaller than ImageNet. A backbone pretrained on ImageNet
already knows edges, textures, and object parts. We reuse those features and only
learn the air-quality-specific mapping — this gives higher accuracy, faster
training, and less overfitting than training from scratch.

## 4. The pipeline, stage by stage

### 4.1 Environment audit (`env_audit.py`)
*Why:* we must adapt to the real machine (GPU vs CPU, RAM, libraries) instead of
assuming. It also decides safe settings such as enabling mixed precision only if
a GPU exists.

### 4.2 Discovery (`dataset.py`)
*Why:* the dataset layout is not assumed. We scan for images and CSVs, decide
whether labels come from folders or a CSV column, and build one **records table**
(`path, label, group_id, …`) used everywhere.

### 4.3 Dataset audit + label verification + EDA (`dataset_analysis.py`)
*Why:* you cannot fix what you have not measured. We compute class balance,
resolution/format statistics, corrupted files, and verify what each class means
(including checking any AQI columns against labels). EDA plots make the data
legible for the report.

### 4.4 Duplicates & near-duplicates (`duplicates.py`)
*Why:* identical or almost-identical photos inflate scores if split carelessly.
MD5 catches exact copies; perceptual hashing (pHash/dHash) + union-find groups
near-duplicates.
*On the real data:* the audit found **129 exact-duplicate copies** (removed) and
many near-duplicate clusters. Because transitive perceptual-hash chaining over
low-detail hazy scenes can merge thousands of genuinely distinct images into one
cluster, near-duplicates are **reported but not merged into leakage groups**;
same-session leakage is instead handled precisely by the metadata group key
(location + date + hour). Exact (MD5) duplicates are always merged.

### 4.5 Cleaning (`cleaning.py`)
*Why:* remove only genuinely unusable files (corrupted, wrong format, invalid
label) and redundant exact copies — **with a documented reason for each**. Hard
and near-duplicate images are *kept* so metrics stay honest.

### 4.6 Leakage-safe split (`splits.py`)
*Why:* this is the single most important guardrail. Duplicate groups + metadata
groups (same city/location/session) are merged into **leakage groups**; whole
groups go to exactly one split. The manifest verifies **zero group overlap**, so
the test score reflects genuine generalization. Split is 70/15/15, stratified,
seed-fixed.
*On the real data:* from 12,111 kept images (12,240 − 129 exact dups) across 115
metadata sessions, the split is **train 8,434 / val 1,862 / test 1,815** with
`"zero_leakage": true` (all pairwise group overlaps = 0) and all 6 classes present
in every split.

### 4.7 Preprocessing (`preprocessing.py`)
*Why:* consistent, backbone-correct input. Training gets realistic, mild
augmentation to improve robustness without destroying pollution cues; val/test
are deterministic. Inference reuses the same function, so there is no
train/serve skew.

### 4.8 Screening → selection → training (`screening.py`, `train.py`)
*Why:* training every architecture fully is wasteful. We *screen* all candidates
cheaply (frozen backbone, few epochs, identical protocol), keep the best by
validation Macro-F1 with an efficiency tie-break, then fully fine-tune only those.
Every run is logged to `experiment_results.csv` with hypothesis/notes.

### 4.9 Final selection (`screening.select_final_model`)
*Why:* the choice must be evidence-based and reproducible. Rule: highest
validation Macro-F1; ties within 0.01 broken toward the smaller model. **The test
set is never used to choose.**

### 4.10 One-shot test evaluation (`evaluate.py`)
*Why:* to get an unbiased estimate, the untouched test set is evaluated exactly
once, producing the full metric suite, confusion matrices, and the
generalization gap.

### 4.11 Error analysis + Grad-CAM (`error_analysis.py`, `explainability.py`)
*Why:* metrics tell you *how much*; these tell you *where and why*. We surface the
most-confused class pairs and visualize the regions that drove predictions
(described as *possible* factors, not proven causes).

### 4.12 Prediction + UI (`predict.py`, `app/`)
*Why:* a demo-ready product. Upload an image → class, confidence, top-3,
distribution, uncertainty warning, Grad-CAM. All dashboard metrics load from real
result files; before training they read "Not evaluated yet".

## 5. How honesty is enforced
- Test set touched once, after selection.
- No fabricated numbers anywhere; UI reads generated files.
- Leakage verified in the split manifest.
- Removed files are individually justified.

## 6. Methodology flow (for slides)
`Audit → Discover → Clean → Dedup → Leakage-safe split → EDA →
Screen → Fine-tune best → Select (val) → Test once → Explain → Deploy UI`

## 7. What to demo live
1. `results/audit/dataset_audit.md` (real dataset facts).
2. `results/splits/split_manifest.json` (zero leakage).
3. `results/experiments/experiment_results.csv` (controlled experiments).
4. `results/evaluation/` (confusion matrix, per-class F1).
5. The dashboard: upload an image, show Grad-CAM.
