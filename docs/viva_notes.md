# Viva Notes — Air Quality Image Classification

Short, clear answers to the questions examiners usually ask.

**What is air-quality image classification?**
Predicting a discrete air-quality *category* for an outdoor scene from a single
RGB image. The model learns visual cues (haze, contrast, color) that correlate
with the dataset's ground-truth air-quality labels.

**Why deep learning here?**
The relevant cues are subtle and spatial. Deep CNNs learn these features
automatically instead of us hand-crafting them.

**Why a CNN specifically?**
Convolutions exploit locality and translation-invariance in images and build up
from edges → textures → scene-level features, which suits visual haze/pollution
patterns.

**Why transfer learning?**
Our dataset is small relative to ImageNet. A pretrained backbone already encodes
generic visual features, so we get better accuracy, faster training, and less
overfitting than training from scratch.

**Why compare VGG16, EfficientNet, MobileNet, etc.?**
- **VGG16**: classic, simple, large and slow — a good *academic baseline*.
- **EfficientNet (B0/B1/V2-S)**: compound-scaled; excellent accuracy per FLOP.
- **MobileNetV3**: designed for efficiency/mobile; strong accuracy/size trade-off.
- **ResNet50 / ConvNeXt-Tiny**: strong, well-understood mid-size models.
We screen them all and let evidence pick the winner — VGG16 is not allowed to
win automatically.

**Why data augmentation?**
It synthetically enlarges the training set with realistic variations (flip, small
rotation/zoom, mild brightness/contrast), improving robustness and reducing
overfitting. We avoid extreme transforms that would destroy pollution cues.

**Why class weights?**
If classes are imbalanced, the model can ignore minority classes. Weighting the
loss by inverse frequency makes minority-class errors cost more. We confirm the
benefit using Macro-F1.

**Why train/validation/test?**
Train fits parameters, validation tunes choices (model, hyper-parameters, early
stopping), and test gives a final unbiased estimate. Mixing these roles leaks
information and inflates results.

**What is data leakage?**
When information from the test set influences training/selection — e.g. a
duplicate or same-scene image appears in both train and test. It makes scores
look better than reality. We prevent it with group-aware splitting.

**What is overfitting / underfitting?**
Overfitting: the model memorizes training data and generalizes poorly (train ≫
val). Underfitting: the model is too weak to capture the pattern (both low). We
watch the train–val gap and use dropout, augmentation, early stopping, and
fine-tuning depth to balance them.

**Why Macro-F1 (not just accuracy)?**
Accuracy is dominated by frequent classes. Macro-F1 averages per-class F1 equally,
so minority-class performance matters — crucial for imbalanced data.

**Why a confusion matrix?**
It shows *which* classes are confused with which, not just the overall rate —
guiding targeted improvements (e.g. adjacent AQI categories).

**What is Grad-CAM?**
Gradient-weighted Class Activation Mapping: it uses gradients flowing into the
last conv layer to produce a heatmap of image regions that most influenced the
predicted class — a sanity check on *where* the model looks.

**Why not a 3D CNN?**
"3D" in this project refers only to the **UI's visual depth**. The data are 2D
images, so the model is a 2D CNN. A 3D CNN is for volumetric/temporal data (e.g.
video, MRI) and is unnecessary and wasteful here.

**Limitations of image-based air-quality prediction?**
- Cannot measure actual pollutant concentrations.
- Sensitive to camera, weather, lighting, and geography.
- Softmax confidence isn't a calibrated correctness probability.

**Future improvements?**
Multimodal fusion with sensor/AQI data, confidence calibration, model
quantization for deployment, and a larger, more diverse dataset.

---

### Dataset-specific facts (this project)
- **Dataset:** Air Pollution Image Dataset from India and Nepal (Kaggle).
- **Classes (6):** a_Good, b_Moderate, c_Unhealthy_for_Sensitive_Groups,
  d_Unhealthy, e_Very_Unhealthy, f_Severe.
- **Size:** 12,240 images, all 224×224 RGB JPEG, 0 corrupted; imbalance ratio ≈ 1.98.
- **Redundant copies:** the download ships the same images three times
  (`All_img`, `IND_and_NEP`, `Country_wise_Dataset`). We use only
  `Country_wise_Dataset` as the canonical root because it also encodes the city,
  which strengthens leakage grouping. Using all three would triple-count and
  guarantee leakage.
- **Duplicates:** 129 exact duplicates removed; near-duplicates kept but excluded
  from leakage grouping (they'd transitively over-merge low-detail hazy scenes).
- **Leakage-safe split:** train 8,434 / val 1,862 / test 1,815, verified zero
  group overlap (`results/splits/split_manifest.json`).
- **Grouping key:** Location + Year + Month + Day + Hour — so burst frames from
  the same capture session never straddle splits.

**Q: If a camera can't measure PM2.5, what is the model actually learning?**
It learns the visual signature of each AQI category — haze density, contrast loss,
sky/scene color — which correlates with (but does not measure) pollutant levels.
The AQI/PM columns in the CSVs are used only to verify labels, never as inputs.

---

### One-line methodology (for the board)
Audit → Discover → Clean → Deduplicate → Leakage-safe split → EDA → Screen →
Fine-tune best → Select on validation → Evaluate test once → Explain → UI.
