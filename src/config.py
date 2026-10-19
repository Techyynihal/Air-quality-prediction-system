"""
Central configuration for the Air Quality Image Classification project.

Everything that another module might need to agree on (paths, seeds,
hyper-parameters, the class list produced during splitting) lives here so the
pipeline stays reproducible and there is a single source of truth.

Nothing here assumes the dataset's structure, class names, or size. Those are
DISCOVERED from the real files (see src/dataset.py) and any values that depend
on them are loaded at runtime from generated artifacts under results/.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #
# Project root = parent of this file's parent (…/air-quality-deep-learning).
ROOT = Path(__file__).resolve().parents[1]

# Base folder where the user drops the (possibly nested / zipped-then-extracted)
# Kaggle download. The *actual* image root inside it is auto-resolved below so we
# never assume the archive's internal layout and never count redundant copies.
DATASET_BASE = ROOT / "dataset"


# AQI class folders look like "a_Good", "b_Moderate", ... "f_Severe".
_CLASS_FOLDER_RE = re.compile(r"^[a-z]_[A-Za-z_]+$")


def _dir_has_class_folders(d: Path) -> int:
    """Count immediate sub-folders that look like AQI class folders."""
    try:
        return sum(1 for c in d.iterdir()
                   if c.is_dir() and _CLASS_FOLDER_RE.match(c.name))
    except (OSError, PermissionError):
        return 0


def find_dataset_root(base: Path = DATASET_BASE) -> Path:
    """Resolve the single canonical image root inside the dataset download.

    The Kaggle "Air Pollution Image Dataset from India and Nepal" ships THREE
    redundant copies of the same 12,240 images:
        Combined_Dataset/All_img        (flat, unlabeled)
        Combined_Dataset/IND_and_NEP    (class folders only)
        Country_wise_Dataset/<country>/<city>/<class>   (class + location)
    We must pick exactly ONE, otherwise every image is counted 2-3x and leakage
    is guaranteed. We prefer `Country_wise_Dataset` because it also encodes the
    city/location needed for group-aware, leakage-safe splitting.

    Resolution order (all auto-detected, nothing hard-coded to the odd
    "archive (1)/..." nesting):
      1. A directory literally named `Country_wise_Dataset` (best: has location).
      2. Otherwise the directory with the most AQI-style class sub-folders.
      3. Otherwise `base` itself (generic fallback for other layouts).
    """
    base = Path(base)
    if not base.exists():
        return base

    # 1. Prefer a Country_wise_Dataset directory that actually contains images.
    country_dirs = [d for d in base.rglob("Country_wise_Dataset") if d.is_dir()]
    country_dirs = [d for d in country_dirs
                    if any(d.rglob("*.jpg")) or any(d.rglob("*.jpeg"))]
    if country_dirs:
        return sorted(country_dirs, key=lambda p: len(p.parts))[0]

    # 2. Fall back to whichever directory directly holds the most class folders.
    best_dir, best_n = None, 0
    for d in [base, *[p for p in base.rglob("*") if p.is_dir()]]:
        n = _dir_has_class_folders(d)
        if n > best_n:
            best_dir, best_n = d, n
    if best_dir is not None and best_n >= 2:
        return best_dir

    # 3. Generic fallback.
    return base


# The resolved image root every downstream stage should read from.
DATASET_DIR = find_dataset_root(DATASET_BASE)
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

AUDIT_DIR = RESULTS_DIR / "audit"
EDA_DIR = RESULTS_DIR / "eda"
SPLITS_DIR = RESULTS_DIR / "splits"
TRAINING_DIR = RESULTS_DIR / "training"
EVAL_DIR = RESULTS_DIR / "evaluation"
ERROR_DIR = RESULTS_DIR / "error_analysis"
EXPLAIN_DIR = RESULTS_DIR / "explainability"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"

# Key artifact files (used across modules AND by the web app).
ENV_AUDIT_JSON = AUDIT_DIR / "environment_audit.json"
DATASET_AUDIT_JSON = AUDIT_DIR / "dataset_audit.json"
DATASET_AUDIT_MD = AUDIT_DIR / "dataset_audit.md"
CLEANING_REPORT_JSON = AUDIT_DIR / "cleaning_report.json"
DUPLICATES_JSON = AUDIT_DIR / "duplicates_report.json"
LABEL_REPORT_MD = AUDIT_DIR / "label_verification.md"
DISCOVERY_JSON = AUDIT_DIR / "dataset_discovery.json"

CLASS_MAP_JSON = SPLITS_DIR / "class_mapping.json"
TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV = SPLITS_DIR / "val.csv"
TEST_CSV = SPLITS_DIR / "test.csv"
SPLIT_MANIFEST_JSON = SPLITS_DIR / "split_manifest.json"

EXPERIMENTS_CSV = EXPERIMENTS_DIR / "experiment_results.csv"
FINAL_METRICS_JSON = EVAL_DIR / "final_metrics.json"
MODEL_SELECTION_JSON = EXPERIMENTS_DIR / "model_selection.json"
FINAL_MODEL_META_JSON = MODELS_DIR / "final_model_meta.json"

ALL_RESULT_DIRS = [
    AUDIT_DIR, EDA_DIR, SPLITS_DIR, TRAINING_DIR, EVAL_DIR,
    ERROR_DIR, EXPLAIN_DIR, EXPERIMENTS_DIR, MODELS_DIR,
]

# --------------------------------------------------------------------------- #
# Reproducibility                                                              #
# --------------------------------------------------------------------------- #
SEED = 42

# --------------------------------------------------------------------------- #
# Image / data settings                                                        #
# --------------------------------------------------------------------------- #
IMAGE_SIZE = 224          # default square input; screening may probe others
CHANNELS = 3
SPLIT_RATIOS = (0.70, 0.15, 0.15)   # train / val / test

# Files we treat as images during discovery.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Thresholds used by the audit (documented, not magic).
SMALL_IMAGE_MAX = 64      # <= this on either side is flagged "very small"
LARGE_IMAGE_MIN = 4000    # >= this on either side is flagged "very large"
PHASH_HAMMING_NEAR_DUP = 5  # perceptual-hash distance <= this => near-duplicate


@dataclass
class TrainConfig:
    """Hyper-parameters for a single training run / experiment."""
    model_name: str = "efficientnetb0"
    image_size: int = IMAGE_SIZE
    batch_size: int = 32
    head_epochs: int = 15          # stage 1: frozen backbone
    finetune_epochs: int = 25      # stage 2: unfrozen top
    head_lr: float = 1e-3
    finetune_lr: float = 1e-5
    finetune_unfreeze: int = 40    # number of top backbone layers to unfreeze
    dropout: float = 0.3
    weight_decay: float = 1e-5
    label_smoothing: float = 0.05
    use_class_weights: bool = True
    augment: bool = True
    early_stop_patience: int = 6
    reduce_lr_patience: int = 3
    mixed_precision: bool = False  # set by env audit when GPU supports it

    def to_dict(self) -> dict:
        return asdict(self)


# Candidate models for screening (see src/models.py for the builders).
# Tiered exactly as required: efficient -> strong -> academic baseline.
CANDIDATE_MODELS = [
    "mobilenetv3large",   # Tier 1 - efficient
    "efficientnetb0",     # Tier 1 - efficient
    "efficientnetb1",     # Tier 1 - efficient
    "resnet50",           # Tier 2 - strong
    "convnexttiny",       # Tier 2 - strong
    "efficientnetv2s",    # Tier 2 - strong
    "vgg16",              # Tier 3 - academic comparison only
]


def ensure_dirs() -> None:
    """Create every results/model directory if missing (idempotent)."""
    for d in ALL_RESULT_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def load_class_names() -> Optional[List[str]]:
    """Return the ordered class list produced during splitting, or None."""
    if CLASS_MAP_JSON.exists():
        data = json.loads(CLASS_MAP_JSON.read_text(encoding="utf-8"))
        # stored as {"classes": [...], "class_to_idx": {...}}
        return data.get("classes")
    return None


if __name__ == "__main__":
    ensure_dirs()
    print(f"Project root: {ROOT}")
    print(f"Results dirs ensured under: {RESULTS_DIR}")
    print(f"Classes (if split done): {load_class_names()}")
