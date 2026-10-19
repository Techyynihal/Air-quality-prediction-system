"""
Preprocessing + tf.data input pipelines.

- Deterministic decode/resize/backbone-preprocess for ALL splits.
- Realistic, mild augmentation applied to TRAIN ONLY (flip, small rotation,
  small zoom/translation, mild contrast/brightness) - nothing that destroys
  pollution-related visual cues (no extreme rotations or color inversions).
- Caching + prefetch for throughput.

Inference preprocessing (predict.py) reuses `decode_and_preprocess` so it
matches training exactly.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from . import config, utils
from . import models

log = utils.get_logger("preprocessing")
AUTOTUNE = tf.data.AUTOTUNE


def build_augmenter() -> tf.keras.Sequential:
    """Realistic, conservative augmentation stack."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.06),       # ~ +/-21 deg (0.06 of 360)
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomTranslation(0.06, 0.06),
        tf.keras.layers.RandomContrast(0.10),
        tf.keras.layers.RandomBrightness(0.10, value_range=(0, 255)),
    ], name="augmenter")


def decode_and_preprocess(path: tf.Tensor, image_size: int,
                          preprocess_fn: Callable) -> tf.Tensor:
    """Read -> RGB -> resize -> backbone preprocess. Deterministic."""
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [image_size, image_size], method="bilinear")
    img = tf.cast(img, tf.float32)
    return preprocess_fn(img)


def _resolve_path(row) -> str:
    """Return a usable absolute path, portable across machines.

    Split CSVs store both an absolute ``path`` (valid on the machine that
    generated them) and a project-root-relative ``rel_path``. We prefer
    whichever actually exists on THIS machine, resolving ``rel_path`` against
    the current project ROOT. This lets a split generated on one computer be
    consumed on another (e.g. audited in a sandbox, trained on a local GPU).
    """
    import os
    p = row.get("path")
    if isinstance(p, str) and os.path.exists(p):
        return p
    rel = row.get("rel_path")
    if isinstance(rel, str) and rel:
        cand = os.path.join(str(config.ROOT), rel)
        if os.path.exists(cand):
            return cand
    return p


def _load_split_csv(csv_path, class_to_idx) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(csv_path)
    if "rel_path" in df.columns:
        paths = df.apply(_resolve_path, axis=1).tolist()
    else:
        paths = df["path"].tolist()
    labels = df["label"].astype(str).map(class_to_idx).to_numpy()
    return paths, labels


def make_dataset(csv_path, class_to_idx, model_name: str, image_size: int,
                 batch_size: int, training: bool,
                 augment: bool = True) -> tf.data.Dataset:
    """Build a batched tf.data pipeline for one split CSV."""
    preprocess_fn = models.get_preprocess_fn(model_name)
    paths, labels = _load_split_csv(csv_path, class_to_idx)
    num_classes = len(class_to_idx)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(paths), 2048), seed=config.SEED,
                        reshuffle_each_iteration=True)

    def _map(path, label):
        img = decode_and_preprocess(path, image_size, preprocess_fn)
        onehot = tf.one_hot(label, num_classes)
        return img, onehot

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)

    if training and augment:
        augmenter = build_augmenter()
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    return ds.prefetch(AUTOTUNE)


def compute_class_weights(class_to_idx) -> dict:
    """Inverse-frequency class weights from the TRAIN split only."""
    df = pd.read_csv(config.TRAIN_CSV)
    counts = df["label"].astype(str).value_counts().to_dict()
    total = sum(counts.values())
    n_classes = len(class_to_idx)
    weights = {}
    for cls, idx in class_to_idx.items():
        c = counts.get(cls, 0)
        weights[idx] = (total / (n_classes * c)) if c else 1.0
    return weights


def steps_per_epoch(csv_path, batch_size: int) -> int:
    return int(np.ceil(len(pd.read_csv(csv_path)) / batch_size))
