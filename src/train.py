"""
Two-stage transfer-learning trainer (TensorFlow / Keras).

Stage 1: freeze the ImageNet backbone, train the compact head.
Stage 2: unfreeze the top of the backbone, fine-tune at a low LR.

Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint (best val restored).
Every run appends a row to results/experiments/experiment_results.csv with a
hypothesis/notes field, and validation (NOT test) metrics drive selection.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from . import config, utils, models, preprocessing as pp

log = utils.get_logger("train")

EXPERIMENT_COLUMNS = [
    "experiment", "model", "image_size", "augmentation", "class_weights",
    "frozen_backbone_stage1", "finetune_unfreeze", "learning_rate_head",
    "learning_rate_finetune", "optimizer", "epochs_total", "best_epoch",
    "val_accuracy", "val_macro_f1", "val_weighted_f1", "val_loss",
    "training_time_sec", "parameters", "trainable_parameters",
    "model_size_mb", "inference_latency_ms", "notes",
]


def _compile(model: keras.Model, lr: float, label_smoothing: float) -> None:
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=loss,
                  metrics=["accuracy"])


def _callbacks(ckpt: Path, cfg) -> list:
    return [
        keras.callbacks.ModelCheckpoint(
            str(ckpt), monitor="val_loss", save_best_only=True,
            save_weights_only=True, mode="min"),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.early_stop_patience,
            restore_best_weights=True, mode="min"),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=cfg.reduce_lr_patience,
            min_lr=1e-7, mode="min"),
    ]


def _val_metrics(model, val_ds, n_classes) -> Dict[str, float]:
    from sklearn.metrics import f1_score, accuracy_score
    y_true, y_pred = [], []
    for x, y in val_ds:
        p = model.predict(x, verbose=0)
        y_true.extend(np.argmax(y.numpy(), axis=1))
        y_pred.extend(np.argmax(p, axis=1))
    return {
        "val_accuracy": float(accuracy_score(y_true, y_pred)),
        "val_macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                       zero_division=0)),
        "val_weighted_f1": float(f1_score(y_true, y_pred, average="weighted",
                                          zero_division=0)),
    }


def _inference_latency_ms(model, image_size: int, n_warmup: int = 3,
                          n_runs: int = 20) -> float:
    """Mean single-image forward-pass latency in milliseconds (batch size 1).

    Warms up first (graph tracing / cuDNN autotune) so the measurement reflects
    steady-state inference rather than one-off compilation cost.
    """
    dummy = tf.zeros((1, image_size, image_size, 3), dtype=tf.float32)
    for _ in range(n_warmup):
        model.predict(dummy, verbose=0)
    start = time.time()
    for _ in range(n_runs):
        model.predict(dummy, verbose=0)
    return round((time.time() - start) / n_runs * 1000.0, 2)


def train_model(cfg, experiment_id: str, notes: str = "",
                screening: bool = False) -> Dict[str, Any]:
    """Run one full (or screening) training experiment and log the result."""
    config.ensure_dirs()
    utils.set_global_seed()
    cm = utils.load_json(config.CLASS_MAP_JSON)
    if not cm:
        raise RuntimeError("Run the split stage first (no class_mapping.json).")
    classes, class_to_idx = cm["classes"], cm["class_to_idx"]
    n_classes = len(classes)

    if cfg.mixed_precision:
        try:
            keras.mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            pass

    train_ds = pp.make_dataset(config.TRAIN_CSV, class_to_idx, cfg.model_name,
                               cfg.image_size, cfg.batch_size, training=True,
                               augment=cfg.augment)
    val_ds = pp.make_dataset(config.VAL_CSV, class_to_idx, cfg.model_name,
                             cfg.image_size, cfg.batch_size, training=False)

    class_weight = pp.compute_class_weights(class_to_idx) \
        if cfg.use_class_weights else None

    model = models.build_model(cfg.model_name, n_classes, cfg.image_size,
                               dropout=cfg.dropout,
                               weight_decay=cfg.weight_decay,
                               trainable_backbone=False)
    ckpt = config.MODELS_DIR / f"{experiment_id}_best.weights.h5"
    start = time.time()

    # ---- Stage 1: frozen backbone, train head ------------------------- #
    _compile(model, cfg.head_lr, cfg.label_smoothing)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=cfg.head_epochs,
                   class_weight=class_weight, callbacks=_callbacks(ckpt, cfg),
                   verbose=2)

    # ---- Stage 2: fine-tune top of backbone --------------------------- #
    h2 = None
    if not screening and cfg.finetune_epochs > 0:
        n_unfrozen = models.unfreeze_top(model, cfg.finetune_unfreeze)
        log.info("Fine-tuning: unfroze %d top backbone layers", n_unfrozen)
        _compile(model, cfg.finetune_lr, cfg.label_smoothing)
        h2 = model.fit(train_ds, validation_data=val_ds,
                       epochs=cfg.finetune_epochs, class_weight=class_weight,
                       callbacks=_callbacks(ckpt, cfg), verbose=2)

    elapsed = time.time() - start
    if ckpt.exists():
        model.load_weights(str(ckpt))  # restore BEST, not last epoch

    metrics = _val_metrics(model, val_ds, n_classes)
    val_loss = float(model.evaluate(val_ds, verbose=0)[0])
    best_epoch = _best_epoch(h1, h2)
    latency_ms = _inference_latency_ms(model, cfg.image_size)
    trainable_params = models.count_trainable_params(model)

    # Persist the full model for the final candidates.
    if not screening:
        model.save(config.MODELS_DIR / f"{experiment_id}.keras")

    result = {
        "experiment": experiment_id, "model": cfg.model_name,
        "image_size": cfg.image_size, "augmentation": cfg.augment,
        "class_weights": cfg.use_class_weights,
        "frozen_backbone_stage1": True,
        "finetune_unfreeze": 0 if screening else cfg.finetune_unfreeze,
        "learning_rate_head": cfg.head_lr,
        "learning_rate_finetune": cfg.finetune_lr, "optimizer": "adam",
        "epochs_total": len(h1.history["loss"]) +
        (len(h2.history["loss"]) if h2 else 0),
        "best_epoch": best_epoch,
        "val_accuracy": round(metrics["val_accuracy"], 4),
        "val_macro_f1": round(metrics["val_macro_f1"], 4),
        "val_weighted_f1": round(metrics["val_weighted_f1"], 4),
        "val_loss": round(val_loss, 4),
        "training_time_sec": round(elapsed, 1),
        "parameters": int(model.count_params()),
        "trainable_parameters": int(trainable_params),
        "model_size_mb": models.model_size_mb(model),
        "inference_latency_ms": latency_ms,
        "notes": notes or ("screening" if screening else "full"),
    }
    _log_experiment(result)
    _plot_history(h1, h2, experiment_id)
    utils.save_json(config.TRAINING_DIR / f"{experiment_id}_history.json",
                    {"stage1": h1.history,
                     "stage2": h2.history if h2 else None})
    log.info("[%s] val_acc=%.4f val_macroF1=%.4f (%.0fs)", experiment_id,
             result["val_accuracy"], result["val_macro_f1"], elapsed)
    return result


def _best_epoch(h1, h2) -> int:
    hist = (h2 or h1).history.get("val_loss", [0])
    return int(np.argmin(hist)) + 1


def _log_experiment(row: Dict[str, Any]) -> None:
    df_row = pd.DataFrame([row], columns=EXPERIMENT_COLUMNS)
    if config.EXPERIMENTS_CSV.exists():
        prev = pd.read_csv(config.EXPERIMENTS_CSV)
        out = pd.concat([prev, df_row], ignore_index=True)
    else:
        out = df_row
    out.to_csv(config.EXPERIMENTS_CSV, index=False)


def _plot_history(h1, h2, experiment_id: str) -> None:
    utils.apply_plot_style()
    import matplotlib.pyplot as plt

    def cat(key):
        v = list(h1.history.get(key, []))
        if h2:
            v += list(h2.history.get(key, []))
        return v

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(cat("accuracy"), label="train")
    a1.plot(cat("val_accuracy"), label="val")
    a1.set(title="Accuracy", xlabel="epoch"); a1.legend()
    a2.plot(cat("loss"), label="train")
    a2.plot(cat("val_loss"), label="val")
    a2.set(title="Loss", xlabel="epoch"); a2.legend()
    if h2:
        for ax in (a1, a2):
            ax.axvline(len(h1.history["loss"]) - 0.5, color="grey",
                       ls="--", alpha=0.6)
    fig.suptitle(f"Training curves - {experiment_id}")
    fig.savefig(config.TRAINING_DIR / f"{experiment_id}_curves.png")
    plt.close(fig)


if __name__ == "__main__":
    cfg = config.TrainConfig()
    train_model(cfg, experiment_id="manual_run")
