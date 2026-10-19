"""
Final test-set evaluation - run ONCE, after the model is finalised.

Loads the selected model, evaluates the UNTOUCHED test split, and reports the
full metric suite (accuracy, macro/weighted precision-recall-F1, per-class
metrics, confusion matrix + normalised version) plus the train/val->test
generalization gap. Saves machine- and human-readable outputs and the raw
test predictions used by error analysis and Grad-CAM.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from . import config, utils
from . import preprocessing as pp

log = utils.get_logger("evaluate")

TEST_PRED_CSV = config.EVAL_DIR / "test_predictions.csv"


def _load_final():
    meta = utils.load_json(config.FINAL_MODEL_META_JSON)
    if not meta:
        raise RuntimeError("No final model selected. Run screening.select_final_model().")
    model = keras.models.load_model(meta["model_path"])
    return model, meta


def _predict_split(model, csv_path, class_to_idx, model_name, image_size,
                   batch_size=32):
    ds = pp.make_dataset(csv_path, class_to_idx, model_name, image_size,
                         batch_size, training=False)
    probs = model.predict(ds, verbose=0)
    df = pd.read_csv(csv_path)
    y_true = df["label"].astype(str).map(class_to_idx).to_numpy()
    return df, y_true, probs


def evaluate_final() -> Dict[str, Any]:
    """Evaluate the untouched test set and persist the full metric suite."""
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                  classification_report, confusion_matrix,
                                  f1_score)
    config.ensure_dirs()
    model, meta = _load_final()
    cm = utils.load_json(config.CLASS_MAP_JSON)
    classes, class_to_idx = cm["classes"], cm["class_to_idx"]
    name, size = meta["selected_model"], meta["image_size"]

    df, y_true, probs = _predict_split(model, config.TEST_CSV, class_to_idx,
                                       name, size)
    y_pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    acc = accuracy_score(y_true, y_pred)
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=classes,
                                   output_dict=True, zero_division=0)
    cmat = confusion_matrix(y_true, y_pred)

    # Generalization gap vs the selected model's validation Macro-F1.
    gap = round(float(meta.get("val_macro_f1", np.nan)) - float(mf), 4)

    metrics = {
        "model": name, "image_size": size,
        "n_test_images": int(len(df)),
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(mp), 4),
        "macro_recall": round(float(mr), 4),
        "macro_f1": round(float(mf), 4),
        "weighted_precision": round(float(wp), 4),
        "weighted_recall": round(float(wr), 4),
        "weighted_f1": round(float(wf), 4),
        "generalization_gap_valF1_minus_testF1": gap,
        "per_class": {c: {k: round(report[c][k], 4)
                          for k in ("precision", "recall", "f1-score")}
                      for c in classes},
        "confusion_matrix": cmat.tolist(),
        "classes": classes,
        "note": "Single, final evaluation on the untouched test split.",
    }
    utils.save_json(config.FINAL_METRICS_JSON, metrics)
    pd.DataFrame(report).transpose().to_csv(
        config.EVAL_DIR / "classification_report.csv")

    # Save raw predictions for error analysis / Grad-CAM.
    out = df.copy()
    out["true_idx"] = y_true
    out["pred_idx"] = y_pred
    out["true_label"] = [classes[i] for i in y_true]
    out["pred_label"] = [classes[i] for i in y_pred]
    out["confidence"] = conf
    out.to_csv(TEST_PRED_CSV, index=False)

    _plot_confusion(cmat, classes)
    _write_summary(metrics)
    log.info("TEST accuracy=%.4f macroF1=%.4f (gap=%.4f)",
             acc, mf, gap)
    return metrics


def _plot_confusion(cmat, classes) -> None:
    utils.apply_plot_style()
    import matplotlib.pyplot as plt

    for norm in (False, True):
        m = cmat.astype(float)
        if norm:
            m = m / m.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(1.2 * len(classes) + 3,
                                        1.0 * len(classes) + 3))
        im = ax.imshow(m, cmap="Blues")
        ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)
        ax.set(xlabel="Predicted", ylabel="Actual",
               title="Normalized confusion matrix" if norm
               else "Confusion matrix")
        thr = m.max() / 2
        for i in range(len(classes)):
            for j in range(len(classes)):
                txt = f"{m[i, j]:.2f}" if norm else f"{int(cmat[i, j])}"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if m[i, j] > thr else "black",
                        fontsize=8)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        fname = "confusion_matrix_normalized.png" if norm \
            else "confusion_matrix.png"
        fig.savefig(config.EVAL_DIR / fname)
        plt.close(fig)


def _write_summary(m: Dict[str, Any]) -> None:
    lines = [
        "# Final Test Evaluation", "",
        f"- **Model:** {m['model']}  |  **Test images:** {m['n_test_images']}",
        f"- **Accuracy:** {m['accuracy']}",
        f"- **Macro F1:** {m['macro_f1']}  |  **Weighted F1:** {m['weighted_f1']}",
        f"- **Macro P/R:** {m['macro_precision']} / {m['macro_recall']}",
        f"- **Generalization gap (valF1 - testF1):** "
        f"{m['generalization_gap_valF1_minus_testF1']}", "",
        "## Per-class metrics", "",
        "| Class | Precision | Recall | F1 |", "| --- | ---: | ---: | ---: |",
    ]
    for c, v in m["per_class"].items():
        lines.append(f"| {c} | {v['precision']} | {v['recall']} | {v['f1-score']} |")
    lines += ["", "> Single final evaluation on the untouched test set.", ""]
    (config.EVAL_DIR / "evaluation_summary.md").write_text(
        "\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    evaluate_final()
