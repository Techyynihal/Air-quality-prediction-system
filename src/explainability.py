"""
Grad-CAM explainability for the final CNN model.

Produces Original -> Heatmap -> Overlay triptychs. The last 4-D convolutional
feature map is discovered automatically (models.last_conv_layer_name). Used for
test examples, representative errors, and user-uploaded images (via the app).

Note: Grad-CAM highlights image regions that most influenced the prediction.
We describe these as "possible visual factors", never as proven causes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import config, utils, models
from . import preprocessing as pp

log = utils.get_logger("explainability")


def compute_heatmap(model: keras.Model, img_batch: tf.Tensor,
                    last_conv: str, pred_index: Optional[int] = None
                    ) -> np.ndarray:
    """Return a [0,1] Grad-CAM heatmap for a single preprocessed image batch."""
    conv_layer = model.backbone.get_layer(last_conv)
    grad_model = keras.Model(model.inputs,
                             [conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_batch)
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def _colorize_and_overlay(orig_rgb: np.ndarray, heatmap: np.ndarray,
                          alpha: float = 0.4) -> np.ndarray:
    import matplotlib.cm as cm
    from PIL import Image
    h, w = orig_rgb.shape[:2]
    hm = np.array(Image.fromarray(np.uint8(255 * heatmap)).resize((w, h)))
    colored = cm.get_cmap("jet")(hm / 255.0)[..., :3]
    overlay = (1 - alpha) * (orig_rgb / 255.0) + alpha * colored
    return np.uint8(255 * np.clip(overlay, 0, 1))


def generate_for_image(model: keras.Model, image_path: str, model_name: str,
                       image_size: int, out_path: Path,
                       class_index: Optional[int] = None) -> Path:
    """Save an Original | Heatmap | Overlay figure for one image."""
    utils.apply_plot_style()
    import matplotlib.pyplot as plt
    from PIL import Image

    preprocess_fn = models.get_preprocess_fn(model_name)
    batch = pp.decode_and_preprocess(tf.constant(image_path), image_size,
                                     preprocess_fn)[tf.newaxis, ...]
    last_conv = models.last_conv_layer_name(model)
    heatmap = compute_heatmap(model, batch, last_conv, class_index)

    orig = np.array(Image.open(image_path).convert("RGB").resize(
        (image_size, image_size)))
    overlay = _colorize_and_overlay(orig, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, img, title in zip(
            axes, [orig, heatmap, overlay],
            ["Original", "Grad-CAM heatmap", "Overlay"]):
        ax.imshow(img, cmap="jet" if title.endswith("heatmap") else None)
        ax.set_title(title); ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def generate_test_examples(n: int = 8) -> None:
    """Grad-CAM for a mix of correct + misclassified test images."""
    import pandas as pd
    from .evaluate import TEST_PRED_CSV, _load_final

    if not TEST_PRED_CSV.exists():
        log.warning("No test predictions yet; run evaluate first.")
        return
    model, meta = _load_final()
    preds = pd.read_csv(TEST_PRED_CSV)
    wrong = preds[preds["true_idx"] != preds["pred_idx"]].head(n // 2)
    right = preds[preds["true_idx"] == preds["pred_idx"]].head(n - len(wrong))
    for tag, subset in (("correct", right), ("error", wrong)):
        for i, (_, row) in enumerate(subset.iterrows()):
            out = config.EXPLAIN_DIR / f"gradcam_{tag}_{i}.png"
            try:
                generate_for_image(model, row["path"], meta["selected_model"],
                                   meta["image_size"], out,
                                   int(row["pred_idx"]))
            except Exception as exc:
                log.warning("Grad-CAM failed for %s: %s", row["path"], exc)
    log.info("Grad-CAM examples written to %s", config.EXPLAIN_DIR)


if __name__ == "__main__":
    generate_test_examples()
