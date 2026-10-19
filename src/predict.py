"""
Single-image prediction pipeline.

Given one RGB image, returns the predicted air-quality class, confidence, the
full probability distribution, top-3 predictions, a plain-language description,
an uncertainty warning when confidence is low, and (optionally) a Grad-CAM
overlay path.

Class descriptions are generic, keyword-matched to common AQI category names in
the dataset. They describe the CATEGORY, not a measured pollutant value - a
visual classifier does not physically measure PM2.5/PM10/etc.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import config, utils, models
from . import preprocessing as pp

log = utils.get_logger("predict")

# Confidence below this => surface an uncertainty warning to the user.
DEFAULT_UNCERTAINTY_THRESHOLD = 0.55

# Generic, non-fabricated descriptions for common AQI category keywords.
_CATEGORY_HINTS = {
    "good": "Air quality is considered satisfactory; air pollution poses little or no risk.",
    "moderate": "Air quality is acceptable; some pollutants may be a concern for a very small number of sensitive individuals.",
    "sensitive": "Members of sensitive groups may experience health effects; the general public is less likely to be affected.",
    "unhealthy": "Some members of the general public may experience health effects; sensitive groups may experience more serious effects.",
    "very": "Health alert: the risk of health effects is increased for everyone.",
    "severe": "Health warning of emergency conditions; everyone is more likely to be affected.",
    "hazardous": "Health warning of emergency conditions; the entire population is likely to be affected.",
    "poor": "Air quality is poor and may cause health effects for many people.",
}


def describe_class(class_name: str) -> str:
    key = class_name.lower()
    for hint, text in _CATEGORY_HINTS.items():
        if hint in key:
            return text
    return ("Predicted air-quality category from the trained image classifier. "
            "See dataset documentation for this category's definition.")


@lru_cache(maxsize=1)
def load_predictor():
    """Load the final model + metadata once (cached)."""
    meta = utils.load_json(config.FINAL_MODEL_META_JSON)
    if not meta:
        raise RuntimeError("No final model available. Train + select first.")
    model = keras.models.load_model(meta["model_path"])
    cm = utils.load_json(config.CLASS_MAP_JSON)
    return model, meta, cm["classes"]


def predict_image(image_path: str, with_gradcam: bool = False,
                  gradcam_out: Optional[str] = None,
                  threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD
                  ) -> Dict[str, Any]:
    """Predict the air-quality class for a single image."""
    model, meta, classes = load_predictor()
    name, size = meta["selected_model"], meta["image_size"]
    preprocess_fn = models.get_preprocess_fn(name)

    batch = pp.decode_and_preprocess(tf.constant(image_path), size,
                                     preprocess_fn)[tf.newaxis, ...]
    probs = model.predict(batch, verbose=0)[0].astype(float)
    order = np.argsort(probs)[::-1]
    top_idx = int(order[0])
    confidence = float(probs[top_idx])

    result: Dict[str, Any] = {
        "image": Path(image_path).name,
        "predicted_class": classes[top_idx],
        "confidence": round(confidence, 4),
        "description": describe_class(classes[top_idx]),
        "top3": [{"class": classes[int(i)], "probability": round(float(probs[i]), 4)}
                 for i in order[:3]],
        "distribution": {classes[i]: round(float(probs[i]), 4)
                         for i in range(len(classes))},
        "uncertain": confidence < threshold,
        "uncertainty_note": (
            "Low confidence - treat this prediction with caution."
            if confidence < threshold else ""),
        "disclaimer": ("Image-based air-quality CLASSIFICATION; does not "
                       "physically measure pollutant concentrations."),
    }

    if with_gradcam:
        from . import explainability as xai
        out = Path(gradcam_out or (config.EXPLAIN_DIR / "prediction_gradcam.png"))
        try:
            xai.generate_for_image(model, image_path, name, size, out, top_idx)
            result["gradcam_path"] = str(out)
        except Exception as exc:
            result["gradcam_error"] = str(exc)
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Predict air-quality class.")
    ap.add_argument("image")
    ap.add_argument("--gradcam", action="store_true")
    args = ap.parse_args()
    import json
    print(json.dumps(predict_image(args.image, with_gradcam=args.gradcam),
                     indent=2))
