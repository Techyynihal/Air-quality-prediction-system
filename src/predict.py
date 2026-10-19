
"""
Single-image prediction pipeline for the Air Quality Image Classifier.

Supports local Windows paths and cloud deployment paths.
Returns the predicted class, confidence, probability distribution,
top-3 predictions, category description, uncertainty warning,
and optional Grad-CAM output.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import config, utils, models
from . import preprocessing as pp

log = utils.get_logger("predict")

DEFAULT_UNCERTAINTY_THRESHOLD = 0.55

_CATEGORY_HINTS = {
    "good": (
        "Air quality is considered satisfactory; "
        "air pollution poses little or no risk."
    ),
    "moderate": (
        "Air quality is acceptable; some pollutants may be a concern "
        "for a very small number of sensitive individuals."
    ),
    "sensitive": (
        "Members of sensitive groups may experience health effects; "
        "the general public is less likely to be affected."
    ),
    "unhealthy": (
        "Some members of the general public may experience health "
        "effects; sensitive groups may experience more serious effects."
    ),
    "very": "Health alert: the risk of health effects is increased for everyone.",
    "severe": (
        "Health warning of emergency conditions; "
        "everyone is more likely to be affected."
    ),
    "hazardous": (
        "Health warning of emergency conditions; "
        "the entire population is likely to be affected."
    ),
    "poor": "Air quality is poor and may cause health effects for many people.",
}


def describe_class(class_name: str) -> str:
    """Return a generic description for a predicted category."""
    key = class_name.lower()

    for hint, description in _CATEGORY_HINTS.items():
        if hint in key:
            return description

    return (
        "Predicted air-quality category from the trained image classifier. "
        "See dataset documentation for this category's definition."
    )


def _resolve_model_path(raw_path: str) -> Path:
    """
    Resolve a model path on Windows or a cloud server.

    Supports:
    - An existing absolute path.
    - A project-relative path such as models/manual_run.keras.
    - An old Windows path stored in metadata, using the model filename
      inside the current project's models directory as a fallback.
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise RuntimeError(
            "Model path is missing from final_model_meta.json."
        )

    root = Path(config.ROOT).resolve()
    supplied = Path(raw_path).expanduser()

    candidates = []

    # First, preserve a path that already exists on this machine.
    if supplied.is_absolute():
        candidates.append(supplied)
    else:
        candidates.append(root / supplied)

    # pathlib on Linux does not recognize a Windows C:\\... path as
    # a native absolute path, so use its filename as a safe fallback.
    windows_path = PureWindowsPath(raw_path)
    model_filename = windows_path.name

    if model_filename:
        candidates.append(root / "models" / model_filename)

    # Also support the configured models directory.
    candidates.append(Path(config.MODELS_DIR) / model_filename)

    seen = set()

    for candidate in candidates:
        try:
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)

            if candidate.is_file():
                return candidate
        except (OSError, RuntimeError):
            continue

    checked_paths = "\n".join(str(path) for path in candidates)

    raise FileNotFoundError(
        "Trained model file was not found. Make sure the .keras file "
        "is available on the deployment server.\n"
        f"Paths checked:\n{checked_paths}"
    )


@lru_cache(maxsize=1)
def load_predictor():
    """Load the selected model and class mapping once per process."""
    meta = utils.load_json(config.FINAL_MODEL_META_JSON)

    if not isinstance(meta, dict):
        raise RuntimeError(
            f"Model metadata is missing or invalid: "
            f"{config.FINAL_MODEL_META_JSON}"
        )

    required = ("model_path", "selected_model", "image_size")
    missing = [key for key in required if key not in meta]

    if missing:
        raise RuntimeError(
            "Model metadata is missing required fields: "
            + ", ".join(missing)
        )

    model_path = _resolve_model_path(meta["model_path"])

    log.info("Loading trained model from %s", model_path)

    # Load the trained model. Errors are allowed to surface with context
    # rather than silently returning fake predictions.
    try:
        model = keras.models.load_model(str(model_path))
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the trained model at {model_path}: {exc}"
        ) from exc

    class_mapping = utils.load_json(config.CLASS_MAP_JSON)

    if not isinstance(class_mapping, dict):
        raise RuntimeError(
            f"Class mapping is missing or invalid: {config.CLASS_MAP_JSON}"
        )

    classes = class_mapping.get("classes")

    if (
        not isinstance(classes, list)
        or not classes
        or not all(isinstance(name, str) for name in classes)
    ):
        raise RuntimeError(
            "Class mapping must contain a non-empty list of class names."
        )

    # Store the resolved path in runtime metadata so callers see the
    # path actually used on this machine, without rewriting the JSON file.
    runtime_meta = dict(meta)
    runtime_meta["model_path"] = str(model_path)

    return model, runtime_meta, classes


def predict_image(
    image_path: str,
    with_gradcam: bool = False,
    gradcam_out: Optional[str] = None,
    threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
) -> Dict[str, Any]:
    """Predict the air-quality category for a single image."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    source_image = Path(image_path)

    if not source_image.is_file():
        raise FileNotFoundError(
            f"Input image was not found: {source_image}"
        )

    model, meta, classes = load_predictor()

    name = meta["selected_model"]
    size = int(meta["image_size"])

    if size <= 0:
        raise RuntimeError("Model image_size must be a positive integer.")

    preprocess_fn = models.get_preprocess_fn(name)

    batch = pp.decode_and_preprocess(
        tf.constant(str(source_image.resolve())),
        size,
        preprocess_fn,
    )[tf.newaxis, ...]

    probs = np.asarray(model.predict(batch, verbose=0)[0], dtype=float)

    if probs.ndim != 1 or len(probs) != len(classes):
        raise RuntimeError(
            "Model output does not match the configured class mapping: "
            f"{len(probs)} outputs for {len(classes)} classes."
        )

    if not np.all(np.isfinite(probs)):
        raise RuntimeError("Model produced invalid probability values.")

    order = np.argsort(probs)[::-1]
    top_idx = int(order[0])
    confidence = float(probs[top_idx])

    result: Dict[str, Any] = {
        "image": source_image.name,
        "predicted_class": classes[top_idx],
        "confidence": round(confidence, 4),
        "description": describe_class(classes[top_idx]),
        "top3": [
            {
                "class": classes[int(i)],
                "probability": round(float(probs[i]), 4),
            }
            for i in order[:3]
        ],
        "distribution": {
            classes[i]: round(float(probs[i]), 4)
            for i in range(len(classes))
        },
        "uncertain": confidence < threshold,
        "uncertainty_note": (
            "Low confidence - treat this prediction with caution."
            if confidence < threshold
            else ""
        ),
        "disclaimer": (
            "Image-based air-quality classification; it does not "
            "physically measure pollutant concentrations."
        ),
    }

    if with_gradcam:
        from . import explainability as xai

        out = Path(
            gradcam_out
            or (config.EXPLAIN_DIR / "prediction_gradcam.png")
        )
        out.parent.mkdir(parents=True, exist_ok=True)

        try:
            xai.generate_for_image(
                model,
                str(source_image),
                name,
                size,
                out,
                top_idx,
            )

            if out.is_file():
                result["gradcam_path"] = str(out)
        except Exception as exc:
            log.warning("Grad-CAM generation failed: %s", exc)
            result["gradcam_error"] = str(exc)

    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Predict an air-quality category from an image."
    )
    parser.add_argument("image", help="Path to an input image.")
    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Generate an optional Grad-CAM visualization.",
    )
    args = parser.parse_args()

    print(
        json.dumps(
            predict_image(args.image, with_gradcam=args.gradcam),
            indent=2,
        )
    )
