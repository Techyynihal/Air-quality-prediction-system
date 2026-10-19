"""
Model zoo (TensorFlow / Keras).

A small registry maps a model name to:
  * the keras.applications backbone constructor,
  * its official preprocess_input (so training and inference match EXACTLY),
  * a sensible default input size.

build_model() attaches a compact classification head:
    backbone -> GlobalAveragePooling -> Dropout -> Dense(softmax)

Staged transfer learning is supported: the backbone is frozen by default and
can be partially unfrozen for fine-tuning via unfreeze_top().
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications as ka


# name -> (constructor, preprocess_input, default_size, tier)
BACKBONES: Dict[str, Tuple[Callable, Callable, int, int]] = {
    "mobilenetv3large": (ka.MobileNetV3Large,
                         ka.mobilenet_v3.preprocess_input, 224, 1),
    "efficientnetb0": (ka.EfficientNetB0,
                       ka.efficientnet.preprocess_input, 224, 1),
    "efficientnetb1": (ka.EfficientNetB1,
                       ka.efficientnet.preprocess_input, 240, 1),
    "resnet50": (ka.ResNet50, ka.resnet.preprocess_input, 224, 2),
    "convnexttiny": (ka.ConvNeXtTiny, ka.convnext.preprocess_input, 224, 2),
    "efficientnetv2s": (ka.EfficientNetV2S,
                        ka.efficientnet_v2.preprocess_input, 300, 2),
    "vgg16": (ka.VGG16, ka.vgg16.preprocess_input, 224, 3),
}


def get_preprocess_fn(model_name: str) -> Callable:
    """Return the official preprocess_input for a backbone."""
    return BACKBONES[model_name.lower()][1]


def default_size(model_name: str) -> int:
    return BACKBONES[model_name.lower()][2]


def build_model(model_name: str, num_classes: int, image_size: int,
                dropout: float = 0.3, weight_decay: float = 1e-5,
                trainable_backbone: bool = False) -> keras.Model:
    """Build a transfer-learning model with a compact head."""
    name = model_name.lower()
    if name not in BACKBONES:
        raise ValueError(f"Unknown model '{model_name}'. "
                         f"Options: {list(BACKBONES)}")
    ctor, _, _, _ = BACKBONES[name]

    inputs = keras.Input(shape=(image_size, image_size, 3))
    backbone = ctor(include_top=False, weights="imagenet", input_tensor=inputs)
    backbone.trainable = trainable_backbone

    x = keras.layers.GlobalAveragePooling2D(name="gap")(backbone.output)
    x = keras.layers.Dropout(dropout, name="head_dropout")(x)
    reg = keras.regularizers.l2(weight_decay) if weight_decay else None
    # Force the softmax output to float32 even under mixed_float16. Keras
    # applies the global mixed-precision policy to every layer, but a float16
    # softmax is numerically unstable and can produce NaN losses; the official
    # guidance is to keep the final activation in float32. This is a no-op when
    # the global policy is already float32 (CPU / no-GPU runs).
    outputs = keras.layers.Dense(num_classes, activation="softmax",
                                 kernel_regularizer=reg, dtype="float32",
                                 name="predictions")(x)

    model = keras.Model(inputs, outputs, name=f"{name}_aqi")
    model.backbone = backbone  # convenience handle for fine-tuning
    return model


def count_trainable_params(model: keras.Model) -> int:
    """Number of trainable scalar parameters (reflects current freeze state)."""
    try:
        return int(sum(int(tf.size(w)) for w in model.trainable_weights))
    except Exception:
        # Fallback for older/newer Keras variable APIs.
        import numpy as np
        return int(sum(np.prod(w.shape) for w in model.trainable_weights))


def unfreeze_top(model: keras.Model, n_layers: int) -> int:
    """Unfreeze the top `n_layers` of the backbone for fine-tuning.

    BatchNorm layers are kept frozen (best practice for small datasets).
    Returns the number of layers actually set trainable.
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return 0
    backbone.trainable = True
    layers = backbone.layers
    cutoff = max(0, len(layers) - n_layers)
    trainable = 0
    for i, layer in enumerate(layers):
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
        elif i >= cutoff:
            layer.trainable = True
            trainable += 1
        else:
            layer.trainable = False
    return trainable


def last_conv_layer_name(model: keras.Model) -> str:
    """Best-effort discovery of the last 4-D conv feature map (for Grad-CAM)."""
    for layer in reversed(model.backbone.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue
    raise ValueError("No 4-D convolutional layer found for Grad-CAM.")


def model_size_mb(model: keras.Model) -> float:
    params = model.count_params()
    return round(params * 4 / (1024 ** 2), 2)  # float32 estimate
