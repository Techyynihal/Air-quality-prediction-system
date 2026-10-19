"""Shared utilities: logging, deterministic seeding, safe I/O, plotting."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import config

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "aqi") -> logging.Logger:
    """Return a module logger configured once (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_global_seed(seed: int = config.SEED) -> None:
    """Seed python, numpy, and (if importable) tensorflow for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # noqa: WPS433 (local import on purpose)

        tf.random.set_seed(seed)
        # Best-effort determinism; harmless if the op is unavailable.
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:  # pragma: no cover - version dependent
            pass
    except Exception:  # pragma: no cover - TF optional at import time
        pass


def save_json(path: os.PathLike | str, data: Any, indent: int = 2) -> None:
    """Write JSON, creating parent dirs. Falls back to str() for odd types."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent, default=_json_default),
                    encoding="utf-8")


def load_json(path: os.PathLike | str, default: Any = None) -> Any:
    """Load JSON if the file exists, else return `default`."""
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def human_bytes(num: float) -> str:
    """Human-readable byte size (e.g. 12.3 MB)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


def apply_plot_style() -> None:
    """Consistent, presentation-quality matplotlib style (dark-friendly)."""
    import matplotlib

    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
