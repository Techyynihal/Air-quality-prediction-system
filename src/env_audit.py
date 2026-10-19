"""
Phase 1 - Environment audit.

Inspects the real machine (OS, Python, CPU, RAM, GPU, CUDA, key libraries)
and writes results/audit/environment_audit.json. It also decides safe runtime
settings (whether mixed precision is usable) WITHOUT assuming any specific GPU.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
from typing import Any, Dict

from . import config, utils

log = utils.get_logger("env_audit")


def _safe(fn):
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - defensive
        return f"unavailable ({exc})"


def _cpu_count() -> Any:
    import os
    return os.cpu_count()


def _total_ram_gb() -> Any:
    try:
        import psutil  # optional
        return round(psutil.virtual_memory().total / 1e9, 2)
    except Exception:
        # Fallback via os.sysconf (Linux/mac); may be unavailable on Windows.
        try:
            import os
            return round(
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 2
            )
        except Exception:
            return "unknown"


def _nvidia_gpus() -> Any:
    """Return GPU names from nvidia-smi if present, else None."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            text=True, timeout=15,
        )
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        return None


def _tf_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"available": False}
    try:
        import tensorflow as tf
        info["available"] = True
        info["version"] = tf.__version__
        gpus = tf.config.list_physical_devices("GPU")
        info["gpu_devices"] = [g.name for g in gpus]
        info["gpu_count"] = len(gpus)
        info["built_with_cuda"] = bool(getattr(tf.test, "is_built_with_cuda",
                                               lambda: False)())
        # Enable memory growth so we never grab all VRAM at once.
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
    except Exception as exc:
        info["error"] = str(exc)
    return info


def run_environment_audit(verbose: bool = True) -> Dict[str, Any]:
    """Collect environment facts, persist them, and return the dict."""
    config.ensure_dirs()

    tf_info = _tf_info()
    gpus = _nvidia_gpus()
    has_gpu = bool(gpus) or tf_info.get("gpu_count", 0) > 0

    audit: Dict[str, Any] = {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python_version": platform.python_version(),
        "cpu": {
            "processor": _safe(platform.processor),
            "logical_cores": _safe(_cpu_count),
        },
        "ram_gb": _total_ram_gb(),
        "gpu": {
            "nvidia_smi_devices": gpus,
            "has_gpu": has_gpu,
        },
        "tensorflow": tf_info,
        "libraries": _library_versions(),
    }

    # Decide safe runtime settings without assuming any specific device.
    audit["recommended"] = {
        "device": "GPU" if has_gpu else "CPU",
        # Mixed precision only when a GPU is actually present.
        "mixed_precision": bool(has_gpu),
        "suggested_batch_size": 32 if has_gpu else 16,
        "note": ("GPU detected - training full candidates is reasonable."
                 if has_gpu else
                 "No GPU detected - prefer screening + fewer candidates; "
                 "training will be slow on CPU."),
    }

    utils.save_json(config.ENV_AUDIT_JSON, audit)
    if verbose:
        log.info("Environment: %s | Python %s | device=%s | mixed_precision=%s",
                 audit["os"]["system"], audit["python_version"],
                 audit["recommended"]["device"],
                 audit["recommended"]["mixed_precision"])
        log.info("Saved -> %s", config.ENV_AUDIT_JSON)
    return audit


def _library_versions() -> Dict[str, Any]:
    libs = ["numpy", "pandas", "sklearn", "PIL", "cv2", "matplotlib",
            "seaborn", "imagehash", "flask"]
    out: Dict[str, Any] = {}
    for name in libs:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "installed")
        except Exception:
            out[name] = None
    return out


if __name__ == "__main__":
    run_environment_audit()
