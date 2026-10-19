"""
Error analysis over the final test predictions.

Identifies the most-confused class pairs, high-confidence mistakes, low-
confidence predictions, and per-class failure rates, and renders a grid of
representative errors (actual vs predicted + confidence). Visual explanations
are phrased as "possible visual factors", never as proven causes.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict

import numpy as np
import pandas as pd

from . import config, utils
from .evaluate import TEST_PRED_CSV

log = utils.get_logger("error_analysis")


def analyze() -> Dict[str, Any]:
    config.ensure_dirs()
    if not TEST_PRED_CSV.exists():
        raise RuntimeError("Run evaluate.evaluate_final() first.")
    df = pd.read_csv(TEST_PRED_CSV)
    errors = df[df["true_idx"] != df["pred_idx"]].copy()

    confused = Counter(
        zip(errors["true_label"], errors["pred_label"])
    ).most_common(10)

    per_class = {}
    for cls, grp in df.groupby("true_label"):
        n = len(grp)
        n_err = int((grp["true_idx"] != grp["pred_idx"]).sum())
        per_class[str(cls)] = {
            "support": n, "errors": n_err,
            "error_rate": round(n_err / n, 4) if n else 0.0,
        }

    high_conf_err = (errors.sort_values("confidence", ascending=False)
                     .head(15)[["filename", "true_label", "pred_label",
                                "confidence"]].to_dict("records"))
    low_conf = (df.sort_values("confidence")
                .head(15)[["filename", "true_label", "pred_label",
                           "confidence"]].to_dict("records"))

    summary = {
        "n_test": int(len(df)),
        "n_errors": int(len(errors)),
        "error_rate": round(len(errors) / len(df), 4) if len(df) else 0.0,
        "most_confused_pairs": [
            {"actual": a, "predicted": p, "count": c} for (a, p), c in confused
        ],
        "per_class_error": per_class,
        "high_confidence_errors": high_conf_err,
        "low_confidence_predictions": low_conf,
        "caveat": ("Confusions may reflect possible visual factors (haze, "
                   "lighting, similar scenes); not asserted as proven causes."),
    }
    utils.save_json(config.ERROR_DIR / "error_analysis.json", summary)
    _plot_error_grid(errors)
    _write_md(summary)
    log.info("Error analysis: %d/%d errors (%.2f%%)", len(errors), len(df),
             100 * summary["error_rate"])
    return summary


def _plot_error_grid(errors: pd.DataFrame, n: int = 9) -> None:
    if errors.empty:
        return
    utils.apply_plot_style()
    import matplotlib.pyplot as plt
    from PIL import Image

    subset = errors.head(n)
    cols = 3
    rows = int(np.ceil(len(subset) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, (_, r) in zip(axes, subset.iterrows()):
        try:
            ax.imshow(Image.open(r["path"]).convert("RGB"))
        except Exception:
            continue
        ax.set_title(f"A:{r['true_label']}\nP:{r['pred_label']} "
                     f"({r['confidence']:.2f})", fontsize=8)
    fig.suptitle("Representative misclassifications")
    fig.savefig(config.ERROR_DIR / "representative_errors.png")
    plt.close(fig)


def _write_md(s: Dict[str, Any]) -> None:
    lines = ["# Error Analysis", "",
             f"- Test images: {s['n_test']}",
             f"- Errors: {s['n_errors']} ({100 * s['error_rate']:.2f}%)", "",
             "## Most confused class pairs", "",
             "| Actual | Predicted | Count |", "| --- | --- | ---: |"]
    for p in s["most_confused_pairs"]:
        lines.append(f"| {p['actual']} | {p['predicted']} | {p['count']} |")
    lines += ["", "## Per-class error rate", "",
              "| Class | Support | Errors | Rate |",
              "| --- | ---: | ---: | ---: |"]
    for c, v in s["per_class_error"].items():
        lines.append(f"| {c} | {v['support']} | {v['errors']} | {v['error_rate']} |")
    lines += ["", f"> {s['caveat']}", ""]
    (config.ERROR_DIR / "error_analysis.md").write_text("\n".join(lines),
                                                        encoding="utf-8")


if __name__ == "__main__":
    analyze()
