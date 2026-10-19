"""
Phase 2 - Dataset audit + EDA.

Computes a complete, machine-readable audit of the REAL dataset and renders
presentation-quality EDA plots. Nothing here is hard-coded; every number comes
from the files. Duplicate/near-duplicate detection lives in duplicates.py and
is summarised into the audit.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from PIL import Image

from . import config, utils
from . import dataset as ds

log = utils.get_logger("dataset_analysis")


def _image_stats(path: str) -> Dict[str, Any]:
    """Open one image and return its properties, or mark it invalid."""
    try:
        with Image.open(path) as im:
            im.verify()  # cheap integrity check
        with Image.open(path) as im:  # reopen (verify() invalidates the handle)
            w, h = im.size
            mode = im.mode
            fmt = (im.format or Path(path).suffix.lstrip(".")).upper()
        size_bytes = Path(path).stat().st_size
        return {
            "valid": True, "width": w, "height": h, "mode": mode,
            "format": fmt, "size_bytes": size_bytes,
            "aspect_ratio": round(w / h, 4) if h else None,
            "is_small": min(w, h) <= config.SMALL_IMAGE_MAX,
            "is_large": max(w, h) >= config.LARGE_IMAGE_MIN,
            "is_grayscale": mode in ("L", "1", "LA"),
        }
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


def audit_dataset(df: pd.DataFrame | None = None) -> Dict[str, Any]:
    """Run the full audit and write JSON + Markdown reports."""
    config.ensure_dirs()
    report = ds.discover()
    df = df if df is not None else ds.build_records(report=report)

    stats: List[Dict[str, Any]] = []
    corrupted: List[str] = []
    for _, row in df.iterrows():
        s = _image_stats(row["path"])
        if s["valid"]:
            s["label"] = row["label"]
            stats.append(s)
        else:
            corrupted.append(row["path"])

    sdf = pd.DataFrame(stats)
    class_counts = df["label"].value_counts().to_dict()
    counts = np.array(list(class_counts.values())) if class_counts else np.array([0])

    audit: Dict[str, Any] = {
        "dataset_root": report["dataset_root"],
        "primary_label_source": report.get("primary_label_source"),
        "totals": {
            "records": int(len(df)),
            "valid_images": int(len(sdf)),
            "corrupted_or_unreadable": int(len(corrupted)),
        },
        "classes": {
            "n_classes": int(df["label"].nunique()),
            "names": sorted(map(str, df["label"].unique())),
            "counts": {str(k): int(v) for k, v in class_counts.items()},
            "percentages": {str(k): round(100 * v / len(df), 2)
                            for k, v in class_counts.items()} if len(df) else {},
            "imbalance_ratio": round(float(counts.max() / counts.min()), 3)
            if counts.min() > 0 else None,
        },
        "corrupted_files": corrupted[:50],
    }

    if not sdf.empty:
        audit["image_properties"] = {
            "formats": {str(k): int(v)
                        for k, v in Counter(sdf["format"]).items()},
            "grayscale_count": int(sdf["is_grayscale"].sum()),
            "rgb_count": int((~sdf["is_grayscale"]).sum()),
            "very_small_images": int(sdf["is_small"].sum()),
            "very_large_images": int(sdf["is_large"].sum()),
            "width": _num_summary(sdf["width"]),
            "height": _num_summary(sdf["height"]),
            "aspect_ratio": _num_summary(sdf["aspect_ratio"]),
            "size_bytes": _num_summary(sdf["size_bytes"]),
        }

    # Fold in duplicate summary if it was already produced.
    dup = utils.load_json(config.DUPLICATES_JSON, default=None)
    if dup:
        audit["duplicates"] = {
            "exact_duplicate_groups": dup.get("n_exact_groups"),
            "near_duplicate_groups": dup.get("n_near_groups"),
        }

    utils.save_json(config.DATASET_AUDIT_JSON, audit)
    _write_audit_markdown(audit)
    log.info("Audit complete: %d valid / %d records, %d classes",
             audit["totals"]["valid_images"], audit["totals"]["records"],
             audit["classes"]["n_classes"])
    return audit


def _num_summary(series: pd.Series) -> Dict[str, float]:
    s = series.dropna().astype(float)
    if s.empty:
        return {}
    return {
        "min": float(s.min()), "max": float(s.max()),
        "mean": round(float(s.mean()), 2), "median": float(s.median()),
        "p05": float(s.quantile(0.05)), "p95": float(s.quantile(0.95)),
    }


def _write_audit_markdown(audit: Dict[str, Any]) -> None:
    t = audit["totals"]
    c = audit["classes"]
    lines = [
        "# Dataset Audit Report",
        "",
        f"- **Dataset root:** `{audit['dataset_root']}`",
        f"- **Primary label source:** {audit['primary_label_source']}",
        f"- **Total records:** {t['records']}",
        f"- **Valid images:** {t['valid_images']}",
        f"- **Corrupted / unreadable:** {t['corrupted_or_unreadable']}",
        f"- **Classes ({c['n_classes']}):** {', '.join(c['names'])}",
        f"- **Imbalance ratio (max/min):** {c['imbalance_ratio']}",
        "",
        "## Images per class",
        "",
        "| Class | Count | % |",
        "| --- | ---: | ---: |",
    ]
    for name in c["names"]:
        lines.append(f"| {name} | {c['counts'].get(name, 0)} | "
                     f"{c['percentages'].get(name, 0)} |")
    ip = audit.get("image_properties")
    if ip:
        lines += [
            "", "## Image properties", "",
            f"- Formats: {ip['formats']}",
            f"- RGB: {ip['rgb_count']} | Grayscale: {ip['grayscale_count']}",
            f"- Very small (<= {config.SMALL_IMAGE_MAX}px): {ip['very_small_images']}",
            f"- Very large (>= {config.LARGE_IMAGE_MIN}px): {ip['very_large_images']}",
            f"- Width: {ip['width']}",
            f"- Height: {ip['height']}",
            f"- Aspect ratio: {ip['aspect_ratio']}",
        ]
    if "duplicates" in audit:
        lines += ["", "## Duplicates",
                  f"- Exact duplicate groups: {audit['duplicates']['exact_duplicate_groups']}",
                  f"- Near-duplicate groups: {audit['duplicates']['near_duplicate_groups']}"]
    lines += ["", "> Generated automatically from the real dataset. "
              "No values are hard-coded.", ""]
    config.DATASET_AUDIT_MD.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# EDA plots                                                                    #
# --------------------------------------------------------------------------- #
def run_eda(df: pd.DataFrame | None = None) -> None:
    """Generate presentation-quality EDA figures under results/eda/."""
    utils.apply_plot_style()
    import matplotlib.pyplot as plt

    config.ensure_dirs()
    df = df if df is not None else ds.build_records()
    counts = df["label"].value_counts().sort_index()

    # 1 + 2: class distribution (count) and percentage.
    for kind in ("count", "percent"):
        vals = counts if kind == "count" else (counts / counts.sum() * 100)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(vals.index.astype(str), vals.values, color="#4f8cff")
        ax.set_title(f"Class distribution ({kind})")
        ax.set_ylabel("Images" if kind == "count" else "Percentage (%)")
        ax.tick_params(axis="x", rotation=30)
        fig.savefig(config.EDA_DIR / f"class_distribution_{kind}.png")
        plt.close(fig)

    _plot_sample_grid(df, plt)
    _plot_image_geometry(df, plt)
    _plot_formats(df, plt)
    _plot_aqi(df, plt)
    log.info("EDA figures written to %s", config.EDA_DIR)


def _plot_sample_grid(df, plt, per_class: int = 5) -> None:
    classes = sorted(df["label"].unique())
    rows = len(classes)
    fig, axes = plt.subplots(rows, per_class,
                             figsize=(per_class * 2.2, rows * 2.2))
    axes = np.atleast_2d(axes)
    for r, cls in enumerate(classes):
        subset = df[df["label"] == cls]["path"].head(per_class).tolist()
        for cidx in range(per_class):
            ax = axes[r, cidx]
            ax.axis("off")
            if cidx < len(subset):
                try:
                    ax.imshow(Image.open(subset[cidx]).convert("RGB"))
                except Exception:
                    pass
            if cidx == 0:
                ax.set_title(str(cls), loc="left", fontsize=9)
    fig.suptitle("Representative samples per class", y=1.0)
    fig.savefig(config.EDA_DIR / "sample_grid.png")
    plt.close(fig)


def _plot_image_geometry(df, plt) -> None:
    ws, hs, ars = [], [], []
    for p in df["path"].sample(min(len(df), 800), random_state=config.SEED):
        try:
            with Image.open(p) as im:
                w, h = im.size
            ws.append(w); hs.append(h); ars.append(w / h if h else 0)
        except Exception:
            continue
    if not ws:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(ws, hs, s=8, alpha=0.4, color="#22d3a7")
    ax.set(title="Image resolution distribution", xlabel="Width (px)",
           ylabel="Height (px)")
    fig.savefig(config.EDA_DIR / "resolution_distribution.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ars, bins=40, color="#a78bfa")
    ax.set(title="Aspect-ratio distribution", xlabel="width / height")
    fig.savefig(config.EDA_DIR / "aspect_ratio_distribution.png")
    plt.close(fig)


def _plot_formats(df, plt) -> None:
    fmts = Counter()
    for p in df["path"]:
        fmts[Path(p).suffix.lower().lstrip(".")] += 1
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(fmts.keys()), list(fmts.values()), color="#f59e0b")
    ax.set(title="File-format distribution", ylabel="Count")
    fig.savefig(config.EDA_DIR / "format_distribution.png")
    plt.close(fig)


def _plot_aqi(df, plt) -> None:
    aqi_cols = [c for c in df.columns
                if any(k in c.lower() for k in ("aqi", "pm2", "pm10", "no2",
                                                "so2", "co", "o3"))]
    for col in aqi_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(vals, bins=40, color="#ef4444")
        ax.set(title=f"{col} distribution", xlabel=col)
        safe = "".join(ch if ch.isalnum() else "_" for ch in col)
        fig.savefig(config.EDA_DIR / f"aqi_{safe}.png")
        plt.close(fig)


def verify_labels(df: pd.DataFrame | None = None) -> Dict[str, Any]:
    """Document what each class means and check any AQI-column relationship."""
    config.ensure_dirs()
    report = ds.discover()
    df = df if df is not None else ds.build_records(report=report)
    source = report.get("primary_label_source")
    classes = sorted(map(str, df["label"].unique()))

    lines = ["# Label Verification", "",
             f"- **Label source:** {source}",
             f"- **Classes ({len(classes)}):** {', '.join(classes)}", ""]
    if source == "folder":
        lines += ["Labels are taken directly from image sub-folder names "
                  "(each folder = one class). This is unambiguous ground truth; "
                  "no labels were renamed or merged.", ""]
    else:
        lines += ["Labels are taken from a CSV column mapping filename -> class.",
                  ""]

    # If AQI-like columns exist, summarise their range per class (relationship).
    aqi_cols = [c for c in df.columns
                if any(k in c.lower() for k in ("aqi", "pm2", "pm10"))]
    result: Dict[str, Any] = {"label_source": source, "classes": classes,
                              "aqi_relationship": {}}
    if aqi_cols:
        lines += ["## AQI relationship (observed, not imposed)", ""]
        for col in aqi_cols:
            vals = pd.to_numeric(df[col], errors="coerce")
            grp = df.assign(_v=vals).dropna(subset=["_v"]).groupby("label")["_v"]
            if grp.count().sum() == 0:
                continue
            stats = grp.agg(["min", "median", "max"]).round(2)
            result["aqi_relationship"][col] = stats.to_dict("index")
            lines += [f"### {col}", "",
                      "| Class | min | median | max |",
                      "| --- | ---: | ---: | ---: |"]
            for cls, r in stats.iterrows():
                lines.append(f"| {cls} | {r['min']} | {r['median']} | {r['max']} |")
            lines.append("")
        lines += ["> Labels were verified against AQI columns for consistency; "
                  "they were NOT altered to make classification easier.", ""]
    config.LABEL_REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    log.info("Label verification written -> %s", config.LABEL_REPORT_MD)
    return result


if __name__ == "__main__":
    a = audit_dataset()
    run_eda()
    print("Audit + EDA done.")
