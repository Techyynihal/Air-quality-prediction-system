"""
Dataset discovery.

Auto-discovers the real dataset without assuming its layout. It finds image
files, CSV/metadata files, infers the label source (folder-based vs CSV-based),
and builds a unified records table:

    columns: path, label, label_source, group_id, <extra metadata columns...>

The `group_id` is used later for leakage-safe splitting (images that likely
belong to the same scene/location/session must not cross the train/test line).

Design choices (documented in results/audit/dataset_discovery.json):
  * If class sub-folders are present, folder names are the PRIMARY labels
    (reliable, unambiguous). Any CSV is then used to ENRICH metadata
    (city/location/date/AQI) and to build group_ids where possible.
  * If no class folders exist but a CSV maps filenames -> a categorical
    column, CSV labels are used instead.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from . import config, utils

log = utils.get_logger("dataset")


# --------------------------------------------------------------------------- #
# Low-level scanning                                                           #
# --------------------------------------------------------------------------- #
def find_images(root: Path) -> List[Path]:
    """Recursively list all files with a known image extension."""
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    )


def find_csvs(root: Path) -> List[Path]:
    return sorted(root.rglob("*.csv"))


def _folder_label(path: Path, root: Path) -> str:
    """Immediate parent folder name relative to the dataset root."""
    rel = path.relative_to(root)
    parts = rel.parts
    # parts[-1] is the filename; the label is the deepest folder above it.
    return parts[-2] if len(parts) >= 2 else "_root"


# --------------------------------------------------------------------------- #
# CSV inspection                                                               #
# --------------------------------------------------------------------------- #
def _read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return None


def _guess_filename_column(df: pd.DataFrame, basenames: set) -> Optional[str]:
    """Find the CSV column whose values best match image basenames."""
    best_col, best_hit = None, 0.0
    for col in df.columns:
        vals = df[col].astype(str)
        # Compare on basename (strip any path the CSV may contain).
        hits = vals.map(lambda v: Path(v).name in basenames).mean()
        if hits > best_hit:
            best_hit, best_col = hits, col
    return best_col if best_hit >= 0.5 else None


def _guess_group_columns(df: pd.DataFrame) -> List[str]:
    """Columns that likely identify a scene/location/session for grouping.

    Includes date/time components (year/month/day/hour) so that images captured
    in the *same session* (same place + same timestamp -> near-identical burst
    frames) are grouped together and never split across train/val/test.
    """
    keys = ("city", "location", "place", "state", "province", "country",
            "station", "site", "date", "time", "session",
            "year", "month", "day", "hour")
    # Preserve a sensible session ordering: location first, then date/time parts.
    order = {"location": 0, "city": 0, "place": 0, "site": 0, "station": 0,
             "state": 1, "province": 1, "country": 1,
             "year": 2, "month": 3, "day": 4, "date": 4, "hour": 5,
             "time": 5, "session": 6}

    def rank(col: str) -> int:
        cl = col.lower()
        return min((order[k] for k in order if k in cl), default=9)

    cols = [c for c in df.columns if any(k in c.lower() for k in keys)]
    return sorted(cols, key=rank)


def _guess_aqi_columns(df: pd.DataFrame) -> List[str]:
    keys = ("aqi", "pm2", "pm10", "pm25", "no2", "so2", "co", "o3", "ozone",
            "pollut")
    return [c for c in df.columns
            if any(k in c.lower() for k in keys)]


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def discover(root: Path = config.DATASET_DIR) -> Dict[str, Any]:
    """Return a discovery report describing what is really in `root`."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}. See dataset/README.txt."
        )

    images = find_images(root)
    csvs = find_csvs(root)
    folder_labels = sorted({_folder_label(p, root) for p in images})

    report: Dict[str, Any] = {
        "dataset_root": str(root),
        "n_images_found": len(images),
        "n_csv_files": len(csvs),
        "csv_files": [str(c.relative_to(root)) for c in csvs],
        "folder_label_candidates": folder_labels,
        "n_folder_label_candidates": len(folder_labels),
    }

    if not images:
        report["primary_label_source"] = "none"
        report["reason"] = "No image files were discovered under dataset/."
        utils.save_json(config.DISCOVERY_JSON, report)
        log.warning("No images found under %s", root)
        return report

    # Decide primary label source.
    has_folder_labels = len(folder_labels) >= 2 and folder_labels != ["_root"]

    csv_summaries = []
    basenames = {p.name for p in images}
    for c in csvs:
        df = _read_csv_safe(c)
        if df is None:
            continue
        fn_col = _guess_filename_column(df, basenames)
        csv_summaries.append({
            "file": str(c.relative_to(root)),
            "columns": list(map(str, df.columns)),
            "n_rows": int(len(df)),
            "filename_column": fn_col,
            "group_columns": _guess_group_columns(df),
            "aqi_columns": _guess_aqi_columns(df),
        })
    report["csv_summaries"] = csv_summaries

    if has_folder_labels:
        report["primary_label_source"] = "folder"
        report["reason"] = (
            "Class sub-folders detected; folder names are used as the primary, "
            "unambiguous ground-truth labels. CSV metadata (if any) is used only "
            "to enrich location/AQI info and to build leakage groups."
        )
    else:
        usable = next((s for s in csv_summaries if s["filename_column"]), None)
        report["primary_label_source"] = "csv" if usable else "unknown"
        report["reason"] = (
            "No class folders found; a CSV mapping filenames to a label column "
            "will be used." if usable else
            "Could not determine labels from folders or CSV; manual review needed."
        )

    utils.save_json(config.DISCOVERY_JSON, report)
    log.info("Discovery: %d images, %d classes (folder), primary source = %s",
             len(images), len(folder_labels), report["primary_label_source"])
    return report


def _load_metadata(root: Path, report: Dict[str, Any],
                   basenames: set) -> Optional[pd.DataFrame]:
    """Build a per-basename metadata frame merged from ALL usable CSVs.

    Real datasets often split metadata across several files (e.g. one AQI CSV
    per city). We therefore concatenate every CSV that has a filename column so
    the metadata covers the whole image set, not just the single largest file.
    """
    frames = []
    for summary in report.get("csv_summaries", []):
        fn_col = summary.get("filename_column")
        if not fn_col:
            continue
        df = _read_csv_safe(root / summary["file"])
        if df is None or fn_col not in df.columns:
            continue
        df = df.copy()
        df["__basename"] = df[fn_col].astype(str).str.strip().map(
            lambda v: Path(v).name)
        # Keep only rows whose basename is an actual image in this root.
        df = df[df["__basename"].isin(basenames)]
        frames.append(df)

    if not frames:
        return None

    meta = pd.concat(frames, ignore_index=True, sort=False)
    meta = meta.drop_duplicates("__basename").set_index("__basename")
    return meta


def build_records(root: Path = config.DATASET_DIR,
                  report: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Return the unified records table used by every downstream stage."""
    root = Path(root)
    report = report or discover(root)
    images = find_images(root)
    basenames = {p.name for p in images}
    meta = _load_metadata(root, report, basenames)

    rows: List[Dict[str, Any]] = []
    source = report.get("primary_label_source", "folder")
    for p in images:
        rec: Dict[str, Any] = {
            "path": str(p),
            "filename": p.name,
            "label_source": source,
        }
        rec["label"] = _folder_label(p, root) if source == "folder" else None
        rows.append(rec)

    df = pd.DataFrame(rows)

    # Enrich with CSV metadata + derive group_id / CSV label if needed.
    if meta is not None:
        group_cols = _guess_group_columns(meta)
        aqi_cols = _guess_aqi_columns(meta)
        keep = [c for c in (group_cols + aqi_cols) if c in meta.columns]
        joined = df.join(meta[keep], on="filename") if keep else df

        if source == "csv":
            # Choose the categorical column with the fewest, most balanced uniques.
            cat_candidates = [c for c in meta.columns
                              if 2 <= meta[c].nunique() <= 20
                              and meta[c].dtype == object]
            if cat_candidates:
                label_col = cat_candidates[0]
                joined["label"] = df["filename"].map(meta[label_col])

        if group_cols:
            joined["group_id"] = joined[group_cols].astype(str).agg("|".join,
                                                                     axis=1)
            # Rows the metadata did not cover collapse to "nan|nan|...";
            # give each its own group (its filename) so they are never merged.
            def _is_empty_group(g: str) -> bool:
                parts = [p.strip().lower() for p in str(g).split("|")]
                return all(p in ("", "nan", "none") for p in parts)
            mask = joined["group_id"].map(_is_empty_group)
            joined.loc[mask, "group_id"] = joined.loc[mask, "filename"]
        df = joined

    if "group_id" not in df.columns:
        # No metadata to group by; each image is its own group (safe default).
        df["group_id"] = df["filename"]

    df = df.dropna(subset=["label"]).reset_index(drop=True)
    log.info("Built %d labelled records across %d classes",
             len(df), df["label"].nunique())
    return df


if __name__ == "__main__":
    rep = discover()
    print(f"Primary label source: {rep.get('primary_label_source')}")
    recs = build_records(report=rep)
    print(recs.head())
    print(recs["label"].value_counts())
