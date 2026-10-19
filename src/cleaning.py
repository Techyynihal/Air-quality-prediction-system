"""
Data cleaning pipeline.

Removes only files that are genuinely unusable, each with a documented reason:
  * corrupted / unreadable / truncated images
  * unsupported formats
  * records with a missing/invalid label
  * redundant EXACT duplicate copies (one representative per md5 group is kept)

Deliberately does NOT remove:
  * near-duplicates (may be legitimate; handled by leakage-safe grouping instead)
  * "hard" or visually difficult images (removing them would inflate metrics)

Writes results/audit/cleaning_report.json and returns the cleaned records.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from PIL import Image

from . import config, utils
from . import dataset as ds

log = utils.get_logger("cleaning")


def _readable(path: str) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean(df: pd.DataFrame | None = None,
          drop_exact_duplicates: bool = True) -> pd.DataFrame:
    """Return cleaned records and persist a documented cleaning report."""
    config.ensure_dirs()
    df = df.copy() if df is not None else ds.build_records()
    removed: List[Dict[str, Any]] = []
    keep_mask = []

    seen_md5: Dict[str, str] = {}
    for _, row in df.iterrows():
        path = row["path"]
        reason = None

        if Path(path).suffix.lower() not in config.IMAGE_EXTENSIONS:
            reason = "unsupported_format"
        elif not Path(path).exists():
            reason = "missing_file"
        elif not _readable(path):
            reason = "corrupted_or_unreadable"
        elif pd.isna(row.get("label")) or str(row.get("label")).strip() == "":
            reason = "missing_or_invalid_label"
        elif drop_exact_duplicates:
            try:
                digest = _md5(path)
                if digest in seen_md5:
                    reason = f"exact_duplicate_of:{Path(seen_md5[digest]).name}"
                else:
                    seen_md5[digest] = path
            except Exception:
                reason = "unreadable_bytes"

        if reason is None:
            keep_mask.append(True)
        else:
            keep_mask.append(False)
            removed.append({"file": Path(path).name, "reason": reason})

    cleaned = df[pd.Series(keep_mask, index=df.index)].reset_index(drop=True)

    report = {
        "input_records": int(len(df)),
        "kept_records": int(len(cleaned)),
        "removed_records": int(len(removed)),
        "removed_by_reason": _count_reasons(removed),
        "removed_detail": removed[:200],
        "policy": {
            "kept_hard_examples": True,
            "kept_near_duplicates": True,
            "dropped_exact_duplicates": drop_exact_duplicates,
            "rationale": ("Hard and near-duplicate images are retained to avoid "
                          "inflating metrics; only unusable files and redundant "
                          "exact copies are removed."),
        },
    }
    utils.save_json(config.CLEANING_REPORT_JSON, report)
    log.info("Cleaning: kept %d / %d records (removed %d)",
             len(cleaned), len(df), len(removed))
    return cleaned


def _count_reasons(removed: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in removed:
        key = r["reason"].split(":")[0]
        out[key] = out.get(key, 0) + 1
    return out


if __name__ == "__main__":
    clean()
