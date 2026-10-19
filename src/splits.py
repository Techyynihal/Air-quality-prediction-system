"""
Leakage-safe, group-aware, stratified train/val/test split (70/15/15).

Why groups? Exact duplicates, near-duplicates, and images of the same
scene/location/session share visual information. If any member of such a group
lands in both train and test, the test score is optimistic (leakage). We
therefore assign WHOLE groups to a single split.

Grouping key = union of:
  * metadata group_id (from dataset.build_records, e.g. city|date)
  * duplicate/near-duplicate group (from duplicates.detect)

We then greedily allocate groups to splits to approximate the 70/15/15 ratio
AND each class's distribution. The split is deterministic (fixed seed) and
verified to have ZERO group overlap across splits.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from . import config, utils
from . import dataset as ds
from . import duplicates as dup

log = utils.get_logger("splits")


def _merge_groups(df: pd.DataFrame, dup_group: Dict[str, str]) -> pd.Series:
    """Merge metadata group_id with duplicate groups via union-find on keys."""
    # Each row starts in its metadata group; duplicate groups then merge rows.
    parent: Dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    meta_key = df.set_index("path")["group_id"].astype(str).to_dict()
    for path, g in meta_key.items():
        union(f"path::{path}", f"meta::{g}")
    for path, dg in dup_group.items():
        if path in meta_key:
            union(f"path::{path}", f"dup::{dg}")
    return df["path"].map(lambda p: find(f"path::{p}"))


def make_split(seed: int = config.SEED,
               df: pd.DataFrame | None = None,
               dup_map: Dict[str, str] | None = None) -> Dict[str, Any]:
    """Build and persist the leakage-safe split. Returns the manifest.

    ``df`` and ``dup_map`` may be supplied to reuse an already-built records
    table and duplicate mapping (avoids re-scanning the dataset). When omitted
    they are computed here so the function still works standalone.
    """
    config.ensure_dirs()
    utils.set_global_seed(seed)

    if df is None:
        df = ds.build_records()
        from . import cleaning
        df = cleaning.clean(df)
    if dup_map is None:
        dup_map = dup.detect(df)["dup_group"]
    df["leak_group"] = _merge_groups(df, dup_map)

    classes = sorted(map(str, df["label"].unique()))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Per-group class profile (a group is usually single-class, but be safe).
    groups = df.groupby("leak_group")
    group_ids = list(groups.groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(group_ids)

    train_ratio, val_ratio, test_ratio = config.SPLIT_RATIOS
    total = len(df)
    targets = {"train": train_ratio * total,
               "val": val_ratio * total,
               "test": test_ratio * total}
    # Per-class targets to steer stratification.
    class_counts = df["label"].astype(str).value_counts().to_dict()
    class_targets = {sp: {c: r * class_counts[c] for c in classes}
                     for sp, r in zip(("train", "val", "test"),
                                      config.SPLIT_RATIOS)}

    assigned = {"train": [], "val": [], "test": []}
    filled = {"train": defaultdict(int), "val": defaultdict(int),
              "test": defaultdict(int)}

    # Assign larger groups first for better balance.
    group_ids.sort(key=lambda g: len(groups.get_group(g)), reverse=True)
    for g in group_ids:
        sub = groups.get_group(g)
        gcls = sub["label"].astype(str).value_counts().to_dict()
        # Score each split by how much it still needs this group's classes.
        best_split, best_score = "train", -np.inf
        for sp in ("train", "val", "test"):
            deficit = sum((class_targets[sp][c] - filled[sp][c]) * n
                          for c, n in gcls.items())
            # Normalise by remaining capacity so val/test are not starved.
            capacity = targets[sp] - sum(filled[sp].values()) + 1e-6
            score = deficit * (capacity / targets[sp])
            if score > best_score:
                best_score, best_split = score, sp
        assigned[best_split].append(g)
        for c, n in gcls.items():
            filled[best_split][c] += n

    split_frames = {}
    for sp in ("train", "val", "test"):
        mask = df["leak_group"].isin(set(assigned[sp]))
        split_frames[sp] = df[mask].copy()

    _write_splits(split_frames, classes, class_to_idx)
    manifest = _verify_and_manifest(split_frames, classes, seed)
    return manifest


def _rel_to_root(path: str) -> str:
    """Project-root-relative POSIX path so split CSVs are machine-portable."""
    import os
    try:
        return os.path.relpath(path, config.ROOT).replace("\\", "/")
    except Exception:
        return path


def _write_splits(frames, classes, class_to_idx) -> None:
    cols = ["path", "rel_path", "filename", "label", "leak_group"]
    for sp in ("train", "val", "test"):
        frames[sp] = frames[sp].assign(
            rel_path=frames[sp]["path"].map(_rel_to_root))
    frames["train"][cols].to_csv(config.TRAIN_CSV, index=False)
    frames["val"][cols].to_csv(config.VAL_CSV, index=False)
    frames["test"][cols].to_csv(config.TEST_CSV, index=False)
    utils.save_json(config.CLASS_MAP_JSON,
                    {"classes": classes, "class_to_idx": class_to_idx})


def _verify_and_manifest(frames, classes, seed) -> Dict[str, Any]:
    g = {sp: set(frames[sp]["leak_group"]) for sp in frames}
    overlap = {
        "train_val": len(g["train"] & g["val"]),
        "train_test": len(g["train"] & g["test"]),
        "val_test": len(g["val"] & g["test"]),
    }
    manifest = {
        "seed": seed,
        "ratios_target": dict(zip(("train", "val", "test"),
                                  config.SPLIT_RATIOS)),
        "counts": {sp: int(len(frames[sp])) for sp in frames},
        "class_distribution": {
            sp: {str(k): int(v) for k, v in
                 frames[sp]["label"].astype(str).value_counts().items()}
            for sp in frames
        },
        "group_overlap": overlap,
        "zero_leakage": all(v == 0 for v in overlap.values()),
        "classes": classes,
    }
    utils.save_json(config.SPLIT_MANIFEST_JSON, manifest)
    status = "OK (zero leakage)" if manifest["zero_leakage"] else "LEAKAGE!"
    log.info("Split: train=%d val=%d test=%d | %s",
             manifest["counts"]["train"], manifest["counts"]["val"],
             manifest["counts"]["test"], status)
    return manifest


if __name__ == "__main__":
    make_split()
