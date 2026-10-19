"""
Duplicate and near-duplicate detection.

- Exact duplicates : MD5 content hash (identical bytes).
- Near duplicates   : perceptual hash (pHash) grouped by Hamming distance via a
                      union-find over a BK-tree-free O(n^2) pass on the (usually
                      modest) pHash set. For very large datasets this is capped.

Outputs results/audit/duplicates_report.json and returns a per-file mapping
`filename -> dup_group_id` used by leakage-safe splitting so identical / almost
identical images never straddle the train/val/test boundary.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import config, utils
from . import dataset as ds

log = utils.get_logger("duplicates")

# Popcount lookup table (0..255) -> number of set bits. Lets us compute Hamming
# distance with plain numpy (works on numpy<2.0, no scipy / np.bitwise_count).
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _md5(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _phash_bits(path: str) -> Optional[np.ndarray]:
    """Return a 64-bit perceptual hash as a packed (8,) uint8 array, or None.

    Uses `imagehash` when available (pHash), otherwise falls back to a
    dependency-free difference-hash (dHash) computed with PIL + numpy so
    near-duplicate detection works on any machine.
    """
    try:
        import imagehash
        from PIL import Image
        with Image.open(path) as im:
            h = imagehash.phash(im.convert("RGB"))  # 8x8 bool matrix
        bits = h.hash.flatten().astype(bool)
        return np.packbits(bits)
    except ImportError:
        pass
    except Exception:
        return None
    # ---- dHash fallback (no imagehash needed) ----
    try:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("L").resize((9, 8))
            arr = np.asarray(im, dtype=np.int16)
        diff = arr[:, 1:] > arr[:, :-1]      # 8x8 bool
        return np.packbits(diff.flatten())
    except Exception:
        return None


class _UnionFind:
    def __init__(self, items: List[str]):
        self.parent = {i: i for i in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def detect(df: pd.DataFrame | None = None,
           max_near_dup_scan: int = 6000,
           include_near_in_groups: bool = False) -> Dict[str, Any]:
    """Detect exact + near duplicates; persist a report and group mapping.

    ``include_near_in_groups`` controls whether near-duplicate clusters are
    folded into the returned ``dup_group`` (which leakage-safe splitting merges
    into split groups). It defaults to **False**: perceptual-hash near-dup
    grouping uses transitive union-find, so on large sets of low-detail images
    (e.g. uniformly hazy "Severe" scenes) it can chain thousands of visually
    similar but genuinely distinct scenes into one giant cluster, which would
    wreck a 70/15/15 split. Near-duplicates are still fully detected and
    reported; same-session leakage is handled precisely by the metadata
    ``group_id`` (location + date + hour) instead. Set True for small datasets
    where transitive chaining is not a concern.
    """
    config.ensure_dirs()
    df = df if df is not None else ds.build_records()
    paths = df["path"].tolist()

    # --- Exact duplicates via MD5 --------------------------------------- #
    md5_groups: Dict[str, List[str]] = defaultdict(list)
    for p in paths:
        try:
            md5_groups[_md5(p)].append(p)
        except Exception:
            continue
    exact_groups = [g for g in md5_groups.values() if len(g) > 1]

    # --- Near duplicates via perceptual hash (vectorised Hamming) ------- #
    scan_paths = paths[:max_near_dup_scan]
    hashed_paths: List[str] = []
    bit_rows: List[np.ndarray] = []
    for p in scan_paths:
        h = _phash_bits(p)
        if h is not None:
            hashed_paths.append(p)
            bit_rows.append(h)

    uf = _UnionFind(hashed_paths)
    if bit_rows:
        hb = np.vstack(bit_rows)               # (N, 8) uint8, packed 64-bit
        n = hb.shape[0]
        thr = config.PHASH_HAMMING_NEAR_DUP
        for i in range(n - 1):
            xor = np.bitwise_xor(hb[i + 1:], hb[i])      # (n-i-1, 8)
            dist = _POPCOUNT[xor].sum(axis=1)            # (n-i-1,)
            for off in np.nonzero(dist <= thr)[0]:
                uf.union(hashed_paths[i], hashed_paths[i + 1 + int(off)])
    near_clusters: Dict[str, List[str]] = defaultdict(list)
    for p in hashed_paths:
        near_clusters[uf.find(p)].append(p)
    near_groups = [g for g in near_clusters.values() if len(g) > 1]

    # --- Combined dup group id per file (for leakage prevention) -------- #
    # Exact duplicates are ALWAYS safe to merge. Near-duplicate clusters are
    # only merged when explicitly requested (see docstring).
    dup_group: Dict[str, str] = {}
    gid = 0
    groups_for_leakage = list(exact_groups)
    if include_near_in_groups:
        groups_for_leakage += near_groups
    for group in groups_for_leakage:
        gname = f"dup_{gid}"
        for p in group:
            dup_group[p] = gname
        gid += 1

    report = {
        "n_files_scanned": len(paths),
        "n_files_phashed": len(hashed_paths),
        "n_exact_groups": len(exact_groups),
        "n_near_groups": len(near_groups),
        "near_dups_merged_into_leakage_groups": include_near_in_groups,
        "near_dup_policy": (
            "Near-duplicate clusters are reported but NOT used as split/leakage "
            "groups by default: transitive perceptual-hash chaining over "
            "low-detail images can merge thousands of distinct scenes. "
            "Same-session leakage is handled precisely by the metadata group_id "
            "(location + date + hour), and exact (MD5) duplicates are always "
            "merged."),
        "largest_exact_group": max((len(g) for g in exact_groups), default=0),
        "largest_near_group": max((len(g) for g in near_groups), default=0),
        "examples_exact": [[Path(x).name for x in g][:6]
                           for g in exact_groups[:5]],
        "examples_near": [[Path(x).name for x in g][:6]
                          for g in near_groups[:5]],
        "note": ("Near-dup scan capped at %d images for tractability; "
                 "raise max_near_dup_scan for exhaustive analysis."
                 % max_near_dup_scan) if len(paths) > max_near_dup_scan else "",
    }
    utils.save_json(config.DUPLICATES_JSON, report)
    log.info("Duplicates: %d exact groups, %d near-duplicate groups",
             len(exact_groups), len(near_groups))
    return {"report": report, "dup_group": dup_group}


if __name__ == "__main__":
    detect()
