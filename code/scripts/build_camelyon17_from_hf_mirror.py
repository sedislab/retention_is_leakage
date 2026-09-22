#!/usr/bin/env python3
"""FIG18 (Camelyon17 natural-federation check, `00_BUILD_PLAN.md`'s "never cut" list): the official
`wilds.get_dataset(dataset="camelyon17", download=True)` path is blocked -- `worksheets.codalab.org`
(the WILDS package's hardcoded download URL) times out on every connection attempt from this login
node (checked directly: TCP connect to 20.232.203.197:443 times out after 10s; general internet
access is fine -- pypi.org, huggingface.co, google.com all respond normally, so this is specific to
that one host, not a general outbound block).

**Workaround, not a new data source**: `wltjr1007/Camelyon17-WILDS` on Hugging Face Hub is a
community re-hosting of the *same* WILDS Camelyon17 release (verified: identical schema -- image
bytes, tumor label, center/hospital id 0-4, patient, node, slide, x/y coordinates -- and identical
per-center row counts to what the official dataset card describes). Camelyon17 itself is CC0 public
domain (see `wilds`'s own `camelyon17_dataset.py` docstring), so a faithful re-hosting carries the
same license.

This script reconstructs the on-disk layout `wilds.datasets.camelyon17_dataset.Camelyon17Dataset`
expects when loaded with `download=False` (`<root>/camelyon17_v1.0/metadata.csv` +
`<root>/camelyon17_v1.0/patches/patient_<p>_node_<n>/patch_patient_<p>_node_<n>_x_<x>_y_<y>.png`) --
NOT to match the original dataset's actual file paths (there is no requirement to), only to be
internally self-consistent so `p3fcl.get_data.prepare_camelyon17()` can run completely unmodified
against it, exactly as it would against a real official download.

Two-phase, matching this project's own image-materialization discipline: the full 455k-row
`metadata.csv` is written first (cheap, no image decoding) so `prepare_camelyon17()`'s existing
class+hospital-stratified subsampling logic can run over the *full* population and pick which
`wilds_index` values it actually wants -- then (phase 2, `--materialize-selected`) only THOSE
specific images are decoded from the parquet shards and written as PNGs, not all 455k.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
HF_DIR = REPO_ROOT / "raw_data" / "camelyon17_hf" / "data"
CAMELYON17_ROOT = REPO_ROOT / "datasets" / "camelyon17_wilds_raw"
VERSIONED_DIR = CAMELYON17_ROOT / "camelyon17_v1.0"
ROW_MAP_PATH = VERSIONED_DIR / "_row_map.json"  # global_row_index -> [parquet_filename, local_row_index]


def _parquet_files() -> list:
    return sorted(HF_DIR.glob("*.parquet"))


def build_metadata() -> None:
    VERSIONED_DIR.mkdir(parents=True, exist_ok=True)
    metadata_csv = VERSIONED_DIR / "metadata.csv"
    row_map = []

    with open(metadata_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "patient", "node", "x_coord", "y_coord", "tumor", "center", "slide", "split"])
        global_idx = 0
        for pq_path in _parquet_files():
            df = pd.read_parquet(pq_path, columns=["label", "center", "patient", "node", "x_coord", "y_coord", "slide"])
            for local_idx, row in enumerate(df.itertuples(index=False)):
                w.writerow([
                    global_idx, int(row.patient), int(row.node), int(row.x_coord), int(row.y_coord),
                    int(row.label), int(row.center), int(row.slide), 0,
                ])
                row_map.append([pq_path.name, local_idx])
                global_idx += 1
            print(f"processed {pq_path.name}: {len(df)} rows (running total {global_idx})", flush=True)
            del df

    ROW_MAP_PATH.write_text(json.dumps(row_map))
    print(f"wrote {metadata_csv} ({global_idx} rows) and {ROW_MAP_PATH}")


def materialize_selected(index_json: Path) -> None:
    """Reads `datasets/camelyon17/index.json` (written by `prepare_camelyon17()`), and decodes +
    writes only the PNG files for the `wilds_index` values it actually selected."""
    index = json.loads(index_json.read_text())
    wanted = sorted({s["wilds_index"] for s in index["samples"]})
    row_map = json.loads(ROW_MAP_PATH.read_text())

    by_file: dict = {}
    for gid in wanted:
        fname, local_idx = row_map[gid]
        by_file.setdefault(fname, []).append((gid, local_idx))

    metadata_csv = VERSIONED_DIR / "metadata.csv"
    meta_rows = pd.read_csv(metadata_csv, index_col=0, dtype={"patient": "str"})

    written = 0
    for fname, pairs in by_file.items():
        local_idx_wanted = {li: gid for gid, li in pairs}
        df = pd.read_parquet(HF_DIR / fname, columns=["image"])
        for local_idx, img_row in enumerate(df.itertuples(index=False)):
            if local_idx not in local_idx_wanted:
                continue
            gid = local_idx_wanted[local_idx]
            meta = meta_rows.loc[gid]
            patch_dir = VERSIONED_DIR / "patches" / f"patient_{meta['patient']}_node_{int(meta['node'])}"
            patch_dir.mkdir(parents=True, exist_ok=True)
            out_path = patch_dir / (
                f"patch_patient_{meta['patient']}_node_{int(meta['node'])}"
                f"_x_{int(meta['x_coord'])}_y_{int(meta['y_coord'])}.png"
            )
            if not out_path.exists():
                out_path.write_bytes(img_row.image["bytes"])
                written += 1
        del df
        print(f"materialized from {fname} (running total {written})", flush=True)

    print(f"wrote {written} PNG files for {len(wanted)} requested wilds_index values")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--materialize-selected":
        materialize_selected(REPO_ROOT / "datasets" / "camelyon17" / "index.json")
    else:
        build_metadata()
    return 0


if __name__ == "__main__":
    sys.exit(main())
