#!/usr/bin/env python3
"""FX9: rebuild merged summaries ONLY from current per-combo producers."""

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from p3fcl import provenance
from p3fcl.experiment import CHANGED_METHODS, FX9_START

REPO_ROOT = Path(__file__).resolve().parents[2]


def _merge(shared_name, per_combo_glob, key_cols, validate=False):
    files = sorted((REPO_ROOT / "results").glob(per_combo_glob))
    if not files:
        raise ValueError(f"no per-combo files for {shared_name}")
    rows = []
    for path in files:
        with path.open() as f:
            rr = list(csv.DictReader(f))
        if validate and any(r["method"] in CHANGED_METHODS for r in rr):
            meta = json.loads(path.with_suffix(".csv.meta.json").read_text())
            assert meta["utc_start"] >= FX9_START, f"stale FX9 source: {path}"
            assert meta["pbs_jobid"], f"non-PBS FX9 source: {path}"
        rows.extend(rr)
    keys = [tuple(str(r[k]) for k in key_cols) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate per-combo keys in {shared_name}")
    if validate:
        expected = 462 if shared_name.startswith("a1_") else 165
        assert len(rows) == expected, (shared_name, len(rows), expected)
    path = REPO_ROOT / "results" / shared_name
    if path.exists():
        archive = REPO_ROOT / "archive/fx9_shared" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        archive.mkdir(parents=True)
        shutil.copy2(path, archive / path.name)
    rows.sort(key=lambda r: tuple(str(r[k]) for k in key_cols))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = provenance.run_manifest(
        dict(phase="FX9-5", hypothesis="H2", sources=[str(f) for f in files]), seed=0
    )
    provenance.finalize(manifest, [path])
    print(
        f"FX9 MERGE {shared_name} per_combo_files={len(files)} rows={len(rows)} reused_shared_rows=0",
        flush=True,
    )


def main():
    _merge(
        "a1_lira_fixedk_summary.csv",
        "a1_lira_fixedk_summary_*.csv",
        ["dataset", "method", "family", "view", "ablation", "elapsed"],
        validate=True,
    )
    _merge(
        "fig02_halflife.csv",
        "fig02_halflife_*.csv",
        ["dataset", "method", "family", "view", "quantity", "ablation"],
        validate=True,
    )


if __name__ == "__main__":
    main()
