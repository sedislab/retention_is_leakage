#!/usr/bin/env python3
"""Merges every per-combo `results/a1_lira_fixedk_summary_<dataset>_<method>_<view>.csv` /
`results/fig02_halflife_<dataset>_<method>_<view>.csv` (written by `build_fx2_summary.py` when run
via `code/scripts/pbs/fx2_leak_wave.pbs`'s manifest-driven array, one file per array task to avoid
concurrent-write races on one shared file) into the single `results/a1_lira_fixedk_summary.csv` /
`results/fig02_halflife.csv` FX2 names. Also folds in whatever those shared files already contain
(e.g. `m0_fedavg`'s standalone run, which writes directly to the shared files since it never runs
concurrently with anything else touching them) -- run this only after every producer has finished, not
concurrently with any of them. No computation, pure concatenation + de-dup by key columns.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import provenance  # noqa: E402


def _merge(shared_name: str, per_combo_glob: str, key_cols: list) -> None:
    shared_path = REPO_ROOT / "results" / shared_name
    rows = []
    if shared_path.exists():
        with open(shared_path, newline="") as f:
            rows.extend(csv.DictReader(f))

    per_combo_files = sorted((REPO_ROOT / "results").glob(per_combo_glob))
    for path in per_combo_files:
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))

    if not rows:
        print(f"no rows found for {shared_name} (no shared file, no per-combo files matching {per_combo_glob})", file=sys.stderr)
        return

    seen = set()
    deduped = []
    for r in reversed(rows):  # later (per-combo) sources win over the shared file's stale copy
        key = tuple(r[c] for c in key_cols)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    deduped.reverse()

    fieldnames = list(rows[0].keys())
    with open(shared_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(deduped)
    print(f"merged {len(per_combo_files)} per-combo files + existing shared file into {shared_path}: {len(deduped)} rows")

    config = {"seed": 0, "purpose": f"FX2 merge of per-combo leak summary files into {shared_name}"}
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [shared_path])


def main() -> int:
    _merge(
        "a1_lira_fixedk_summary.csv", "a1_lira_fixedk_summary_*.csv",
        key_cols=["dataset", "method", "family", "view", "ablation", "elapsed"],
    )
    _merge(
        "fig02_halflife.csv", "fig02_halflife_*.csv",
        key_cols=["dataset", "method", "family", "view", "quantity", "ablation"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
