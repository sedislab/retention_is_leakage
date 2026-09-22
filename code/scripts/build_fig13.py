#!/usr/bin/env python3
"""FIG13 — Gram inversion n-curve (claim C1's supporting evidence, H5). Aggregates the already-computed
per-trial reconstruction records in `results/fig13_gram_inversion.csv` (5 trials per
(dataset, n_per_class, ref_quality), each trial a different randomly-sampled class -- no new
reconstruction runs here) into a mean + CI curve, since the raw file has per-trial rows, not the
mean/CI columns a plot script can read directly (CLAUDE.md non-negotiable #3: plot scripts do no
computation, so that aggregation step lives here, not in `analysis/fig13_gram_inversion.py`).
Writes `results/fig13_gram_inversion_summary.csv`.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import metrics, provenance  # noqa: E402


def main() -> int:
    in_csv = REPO_ROOT / "results" / "fig13_gram_inversion.csv"
    with open(in_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    groups: dict = defaultdict(list)
    for r in rows:
        key = (r["dataset"], r["backbone"], int(r["n_per_class"]), r["ref_quality"])
        groups[key].append(float(r["mean_cos"]))

    out_rows = []
    for (dataset, backbone, n_per_class, ref_quality), values in groups.items():
        mean, lo, hi = metrics.seed_ci(values)
        out_rows.append({
            "dataset": dataset, "backbone": backbone, "n_per_class": n_per_class,
            "ref_quality": ref_quality, "mean_cos": mean, "ci_lo": lo, "ci_hi": hi,
            "n_trials": len(values),
        })
    out_rows.sort(key=lambda r: (r["dataset"], r["ref_quality"], r["n_per_class"]))

    out_csv = REPO_ROOT / "results" / "fig13_gram_inversion_summary.csv"
    fieldnames = ["dataset", "backbone", "n_per_class", "ref_quality", "mean_cos", "ci_lo", "ci_hi", "n_trials"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows to {out_csv}")

    config = {
        "seed": 0, "purpose": "FIG13 Gram-inversion curve: aggregate per-trial records to mean+CI (H5)",
        "n_input_rows": len(rows), "n_groups": len(groups),
    }
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
