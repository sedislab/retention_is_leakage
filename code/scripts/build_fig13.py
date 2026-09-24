#!/usr/bin/env python3
"""FX9 FIG13: 25 independent class draws per feasible cell; median of per-trial mean cosine, with percentile bootstrap CI. No anisotropy panel or causal anisotropy claim."""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

import numpy as np  # noqa: E402
from p3fcl import provenance  # noqa: E402
from p3fcl import rng as rng_mod  # noqa: E402


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
        assert len(values) == 25, (dataset, n_per_class, ref_quality, len(values))
        rng = rng_mod.seeded(f"fx9.fig13::{dataset}::{n_per_class}::{ref_quality}", 0)
        values = np.asarray(values)
        draws = rng.choice(values, size=(2000, len(values)), replace=True)
        mean = float(np.median(values))
        lo, hi = np.percentile(np.median(draws, axis=1), [2.5, 97.5])
        out_rows.append({
            "dataset": dataset, "backbone": backbone, "n_per_class": n_per_class,
            "ref_quality": ref_quality, "median_cos": mean, "ci_lo": lo, "ci_hi": hi,
            "n_trials": len(values),
        })
    out_rows.sort(key=lambda r: (r["dataset"], r["ref_quality"], r["n_per_class"]))

    out_csv = REPO_ROOT / "results" / "fig13_gram_inversion_summary.csv"
    fieldnames = ["dataset", "backbone", "n_per_class", "ref_quality", "median_cos", "ci_lo", "ci_hi", "n_trials"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"FX9-8 FIG13 ACCEPT cells={len(out_rows)} trials_per_cell=25 min_trials={min(r['n_trials'] for r in out_rows)} statistic=median bootstrap_replicates=2000")

    config = {
        "seed": 0, "purpose": "FIG13 Gram-inversion curve: median of per-trial mean cosine + bootstrap CI (H5)",
        "n_input_rows": len(rows), "n_groups": len(groups),
    }
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
