#!/usr/bin/env python3
"""FIG16 from the exact already-calibrated seed-0 fixed-K score sidecars."""

import numpy as np
from build_fig16 import _roc_curve
from build_fx2_summary import _load_seed_npz, _seed_arrays
from fx9_io import ROOT, merge, write
from p3fcl.experiment import DATASETS, METHODS
from p3fcl.shadow_runner import METHOD_REGISTRY

if __name__ == "__main__":
    paths = []
    for ds in DATASETS:
        for method in METHODS:
            scores, labels = _seed_arrays(_load_seed_npz(ds, method, 0, "full"), "trajectory")
            keep = np.isfinite(scores[0])
            fpr, tpr = _roc_curve(scores[0][keep], labels[keep])
            rows = [
                dict(dataset=ds, method=method, family=METHOD_REGISTRY[method][1].value, seed=0, fpr=x, tpr=y)
                for x, y in zip(fpr, tpr)
            ]
            path = ROOT / f"results/fig16_roc_{ds}_{method}.csv"
            write(path, rows, dict(hypothesis="H2", source=f"a1_lira_pertask_{ds}_{method}_seed0_full.npz"))
            paths.append(path)
    merge(paths, ROOT / "results/fig16_roc.csv", ["dataset", "method", "seed", "fpr"], 21 * 201)
