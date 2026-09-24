#!/usr/bin/env python3
"""Preserved scores -> new curve/paired-bootstrap products; no shadow rerun."""

import os

from p3fcl.experiment import DATASETS
from run_fx9_score_combo import run


def main():
    combos = [
        (ds, method, view)
        for ds in DATASETS
        for method in ["m0_fedavg", "m3_fot", "m8_analytic"]
        for view in (["full", "aggregate", "global"] if method == "m8_analytic" else ["full"])
    ]
    ds, method, view = combos[int(os.environ["PBS_ARRAY_INDEX"])]
    if (ds, method) == ("cifar100", "m0_fedavg"):
        # The historical merged row exists, but its per-combo producer was absent.
        run("build_fx2_summary", ds, method, "F1", view, 0, 1, 2, 3, 4)
    else:
        run("build_fx9_preserved_curves", ds, method, view)
    run("build_fx9_joint", ds, method, view)


if __name__ == "__main__":
    main()
