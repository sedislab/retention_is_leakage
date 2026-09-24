#!/usr/bin/env python3
"""Current V3 score producers, then their bootstrap and paired horizon producer."""

import os
import subprocess
import sys

from p3fcl.experiment import DATASETS
from p3fcl.shadow_runner import METHOD_REGISTRY


def run(script, *args):
    subprocess.run([sys.executable, f"code/scripts/{script}.py", *map(str, args)], check=True)


def main():
    combos = [
        (ds, method, view)
        for ds in DATASETS
        for method in ["m1_glfc", "m2_target", "m4_proto", "m5_hybrid_replay"]
        for view in (["full", "aggregate", "global"] if method == "m4_proto" else ["full"])
    ]
    ds, method, view = combos[int(os.environ["PBS_ARRAY_INDEX"])]
    # Existing outputs are from V2 and MUST be replaced, never resumed by existence.
    for seed in range(3):
        run("run_lira_pertask", ds, method, seed, view)
    run("build_fx2_summary", ds, method, METHOD_REGISTRY[method][1].value, view, 0, 1, 2)
    run("build_fx9_joint", ds, method, view)


if __name__ == "__main__":
    main()
