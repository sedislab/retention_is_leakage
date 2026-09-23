#!/usr/bin/env python3
"""FX4i (`08_FIX_PLAN.md` §4i): "Accept when every (dataset, method, seed) has 1024 npz files that
all load." Checks file counts AND that every file actually loads (a truncated/corrupt npz from a
killed subjob would still show up in a file count). Login-node safe if under ~2 min; if it grows
past that (many thousand npz files), submit as a PBS job -- do not assume it stays quick.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

METHODS = ["m1_glfc", "m2_target", "m3_fot", "m5_hybrid_replay", "m4_proto", "m8_analytic"]
DATASETS = ["cifar100", "cub200", "imagenet_r"]
SEEDS = [0, 1, 2]
EXPECTED = 1024


def main() -> int:
    base = REPO_ROOT / "shadows_v2"
    problems = []
    for dataset in DATASETS:
        for method in METHODS:
            for seed in SEEDS:
                d = base / dataset / method / f"seed{seed}"
                files = sorted(d.glob("shadow_*.npz")) if d.exists() else []
                if len(files) != EXPECTED:
                    problems.append(f"{dataset}/{method}/seed{seed}: {len(files)} files, expected {EXPECTED}")
                    continue
                bad = []
                for f in files:
                    try:
                        with np.load(f) as z:
                            _ = z["scores_full"]
                    except Exception as e:  # noqa: BLE001
                        bad.append((f.name, str(e)))
                if bad:
                    problems.append(f"{dataset}/{method}/seed{seed}: {len(bad)} unloadable files, e.g. {bad[0]}")
                else:
                    print(f"OK  {dataset}/{method}/seed{seed}: {EXPECTED}/{EXPECTED} load")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  {p}")
        return 1
    print("\nAll (dataset, method, seed) combinations have 1024 loadable npz files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
