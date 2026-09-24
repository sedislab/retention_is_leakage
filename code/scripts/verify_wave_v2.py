#!/usr/bin/env python3
"""FX4i (`08_FIX_PLAN.md` §4i): "Accept when every (dataset, method, seed) has 1024 npz files that
all load." Checks file counts AND that every file actually loads (a truncated/corrupt npz from a
killed subjob would still show up in a file count). Login-node safe if under ~2 min; if it grows
past that (many thousand npz files), submit as a PBS job -- do not assume it stays quick.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from p3fcl.paths import V2_SHADOW_ROOT

REPO_ROOT = Path(__file__).resolve().parents[2]

METHODS = ["m1_glfc", "m2_target", "m3_fot", "m5_hybrid_replay", "m4_proto", "m8_analytic"]
DATASETS = ["cifar100", "cub200", "imagenet_r"]
SEEDS = [0, 1, 2]
EXPECTED = 1024


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=REPO_ROOT / V2_SHADOW_ROOT)
    parser.add_argument('--methods', nargs='+', default=METHODS)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    base = args.root
    problems = []
    for dataset in DATASETS:
        for method in args.methods:
            for seed in SEEDS:
                d = base / dataset / method / f"seed{seed}"
                files = sorted(d.glob("shadow_*.npz")) if d.exists() else []
                if len(files) != EXPECTED:
                    problems.append(f"{dataset}/{method}/seed{seed}: {len(files)} files, expected {EXPECTED}")
                    continue
                assert {int(f.stem.split("_")[-1]) for f in files} == set(range(EXPECTED)), d
                bad = []
                for f in files:
                    try:
                        with np.load(f) as z:
                            for key in z.files:
                                _ = z[key]
                            assert z["shadow_id"].shape == ()
                            assert int(z["shadow_id"]) == int(f.stem.split("_")[-1])
                            for view in z["views"]:
                                assert np.isfinite(z[f"scores_{view}"]).any(), view
                            if method == "m4_proto":
                                np.testing.assert_allclose(z["scores_global"], z["scores_aggregate"], atol=1e-10, rtol=0, equal_nan=True)
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
    print(f"FX9-5 LOAD ACCEPT combos={len(DATASETS)*len(args.methods)*len(SEEDS)} loadable={len(DATASETS)*len(args.methods)*len(SEEDS)*EXPECTED} per_combo={EXPECTED} bad=0 M4_global_aggregate_max_error<=1e-10", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
