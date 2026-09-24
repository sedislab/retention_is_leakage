#!/usr/bin/env python3
"""Every V3 dose file must fully decompress; count alone is insufficient."""

import numpy as np
from fx9_io import ROOT
from p3fcl.paths import SHADOW_ROOT
from run_dose_response_lira import KNOBS, LEVELS, SEEDS

if __name__ == "__main__":
    total = 0
    for method, knob in KNOBS.items():
        for level in LEVELS:
            for seed in SEEDS:
                path = (
                    ROOT / SHADOW_ROOT[method] / "cifar100" / method / f"dose_{knob}_{level}" / f"seed{seed}"
                )
                files = sorted(path.glob("shadow_*.npz"))
                assert len(files) == 512, (path, len(files))
                assert {int(f.stem.split("_")[-1]) for f in files} == set(range(512)), path
                for file in files:
                    with np.load(file) as z:
                        for key in z.files:
                            _ = z[key]
                        assert int(z["shadow_id"]) == int(file.stem.split("_")[-1])
                        assert np.isfinite(z["scores_full"]).any()
                total += len(files)
                print(f"FX9-6 LOAD ACCEPT {method} {knob}={level} seed={seed} loadable=512", flush=True)
    print(f"FX9-6 LOAD ACCEPT combos=36 loadable={total} bad=0", flush=True)
