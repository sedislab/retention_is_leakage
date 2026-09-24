#!/usr/bin/env python3
"""FX8 (`08_FIX_PLAN.md` §10, C2 dose-response rerun): real (non-shadow) accuracy/BWT per
(method, knob_value, seed) on CIFAR-100's held-out test split, via `eval_sets` (FX4a's fix -- the old
pilot's `_run_accuracy` used `bwt_train` as a stopgap, explicitly not to be cited). Cheap (one real
federation per method x level x seed, not shadows); still goes through qsub per Kodiak policy since
it is real training, however small. Independent of the (much more expensive) shadow generation --
can run before, during, or after it.

Writes `results/dose_response_accuracy.csv`, one row per (method, knob_value, seed):
`dataset, method, knob_name, knob_value, seed, retention_bwt, final_acc`. This is later joined with
the leakage side (`run_dose_response_lira.py`'s output) into the FIG03/FIG04 v2 schemas by
`build_fig03_dose_response.py`/`build_fig04_semantic_vs_individual.py`.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import features, provenance, sim, streams  # noqa: E402
from p3fcl.experiment import attacked_method_config  # noqa: E402
from p3fcl.methods.m2_target import TARGET  # noqa: E402
from p3fcl.methods.m5_hybrid_replay import HybridReplay  # noqa: E402

BACKBONE = "vit_base_patch16_224.augreg_in21k"
DATASET = "cifar100"
N_TASKS, N_CLIENTS, BETA = 10, 10, 0.5
SEEDS = [0, 1, 2]
LEVELS = [1, 2, 5, 10, 20, 50]

# Same TAB05 headline base config as shadow_runner._method_config, with the dose-response knob
# overridden per level below -- not an attack-specific weakening.
METHODS = {
    "m5_hybrid_replay": (HybridReplay, "buffer_size_per_class"),
    "m2_target": (TARGET, "n_synthetic_per_class"),
}


def main() -> int:
    cache = features.load_cache(REPO_ROOT / "features", DATASET, BACKBONE, "train")
    X, y = cache["features"], cache["labels"]
    n_classes = int(y.max()) + 1
    feature_dim = X.shape[1]

    test_cache = features.load_cache(REPO_ROOT / "features", DATASET, BACKBONE, "test")
    X_test, y_test = test_cache["features"], test_cache["labels"]
    idx_test = np.arange(len(y_test))

    combos = [(m, level, seed) for m in METHODS for level in LEVELS for seed in SEEDS]
    selected = combos[int(os.environ["PBS_ARRAY_INDEX"])]
    manifest = provenance.run_manifest(dict(phase="FX9-6", hypothesis="H2", combo=selected), seed=selected[2])
    rows = []
    for method_name, (method_cls, knob_name) in METHODS.items():
        for level in LEVELS:
            for seed in SEEDS:
                if (method_name, level, seed) != selected:
                    continue
                stream = streams.build_stream(
                    y,
                    np.arange(len(y)),
                    n_tasks=N_TASKS,
                    n_clients=N_CLIENTS,
                    beta=BETA,
                    seed=seed,
                )
                eval_id_lists = streams.task_eval_sets(y_test, idx_test, n_tasks=N_TASKS, seed=seed)
                eval_sets = [(X_test[ids], y_test[ids]) for ids in eval_id_lists]

                cfg = {
                    **attacked_method_config(method_name, n_classes, feature_dim, DATASET),
                    knob_name: level,
                }
                method = method_cls(cfg)
                result = sim.run(method, X, y, stream, seed=seed, eval_sets=eval_sets)
                rows.append(
                    {
                        "dataset": DATASET,
                        "method": method_name,
                        "knob_name": knob_name,
                        "knob_value": level,
                        "seed": seed,
                        "retention_bwt": result["bwt"],
                        "final_acc": result["final_avg_acc"],
                    }
                )
                print(
                    f"{method_name} {knob_name}={level} seed={seed}: "
                    f"final_acc={result['final_avg_acc']:.4f} bwt={result['bwt']:.4f}"
                )

    assert len(rows) == 1
    method, level, seed = selected
    out_csv = REPO_ROOT / "results" / f"dose_response_accuracy_{method}_{level}_seed{seed}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")

    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
