#!/usr/bin/env python3
"""TAB05 deliverable (Gate P2): one (method, dataset, n_tasks, seed) utility-baseline run on real
cached features. Writes a small per-run JSON to `runs/utility_baseline/` (collected later by
`collect_tab05.py` into `results/tab05_utility_baselines.csv`), with a provenance-stamped manifest.

FX9 uses raw cached features and exactly the shadow runner configuration plus the FX9 gate.

Usage:
  python scripts/run_utility_baseline.py --method M0 --dataset cifar100 --n_tasks 10 --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import features, provenance, sim, streams  # noqa: E402
from p3fcl.experiment import METHODS, attacked_method_config  # noqa: E402
from p3fcl.shadow_runner import METHOD_REGISTRY as SHADOW_METHODS  # noqa: E402

BACKBONE = "vit_base_patch16_224.augreg_in21k"
FEATURES_DIR = REPO_ROOT / "features"
OUT_DIR = REPO_ROOT / "runs" / "fx9_utility_baseline"

METHOD_REGISTRY = {m.split("_")[0].upper(): m for m in METHODS}

# per-dataset stream spec: n_classes, n_clients, beta, domain (None for class-incremental)
DATASET_SPEC = {
    "cifar100": {"n_classes": 100, "n_clients": 10, "beta": 0.5, "domain": False},
    "imagenet_r": {"n_classes": 200, "n_clients": 10, "beta": 0.5, "domain": False},
    "cub200": {"n_classes": 200, "n_clients": 10, "beta": 0.5, "domain": False},
    "camelyon17": {"n_classes": 2, "n_clients": 1, "beta": 0.5, "domain": True},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHOD_REGISTRY))
    ap.add_argument("--dataset", required=True, choices=list(DATASET_SPEC))
    ap.add_argument("--n_tasks", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    spec = DATASET_SPEC[args.dataset]
    cache = features.load_cache(FEATURES_DIR, args.dataset, BACKBONE, "train")
    X, y = cache["features"], cache["labels"]
    test_cache = features.load_cache(FEATURES_DIR, args.dataset, BACKBONE, "test")
    X_test, y_test = test_cache["features"], test_cache["labels"]

    domain_field = None
    domain_field_test = None
    if spec["domain"]:
        # Camelyon17's cached "labels" is the tumor/normal target; hospital domain isn't stored in
        # the generic feature cache (which only knows dataset/backbone/split/features/labels/ids) --
        # recover it from datasets/camelyon17/index.json, keyed by the same wilds_index in `ids`.
        index = json.loads((REPO_ROOT / "datasets" / "camelyon17" / "index.json").read_text())
        by_wilds_idx = {s["wilds_index"]: s["domain"] for s in index["samples"]}
        domain_field = np.array([by_wilds_idx[int(i)] for i in cache["ids"]])
        domain_field_test = np.array([by_wilds_idx[int(i)] for i in test_cache["ids"]])

    idx = np.arange(len(y))
    stream = streams.build_stream(
        y, idx, n_tasks=args.n_tasks, n_clients=spec["n_clients"], beta=spec["beta"],
        seed=args.seed, domain_field=domain_field,
    )
    n_tasks_actual = len(stream)

    # FX4a (08_FIX_PLAN.md R1): held-out test-split accuracy is now primary. `task_eval_sets` uses
    # the exact same task-to-class/domain partition `build_stream` used above (same seed/n_tasks/
    # domain values), so eval-task k lines up with train-task k.
    idx_test = np.arange(len(y_test))
    eval_id_lists = streams.task_eval_sets(
        y_test, idx_test, n_tasks=n_tasks_actual, seed=args.seed, domain_field_eval=domain_field_test,
    )
    eval_sets = [(X_test[ids], y_test[ids]) for ids in eval_id_lists]

    method_id = METHOD_REGISTRY[args.method]
    cls = SHADOW_METHODS[method_id][0]
    cfg = attacked_method_config(method_id, spec["n_classes"], X.shape[1], args.dataset)
    method = cls(cfg)
    result = sim.run(method, X, y, stream, seed=args.seed, eval_sets=eval_sets)

    record = {
        "phase": "FX9",
        "method_config": cfg,
        "method": args.method,
        "dataset": args.dataset,
        "n_tasks_requested": args.n_tasks,
        "n_tasks_actual": n_tasks_actual,
        "seed": args.seed,
        "final_avg_acc": result["final_avg_acc"],
        "bwt": result["bwt"],
        "avg_incremental_acc": result["avg_incremental_acc"],
        "final_avg_acc_train": result["final_avg_acc_train"],
        "bwt_train": result["bwt_train"],
        "avg_incremental_acc_train": result["avg_incremental_acc_train"],
        "n_ledger_records": len(result["ledger"]),
    }
    print(json.dumps(record, indent=2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.method}_{args.dataset}_t{args.n_tasks}_s{args.seed}.json"
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(record))
    tmp_path.replace(out_path)

    config = {"phase": "FX9-4", "method_config": cfg, "seed": args.seed, "method": args.method, "dataset": args.dataset, "n_tasks": args.n_tasks}
    manifest = provenance.run_manifest(config, seed=args.seed)
    provenance.finalize(manifest, [out_path])
    return 0


if __name__ == "__main__":
    sys.exit(main())
