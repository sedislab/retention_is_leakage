#!/usr/bin/env python3
"""TAB05 deliverable (Gate P2): one (method, dataset, n_tasks, seed) utility-baseline run on real
cached features. Writes a small per-run JSON to `runs/utility_baseline/` (collected later by
`collect_tab05.py` into `results/tab05_utility_baselines.csv`), with a provenance-stamped manifest.

Features are z-score standardised (fit on the train split) before being handed to any method. This
is a per-run *modelling* choice for these plain-gradient-descent baselines' numerical stability, not
a change to the stored cache — `features.py` stores unnormalised vectors precisely so this kind of
decision stays a method/runner concern (RESEARCH_PLAN.md, `06_PACKAGE_SPEC.md §4`), not baked into
extraction.

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
from p3fcl.methods.m0_fedavg import FedAvgSequential  # noqa: E402
from p3fcl.methods.m1_glfc import GLFC  # noqa: E402
from p3fcl.methods.m2_target import TARGET  # noqa: E402
from p3fcl.methods.m3_fot import FOT  # noqa: E402
from p3fcl.methods.m4_proto import PrototypeFCL  # noqa: E402
from p3fcl.methods.m5_hybrid_replay import HybridReplay  # noqa: E402
from p3fcl.methods.m8_analytic import AnalyticFCL  # noqa: E402

BACKBONE = "vit_base_patch16_224.augreg_in21k"
FEATURES_DIR = REPO_ROOT / "features"
OUT_DIR = REPO_ROOT / "runs" / "utility_baseline"

# (method_id, class, default hyperparameters — one representative setting, not a dose-response sweep)
METHOD_REGISTRY = {
    "M0": (FedAvgSequential, {"local_epochs": 30, "lr": 0.5}),
    "M1": (GLFC, {"local_epochs": 30, "lr": 0.5, "exemplar_budget": 10, "distillation_weight": 1.0, "temperature": 2.0}),
    "M2": (TARGET, {"local_epochs": 30, "lr": 0.5, "replay_ratio": 1.0, "n_synthetic_per_class": 20}),
    "M3": (FOT, {"local_epochs": 30, "lr": 0.5, "subspace_rank": 8, "projection_strength": 1.0}),
    "M4": (PrototypeFCL, {"prototype_momentum": 0.5, "release_counts": True}),
    "M5": (HybridReplay, {"local_epochs": 30, "lr": 0.5, "buffer_size_per_class": 10}),
    "M8": (AnalyticFCL, {"ridge_lambda": 1.0}),
}

# per-dataset stream spec: n_classes, n_clients, beta, domain (None for class-incremental)
DATASET_SPEC = {
    "cifar100": {"n_classes": 100, "n_clients": 10, "beta": 0.5, "domain": False},
    "imagenet_r": {"n_classes": 200, "n_clients": 10, "beta": 0.5, "domain": False},
    "cub200": {"n_classes": 200, "n_clients": 10, "beta": 0.5, "domain": False},
    "camelyon17": {"n_classes": 2, "n_clients": 1, "beta": 0.5, "domain": True},
}


def _standardize(X_train, *others):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6
    out = [(X_train - mean) / std]
    for X in others:
        out.append((X - mean) / std)
    return out


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

    domain_field = None
    if spec["domain"]:
        # Camelyon17's cached "labels" is the tumor/normal target; hospital domain isn't stored in
        # the generic feature cache (which only knows dataset/backbone/split/features/labels/ids) --
        # recover it from datasets/camelyon17/index.json, keyed by the same wilds_index in `ids`.
        index = json.loads((REPO_ROOT / "datasets" / "camelyon17" / "index.json").read_text())
        by_wilds_idx = {s["wilds_index"]: s["domain"] for s in index["samples"]}
        domain_field_full = np.array([by_wilds_idx[int(i)] for i in cache["ids"]])
        domain_field = domain_field_full

    (X_std,) = _standardize(X)
    idx = np.arange(len(y))
    stream = streams.build_stream(
        y, idx, n_tasks=args.n_tasks, n_clients=spec["n_clients"], beta=spec["beta"],
        seed=args.seed, domain_field=domain_field,
    )

    cls, base_cfg = METHOD_REGISTRY[args.method]
    cfg = {**base_cfg, "n_classes": spec["n_classes"], "feature_dim": X_std.shape[1]}
    method = cls(cfg)
    result = sim.run(method, X_std, y, stream, seed=args.seed)

    n_tasks_actual = len(stream)
    record = {
        "method": args.method,
        "dataset": args.dataset,
        "n_tasks_requested": args.n_tasks,
        "n_tasks_actual": n_tasks_actual,
        "seed": args.seed,
        "final_avg_acc": result["final_avg_acc"],
        "bwt": result["bwt"],
        "avg_incremental_acc": result["avg_incremental_acc"],
        "n_ledger_records": len(result["ledger"]),
    }
    print(json.dumps(record, indent=2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.method}_{args.dataset}_t{args.n_tasks}_s{args.seed}.json"
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(record))
    tmp_path.replace(out_path)

    config = {"seed": args.seed, "method": args.method, "dataset": args.dataset, "n_tasks": args.n_tasks}
    manifest = provenance.run_manifest(config, seed=args.seed)
    provenance.finalize(manifest, [out_path])
    return 0


if __name__ == "__main__":
    sys.exit(main())
