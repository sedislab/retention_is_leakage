#!/usr/bin/env python3
"""Gate P2 deliverable: `results/disjointness_report.csv` — the empirical half of hypothesis H6
(build/00_BUILD_PLAN.md: "each method's disjointness report is recorded... it must be produced
*before* any DP claim"). Runs every implemented method through `sim.run` on synthetic features and
records what `dp.accountant.check_disjointness` certifies for its real, honestly-tracked ledger.

This is a legitimate use of synthetic data (CLAUDE.md non-negotiable #3 governs *statistical*
results, not this): task-disjointness is a structural property of a method's release schedule and
`touched` bookkeeping, not of the feature values themselves — a Gaussian blob and a real ViT feature
trigger the identical V1-V5 checks given the same ids and rounds. The "without accuracy collapse"
half of H6 (agents/OPEN_QUESTIONS.md) still needs real per-dataset accuracy numbers once features
exist; this script only settles the formal half.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import provenance, sim  # noqa: E402
from p3fcl.dp.accountant import check_disjointness  # noqa: E402
from p3fcl.methods.m0_fedavg import FedAvgSequential  # noqa: E402
from p3fcl.methods.m1_glfc import GLFC  # noqa: E402
from p3fcl.methods.m2_target import TARGET  # noqa: E402
from p3fcl.methods.m3_fot import FOT  # noqa: E402
from p3fcl.methods.m4_proto import PrototypeFCL  # noqa: E402
from p3fcl.methods.m5_hybrid_replay import HybridReplay  # noqa: E402
from p3fcl.methods.m8_analytic import AnalyticFCL  # noqa: E402
from p3fcl.streams import build_stream  # noqa: E402


def _synthetic(n_classes=6, d=8, per_class=40, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_classes, d)) * 3
    X, y = [], []
    for c in range(n_classes):
        X.append(centers[c] + rng.standard_normal((per_class, d)) * 0.3)
        y.append(np.full(per_class, c))
    return np.concatenate(X), np.concatenate(y)


# (label, method_class, config, default_and_"retention-on" variants to show the mechanism honestly)
CONFIGS = [
    ("M0_fedavg_sequential (local_epochs=1)", FedAvgSequential, {"local_epochs": 1}),
    ("M0_fedavg_sequential (local_epochs=5)", FedAvgSequential, {"local_epochs": 5}),
    ("M1_glfc (no replay, no distill)", GLFC, {"local_epochs": 1, "exemplar_budget": 0, "distillation_weight": 0.0}),
    ("M1_glfc (replay + distill)", GLFC, {"local_epochs": 1, "exemplar_budget": 3, "distillation_weight": 1.0}),
    ("M2_target (replay_ratio=0)", TARGET, {"local_epochs": 1, "replay_ratio": 0.0}),
    ("M2_target (replay_ratio=1)", TARGET, {"local_epochs": 1, "replay_ratio": 1.0}),
    ("M3_fot (projection_strength=1)", FOT, {"local_epochs": 1, "subspace_rank": 2, "projection_strength": 1.0}),
    ("M4_prototype_half (momentum=0, class-IL)", PrototypeFCL, {"prototype_momentum": 0.0}),
    ("M4_prototype_half (momentum=0.9, class-IL)", PrototypeFCL, {"prototype_momentum": 0.9}),
    ("M5_hybrid_replay (buffer=0, no replay)", HybridReplay, {"local_epochs": 1, "buffer_size_per_class": 0}),
    ("M5_hybrid_replay (buffer=4, replay on)", HybridReplay, {"local_epochs": 1, "buffer_size_per_class": 4}),
    ("M8_analytic_fcl", AnalyticFCL, {"ridge_lambda": 1.0}),
]


def main() -> int:
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)

    rows = []
    for label, cls, extra_cfg in CONFIGS:
        cfg = {"n_classes": 6, "feature_dim": X.shape[1], **extra_cfg}
        method = cls(cfg)
        result = sim.run(method, X, y, stream, seed=0)
        report = check_disjointness(result["ledger"])
        rows.append({
            "method": label,
            "family_declared": "+".join(f.value for f in method.spec.families),
            "cacheable": method.spec.cacheable,
            "retention_type": method.spec.retention_type,
            "retention_knob": method.spec.retention_knob_name,
            "task_disjoint": report.task_disjoint,
            "violations": ";".join(report.violations),
            "final_avg_acc_synthetic": round(result["final_avg_acc"], 4),
        })
        print(f"{label}: task_disjoint={report.task_disjoint} violations={report.violations}")

    out_csv = REPO_ROOT / "results" / "disjointness_report.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    config = {"seed": 0, "purpose": "H6 structural disjointness check, synthetic features"}
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv])
    print(f"\nwrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
