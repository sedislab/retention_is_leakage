#!/usr/bin/env python3
"""P5 dose-response PILOT — explicitly small-scale and preliminary, not the full spec
(`00_BUILD_PLAN.md` P5: >=6 retention-strength levels x >=3 methods x >=2 datasets x 3 seeds). This
pilot is 1 method (M5 HybridReplay) x 1 dataset (CIFAR-100) x 3 levels of `buffer_size_per_class`
(0, 10, 20) x 3 seeds, with a reduced 1024-shadow budget (not the 4,096 headline budget) per
(level, seed) -- deliberately scoped down under a hard deadline
(`notes/2026-09-21_p5_dose_response_pilot.md` explains why and what full validation would need).

For each level, runs the real (non-shadow) accuracy federation to get -BWT (retention strength), and
reads the already-generated shadow store to get A1's TPR@1%FPR at a representative elapsed value
(leakage). Writes `results/fig03_dose_response_pilot.csv`.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

import run_lira  # noqa: E402
from p3fcl import features, metrics, provenance, sim, streams  # noqa: E402
from p3fcl import rng as rng_mod  # noqa: E402
from p3fcl.methods.m5_hybrid_replay import HybridReplay  # noqa: E402

DATASET = "cifar100"
METHOD = "m5_hybrid_replay"
KNOB_NAME = "buffer_size_per_class"
LEVELS = [0, 10, 20]
SEEDS = [0, 1, 2]
BACKBONE = "vit_base_patch16_224.augreg_in21k"
N_TASKS, N_CLIENTS, BETA = 10, 10, 0.5
ELAPSED_FOR_LEAK = 5  # representative mid-horizon point, not just elapsed=0
CALIB_FRAC = 0.8


def _shadow_dir(level: int, seed: int) -> Path:
    return REPO_ROOT / "shadows" / DATASET / METHOD / f"dose_{KNOB_NAME}_{level}" / f"seed{seed}"


def _run_accuracy(level: int, seed: int, X, y, n_classes: int, feature_dim: int) -> float:
    stream = streams.build_stream(y, np.arange(len(y)), n_tasks=N_TASKS, n_clients=N_CLIENTS, beta=BETA, seed=seed)
    cfg = {
        "n_classes": n_classes, "feature_dim": feature_dim, "local_epochs": 30, "lr": 0.5,
        KNOB_NAME: level,
    }
    result = sim.run(HybridReplay(cfg), X, y, stream, seed=seed)
    return result["bwt_train"]  # FX4a stopgap: proper eval_sets wiring deferred to FX8


def _leakage_tpr1_at_elapsed(level: int, seed: int, elapsed: int) -> float:
    store = run_lira.load_shadow_store(_shadow_dir(level, seed))
    targets = store["targets"]
    n_shadows = len(store["shadow_ids"])
    r = rng_mod.seeded(f"run_dose_response_pilot::calib_split::{DATASET}::{METHOD}::{level}::{seed}", 0)
    perm = r.permutation(n_shadows)
    n_calib = int(round(CALIB_FRAC * n_shadows))
    calib_idx, eval_idx = perm[:n_calib], perm[n_calib:]

    surfaces = run_lira.compute_log_lr_surfaces(store, calib_idx)
    traj = surfaces["trajectory"][eval_idx]  # (n_eval, n_targets, n_rounds)
    in_out_eval = store["in_out"][eval_idx]
    n_rounds = store["n_rounds"]

    scores_list, labels_list = [], []
    for j, t in enumerate(targets):
        T = t["task"] + elapsed
        if not (0 <= T < n_rounds):
            continue
        col = traj[:, j, T]
        keep = ~np.isnan(col)
        scores_list.append(col[keep])
        labels_list.append(in_out_eval[keep, j])
    if not scores_list:
        return float("nan")
    scores_arr = np.concatenate(scores_list)
    labels_arr = np.concatenate(labels_list).astype(int)
    report = metrics.membership_report(scores_arr, labels_arr)
    return report["tpr_at_1pct_fpr"]


def main() -> int:
    cache = features.load_cache(REPO_ROOT / "features", DATASET, BACKBONE, "train")
    X, y = cache["features"], cache["labels"]
    n_classes = int(y.max()) + 1
    feature_dim = X.shape[1]

    rows = []
    for level in LEVELS:
        bwts, tprs = [], []
        for seed in SEEDS:
            shadow_dir = _shadow_dir(level, seed)
            if not shadow_dir.exists():
                print(f"(skipping level={level} seed={seed}: {shadow_dir} not found)")
                continue
            bwt = _run_accuracy(level, seed, X, y, n_classes, feature_dim)
            tpr1 = _leakage_tpr1_at_elapsed(level, seed, ELAPSED_FOR_LEAK)
            bwts.append(bwt)
            tprs.append(tpr1)
            print(f"level={level} seed={seed}: bwt={bwt:.4f} tpr1(elapsed={ELAPSED_FOR_LEAK})={tpr1:.4f}")

        if not bwts:
            continue
        bwt_mean, bwt_lo, bwt_hi = metrics.seed_ci(bwts)
        tpr_mean, tpr_lo, tpr_hi = metrics.seed_ci(tprs)
        rows.append({
            "dataset": DATASET, "method": METHOD, "knob_name": KNOB_NAME, "knob_value": level,
            "n_seeds": len(bwts),
            "retention_bwt_mean": bwt_mean, "retention_bwt_ci_lo": bwt_lo, "retention_bwt_ci_hi": bwt_hi,
            "tpr1_mean": tpr_mean, "tpr1_ci_lo": tpr_lo, "tpr1_ci_hi": tpr_hi,
            "elapsed": ELAPSED_FOR_LEAK,
        })

    if not rows:
        print("no levels had complete data -- nothing written")
        return 1

    out_csv = REPO_ROOT / "results" / "fig03_dose_response_pilot.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")

    config = {
        "seed": 0, "purpose": "P5 dose-response PILOT, preliminary/small-scale (claim C2, H2)",
        "dataset": DATASET, "method": METHOD, "knob_name": KNOB_NAME, "levels": LEVELS, "seeds": SEEDS,
    }
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
