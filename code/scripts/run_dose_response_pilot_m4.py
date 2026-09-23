#!/usr/bin/env python3
"""P5 dose-response PILOT, M4 (semantic retention) arm -- the direct companion to
`run_dose_response_pilot.py`'s M5 (individual/exemplar retention) arm, added 2026-09-21 specifically
to answer the Red Team's "retention is semantic, membership is individual" objection
(`agents/OPEN_QUESTIONS.md`'s H2 entry, `00_BUILD_PLAN.md` P5: "include at least one method whose
retention is purely semantic (M4's prototype half, counts off) and one whose retention is
example-level (M5 Hybrid Replay's latent exemplars). If the dose-response holds for the second and
not the first, that is the finding.").

Same design as the M5 pilot: 1 dataset (CIFAR-100) x 3 levels of M4's `prototype_momentum` knob
(0.0, 0.5, 0.9 -- 0 = each task's class prototype simply overwrites the last, no cross-task value
carry; higher = more blending with the previous released value, i.e. stronger semantic retention) x
3 seeds x a reduced 1,024-shadow budget. `release_counts=False` for the whole pilot (both this and
the M5 arm), so M4's leakage signal here is attributable to the *prototype* channel (F2) alone, not
the F7 counts channel -- the "counts off" condition `00_BUILD_PLAN.md` explicitly asks for.

Writes `results/fig04_semantic_vs_individual.csv` (schema per `03_RESULTS_SPEC.md` FIG04:
`dataset, method, retention_type, knob_value, tpr1, retention_bwt, seed, ci_lo, ci_hi` -- long format,
one row per (method, knob_value) pair aggregated over seeds, tagged with `retention_type` so the two
arms can be plotted together).
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
from p3fcl.methods.m4_proto import PrototypeFCL  # noqa: E402

DATASET = "cifar100"
METHOD = "m4_proto"
RETENTION_TYPE = "semantic"
KNOB_NAME = "prototype_momentum"
LEVELS = [0.0, 0.5, 0.9]
SEEDS = [0, 1, 2]
BACKBONE = "vit_base_patch16_224.augreg_in21k"
N_TASKS, N_CLIENTS, BETA = 10, 10, 0.5
ELAPSED_FOR_LEAK = 5  # same representative mid-horizon point as the M5 arm, for comparability
CALIB_FRAC = 0.8


def _shadow_dir(level, seed: int) -> Path:
    return REPO_ROOT / "shadows" / DATASET / METHOD / f"dose_{KNOB_NAME}_{level}" / f"seed{seed}"


def _run_accuracy(level, seed: int, X, y, n_classes: int, feature_dim: int) -> float:
    stream = streams.build_stream(y, np.arange(len(y)), n_tasks=N_TASKS, n_clients=N_CLIENTS, beta=BETA, seed=seed)
    cfg = {
        "n_classes": n_classes, "feature_dim": feature_dim, "release_counts": False,
        KNOB_NAME: level,
    }
    result = sim.run(PrototypeFCL(cfg), X, y, stream, seed=seed)
    return result["bwt_train"]  # FX4a stopgap: proper eval_sets wiring deferred to FX8


def _leakage_tpr1_at_elapsed(level, seed: int, elapsed: int) -> float:
    store = run_lira.load_shadow_store(_shadow_dir(level, seed))
    targets = store["targets"]
    n_shadows = len(store["shadow_ids"])
    r = rng_mod.seeded(f"run_dose_response_pilot_m4::calib_split::{DATASET}::{METHOD}::{level}::{seed}", 0)
    perm = r.permutation(n_shadows)
    n_calib = int(round(CALIB_FRAC * n_shadows))
    calib_idx, eval_idx = perm[:n_calib], perm[n_calib:]

    surfaces = run_lira.compute_log_lr_surfaces(store, calib_idx)
    traj = surfaces["trajectory"][eval_idx]
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
            "dataset": DATASET, "method": METHOD, "retention_type": RETENTION_TYPE,
            "knob_value": level, "n_seeds": len(bwts),
            "retention_bwt_mean": bwt_mean, "retention_bwt_ci_lo": bwt_lo, "retention_bwt_ci_hi": bwt_hi,
            "tpr1_mean": tpr_mean, "tpr1_ci_lo": tpr_lo, "tpr1_ci_hi": tpr_hi,
            "elapsed": ELAPSED_FOR_LEAK,
        })

    if not rows:
        print("no levels had complete data -- nothing written")
        return 1

    out_csv = REPO_ROOT / "results" / "fig04_semantic_arm_m4.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")

    config = {
        "seed": 0, "purpose": "P5 dose-response PILOT, M4 semantic arm (claim C2, Red Team objection)",
        "dataset": DATASET, "method": METHOD, "knob_name": KNOB_NAME, "levels": LEVELS, "seeds": SEEDS,
    }
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
