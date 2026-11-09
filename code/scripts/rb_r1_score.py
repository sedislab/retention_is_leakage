#!/usr/bin/env python3
"""R1 LiRA scoring (`rebuttal/R1_PROTOCOL.md`). PBS array, one task per (dataset, method, view),
index in 0..29.

Views: the two release views of the method, `combined`, `combined_once`, and `margin_repro`.
`margin_repro` reruns the paper's own margin attack (paper shadows, `scores_full`) through this
script. It must reproduce `results/a1_lira_fixedk_summary_<ds>_<method>_full.csv` exactly, which is
the check that this pipeline is paired with the paper.

Reuses the paper's code without copying it: `run_lira.load_shadow_store` /
`compute_log_lr_surfaces`, `run_lira_pertask._calibration_key` (the paper's `full` split) and K/E,
and `build_fx2_summary.build_summary` (hierarchical bootstrap, 2,000 replicates, seed 0). Its
`_load_seed_npz` is pointed at the in-memory arrays here, and no file of the paper is written.
Intermediate output: `shadows_rb/_lira/` (per-seed npz plus a per-view summary CSV).
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))
sys.path.insert(0, str(REPO_ROOT / "code" / "scripts"))

import build_fx2_summary  # noqa: E402
import run_lira  # noqa: E402
import run_lira_pertask  # noqa: E402
from p3fcl import provenance  # noqa: E402
from p3fcl import rng as rng_mod  # noqa: E402
from p3fcl.paths import shadow_dir  # noqa: E402
from p3fcl.rebuttal import release_audit as ra  # noqa: E402
from p3fcl.rebuttal.common import attack_config, manifest  # noqa: E402

DATASETS = ["cifar100", "cub200", "imagenet_r"]
METHODS = ["m2_target", "m3_fot"]
SEEDS = [0, 1, 2]
COMBOS = [(d, m, v) for d in DATASETS for m in METHODS
          for v in (*ra.RELEASE_VIEWS[m], "combined", "combined_once", "margin_repro")]
RB_ROOT = REPO_ROOT / "shadows_rb"
LIRA_DIR = RB_ROOT / "_lira"


def _check_aligned(a: dict, b: dict, what: str) -> None:
    if not (np.array_equal(a["shadow_ids"], b["shadow_ids"]) and np.array_equal(a["in_out"], b["in_out"])):
        raise AssertionError(f"{what}: release and paper stores are not shadow-aligned")


def seed_record(dataset: str, method: str, seed: int, view: str, calib_seed: int, calib_frac: float) -> dict:
    """Per-seed arrays in `run_lira_pertask`'s npz layout (eval shadows, K-set targets only)."""
    paper = run_lira.load_shadow_store(shadow_dir(dataset, method, seed, root=REPO_ROOT), view="full")
    n = len(paper["shadow_ids"])
    perm = rng_mod.seeded(run_lira_pertask._calibration_key(dataset, method, "full"), calib_seed).permutation(n)
    n_calib = int(round(calib_frac * n))
    calib_idx, eval_idx = perm[:n_calib], perm[n_calib:]

    if view == "margin_repro":
        surfaces, store = run_lira.compute_log_lr_surfaces(paper, calib_idx), paper
    else:
        rel_view = view if view in ra.RELEASE_VIEWS[method] else "broadcast"
        rel = run_lira.load_shadow_store(RB_ROOT / dataset / method / f"seed{seed}", view=rel_view)
        _check_aligned(rel, paper, f"{dataset}/{method}/seed{seed}")
        rel_s = run_lira.compute_log_lr_surfaces(rel, calib_idx)
        if view in ra.RELEASE_VIEWS[method]:
            surfaces = rel_s
        else:
            mar_s = run_lira.compute_log_lr_surfaces(paper, calib_idx)
            surfaces = ra.combine_log_lr(mar_s["log_lr"], rel_s["log_lr"], once=(view == "combined_once"))
        store = rel

    k_of = np.array([t["task"] for t in store["targets"]])
    k_idx = [j for k in run_lira_pertask.K_SET for j in np.flatnonzero(k_of == k)]
    rec = {
        "k_of_target": k_of[k_idx],
        "surf_trajectory": surfaces["trajectory"][eval_idx][:, k_idx, :],
        "surf_last_round": surfaces["log_lr"][eval_idx][:, k_idx, :],
        "in_out": store["in_out"][eval_idx][:, k_idx],
    }
    LIRA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(LIRA_DIR / f"{dataset}_{method}_seed{seed}_{view}.npz", **rec,
                        calib_idx=calib_idx, eval_idx=eval_idx)
    return rec


def main() -> int:
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ["PBS_ARRAY_INDEX"])
    dataset, method, view = COMBOS[idx]
    cfg = attack_config()
    calib_seed, calib_frac = int(cfg["seed"]), float(cfg["shadow"]["calibration_frac"])
    man = manifest({"seed": calib_seed, "purpose": "rebuttal R1 LiRA scoring", "dataset": dataset,
                    "method": method, "view": view, "seeds": SEEDS, "calibration_key_view": "full",
                    "k_set": run_lira_pertask.K_SET, "e_max": run_lira_pertask.E_MAX,
                    "n_replicates": build_fx2_summary.N_REPLICATES, "protocol": "rebuttal/R1_PROTOCOL.md"})

    records = {s: seed_record(dataset, method, s, view, calib_seed, calib_frac) for s in SEEDS}
    build_fx2_summary._load_seed_npz = lambda _d, _m, s, _v: records[s]
    family = "F6" if method == "m2_target" and view in ra.RELEASE_VIEWS[method] else \
             "F5" if method == "m3_fot" and view in ra.RELEASE_VIEWS[method] else \
             "F1" if view == "margin_repro" else "F1+release"
    rows, _ = build_fx2_summary.build_summary(dataset, method, family, view, SEEDS)

    out = LIRA_DIR / f"summary_{dataset}_{method}_{view}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["elapsed"] in (0, 6):
            print(f"R1-SCORE {dataset}/{method}/{view} [{r['ablation']}] e={r['elapsed']}: "
                  f"tpr1={r['tpr1']:.4f} [{r['tpr1_ci_lo']}, {r['tpr1_ci_hi']}] tpr01={r['tpr01']:.4f} auc={r['auc']:.4f}")
    provenance.finalize(man, [out, *(LIRA_DIR / f"{dataset}_{method}_seed{s}_{view}.npz" for s in SEEDS)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
