#!/usr/bin/env python3
"""FX8 (`08_FIX_PLAN.md` §10, C2 dose-response rerun): per-(method, knob_value, seed) leakage number
for FIG03/FIG04 v2. Reuses `run_lira_pertask.py`'s fixed-k population machinery (`K_SET=[0,1,2,3]`,
`E_MAX=6`, `_pooled_scores_labels`, `_rebuild_and_verify_targets`) rather than reinventing it -- FX2
already solved "how do you get a leakage number that's comparable across different configurations of
the same method," which is exactly FX8's cross-LEVEL comparability problem too.

**Elapsed choice**: the target schema (`03_RESULTS_SPEC.md` FIG03/FIG04) has no `elapsed` column, so
exactly one representative point is needed, not a curve. Uses `elapsed=E_MAX=6` (not 0): C2's claim is
that retention strength causally drives leakage, and both M5 (exemplar replay) and M2 (generative
replay) affect leakage mainly through what gets carried into LATER training via the replay
buffer/generator -- measuring at elapsed=0 (the target's own immediate release) would undersell
exactly the mechanism this figure is trying to show. `E_MAX=6` reuses FX2's own already-vetted horizon
choice (the largest elapsed at which every k in K_SET still has a valid round, given n_tasks=10) rather
than inventing a new one.

Reads shadow stores from `shadows_v2/cifar100/<method>/dose_<knob_name>_<level>/seed<seed>/` (written
by the FX8 manifest-driven PBS array, `code/scripts/pbs/fx8_dose_response.pbs`). Writes
`results/dose_response_lira.csv`, one row per (method, knob_value, seed):
`dataset, method, knob_name, knob_value, seed, tpr1, tpr01, ci_lo, ci_hi, auc, n_targets`.
`ci_lo`/`ci_hi` are the Clopper-Pearson interval on `tpr1` specifically (matching FIG03's schema,
which pairs one `ci_lo`/`ci_hi` with `tpr1` as the headline metric; `tpr01`'s own CI is not part of
the target schema but is computed anyway and available in this CSV for the appendix).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))
sys.path.insert(0, str(REPO_ROOT / "code" / "scripts"))

import run_lira  # noqa: E402
from p3fcl import metrics, provenance  # noqa: E402
from p3fcl import rng as rng_mod  # noqa: E402
from run_lira_pertask import E_MAX, K_SET, _pooled_scores_labels, _rebuild_and_verify_targets  # noqa: E402

DATASET = "cifar100"
KNOBS = {"m5_hybrid_replay": "buffer_size_per_class", "m2_target": "n_synthetic_per_class"}
LEVELS = [1, 2, 5, 10, 20, 50]
SEEDS = [0, 1, 2]


def _shadow_dir(method: str, knob_name: str, level: int, seed: int) -> Path:
    return REPO_ROOT / "shadows_v2" / DATASET / method / f"dose_{knob_name}_{level}" / f"seed{seed}"


def main() -> int:
    with open(REPO_ROOT / "code" / "configs" / "attack_lira.yaml") as f:
        attack_cfg = yaml.safe_load(f)
    calib_frac = float(attack_cfg["shadow"]["calibration_frac"])

    rows = []
    for method, knob_name in KNOBS.items():
        for level in LEVELS:
            for seed in SEEDS:
                shadow_dir = _shadow_dir(method, knob_name, level, seed)
                if not shadow_dir.exists() or not any(shadow_dir.glob("shadow_*.npz")):
                    print(f"(skipping {method} {knob_name}={level} seed={seed}: {shadow_dir} not found)")
                    continue

                store = run_lira.load_shadow_store(shadow_dir)
                n_shadows = len(store["shadow_ids"])
                targets = _rebuild_and_verify_targets(DATASET, seed, shadow_dir, attack_cfg)
                idx_by_k: dict = {}
                for j, t in enumerate(targets):
                    idx_by_k.setdefault(t["task"], []).append(j)

                r = rng_mod.seeded(
                    f"run_dose_response_lira::calib_split::{DATASET}::{method}::{knob_name}::{level}::{seed}",
                    int(attack_cfg["seed"]),
                )
                perm = r.permutation(n_shadows)
                n_calib = int(round(calib_frac * n_shadows))
                calib_idx, eval_idx = perm[:n_calib], perm[n_calib:]

                surfaces = run_lira.compute_log_lr_surfaces(store, calib_idx)
                traj_eval = surfaces["trajectory"][eval_idx]
                in_out_eval = store["in_out"][eval_idx]
                n_rounds = store["n_rounds"]

                scores_arr, labels_arr, n_t = _pooled_scores_labels(
                    traj_eval, in_out_eval, idx_by_k, K_SET, E_MAX, n_rounds,
                )
                if scores_arr is None:
                    print(f"(skipping {method} {knob_name}={level} seed={seed}: no usable pooled scores at e={E_MAX})")
                    continue

                report = metrics.membership_report(scores_arr, labels_arr)
                rows.append({
                    "dataset": DATASET, "method": method, "knob_name": knob_name, "knob_value": level,
                    "seed": seed, "tpr1": report["tpr_at_1pct_fpr"], "tpr01": report["tpr_at_0.1pct_fpr"],
                    "ci_lo": report["tpr_at_1pct_fpr_ci_lo"], "ci_hi": report["tpr_at_1pct_fpr_ci_hi"],
                    "auc": report["auc"], "n_targets": n_t,
                })
                print(f"{method} {knob_name}={level} seed={seed}: elapsed={E_MAX} tpr1={report['tpr_at_1pct_fpr']:.4f} "
                      f"auc={report['auc']:.4f} n_targets={n_t}")

    if not rows:
        print("no (method, level, seed) combos had complete data -- nothing written")
        return 1

    out_csv = REPO_ROOT / "results" / "dose_response_lira.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")

    config = {
        "seed": 0, "purpose": f"FX8 dose-response leakage, pooled K={K_SET} at elapsed={E_MAX} (claim C2, H2)",
        "dataset": DATASET, "knobs": KNOBS, "levels": LEVELS, "seeds": SEEDS,
    }
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
