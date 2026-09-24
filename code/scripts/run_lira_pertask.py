#!/usr/bin/env python3
"""FX2 (`08_FIX_PLAN.md` §7a): per-task LiRA over a FIXED target population, replacing
`run_lira.py`'s old elapsed grid, which pooled across every target regardless of the target's own
origin task `k` -- as elapsed `e` grows, fewer `k`s have a valid round `k+e` left in the horizon, so
the pooled population silently shrank and shifted composition as `e` increased, confounding H2/H3's
headline half-life numbers with a population-composition effect, not just genuine signal decay.

**Fixed-k set K = {0, 1, 2, 3}, elapsed e in {0, ..., 6}.** These are chosen jointly, not
independently: with this project's headline stream (10 tasks, 0-indexed 0..9), `e=6` is the largest
elapsed time at which EVERY k in K still has a valid round `k+e < 10` (`3+6=9`) -- no e in the
reported range silently drops one of K's contributions. Keep K/E_MAX in sync with `n_tasks` if either
changes.

**Old (pre-FX4h) shadow files** don't store `target_task`/`target_client` per shadow -- this script
recovers them by rebuilding the target pool with `shadow_runner.build_targets` (a pure function of
the base stream + target config, the same one every array task already recomputes independently) and
asserting the rebuilt `target_id`s match the ones actually stored in the npz files, rather than
trusting `targets.json`'s convenience copy.

Usage: `run_lira_pertask.py <dataset> <method> <seed> [view]` (`view` default `full`). PBS only.

Output: `results/a1_lira_pertask_<dataset>_<method>_seed<S>_<view>.csv`, one row per
(ablation, task_k, elapsed) -- `task_k="pooled"` rows are primary (H2/H3's headline numbers, pooled
over K at each e); integer `task_k` rows are the per-k appendix `08_FIX_PLAN.md` §7a asks for. The
per-pair Clopper-Pearson interval on TPR@1%FPR is reported as `cp_lo`/`cp_hi` -- NOT the primary CI
(that's the hierarchical bootstrap over `a1_lira_fixedk_summary.csv`, not yet built by this script).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))
sys.path.insert(0, str(REPO_ROOT / "code" / "scripts"))

import run_lira  # noqa: E402
from p3fcl import features, metrics, provenance, streams  # noqa: E402
from p3fcl import rng as rng_mod  # noqa: E402
from p3fcl.paths import shadow_dir  # noqa: E402
from p3fcl.shadow_runner import METHOD_REGISTRY, build_targets  # noqa: E402

K_SET = [0, 1, 2, 3]
E_MAX = 6
BACKBONE = "vit_base_patch16_224.augreg_in21k"


def _calibration_key(dataset, method, view, file_suffix=""):
    # FX9 M4 global and aggregate are the SAME score in class-incremental streams.
    # Pair their calibration split too, so reporting differences cannot be split noise.
    calibration_view = "aggregate" if method == "m4_proto" and view == "global" else view
    return f"run_lira_pertask::calib_split::{dataset}::{method}::{calibration_view}{file_suffix}"


def _shadow_dir(dataset: str, method: str, seed: int) -> Path:
    return shadow_dir(dataset, method, seed, root=REPO_ROOT)


def _rebuild_and_verify_targets(dataset: str, seed: int, shadow_dir: Path, attack_cfg: dict) -> list:
    stream_cfg = attack_cfg["stream"]
    cache = features.load_cache(REPO_ROOT / "features", dataset, BACKBONE, "train")
    y = cache["labels"]
    base_stream = streams.build_stream(
        y, np.arange(len(y)), n_tasks=stream_cfg["n_tasks"], n_clients=stream_cfg["n_clients"],
        beta=stream_cfg["beta"], seed=seed,
    )
    # `shadow_array.pbs`'s real invocation overrides BOTH `stream.base_seed` and the output dir with
    # the same `SEED` value (`--set stream.base_seed="${SEED}"`) for every non-zero stream seed --
    # `attack_lira.yaml`'s literal `stream.base_seed: 0` only applies to seed 0's original store, so
    # `build_targets` must use `seed` here (matching what actually generated the store), not the
    # static config value, or the id-match assertion below fails for every seed > 0.
    targets = build_targets(base_stream, targets_per_shard=attack_cfg["targets"]["per_shard"], base_seed=seed)

    first_npz = sorted(shadow_dir.glob("shadow_*.npz"))[0]
    with np.load(first_npz) as z:
        stored_ids = [int(i) for i in z["target_ids"]]
    rebuilt_ids = [t["target_id"] for t in targets]
    if rebuilt_ids != stored_ids:
        raise AssertionError(
            f"{shadow_dir}: rebuilt target_ids (n={len(rebuilt_ids)}) do not match the stored "
            f"target_ids (n={len(stored_ids)}) -- the stream/target config used to regenerate the "
            f"pool does not match what actually produced this shadow store"
        )
    return targets


def _report_row(dataset, method, family, view, seed, n_shadows, n_eval_shadows, task_k, elapsed, ablation, n_targets, report):
    return {
        "dataset": dataset, "method": method, "family": family, "view": view, "seed": seed,
        "n_shadows": n_shadows, "n_eval_shadows": n_eval_shadows, "task_k": task_k, "elapsed": elapsed,
        "ablation": ablation, "auc": report["auc"], "tpr1": report["tpr_at_1pct_fpr"],
        "tpr01": report["tpr_at_0.1pct_fpr"], "n_targets": n_targets, "n_pos": report["n_pos"],
        "n_neg": report["n_neg"], "cp_lo": report["tpr_at_1pct_fpr_ci_lo"], "cp_hi": report["tpr_at_1pct_fpr_ci_hi"],
        "tpr01_cp_lo": report["tpr_at_0.1pct_fpr_ci_lo"], "tpr01_cp_hi": report["tpr_at_0.1pct_fpr_ci_hi"],
    }


def _pooled_scores_labels(surf_eval, in_out_eval, idx_by_k: dict, ks: list, e: int, n_rounds: int):
    scores_list, labels_list, n_targets = [], [], 0
    for k in ks:
        T = k + e
        idx_k = idx_by_k.get(k, [])
        if not (0 <= T < n_rounds) or not idx_k:
            continue
        cols = surf_eval[:, idx_k, T]
        labs = in_out_eval[:, idx_k]
        keep = ~np.isnan(cols)
        if not keep.any():
            continue
        scores_list.append(cols[keep])
        labels_list.append(labs[keep])
        n_targets += len(idx_k)
    if not scores_list:
        return None, None, 0
    return np.concatenate(scores_list), np.concatenate(labels_list).astype(int), n_targets


def main() -> int:
    if len(sys.argv) not in (4, 5, 6):
        print("usage: run_lira_pertask.py <dataset> <method> <seed> [view] [max_shadows]", file=sys.stderr)
        return 2
    dataset, method, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    view = sys.argv[4] if len(sys.argv) >= 5 else "full"
    max_shadows = int(sys.argv[5]) if len(sys.argv) == 6 else None
    # FX2's a1_m0_budget_check.csv: restricting to max_shadows must never overwrite the full-budget
    # output files (a1_lira_pertask_<dataset>_<method>_seed<S>_<view>.{csv,npz}) that everything else
    # in the pipeline reads -- give the restricted run its own filename suffix instead.
    file_suffix = f"_max{max_shadows}" if max_shadows is not None else ""

    with open(REPO_ROOT / "code" / "configs" / "attack_lira.yaml") as f:
        attack_cfg = yaml.safe_load(f)
    calib_frac = float(attack_cfg["shadow"]["calibration_frac"])
    calib_seed = int(attack_cfg["seed"])

    shadow_dir = _shadow_dir(dataset, method, seed)
    store = run_lira.load_shadow_store(shadow_dir, view=view, max_shadows=max_shadows)
    n_shadows = len(store["shadow_ids"])
    n_rounds = store["n_rounds"]
    _, family = METHOD_REGISTRY[method]

    targets = _rebuild_and_verify_targets(dataset, seed, shadow_dir, attack_cfg)
    idx_by_k: dict = {}
    for j, t in enumerate(targets):
        idx_by_k.setdefault(t["task"], []).append(j)
    print(f"{dataset}/{method}/seed{seed}/{view}: {n_shadows} shadows, {len(targets)} targets, {n_rounds} rounds")

    r = rng_mod.seeded(_calibration_key(dataset, method, view, file_suffix), calib_seed)
    perm = r.permutation(n_shadows)
    n_calib = int(round(calib_frac * n_shadows))
    calib_idx, eval_idx = perm[:n_calib], perm[n_calib:]

    surfaces = run_lira.compute_log_lr_surfaces(store, calib_idx)
    in_out_eval = store["in_out"][eval_idx]

    rows = []
    for ablation, surface in (("trajectory", surfaces["trajectory"]), ("last_round", surfaces["log_lr"])):
        surf_eval = surface[eval_idx]

        # Per-k appendix rows, every valid (k, e).
        for k in K_SET:
            for e in range(E_MAX + 1):
                scores_arr, labels_arr, n_t = _pooled_scores_labels(surf_eval, in_out_eval, idx_by_k, [k], e, n_rounds)
                if scores_arr is None:
                    continue
                report = metrics.membership_report(scores_arr, labels_arr)
                rows.append(_report_row(dataset, method, family.value, view, seed, n_shadows, len(eval_idx), k, e, ablation, n_t, report))

        # Primary rows: pooled over the fixed-k set K at each e -- the "same target population" claim.
        for e in range(E_MAX + 1):
            scores_arr, labels_arr, n_t = _pooled_scores_labels(surf_eval, in_out_eval, idx_by_k, K_SET, e, n_rounds)
            if scores_arr is None:
                continue
            report = metrics.membership_report(scores_arr, labels_arr)
            rows.append(_report_row(dataset, method, family.value, view, seed, n_shadows, len(eval_idx), "pooled", e, ablation, n_t, report))
            print(f"[{ablation}] e={e} (pooled over K={K_SET}): auc={report['auc']:.4f} "
                  f"tpr1={report['tpr_at_1pct_fpr']:.4f} n_targets={n_t}")

    out_csv = REPO_ROOT / "results" / f"a1_lira_pertask_{dataset}_{method}_seed{seed}_{view}{file_suffix}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")

    # Companion npz: per-K-target raw eval-shadow score/label arrays, restricted to K_SET targets only
    # (not the full target pool) -- this is what the FX2 §7c hierarchical bootstrap resamples at the
    # target level; the CSV above only has pre-pooled summary statistics, which cannot be re-pooled
    # over a resampled target subset without the raw per-target arrays.
    k_target_idx = [j for k in K_SET for j in idx_by_k.get(k, [])]
    k_of_target = np.array([targets[j]["task"] for j in k_target_idx], dtype=int)
    out_npz = REPO_ROOT / "results" / f"a1_lira_pertask_{dataset}_{method}_seed{seed}_{view}{file_suffix}.npz"
    np.savez_compressed(
        out_npz,
        k_of_target=k_of_target,
        surf_trajectory=surfaces["trajectory"][eval_idx][:, k_target_idx, :],
        surf_last_round=surfaces["log_lr"][eval_idx][:, k_target_idx, :],
        in_out=in_out_eval[:, k_target_idx],
        k_set=np.array(K_SET), e_max=E_MAX, n_rounds=n_rounds,
    )
    print(f"wrote {out_npz} ({len(k_target_idx)} K-set targets)")

    config = {
        "seed": calib_seed, "purpose": "FX2 per-task LiRA, fixed-k population (H2/H3/H11)",
        "dataset": dataset, "method": method, "stream_seed": seed, "view": view,
        "calibration_frac": calib_frac, "k_set": K_SET, "e_max": E_MAX,
        "n_shadows": n_shadows, "n_calib": len(calib_idx), "n_eval": len(eval_idx),
        "max_shadows": max_shadows,
    }
    manifest = provenance.run_manifest(config, seed=calib_seed)
    provenance.finalize(manifest, [out_csv, out_npz])
    return 0


if __name__ == "__main__":
    sys.exit(main())
