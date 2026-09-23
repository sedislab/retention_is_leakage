#!/usr/bin/env python3
"""A1 deliverable (H2/H3/H11's machinery, carries claim C1 alongside A6): cross-task LiRA over the
transcript, on the real shadow store `run_shadows.py` writes. Consumes ONLY the shadow `.npz`
statistics -- never re-runs a federation itself -- per `04_METHODS_AND_ATTACKS.md §3`'s "write
statistics, not ledgers" design.

Calibration/evaluation split (CLAUDE.md non-negotiable #6: never tune on the evaluation split):
shadows are split by a seeded permutation into a calibration set (fits the per-target, per-round
IN/OUT Gaussians LiRA needs) and a held-out evaluation set (supplies the real IN/OUT-labelled score
whose membership report we actually trust). `calibration_frac` is read from `configs/attack_lira.yaml`
and recorded in the run manifest.

Reports two ablations per `elapsed = T - k` bucket (`k` = the target's own task, `T` = the observer's
horizon): `trajectory` (H3's "transcript" arm, sums the per-round log-LR from release to T) and
`last_round` (H3's "checkpoint" arm, uses only round T's log-LR). See `shadow_runner.py`'s module
docstring for why these are expected to be close for M4/M8 per-target (each releases evidence once,
so the trajectory is a constant repeated) but NOT identical once pooled across targets released at
different tasks k -- report whatever the real numbers show.

`compute_log_lr_surfaces` below is a vectorised reimplementation of `attacks.lira.per_round_log_lr` /
`lira_score` (that module's docstring is the readable, unit-tested reference for the exact math;
looping it per-target/per-shadow/per-elapsed at 4,000 shadows is too slow, so this script fits the
Gaussians once with numpy broadcasting instead) -- keep the two in sync if either changes.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import metrics, provenance  # noqa: E402
from p3fcl import rng as rng_mod  # noqa: E402

_EPS = 1e-6


def load_shadow_store(shadow_dir: Path, view: str = "full", max_shadows: int | None = None) -> dict:
    """Load a shadow store written by `shadow_runner._run_one_shadow`.

    FX4h (`08_FIX_PLAN.md` §4h) gave shadow npz files a `views` array plus one `scores_<view>` key
    per applicable view (`full`, `aggregate`, `global`). Pre-fix shadow files predate that and carry
    a single bare `scores` key -- those are treated as view `full` (the only view they ever computed)
    regardless of the `view` argument, so old stores keep loading unchanged.

    `max_shadows`: if given, only `shadow_id < max_shadows` files are loaded (a store's `shadow_*.npz`
    filenames ARE the shadow id, zero-padded, so this is a filename-prefix restriction, not a
    behind-the-scenes resample) -- FX2's `a1_m0_budget_check.csv` needs M0's existing 4096-shadow/
    5-seed store subsampled down to the 1024-shadow budget every other method got in wave V2, without
    regenerating or duplicating any files.
    """
    import json

    targets = json.loads((shadow_dir / "targets.json").read_text())
    paths = sorted(shadow_dir.glob("shadow_*.npz"))
    if max_shadows is not None:
        paths = [p for p in paths if int(p.stem.split("_")[1]) < max_shadows]
    if not paths:
        raise FileNotFoundError(f"no shadow_*.npz files under {shadow_dir} (max_shadows={max_shadows})")

    shadow_ids, all_scores, all_in_out = [], [], []
    ref_target_ids = None
    for p in paths:
        with np.load(p) as d:
            tids = d["target_ids"]
            if ref_target_ids is None:
                ref_target_ids = tids
            elif not np.array_equal(tids, ref_target_ids):
                raise ValueError(f"{p} has a different target_ids ordering than the rest of the store")
            shadow_ids.append(int(d["shadow_id"]))
            if "views" in d.files:
                views_present = {str(v) for v in d["views"]}
                if view not in views_present:
                    raise ValueError(f"{p} has views {sorted(views_present)}, requested {view!r}")
                all_scores.append(d[f"scores_{view}"])
            else:
                if view != "full":
                    raise ValueError(f"{p} is a pre-FX4h shadow file (bare 'scores' key); only view='full' is available")
                all_scores.append(d["scores"])
            all_in_out.append(d["in_out"])

    order = np.argsort(shadow_ids)
    scores = np.stack(all_scores, axis=0)[order]  # (n_shadows, n_targets, n_rounds)
    in_out = np.stack(all_in_out, axis=0)[order]  # (n_shadows, n_targets)
    return {
        "targets": targets,
        "shadow_ids": np.array(shadow_ids)[order],
        "scores": scores,
        "in_out": in_out,
        "n_rounds": scores.shape[2],
        "view": view,
    }


def _masked_mean_std(scores: np.ndarray, mask: np.ndarray) -> tuple:
    """`scores`: (n_shadows, n_targets, n_rounds). `mask`: (n_shadows, n_targets) bool -- which
    shadows count for each target. Returns `(mu, sd)`, each `(n_targets, n_rounds)`, NaN where fewer
    than 2 masked shadows have a defined (non-NaN) score for that (target, round)."""
    m = mask[:, :, None]
    masked = np.where(m, scores, np.nan)
    n = np.sum(~np.isnan(masked), axis=0)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(masked, axis=0)
        sd = np.nanstd(masked, axis=0, ddof=1) + _EPS
    mu = np.where(n >= 2, mu, np.nan)
    sd = np.where(n >= 2, sd, np.nan)
    return mu, sd


def compute_log_lr_surfaces(store: dict, calib_idx: np.ndarray) -> dict:
    """Vectorised per-round log-LR and its cumulative ("trajectory") sum, both `(n_shadows, n_targets,
    n_rounds)`, fit once from the calibration shadows and evaluated for every shadow (including
    calibration ones -- callers must restrict to the evaluation split before reporting)."""
    from scipy.stats import norm

    scores = store["scores"]
    in_out = store["in_out"]
    mu_in, sd_in = _masked_mean_std(scores[calib_idx], in_out[calib_idx])
    mu_out, sd_out = _masked_mean_std(scores[calib_idx], ~in_out[calib_idx])

    with np.errstate(invalid="ignore"):
        log_lr = norm.logpdf(scores, mu_in[None], sd_in[None]) - norm.logpdf(scores, mu_out[None], sd_out[None])

    valid = ~np.isnan(log_lr)
    filled = np.where(valid, log_lr, 0.0)
    cum_sum = np.cumsum(filled, axis=2)
    cum_valid = np.cumsum(valid, axis=2) > 0
    traj = np.where(cum_valid, cum_sum, np.nan)
    return {"log_lr": log_lr, "trajectory": traj}


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print("usage: run_lira.py <dataset> <method> [stream_seed]", file=sys.stderr)
        return 2
    dataset, method = sys.argv[1], sys.argv[2]
    stream_seed = int(sys.argv[3]) if len(sys.argv) == 4 else 0

    with open(REPO_ROOT / "code" / "configs" / "attack_lira.yaml") as f:
        cfg = yaml.safe_load(f)
    calib_frac = float(cfg["shadow"]["calibration_frac"])
    seed = int(cfg["seed"])

    # stream_seed 0 keeps the original (pre-P4) directory layout so every already-computed store and
    # its results CSV stay valid; P4's extra seeds get their own subdirectory and a seed-suffixed CSV.
    shadow_dir = REPO_ROOT / "shadows" / dataset / method
    if stream_seed != 0:
        shadow_dir = shadow_dir / f"seed{stream_seed}"
    store = load_shadow_store(shadow_dir)
    targets = store["targets"]
    n_shadows = len(store["shadow_ids"])
    n_rounds = store["n_rounds"]
    print(f"{dataset}/{method}: {n_shadows} shadows, {len(targets)} targets, {n_rounds} rounds")

    r = rng_mod.seeded(f"run_lira::calib_split::{dataset}::{method}", seed)
    perm = r.permutation(n_shadows)
    n_calib = int(round(calib_frac * n_shadows))
    calib_idx = perm[:n_calib]
    eval_idx = perm[n_calib:]
    print(f"calibration shadows: {len(calib_idx)}, evaluation shadows: {len(eval_idx)} (frac={calib_frac})")

    surfaces = compute_log_lr_surfaces(store, calib_idx)
    in_out_eval = store["in_out"][eval_idx]  # (n_eval, n_targets)

    task_by_target = np.array([t["task"] for t in targets], dtype=int)
    max_elapsed = n_rounds - 1 - int(task_by_target.min())
    elapsed_grid = list(range(0, max_elapsed + 1))

    rows = []
    for ablation, surface in (("trajectory", surfaces["trajectory"]), ("last_round", surfaces["log_lr"])):
        surf_eval = surface[eval_idx]  # (n_eval, n_targets, n_rounds)
        for e in elapsed_grid:
            scores_list, labels_list = [], []
            for j, t in enumerate(targets):
                T = t["task"] + e
                if not (0 <= T < n_rounds):
                    continue
                col = surf_eval[:, j, T]
                lab = in_out_eval[:, j]
                keep = ~np.isnan(col)
                scores_list.append(col[keep])
                labels_list.append(lab[keep])
            if not scores_list:
                continue
            scores_arr = np.concatenate(scores_list)
            labels_arr = np.concatenate(labels_list).astype(int)
            report = metrics.membership_report(scores_arr, labels_arr)
            rows.append({"ablation": ablation, "elapsed": e, **report})
            print(
                f"[{ablation}] elapsed={e}: auc={report['auc']:.4f} "
                f"tpr@1%={report['tpr_at_1pct_fpr']:.4f} tpr@0.1%={report['tpr_at_0.1pct_fpr']:.4f} "
                f"n_pos={report['n_pos']} n_neg={report['n_neg']}"
            )

    suffix = f"_seed{stream_seed}" if stream_seed != 0 else ""
    out_csv = REPO_ROOT / "results" / f"a1_lira_{dataset}_{method}{suffix}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")

    config = {
        "seed": seed, "purpose": "A1 cross-task LiRA membership report (H2/H3/H11)",
        "dataset": dataset, "method": method, "stream_seed": stream_seed, "calibration_frac": calib_frac,
        "n_shadows": n_shadows, "n_calib": len(calib_idx), "n_eval": len(eval_idx),
    }
    manifest = provenance.run_manifest(config, seed=seed)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
