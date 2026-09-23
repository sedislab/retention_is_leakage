#!/usr/bin/env python3
"""FX5 M9 audit (`08_FIX_PLAN.md` §8, the empirical check of Theorem 11; see
`notes/2026-09-23_m9_audit_design_notes.md` for the `shadow_runner.py` subtleties this resolves,
all fixed there before this script was written -- PCA-space dimension mismatch in GRAM `global`/
`aggregate` scoring, `gamma`-decay generalization, fitting the PCA basis once per eps rather than
once per shadow, and a fourth one found while writing this script: `noise_seed` must vary per shadow
or every shadow draws identical DP noise, which would audit a mechanism with no real privacy).

Online LiRA against M9 on CIFAR-100, U1, eps in {1, 4, inf}, gamma=1 (recovers M8's unbounded running
sum -- this audit is about whether the PER-RELEASE analytic-Gaussian mechanism holds up under attack,
not about the contractive C3/C4 story), 1024 shadows x 1 seed, scored by leverage on the `global`
state (the running R/Q the server actually inverts to predict -- the real retention measurement; M9's
`full` view is structurally undefined since it only ever releases one `client=-1` aggregate per task,
never a per-client record). (p, lambda, clip_C) per eps come from `results/m9_hparam_selection.csv`'s
already-selected row (ref-split-only selection, CLAUDE.md non-negotiable #6) -- never re-tuned here.

Two phases, resumable independently (shadow generation is the expensive, PBS-only part; scoring is
CPU-light login-node-safe post-processing of the already-written shadow stores):
    python run_m9_audit.py --shadows --workers 16     # PBS only
    python run_m9_audit.py --score                    # writes results/fig10_m9_audit.csv

Compares the empirical ROC (elapsed=0, trajectory ablation -- this project's standard, primary MIA
setting, matching FIG16's own convention) against the analytic Gaussian mechanism's DP bound
`TPR <= e^eps * FPR + delta`. Per the plan: if the empirical curve exceeds the bound beyond its
Clopper-Pearson CI anywhere, stop the M9 line, debug it, and log what was found -- this script's
`--score` phase prints that check plainly rather than only rendering a figure.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))
sys.path.insert(0, str(REPO_ROOT / "code" / "scripts"))

import run_lira  # noqa: E402
from p3fcl import features, metrics, provenance, shadow_runner  # noqa: E402
from p3fcl import rng as rng_mod  # noqa: E402
from run_m9_hparam_selection import _fit_pca  # noqa: E402

BACKBONE = "vit_base_patch16_224.augreg_in21k"
DATASET = "cifar100"
UNIT = "U1"
GAMMA = 1.0
DELTA = 1e-5
N_SHADOWS = 1024
EPS_LIST = [1.0, 4.0, float("inf")]
with open(REPO_ROOT / "code" / "configs" / "attack_lira.yaml") as _f:
    CALIB_FRAC = float(yaml.safe_load(_f)["shadow"]["calibration_frac"])  # same calib/eval split every other A1 script uses
CALIB_SEED = 0
N_FPR_GRID = 200


def _eps_tag(eps: float) -> str:
    return "inf" if np.isinf(eps) else str(int(eps))


def _hparams_for(eps: float) -> tuple:
    path = REPO_ROOT / "results" / "m9_hparam_selection.csv"
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if (r["dataset"] == DATASET and r["unit"] == UNIT and r["selected"] == "1"
                    and float(r["eps"]) == eps):
                return int(r["p"]), float(r["lambda"]), float(r["clip_C"])
    raise ValueError(f"no selected m9_hparam_selection.csv row for dataset={DATASET} unit={UNIT} eps={eps}")


def _shadow_dir(eps: float) -> Path:
    return REPO_ROOT / "shadows_v2" / DATASET / "m9_contractive" / f"eps{_eps_tag(eps)}"


def generate_shadows(workers: int) -> None:
    cache = features.load_cache(REPO_ROOT / "features", DATASET, BACKBONE, "ref")
    X_ref = cache["features"]
    for eps in EPS_LIST:
        p, lam, clip_C = _hparams_for(eps)
        P = _fit_pca(X_ref, p_max=p, seed=0)
        config = {
            "stream": {"n_tasks": 10, "n_clients": 10, "beta": 0.5, "base_seed": 0},
            "targets": {"per_shard": 5, "p_in": 0.5},
            "method_config_override": {
                "pca_basis": P, "gamma": GAMMA, "eps": eps, "delta": DELTA,
                "ridge_lambda": lam, "unit": UNIT, "clip_C": clip_C, "B": 1.0,
            },
        }
        out_dir = _shadow_dir(eps)
        manifest = provenance.run_manifest(
            {"seed": 0, "purpose": "M9 audit shadow generation (08_FIX_PLAN.md §8)", "dataset": DATASET,
             "eps": eps, "p": p, "lambda": lam, "clip_C": clip_C, "gamma": GAMMA, "unit": UNIT,
             "n_shadows": N_SHADOWS},
            seed=0,
        )
        result = shadow_runner.run_shadow_range(
            dataset=DATASET, method="m9_contractive", start=0, count=N_SHADOWS,
            workers=workers, out_dir=out_dir, config=config,
        )
        print(f"eps={eps} (p={p}, lambda={lam}, clip_C={clip_C:.3f}): "
              f"{result['n_written']} written, {result['n_skipped']} skipped, "
              f"{result['n_targets']} targets, {result['n_rounds']} rounds -> {out_dir}")
        provenance.finalize(manifest, result["outputs"])


def _roc_curve(scores: np.ndarray, labels: np.ndarray, n_points: int = N_FPR_GRID):
    """Same construction as `build_fig16.py::_roc_curve` -- exact empirical ROC, read off a
    log-spaced FPR grid via monotonic interpolation of the step function."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos, n_neg = int(np.sum(labels == 1)), int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return np.array([]), np.array([])
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    tpr_full = np.concatenate([[0.0], np.cumsum(labels_sorted == 1) / n_pos])
    fpr_full = np.concatenate([[0.0], np.cumsum(labels_sorted == 0) / n_neg])
    fpr_grid = np.concatenate([[0.0], np.logspace(-4, 0, n_points)])
    tpr_grid = np.interp(fpr_grid, fpr_full, tpr_full)
    return fpr_grid, tpr_grid


def score_audit() -> int:
    out_rows = []
    summary_rows = []
    violations = []
    for eps in EPS_LIST:
        shadow_dir = _shadow_dir(eps)
        if not shadow_dir.exists():
            print(f"eps={eps}: {shadow_dir} does not exist yet -- run --shadows first")
            return 1
        store = run_lira.load_shadow_store(shadow_dir, view="global")
        n_shadows = len(store["shadow_ids"])
        r = rng_mod.seeded(f"run_m9_audit::calib_split::{DATASET}::eps{_eps_tag(eps)}", CALIB_SEED)
        perm = r.permutation(n_shadows)
        n_calib = int(round(CALIB_FRAC * n_shadows))
        calib_idx, eval_idx = perm[:n_calib], perm[n_calib:]

        surfaces = run_lira.compute_log_lr_surfaces(store, calib_idx)
        targets = store["targets"]
        traj_eval = surfaces["trajectory"][eval_idx]
        in_out_eval = store["in_out"][eval_idx]

        scores_list, labels_list = [], []
        for j, t in enumerate(targets):
            T = t["task"]  # elapsed=0
            if not (0 <= T < store["n_rounds"]):
                continue
            col = traj_eval[:, j, T]
            lab = in_out_eval[:, j]
            keep = ~np.isnan(col)
            scores_list.append(col[keep])
            labels_list.append(lab[keep])
        if not scores_list:
            print(f"eps={eps}: no usable elapsed=0 scores, skipping")
            continue
        scores_arr = np.concatenate(scores_list)
        labels_arr = np.concatenate(labels_list).astype(int)
        n_pos, n_neg = int(labels_arr.sum()), int((1 - labels_arr).sum())

        # The full log-log curve, for FIG10's plot only -- 200 points read off the empirical step
        # function by monotonic interpolation (same construction as `build_fig16.py`). No CI claim
        # is attached to any individual interpolated point; the pass/fail check below is separate and
        # deliberately does NOT use this curve.
        fpr, tpr = _roc_curve(scores_arr, labels_arr)
        bound = np.exp(eps) * fpr + DELTA if not np.isinf(eps) else np.ones_like(fpr)
        for f_, t_, b_ in zip(fpr, tpr, bound):
            out_rows.append({
                "dataset": DATASET, "unit": UNIT, "eps": eps, "gamma": GAMMA, "delta": DELTA,
                "fpr": f_, "tpr_empirical": t_, "tpr_dp_bound": min(b_, 1.0),
                "n_shadows": n_shadows, "n_pos": n_pos, "n_neg": n_neg,
            })

        # The actual pass/fail check, at the two CLAUDE.md-mandated headline FPR points (1%, 0.1%):
        # `membership_report`'s own `tpr_at_Xpct_fpr_ci_hi` is an EXACT Clopper-Pearson CI computed
        # from the real (k hits, n_pos) count at that exact threshold -- not backed out from an
        # interpolated curve point, which the first version of this check did and which produced
        # spurious "violations" from interpolation/tie artifacts on small samples (caught by this
        # script's own synthetic test, `test_run_m9_audit.py`, before ever touching real shadows).
        report = metrics.membership_report(scores_arr, labels_arr)
        exceeds = []
        if not np.isinf(eps):
            for fpr_target, tpr_key, ci_hi_key in (
                (0.01, "tpr_at_1pct_fpr", "tpr_at_1pct_fpr_ci_hi"),
                (0.001, "tpr_at_0.1pct_fpr", "tpr_at_0.1pct_fpr_ci_hi"),
            ):
                bound_here = min(np.exp(eps) * fpr_target + DELTA, 1.0)
                if report[ci_hi_key] > bound_here + 1e-9 and report[tpr_key] > bound_here + 1e-6:
                    exceeds.append((fpr_target, report[tpr_key], report[ci_hi_key], bound_here))
            if exceeds:
                violations.append((eps, exceeds))

        bound_1pct = min(np.exp(eps) * 0.01 + DELTA, 1.0) if not np.isinf(eps) else 1.0
        bound_01pct = min(np.exp(eps) * 0.001 + DELTA, 1.0) if not np.isinf(eps) else 1.0
        summary_rows.append({
            "dataset": DATASET, "unit": UNIT, "eps": eps, "gamma": GAMMA, "delta": DELTA,
            "n_shadows": n_shadows, "n_pos": n_pos, "n_neg": n_neg, "auc": report["auc"],
            "tpr1": report["tpr_at_1pct_fpr"], "tpr1_ci_hi": report["tpr_at_1pct_fpr_ci_hi"],
            "bound_at_1pct_fpr": bound_1pct,
            "tpr01": report["tpr_at_0.1pct_fpr"], "tpr01_ci_hi": report["tpr_at_0.1pct_fpr_ci_hi"],
            "bound_at_0.1pct_fpr": bound_01pct,
            "audit_passed": int(not (not np.isinf(eps) and exceeds)),
        })
        print(f"eps={eps}: n_shadows={n_shadows}, n_targets(pos+neg)={n_pos}+{n_neg}, "
              f"auc={report['auc']:.4f}, tpr1={report['tpr_at_1pct_fpr']:.4f} "
              f"(95% CI hi={report['tpr_at_1pct_fpr_ci_hi']:.4f}), "
              f"bound_at_1pct_fpr={bound_1pct:.4f}")

    out_csv = REPO_ROOT / "results" / "fig10_m9_audit.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows to {out_csv}")

    summary_csv = REPO_ROOT / "results" / "m9_audit_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"wrote {len(summary_rows)} rows to {summary_csv}")

    config = {
        "seed": CALIB_SEED, "purpose": "M9 audit: empirical ROC vs analytic DP bound (08_FIX_PLAN.md §8)",
        "dataset": DATASET, "unit": UNIT, "eps_list": EPS_LIST, "gamma": GAMMA, "delta": DELTA,
    }
    manifest = provenance.run_manifest(config, seed=CALIB_SEED)
    provenance.finalize(manifest, [out_csv, summary_csv])

    if violations:
        print("\n*** AUDIT FAILED: empirical TPR exceeds the DP bound beyond its 95% CI ***")
        for eps, exceeds in violations:
            for fpr_target, tpr_val, ci_hi, bound_here in exceeds:
                print(f"  eps={eps}, FPR={fpr_target}: empirical TPR={tpr_val:.4f} "
                      f"(95% CI hi={ci_hi:.4f}) > bound={bound_here:.4f}")
        print("Per 08_FIX_PLAN.md §8: stop the M9 line, debug it, and log what was found. Do not proceed.")
        return 2
    print("\nAudit passed: empirical TPR stays within the DP bound (95% CI) at FPR in {1%, 0.1%}, for every finite eps.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadows", action="store_true", help="generate shadows (PBS only, real compute)")
    ap.add_argument("--score", action="store_true", help="score against the DP bound (login-node safe)")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if not args.shadows and not args.score:
        print("usage: run_m9_audit.py --shadows [--workers N] | --score", file=sys.stderr)
        return 2
    if args.shadows:
        generate_shadows(args.workers)
    if args.score:
        return score_audit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
