#!/usr/bin/env python3
"""FX2/FX6: `results/fig01_decoupling.csv`, combining the leak side (already computed with a real
hierarchical bootstrap CI, `results/a1_lira_fixedk_summary.csv`) and the accuracy side (computed here
directly from `results/accuracy_matrix_<dataset>.csv` with a simple across-seed t-interval CI, the
same convention `build_fig08_pareto.py`/`build_tab08.py` use -- NOT the half-life hierarchical
bootstrap `build_fx2_accuracy_summary.py` uses, since this is a per-elapsed CURVE, not a half-life
point estimate, and doesn't need the K-resampling machinery). No computation on the leak side
(CLAUDE.md non-negotiable #3 -- reads `a1_lira_fixedk_summary.csv`'s already-bootstrapped numbers
as-is); the accuracy side is a real, simple aggregation, not a re-derivation of anything already
computed elsewhere.

Schema (03_RESULTS_SPEC.md's FIG01 + FX2's `view`/`acc_norm`/`leak_norm`/`acc_train` additions):
`dataset, method, family, view, elapsed, acc_mean, acc_ci_lo, acc_ci_hi, acc_train_mean,
tpr1_mean, tpr1_ci_lo, tpr1_ci_hi, tpr01_mean, tpr01_ci_lo, tpr01_ci_hi, auc_mean, auc_ci_lo,
auc_ci_hi, acc_norm, leak_norm`.

`acc_norm`/`leak_norm`: each series normalised to its own elapsed=0 value (1.0 at e=0), matching
`08_FIX_PLAN.md`'s normalization convention (07b) applied per-curve rather than only at the
half-life step -- lets FIG01 show both accuracy and leakage on a shared relative-decay scale.
Restricted to `ablation=trajectory` (H3's transcript arm, the paper's headline) and the fixed-k set
K={0,1,2,3} (same population as the leak side, e in 0..6) for the accuracy side too, so both curves
describe the exact same target population at every elapsed time.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import provenance  # noqa: E402

K_SET = [0, 1, 2, 3]
E_MAX = 6
N_TASKS = 10


def _ci(values: list) -> tuple:
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, float("nan"), float("nan")
    sem = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    if sem == 0:
        return mean, mean, mean
    lo, hi = stats.t.interval(0.95, df=len(values) - 1, loc=mean, scale=sem)
    return mean, float(lo), float(hi)


def _load_acc_matrix(dataset: str, method: str) -> dict:
    path = REPO_ROOT / "results" / f"accuracy_matrix_{dataset}.csv"
    out: dict = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] != method:
                continue
            key = (int(row["seed"]), int(row["task_k"]), int(row["elapsed"]))
            out[key] = {
                "acc": float(row["acc"]) if row["acc"] != "" else float("nan"),
                "acc_train": float(row["acc_train"]) if row["acc_train"] != "" else float("nan"),
            }
    return out


def _accuracy_curve(dataset: str, method: str, seeds: list) -> dict:
    """`{e: (acc_mean, acc_lo, acc_hi, acc_train_mean)}`, pooled over K_SET and seeds at each e."""
    acc_by_key = _load_acc_matrix(dataset, method)
    curve = {}
    for e in range(E_MAX + 1):
        accs, accs_train = [], []
        for k in K_SET:
            T = k + e
            if T >= N_TASKS:
                continue
            for s in seeds:
                rec = acc_by_key.get((s, k, T))
                if rec is None or np.isnan(rec["acc"]):
                    continue
                accs.append(rec["acc"])
                accs_train.append(rec["acc_train"])
        if not accs:
            continue
        mean, lo, hi = _ci(accs)
        curve[e] = (mean, lo, hi, float(np.mean(accs_train)) if accs_train else float("nan"))
    return curve


def _load_leak_curve(dataset: str, method: str, family: str, view: str) -> dict:
    path = REPO_ROOT / "results" / "a1_lira_fixedk_summary.csv"
    with open(path, newline="") as f:
        rows = [
            r for r in csv.DictReader(f)
            if r["dataset"] == dataset and r["method"] == method and r["family"] == family
            and r["view"] == view and r["ablation"] == "trajectory"
        ]
    return {int(r["elapsed"]): r for r in rows}


def build_one(dataset: str, method: str, family: str, view: str, seeds: list) -> list:
    acc_curve = _accuracy_curve(dataset, method, seeds)
    leak_curve = _load_leak_curve(dataset, method, family, view)
    if not acc_curve or not leak_curve:
        return []

    acc_base = acc_curve.get(0, (None,))[0]
    leak_base = float(leak_curve[0]["tpr1"]) if 0 in leak_curve else None

    rows = []
    for e in sorted(set(acc_curve) & set(leak_curve)):
        acc_mean, acc_lo, acc_hi, acc_train_mean = acc_curve[e]
        leak_row = leak_curve[e]
        rows.append({
            "dataset": dataset, "method": method, "family": family, "view": view, "elapsed": e,
            "acc_mean": acc_mean, "acc_ci_lo": acc_lo, "acc_ci_hi": acc_hi,
            "acc_train_mean": acc_train_mean,
            "tpr1_mean": leak_row["tpr1"], "tpr1_ci_lo": leak_row["tpr1_ci_lo"], "tpr1_ci_hi": leak_row["tpr1_ci_hi"],
            "tpr01_mean": leak_row["tpr01"], "tpr01_ci_lo": leak_row["tpr01_ci_lo"], "tpr01_ci_hi": leak_row["tpr01_ci_hi"],
            "auc_mean": leak_row["auc"], "auc_ci_lo": leak_row["auc_ci_lo"], "auc_ci_hi": leak_row["auc_ci_hi"],
            "acc_norm": (acc_mean / acc_base) if acc_base else "",
            "leak_norm": (float(leak_row["tpr1"]) / leak_base) if leak_base else "",
        })
    return rows


def main() -> int:
    combos = [
        ("cifar100", "m0_fedavg", "F1", "full"), ("cifar100", "m1_glfc", "F1", "full"),
        ("cifar100", "m2_target", "F1", "full"), ("cifar100", "m3_fot", "F1", "full"),
        ("cifar100", "m5_hybrid_replay", "F1", "full"),
        ("cifar100", "m4_proto", "F2", "full"), ("cifar100", "m4_proto", "F2", "aggregate"), ("cifar100", "m4_proto", "F2", "global"),
        ("cifar100", "m8_analytic", "F5", "full"), ("cifar100", "m8_analytic", "F5", "aggregate"), ("cifar100", "m8_analytic", "F5", "global"),
    ]
    for dataset in ("cub200", "imagenet_r"):
        combos.append((dataset, "m1_glfc", "F1", "full"))
        combos.append((dataset, "m2_target", "F1", "full"))
        combos.append((dataset, "m3_fot", "F1", "full"))
        combos.append((dataset, "m5_hybrid_replay", "F1", "full"))
        for view in ("full", "aggregate", "global"):
            combos.append((dataset, "m4_proto", "F2", view))
            combos.append((dataset, "m8_analytic", "F5", view))
        m0_summary = REPO_ROOT / "results" / "a1_lira_fixedk_summary.csv"
        with open(m0_summary, newline="") as f:
            if any(r["dataset"] == dataset and r["method"] == "m0_fedavg" for r in csv.DictReader(f)):
                combos.append((dataset, "m0_fedavg", "F1", "full"))

    all_rows = []
    for dataset, method, family, view in combos:
        seeds = [0, 1, 2, 3, 4] if method == "m0_fedavg" else [0, 1, 2]
        all_rows.extend(build_one(dataset, method, family, view, seeds))

    if not all_rows:
        print("no rows produced -- check that a1_lira_fixedk_summary.csv and accuracy_matrix_*.csv exist", file=sys.stderr)
        return 1

    out_csv = REPO_ROOT / "results" / "fig01_decoupling.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows to {out_csv} ({len(combos)} combos attempted)")

    config = {"seed": 0, "purpose": "FX6 FIG01 v2 decoupling curve (leak + accuracy)"}
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
