#!/usr/bin/env python3
"""R7 test-split sensitivity of the existing selection grids (`rebuttal/R7_PROTOCOL.md`). PBS only.

`rb_r7_sensitivity.py cell <idx>` (array, idx 0..43): run every grid point of one cell on train/test
for seeds 0-2. Writes `rebuttal/work/r7/cell_<idx>.csv`.
`rb_r7_sensitivity.py merge`: check that each selected point's rerun equals the reported values bit for
bit, then write `results/rb_sensitivity.csv` and `results/rb_sensitivity_grid.csv`. On any mismatch
it writes `rebuttal/r7_merge_FAILED.csv` and nothing to results/.
This is a sensitivity analysis: nothing here changes a reported result or a selection.
"""
from __future__ import annotations

import csv
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))
sys.path.insert(0, str(REPO_ROOT / "code" / "scripts"))

import rb_r3_client_scaling as r3  # noqa: E402
import rb_r5_dp_fedavg as r5  # noqa: E402
from p3fcl import provenance  # noqa: E402
from p3fcl.methods.m9_contractive import ContractiveDPAnalytic  # noqa: E402
from p3fcl.rebuttal.common import manifest  # noqa: E402
from p3fcl.rebuttal.sensitivity import ORACLE_LABEL, mean_ci, summarize_cell  # noqa: E402
from run_m9_hparam_selection import _fit_pca  # noqa: E402
from run_m9_sweep import DELTA  # noqa: E402

SEEDS = [0, 1, 2]
EPS_DP = [1.0, 4.0, 8.0, float("inf")]
CELLS = ([("m0_dpsgd_u1", ds, 10, e) for ds in ("cifar100", "cub200", "imagenet_r") for e in EPS_DP]
         + [("m0_dpfedavg_u2", ds, n, e) for ds, n in (("cifar100", 10), ("cifar100", 100), ("cifar100", 1000),
                                                         ("cub200", 10), ("imagenet_r", 10)) for e in EPS_DP]
         + [("m9_contractive", "cifar100", n, e) for n in (100, 500, 1000, 2000) for e in (1.0, 4.0, 8.0)])
WORK = REPO_ROOT / "rebuttal" / "work" / "r7"
METRICS = ["final_avg_acc", "bwt", "newest_task_acc"]


def _csv(name):
    with open(REPO_ROOT / name, newline="") as f:
        return list(csv.DictReader(f))


def grid_points(method, ds, n, eps) -> list:
    """The cell's existing selection grid in grid order, copied from the selection CSVs."""
    if method == "m9_contractive":
        rows = [r for r in _csv("results/rb_client_scaling_selection.csv")
                if r["dataset"] == ds and int(r["n_clients"]) == n and float(r["eps"]) == eps]
        pts = sorted(({"p": int(r["p"]), "lambda": float(r["lambda"]), "clip": float(r["clip_C"])} for r in rows),
                     key=lambda g: (g["p"], g["lambda"]))
        expected = 25
    else:
        src = "results/rb_dp_fedavg_selection.csv" if n == 10 else "results/rb_dp_fedavg_nscale_selection.csv"
        rows = [r for r in _csv(src) if r["dataset"] == ds and r["method"] == method and float(r["eps"]) == eps
                and int(r.get("n_clients") or 10) == n]
        pts = sorted(({"lr": float(r["lr"]), "clip_multiplier": float(r["clip_multiplier"]), "clip": float(r["clip"])} for r in rows),
                     key=lambda g: (g["lr"], g["clip_multiplier"]))
        expected = 15
    if len(pts) != expected:
        raise RuntimeError(f"{method}/{ds}/n={n}/eps={eps}: {len(pts)} grid points, expected {expected}")
    return pts


def point_key(method, g):
    return (g["p"], g["lambda"]) if method == "m9_contractive" else (g["lr"], g["clip_multiplier"])


_S: dict = {}


def _one(args):
    j, seed = args
    s = _S
    g = s["pts"][j]
    if s["method"] == "m9_contractive":
        m = ContractiveDPAnalytic({"n_classes": s["n_classes"], "pca_basis": s["bases"][g["p"]], "noise_seed": seed, "gamma": 1.0,
                                   "eps": s["eps"], "delta": DELTA, "ridge_lambda": g["lambda"], "unit": "U2", "clip_C": g["clip"]})
        _, res = r3._run(m, s["X"], s["y"], s["Xe"], s["ye"], s["n"], seed)
        T = res["acc_matrix"].shape[0]
        met = {"final_avg_acc": res["final_avg_acc"], "bwt": res["bwt"], "newest_task_acc": float(res["acc_matrix"][T - 1, T - 1])}
    else:
        m = r5._make(s["method"], s["n_classes"], s["d"], g["lr"], g["clip"], s["eps"], seed, w_cap=s["w_cap"])
        _, _, met = r5._run(m, s["X"], s["y"], s["Xe"], s["ye"], s["n"], seed)
    return {"point": j, **g, "seed": seed, **{k: met[k] for k in METRICS}}


def run_cell(idx: int) -> int:
    method, ds, n, eps = CELLS[idx]
    man = manifest({"seed": 0, "purpose": "rebuttal R7 test-split sensitivity (not selection)", "cell": idx,
                    "method": method, "dataset": ds, "n_clients": n, "eps": eps, "seeds": SEEDS, "protocol": "rebuttal/R7_PROTOCOL.md"})
    X, y = r5._load(ds, "train")
    Xe, ye = r5._load(ds, "test")
    pts = grid_points(method, ds, n, eps)
    _S.update(method=method, n=n, eps=eps, X=X, y=y, Xe=Xe, ye=ye, pts=pts, n_classes=int(y.max()) + 1, d=X.shape[1])
    if method == "m9_contractive":
        Xref = r5._load(ds, "ref")[0]
        _S["bases"] = {p: _fit_pca(Xref, p, seed=0) for p in sorted({g["p"] for g in pts})}
    else:
        _S["w_cap"] = r5.read_wcap(r5.WCAP_CSV if n == 10 else r5.WCAP_NSCALE_CSV)[(ds, n)]
    work = [(j, s) for j in range(len(pts)) for s in SEEDS]
    with mp.get_context("fork").Pool(int(os.environ.get("NCPUS", "8"))) as pool:
        rows = sorted(pool.map(_one, work, chunksize=1), key=lambda r: (r["point"], r["seed"]))
    for r in rows:
        r.update({"cell": idx, "method": method, "dataset": ds, "n_clients": n, "eps": eps})
    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / f"cell_{idx:02d}.csv"
    cols = ["cell", "method", "dataset", "n_clients", "eps", "point", "lr", "clip_multiplier", "p", "lambda", "clip", "seed", *METRICS]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval="n/a")
        w.writeheader()
        w.writerows(rows)
    print(f"R7 cell {idx} {method}/{ds}/n={n}/eps={eps}: {len(rows)} runs -> {out}")
    provenance.finalize(man, [out])
    return 0


def reported_selected(method, ds, n, eps) -> tuple:
    """`(key, {seed: final_avg_acc})` of the reported, selected configuration."""
    if method == "m9_contractive":
        rows = [r for r in _csv("results/rb_client_scaling.csv") if r["calibration"] == "size_matched" and r["dataset"] == ds
                and int(r["n_clients"]) == n and float(r["eps"]) == eps]
        keys = {(int(r["p"]), float(r["lambda"])) for r in rows}
        src = "results/rb_client_scaling.csv (calibration=size_matched)"
    else:
        name = "results/rb_dp_fedavg.csv" if n == 10 else "results/rb_dp_fedavg_nscale.csv"
        rows = [r for r in _csv(name) if r["dataset"] == ds and r["method"] == method and float(r["eps"]) == eps
                and int(r.get("n_clients") or 10) == n]
        keys = {(float(r["lr"]), float(r["clip_multiplier"])) for r in rows}
        src = name
    if len(keys) != 1 or len(rows) != len(SEEDS):
        raise RuntimeError(f"reported rows for {method}/{ds}/n={n}/eps={eps}: keys={keys} rows={len(rows)}")
    return keys.pop(), {int(r["seed"]): float(r["final_avg_acc"]) for r in rows}, src


def merge() -> int:
    man = manifest({"seed": 0, "purpose": "rebuttal R7 merge (sensitivity, not selection)", "protocol": "rebuttal/R7_PROTOCOL.md"})
    grid_rows, out_rows, failures = [], [], []
    for idx, (method, ds, n, eps) in enumerate(CELLS):
        rows = _csv(f"rebuttal/work/r7/cell_{idx:02d}.csv")
        grid_rows += rows
        pts = grid_points(method, ds, n, eps)
        acc = {}
        for r in rows:
            acc.setdefault(int(r["point"]), {})[int(r["seed"])] = float(r["final_avg_acc"])
        points = [{"key": point_key(method, g), "g": g, "accs": [acc[j][s] for s in SEEDS]} for j, g in enumerate(pts)]
        sel_key, sel_rep, src = reported_selected(method, ds, n, eps)
        rerun = dict(zip(SEEDS, next(p["accs"] for p in points if p["key"] == sel_key)))
        if any(rerun[s] != sel_rep[s] for s in SEEDS):
            failures.append(f"{method}/{ds}/n={n}/eps={eps}: rerun {rerun} != reported {sel_rep}")
        summ = summarize_cell(points, sel_key)
        for role in ("selected", "best", "worst"):
            p = summ[role]
            vals = [sel_rep[s] for s in SEEDS] if role == "selected" else p["accs"]
            m, lo, hi = mean_ci(vals)
            g = p["g"]
            out_rows.append({
                "method": method, "dataset": ds, "n_clients": n, "eps": eps, "role": role,
                "label": {"selected": "selected on public ref-val (reported, unchanged)", "best": ORACLE_LABEL,
                          "worst": "worst grid point on test (sensitivity)"}[role],
                "lr": g.get("lr", "n/a"), "clip_multiplier": g.get("clip_multiplier", "n/a"), "p": g.get("p", "n/a"),
                "lambda": g.get("lambda", "n/a"), "clip": g["clip"],
                "final_avg_acc_mean": m, "ci_lo": lo, "ci_hi": hi, **{f"acc_seed{s}": v for s, v in zip(SEEDS, vals)},
                "n_grid_points": summ["n_points"], "n_tied": summ["n_tied_best"] if role == "best" else summ["n_tied_worst"] if role == "worst" else "",
                "selected_rank_oracle": summ["selected_rank"] if role == "selected" else "",
                "oracle_gap_best_minus_selected": summ["oracle_gap"] if role == "best" else "",
                "selected_rerun_matches_reported": int(all(rerun[s] == sel_rep[s] for s in SEEDS)) if role == "selected" else "",
                "source_selected": src if role == "selected" else "results/rb_sensitivity_grid.csv",
            })
    dest = REPO_ROOT / "results" / "rb_sensitivity.csv" if not failures else REPO_ROOT / "rebuttal" / "r7_merge_FAILED.csv"
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    for msg in failures:
        print("R7 MERGE FAILURE:", msg)
    for r in out_rows:
        print(f"R7 {r['method']:15s} {r['dataset']:10s} n={r['n_clients']:<5} eps={r['eps']:<4} {r['role']:8s} "
              f"acc={r['final_avg_acc_mean']:.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}] rank={r['selected_rank_oracle']} gap={r['oracle_gap_best_minus_selected']}")
    print(f"R7 MERGE cells={len(CELLS)} rows={len(out_rows)} grid_rows={len(grid_rows)} failures={len(failures)} -> {dest}")
    if failures:
        return 1
    grid_out = REPO_ROOT / "results" / "rb_sensitivity_grid.csv"
    with open(grid_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grid_rows[0]))
        w.writeheader()
        w.writerows(grid_rows)
    provenance.finalize(man, [dest, grid_out])
    return 0


if __name__ == "__main__":
    mode = sys.argv[1]
    sys.exit(run_cell(int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ["PBS_ARRAY_INDEX"])) if mode == "cell" else merge())
