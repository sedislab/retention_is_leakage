#!/usr/bin/env python3
"""R8 symmetric re-selection (`rebuttal/R8_PROTOCOL.md`). PBS only.

Test-informed design: the grids and the selection federation were motivated by R7/R7b's test-split
sensitivity results (stated in the protocol). Selection uses public ref data only.

`rb_r8_reselect.py cap`: the missing DP-FedAvg caps ŵ(n), via R5's procedure → `rebuttal/r8_wcap.csv`.
`rb_r8_reselect.py cell <idx>` (array, 0..71): select on the size-matched public federation, then run
the finals. Writes `rebuttal/work/r8/{final,sel}_<idx>.csv`.
`rb_r8_reselect.py merge`: writes `results/rb_r8.csv`, `rb_r8_summary.csv` and `rb_r8_selection.csv`.
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
from p3fcl.methods.m0_dpsgd_u1 import per_example_grad_norms  # noqa: E402
from p3fcl.methods.m0_fedavg import _softmax  # noqa: E402
from p3fcl.methods.m9_contractive import ContractiveDPAnalytic  # noqa: E402
from p3fcl.rebuttal.common import manifest  # noqa: E402
from p3fcl.rebuttal.m9_select import select_first_max  # noqa: E402
from p3fcl.rebuttal.r8 import on_edge, selection_federation  # noqa: E402
from p3fcl.rebuttal.sensitivity import ORACLE_LABEL, mean_ci  # noqa: E402
from run_m9_hparam_selection import _fit_pca, _ref_train_val_split  # noqa: E402
from run_m9_sweep import DELTA  # noqa: E402

EPS = [1.0, 4.0, 8.0]
SEEDS = [0, 1, 2]
DATASETS = ["cifar100", "cub200", "imagenet_r"]
U2_N = [("cifar100", 10), ("cifar100", 100), ("cifar100", 500), ("cifar100", 1000), ("cifar100", 2000),
        ("cub200", 10), ("cub200", 100), ("imagenet_r", 10), ("imagenet_r", 100)]
CELLS = ([("m0_dpsgd_u1", "U1", ds, 10, e) for ds in DATASETS for e in EPS]
         + [("m9_contractive", "U1", ds, 10, e) for ds in DATASETS for e in EPS]
         + [("m0_dpfedavg_u2", "U2", ds, n, e) for ds, n in U2_N for e in EPS]
         + [("m9_contractive", "U2", ds, n, e) for ds, n in U2_N for e in EPS])
LR_GRID = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.5]
MULT_GRID = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1.0, 4.0]
P_GRID = [32, 64, 128, 256, 768]
LAM_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
WORK = REPO_ROOT / "rebuttal" / "work" / "r8"
R8_WCAP = REPO_ROOT / "rebuttal" / "r8_wcap.csv"
METRICS = ["final_avg_acc", "bwt", "newest_task_acc"]


def _csv(name):
    with open(REPO_ROOT / name, newline="") as f:
        return list(csv.DictReader(f))


def grid(method):
    if method == "m9_contractive":
        return [{"p": p, "lambda": lam} for p in P_GRID for lam in LAM_GRID], {"p": P_GRID, "lambda": LAM_GRID}
    return [{"lr": lr, "clip_multiplier": m} for lr in LR_GRID for m in MULT_GRID], {"lr": LR_GRID, "clip_multiplier": MULT_GRID}


def wcap(ds, n):
    caps = {**r5.read_wcap(r5.WCAP_CSV), **r5.read_wcap(r5.WCAP_NSCALE_CSV)}
    if R8_WCAP.exists():
        caps.update(r5.read_wcap(R8_WCAP))
    return caps[(ds, n)]


_S: dict = {}


def _run_point(g, seed, final):
    """One run. `final=False`: selection federation → ref-val, seed 0. `final=True`: train → test."""
    s = _S
    Xa, ya, Xb, yb = (s["X"], s["y"], s["Xe"], s["ye"]) if final else (s["Xsel"], s["ysel"], s["Xva"], s["yva"])
    if s["method"] == "m9_contractive":
        P = s["final_bases"][g["p"]] if final else s["sel_basis"][:, :g["p"]]
        C = s["clip_C"][g["p"]] if s["unit"] == "U2" else 1.0  # U1 ignores clip_C
        m = ContractiveDPAnalytic({"n_classes": s["n_classes"], "pca_basis": P, "noise_seed": seed, "gamma": 1.0, "eps": s["eps"],
                                   "delta": DELTA, "ridge_lambda": g["lambda"], "unit": s["unit"], "clip_C": C})
        _, res = r3._run(m, Xa, ya, Xb, yb, s["n"], seed)
        T = res["acc_matrix"].shape[0]
        return {"final_avg_acc": res["final_avg_acc"], "bwt": res["bwt"], "newest_task_acc": float(res["acc_matrix"][T - 1, T - 1]), "clip": C}
    clip = s["med"][g["lr"]] * g["clip_multiplier"]
    m = r5._make(s["method"], s["n_classes"], s["d"], g["lr"], clip, s["eps"], seed, w_cap=s.get("w_cap"))
    _, _, met = r5._run(m, Xa, ya, Xb, yb, s["n"], seed)
    return {**{k: met[k] for k in METRICS}, "clip": clip}


def _sel(j):
    return {"point": j, **_S["grid"][j], **_run_point(_S["grid"][j], 0, final=False)}


def _fin(seed):
    return {"seed": seed, **_run_point(_S["winner"], seed, final=True)}


def _medians():
    """R5's public medians on its ref-train federation (10 clients, seed 0)."""
    s = _S
    if s["method"] == "m0_dpsgd_u1":
        Y0 = np.zeros((len(s["ytr"]), s["n_classes"]))
        Y0[np.arange(len(s["ytr"])), s["ytr"]] = 1.0
        med = float(np.median(per_example_grad_norms(s["Xtr"], _softmax(np.zeros((len(s["ytr"]), s["n_classes"]))) - Y0)))
        return {lr: med for lr in LR_GRID}
    out = {}
    for lr in LR_GRID:
        _, res, _ = r5._run(r5._make("m0", s["n_classes"], s["d"], lr, np.inf, np.inf, 0), s["Xtr"], s["ytr"], s["Xva"], s["yva"],
                            r5.N_CLIENTS_DEFAULT, 0)
        out[lr] = float(np.median([np.linalg.norm(rec.payload) for rec in res["ledger"]]))
    return out


def run_cell(idx: int) -> int:
    method, unit, ds, n, eps = CELLS[idx]
    man = manifest({"seed": 0, "purpose": "rebuttal R8 symmetric re-selection (test-informed design, public selection)",
                    "cell": idx, "method": method, "unit": unit, "dataset": ds, "n_clients": n, "eps": eps,
                    "lr_grid": LR_GRID, "mult_grid": MULT_GRID, "p_grid": P_GRID, "lambda_grid": LAM_GRID,
                    "protocol": "rebuttal/R8_PROTOCOL.md"})
    X, y = r5._load(ds, "train")
    Xe, ye = r5._load(ds, "test")
    Xr, yr = r5._load(ds, "ref")
    tr, va = _ref_train_val_split(yr, seed=0, frac=0.5)
    Xsel, ysel, _ = selection_federation(Xr[tr], yr[tr], y, ds)
    pts, axes = grid(method)
    _S.update(method=method, unit=unit, n=n, eps=eps, X=X, y=y, Xe=Xe, ye=ye, Xtr=Xr[tr], ytr=yr[tr], Xva=Xr[va], yva=yr[va],
              Xsel=Xsel, ysel=ysel, n_classes=int(y.max()) + 1, d=X.shape[1], grid=pts)
    if method == "m9_contractive":
        _S["sel_basis"] = _fit_pca(Xr[tr], max(P_GRID), seed=0)
        if unit == "U2":
            _S["clip_C"] = {int(r["p"]): float(r["clip_C"]) for r in _csv("results/rb_client_scaling_selection.csv")
                            if r["dataset"] == ds and int(r["n_clients"]) == n}
            assert set(_S["clip_C"]) == set(P_GRID), f"C(n,p) missing for {ds} n={n}"
    else:
        _S["med"] = _medians()
        stored = {float(r["lr"]): float(r["median_norm"]) for r in _csv("results/rb_dp_fedavg_selection.csv")
                  if r["dataset"] == ds and r["method"] == method}
        for lr in (0.1, 0.5):
            if _S["med"][lr] != stored[lr]:
                raise AssertionError(f"median lr={lr}: {_S['med'][lr]!r} != R5 stored {stored[lr]!r}")
        if method == "m0_dpfedavg_u2":
            _S["w_cap"] = wcap(ds, n)
    nproc = int(os.environ.get("NCPUS", "8"))
    with mp.get_context("fork").Pool(nproc) as pool:
        sel = sorted(pool.map(_sel, range(len(pts)), chunksize=1), key=lambda r: r["point"])
    win = select_first_max([{**r, "acc": r["final_avg_acc"]} for r in sel])
    _S["winner"] = {k: win[k] for k in axes}
    if method == "m9_contractive":
        _S["final_bases"] = {win["p"]: _fit_pca(Xr, win["p"], seed=0)}
    edges = on_edge(_S["winner"], axes)
    with mp.get_context("fork").Pool(len(SEEDS)) as pool:
        fin = pool.map(_fin, SEEDS)
    base = {"cell": idx, "method": method, "unit": unit, "dataset": ds, "n_clients": n, "eps": eps}
    sel_rows = [{**base, **r, "selected": int(r["point"] == win["point"])} for r in sel]
    fin_rows = [{**base, **{k: _S["winner"].get(k, "n/a") for k in ("lr", "clip_multiplier", "p", "lambda")}, "clip": r["clip"],
                 "w_cap": _S.get("w_cap", "n/a"), "ref_val_score": win["final_avg_acc"], "on_edge": int(bool(edges)),
                 "edge_axes": ";".join(edges) if edges else "none", **{k: r[k] for k in ("seed", *METRICS)}} for r in fin]
    print(f"R8 cell {idx} {method}/{unit}/{ds}/n={n}/eps={eps}: selected {_S['winner']} ref_val={win['final_avg_acc']:.4f} "
          f"edges={edges or 'none'} test acc={np.mean([r['final_avg_acc'] for r in fin]):.4f}", flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    outs = []
    for name, data in (("sel", sel_rows), ("final", fin_rows)):
        p = WORK / f"{name}_{idx:02d}.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]), restval="n/a")
            w.writeheader()
            w.writerows(data)
        outs.append(p)
    provenance.finalize(man, outs)
    return 0


def cap() -> int:
    man = manifest({"seed": 0, "purpose": "rebuttal R8 missing DP-FedAvg caps (R5 amendment-1 procedure)", "protocol": "rebuttal/R8_PROTOCOL.md"})
    have = {**r5.read_wcap(r5.WCAP_CSV), **r5.read_wcap(r5.WCAP_NSCALE_CSV)}
    rows = [r5.compute_wcap(ds, n) for ds, n in U2_N if (ds, n) not in have]
    with open(R8_WCAP, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"R8 W_CAP {r['dataset']} n={r['n_clients']}: {r['w_cap']} (median {r['median_size']}, max {r['max_size']})")
    provenance.finalize(man, [R8_WCAP])
    return 0


def _prev_selected(method, unit, ds, n, eps):
    if method == "m9_contractive" and unit == "U1":
        v = [float(r["final_avg_acc"]) for r in _csv(f"results/m9_sweep_{ds}.csv") if r["method"] == "m9_contractive" and r["unit"] == "U1"
             and float(r["gamma"]) == 1.0 and int(r["n_clients"]) == 10 and float(r["eps"]) == eps]
        return (float(np.mean(v)), f"results/m9_sweep_{ds}.csv (paper)") if v else ("n/a", "n/a")
    if method == "m9_contractive":
        v = [float(r["final_avg_acc"]) for r in _csv("results/rb_client_scaling.csv") if r["calibration"] == "size_matched"
             and r["dataset"] == ds and int(r["n_clients"]) == n and float(r["eps"]) == eps]
        return (float(np.mean(v)), "results/rb_client_scaling.csv (R3 size_matched)") if v else ("n/a", "n/a")
    for name in ("results/rb_dp_fedavg.csv", "results/rb_dp_fedavg_nscale.csv"):
        v = [float(r["final_avg_acc"]) for r in _csv(name) if r["method"] == method and r["dataset"] == ds
             and int(r.get("n_clients") or 10) == n and float(r["eps"]) == eps]
        if v:
            return float(np.mean(v)), f"{name} (R5)"
    return "n/a", "n/a"


def _oracle(method, unit, ds, n, eps):
    for r in _csv("results/rb_sensitivity_ext.csv"):
        if r["method"] == method and r["dataset"] == ds and int(r["n_clients"]) == n and float(r["eps"]) == eps:
            return float(r["best_final_avg_acc_mean"]), f"R7b (interior={r['interior']})"
    for r in _csv("results/rb_sensitivity.csv"):
        if r["role"] == "best" and r["method"] == method and r["dataset"] == ds and int(r["n_clients"]) == n and float(r["eps"]) == eps \
                and (method != "m9_contractive" or unit == "U2"):
            return float(r["final_avg_acc_mean"]), "R7"
    return "n/a", "n/a"


def merge() -> int:
    man = manifest({"seed": 0, "purpose": "rebuttal R8 merge", "protocol": "rebuttal/R8_PROTOCOL.md"})
    finals, sels, summ = [], [], []
    for idx, (method, unit, ds, n, eps) in enumerate(CELLS):
        f = _csv(f"rebuttal/work/r8/final_{idx:02d}.csv")
        finals += f
        sels += _csv(f"rebuttal/work/r8/sel_{idx:02d}.csv")
        m, lo, hi = mean_ci([float(r["final_avg_acc"]) for r in f])
        prev, prev_src = _prev_selected(method, unit, ds, n, eps)
        orc, orc_src = _oracle(method, unit, ds, n, eps)
        summ.append({**{k: f[0][k] for k in ("method", "unit", "dataset", "n_clients", "eps", "lr", "clip_multiplier", "p", "lambda",
                                              "clip", "w_cap", "ref_val_score", "on_edge", "edge_axes")},
                     "final_avg_acc_mean": m, "ci_lo": lo, "ci_hi": hi,
                     "bwt_mean": float(np.mean([float(r["bwt"]) for r in f])),
                     "newest_task_acc_mean": float(np.mean([float(r["newest_task_acc"]) for r in f])),
                     "prev_selected_mean": prev, "prev_selected_source": prev_src,
                     "delta_vs_prev_selected": m - prev if prev != "n/a" else "n/a",
                     "oracle_best_mean": orc, "oracle_source": orc_src, "oracle_label": ORACLE_LABEL if orc != "n/a" else "n/a"})
    outs = []
    for name, data in (("rb_r8.csv", finals), ("rb_r8_summary.csv", summ), ("rb_r8_selection.csv", sels)):
        p = REPO_ROOT / "results" / name
        cols = list(dict.fromkeys(k for row in data for k in row))  # union: baseline and M9 rows differ
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, restval="n/a")
            w.writeheader()
            w.writerows(data)
        outs.append(p)
    for s in summ:
        print(f"R8 {s['method']:15s} {s['unit']} {s['dataset']:10s} n={s['n_clients']:<5} eps={s['eps']:<4} acc={s['final_avg_acc_mean']:.3f} "
              f"edge={s['edge_axes']} prev={s['prev_selected_mean']} oracle={s['oracle_best_mean']}")
    print(f"R8 MERGE cells={len(summ)} final_rows={len(finals)} selection_rows={len(sels)}")
    provenance.finalize(man, outs)
    return 0


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "cap":
        sys.exit(cap())
    if mode == "merge":
        sys.exit(merge())
    sys.exit(run_cell(int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ["PBS_ARRAY_INDEX"])))
