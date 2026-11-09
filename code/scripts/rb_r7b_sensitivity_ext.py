#!/usr/bin/env python3
"""R7b extended test-split sensitivity grid for the DP baselines (`rebuttal/R7b_PROTOCOL.md`). PBS only.

`rb_r7b_sensitivity_ext.py cell <idx>` (array, idx 0..17): base 6×7 grid, then up to 2 edge
extension rounds, for one cell. Writes `rebuttal/work/r7b/cell_<idx>.csv` (grid) and `summary_<idx>.csv`.
`rb_r7b_sensitivity_ext.py merge`: writes `results/rb_sensitivity_ext.csv` and `_grid.csv`.
This is a sensitivity analysis: the best point is a test-oracle upper bound, not used for selection.
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

import rb_r5_dp_fedavg as r5  # noqa: E402
from p3fcl import provenance  # noqa: E402
from p3fcl.methods.m0_dpsgd_u1 import per_example_grad_norms  # noqa: E402
from p3fcl.methods.m0_fedavg import _softmax  # noqa: E402
from p3fcl.rebuttal.common import manifest  # noqa: E402
from p3fcl.rebuttal.sensitivity import ORACLE_LABEL, best_point, edges_hit, extend_grid, mean_ci  # noqa: E402
from run_m9_hparam_selection import _ref_train_val_split  # noqa: E402

SEEDS = [0, 1, 2]
EPS = [1.0, 4.0, 8.0]
CELLS = ([("m0_dpsgd_u1", ds, 10, e) for ds in ("cifar100", "cub200", "imagenet_r") for e in EPS]
         + [("m0_dpfedavg_u2", "cifar100", n, e) for n in (10, 100, 1000) for e in EPS])
LR_BASE = [0.01, 0.02, 0.05, 0.1, 0.5, 1.0]
MULT_BASE = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
MAX_ROUNDS = 2
WORK = REPO_ROOT / "rebuttal" / "work" / "r7b"
METRICS = ["final_avg_acc", "bwt", "newest_task_acc"]


def _csv(name):
    with open(REPO_ROOT / name, newline="") as f:
        return list(csv.DictReader(f))


_S: dict = {}


def _median(lr: float) -> float:
    """R5's public medians (main_dataset): DP-SGD per-example gradient norm at W = 0 (lr-free);
    DP-FedAvg non-private M0 client update norm at `lr` on the ref-train federation (10 clients, seed 0)."""
    s = _S
    if s["method"] == "m0_dpsgd_u1":
        if "med_g" not in s:
            Y0 = np.zeros((len(s["ytr"]), s["n_classes"]))
            Y0[np.arange(len(s["ytr"])), s["ytr"]] = 1.0
            s["med_g"] = float(np.median(per_example_grad_norms(s["Xtr"], _softmax(np.zeros((len(s["ytr"]), s["n_classes"]))) - Y0)))
        return s["med_g"]
    if lr not in s["med_u"]:
        _, res, _ = r5._run(r5._make("m0", s["n_classes"], s["d"], lr, np.inf, np.inf, 0), s["Xtr"], s["ytr"], s["Xva"], s["yva"],
                            r5.N_CLIENTS_DEFAULT, 0)
        s["med_u"][lr] = float(np.median([np.linalg.norm(rec.payload) for rec in res["ledger"]]))
    return s["med_u"][lr]


def _one(args):
    lr, mult, seed = args
    s = _S
    clip = s["med"][lr] * mult
    m = r5._make(s["method"], s["n_classes"], s["d"], lr, clip, s["eps"], seed, w_cap=s["w_cap"])
    _, _, met = r5._run(m, s["X"], s["y"], s["Xe"], s["ye"], s["n"], seed)
    return {"lr": lr, "clip_multiplier": mult, "clip": clip, "seed": seed, **{k: met[k] for k in METRICS}, "source": "R7b run"}


def _pool_run(work):
    if not work:
        return []
    with mp.get_context("fork").Pool(int(os.environ.get("NCPUS", "8"))) as pool:
        return pool.map(_one, work, chunksize=1)


def run_cell(idx: int) -> int:
    method, ds, n, eps = CELLS[idx]
    man = manifest({"seed": 0, "purpose": "rebuttal R7b extended sensitivity (not selection)", "cell": idx, "method": method,
                    "dataset": ds, "n_clients": n, "eps": eps, "lr_base": LR_BASE, "mult_base": MULT_BASE,
                    "max_rounds": MAX_ROUNDS, "protocol": "rebuttal/R7b_PROTOCOL.md"})
    X, y = r5._load(ds, "train")
    Xe, ye = r5._load(ds, "test")
    Xr, yr = r5._load(ds, "ref")
    tr, va = _ref_train_val_split(yr, seed=0, frac=0.5)
    _S.update(method=method, n=n, eps=eps, X=X, y=y, Xe=Xe, ye=ye, Xtr=Xr[tr], ytr=yr[tr], Xva=Xr[va], yva=yr[va],
              n_classes=int(y.max()) + 1, d=X.shape[1], med_u={},
              w_cap=r5.read_wcap(r5.WCAP_CSV if n == 10 else r5.WCAP_NSCALE_CSV)[(ds, n)])

    # Check 1: recomputed medians for the existing lrs equal the stored ones.
    sel_src = "results/rb_dp_fedavg_selection.csv" if n == 10 else "results/rb_dp_fedavg_nscale_selection.csv"
    stored = {float(r["lr"]): float(r["median_norm"]) for r in _csv(sel_src)
              if r["dataset"] == ds and r["method"] == method and int(r.get("n_clients") or 10) == n}
    for lr, v in stored.items():
        if _median(lr) != v:
            raise AssertionError(f"median at lr={lr}: recomputed {_median(lr)!r} != stored {v!r}")

    # Reuse R7's 15 points, after checking that rerunning R7's best point is bit-identical.
    res: dict = {}
    for r in _csv("results/rb_sensitivity_grid.csv"):
        if r["method"] == method and r["dataset"] == ds and int(r["n_clients"]) == n and float(r["eps"]) == eps:
            res.setdefault((float(r["lr"]), float(r["clip_multiplier"])), {})[int(r["seed"])] = {
                "lr": float(r["lr"]), "clip_multiplier": float(r["clip_multiplier"]), "clip": float(r["clip"]), "seed": int(r["seed"]),
                **{k: float(r[k]) for k in METRICS}, "source": "R7 reuse (results/rb_sensitivity_grid.csv)"}
    if len(res) != 15:
        raise AssertionError(f"expected 15 R7 points, found {len(res)}")
    r7best = next(r for r in _csv("results/rb_sensitivity.csv") if r["role"] == "best" and r["method"] == method
                  and r["dataset"] == ds and int(r["n_clients"]) == n and float(r["eps"]) == eps)
    bkey = (float(r7best["lr"]), float(r7best["clip_multiplier"]))
    _S["med"] = {lr: _median(lr) for lr in {bkey[0]}}
    for row in _pool_run([(bkey[0], bkey[1], s) for s in SEEDS]):
        if row["final_avg_acc"] != res[bkey][row["seed"]]["final_avg_acc"]:
            raise AssertionError(f"R7 best rerun {row['final_avg_acc']!r} != stored {res[bkey][row['seed']]['final_avg_acc']!r}")
    print(f"R7b cell {idx}: medians match stored; R7 best {bkey} reruns bit-identically", flush=True)

    lr_grid, mult_grid, used, rounds, history = list(LR_BASE), list(MULT_BASE), {}, 0, []
    while True:
        _S["med"] = {lr: _median(lr) for lr in lr_grid}
        todo = [(lr, m, s) for lr in lr_grid for m in mult_grid for s in SEEDS if s not in res.get((lr, m), {})]
        for row in _pool_run(todo):
            res.setdefault((row["lr"], row["clip_multiplier"]), {})[row["seed"]] = row
        keys = [(lr, m) for lr in lr_grid for m in mult_grid]
        pts = [{"key": k, "accs": [res[k][s]["final_avg_acc"] for s in SEEDS]} for k in keys]
        b = pts[best_point(pts)]["key"]
        e = edges_hit(b[0], b[1], lr_grid, mult_grid)
        history.append(f"round{rounds}: best={b} edges={e or 'none'} ({len(keys)} pts)")
        print(f"R7b cell {idx} {method}/{ds}/n={n}/eps={eps} {history[-1]}", flush=True)
        if not e or rounds >= MAX_ROUNDS:
            break
        lr_grid, mult_grid, done = extend_grid(lr_grid, mult_grid, e, used)
        if not done:
            break
        rounds += 1

    accs = [res[b][s]["final_avg_acc"] for s in SEEDS]
    mm, lo, hi = mean_ci(accs)
    summary = {"cell": idx, "method": method, "dataset": ds, "n_clients": n, "eps": eps, "label": ORACLE_LABEL,
               "best_lr": b[0], "best_clip_multiplier": b[1], "best_clip": res[b][0]["clip"],
               "best_final_avg_acc_mean": mm, "ci_lo": lo, "ci_hi": hi, **{f"acc_seed{s}": a for s, a in zip(SEEDS, accs)},
               "best_bwt_mean": float(np.mean([res[b][s]["bwt"] for s in SEEDS])),
               "best_newest_task_acc_mean": float(np.mean([res[b][s]["newest_task_acc"] for s in SEEDS])),
               "interior": int(not e), "edges_hit": ";".join(e) if e else "none", "extension_rounds": rounds,
               "lr_grid": ";".join(f"{v:g}" for v in lr_grid), "mult_grid": ";".join(f"{v:g}" for v in mult_grid),
               "n_points": len(lr_grid) * len(mult_grid), "history": " | ".join(history)}
    WORK.mkdir(parents=True, exist_ok=True)
    grid_rows = [{"cell": idx, "method": method, "dataset": ds, "n_clients": n, "eps": eps, **row}
                 for k in sorted(res) for row in (res[k][s] for s in SEEDS)]
    out_g, out_s = WORK / f"cell_{idx:02d}.csv", WORK / f"summary_{idx:02d}.csv"
    for path, data in ((out_g, grid_rows), (out_s, [summary])):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    provenance.finalize(man, [out_g, out_s])
    return 0


def merge() -> int:
    man = manifest({"seed": 0, "purpose": "rebuttal R7b merge (sensitivity, not selection)", "protocol": "rebuttal/R7b_PROTOCOL.md"})
    r7 = _csv("results/rb_sensitivity.csv")
    rows, grid = [], []
    for idx, (method, ds, n, eps) in enumerate(CELLS):
        s = _csv(f"rebuttal/work/r7b/summary_{idx:02d}.csv")[0]
        grid += _csv(f"rebuttal/work/r7b/cell_{idx:02d}.csv")
        ref = {r["role"]: r for r in r7 if r["method"] == method and r["dataset"] == ds and int(r["n_clients"]) == n and float(r["eps"]) == eps}
        s.update({"selected_reported_mean": ref["selected"]["final_avg_acc_mean"],
                  "selected_reported_lr": ref["selected"]["lr"], "selected_reported_clip_multiplier": ref["selected"]["clip_multiplier"],
                  "r7_15pt_best_mean": ref["best"]["final_avg_acc_mean"],
                  "source_reference": "results/rb_sensitivity.csv (selected = reported, unchanged)"})
        rows.append(s)
        print(f"R7b {method:15s} {ds:10s} n={n:<5} eps={eps:<4} best lr={s['best_lr']:<6} mult={s['best_clip_multiplier']:<5} "
              f"acc={float(s['best_final_avg_acc_mean']):.4f} interior={s['interior']} edges={s['edges_hit']} rounds={s['extension_rounds']} "
              f"| selected={float(s['selected_reported_mean']):.4f} R7best={float(s['r7_15pt_best_mean']):.4f}")
    out, out_g = REPO_ROOT / "results" / "rb_sensitivity_ext.csv", REPO_ROOT / "results" / "rb_sensitivity_ext_grid.csv"
    for path, data in ((out, rows), (out_g, grid)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print(f"R7b MERGE cells={len(rows)} grid_rows={len(grid)}")
    provenance.finalize(man, [out, out_g])
    return 0


if __name__ == "__main__":
    mode = sys.argv[1]
    sys.exit(run_cell(int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ["PBS_ARRAY_INDEX"])) if mode == "cell" else merge())
