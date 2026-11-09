#!/usr/bin/env python3
"""R5 DP-FedAvg baselines (`rebuttal/R5_PROTOCOL.md`). PBS only.

`rb_r5_dp_fedavg.py gate`: correctness gate. Writes `rebuttal/r5_gate.csv` and exits 1 on failure.
`rb_r5_dp_fedavg.py main`: selection on ref plus final runs. Writes `results/rb_dp_fedavg.csv` and
`results/rb_dp_fedavg_selection.csv`. Submitted afterok on the gate.
"""
from __future__ import annotations

import csv
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))
sys.path.insert(0, str(REPO_ROOT / "code" / "scripts"))

from p3fcl import features, provenance, sim, streams  # noqa: E402
from p3fcl.dp.accountant import account  # noqa: E402
from p3fcl.methods.m0_dpfedavg_u2 import DPFedAvgU2  # noqa: E402
from p3fcl.methods.m0_dpsgd_u1 import DPSGDFedAvgU1, noise_multiplier_for, per_example_grad_norms  # noqa: E402
from p3fcl.methods.m0_fedavg import FedAvgSequential, _softmax  # noqa: E402
from p3fcl.rebuttal.common import manifest  # noqa: E402
from p3fcl.units import Unit  # noqa: E402
from run_m9_hparam_selection import _ref_train_val_split  # noqa: E402
from run_m9_sweep import BACKBONE, BETA, N_CLIENTS_DEFAULT, N_TASKS  # noqa: E402

DATASETS = ["cifar100", "cub200", "imagenet_r"]
EPS = [1.0, 4.0, 8.0, float("inf")]
SEEDS = [0, 1, 2]
LR_GRID = [0.1, 0.5, 1.0]
MULT_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
EPOCHS, DELTA, GATE_TOL = 30, 1e-5, 0.01


WCAP_CSV = REPO_ROOT / "rebuttal" / "r5_wcap.csv"


def compute_wcap(ds, n_clients=N_CLIENTS_DEFAULT):
    """McMahan weight cap from public data only (R5_PROTOCOL.md amendment 1)."""
    from p3fcl.rebuttal.client_scaling import CAP_SEEDS, PERCENTILE, public_client_sizes
    _, yr = _load(ds, "ref")
    tr, _ = _ref_train_val_split(yr, seed=0, frac=0.5)
    n_train = len(_load(ds, "train")[1])  # dataset size only (public metadata), no features or labels
    sizes = public_client_sizes(yr[tr], n_train, N_TASKS, n_clients, BETA)
    return {"dataset": ds, "n_clients": n_clients, "w_cap": float(np.percentile(sizes, PERCENTILE)), "percentile": PERCENTILE,
            "n_pseudo_clients": len(sizes), "median_size": float(np.median(sizes)), "max_size": int(sizes.max()),
            "cap_seeds": f"{CAP_SEEDS[0]}-{CAP_SEEDS[-1]}", "n_train_public": n_train}


WCAP_NSCALE_CSV = REPO_ROOT / "rebuttal" / "r5_wcap_nscale.csv"
NSCALE_N = [100, 1000]


def read_wcap(path=WCAP_CSV) -> dict:
    with open(path, newline="") as f:
        return {(r["dataset"], int(r["n_clients"])): float(r["w_cap"]) for r in csv.DictReader(f)}


def _load(ds, split):
    c = features.load_cache(REPO_ROOT / "features", ds, BACKBONE, split)
    return c["features"], c["labels"]


def _run(method, X, y, Xe, ye, n_clients, seed):
    stream = streams.build_stream(y, np.arange(len(y)), n_tasks=N_TASKS, n_clients=n_clients, beta=BETA, seed=seed)
    eval_ids = streams.task_eval_sets(ye, np.arange(len(ye)), n_tasks=N_TASKS, seed=seed)
    res = sim.run(method, X, y, stream, seed=seed, eval_sets=[(Xe[i], ye[i]) for i in eval_ids])
    T = res["acc_matrix"].shape[0]
    return stream, res, {"final_avg_acc": res["final_avg_acc"], "bwt": res["bwt"],
                         "newest_task_acc": float(res["acc_matrix"][T - 1, T - 1]), "avg_incremental_acc": res["avg_incremental_acc"]}


def _make(method, n_classes, d, lr, clip, eps, seed, aggregation="fixed_weighted", w_cap=None):
    base = {"n_classes": n_classes, "feature_dim": d, "local_epochs": EPOCHS, "lr": lr, "noise_seed": seed}
    if method == "m0_dpsgd_u1":
        return DPSGDFedAvgU1({**base, "clip_Cg": clip, "noise_multiplier": noise_multiplier_for(eps, EPOCHS, DELTA)})
    if method == "m0_dpfedavg_u2":
        cfg = {**base, "clip_S": clip, "eps": eps, "delta": DELTA, "aggregation": aggregation}
        if w_cap is not None:
            cfg["w_cap"] = w_cap
        return DPFedAvgU2(cfg)
    return FedAvgSequential(base)


# ------------------------------------------------------------------ gate

def gate_dataset(ds):
    X, y = _load(ds, "train")
    Xe, ye = _load(ds, "test")
    n_classes, d = int(y.max()) + 1, X.shape[1]
    with open(REPO_ROOT / "results" / f"m9_sweep_{ds}.csv", newline="") as f:
        paper = {int(r["seed"]): float(r["final_avg_acc"]) for r in csv.DictReader(f) if r["method"] == "m0_fedavg"}
    rows = []
    for seed in SEEDS:
        _, _, m0 = _run(_make("m0", n_classes, d, 0.5, np.inf, np.inf, seed), X, y, Xe, ye, N_CLIENTS_DEFAULT, seed)
        rows.append({"dataset": ds, "seed": seed, "method": "m0_fedavg", "acc": m0["final_avg_acc"], "m0_acc": m0["final_avg_acc"],
                     "abs_diff": abs(m0["final_avg_acc"] - paper[seed]), "tol": 0.0, "check": "equals results/m9_sweep M0 row",
                     "pass": int(m0["final_avg_acc"] == paper[seed])})
        stream = streams.build_stream(y, np.arange(len(y)), n_tasks=N_TASKS, n_clients=N_CLIENTS_DEFAULT, beta=BETA, seed=seed)
        big_cap = float(max(len(sh.ids) for t in stream for sh in t))  # no cap binds
        for method, agg, tol, cap in (("m0_dpsgd_u1", "n/a", GATE_TOL, None),
                                      ("m0_dpfedavg_u2", "fixed_weighted", GATE_TOL, big_cap),
                                      ("m0_dpfedavg_u2", "weighted", 0.0, None)):
            _, _, r = _run(_make(method, n_classes, d, 0.5, np.inf, np.inf, seed, agg, cap), X, y, Xe, ye, N_CLIENTS_DEFAULT, seed)
            diff = abs(r["final_avg_acc"] - m0["final_avg_acc"])
            rows.append({"dataset": ds, "seed": seed, "method": f"{method}[{agg}]", "acc": r["final_avg_acc"], "m0_acc": m0["final_avg_acc"],
                         "abs_diff": diff, "tol": tol, "check": "no clip, no noise" + (f", w_cap={cap:g} (no cap binds)" if cap else "") + " vs M0",
                         "pass": int(diff <= tol)})
            print(f"R5-GATE {ds} seed={seed} {method}[{agg}]: acc={r['final_avg_acc']:.6f} M0={m0['final_avg_acc']:.6f} diff={diff:.2e} "
                  f"{'PASS' if diff <= tol else 'FAIL'}", flush=True)
    return rows


# ------------------------------------------------------------------ main

def main_dataset(ds, n_clients=N_CLIENTS_DEFAULT, methods=("m0_dpsgd_u1", "m0_dpfedavg_u2")):
    """Selection always uses the paper's ref-train federation (10 clients), as in R3. Only the
    n-dependent quantity (here w_cap(n)) changes; the final runs use `n_clients`."""
    w_cap = read_wcap(WCAP_CSV if n_clients == N_CLIENTS_DEFAULT else WCAP_NSCALE_CSV)[(ds, n_clients)]
    X, y = _load(ds, "train")
    Xe, ye = _load(ds, "test")
    Xr, yr = _load(ds, "ref")
    n_classes, d = int(y.max()) + 1, X.shape[1]
    tr, va = _ref_train_val_split(yr, seed=0, frac=0.5)
    Xtr, ytr, Xva, yva = Xr[tr], yr[tr], Xr[va], yr[va]

    # Medians on ref-train.
    Y0 = np.zeros((len(ytr), n_classes))
    Y0[np.arange(len(ytr)), ytr] = 1.0
    med_g = float(np.median(per_example_grad_norms(Xtr, _softmax(np.zeros((len(ytr), n_classes))) - Y0)))
    med_u = {}
    for lr in LR_GRID:
        _, res, _ = _run(_make("m0", n_classes, d, lr, np.inf, np.inf, 0), Xtr, ytr, Xva, yva, N_CLIENTS_DEFAULT, 0)
        med_u[lr] = float(np.median([np.linalg.norm(rec.payload) for rec in res["ledger"]]))
    print(f"R5 {ds} median per-example grad norm={med_g:.4f} median update norm={med_u}", flush=True)

    sel_rows, winners = [], {}
    for method in methods:
        for eps in EPS:
            best = None
            for lr in LR_GRID:
                med = med_g if method == "m0_dpsgd_u1" else med_u[lr]
                for mult in MULT_GRID:
                    _, _, r = _run(_make(method, n_classes, d, lr, med * mult, eps, 0, w_cap=w_cap), Xtr, ytr, Xva, yva, N_CLIENTS_DEFAULT, 0)
                    row = {"dataset": ds, "n_clients": n_clients, "method": method, "eps": eps, "lr": lr, "clip_multiplier": mult, "clip": med * mult,
                           "median_norm": med, "ref_val_final_avg_acc": r["final_avg_acc"]}
                    sel_rows.append(row)
                    if best is None or row["ref_val_final_avg_acc"] > best["ref_val_final_avg_acc"]:
                        best = row
            winners[(method, eps)] = best
            print(f"R5 {ds} {method} eps={eps}: selected lr={best['lr']} mult={best['clip_multiplier']} ref_val={best['ref_val_final_avg_acc']:.4f}", flush=True)
    for r in sel_rows:
        w = winners[(r["method"], r["eps"])]
        r["selected"] = int(r is w)

    out = []
    for (method, eps), w in winners.items():
        for seed in SEEDS:
            meth = _make(method, n_classes, d, w["lr"], w["clip"], eps, seed, w_cap=w_cap)
            stream, res, m = _run(meth, X, y, Xe, ye, n_clients, seed)
            unit = Unit.EXAMPLE if method == "m0_dpsgd_u1" else Unit.TASK
            max_m = int(account(res["ledger"], stream, unit, sigma=1.0, delta=DELTA)["m_T"].max()) if seed == 0 else ""
            frac = float(np.mean(meth.frac_clipped_steps)) if method == "m0_dpsgd_u1" else float(np.mean(meth.frac_clipped_rounds))
            noise = meth.z if method == "m0_dpsgd_u1" else meth.sigma
            out.append({"dataset": ds, "n_clients": n_clients, "method": method, "unit": unit.value, "eps": eps, "seed": seed, "lr": w["lr"],
                        "clip_multiplier": w["clip_multiplier"], "clip": w["clip"], "noise": noise,
                        "noise_kind": "z (noise multiplier)" if method == "m0_dpsgd_u1" else "sigma (absolute)",
                        "frac_clipped": frac, "w_cap": w_cap if method == "m0_dpfedavg_u2" else "n/a",
                        "frac_capped": float(np.mean(meth.frac_capped_rounds)) if method == "m0_dpfedavg_u2" else "n/a",
                        "aggregation": "fixed_weighted (McMahan)" if method == "m0_dpfedavg_u2" else "n/a", "ledger_max_m_T": max_m, "lifelong_eps": eps if max_m in ("", 1) else "CHECK", **m})
            print(f"R5 {ds} {method} eps={eps} seed={seed}: acc={m['final_avg_acc']:.4f} bwt={m['bwt']:.4f} newest={m['newest_task_acc']:.4f} "
                  f"clipped={frac:.3f} noise={noise:.4f} max_m_T={max_m}", flush=True)
    return out, sel_rows


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    mode = sys.argv[1]
    man = manifest({"seed": 0, "purpose": f"rebuttal R5 DP-FedAvg baselines ({mode})", "datasets": DATASETS, "eps": EPS,
                    "seeds": SEEDS, "lr_grid": LR_GRID, "mult_grid": MULT_GRID, "epochs": EPOCHS, "delta": DELTA,
                    "gate_tol": GATE_TOL, "aggregation": "fixed_weighted (McMahan 2018)", "protocol": "rebuttal/R5_PROTOCOL.md amendment 1"})
    if mode == "nscale_cap":
        caps = [compute_wcap("cifar100", n) for n in NSCALE_N]
        _write(WCAP_NSCALE_CSV, caps)
        for c in caps:
            print(f"R5 W_CAP cifar100 n={c['n_clients']}: w_cap={c['w_cap']:.2f} (median {c['median_size']}, max {c['max_size']}, n={c['n_pseudo_clients']})")
        provenance.finalize(man, [WCAP_NSCALE_CSV])
        return 0
    if mode == "nscale":
        with mp.get_context("fork").Pool(len(NSCALE_N)) as pool:
            parts = pool.starmap(main_dataset, [("cifar100", n, ("m0_dpfedavg_u2",)) for n in NSCALE_N])
        rows = [r for p in parts for r in p[0]]
        sel = [r for p in parts for r in p[1]]
        out, out_sel = REPO_ROOT / "results" / "rb_dp_fedavg_nscale.csv", REPO_ROOT / "results" / "rb_dp_fedavg_nscale_selection.csv"
        _write(out, rows)
        _write(out_sel, sel)
        print(f"R5 NSCALE DONE rows={len(rows)} selection_rows={len(sel)}")
        provenance.finalize(man, [out, out_sel])
        return 0
    with mp.get_context("fork").Pool(len(DATASETS)) as pool:
        parts = pool.map(gate_dataset if mode == "gate" else main_dataset, DATASETS)
    if mode == "gate":
        caps = [compute_wcap(ds) for ds in DATASETS]
        _write(WCAP_CSV, caps)
        for c in caps:
            print(f"R5 W_CAP {c['dataset']}: w_cap={c['w_cap']:.2f} (median size {c['median_size']}, max {c['max_size']}, n={c['n_pseudo_clients']})")
        rows = [r for p in parts for r in p]
        out = REPO_ROOT / "rebuttal" / "r5_gate.csv"
        _write(out, rows)
        n_pass = sum(r["pass"] for r in rows)
        print(f"R5-GATE SUMMARY checks={len(rows)} passed={n_pass}")
        provenance.finalize(man, [out, WCAP_CSV])
        return 0 if n_pass == len(rows) else 1
    rows = [r for p in parts for r in p[0]]
    sel = [r for p in parts for r in p[1]]
    out, out_sel = REPO_ROOT / "results" / "rb_dp_fedavg.csv", REPO_ROOT / "results" / "rb_dp_fedavg_selection.csv"
    _write(out, rows)
    _write(out_sel, sel)
    print(f"R5 DONE rows={len(rows)} selection_rows={len(sel)}")
    provenance.finalize(man, [out, out_sel])
    return 0


if __name__ == "__main__":
    sys.exit(main())
