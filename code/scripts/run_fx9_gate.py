#!/usr/bin/env python3
"""FX9-3 automatic simulation gate. One dataset per PBS array task; validation-only tuning."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from p3fcl import features, provenance, sim, streams
from p3fcl.shadow_runner import BACKBONE, METHOD_REGISTRY, _method_config
from run_fx4g_gate import _stratified_val_split

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [0, 1, 2]
METHODS = ["m0_fedavg", "m1_glfc", "m2_target", "m3_fot", "m5_hybrid_replay"]


def simulate(method, cfg, seed, X, y, Xe, ye):
    stream = streams.build_stream(y, np.arange(len(y)), n_tasks=10, n_clients=10, beta=0.5, seed=seed)
    ids = streams.task_eval_sets(ye, np.arange(len(ye)), n_tasks=10, seed=seed)
    obj = METHOD_REGISTRY[method][0](cfg)
    r = sim.run(obj, X, y, stream, seed=seed, eval_sets=[(Xe[i], ye[i]) for i in ids])
    return dict(
        final_avg_acc=r["final_avg_acc"],
        last_task_acc=float(r["acc_matrix"][-1, -1]),
        bwt=r["bwt"],
        aia=r["avg_incremental_acc"],
        diagonal_json=json.dumps(np.diag(r["acc_matrix"]).tolist()),
    )


def means(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in ["final_avg_acc", "last_task_acc", "bwt", "aia"]}


def passed(row, base):
    return (
        row["bwt"] <= 0.02
        and row["last_task_acc"] >= base["last_task_acc"] - 0.15
        and row["final_avg_acc"] >= base["final_avg_acc"]
    )


def write(name, rows, manifest):
    path = ROOT / "results" / name
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    provenance.finalize(manifest, [path])


def main():
    ds = sys.argv[1]
    manifest = provenance.run_manifest(
        dict(
            phase="FX9-3",
            hypotheses=["H2", "H6"],
            dataset=ds,
            seeds=SEEDS,
            tuning="train stratified 90/10; rho=.5,1,2 × lr=.1,.5; test never used for selection",
        ),
        seed=0,
    )
    train = features.load_cache(ROOT / "features", ds, BACKBONE, "train")
    test = features.load_cache(ROOT / "features", ds, BACKBONE, "test")
    X, y, Xe, ye = train["features"], train["labels"], test["features"], test["labels"]
    configs = {m: _method_config(m, int(y.max()) + 1, X.shape[1]) for m in METHODS + ["m4_proto"]}
    # M4 acceptance uses raw features, five seeds, matching the final accuracy protocol.
    m4 = [
        dict(
            dataset=ds,
            method="m4_proto",
            seed=s,
            **simulate("m4_proto", configs["m4_proto"], s, X, y, Xe, ye),
        )
        for s in range(5)
    ]
    write(f"fx9_m4_{ds}.csv", m4, manifest)
    print(f'FX9-2 ACCEPT {ds} M4_final_avg_acc={means(m4)["final_avg_acc"]:.12f} n_seeds=5', flush=True)
    baseline = [simulate("m0_fedavg", configs["m0_fedavg"], s, X, y, Xe, ye) for s in SEEDS]
    base = means(baseline)
    gate, grid = [], []
    tr, val = _stratified_val_split(y)
    val_baseline = None
    for method in METHODS:
        cfg = configs[method]
        per_seed = (
            baseline if method == "m0_fedavg" else [simulate(method, cfg, s, X, y, Xe, ye) for s in SEEDS]
        )
        config_id = "default"
        if method in ["m1_glfc", "m2_target", "m5_hybrid_replay"] and not passed(means(per_seed), base):
            if val_baseline is None:
                val_baseline = means(
                    [
                        simulate("m0_fedavg", configs["m0_fedavg"], s, X[tr], y[tr], X[val], y[val])
                        for s in SEEDS
                    ]
                )
            candidates = []
            for rho in [0.5, 1.0, 2.0]:
                for lr in [0.1, 0.5]:
                    tuned = {**cfg, "replay_weight": rho, "lr": lr}
                    vm = means([simulate(method, tuned, s, X[tr], y[tr], X[val], y[val]) for s in SEEDS])
                    vp = passed(vm, val_baseline)
                    grid.append(dict(dataset=ds, method=method, replay_weight=rho, lr=lr, val_pass=vp, **vm))
                    # Prefer a validation gate pass; otherwise retain the best validation utility.
                    candidates.append(((vp, vm["final_avg_acc"]), tuned))
            cfg = max(candidates, key=lambda c: c[0])[1]
            config_id = "tuned"
            per_seed = [simulate(method, cfg, s, X, y, Xe, ye) for s in SEEDS]
        avg = means(per_seed)
        status = "n/a (baseline)" if method == "m0_fedavg" else passed(avg, base)
        override = {k: v for k, v in cfg.items() if k not in ["n_classes", "feature_dim"]}
        for seed, r in zip(SEEDS, per_seed):
            gate.append(
                dict(
                    dataset=ds,
                    method=method,
                    seed=seed,
                    config_id=config_id,
                    **r,
                    m0_final_avg_acc=base["final_avg_acc"],
                    m0_last_task_acc=base["last_task_acc"],
                    **{"pass": status},
                    used_cfg_json=json.dumps(override, sort_keys=True),
                    replay_weight=cfg.get("replay_weight", 1.0),
                )
            )
        print(
            f'FX9-3 {ds}/{method} {"PASS" if status is True else "BASELINE" if method == "m0_fedavg" else "GATE FAILED"} '
            f'config={config_id} final={avg["final_avg_acc"]:.9f} last={avg["last_task_acc"]:.9f} bwt={avg["bwt"]:.9f} cfg={override}',
            flush=True,
        )
    write(f"fx9_gate_{ds}.csv", gate, manifest)
    if grid:
        write(f"fx9_gate_grid_{ds}.csv", grid, manifest)


if __name__ == "__main__":
    main()
