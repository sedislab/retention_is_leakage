#!/usr/bin/env python3
"""FX9-9 paired seed/task/target bootstrap for horizon retention and the M0 ratio."""

import csv
import sys
from pathlib import Path

import numpy as np
from build_fx2_summary import _load_seed_npz, _seed_arrays
from p3fcl import bootstrap_metrics, provenance
from p3fcl import rng as rng_mod
from p3fcl.halflife import (
    _pool_accuracy,
    accuracy_curve,
    bootstrap_accuracy_curve,
    decoupling_ratio,
    half_life,
)
from p3fcl.shadow_runner import METHOD_REGISTRY

ROOT = Path(__file__).resolve().parents[2]
N_REPLICATES = 2000


def draw_indices(seed_data, rng):
    """One draw reused by BOTH quantities: seeds, then K, then targets within K."""
    drawn = []
    for si in rng.integers(0, len(seed_data), size=len(seed_data)):
        tasks = rng.integers(0, 4, size=4)
        target_idx = []
        for k in tasks:
            candidates = np.flatnonzero(seed_data[si]["k"] == k)
            if len(candidates) == 0:
                raise ValueError(f"no targets for task {k}")
            target_idx.extend(rng.choice(candidates, size=len(candidates), replace=True))
        drawn.append((int(si), tasks, np.asarray(target_idx)))
    return drawn


def statistic(data, accuracy, draws, elapsed):
    acc_raw = np.stack([accuracy.samples[si, tasks, :] for si, tasks, _ in draws])
    acc_floors = np.stack([accuracy.floors[si, tasks, :] for si, tasks, _ in draws])
    raw, norm, _ = _pool_accuracy(acc_raw, acc_floors)
    leak = {}
    for e in elapsed:
        scores = np.concatenate([data[si]["scores"][e][:, targets].ravel() for si, _, targets in draws])
        labels = np.concatenate([data[si]["labels"][:, targets].ravel() for si, _, targets in draws])
        keep = np.isfinite(scores)
        leak[e] = bootstrap_metrics.tpr_at_fpr(scores[keep], labels[keep], 0.01)
    base = leak[0] - 0.01
    leak_norm = {e: (v - 0.01) / base if base > 0 else np.nan for e, v in leak.items()}
    return raw, norm, leak, leak_norm


def main():
    ds, method, view = sys.argv[1:4]
    seeds = list(range(5)) if method == "m0_fedavg" else list(range(3))
    with (ROOT / f"results/accuracy_matrix_{ds}.csv").open() as f:
        acc_rows = [r for r in csv.DictReader(f) if r["method"] == method]
    accuracy = accuracy_curve(acc_rows)
    data = []
    for seed in seeds:
        z = _load_seed_npz(ds, method, seed, view)
        scores, labels = _seed_arrays(z, "trajectory")
        data.append(dict(scores=scores, labels=labels, k=z["k_of_target"]))
    elapsed = list(range(7)) if method == "m0_fedavg" else [0, 6]
    point_draws = [(i, np.arange(4), np.arange(len(d["k"]))) for i, d in enumerate(data)]
    point = statistic(data, accuracy, point_draws, elapsed)
    rng = rng_mod.seeded("fx9.joint_bootstrap", 0)
    manifest = provenance.run_manifest(
        dict(
            phase="FX9-9",
            hypothesis="H2",
            dataset=ds,
            method=method,
            view=view,
            seeds=seeds,
            replicates=N_REPLICATES,
            sampling="paired seeds/tasks/targets",
        ),
        seed=0,
    )
    horizon = []
    ratios = []
    censored = 0
    for i in range(N_REPLICATES):
        draw = statistic(data, accuracy, draw_indices(data, rng), elapsed)
        horizon.append([draw[1][6], draw[3][6]])
        if method == "m0_fedavg" and np.isfinite(list(draw[3].values())).all():
            ha = half_life(dict(enumerate(draw[1])))
            hl = half_life(draw[3])
            ratio = decoupling_ratio(hl, ha)
            if ratio["ratio"] is not None:
                ratios.append(ratio["ratio"])
                censored += int(ratio["ratio_type"] == "lower_bound")
        if (i + 1) % 250 == 0:
            print(f"{ds}/{method}/{view} joint bootstrap {i+1}/{N_REPLICATES}", flush=True)
    ci = np.nanpercentile(horizon, [2.5, 97.5], axis=0)
    if method != "m0_fedavg":
        _, acc_boot = bootstrap_accuracy_curve(acc_rows, n_replicates=N_REPLICATES)
        ci[:, 0] = np.percentile(acc_boot["norm"][:, 6], [2.5, 97.5])
    row = dict(
        dataset=ds,
        method=method,
        family=METHOD_REGISTRY[method][1].value,
        view=view,
        acc_norm=accuracy.norm[6],
        acc_norm_ci_lo=ci[0, 0],
        acc_norm_ci_hi=ci[1, 0],
        leak_norm=point[3][6],
        leak_norm_ci_lo=ci[0, 1],
        leak_norm_ci_hi=ci[1, 1],
        n_seeds=len(seeds),
        n_accuracy_seeds=len(accuracy.seeds),
        n_leakage_seeds=len(seeds),
        n_replicates=N_REPLICATES,
    )
    outputs = []
    out = ROOT / f"results/fx9_horizon_{ds}_{method}_{view}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    outputs.append(out)
    if method == "m0_fedavg":
        ha = half_life(dict(enumerate(point[1])))
        hl = half_life(point[3])
        ratio = decoupling_ratio(hl, ha)
        ci_ratio = np.percentile(ratios, [2.5, 97.5]) if ratios else [np.nan, np.nan]
        rr = dict(
            dataset=ds,
            method=method,
            family="F1",
            view=view,
            h_acc=ha["halflife"],
            h_leak=hl["halflife"],
            **ratio,
            ratio_ci_lo=ci_ratio[0],
            ratio_ci_hi=ci_ratio[1],
            n_valid=len(ratios),
            n_replicates=N_REPLICATES,
            n_leak_censored=censored,
            ci_interpretation="interval for lower-bound statistic; not an upper bound on true ratio"
            if ratio["ratio_type"] == "lower_bound" or censored > 0
            else "paired percentile 95% interval; evaluable denominator replicates only",
        )
        out = ROOT / f"results/fx9_ratio_{ds}_{method}_{view}.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rr))
            w.writeheader()
            w.writerow(rr)
        outputs.append(out)
        print(f"FX9-9 ACCEPT {rr}", flush=True)
    provenance.finalize(manifest, outputs)


if __name__ == "__main__":
    main()
