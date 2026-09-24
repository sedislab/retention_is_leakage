#!/usr/bin/env python3
"""Merge current per-dataset gate files and print every FX9-2/3 acceptance number."""

import csv
import json
from pathlib import Path

import numpy as np
from p3fcl import provenance
from p3fcl.experiment import DATASETS, REPLAY_METHODS

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "archive/2026-09-23b_pre_fx9/results"


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write(path, rows, manifest):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    provenance.finalize(manifest, [path])


def main():
    rows = [r for ds in DATASETS for r in read(ROOT / f"results/fx9_gate_{ds}.csv")]
    assert len(rows) == 45
    assert len({(r["dataset"], r["method"], r["seed"]) for r in rows}) == 45
    manifest = provenance.run_manifest(
        dict(phase="FX9-3", hypotheses=["H2", "H6"], inputs=[f"fx9_gate_{d}.csv" for d in DATASETS]), seed=0
    )
    old = read(ARCHIVE / "fx4_gate.csv")
    for ds, old_acc in zip(DATASETS, [0.66, 0.56, 0.29]):
        m4 = read(ROOT / f"results/fx9_m4_{ds}.csv")
        print(
            f'FX9-2 ACCEPT {ds} M4 old_TAB05={old_acc:.2f} new_raw_final={np.mean([float(r["final_avg_acc"]) for r in m4]):.12f} n_seeds=5'
        )
        for method in sorted(REPLAY_METHODS):
            before = [r for r in old if r["dataset"] == ds and r["method"] == method]
            after = [r for r in rows if r["dataset"] == ds and r["method"] == method]

            def avg(rr, col):
                return np.mean([float(r[col]) for r in rr])

            print(
                f'FX9-3 ACCEPT {ds}/{method} last_before={avg(before,"last_task_acc"):.12f} last_after={avg(after,"last_task_acc"):.12f} '
                f'bwt_before={avg(before,"bwt"):.12f} bwt_after={avg(after,"bwt"):.12f} gate={after[0]["pass"]} cfg={after[0]["used_cfg_json"]}'
            )
    write(ROOT / "results/fx9_gate.csv", rows, manifest)
    matrix = read(ARCHIVE / "accuracy_matrix_cub200.csv")
    diag = []
    for method in sorted(REPLAY_METHODS):
        after = next(
            r for r in rows if r["dataset"] == "cub200" and r["method"] == method and r["seed"] == "0"
        )
        values = json.loads(after["diagonal_json"])
        for k in range(10):
            before = next(
                r
                for r in matrix
                if r["method"] == method
                and r["seed"] == "0"
                and int(r["task_k"]) == k
                and r["elapsed"] == "0"
            )
            diag.append(
                dict(
                    dataset="cub200",
                    method=method,
                    seed=0,
                    task_k=k,
                    acc_before=before["acc"],
                    acc_after=values[k],
                )
            )
    write(ROOT / "results/fx9_lag_diagnostic.csv", diag, manifest)


if __name__ == "__main__":
    main()
