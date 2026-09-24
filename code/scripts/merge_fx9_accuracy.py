#!/usr/bin/env python3
"""Validate independent utility reruns before replacing shared accuracy and TAB05 outputs."""

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from p3fcl import provenance
from p3fcl.experiment import DATASETS, FX9_START, METHODS

ROOT = Path(__file__).resolve().parents[2]


def main():
    matrices, utilities, errors = {}, [], []
    for ds in DATASETS:
        matrices[ds] = []
        for method in METHODS:
            path = ROOT / f"results/accuracy_matrix_{ds}_{method}.csv"
            meta = json.loads(path.with_suffix(".csv.meta.json").read_text())
            assert meta["utc_start"] >= FX9_START and meta["pbs_jobid"]
            with path.open() as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 275
            for seed in range(5):
                final = [float(r["acc"]) for r in rows if int(r["seed"]) == seed and int(r["task_T"]) == 9]
                assert len(final) == 10
                short = method.split("_")[0].upper()
                record = json.loads(
                    (ROOT / f"runs/fx9_utility_baseline/{short}_{ds}_t10_s{seed}.json").read_text()
                )
                assert record["phase"] == "FX9"
                err = abs(np.mean(final) - record["final_avg_acc"])
                errors.append(err)
                assert err < 1e-9, (ds, method, seed, err)
                utilities.append(record)
            matrices[ds].extend(rows)
    archive = ROOT / "archive/fx9_shared" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    archive.mkdir(parents=True)
    outputs = []
    manifest = provenance.run_manifest(dict(phase="FX9-4", hypotheses=["H2", "H12"], n_combos=21), seed=0)
    for ds, rows in matrices.items():
        out = ROOT / f"results/accuracy_matrix_{ds}.csv"
        if out.exists():
            shutil.copy2(out, archive / out.name)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        outputs.append(out)
    provenance.finalize(manifest, outputs)
    print(
        f"FX9-4 ACCEPT dataset_method_combos=21 seeds_per_combo=5 compared_final_rows={len(errors)} max_abs_error={max(errors):.12g}",
        flush=True,
    )
    for ds in DATASETS:
        for m in METHODS:
            rs = [r for r in utilities if r["dataset"] == ds and r["method"] == m.split("_")[0].upper()]
            print(
                f'FX9-4 ACCEPT {ds}/{m} final_avg_acc={np.mean([r["final_avg_acc"] for r in rs]):.12f}',
                flush=True,
            )


if __name__ == "__main__":
    main()
