#!/usr/bin/env python3
"""R1 equivalence gate (`rebuttal/R1_PROTOCOL.md`, "Gates inside R1" 2). PBS only.

For 3 (seed, shadow id) pairs per (dataset, method), this rebuilds the paper's masked stream and runs
the real method (`TARGET` / `FOT`, with the paper's shadow config plus wave override) through tasks
0..3 with `sim.run`'s rng. After each task it compares the cheap-path release with the method's own
release:
- M2: every client's F6 GENERATIVE mean/var, and `_generators` (the broadcast);
- M3: the GRAM payload R, and the new directions V lying in the updated subspace U.
It also checks that the rebuilt target_ids / in_out equal the paper's shadow file for that id.
Pass = every max |diff| < 1e-10. Writes `rebuttal/r1_equivalence.csv`. Exit 1 on any failure, so
the array that depends on this job (afterok) never starts.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import provenance  # noqa: E402
from p3fcl import rng as rng_mod  # noqa: E402
from p3fcl.artifacts import Family  # noqa: E402
from p3fcl.rebuttal import release_audit as ra  # noqa: E402
from p3fcl.rebuttal.common import manifest, paired_setup, wave_override  # noqa: E402
from p3fcl.shadow_runner import (  # noqa: E402
    METHOD_REGISTRY, _apply_population_mask, _method_config, _population_mask,
)

TOL = 1e-10
TOL_FLOAT32_SPAN = 1e-5   # ~100x float32 eps; only for the secondary span check (R1_PROTOCOL.md amendment 1)
PAIRS = [(0, 0), (1, 511), (2, 1023)]   # (seed, shadow id)
TASKS = [0, 1, 2, 3]                    # the K set
OUT_CSV = REPO_ROOT / "rebuttal" / "r1_equivalence.csv"


def main() -> int:
    man = manifest({"seed": 0, "purpose": "rebuttal R1 equivalence gate", "pairs": PAIRS, "tasks": TASKS, "tol": TOL,
                    "tol_float32_span": TOL_FLOAT32_SPAN, "amendment": "R1_PROTOCOL.md amendment 1"})
    rows = []
    for dataset in ["cifar100", "cub200", "imagenet_r"]:
        for method in ["m2_target", "m3_fot"]:
            for seed, sid in PAIRS:
                s = paired_setup(dataset, method, seed)
                X, y = s["X"], s["y"]
                keep = _population_mask(dataset, method, sid, len(X), s["p_in"])
                stream = _apply_population_mask(s["base"], keep)
                with np.load(s["paper_dir"] / f"shadow_{sid:06d}.npz") as z:
                    tids = np.array([t["target_id"] for t in s["targets"]])
                    pairing_ok = bool(np.array_equal(z["target_ids"], tids) and np.array_equal(z["in_out"], keep[tids]))

                cls, _ = METHOD_REGISTRY[method]
                cfg = {**_method_config(method, int(y.max()) + 1, X.shape[1]), **wave_override(dataset, method, seed)}
                m = cls(cfg)
                r = rng_mod.seeded(f"sim.run::{m.spec.name}", sid)
                for t in TASKS:
                    recs = m.fit_task(t, X, y, ids=None, client_shards=stream[t], rng=r)
                    diffs = {}
                    if method == "m2_target":
                        client_stats, generators = ra.m2_task_release(X, y, stream[t])
                        dmax, n = 0.0, 0
                        for rec in (rc for rc in recs if rc.family == Family.GENERATIVE):
                            for key, val in rec.payload.items():
                                kind, c = key.split("_")
                                mean, var = client_stats[(rec.client, int(c))]
                                dmax = max(dmax, float(np.max(np.abs((mean if kind == "mean" else var) - val))))
                                n += 1
                        diffs["client_mean_var"] = dmax if n == 2 * len(client_stats) else float("inf")
                        gm = max(float(np.max(np.abs(generators[c][0] - m._generators[c][0]))) for c in generators)
                        gv = max(float(np.max(np.abs(generators[c][1] - m._generators[c][1]))) for c in generators)
                        diffs["broadcast_generator"] = max(gm, gv)
                    else:
                        R, V = ra.m3_task_release(X, stream[t])
                        gram = [rc for rc in recs if rc.family == Family.GRAM]
                        diffs["server_R"] = float(np.max(np.abs(gram[0].payload["R"] - R))) if len(gram) == 1 else float("inf")
                        # FOT's own lines applied to its own released R -> its `new_dirs`.
                        ev, evec = np.linalg.eigh(gram[0].payload["R"])
                        method_dirs = evec[:, np.argsort(ev)[::-1][:min(cfg["subspace_rank"], X.shape[1])]]
                        diffs["broadcast_subspace_exact"] = float(np.max(np.abs(V - method_dirs)))
                        # Secondary: V lies in FOT's updated U. U comes from a float32 QR, so the floor is ~1e-7.
                        diffs["broadcast_subspace_in_U"] = float(np.max(np.abs(V - m.U @ (m.U.T @ V))))
                    for stat, d in diffs.items():
                        ok = d < (TOL_FLOAT32_SPAN if stat == "broadcast_subspace_in_U" else TOL) and pairing_ok
                        rows.append({"dataset": dataset, "method": method, "seed": seed, "shadow_id": sid, "task": t,
                                     "statistic": stat, "max_abs_diff": repr(d), "pairing_ok": int(pairing_ok), "pass": int(ok)})
                        print(f"R1-EQ {dataset} {method} seed={seed} shadow={sid} task={t} {stat}: "
                              f"max|diff|={d:.3e} pairing_ok={pairing_ok} {'PASS' if ok else 'FAIL'}", flush=True)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_pass = sum(r["pass"] for r in rows)
    print(f"R1-EQ SUMMARY checks={len(rows)} passed={n_pass} max_abs_diff={max(float(r['max_abs_diff']) for r in rows):.3e} tol={TOL}")
    provenance.finalize(man, [OUT_CSV])
    return 0 if n_pass == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
