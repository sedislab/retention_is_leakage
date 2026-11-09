#!/usr/bin/env python3
"""R1 release-only shadow scores (`rebuttal/R1_PROTOCOL.md`). PBS array, one task per
(dataset, method, seed), index = `$PBS_ARRAY_INDEX` in 0..17.

For shadow ids 0..1023: rebuild the paper's population mask, compute the task-k release statistic
for every target and write `shadows_rb/<ds>/<method>/seed<S>/shadow_XXXXXX.npz` in the existing
schema. Each file's target_ids / in_out must equal the paper's shadow file with the same id, or the
task fails. Resumable: an existing output is re-checked for pairing, not recomputed. Ends with
`index.csv` (sha256 per file, NaN counts), which carries the provenance sidecar.
"""
from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import provenance  # noqa: E402
from p3fcl.rebuttal import release_audit as ra  # noqa: E402
from p3fcl.rebuttal.common import manifest, paired_setup  # noqa: E402
from p3fcl.shadow_runner import _apply_population_mask, _population_mask  # noqa: E402

DATASETS = ["cifar100", "cub200", "imagenet_r"]
METHODS = ["m2_target", "m3_fot"]
SEEDS = [0, 1, 2]
N_SHADOWS = 1024
COMBOS = [(d, m, s) for d in DATASETS for m in METHODS for s in SEEDS]
OUT_ROOT = REPO_ROOT / "shadows_rb"

_W: dict = {}


def _one(sid: int) -> dict:
    w = _W
    out = w["out_dir"] / f"shadow_{sid:06d}.npz"
    keep = _population_mask(w["dataset"], w["method"], sid, len(w["X"]), w["p_in"])
    in_out = keep[w["tids"]]
    with np.load(w["paper_dir"] / f"shadow_{sid:06d}.npz") as z:
        if not (np.array_equal(z["target_ids"], w["tids"]) and np.array_equal(z["in_out"], in_out)):
            raise AssertionError(f"shadow {sid}: target_ids/in_out differ from the paper store")
    if not out.exists():
        stream = _apply_population_mask(w["base"], keep)
        scores = ra.release_scores(w["method"], w["X"], w["y"], stream, w["targets"], ref_centres=w["centres"])
        views = list(ra.RELEASE_VIEWS[w["method"]])
        tmp = out.with_name(out.name + f".tmp.{os.getpid()}.npz")
        np.savez_compressed(
            tmp, shadow_id=np.int64(sid), target_ids=w["tids"], target_task=w["ttask"],
            target_client=w["tclient"], in_out=in_out, views=np.array(views),
            **{f"scores_{v}": scores[v] for v in views},
        )
        os.replace(tmp, out)
    with np.load(out) as z:
        if not (np.array_equal(z["in_out"], in_out) and np.array_equal(z["target_ids"], w["tids"])):
            raise AssertionError(f"{out}: stored in_out/target_ids do not match the rebuilt mask")
        nan = {v: int(sum(np.isnan(z[f"scores_{v}"][j, t["task"]]) for j, t in enumerate(w["targets"])))
               for v in ra.RELEASE_VIEWS[w["method"]]}
    return {"shadow_id": sid, "file": out.name, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            **{f"nan_at_k_{v}": n for v, n in nan.items()}}


def main() -> int:
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ["PBS_ARRAY_INDEX"])
    workers = int(os.environ.get("NCPUS", "8"))
    dataset, method, seed = COMBOS[idx]
    man = manifest({"seed": seed, "purpose": "rebuttal R1 release-only shadow scores", "dataset": dataset,
                    "method": method, "n_shadows": N_SHADOWS, "views": list(ra.RELEASE_VIEWS[method]),
                    "m3_subspace_rank": ra.M3_SUBSPACE_RANK, "m3_server_lambda": ra.M3_SERVER_LAMBDA,
                    "m2_var_floor": ra.M2_VAR_FLOOR, "protocol": "rebuttal/R1_PROTOCOL.md"}, seed=seed)
    s = paired_setup(dataset, method, seed)
    out_dir = OUT_ROOT / dataset / method / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("shadow_*.npz.tmp.*.npz"):  # left by a killed earlier attempt
        stale.unlink()
    (out_dir / "targets.json").write_text(json.dumps(s["targets"]))
    _W.update(dataset=dataset, method=method, out_dir=out_dir, **s,
              tids=np.array([t["target_id"] for t in s["targets"]], dtype=np.int64),
              ttask=np.array([t["task"] for t in s["targets"]], dtype=np.int64),
              tclient=np.array([t["client"] for t in s["targets"]], dtype=np.int64))
    print(f"R1-SHADOWS {dataset}/{method}/seed{seed}: {len(s['targets'])} targets, {workers} workers -> {out_dir}", flush=True)
    with mp.get_context("fork").Pool(workers) as pool:
        rows = sorted(pool.imap_unordered(_one, range(N_SHADOWS), chunksize=8), key=lambda r: r["shadow_id"])
    index = out_dir / "index.csv"
    with open(index, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_t = len(s["targets"]) * N_SHADOWS
    for v in ra.RELEASE_VIEWS[method]:
        n_nan = sum(r[f"nan_at_k_{v}"] for r in rows)
        print(f"R1-SHADOWS {dataset}/{method}/seed{seed} view={v}: NaN at own task {n_nan}/{n_t} ({n_nan / n_t:.4%})")
    print(f"R1-SHADOWS DONE {dataset}/{method}/seed{seed} files={len(rows)} paired_with_paper={len(rows)}")
    provenance.finalize(man, [index])
    return 0


if __name__ == "__main__":
    sys.exit(main())
