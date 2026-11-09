#!/usr/bin/env python3
"""R3 merge (`rebuttal/R3_PROTOCOL.md`): combines `rebuttal/work/r3/part_*.csv` into
`results/rb_client_scaling.csv` (+ `_selection.csv`). PBS only.

Hard checks against `results/m9_sweep_<ds>.csv` (bit for bit): `old_fixed_C` U2 rows at n = 10
(every dataset) and at n ∈ {50, 100} (CIFAR-100, the paper's arm); `u1_reference` and `m8_reference`
rows at n = 10. On any failure the merged table goes to `rebuttal/r3_merge_FAILED.csv`.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))
sys.path.insert(0, str(REPO_ROOT / "code" / "scripts"))

from p3fcl import provenance  # noqa: E402
from p3fcl.rebuttal.common import manifest  # noqa: E402
from rb_r3_client_scaling import COMBOS, WORK  # noqa: E402

METRICS = ["final_avg_acc", "bwt", "avg_incremental_acc"]


def _paper_row(paper, ds, method, unit, eps, n, seed):
    hits = [r for r in paper[ds] if r["method"] == method and int(r["seed"]) == seed and int(r["n_clients"]) == n
            and r["unit"] == unit and (method != "m9_contractive" or (float(r["gamma"]) == 1.0 and float(r["eps"]) == eps))]
    return hits[0] if len(hits) == 1 else None


def main() -> int:
    man = manifest({"seed": 0, "purpose": "rebuttal R3 merge", "protocol": "rebuttal/R3_PROTOCOL.md"})
    paper = {}
    for ds in {d for d, _ in COMBOS}:
        with open(REPO_ROOT / "results" / f"m9_sweep_{ds}.csv", newline="") as f:
            paper[ds] = list(csv.DictReader(f))
    rows, sel = [], []
    for ds, n in COMBOS:
        with open(WORK / f"part_{ds}_n{n}.csv", newline="") as f:
            rows += list(csv.DictReader(f))
        with open(WORK / f"sel_{ds}_n{n}.csv", newline="") as f:
            sel += list(csv.DictReader(f))

    failures, n_checked = [], 0
    for r in rows:
        n, eps, seed, ds = int(r["n_clients"]), float(r["eps"]), int(r["seed"]), r["dataset"]
        check = (r["calibration"] == "old_fixed_C" and (n == 10 or (ds == "cifar100" and n in (50, 100)))) or \
                (r["calibration"] in ("u1_reference", "m8_reference") and n == 10)
        r["paper_check"] = "n/a"
        if not check:
            continue
        p = _paper_row(paper, ds, r["method"], r["unit"], eps, n, seed)
        ok = p is not None and all(float(p[k]) == float(r[k]) for k in METRICS)
        r["paper_check"] = "matches_paper" if ok else "MISMATCH"
        n_checked += 1
        if not ok:
            failures.append(f"{ds} {r['calibration']} {r['unit']} n={n} eps={eps} seed={seed}")

    dest = REPO_ROOT / "results" / "rb_client_scaling.csv" if not failures else REPO_ROOT / "rebuttal" / "r3_merge_FAILED.csv"
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for msg in failures:
        print("R3 MERGE FAILURE:", msg)
    print(f"R3 MERGE rows={len(rows)} paper_checks={n_checked} failures={len(failures)} -> {dest}")
    if failures:
        return 1
    dest_sel = REPO_ROOT / "results" / "rb_client_scaling_selection.csv"
    with open(dest_sel, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sel[0]))
        w.writeheader()
        w.writerows(sel)
    provenance.finalize(man, [dest, dest_sel])
    return 0


if __name__ == "__main__":
    sys.exit(main())
