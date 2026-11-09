#!/usr/bin/env python3
"""R1 merge (`rebuttal/R1_PROTOCOL.md`): `shadows_rb/_lira/summary_*.csv` plus the paper's margin
summaries become `results/rb_release_audit.csv`. PBS only.

Hard checks, which fail the job if violated:
- `margin_repro` equals the paper's `a1_lira_fixedk_summary_<ds>_<method>_full.csv` (tpr1, tpr01,
  auc and their CIs, every ablation and elapsed), so the pipeline is paired with the paper;
- release views have identical tpr1/tpr01/auc at e = 0 and e = 6 (a constant release).
`prediction_holds` (fixed in advance): release view, trajectory, e = 0: tpr1 >= paper margin tpr1
at e = 6 (trajectory). Elsewhere `n/a`.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))
sys.path.insert(0, str(REPO_ROOT / "code" / "scripts"))

from p3fcl import provenance  # noqa: E402
from p3fcl.rebuttal import release_audit as ra  # noqa: E402
from p3fcl.rebuttal.common import manifest  # noqa: E402
from rb_r1_score import COMBOS, LIRA_DIR  # noqa: E402

OUT = REPO_ROOT / "results" / "rb_release_audit.csv"
STATS = ["tpr1", "tpr1_ci_lo", "tpr1_ci_hi", "tpr01", "tpr01_ci_lo", "tpr01_ci_hi", "auc", "auc_ci_lo", "auc_ci_hi"]


def _f(x):
    return float("nan") if x in ("", None, "None") else float(x)


def _same(a, b):
    a, b = _f(a), _f(b)
    return (math.isnan(a) and math.isnan(b)) or a == b


def main() -> int:
    man = manifest({"seed": 0, "purpose": "rebuttal R1 merge", "protocol": "rebuttal/R1_PROTOCOL.md"})
    paper, paper_src = {}, {}
    for ds, m, _ in COMBOS:
        src = REPO_ROOT / "results" / f"a1_lira_fixedk_summary_{ds}_{m}_full.csv"
        paper_src[(ds, m)] = str(src.relative_to(REPO_ROOT))
        with open(src, newline="") as f:
            for r in csv.DictReader(f):
                paper[(ds, m, r["ablation"], int(r["elapsed"]))] = r

    def nan_frac(ds, m, view):
        """Fraction of (shadow, target) scores that are NaN at the target's own task, over all 500
        targets and all seeds (from `index.csv`). n/a for views without a release score of their own."""
        if view not in ra.RELEASE_VIEWS[m]:
            return "n/a"
        num = den = 0
        for s in (0, 1, 2):
            with open(REPO_ROOT / "shadows_rb" / ds / m / f"seed{s}" / "index.csv", newline="") as f:
                idx = list(csv.DictReader(f))
            num += sum(int(r[f"nan_at_k_{view}"]) for r in idx)
            den += 500 * len(idx)
        return num / den

    out_rows, failures = [], []
    for ds, m, view in COMBOS:
        nf = nan_frac(ds, m, view)
        with open(LIRA_DIR / f"summary_{ds}_{m}_{view}.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        by = {(r["ablation"], int(r["elapsed"])): r for r in rows}
        for r in rows:
            ab, e = r["ablation"], int(r["elapsed"])
            p = paper[(ds, m, ab, e)]
            row = {"dataset": ds, "method": m, "view": view, "ablation": ab, "elapsed": e,
                   **{k: r[k] for k in STATS}, "n_seeds": r["n_seeds"],
                   "paper_margin_tpr1_e0": paper[(ds, m, ab, 0)]["tpr1"],
                   "paper_margin_tpr1_e6": paper[(ds, m, ab, 6)]["tpr1"],
                   "paper_source": paper_src[(ds, m)], "prediction_holds": "n/a", "check": "n/a",
                   "nan_frac": nf}
            if view == "margin_repro":
                ok = all(_same(r[k], p[k]) for k in STATS)
                row["check"] = "matches_paper" if ok else "MISMATCH"
                if not ok:
                    failures.append(f"margin_repro {ds}/{m}/{ab}/e={e} differs from {paper_src[(ds, m)]}")
            elif view in ra.RELEASE_VIEWS[m]:
                const = all(_same(r[k], by[(ab, 0)][k]) for k in ("tpr1", "tpr01", "auc"))
                row["check"] = "constant_in_e" if const else "NOT_CONSTANT"
                if not const:
                    failures.append(f"release {ds}/{m}/{view}/{ab}/e={e} differs from e=0")
                if ab == "trajectory" and e == 0:
                    row["prediction_holds"] = int(_f(r["tpr1"]) >= _f(paper[(ds, m, "trajectory", 6)]["tpr1"]))
            out_rows.append(row)

    dest = OUT if not failures else REPO_ROOT / "rebuttal" / "r1_merge_FAILED.csv"
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    for r in out_rows:
        if r["ablation"] == "trajectory" and r["elapsed"] in (0, 6):
            print(f"R1 {r['dataset']:10s} {r['method']:9s} {r['view']:13s} e={r['elapsed']} "
                  f"tpr1={_f(r['tpr1']):.4f} [{_f(r['tpr1_ci_lo']):.4f},{_f(r['tpr1_ci_hi']):.4f}] "
                  f"tpr01={_f(r['tpr01']):.4f} auc={_f(r['auc']):.4f} paper_e6={_f(r['paper_margin_tpr1_e6']):.4f} "
                  f"holds={r['prediction_holds']} check={r['check']}")
    for msg in failures:
        print("R1 MERGE FAILURE:", msg)
    print(f"R1 MERGE rows={len(out_rows)} failures={len(failures)} -> {dest}")
    if failures:
        return 1
    provenance.finalize(man, [OUT])
    return 0


if __name__ == "__main__":
    sys.exit(main())
