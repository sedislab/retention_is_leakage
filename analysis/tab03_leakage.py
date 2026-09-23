#!/usr/bin/env python3
"""TAB03 — main leakage table: method x dataset x view x {TPR@1%FPR, TPR@0.1%FPR, AUC} at elapsed=0
and elapsed=max, with CIs. Reads `results/fig01_decoupling.csv`
(`code/scripts/build_fig01_decoupling.py`'s output) and writes `tables/tab03_leakage.{csv,tex}` -- no
computation here (CLAUDE.md non-negotiable #3).

Grouped on `(dataset, method, view)`, not just `(dataset, method)` -- the same real bug FIG01 v2 had
(FX4h gave M4/M8 up to 3 distinct scoring views per (dataset, method); grouping without `view` picks
one arbitrarily and silently drops the other two's rows).
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl.plotting import latex_escape  # noqa: E402


def _fmt(mean: str, lo: str, hi: str) -> str:
    return f"{float(mean):.3f} [{float(lo):.3f}, {float(hi):.3f}]"


def main() -> int:
    with open(REPO_ROOT / "results" / "fig01_decoupling.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    by_method: dict = defaultdict(list)
    for r in rows:
        by_method[(r["dataset"], r["method"], r["view"])].append(r)

    table_rows = []
    for (dataset, method, view), method_rows in sorted(by_method.items()):
        method_rows.sort(key=lambda r: int(r["elapsed"]))
        for tag, r in (("elapsed=0", method_rows[0]), ("elapsed=max", method_rows[-1])):
            table_rows.append({
                "dataset": dataset, "method": method, "view": view, "condition": tag,
                "elapsed": r["elapsed"],
                "tpr1": _fmt(r["tpr1_mean"], r["tpr1_ci_lo"], r["tpr1_ci_hi"]),
                "tpr01": _fmt(r["tpr01_mean"], r["tpr01_ci_lo"], r["tpr01_ci_hi"]),
                "auc": _fmt(r["auc_mean"], r["auc_ci_lo"], r["auc_ci_hi"]),
            })

    out_dir = REPO_ROOT / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "tab03_leakage.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        w.writeheader()
        w.writerows(table_rows)

    tex_lines = [
        r"\begin{tabular}{lllrlll}",
        r"\toprule",
        r"Dataset & Method & View & Elapsed & TPR@1\%FPR & TPR@0.1\%FPR & AUC \\",
        r"\midrule",
    ]
    for r in table_rows:
        tex_lines.append(
            f"{latex_escape(r['dataset'])} & {latex_escape(r['method'])} & {latex_escape(r['view'])} & "
            f"{r['elapsed']} & {r['tpr1']} & {r['tpr01']} & {r['auc']} \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "tab03_leakage.tex").write_text("\n".join(tex_lines) + "\n")

    print(f"wrote {len(table_rows)} rows to {csv_path} and tab03_leakage.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
