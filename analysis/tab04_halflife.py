#!/usr/bin/env python3
"""TAB04 — half-life estimates per family, with CI, R^2, censoring flag. Reads
`results/fig02_halflife.csv` (`scripts/build_fig02.py`'s output -- same underlying data as FIG02, a
tabular presentation of it, not a separate computation) and writes `tables/tab04_halflife.{csv,tex}`.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl.plotting import latex_escape  # noqa: E402


def main() -> int:
    with open(REPO_ROOT / "results" / "fig02_halflife.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    out_dir = REPO_ROOT / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "tab04_halflife.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    tex_lines = [
        r"\begin{tabular}{llllrrl}",
        r"\toprule",
        r"Method & Family & Quantity & Half-life & CI & $R^2$ & Censored \\",
        r"\midrule",
    ]
    for r in rows:
        censored = r["censored"] in ("True", "true", "1")
        halflife = "no decay ($>$ " + r["horizon_tasks"] + ")" if censored else f"{float(r['halflife']):.2f}"
        ci = "--" if censored else f"[{float(r['ci_lo']):.2f}, {float(r['ci_hi']):.2f}]"
        r2 = "--" if r["r2"] in ("nan", "") else f"{float(r['r2']):.3f}"
        tex_lines.append(
            f"{latex_escape(r['method'])} & {latex_escape(r['family'])} & "
            f"{latex_escape(r['quantity'])} & {halflife} & {ci} & {r2} & "
            f"{'yes' if censored else 'no'} \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "tab04_halflife.tex").write_text("\n".join(tex_lines) + "\n")

    print(f"wrote {len(rows)} rows to {csv_path} and tab04_halflife.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
