#!/usr/bin/env python3
"""FX5 (`08_FIX_PLAN.md` §8 Outputs): TAB08, dataset x eps x unit -> final average accuracy
(mean and 95% CI over seeds), from the already-computed `results/m9_sweep.csv` (CLAUDE.md
non-negotiable #3 -- no recomputation). gamma=1.0, n_clients=10 (main grid) only.

**Certified eps at T=10/T=50 for U1/U2/U3(W=3)/U4, taken from the FX1 accountant, is NOT included in
this version.** That column needs each (dataset, eps, unit)'s actual M9 ledger re-run through
`dp.accountant.account`/`extrapolate_lifelong` -- `run_m9_sweep.py` did not retain ledgers (only
`final_avg_acc`/`bwt`/`avg_incremental_acc` from `sim.run`'s result dict), so this is real, separate
follow-on work, not a trivial addition. Flagged here rather than fabricated or silently dropped.

Files: `tables/tab08_dp_utility.csv` and `.tex`.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl.plotting import latex_escape  # noqa: E402


def _mean_ci(values: list) -> tuple:
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, float("nan"), float("nan")
    sem = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    if sem == 0:
        return mean, mean, mean
    lo, hi = stats.t.interval(0.95, df=len(values) - 1, loc=mean, scale=sem)
    return mean, float(lo), float(hi)


def main() -> int:
    sweep_csv = REPO_ROOT / "results" / "m9_sweep.csv"
    with open(sweep_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["method"] == "m9_contractive" and r["gamma"] == "1.0" and r["n_clients"] == "10"]

    groups: dict = {}
    for r in rows:
        key = (r["dataset"], r["unit"], float(r["eps"]))
        groups.setdefault(key, []).append(float(r["final_avg_acc"]))

    out_rows = []
    for (dataset, unit, eps), accs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        mean, lo, hi = _mean_ci(accs)
        out_rows.append({
            "dataset": dataset, "unit": unit, "eps": eps, "n_seeds": len(accs),
            "final_avg_acc_mean": mean, "final_avg_acc_ci_lo": lo, "final_avg_acc_ci_hi": hi,
            "eps_certified_T10_U1": "", "eps_certified_T10_U2": "", "eps_certified_T10_U3": "", "eps_certified_T10_U4": "",
            "eps_certified_T50_U1": "", "eps_certified_T50_U2": "", "eps_certified_T50_U3": "", "eps_certified_T50_U4": "",
        })

    out_dir = REPO_ROOT / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "tab08_dp_utility.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    tex_lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
                 r"Dataset & Unit & $\varepsilon$ & N & Final avg acc (mean) & 95\% CI \\", r"\midrule"]
    for r in out_rows:
        eps_str = "inf" if np.isinf(r["eps"]) else f"{r['eps']:g}"
        ci_str = "n/a" if np.isnan(r["final_avg_acc_ci_lo"]) else f"[{r['final_avg_acc_ci_lo']:.3f}, {r['final_avg_acc_ci_hi']:.3f}]"
        tex_lines.append(
            f"{latex_escape(r['dataset'])} & {latex_escape(r['unit'])} & {eps_str} & {r['n_seeds']} & "
            f"{r['final_avg_acc_mean']:.3f} & {latex_escape(ci_str)} \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "tab08_dp_utility.tex").write_text("\n".join(tex_lines) + "\n")

    print(f"wrote {len(out_rows)} rows to {csv_path} and tab08_dp_utility.tex")
    print("NOTE: eps_certified_T10/T50_* columns are blank -- FX1-accountant certified eps needs M9 ledgers re-run, not done this pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
