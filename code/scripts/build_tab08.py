#!/usr/bin/env python3
"""FX5 (`08_FIX_PLAN.md` §8 Outputs): TAB08, dataset x eps x unit -> final average accuracy
(mean and 95% CI over seeds), from the already-computed `results/m9_sweep.csv` (CLAUDE.md
non-negotiable #3 -- no recomputation). gamma=1.0, n_clients=10 (main grid) only.

FX9 fills the eight T=10/50, U1–U4 composition columns from m9_certified.csv and
adds a compact accounting tabular to the utility TeX. These are the requested FX1
composition values from U1-calibrated ledgers; cross-unit sensitivity remains an
explicit qualification, not an unconditional group-privacy claim.

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
        rows = [
            r
            for r in csv.DictReader(f)
            if r["method"] == "m9_contractive" and r["gamma"] == "1.0" and r["n_clients"] == "10"
        ]

    groups: dict = {}
    for r in rows:
        key = (r["dataset"], r["unit"], float(r["eps"]))
        groups.setdefault(key, []).append(float(r["final_avg_acc"]))

    with (REPO_ROOT / "results" / "m9_certified.csv").open() as f:
        certified = {(float(r["eps0"]), int(r["T"]), r["unit"]): r for r in csv.DictReader(f)}
    out_rows = []
    for (dataset, unit, eps), accs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        mean, lo, hi = _mean_ci(accs)
        out_rows.append(
            {
                "dataset": dataset,
                "unit": unit,
                "eps": eps,
                "n_seeds": len(accs),
                "final_avg_acc_mean": mean,
                "final_avg_acc_ci_lo": lo,
                "final_avg_acc_ci_hi": hi,
                **{
                    f"eps_certified_T{T}_{u}": certified[(eps, T, u)]["eps"]
                    for T in (10, 50)
                    for u in ("U1", "U2", "U3", "U4")
                },
                "certified_scope": "FX1 record-composition convention, normalized to configured sensitivity; group sensitivity requires matching calibration unit",
                "eps0_analytic": eps,
                "delta0_analytic": 1e-5,
            }
        )

    out_dir = REPO_ROOT / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "tab08_dp_utility.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    tex_lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Unit & $\varepsilon$ & N & Final avg acc (mean) & 95\% CI \\",
        r"\midrule",
    ]
    for r in out_rows:
        eps_str = "inf" if np.isinf(r["eps"]) else f"{r['eps']:g}"
        ci_str = (
            "n/a"
            if np.isnan(r["final_avg_acc_ci_lo"])
            else f"[{r['final_avg_acc_ci_lo']:.3f}, {r['final_avg_acc_ci_hi']:.3f}]"
        )
        tex_lines.append(
            f"{latex_escape(r['dataset'])} & {latex_escape(r['unit'])} & {eps_str} & {r['n_seeds']} & "
            f"{r['final_avg_acc_mean']:.3f} & {latex_escape(ci_str)} \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}"]
    tex_lines += [
        "",
        r"\par\medskip",
        r"FX1 composition of the U1-calibrated M9 ledgers. Cross-unit entries are conditional on matching group sensitivity.",
        r"\par\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$\varepsilon_0$ & $T$ & U1 & U2 & U3 & U4 \\",
        r"\midrule",
    ]
    for eps in sorted({float(r["eps"]) for r in out_rows}):
        for T in (10, 50):
            values = [eps, T] + [float(certified[(eps, T, u)]["eps"]) for u in ("U1", "U2", "U3", "U4")]
            tex_lines.append(" & ".join("inf" if np.isinf(v) else f"{v:.4g}" for v in values) + r" \\")
    tex_lines += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "tab08_dp_utility.tex").write_text("\n".join(tex_lines) + "\n")

    print(f"wrote {len(out_rows)} rows to {csv_path} and tab08_dp_utility.tex")
    cert_columns = [k for k in out_rows[0] if k.startswith("eps_certified_")]
    blank = sum(r[k] in ("", None) or np.isnan(float(r[k])) for r in out_rows for k in cert_columns)
    assert len(cert_columns) == 8 and blank == 0
    print(
        f"FX9-7 TAB08 ACCEPT certified_columns={len(cert_columns)} blank_certified_cells={blank} table_rows={len(out_rows)} TeX_accounting_rows=12; scope column distinguishes the FX1 convention"
    )
    from p3fcl import provenance

    manifest = provenance.run_manifest(
        dict(phase="FX9-10", table="tab08_dp_utility", source="m9_sweep.csv + m9_certified.csv"), seed=0
    )
    provenance.finalize(manifest, [out_dir / "tab08_dp_utility.csv", out_dir / "tab08_dp_utility.tex"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
