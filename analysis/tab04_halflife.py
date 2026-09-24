#!/usr/bin/env python3
"""TAB04 — half-life estimates per family, with CI and status. Reads `results/fig02_halflife.csv`
(FX2, `code/scripts/build_fx2_summary.py`/`build_fx2_accuracy_summary.py`'s output -- same underlying
data as FIG02, a tabular presentation of it, not a separate computation) and writes
`tables/tab04_halflife.{csv,tex}`.

FX2 rewrite note: `status` is one of `ok` / `censored` / `no_signal` (Definition 10's three-way
outcome, `08_FIX_PLAN.md` §7b/7c), not the old binary `censored` flag -- `no_signal` (the quantity's
own 95%-CI lower bound didn't clear its chance floor by the required margin) is a real, distinct
outcome from `censored` (real signal, but never decayed to half within the observed horizon) and must
not be collapsed into it. `halflife_expfit`/`r2` (the secondary exponential-fit columns Definition 10
allows for the appendix) are blank in the current data -- not built this pass, shown as `--`.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl.plotting import latex_escape  # noqa: E402

_STATUS_LABEL = {"ok": "ok", "censored": "censored ($>E$)", "no_signal": "no signal"}


def main() -> int:
    with open(REPO_ROOT / "results" / "fig02_halflife.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (r["dataset"], r["method"], r["view"], r["quantity"], r["ablation"]))

    out_dir = REPO_ROOT / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "tab04_halflife.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    tex_lines = [
        r"\begin{tabular}{lllllrrl}",
        r"\toprule",
        r"Dataset & Method & View & Quantity & Ablation & Half-life & CI & Status \\",
        r"\midrule",
    ]
    for r in rows:
        status = r["status"]
        if status == "ok":
            halflife = f"{float(r['halflife']):.2f}"
            ci = f"[{float(r['ci_lo']):.2f}, {float(r['ci_hi']):.2f}]" if r["ci_lo"] not in ("", "None") else "--"
        elif status == "censored":
            halflife = f"$>${float(r['halflife']):.0f}"
            ci = "--"
        else:  # no_signal
            halflife = "--"
            ci = "--"
        tex_lines.append(
            f"{latex_escape(r['dataset'])} & {latex_escape(r['method'])} & {latex_escape(r['view'])} & "
            f"{latex_escape(r['quantity'])} & {latex_escape(r['ablation'])} & {halflife} & {ci} & "
            f"{_STATUS_LABEL.get(status, status)} \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "tab04_halflife.tex").write_text("\n".join(tex_lines) + "\n")

    print(f"wrote {len(rows)} rows to {csv_path} and tab04_halflife.tex")
    from p3fcl import provenance
    manifest = provenance.run_manifest(dict(phase="FX9-10", table="tab04_halflife", source="fig02_halflife.csv"), seed=0)
    provenance.finalize(manifest, [out_dir/"tab04_halflife.csv", out_dir/"tab04_halflife.tex"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
