#!/usr/bin/env python3
"""TAB02 — units of privacy in FCL (RESEARCH_PLAN.md §2.3, claim C3). Purely descriptive: reads
`p3fcl.units.Unit`/`neighbouring()` directly (the single source of truth for these definitions, so the
paper's wording and the code's wording cannot drift apart) rather than hand-typing prose that could
diverge from what `dp/accountant.py` actually implements. No new experiments — this table only needed
writing, not computing.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl.plotting import latex_escape  # noqa: E402
from p3fcl.units import Unit, neighbouring  # noqa: E402

# Which composition regime `dp/accountant.py::account()` actually routes each unit to, and which
# claim/hypothesis it is load-bearing for -- both taken directly from that function's own routing
# logic and from agents/OPEN_QUESTIONS.md, not invented for this table.
_REGIME = {
    Unit.EXAMPLE: "Sequential composition over all releases that touch the example (no special-casing)",
    Unit.TASK: "Parallel composition IF the ledger is task-disjoint (flat eps(T)); else sequential",
    Unit.CLIENT_BOUNDED: "Sequential composition over the fixed horizon T",
    Unit.CLIENT_LIFELONG: "Sequential composition over the *entire* observed stream, unconditionally",
    Unit.INDIVIDUAL: "Sequential composition over a person's recurring participation under a renewal model",
}
_RELEVANCE = {
    Unit.EXAMPLE: "Baseline unit most DP-SGD-in-FL papers use implicitly",
    Unit.TASK: "The unit under which claim C3 / theorem T1-fwd holds: task-disjoint retention -> flat eps(T)",
    Unit.CLIENT_BOUNDED: "Standard cross-silo target when a horizon T is fixed in advance",
    Unit.CLIENT_LIFELONG: "The unit under which H1 shows every accountant we tested diverges (theorem T1-conv)",
    Unit.INDIVIDUAL: "The unit theorem T2 targets (person-level guarantee under churn), aspirational",
}


def main() -> int:
    rows = []
    for unit in Unit:
        rows.append({
            "unit_id": unit.value,
            "name": unit.name,
            "neighbouring_relation": neighbouring(unit),
            "composition_regime": _REGIME[unit],
            "relevance": _RELEVANCE[unit],
        })

    out_dir = REPO_ROOT / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "tab02_units.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    tex_lines = [r"\begin{tabular}{lp{5cm}p{4cm}p{4cm}}", r"\toprule",
                 r"Unit & Neighbouring relation & Composition regime & Relevance \\", r"\midrule"]
    for r in rows:
        tex_lines.append(
            f"{latex_escape(r['unit_id'])} & {latex_escape(r['neighbouring_relation'])} & "
            f"{latex_escape(r['composition_regime'])} & {latex_escape(r['relevance'])} \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "tab02_units.tex").write_text("\n".join(tex_lines) + "\n")

    print(f"wrote {len(rows)} rows to {csv_path} and tab02_units.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
