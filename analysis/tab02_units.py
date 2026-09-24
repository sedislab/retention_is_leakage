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

# FX1 (08_FIX_PLAN.md §6) rewrote account() as one unified rule for every unit: eps(T) = max over
# unit instances u of eps_gaussian_composed(sigma, m_T(u), delta), where m_T(u) counts releases that
# touched u's data by horizon T. No unit gets a hard-coded routing branch any more -- "parallel
# composition" for a task-disjoint ledger under U2 is just what falls out when m_T stays at 1 forever
# because no later release ever re-touches an already-released shard. Text below describes the
# resulting behavior, not a branch in the code.
_REGIME = {
    Unit.EXAMPLE: "m_T(id) = releases touching that example by T, maximized over ids; flat iff no example is ever re-touched (U5 routes here too, see neighbouring(U5))",
    Unit.TASK: "m_T(client,task) = releases touching that shard by T; flat (=1) iff task-disjoint, else grows with every re-touch (replay/regularizer)",
    Unit.CLIENT_BOUNDED: "m_T over a fixed-width rolling window (default 3 tasks) of one client's data, worst case over every window start; plateaus at the window width if no replay reaches back that far",
    Unit.CLIENT_LIFELONG: "m_T over one client's ENTIRE history so far (tasks 0..T-1) -- grows at least linearly in T even with zero replay, since every task's own release still counts",
    Unit.INDIVIDUAL: "Identical to U1 under this project's one-person-one-example renewal model; not accounted separately",
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
    from p3fcl import provenance
    manifest = provenance.run_manifest(dict(phase="FX9-10", table="tab02_units", source="code definitions"), seed=0)
    provenance.finalize(manifest, [out_dir/"tab02_units.csv", out_dir/"tab02_units.tex"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
