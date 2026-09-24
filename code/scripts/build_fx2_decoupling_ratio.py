#!/usr/bin/env python3
"""FX2 (`08_FIX_PLAN.md` §7d): `results/decoupling_ratio.csv` -- the point estimate rho = h_leak /
h_acc and its `ratio_type`, read from `results/fig02_halflife.csv` (both scripts that populate it,
`build_fx2_summary.py` for the leak rows and `build_fx2_accuracy_summary.py` for the accuracy row,
must have already run for the given (dataset, method, view)). No recomputation here (CLAUDE.md
non-negotiable #3).

Primary leak quantity is `leak_tpr1` (CLAUDE.md non-negotiable #5: TPR@1%FPR/0.1%FPR are primary,
never AUC alone), `ablation="trajectory"` (H3's transcript arm, the paper's headline claim).

**`ratio_ci_lo`/`ratio_ci_hi` are left blank, not fabricated.** A rigorous CI on the ratio needs a
JOINT bootstrap (the same resampled-seed draw feeding both the leak and accuracy halflife in each
replicate) -- `build_fx2_summary.py`'s leak bootstrap resamples shadow targets and
`build_fx2_accuracy_summary.py`'s accuracy bootstrap resamples the k in K, entirely different data and
resampling loops. Pairing their already-computed replicate arrays positionally would only be a valid
joint bootstrap if the two loops' random draws stayed lock-step through both the outer seed resample
AND the inner (target vs. k) resample, which they do not (different amounts of randomness consumed
per replicate). Flagged here as a real, not-yet-built piece rather than silently approximating it.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import provenance  # noqa: E402
from p3fcl.halflife import decoupling_ratio  # noqa: E402

PRIMARY_LEAK_QUANTITY = "leak_tpr1"
PRIMARY_ABLATION = "trajectory"


def _hl_dict(row: dict) -> dict:
    return {
        "halflife": float(row["halflife"]) if row["halflife"] not in ("", None) else None,
        "halflife_int": int(row["halflife_int"]) if row["halflife_int"] not in ("", None) else None,
        "status": row["status"],
    }


def build_decoupling_ratio(dataset: str, method: str, family: str, view: str, by_construction: bool = False) -> dict:
    halflife_csv = REPO_ROOT / "results" / "fig02_halflife.csv"
    with open(halflife_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    def find(quantity: str, ablation: str, match_view: bool):
        matches = [
            r for r in rows if r["dataset"] == dataset and r["method"] == method and r["family"] == family
            and (r["view"] == view if match_view else True) and r["quantity"] == quantity and r["ablation"] == ablation
        ]
        if not matches:
            raise ValueError(f"no fig02_halflife.csv row for {(dataset, method, family, view, quantity, ablation)}")
        return matches[0]

    leak_row = find(PRIMARY_LEAK_QUANTITY, PRIMARY_ABLATION, match_view=True)
    # Accuracy is not scored per-view (there is one real held-out test accuracy per (dataset,method),
    # independent of which shadow-scoring view is being used to measure leakage) -- built by
    # `build_fx2_accuracy_summary.py` with view="full" as a fixed convention, not a claim that
    # accuracy itself varies by view.
    acc_row = find("acc", "n/a", match_view=False)

    h_leak, h_acc = _hl_dict(leak_row), _hl_dict(acc_row)
    result = decoupling_ratio(h_leak, h_acc, by_construction=by_construction)

    return {
        "dataset": dataset, "method": method, "family": family, "view": view,
        "h_acc": h_acc["halflife"], "h_leak": h_leak["halflife"],
        "ratio": result["ratio"], "ratio_ci_lo": "", "ratio_ci_hi": "",
        "ratio_type": result["ratio_type"],
    }


def main() -> int:
    if len(sys.argv) < 4:
        print("usage: build_fx2_decoupling_ratio.py <dataset> <method> <family> [view] [--by-construction]", file=sys.stderr)
        return 2
    dataset, method, family = sys.argv[1], sys.argv[2], sys.argv[3]
    rest = sys.argv[4:]
    by_construction = "--by-construction" in rest
    rest = [a for a in rest if a != "--by-construction"]
    view = rest[0] if rest else "full"

    row = build_decoupling_ratio(dataset, method, family, view, by_construction=by_construction)

    out_csv = REPO_ROOT / "results" / f"decoupling_ratio_{dataset}_{method}_{view}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"wrote decoupling_ratio row for {dataset}/{method}/{view}: {row}")

    config = {
        "seed": 0, "purpose": "FX2 decoupling ratio (claim C1, H2)",
        "dataset": dataset, "method": method, "view": view,
    }
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
