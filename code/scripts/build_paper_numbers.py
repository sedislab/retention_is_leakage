#!/usr/bin/env python3
"""FX9 exact cell transcription: JSON row filters plus explicit value/CI columns."""

import json
import math

from fx9_io import ROOT, read, write

# Every quoted value is a source cell. Means/intervals are computed by producers,
# never inside this transcription layer. Identity columns uniquely select a row.
SOURCES = [
    (
        "results/fx9_m9_gap_summary.csv",
        ["dataset", "unit", "eps"],
        [("gap_m9_minus_m8", "gap_m9_minus_m8_ci_lo", "gap_m9_minus_m8_ci_hi")],
    ),
    (
        "results/fx9_gate_comparison.csv",
        ["dataset", "method"],
        [(k, "", "") for k in ["last_before", "last_after", "bwt_before", "bwt_after"]],
    ),
    (
        "results/fx9_dose_summary.csv",
        ["dataset", "method", "knob_value"],
        [(k, k + "_ci_lo", k + "_ci_hi") for k in ["forgetting", "tpr1", "final_acc"]],
    ),
    (
        "results/retention_curves.csv",
        ["dataset", "method", "view", "quantity", "elapsed"],
        [("raw", "raw_ci_lo", "raw_ci_hi"), ("norm", "norm_ci_lo", "norm_ci_hi")],
    ),
    (
        "results/fig02_retention_at_horizon.csv",
        ["dataset", "method", "view"],
        [
            ("acc_norm", "acc_norm_ci_lo", "acc_norm_ci_hi"),
            ("leak_norm", "leak_norm_ci_lo", "leak_norm_ci_hi"),
        ],
    ),
    (
        "results/fig02_halflife.csv",
        ["dataset", "method", "view", "quantity", "ablation"],
        [("halflife", "ci_lo", "ci_hi")],
    ),
    (
        "results/decoupling_ratio.csv",
        ["dataset", "method", "view"],
        [("ratio", "ratio_ci_lo", "ratio_ci_hi")],
    ),
    (
        "results/m9_certified.csv",
        ["dataset", "calibration_unit", "eps0", "T", "unit"],
        [("eps", "", ""), ("eps0_analytic", "", ""), ("z", "", "")],
    ),
    (
        "results/m9_audit_summary.csv",
        ["dataset", "unit", "eps"],
        [("eps_lb", "", ""), ("tpr1", "", "tpr1_ci_hi")],
    ),
    (
        "results/fx9_gate_summary.csv",
        ["dataset", "method"],
        [(k, k + "_ci_lo", k + "_ci_hi") for k in ["final_avg_acc", "last_task_acc", "bwt"]]
        + [("gate_pass_numeric", "", "")],
    ),
    (
        "results/fx9_utility_summary.csv",
        ["dataset", "method"],
        [(k, k + "_ci_lo", k + "_ci_hi") for k in ["final_avg_acc", "bwt", "avg_incremental_acc"]],
    ),
    (
        "results/fx9_gate.csv",
        ["dataset", "method", "seed"],
        [("final_avg_acc", "", ""), ("last_task_acc", "", ""), ("bwt", "", ""), ("replay_weight", "", "")],
    ),
    (
        "results/tab07_reproduction_gap.csv",
        ["method", "source_paper"],
        [("our_reimpl_acc_mean", "", ""), ("published_acc", "", ""), ("gap_ours_minus_published", "", "")],
    ),
    (
        "tables/tab08_dp_utility.csv",
        ["dataset", "unit", "eps"],
        [("final_avg_acc_mean", "final_avg_acc_ci_lo", "final_avg_acc_ci_hi")],
    ),
    (
        "results/fig13_gram_inversion_summary.csv",
        ["dataset", "n_per_class", "ref_quality"],
        [("median_cos", "ci_lo", "ci_hi")],
    ),
    (
        "results/fig18_natural_federation.csv",
        ["partition", "elapsed"],
        [("tpr1", "ci_lo", "ci_hi"), ("acc", "", "")],
    ),
    (
        "results/fig03_dose_response.csv",
        ["dataset", "method", "knob_value", "seed"],
        [("tpr1", "ci_lo", "ci_hi"), ("retention_bwt", "", ""), ("final_acc", "", "")],
    ),
    ("results/fig05_eps_of_T.csv", ["dataset", "method", "unit", "eps0", "T"], [("eps", "", "")]),
    ("results/fig17_seed_variance.csv", ["dataset", "method", "seed", "metric"], [("value", "", "")]),
    ("results/a6_property_inference.csv", ["elapsed"], [("balanced_accuracy", "", "")]),
]


def transcribe(source, ids, fields):
    output = []
    rows = read(ROOT / source)
    for row in rows:
        if source.endswith("fig05_eps_of_T.csv") and row["T"] not in ["1", "10", "50", "1000"]:
            continue
        identity = {key: row.get(key, "") for key in ids}
        note = "; ".join(
            f"{k}={row[k]}"
            for k in [
                "status",
                "ratio_type",
                "ci_interpretation",
                "scope",
                "gate_pass",
                "n_seeds",
                "n_trials",
                "audit_scope",
                "interval_scope",
            ]
            if k in row
        )
        for field, lo, hi in fields:
            if field == "halflife" and row.get("status") == "censored":
                lo = hi = ""  # historical [E,E] endpoints are censor markers, not a finite CI
            value = row[field]
            if value == "" or math.isnan(float(value)):
                continue
            output.append(
                dict(
                    key=source.split("/")[-1].removesuffix(".csv")
                    + "__"
                    + "__".join(identity.values())
                    + "__"
                    + field,
                    value=value,
                    ci_lo=row.get(lo, ""),
                    ci_hi=row.get(hi, ""),
                    unit=field,
                    source_csv=source,
                    row_filter=json.dumps(identity, sort_keys=True),
                    value_column=field,
                    ci_lo_column=lo,
                    ci_hi_column=hi,
                    note=note,
                )
            )
    return output


def main():
    rows = [r for source, ids, fields in SOURCES for r in transcribe(source, ids, fields)]
    assert len({r["key"] for r in rows}) == len(rows), "duplicate paper number keys"
    write(
        ROOT / "results/paper_numbers.csv",
        rows,
        dict(sources=[s[0] for s in SOURCES], operation="exact source-cell transcription"),
    )
    print(f"FX9-10 PAPER NUMBERS ACCEPT rows={len(rows)} computed_in_transcriber=0", flush=True)


if __name__ == "__main__":
    main()
