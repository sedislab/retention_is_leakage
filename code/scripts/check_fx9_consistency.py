#!/usr/bin/env python3
"""Mandatory FX9 acceptance: exact source lineage, numerical agreement, figure freshness."""

import json
import math

import numpy as np
from fx9_io import ROOT, current, read
from p3fcl.experiment import CHANGED_METHODS, DATASETS, METHODS
from p3fcl.figure_sources import FIGURE_SOURCES
from p3fcl.halflife import accuracy_curve, half_life


def equal(a, b, tol=1e-9):
    if a == b:
        return True
    try:
        a, b = float(a), float(b)
        return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= tol
    except (ValueError, TypeError):
        return False


def check_paper_rows(rows, root=ROOT):
    cache = {}
    for r in rows:
        source = r["source_csv"]
        if source not in cache:
            cache[source] = read(root / source)
        identity = json.loads(r["row_filter"])
        matches = [v for v in cache[source] if all(v.get(k, "") == value for k, value in identity.items())]
        assert len(matches) == 1, (r["key"], len(matches), identity)
        v = matches[0]
        for field, column_field in [
            ("value", "value_column"),
            ("ci_lo", "ci_lo_column"),
            ("ci_hi", "ci_hi_column"),
        ]:
            column = r[column_field]
            if column:
                assert equal(r[field], v[column], tol=0), (r["key"], field, r[field], v[column])


def exact_merge(shared, pattern, keys):
    wanted = {}
    paths = (
        (ROOT / "results").glob(pattern)
        if isinstance(pattern, str)
        else [ROOT / "results" / name for name in pattern]
    )
    for path in sorted(paths):
        rows = read(path)
        if any(r["method"] in CHANGED_METHODS for r in rows):
            current(path)
        for r in rows:
            k = tuple(r[c] for c in keys)
            assert k not in wanted, (shared, k)
            wanted[k] = r
    actual = read(ROOT / "results" / shared)
    assert len(actual) == len(wanted), (shared, len(actual), len(wanted))
    for r in actual:
        k = tuple(r[c] for c in keys)
        assert r == wanted[k], (shared, k)
    return len(actual)


def main():
    curves = read(ROOT / "results/retention_curves.csv")
    fig01 = read(ROOT / "results/fig01_decoupling.csv")
    halves = read(ROOT / "results/fig02_halflife.csv")
    utility = read(ROOT / "results/tab05_utility_baselines.csv")
    assert len(curves) == 693 and len(utility) == 105
    matrix_checks, diag_checks, errors = 0, 0, []
    for ds in DATASETS:
        matrix = read(ROOT / f"results/accuracy_matrix_{ds}.csv")
        for method in METHODS:
            rr = [r for r in matrix if r["method"] == method]
            point = accuracy_curve(rr)
            diagonal = np.mean(
                [float(r["acc"]) for r in rr if r["elapsed"] == "0" and int(r["task_k"]) in range(4)]
            )
            acc = [
                r for r in curves if r["dataset"] == ds and r["method"] == method and r["quantity"] == "acc"
            ]
            for r in acc:
                assert equal(r["raw"], point.raw[int(r["elapsed"])])
                assert equal(r["norm"], point.norm[int(r["elapsed"])])
                if r["elapsed"] == "0":
                    assert equal(r["raw"], diagonal) and float(r["norm"]) == 1
                    diag_checks += 1
            for r in fig01:
                if r["dataset"] == ds and r["method"] == method and r["elapsed"] == "0":
                    assert equal(r["acc_mean"], diagonal), r
            expected_h = half_life(dict(enumerate(point.norm)))
            for r in halves:
                if (
                    r["dataset"] == ds
                    and r["method"] == method
                    and r["quantity"] == "acc"
                    and r["status"] != "no_signal"
                ):
                    assert equal(r["halflife"], expected_h["halflife"]), r
            for seed in range(5):
                final = np.mean(
                    [float(r["acc"]) for r in rr if int(r["seed"]) == seed and r["task_T"] == "9"]
                )
                u = [
                    r
                    for r in utility
                    if r["dataset"] == ds
                    and r["method"] == method.split("_")[0].upper()
                    and int(r["seed"]) == seed
                ]
                assert len(u) == 1 and u[0]["status"] == "FX9_raw_t10"
                err = abs(final - float(u[0]["final_avg_acc"]))
                assert err < 1e-9, (ds, method, seed, err)
                errors.append(err)
                matrix_checks += 1
    print(
        f"FX9 CONSISTENCY accuracy_diagonals={diag_checks} utility_seed_rows={matrix_checks} utility_max_error={max(errors):.12g}"
    )
    nleak = exact_merge(
        "a1_lira_fixedk_summary.csv",
        "a1_lira_fixedk_summary_*.csv",
        ["dataset", "method", "view", "ablation", "elapsed"],
    )
    nhalf = exact_merge(
        "fig02_halflife.csv", "fig02_halflife_*.csv", ["dataset", "method", "view", "quantity", "ablation"]
    )
    # Changed-method curves and every current fixed-K source are current PBS products.
    for pattern in [
        "retention_acc_*.csv",
        "retention_leak_*.csv",
        "fx9_horizon_*.csv",
        "a1_lira_pertask_*.csv",
    ]:
        for path in (ROOT / "results").glob(pattern):
            rr = read(path)
            if any(r["method"] in CHANGED_METHODS for r in rr):
                current(path)
    curve_sources = {}
    for pattern in ["retention_acc_*.csv", "retention_leak_*.csv"]:
        for path in (ROOT / "results").glob(pattern):
            for r in read(path):
                key = tuple(r[k] for k in ["dataset", "method", "view", "quantity", "elapsed"])
                assert key not in curve_sources
                curve_sources[key] = r
    horizon = {
        (r["dataset"], r["method"], r["view"]): r
        for r in read(ROOT / "results/fig02_retention_at_horizon.csv")
    }
    assert len(horizon) == 33
    for r in curves:
        key = tuple(r[k] for k in ["dataset", "method", "view", "quantity", "elapsed"])
        source = dict(curve_sources[key])
        if r["elapsed"] == "6" and r["quantity"] in ["acc", "leak_tpr1"]:
            h = horizon[(r["dataset"], r["method"], r["view"])]
            prefix = "acc" if r["quantity"] == "acc" else "leak"
            assert equal(r["norm"], h[prefix + "_norm"])
            for bound in ["lo", "hi"]:
                source["norm_ci_" + bound] = h[prefix + "_norm_ci_" + bound]
        assert all(equal(r[k], source[k], tol=0) for k in r), key
    print("FX9 CONSISTENCY retention_rows_traced=693 horizon_rows_traced=33")
    views = read(ROOT / "results/fx3_views_summary.csv")
    summary = {
        (r["dataset"], r["method"], r["view"], r["elapsed"]): r
        for r in read(ROOT / "results/a1_lira_fixedk_summary.csv")
        if r["ablation"] == "trajectory"
    }
    for r in views:
        view = "full" if r["identical_by_construction"] == "True" else r["view"]
        source = summary[(r["dataset"], r["method"], view, r["elapsed"])]
        for field in ["auc", "tpr1", "tpr01", "tpr1_ci_lo", "tpr1_ci_hi"]:
            assert equal(r[field], source[field]), r
        if r["method"] == "m4_proto" and r["view"] == "global":
            for field in ["auc", "tpr1", "tpr01"]:
                assert equal(
                    r[field], summary[(r["dataset"], r["method"], "aggregate", r["elapsed"])][field], 1e-10
                ), r
    print(
        f"FX9 CONSISTENCY per_combo_leak_rows={nleak} per_combo_halflife_rows={nhalf} changed_sources_current=1 M4_global_equals_aggregate=1"
    )
    for kind in ["accuracy", "lira"]:
        assert (
            exact_merge(
                f"dose_response_{kind}.csv",
                f"dose_response_{kind}_*.csv",
                ["method", "knob_value", "seed"],
            )
            == 36
        )
    assert (
        exact_merge("fx9_gate.csv", [f"fx9_gate_{ds}.csv" for ds in DATASETS], ["dataset", "method", "seed"])
        == 45
    )
    assert exact_merge("fig16_roc.csv", "fig16_roc_*.csv", ["dataset", "method", "seed", "fpr"]) == 4221
    seed_rows = read(ROOT / "results/fig17_seed_variance.csv")
    assert len(seed_rows) == 69
    for r in seed_rows:
        path = ROOT / f"results/a1_lira_pertask_{r['dataset']}_{r['method']}_seed{r['seed']}_full.csv"
        if r["method"] in CHANGED_METHODS:
            current(path)
        matches = [
            s
            for s in read(path)
            if s["task_k"] == "pooled" and s["elapsed"] == "0" and s["ablation"] == "trajectory"
        ]
        assert len(matches) == 1 and equal(r["value"], matches[0]["tpr1"], tol=0)
    for name in [
        "retention_curves",
        "fig02_retention_at_horizon",
        "fig01_decoupling",
        "fx3_views_summary",
        "fig16_roc",
        "fig17_seed_variance",
        "fig03_dose_response",
        "fig04_semantic_vs_individual",
        "fx9_dose_summary",
        "fx9_gate_summary",
        "fx9_gate_comparison",
        "fx9_utility_summary",
        "tab05_utility_baselines",
        "tab07_reproduction_gap",
        "fig05_eps_of_T",
    ]:
        current(ROOT / f"results/{name}.csv")
    print("FX9 CONSISTENCY dose_rows_traced=72 gate_rows_traced=45 ROC_rows_traced=4221 seed_rows_traced=69")
    paper = read(ROOT / "results/paper_numbers.csv")
    check_paper_rows(paper)
    print(f"FX9 CONSISTENCY paper_numbers_exact_source_cells={len(paper)}")
    descriptions = {r["method_id"]: r for r in read(ROOT / "results/method_descriptions.csv")}
    assert len(descriptions) == 9 and "one FedAvg round per task" in descriptions["protocol"]["implemented"]
    assert descriptions["m5_hybrid_replay"]["released_families"] == "F1 (model delta)"
    assert "Count-weighted" in descriptions["m4_proto"]["retention_mechanism"]
    for method in ["m1_glfc", "m2_target", "m5_hybrid_replay"]:
        assert "CE_mean(current)" in descriptions[method]["implemented"]
    certificates = {(float(r["eps0"]), r["T"], r["unit"]): r for r in read(ROOT / "results/m9_certified.csv")}
    tab08 = read(ROOT / "tables/tab08_dp_utility.csv")
    assert len(tab08) == 36
    for r in tab08:
        for T in ["10", "50"]:
            for unit in ["U1", "U2", "U3", "U4"]:
                assert equal(
                    r[f"eps_certified_T{T}_{unit}"], certificates[(float(r["eps"]), T, unit)]["eps"], tol=0
                )
    audited = {
        (r["dataset"], r["unit"], float(r["eps"])): r for r in read(ROOT / "results/m9_audit_summary.csv")
    }
    matched = 0
    for r in read(ROOT / "results/fig08_pareto.csv"):
        if r["audit_status"] == "pointwise empirical lower bound":
            source = audited[
                (r["dataset"], r["method"].removeprefix("m9_contractive_"), float(r["eps_target"]))
            ]
            assert equal(r["eps_audited_lb"], source["eps_lb"], tol=0)
            matched += 1
    assert matched == 9
    print("FX9 CONSISTENCY method_rows=9 TAB08_accounting_cells_traced=288 FIG08_audited_seed_rows_traced=9")
    pdfs = sorted((ROOT / "figs").glob("*.pdf"))
    for pdf in pdfs:
        assert pdf.stem in FIGURE_SOURCES, f"unregistered figure {pdf}"
        meta = current(pdf)
        config = __import__("yaml").safe_load(
            (ROOT / f'results/configs/{meta["config_hash"]}.yaml').read_text()
        )
        assert config["width_inches"] <= 5.5 and config["min_fontsize"] >= 7, pdf
        if pdf.stem == "fig01_decoupling":
            assert config["height_inches"] <= 3.4
        if pdf.stem in ["fig02_halflife", "fig02_retention_at_horizon"]:
            assert config["height_inches"] <= 2.4
        for name in FIGURE_SOURCES[pdf.stem] + ["method_descriptions.csv"]:
            assert pdf.stat().st_mtime_ns >= (ROOT / "results" / name).stat().st_mtime_ns, (pdf, name)
    assert len(pdfs) == len(FIGURE_SOURCES)
    briefing = ROOT / "paper/PAPER_BRIEFING.md"
    if briefing.exists():  # internal document; not part of the released tree
        assert len(briefing.read_text().splitlines()) <= 400
    print(
        f"FX9 CONSISTENCY figure_pdfs={len(pdfs)} stale_figures=0 max_width=5.5 min_font=7 PASS", flush=True
    )


if __name__ == "__main__":
    main()
