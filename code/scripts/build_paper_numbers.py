#!/usr/bin/env python3
"""FX7 (`08_FIX_PLAN.md` §11): `results/paper_numbers.csv` -- "one row for every number the paper
may quote," columns `key, value, ci_lo, ci_hi, unit, source_csv, row_filter, note`. Opus (the
paper-writing session) pulls numbers from this file, so CLAUDE.md non-negotiable #3 applies here
exactly as it does to a figure script: every row is a mechanical transcription of an already-computed
cell from an already-committed `results/*.csv` or `tables/*.csv`, never a fresh computation. Nothing
here is hand-typed from memory.

This is a living document (`08_FIX_PLAN.md`: "continuous; final at freeze") -- re-run any time a
source CSV changes, and extend `_SOURCES` as new result tables land (Wave V3, FX8, the M9 audit).
Sources covered so far: TAB03 (headline leakage at elapsed=0/max), TAB04 (half-lives), the
decoupling-ratio table (only the 3 non-degenerate rows -- see the FX2 post-fix correction in
`agents/OPEN_QUESTIONS.md`'s H2 entry for why the other 30 are `undefined`/`by_construction`), FIG05's
eps(T) at the last observed T per (dataset, method, unit), TAB08 (M9's DP-utility Pareto), the FX4
accuracy gate, TAB07 (reproduction gaps vs. published numbers), FIG18 v2 (Camelyon17
matched-5-client TPR@1%FPR/accuracy per partition/elapsed -- the H13 correction), the M9 audit
(`m9_audit_summary.csv` -- Theorem 11's empirical check), and FIG03 (the FX8 dose-response sweep,
per-seed tpr1/retention_bwt for M5/M2 across all 6 knob levels).
"""
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import provenance  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = REPO_ROOT / "tables"

FIELDS = ["key", "value", "ci_lo", "ci_hi", "unit", "source_csv", "row_filter", "note"]

_VALUE_CI_RE = re.compile(r"^([-\d.]+)\s*\[([-\d.]+),\s*([-\d.]+)\]$")


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _parse_value_ci(s: str) -> tuple[str, str, str]:
    """TAB03 cells are `"value [ci_lo, ci_hi]"` strings (Clopper-Pearson for rates, t-interval for
    AUC) -- split them back into separate numeric fields rather than re-deriving the interval."""
    m = _VALUE_CI_RE.match(s.strip())
    if not m:
        return s, "", ""
    return m.group(1), m.group(2), m.group(3)


def _from_tab03(rows_out: list[dict]) -> None:
    path = TABLES_DIR / "tab03_leakage.csv"
    if not path.exists():
        return
    e_tag = {"elapsed=0": "e0", "elapsed=max": "e6"}
    for r in _read_csv(path):
        tag = e_tag.get(r["condition"], r["condition"])
        base_filter = (f"dataset=={r['dataset']!r} & method=={r['method']!r} & "
                       f"view=={r['view']!r} & condition=={r['condition']!r}")
        for metric, unit in (("tpr1", "TPR@1%FPR"), ("tpr01", "TPR@0.1%FPR"), ("auc", "AUC")):
            value, ci_lo, ci_hi = _parse_value_ci(r[metric])
            rows_out.append({
                "key": f"{r['method']}_{r['dataset']}_{r['view']}_{metric}_{tag}",
                "value": value, "ci_lo": ci_lo, "ci_hi": ci_hi, "unit": unit,
                "source_csv": "tables/tab03_leakage.csv", "row_filter": base_filter,
                "note": f"A1 LiRA, trajectory ablation, {r['view']} view, elapsed={r['elapsed']}",
            })


def _from_tab04(rows_out: list[dict]) -> None:
    path = TABLES_DIR / "tab04_halflife.csv"
    if not path.exists():
        return
    for r in _read_csv(path):
        ablation_tag = "" if r["ablation"] == "n/a" else f"_{r['ablation']}"
        note = f"status={r['status']}, horizon E={r['horizon_E']}"
        if r["status"] == "censored":
            note += " (never crossed half its base value within the observed horizon -- report as a lower bound '>E', do not extrapolate, per Definition 10)"
        rows_out.append({
            "key": f"{r['method']}_{r['dataset']}_{r['view']}_halflife_{r['quantity']}{ablation_tag}",
            "value": r["halflife"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "unit": "tasks",
            "source_csv": "tables/tab04_halflife.csv",
            "row_filter": (f"dataset=={r['dataset']!r} & method=={r['method']!r} & "
                           f"view=={r['view']!r} & quantity=={r['quantity']!r} & ablation=={r['ablation']!r}"),
            "note": note,
        })


def _from_decoupling_ratio(rows_out: list[dict]) -> None:
    path = RESULTS_DIR / "decoupling_ratio.csv"
    if not path.exists():
        return
    for r in _read_csv(path):
        if r["ratio_type"] not in ("point", "lower_bound"):
            continue  # "undefined"/"by_construction" carry no real experimental number to quote
        rows_out.append({
            "key": f"{r['method']}_{r['dataset']}_decoupling_ratio",
            "value": r["ratio"], "ci_lo": r["ratio_ci_lo"], "ci_hi": r["ratio_ci_hi"],
            "unit": "ratio (leak half-life / accuracy half-life)",
            "source_csv": "results/decoupling_ratio.csv",
            "row_filter": f"dataset=={r['dataset']!r} & method=={r['method']!r} & view=={r['view']!r}",
            "note": ("real point estimate" if r["ratio_type"] == "point"
                      else "lower bound -- leak half-life censored at the observed horizon, ratio only increases with a longer horizon"),
        })


def _from_fig05(rows_out: list[dict]) -> None:
    path = RESULTS_DIR / "fig05_eps_of_T.csv"
    if not path.exists():
        return
    rows = _read_csv(path)
    max_t: dict = defaultdict(int)
    for r in rows:
        if int(r["observed"]) == 1:
            k = (r["dataset"], r["method"], r["unit"])
            max_t[k] = max(max_t[k], int(r["T"]))
    by_key = {(r["dataset"], r["method"], r["unit"], int(r["T"])): r for r in rows}
    for (dataset, method, unit), t in sorted(max_t.items()):
        r = by_key[(dataset, method, unit, t)]
        rows_out.append({
            "key": f"{method}_{dataset}_eps_T{t}_{unit}",
            "value": r["eps"], "ci_lo": "", "ci_hi": "", "unit": "epsilon (DP)",
            "source_csv": "results/fig05_eps_of_T.csv",
            "row_filter": f"dataset=={dataset!r} & method=={method!r} & unit=={unit!r} & T=={t}",
            "note": f"last observed T (regime={r['regime']}); deterministic accountant computation, no CI",
        })


def _from_tab08(rows_out: list[dict]) -> None:
    path = TABLES_DIR / "tab08_dp_utility.csv"
    if not path.exists():
        return
    for r in _read_csv(path):
        base_filter = f"dataset=={r['dataset']!r} & unit=={r['unit']!r} & eps=={r['eps']!r}"
        rows_out.append({
            "key": f"m9_{r['dataset']}_{r['unit']}_acc_eps{r['eps']}",
            "value": r["final_avg_acc_mean"], "ci_lo": r["final_avg_acc_ci_lo"],
            "ci_hi": r["final_avg_acc_ci_hi"], "unit": "final average accuracy",
            "source_csv": "tables/tab08_dp_utility.csv", "row_filter": base_filter,
            "note": f"M9 contractive DP analytic, gamma=1.0, T=10, n_seeds={r['n_seeds']}",
        })
        for col in ("eps_certified_T10_U1", "eps_certified_T10_U2", "eps_certified_T10_U3", "eps_certified_T10_U4",
                    "eps_certified_T50_U1", "eps_certified_T50_U2", "eps_certified_T50_U3", "eps_certified_T50_U4"):
            if r.get(col):
                t_tag, u_tag = col.split("_")[2], col.split("_")[3]
                rows_out.append({
                    "key": f"m9_{r['dataset']}_{u_tag}_eps_certified_{t_tag}_eps{r['eps']}",
                    "value": r[col], "ci_lo": "", "ci_hi": "", "unit": "epsilon (DP)",
                    "source_csv": "tables/tab08_dp_utility.csv", "row_filter": base_filter,
                    "note": f"contractive M9 lifelong epsilon at {t_tag}, {u_tag}",
                })


def _from_fx4_gate(rows_out: list[dict]) -> None:
    path = RESULTS_DIR / "fx4_gate.csv"
    if not path.exists():
        return
    rows = _read_csv(path)
    by_combo: dict = defaultdict(list)
    for r in rows:
        by_combo[(r["dataset"], r["method"])].append(r)
    for (dataset, method), combo_rows in sorted(by_combo.items()):
        outcome = combo_rows[0]["pass"]
        accs = [float(r["final_avg_acc"]) for r in combo_rows]
        bwts = [float(r["bwt"]) for r in combo_rows]
        n = len(accs)
        row_filter = f"dataset=={dataset!r} & method=={method!r}"
        rows_out.append({
            "key": f"{method}_{dataset}_gate_final_avg_acc_mean",
            "value": sum(accs) / n, "ci_lo": "", "ci_hi": "", "unit": "final average accuracy",
            "source_csv": "results/fx4_gate.csv", "row_filter": row_filter,
            "note": f"mean over {n} seeds at the gate-selected config; gate outcome={outcome}",
        })
        rows_out.append({
            "key": f"{method}_{dataset}_gate_bwt_mean",
            "value": sum(bwts) / n, "ci_lo": "", "ci_hi": "", "unit": "backward transfer",
            "source_csv": "results/fx4_gate.csv", "row_filter": row_filter,
            "note": f"mean over {n} seeds at the gate-selected config; gate outcome={outcome}",
        })


def _from_tab07(rows_out: list[dict]) -> None:
    path = RESULTS_DIR / "tab07_reproduction_gap.csv"
    if not path.exists():
        return
    rows = _read_csv(path)
    seen: dict = defaultdict(int)
    for r in rows:
        # tab07's own `method` column uses display-style names (e.g. "M2_target",
        # "M4_prototype_half") rather than the canonical lowercase method_id every other source
        # here uses ("m2_target", "m4_proto") -- lowercase it for key consistency across this file;
        # row_filter still quotes tab07's own spelling verbatim so it stays a correct lookup.
        method = r["method"].lower()
        rows_out_key = f"{method}_our_reimpl_acc"
        if seen[method] == 0:
            rows_out.append({
                "key": rows_out_key, "value": r["our_reimpl_acc_mean"], "ci_lo": "", "ci_hi": "",
                "unit": "final average accuracy",
                "source_csv": "results/tab07_reproduction_gap.csv",
                "row_filter": f"method=={r['method']!r}",
                "note": f"n_seeds={r['our_reimpl_n_seeds']}, std={r['our_reimpl_acc_std']}; {r['our_protocol']}",
            })
        if r["published_acc"]:
            seen[method] += 1
            idx = seen[method]
            rows_out.append({
                "key": f"{method}_gap_vs_published_{idx}",
                "value": r["gap_ours_minus_published"], "ci_lo": "", "ci_hi": "",
                "unit": "accuracy fraction (ours - published)",
                "source_csv": "results/tab07_reproduction_gap.csv",
                "row_filter": f"method=={r['method']!r} & source_paper=={r['source_paper']!r}",
                "note": f"published={r['published_acc']} ({r['source_paper']}, {r['published_protocol']}). {r['note']}",
            })
        else:
            seen.setdefault(method, 0)


def _from_fig18(rows_out: list[dict]) -> None:
    path = RESULTS_DIR / "fig18_natural_federation.csv"
    if not path.exists():
        return
    for r in _read_csv(path):
        base_filter = f"partition=={r['partition']!r} & elapsed=={r['elapsed']!r}"
        rows_out.append({
            "key": f"m0_camelyon17_{r['partition']}_n{r['n_clients']}_tpr1_e{r['elapsed']}",
            "value": r["tpr1"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "unit": "TPR@1%FPR",
            "source_csv": "results/fig18_natural_federation.csv", "row_filter": base_filter,
            "note": (f"FIG18 v2 matched-{r['n_clients']}-client redesign, n_seeds={r['n_seeds']}; "
                     "H13 correction -- see agents/OPEN_QUESTIONS.md"),
        })
        rows_out.append({
            "key": f"m0_camelyon17_{r['partition']}_n{r['n_clients']}_acc_norm_e{r['elapsed']}",
            "value": r["acc"], "ci_lo": "", "ci_hi": "", "unit": "accuracy, normalised to elapsed=0",
            "source_csv": "results/fig18_natural_federation.csv", "row_filter": base_filter,
            "note": f"FIG18 v2 matched-{r['n_clients']}-client redesign, mean over {r['n_seeds']} seeds",
        })


def _from_m9_audit(rows_out: list[dict]) -> None:
    path = RESULTS_DIR / "m9_audit_summary.csv"
    if not path.exists():
        return
    for r in _read_csv(path):
        tag = "inf" if r["eps"] == "inf" else str(int(float(r["eps"])))
        base_filter = f"eps=={r['eps']!r}"
        for metric_key, ci_key, bound_key, fpr_pct in (
            ("tpr1", "tpr1_ci_hi", "bound_at_1pct_fpr", "1pct"),
            ("tpr01", "tpr01_ci_hi", "bound_at_0.1pct_fpr", "0.1pct"),
        ):
            rows_out.append({
                "key": f"m9_audit_cifar100_eps{tag}_tpr_{fpr_pct}fpr",
                "value": r[metric_key], "ci_lo": "", "ci_hi": r[ci_key], "unit": f"TPR@{fpr_pct}FPR",
                "source_csv": "results/m9_audit_summary.csv", "row_filter": base_filter,
                "note": f"online LiRA vs global state, elapsed=0, trajectory; DP bound={r[bound_key]}, "
                        f"audit_passed={r['audit_passed']}, n_shadows={r['n_shadows']}",
            })
        rows_out.append({
            "key": f"m9_audit_cifar100_eps{tag}_auc",
            "value": r["auc"], "ci_lo": "", "ci_hi": "", "unit": "AUC",
            "source_csv": "results/m9_audit_summary.csv", "row_filter": base_filter,
            "note": f"online LiRA vs global state, n_shadows={r['n_shadows']}",
        })


def _from_fig03(rows_out: list[dict]) -> None:
    path = RESULTS_DIR / "fig03_dose_response.csv"
    if not path.exists():
        return
    for r in _read_csv(path):
        base_filter = (f"method=={r['method']!r} & knob_value=={r['knob_value']!r} & "
                       f"seed=={r['seed']!r}")
        key_base = f"{r['method']}_cifar100_{r['knob_name']}{r['knob_value']}_seed{r['seed']}"
        rows_out.append({
            "key": f"{key_base}_tpr1", "value": r["tpr1"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
            "unit": "TPR@1%FPR", "source_csv": "results/fig03_dose_response.csv",
            "row_filter": base_filter, "note": "FX8 dose-response, pooled K=[0,1,2,3], elapsed=6",
        })
        rows_out.append({
            "key": f"{key_base}_retention_bwt", "value": r["retention_bwt"], "ci_lo": "", "ci_hi": "",
            "unit": "backward transfer", "source_csv": "results/fig03_dose_response.csv",
            "row_filter": base_filter, "note": "FX8 dose-response; -BWT is the retention-strength reading",
        })


def main() -> int:
    rows_out: list[dict] = []
    _from_tab03(rows_out)
    _from_tab04(rows_out)
    _from_decoupling_ratio(rows_out)
    _from_fig05(rows_out)
    _from_fig18(rows_out)
    _from_fig03(rows_out)
    _from_m9_audit(rows_out)
    _from_tab08(rows_out)
    _from_fx4_gate(rows_out)
    _from_tab07(rows_out)

    keys = [r["key"] for r in rows_out]
    dupes = {k for k in keys if keys.count(k) > 1}
    if dupes:
        print(f"ERROR: duplicate keys, would silently shadow each other: {sorted(dupes)}", file=sys.stderr)
        return 1

    out_csv = RESULTS_DIR / "paper_numbers.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {len(rows_out)} rows to {out_csv}")

    config = {"seed": 0, "purpose": "FX7 paper_numbers.csv: mechanical transcription of every landed result table"}
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
