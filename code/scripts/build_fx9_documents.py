#!/usr/bin/env python3
"""Paper briefing and hypothesis updates transcribed from accepted final CSVs."""

import re

from fx9_io import ROOT, read
from p3fcl.figure_sources import FIGURE_SOURCES


def one(rows, **filters):
    matches = [r for r in rows if all(r[k] == str(v) for k, v in filters.items())]
    assert len(matches) == 1, (filters, len(matches))
    return matches[0]


def value(row, key, lo=None, hi=None):
    v = float(row[key])
    low = row.get(lo or key + "_ci_lo", "")
    high = row.get(hi or key + "_ci_hi", "")
    return f"{v:.6g} [{float(low):.6g}, {float(high):.6g}]" if low and high else f"{v:.6g}"


def update_hypothesis(text, hypothesis, status, lines):
    pattern = rf"(^### {hypothesis} —.*?)(?=^### |^## |\Z)"
    match = re.search(pattern, text, re.M | re.S)
    assert match, hypothesis
    block = match.group(1)
    block = re.sub(
        r"^- Status:[^\n]*(?:\n[ \t]+[^\n]*)*", f"- Status: `{status}`", block, count=1, flags=re.M
    )
    block = re.sub(r"\n<!-- FX9 START -->.*?<!-- FX9 END -->\n", "\n", block, flags=re.S)
    insertion = (
        "\n<!-- FX9 START -->\n- **FX9 final evidence supersedes earlier numerical claims below.**\n"
        + "\n".join("  " + line for line in lines)
        + "\n<!-- FX9 END -->\n"
    )
    status_end = re.search(r"^- Status:.*$", block, re.M).end()
    block = block[:status_end] + insertion + block[status_end:]
    return text[: match.start()] + block + text[match.end() :]


def main():
    horizon = read(ROOT / "results/fig02_retention_at_horizon.csv")
    utility = read(ROOT / "results/fx9_utility_summary.csv")
    gates = read(ROOT / "results/fx9_gate_comparison.csv")
    ratios = read(ROOT / "results/decoupling_ratio.csv")
    gaps = read(ROOT / "results/fx9_m9_gap_summary.csv")
    cert = read(ROOT / "results/m9_certified.csv")
    audit = read(ROOT / "results/m9_audit_summary.csv")
    dose = read(ROOT / "results/fx9_dose_summary.csv")
    tab08 = read(ROOT / "tables/tab08_dp_utility.csv")
    fig18 = read(ROOT / "results/fig18_natural_federation.csv")
    gram = read(ROOT / "results/fig13_gram_inversion_summary.csv")
    a6 = read(ROOT / "results/a6_property_inference.csv")
    seed_variance = read(ROOT / "results/fig17_seed_variance.csv")
    names = {r["method_id"]: r["display_name"] for r in read(ROOT / "results/method_descriptions.csv")}
    lines = [
        "# PAPER BRIEFING — FX9 final evidence",
        "",
        "This file supersedes the archived pre-FX9 briefing. Experimental numbers below are formatted from the named final CSVs; exact cells and row filters are in `results/paper_numbers.csv`. Intervals are 95% unless explicitly qualified. No hypothesis is marked CONFIRMED by the agent.",
        "",
        "## ERRATA",
        "",
        "- R1: training accuracy presented as generalization → withdrawn; utility and accuracy matrices now use the held-out test split, with separate acc_train columns.",
        "- R2: late-task replay learning described as retention → withdrawn; client-private buffers, old-class KD, balanced losses and the FX9 gate comparison below replace it.",
        "- R3: privacy touched sets used as FedAvg weights → withdrawn; released aggregation weights reconstruct the actual model; touched still records replay reads.",
        "- R4: M4/M8 full-view persistence called a retention half-life → withdrawn; per-client releases are constant by construction. Aggregate and global state are separate views.",
        "- R5: old lifelong-accountant curves → withdrawn; current FIG05 uses the unchanged, corrected FX1 record-composition accountant on real ledgers. See its sensitivity qualification below.",
        "- R6: extrapolated exponential half-lives → withdrawn; current TAB04 uses fixed K, chance floors, observed first crossing and explicit censoring at E.",
        "- R7: independent-pair headline leakage CIs → withdrawn; current leakage curves use hierarchical seed/target bootstrap, with paired seed/task/target horizon intervals.",
        "- R8: unmatched-client FIG18 comparison → withdrawn; preserved v2 has matched client counts and slide groups within hospital tasks, with limited scope.",
        "- R9: blanket secure-aggregation privacy claim → withdrawn; F1 model-based attacks are identical by construction, while F2/F5 views are measured separately.",
        "- R10: original dose pilot and inert M4 knob → withdrawn; balanced M5/M2 sweeps replace them, with independent replay weight and sample-count knobs.",
        "- R11: old forest plot, small fonts and causal anisotropy interpretation → withdrawn; horizon scatter and consistent styling replace them; FIG13 uses medians and omits the anisotropy panel. A4 remains inconclusive; A6 is explicit count leakage.",
        "- R12: empirical M9 results absent while C4 was asserted → withdrawn; M9 sweep/audit exist, with DP-baseline superiority still untested.",
        "- B1: FIG01/FIG19 elapsed/task-index mix-up → corrected through accuracy_curve and retention_curves.csv; diagonal checks pass for every method.",
        "- B2: last-client prototype overwrite → count-weighted server bank; momentum applies once per round. Global and aggregate scores and leakage now agree in class-incremental streams. The old TAB05 comparison also differs in feature preprocessing, so it is not an isolated bank-only ablation.",
        "- B3: single union-mean replay loss → balanced current and replay means, with unchanged M1 union KD. Gate failures below remain failures; persistent last-task lag confounds retention interpretations on those combinations.",
        "- B4: TAB05 standardized features → raw attacked pipeline, identical configurations and independent per-seed accuracy checks.",
        "- B5: M5 raw-buffer F8 release → removed; only F1 is released, with private buffer IDs still counted in touched.",
        "",
        "## PROTOCOL",
        "",
        "- Frozen ViT-B/16 (IN21k-pretrained) feature caches; raw features, no standardization.",
        "- One FedAvg round per task; 30 local epochs; default learning rate 0.5. Train-validation gate overrides in fx9_gate.csv take precedence and are never selected on test utility.",
        "- 10 class-incremental tasks × 10 clients, Dirichlet β=0.5; test-split predictions restricted to seen classes.",
        "- Main changed-method LiRA: 1024 shadows × 3 seeds; calibration_frac=0.8, K={0,1,2,3}, E=6, trajectory and last-round ablations. M0 preserves its existing larger five-seed budget; see a1_m0_budget_check.csv.",
        "- Accuracy uses 5 seeds. Main leakage uses 3 seeds except M0 (5). Two thousand bootstrap replicates; accuracy resamples seeds/tasks, leakage seeds/targets; horizon leakage adds task resampling.",
        "- M0 ratio uses a joint seed/task/target draw for both half-lives. Lower-bound ratio intervals describe the lower-bound statistic, not a finite upper confidence bound on the uncensored ratio.",
        "- Non-M0 horizon accuracy CIs retain all 5 accuracy seeds; leakage intervals use the available 3 shadow seeds. Do not claim a 5-seed joint privacy/accuracy covariance for these rows.",
        "- FX1 accountant baseline noise multiplier σ=2, δ=1e-5, W=3; multiplicity counts released records. M9 instead uses actual analytic-Gaussian σ/Δ and records exact single-release eps0_analytic.",
        "- M9 hyperparameters and PCA are selected/fitted on the public ref split; sweep and audit remain unchanged. Accounting ledgers use γ=1 at 10 and 50 tasks.",
        "- M9 cross-unit FX1 numbers are conditional on the corresponding group sensitivity. A U1-calibrated ledger alone does not establish a U2–U4 group guarantee; do not cite those conditional values as unconditional certificates.",
        "- Audited epsilon uses the requested maximum of forward/mirror one-sided Clopper–Pearson bounds over thresholds. These are pointwise empirical bounds on dependent shadow/target observations, not a simultaneous formal DP certificate.",
        "- M2 is a Gaussian feature-replay approximation; M4 is prototypes only; M1/M5 train frozen-feature heads. Full backbone/prompt/LoRA reproductions were not delivered.",
        "",
        "## FINAL RESULTS",
        "",
    ]
    descriptions = {
        "fig01_decoupling": "Normalized accuracy and leakage by dataset. Full per-client M4/M8 releases are dashed and constant by construction; global state is distinct. Exact horizon values follow below.",
        "fig02_retention_at_horizon": "A(6) versus L(6), with intervals on both axes and reference lines y=1 and y=x. It replaces the old portrait forest plot; half-lives remain in TAB04.",
        "fig02_halflife": "Compatibility filename for the same horizon scatter as FIG02; it no longer plots half-lives.",
        "fig03_dose_response": "Balanced replay sample-count dose response for M5 and M2 on CIFAR. The utility axis is forgetting (−BWT); curve shape is reported without imposing monotonicity.",
        "fig04_semantic_vs_individual": "Forgetting versus leakage for individual and semantic replay, with dose labels and both intervals. All points come from the same dose CSV as FIG03.",
        "fig05_eps_of_T": "FX1 record-composition values by privacy unit; solid segments are observed and dashed segments extrapolated. M9 uses actual noise, with both calibration epsilons and the cross-unit sensitivity qualification above.",
        "fig08_pareto": "Preserved M9 utility sweep at gamma=1, with non-private M0/M8 references and utility intervals. Audited epsilon is populated only for the actual CIFAR/U1/epsilon audit cells; unsupported cells are explicitly labelled, not inferred.",
        "fig10_m9_audit": "Preserved empirical ROC versus analytic single-release DP envelope. A weak attack supports an audit result, not proof that a mechanism is private.",
        "fig11_secure_agg": "Full, aggregate and global TPR at e=0 and e=6 on CIFAR. F1 bars coincide by construction; M4 aggregate/global coincide after its bank and calibration fixes.",
        "fig11_secure_agg_appendix_cub200": "Same three-view comparison on CUB, retaining M4 aggregate/global equality.",
        "fig11_secure_agg_appendix_imagenet_r": "Same three-view comparison on ImageNet-R, retaining M4 aggregate/global equality.",
        "fig13_gram_inversion": "Median of trial-mean reconstruction cosines with bootstrap intervals, 25 trials per feasible cell. No anisotropy panel or causal anisotropy claim; infeasible large CUB shards are absent.",
        "fig16_roc": "Log-log ROC from the exact calibrated fixed-K score sidecars, e=0, full view, representative seed 0. This appendix visualization does not replace pooled multi-seed headline estimates.",
        "fig17_seed_variance": "Per-seed e=0 full-view TPR. M0 has five seeds; the other methods have three.",
        "fig18_natural_federation": "Preserved v2 appendix experiment: clients are slide groups within each hospital task versus matched Dirichlet groups. This is not a hospital-as-client federation and cannot settle that design objection.",
        "fig19_retention_vs_release": "M4/M8 full-release and global-state leakage alongside accuracy, all from the unified CIFAR curves. Release persistence must not be interpreted as an uncensored accuracy/leakage half-life ratio.",
        "fig_a6_property_inference": "Count-based property inference, retained as an explicit released-count disclosure demonstration. It is not evidence of subtle inferred leakage.",
    }
    for stem in FIGURE_SOURCES:
        lines += [
            f"**{stem.upper()} — `figs/{stem}.pdf`; source `results/{FIGURE_SOURCES[stem][0]}`.** {descriptions[stem]}",
            "",
        ]
    lines += [
        "**Horizon numbers — `results/fig02_retention_at_horizon.csv` (dataset / method / view: A(6); L(6)).**"
    ]
    for r in horizon:
        lines.append(
            f'- {r["dataset"]} / {names[r["method"]]} / {r["view"]}: {value(r,"acc_norm")}; {value(r,"leak_norm")}.'
        )
    lines += ["", "**M0 ratio — `results/decoupling_ratio.csv`.**"]
    ratio_lines = []
    for r in ratios:
        text = f'{r["dataset"]}: ratio {value(r,"ratio")}, {r["ratio_type"]}; {r["ci_interpretation"]}.'
        ratio_lines.append(text)
        lines.append("- " + text)
    lines += [
        "",
        "**Gate comparison — `results/fx9_gate_comparison.csv`: last-task accuracy before → after; BWT before → after.**",
    ]
    for r in gates:
        lines.append(
            f'- {r["dataset"]}/{r["method"]}: {value(r,"last_before")} → {value(r,"last_after")}; {value(r,"bwt_before")} → {value(r,"bwt_after")}; gate={r["gate_pass"]}.'
        )
    lines += [
        "",
        "**Dose response — `results/fx9_dose_summary.csv`: sample count; forgetting; TPR@1%FPR at e=6.**",
    ]
    for r in dose:
        lines.append(
            f'- {names[r["method"]]}, {r["knob_value"]}: {value(r,"forgetting")}; {value(r,"tpr1")}.'
        )
    lines += [
        "",
        "**TAB02 — `tables/tab02_units.csv/.tex`.** Privacy units and neighboring relations; definitional table, no empirical interval.",
        "**TAB03 — `tables/tab03_leakage.csv/.tex`.** TPR@1% and @0.1% FPR plus AUC at e=0/e=6, from current fixed-K summaries. Rates are reported with hierarchical intervals; full M4/M8 transcript persistence is structural.",
        "**TAB04 — `tables/tab04_halflife.csv/.tex`.** Accuracy/leakage half-lives and explicit ok/censored/no_signal status. Blank finite CIs for censored half-lives are intentional, not missing estimates.",
        "**TAB05 — `results/tab05_utility_baselines.csv`; means/CIs in `results/fx9_utility_summary.csv`.** Raw attacked configurations; 10 tasks, five seeds. Every final per-seed value agrees with its accuracy matrix within 1e-9.",
    ]
    for r in utility:
        lines.append(
            f'- {r["dataset"]}/{names[r["method"]]}: final average accuracy {value(r,"final_avg_acc")}; BWT {value(r,"bwt")}.'
        )
    lines += [
        "**TAB07 — `results/tab07_reproduction_gap.csv`.** Updated raw-feature utility versus published protocols. Backbone, federation and algorithm differences prevent treating these gaps as faithful reproduction rankings.",
        "**TAB08 — `tables/tab08_dp_utility.csv/.tex`.** Preserved M9 utility with eight FX1 accounting columns at T=10/50. The columns follow the explicit sensitivity qualification above; infinity denotes the non-private mechanism.",
        "**M9 accounting — `results/m9_certified.csv`: calibration epsilon / T / unit: FX1 epsilon (deterministic, no sampling CI).**",
    ]
    for r in cert:
        if r["eps0"] in ["1.0", "4.0"]:
            lines.append(
                f'- {r["eps0"]} / {r["T"]} / {r["unit"]}: {value(r,"eps")}; exact single-release epsilon={r["eps0_analytic"]}.'
            )
    audit_lines = []
    for r in audit:
        text = f'CIFAR/U1 epsilon={r["eps"]}: empirical epsilon_lb={value(r,"eps_lb")}; TPR@1%FPR={value(r,"tpr1")} (upper endpoint {value(r,"tpr1_ci_hi")}).'
        audit_lines.append(text)
        lines.append("- " + text)
    lines += [
        "**TAB10 — `tables/tab10_taxonomy.csv/.tex`.** Taxonomy and proof-scope table; definitions rather than new measurements.",
        "",
        "**FIG13 selected cells — `results/fig13_gram_inversion_summary.csv`.**",
    ]
    for r in gram:
        if r["n_per_class"] in ["1", "4", "16", "64"] and r["ref_quality"] == "clean_full":
            lines.append(
                f'- {r["dataset"]}, n={r["n_per_class"]}, ref={r["ref_quality"]}: median cosine {value(r,"median_cos","ci_lo","ci_hi")}, trials={r["n_trials"]}.'
            )
    fig18_lines = []
    for r in fig18:
        if r["elapsed"] in ["0", "4"]:
            text = f'{r["partition"]}, e={r["elapsed"]}: TPR@1%FPR {value(r,"tpr1","ci_lo","ci_hi")}; normalized accuracy {value(r,"acc")}; clients={r["n_clients"]}.'
            fig18_lines.append(text)
            lines.append("- FIG18 " + text)
    lines += [
        "",
        "**FIG17 representative seed values — `results/fig17_seed_variance.csv`, CIFAR e=0 full-view TPR@1%FPR.** The strip plot shows individual seeds; pooled intervals are in TAB03.",
    ]
    for method in sorted({r["method"] for r in seed_variance}):
        rr = [r for r in seed_variance if r["dataset"] == "cifar100" and r["method"] == method]
        lines.append(
            f"- {names[method]}: " + "; ".join(f"seed {r['seed']}={value(r,'value')}" for r in rr) + "."
        )
    lines += [
        "**A6 — `results/a6_property_inference.csv`.** Released counts directly disclose the property; this is a disclosure check, with no multi-seed uncertainty estimate.",
    ]
    for r in a6:
        if r["elapsed"] in ["-1", "0", "6"]:
            lines.append(f"- elapsed={r['elapsed']}: balanced accuracy {value(r,'balanced_accuracy')}.")
    lines += [
        "",
        "## CUT",
        "",
        "- DP-FedAvg / DP linear-probe baselines: not delivered; no equal-epsilon superiority claim. H8 is CUT.",
        "- FIG09 / privacy-induced participation drift: cut from this repair schedule to prioritize the mandatory corrected experiments; no deciding simulation.",
        "- 20-task TAB05 and Camelyon TAB05 rows: removed from the current attacked-protocol table. FIG18 appendix retains its own separately scoped accuracy.",
        "- FIG18 hospital-as-client design: not delivered; v2 uses slide groups within hospital tasks on the 5k-patch subsample.",
        "- M6/M7 prompt/LoRA methods and M4 LoRA half: GPU reproduction cuts; main evidence covers seven CPU methods plus the separately scoped M9.",
        "- Full-trajectory empirical M9 epsilon certification: not claimed by the one-release audit. The threshold maximum is not a simultaneous calibrated certificate.",
        "- Unconditional U2–U4 certificates from U1-calibrated M9: not established by the unchanged FX1 composition calculation; sensitivity matching is required.",
        "- Unobserved/undefined retention ratios outside M0: omitted; no infinity-as-point-estimate or exponential extrapolation.",
        "- FIG06/FIG07/FIG12/FIG14/FIG15 and TAB01/TAB06/TAB09: no separate final artifact in this delivery; related evidence is scoped to the listed existing figures/tables.",
        "- C2 universality / superiority to original trainable-backbone methods: not established by these feature-space experiments.",
    ]
    assert len(lines) <= 400, len(lines)
    (ROOT / "paper/PAPER_BRIEFING.md").write_text("\n".join(lines) + "\n")
    questions = (ROOT / "agents/OPEN_QUESTIONS.md").read_text()
    h2 = (
        ratio_lines
        + [
            f'{r["method"]} dose={r["knob_value"]}: forgetting {value(r,"forgetting")}, TPR {value(r,"tpr1")}.'
            for r in dose
            if r["knob_value"] in ["1", "50"]
        ]
        + [
            "Sources: results/decoupling_ratio.csv, results/fx9_dose_summary.csv. Status OPEN: seven CPU methods and two dose arms do not meet the full breadth of the original deciding test. Minus BWT measures forgetting; the old retention-strength label is withdrawn."
        ]
    )
    h7 = (
        audit_lines
        + [
            f'{r["dataset"]} U2 epsilon=1 utility {value(r,"final_avg_acc_mean","final_avg_acc_ci_lo","final_avg_acc_ci_hi")}.'
            for r in tab08
            if r["unit"] == "U2" and r["eps"] == "1.0"
        ]
        + [
            "Source: tables/tab08_dp_utility.csv; no DP-FedAvg comparator, so superiority remains untested. Cross-unit FX1 accounting is conditional on sensitivity."
        ]
    )
    h7 += [
        f"{r['dataset']} U2 epsilon=1 paired M9−M8 accuracy gap {value(r,'gap_m9_minus_m8')}."
        for r in gaps
        if r["unit"] == "U2" and float(r["eps"]) == 1
    ]
    h7_status = (
        "REFUTED"
        if any(
            float(r["gap_m9_minus_m8_ci_hi"]) < -0.05
            for r in gaps
            if r["dataset"] in ["cifar100", "imagenet_r"] and r["unit"] == "U2" and float(r["eps"]) == 1
        )
        else "OPEN"
    )
    h7 += [
        "The within-five-points utility criterion is refuted when its paired gap interval lies below −0.05; the separate DP-baseline superiority comparison remains untested. Source: results/fx9_m9_gap_summary.csv."
    ]
    views = read(ROOT / "results/fx3_views_summary.csv")
    h11 = [
        f'{r["dataset"]} {r["method"]} {r["view"]} e=6 TPR {value(r,"tpr1")}.'
        for r in views
        if r["method"] in ["m4_proto", "m8_analytic"]
        and r["view"] in ["aggregate", "global"]
        and r["elapsed"] == "6"
    ] + [
        "Source: results/fx3_views_summary.csv. F1 invariance is by construction; the blanket no-help claim is refuted as a universal cross-family statement."
    ]
    for hypothesis, status, items in [
        ("H2", "OPEN", h2),
        (
            "H5",
            "SUPPORTED (reconstruction curve only; original threshold not established)",
            [
                f"{r['dataset']}, n={r['n_per_class']}, ref={r['ref_quality']}: median trial-mean cosine {value(r,'median_cos','ci_lo','ci_hi')}; trials={r['n_trials']}."
                for r in gram
                if r["ref_quality"] == "clean_full" and r["n_per_class"] in ["1", "4", "16", "64"]
            ]
            + [
                "Source: results/fig13_gram_inversion_summary.csv. The former five-trials caveat is superseded by 25 trials per feasible cell and 2000 median-bootstrap replicates. Trial-mean cosine is not a fraction of individual samples above 0.8; no image-space identifiability claim is established."
            ],
        ),
        ("H7", h7_status, h7),
        ("H8", "CUT", ["No DP baselines or deciding long-horizon baseline comparison delivered."]),
        ("H11", "REFUTED", h11),
        (
            "H13",
            "OPEN",
            fig18_lines
            + [
                "Source: results/fig18_natural_federation.csv. Preserved matched-client slide-group experiment; hospital-as-client question remains open."
            ],
        ),
    ]:
        questions = update_hypothesis(questions, hypothesis, status, items)
    (ROOT / "agents/OPEN_QUESTIONS.md").write_text(questions)
    print(
        f"FX9-10 DOCUMENTS ACCEPT briefing_lines={len(lines)} errata=R1-R12,B1-B5 figures={len(FIGURE_SOURCES)} H8=CUT updated=H2,H7,H11,H13",
        flush=True,
    )


if __name__ == "__main__":
    main()
