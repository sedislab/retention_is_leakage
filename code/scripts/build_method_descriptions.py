#!/usr/bin/env python3
"""FX7 (`08_FIX_PLAN.md` §11): `results/method_descriptions.csv`, one row per method, for Opus to
cite directly in the paper. Content is hand-curated from each method's own module docstring in
`code/src/p3fcl/methods/*.py` (the source of truth for what each method actually does in this
harness) plus its `MethodSpec.families`/`display_name` -- not re-derived from any experiment, so
(like `tab10_taxonomy.py`) this has no provenance sidecar: there is nothing here that a source-hash
or config-hash would meaningfully version.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

FIELDS = [
    "method_id", "display_name", "implemented", "differs_from_original",
    "retention_mechanism", "released_families", "display_short",
]

ROWS = [
    {
        "method_id": "m0_fedavg",
        "display_name": "FedAvg + sequential fine-tuning",
        "implemented": "Each task's clients locally fine-tune a shared linear softmax head over "
                        "frozen ViT features by plain SGD for local_epochs steps, and the server "
                        "FedAvg-aggregates the resulting weight deltas; no replay, distillation, or "
                        "regulariser.",
        "differs_from_original": "Not a reimplementation of a specific paper -- this is the "
                                  "project's retention lower bound and attack-calibration baseline.",
        "retention_mechanism": "None by design -- whatever survives in the shared weight is the "
                                "only carry-over across tasks.",
        "released_families": "F1 (model delta)",
    },
    {
        "method_id": "m1_glfc",
        "display_name": "GLFC",
        "implemented": "Local training combines cross-entropy on the current task's data with a "
                        "per-client, per-class exemplar buffer and a temperature-scaled knowledge-"
                        "distillation term (restricted to already-seen classes) pulling the new "
                        "head's old-class predictions toward the previous round's broadcast head.",
        "differs_from_original": "The original (CVPR'22) fine-tunes the whole backbone; this "
                                  "harness restricts training to a frozen-ViT linear head "
                                  "(feature-space-only reimplementation).",
        "retention_mechanism": "Per-client exemplar replay plus knowledge distillation against the "
                                "previous broadcast head, restricted to already-seen classes.",
        "released_families": "F1 (model delta), F7 (per-class counts)",
    },
    {
        "method_id": "m2_target",
        "display_name": "Gaussian feature replay (TARGET-style)",
        "implemented": "Each client fits a per-class diagonal Gaussian on its own current-task "
                        "frozen ViT features and releases the statistics once; the server aggregates "
                        "them (count-weighted) into a broadcast generator used to sample synthetic "
                        "replay features in later rounds.",
        "differs_from_original": "Explicitly NOT TARGET (ICCV'23)'s actual data-free generator -- a "
                                  "per-class diagonal-Gaussian approximation kept under the same "
                                  "config name for continuity; display_name states this plainly so "
                                  "it is never mistaken for a faithful TARGET reproduction.",
        "retention_mechanism": "Generative replay: sampling from an already-released, broadcast "
                                "per-class Gaussian to augment later local training.",
        "released_families": "F1 (model delta), F6 (generative)",
    },
    {
        "method_id": "m3_fot",
        "display_name": "FOT",
        "implemented": "Maintains a running orthonormal subspace spanning the dominant feature "
                        "directions of every task seen so far, and projects each task's local "
                        "gradient to remove its component along that subspace before the weight "
                        "update, so new-task learning does not overwrite directions old tasks "
                        "relied on.",
        "differs_from_original": "Reimplemented over frozen ViT features (feature-space-only) "
                                  "rather than the original (ICLR'24)'s raw-input backbone training; "
                                  "the subspace-update rule follows the paper's description.",
        "retention_mechanism": "Orthogonal-projection regularisation via a running feature "
                                "subspace; projection_strength (0=plain FedAvg, 1=full projection) "
                                "is the dose-response knob.",
        "released_families": "F1 (model delta), F5 (Gram/subspace)",
    },
    {
        "method_id": "m4_proto",
        "display_name": "PILoRA (prototype half)",
        "implemented": "Releases per-client, per-class, per-task prototypes (feature-space class "
                        "means) and their counts; no LoRA component (that needs backprop through "
                        "the ViT and is separate GPU-only work).",
        "differs_from_original": "Only the prototype half of PILoRA (ECCV'24) is implemented -- the "
                                  "LoRA fine-tuning component is out of scope for this CPU-array "
                                  "design and is a documented cut, not a hidden omission.",
        "retention_mechanism": "Per-class prototype carry-over; prototype_momentum (0=overwrite, "
                                ">0=blend with the previous release) is the dose-response knob.",
        "released_families": "F2 (prototype), F7 (per-class counts)",
    },
    {
        "method_id": "m5_hybrid_replay",
        "display_name": "Hybrid Replay FCIL",
        "implemented": "Each client keeps a per-class raw-feature exemplar buffer, releases newly-"
                        "added exemplars every round, and locally retrains on current-task data "
                        "plus buffered exemplars; the resulting model delta is released alongside "
                        "the exemplar buffer.",
        "differs_from_original": "Reimplemented over frozen ViT features; both the raw-exemplar "
                                  "release (F8) and the resulting model delta (F1) are reported, "
                                  "since the delta is causally a function of replayed old-task data "
                                  "and omitting it would understate leakage (CLAUDE.md non-"
                                  "negotiable #2).",
        "retention_mechanism": "Raw per-class exemplar replay buffer, keyed per (client, class).",
        "released_families": "F8 (raw exemplar), F1 (model delta)",
    },
    {
        "method_id": "m8_analytic",
        "display_name": "Analytic FCL",
        "implemented": "Each client emits one (Gram, cross-correlation) statistic pair per task from "
                        "that task's local data only; the server keeps a running sum over every task "
                        "and client seen so far and predicts by closed-form ridge regression.",
        "differs_from_original": "Faithful closed-form reimplementation; single-pass per-client-"
                                  "task release makes it task-disjoint by construction (the running "
                                  "sum retains influence, but each release itself never re-touches "
                                  "old data).",
        "retention_mechanism": "Unbounded accumulating running sum of per-task Gram/cross-"
                                "correlation statistics.",
        "released_families": "F5 (Gram)",
    },
    {
        "method_id": "m9_contractive",
        "display_name": "Contractive DP analytic (ours)",
        "implemented": "Same closed-form-ridge family as M8, projected onto a PCA basis fit on the "
                        "public ref split, with analytic-Gaussian DP noise added to one post-secure-"
                        "aggregation release per task, and a contractive (not accumulating) running "
                        "update R<-gamma*R+G_tilde, Q<-gamma*Q+H_tilde.",
        "differs_from_original": "This project's own proposed method (Algorithm 3 in the paper), "
                                  "not a reimplementation of prior work; gamma=1.0 recovers M8's "
                                  "unbounded running sum exactly (mod the ridge-lambda convention).",
        "retention_mechanism": "Contractive (geometrically-decaying) running Gram/cross-correlation "
                                "state, bounding any one release's long-run influence -- the "
                                "mechanism the paper's C3/C4 claims are about.",
        "released_families": "F5 (Gram)",
    },
]


def main() -> int:
    display = {"m0_fedavg": "M0 FedAvg", "m1_glfc": "M1 GLFC", "m2_target": "M2 Gaussian replay (semantic)",
               "m3_fot": "M3 FOT", "m4_proto": "M4 prototypes", "m5_hybrid_replay": "M5 exemplar replay (individual)",
               "m8_analytic": "M8 analytic", "m9_contractive": "M9 DP analytic"}
    for row in ROWS:
        row["display_name"] = display[row["method_id"]]
        row["display_short"] = row["method_id"].split("_")[0].upper()
        if row["method_id"] in {"m1_glfc", "m2_target", "m5_hybrid_replay"}:
            row["implemented"] += " FX9 objective: CE_mean(current) + replay_weight * CE_mean(replay), default replay_weight=1. M1 additionally retains its old-class KD_mean over current plus buffer."
        if row["method_id"] == "m5_hybrid_replay":
            row.update(implemented="Private per-client exemplar buffers train a frozen-feature head with CE_mean(current) + replay_weight * CE_mean(replay), default replay_weight=1. Only the F1 model delta is released; touched includes all replay reads.",
                       differs_from_original="Frozen-feature replay approximation; client exemplars remain private, as in Hybrid Replay.", released_families="F1 (model delta)")
        if row["method_id"] == "m9_contractive":
            row.update(
                implemented="Closed-form ridge on a PCA basis fitted to the public ref split. Each task releases one clipped, analytic-Gaussian-noised Gram/cross-correlation pair after secure aggregation. The code supports R <- gamma*R + G_tilde and Q <- gamma*Q + H_tilde; all reported sweep/audit and FX9 ledgers use gamma=1.",
                differs_from_original="This project's proposed DP analytic method. At gamma=1 the retained state accumulates; the current experiments do not test a contractive-update advantage. Non-private M8 comparisons also differ in projection and regularization, as documented by the sweep.",
                retention_mechanism="Accumulating Gram/cross-correlation state in the reported gamma=1 experiments; optional geometric decay exists in code but is not evaluated in this delivery.",
            )
        if row["method_id"] == "m4_proto":
            row["retention_mechanism"] = "Count-weighted server aggregate of client-local class means; optional momentum blends the server bank once per round. Local F2 means and F7 counts stay unchanged."
    ROWS.append(dict(method_id="protocol", display_name="Shared protocol", display_short="protocol", implemented="one FedAvg round per task; 30 local epochs; raw frozen features; 10 tasks and 10 clients",
                     differs_from_original="one FedAvg round per task is a shared limitation; gate overrides are recorded separately",
                     retention_mechanism="n/a", released_families="n/a"))
    out_csv = REPO_ROOT / "results" / "method_descriptions.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(ROWS)
    from p3fcl import provenance
    manifest = provenance.run_manifest(dict(phase="FX9-10", purpose="method descriptions and shared protocol"), seed=0)
    provenance.finalize(manifest, [out_csv])
    print(f"wrote {len(ROWS)} rows to {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
