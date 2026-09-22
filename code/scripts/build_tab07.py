#!/usr/bin/env python3
"""TAB07 (Gate P2 deliverable, hypothesis H12): reproduction gap between our reimplementation and
each paper's published number. Per CLAUDE.md non-negotiable #8, we do not chase original codebases —
we reimplement inside our harness against the paper's description and report the gap honestly,
including when we could not verify a published number at all.

Published numbers below were looked up from the actual papers (see `source` column) on 2026-09-15,
not recalled from memory — each is a direct quote/reading of a results table, cited by URL. Where a
method has been reproduced under materially different protocols in different papers (common in this
literature), multiple published numbers are recorded rather than picking one and hiding the disagreement
— that disagreement is itself informative about how sensitive these numbers are to protocol details.

Our number is the mean `final_avg_acc` from `results/tab05_utility_baselines.csv` at the closest
matching protocol available in our sweep (CIFAR-100, 10 tasks, 3 seeds) — never a hand-typed number.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import provenance  # noqa: E402

TAB05 = REPO_ROOT / "results" / "tab05_utility_baselines.csv"
OUT_CSV = REPO_ROOT / "results" / "tab07_reproduction_gap.csv"

# Published numbers, looked up 2026-09-15. Multiple rows per method where multiple papers report a
# number for it under different protocols -- see `protocol` and `source` for exactly what differs.
PUBLISHED = [
    {
        "method": "M1_glfc", "published_acc": 0.669, "protocol": "10 tasks, CIFAR-100, ResNet18, 30-120 clients",
        "source_paper": "Dong et al., CVPR 2022 (GLFC's own paper), Table 1",
        "source_url": "https://arxiv.org/abs/2203.11473",
    },
    {
        "method": "M1_glfc", "published_acc": 0.6183, "protocol": "10/10/50/5 (tasks/classes-per-task/clients/active-per-round), CIFAR-100",
        "source_paper": "Nori et al., ICLR 2025 (Hybrid Replay paper's independent reproduction of GLFC), Table 2",
        "source_url": "https://arxiv.org/pdf/2501.15356",
    },
    {
        "method": "M2_target", "published_acc": 0.3631, "protocol": "5 tasks, CIFAR-100, ResNet18, trainable generator",
        "source_paper": "Zhang et al., ICCV 2023 (TARGET's own paper), §5 text",
        "source_url": "https://arxiv.org/pdf/2303.06937",
    },
    {
        "method": "M2_target", "published_acc": 0.713, "protocol": "10 tasks, CIFAR-100, quantity-skew alpha=6, 10 clients",
        "source_paper": "Guo et al., ECCV 2024 (PILoRA paper's independent reproduction of TARGET), Table 1 (avg. accuracy column)",
        "source_url": "https://arxiv.org/abs/2401.02094",
    },
    {
        "method": "M2_target", "published_acc": 0.5003, "protocol": "10/10/50/5, CIFAR-100",
        "source_paper": "Nori et al., ICLR 2025 (Hybrid Replay paper's independent reproduction of TARGET), Table 2",
        "source_url": "https://arxiv.org/pdf/2501.15356",
    },
    {
        "method": "M4_prototype_half", "published_acc": 0.786, "protocol": "10 tasks, CIFAR-100, quantity-skew alpha=6, 10 clients -- FULL PILoRA (prototype+LoRA), not the prototype-only half we reimplement",
        "source_paper": "Guo et al., ECCV 2024 (PILoRA's own paper), Table 1 (avg. accuracy column)",
        "source_url": "https://arxiv.org/abs/2401.02094",
    },
    {
        "method": "M5_hybrid_replay", "published_acc": 0.6584, "protocol": "10/10/50/5, CIFAR-100",
        "source_paper": "Nori et al., ICLR 2025 (Hybrid Replay's own paper), Table 2",
        "source_url": "https://arxiv.org/pdf/2501.15356",
    },
]

BACKBONE_CONFOUND = (
    "CAVEAT (read before citing this gap): our number sits on a frozen, IN21k-pretrained ViT-B/16 "
    "(RESEARCH_PLAN.md's cacheable-methods architecture); the published number trains a ResNet18 "
    "substantially from scratch inside the federated loop. A strong-enough frozen backbone routinely "
    "outperforms a weak-backbone method with sophisticated anti-forgetting machinery on its own -- "
    "this gap is dominated by that protocol difference, not by reimplementation fidelity. See "
    "notes/2026-09-15_p2_tab05_tab07.md."
)

NOTES = {
    "M0_fedavg_sequential": "No single published number exists -- M0 is a generic sequential-fine-tuning "
        "baseline (RESEARCH_PLAN.md §5.1: 'retention lower bound, calibration point for attacks'), not "
        "one paper's proposed method. Reported for completeness; no gap computed.",
    "M1_glfc": BACKBONE_CONFOUND,
    "M2_target": "Three independent papers report three very different numbers for the SAME method "
        "(0.363 / 0.713 / 0.500) under three different protocols (5 vs 10 tasks, different client "
        "counts and skew) -- this is itself evidence for H12 (reproduction attrition): published "
        "numbers for these methods are highly protocol-sensitive, which is a finding, not noise. "
        + BACKBONE_CONFOUND,
    "M4_prototype_half": "We reimplement only PILoRA's prototype half (frozen backbone, no LoRA "
        "fine-tuning of the ViT -- LoRA needs backprop through the network and is out of scope for the "
        "cacheable-methods pass, 00_BUILD_PLAN.md P2). This is the one method in the table where our "
        "number is LOWER than published, which is independent structural evidence for the same "
        "backbone-confound point (see M1/M2/M5's note): the missing LoRA half is exactly the part that "
        "would let the backbone adapt, and its absence costs accuracy in the direction you'd expect.",
    "M3_fot": "Could not extract an exact Split-CIFAR100 accuracy figure from the paper in this "
        "session (ICLR proceedings PDF and OpenReview PDF both resisted automated extraction; "
        "WebSearch surfaced only the qualitative claim of 'up to 15% average accuracy gain, 27% lower "
        "forgetting than SOTA' without a table). Recorded as an open follow-up, not fabricated.",
    "M5_hybrid_replay": BACKBONE_CONFOUND,
    "M8_analytic_fcl": "No independent public benchmark number exists to compare against -- M8 is our "
        "reimplementation of the internal 'Analytic FCL' preprint reference (RESEARCH_PLAN.md, "
        "preprint '26) that motivates M9's design, not a method with an established external leaderboard entry.",
}


def main() -> int:
    tab05 = pd.read_csv(TAB05)
    ours = (
        tab05[(tab05["dataset"] == "cifar100") & (tab05["n_tasks_requested"] == 10)]
        .groupby("method")["final_avg_acc"]
        .agg(["mean", "std", "count"])
    )

    method_id_map = {
        "M0": "M0_fedavg_sequential", "M1": "M1_glfc", "M2": "M2_target", "M3": "M3_fot",
        "M4": "M4_prototype_half", "M5": "M5_hybrid_replay", "M8": "M8_analytic_fcl",
    }

    rows = []
    published_by_method: dict = {}
    for p in PUBLISHED:
        published_by_method.setdefault(p["method"], []).append(p)

    for short, full in method_id_map.items():
        our_mean = ours.loc[short, "mean"] if short in ours.index else None
        our_std = ours.loc[short, "std"] if short in ours.index else None
        our_n = ours.loc[short, "count"] if short in ours.index else 0
        pubs = published_by_method.get(full, [{"published_acc": None, "protocol": "n/a", "source_paper": "n/a", "source_url": ""}])
        for pub in pubs:
            gap = None
            if our_mean is not None and pub["published_acc"] is not None:
                gap = round(our_mean - pub["published_acc"], 4)
            rows.append({
                "method": full,
                "our_reimpl_acc_mean": round(our_mean, 4) if our_mean is not None else None,
                "our_reimpl_acc_std": round(our_std, 4) if our_std is not None else None,
                "our_reimpl_n_seeds": int(our_n),
                "our_protocol": "CIFAR-100, 10 class-incremental tasks, frozen ViT-B/16 features, 10 clients, Dirichlet beta=0.5",
                "published_acc": pub["published_acc"],
                "published_protocol": pub["protocol"],
                "source_paper": pub["source_paper"],
                "source_url": pub["source_url"],
                "gap_ours_minus_published": gap,
                "note": NOTES.get(full, ""),
            })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} rows to {OUT_CSV}")
    for r in rows:
        print(f"  {r['method']}: ours={r['our_reimpl_acc_mean']} published={r['published_acc']} "
              f"gap={r['gap_ours_minus_published']} ({r['source_paper']})")

    config = {"seed": 0, "purpose": "TAB07 reproduction gap"}
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [OUT_CSV])
    return 0


if __name__ == "__main__":
    sys.exit(main())
