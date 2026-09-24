#!/usr/bin/env python3
"""TAB10 — artifact family taxonomy F1-F8 (RESEARCH_PLAN.md §2.2; marked "required, conceptual" in
build/03_RESULTS_SPEC.md — no new experiments needed, only careful transcription + this project's own
`cacheable` determination). Content is taken directly from RESEARCH_PLAN.md §2.2's own table (the
project's source of truth for the taxonomy) plus this codebase's `methods/*.py` `MethodSpec.cacheable`
declarations for the "cacheable in this reimplementation" column, which is a genuinely new column not
in the original prose table -- it reflects how THIS project implemented each family (frozen ViT
features vs. real backprop), not necessarily how the cited published method did it.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl.plotting import latex_escape  # noqa: E402

# (family_id, name, payload_notation, example_published_methods, sufficient_statistic_content,
#  a_priori_risk_per_RESEARCH_PLAN, cacheable_in_this_project, methods_in_this_zoo)
ROWS = [
    ("F1", "Model deltas", "Δθ", "FedAvg, GLFC, FOT",
     "Full gradient signal", "High (known); this project's baseline",
     "Yes (M0/M1/M2/M3/M5 all train only a frozen-ViT linear head in this reimplementation)",
     "M0, M1, M2, M3, M5"),
    ("F2", "Class prototypes / feature moments", "mu_c,y, Sigma_c,y", "PILoRA, HGP, STSA, pFedMxF",
     "Mean (and covariance) of the client's features for class y",
     "High -- a mean over n samples is a linear query; with small n it is nearly a sample",
     "Yes", "M4 (prototype half)"),
    ("F3", "Prompt pools + keys", "P, k_P", "C2Prompt, Fed-CPrompt, TPGP",
     "Learned input-space perturbations optimized on local data",
     "High (shown centrally exploitable, USENIX'24)",
     "No -- needs real backprop through the frozen ViT (M6, not built to completion this pass)",
     "M6 (not built to completion)"),
    ("F4", "Low-rank adapters", "A, B", "PILoRA, DOLFIN, FCIT-style",
     "Rank-r update subspace",
     "Medium-High; row space of updates",
     "No -- needs real backprop (M4's LoRA half, not built)",
     "M4-LoRA (not built)"),
    ("F5", "Analytic Gram / autocorrelation", "R_c = X^T X + lambda I, Q_c = X^T Y",
     "FedRAN and the analytic-FCL line",
     "An exact sufficient statistic of the entire local feature matrix (RESEARCH_PLAN.md §3.4)",
     "Critical -- see H5's real result: n=1 reconstruction is exact",
     "Yes", "M3, M8, M9"),
    ("F6", "Generative replay / synthetic data", "G", "TARGET, FedCIL, FedER",
     "A generative model fitted to client data",
     "Critical (the model *is* the leak)",
     "Yes (M2's per-class diagonal-Gaussian generator, on frozen features)",
     "M2"),
    ("F7", "Class-distribution / count vectors", "n_c,y", "C2Prompt's compensation, GLFC",
     "Exact per-class counts per client per task",
     "Medium alone; high as a join key -- deanonymizes and reveals events (A6's H10 test)",
     "Yes", "M1, M4 (counts)"),
    ("F8", "Exemplar buffers", "E_c", "Potential raw-buffer release (absent from FX9)",
     "Literal raw samples",
     "Trivially critical if released; M1/M5 keep buffers private in FX9",
     "Private state only", "None released (M1/M5 buffers are private)"),
]


def main() -> int:
    out_dir = REPO_ROOT / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "family_id", "name", "payload_notation", "example_published_methods",
        "sufficient_statistic_content", "a_priori_risk", "cacheable_in_this_project", "methods_in_this_zoo",
    ]
    csv_path = out_dir / "tab10_taxonomy.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        w.writerows(ROWS)

    tex_lines = [r"\begin{tabular}{llp{3.5cm}p{3cm}l}", r"\toprule",
                 r"Family & Name & Sufficient-statistic content & A-priori risk & In this zoo \\", r"\midrule"]
    for r in ROWS:
        tex_lines.append(
            f"{latex_escape(r[0])} & {latex_escape(r[1])} & {latex_escape(r[4])} & "
            f"{latex_escape(r[5])} & {latex_escape(r[7])} \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "tab10_taxonomy.tex").write_text("\n".join(tex_lines) + "\n")

    print(f"wrote {len(ROWS)} rows to {csv_path} and tab10_taxonomy.tex")
    from p3fcl import provenance
    manifest = provenance.run_manifest(dict(phase="FX9-10", table="tab10_taxonomy", source="code definitions"), seed=0)
    provenance.finalize(manifest, [out_dir/"tab10_taxonomy.csv", out_dir/"tab10_taxonomy.tex"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
