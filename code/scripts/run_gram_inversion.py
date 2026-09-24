#!/usr/bin/env python3
"""FIG13 (H5's deciding test): Gram-inversion cosine similarity vs per-class n, per reference
quality, per dataset -- a curve, not a single threshold (re-specified per
`notes/2026-08-25_preliminary.md`'s toy-scale finding). Runs entirely on cached real ViT features, no
GPU, no shadow models (04_METHODS_AND_ATTACKS.md's own stated order: settle this first, it's cheap).

For each dataset, each class, each n in a grid: draw n real training-split feature vectors of that
class, form the exact class-conditional Gram statistic `R = X^T X + lambda*I` (exactly what M8/the
analytic-FCL line releases per RESEARCH_PLAN.md §3.4), invert it with the realistic
(reference-based, not oracle) adversary from `attacks/inversion.py`, using the `ref` split (target-
disjoint, per `02_DATASETS.md §7`) as the public reference pool at a chosen "quality" level. Reference
quality is varied two ways: pool size (more candidates competing for a match) and injected Gaussian
noise (degraded reference fidelity) -- both named explicitly in the CSV so a reader can tell which
knob moved. Also records per-dataset feature anisotropy (RESEARCH_PLAN.md's stated hypothesis: more
anisotropic/class-clustered features should make rotation resolution easier).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import features, provenance  # noqa: E402
from p3fcl import rng as rng_mod  # noqa: E402
from p3fcl.attacks.inversion import (  # noqa: E402
    canonical_recovery,
    cosine_similarity_rows,
    feature_anisotropy,
    gram_to_subspace,
    reference_based_recovery,
)

BACKBONE = "vit_base_patch16_224.augreg_in21k"
RIDGE_LAMBDA = 1.0
N_GRID = [1, 2, 4, 8, 16, 32, 64, 128]
REF_QUALITIES = [
    {"name": "clean_full", "noise_std": 0.0, "pool_cap": None},
    {"name": "clean_small_pool", "noise_std": 0.0, "pool_cap": 20},
    {"name": "noisy_full", "noise_std": 0.5, "pool_cap": None},
]
N_TRIALS_PER_N = 25  # repeat draws per (dataset, class, n, ref_quality) for a CI-able spread


def main() -> int:
    datasets = sys.argv[1:] or ["cub200", "cifar100"]
    rows = []
    anisotropy_rows = []

    for dataset in datasets:
        train = features.load_cache(REPO_ROOT / "features", dataset, BACKBONE, "train")
        ref = features.load_cache(REPO_ROOT / "features", dataset, BACKBONE, "ref")
        X_train, y_train = train["features"], train["labels"]
        X_ref, y_ref = ref["features"], ref["labels"]
        d = X_train.shape[1]

        aniso = feature_anisotropy(X_train)
        anisotropy_rows.append({"dataset": dataset, "backbone": BACKBONE, "anisotropy_ratio": aniso})
        print(f"{dataset}: feature anisotropy ratio = {aniso:.3f}")

        classes = sorted(set(y_train.tolist()))
        for n in N_GRID:
            eligible = [c for c in classes if np.sum(y_train == c) >= n and np.any(y_ref == c)]
            if not eligible:
                print(f"{dataset}: n={n} unavailable (no class has enough training samples)", flush=True)
                continue
            for rq in REF_QUALITIES:
                for trial in range(N_TRIALS_PER_N):
                    seed = 0  # stable named RNG includes dataset, n, quality, trial; no process-randomized hash
                    r = rng_mod.seeded(f"run_gram_inversion::{dataset}::{n}::{rq['name']}::{trial}", seed)
                    c = eligible[r.integers(0, len(eligible))]

                    class_idx_train = np.where(y_train == c)[0]
                    if len(class_idx_train) < n:
                        continue
                    chosen = class_idx_train[r.permutation(len(class_idx_train))[:n]]
                    X = X_train[chosen]
                    R = X.T @ X + RIDGE_LAMBDA * np.eye(d)

                    class_idx_ref = np.where(y_ref == c)[0]
                    if len(class_idx_ref) == 0:
                        continue
                    ref_pool = X_ref[class_idx_ref].copy()
                    if rq["pool_cap"] is not None and len(ref_pool) > rq["pool_cap"]:
                        ref_pool = ref_pool[r.permutation(len(ref_pool))[: rq["pool_cap"]]]
                    if rq["noise_std"] > 0:
                        ref_pool = ref_pool + r.standard_normal(ref_pool.shape) * rq["noise_std"] * ref_pool.std()

                    top_vecs, sv = gram_to_subspace(R, RIDGE_LAMBDA, n)
                    recovered = canonical_recovery(top_vecs, sv)
                    rotated = reference_based_recovery(recovered, ref_pool, seed=seed)
                    cos = cosine_similarity_rows(rotated, X)

                    rows.append({
                        "dataset": dataset, "backbone": BACKBONE, "n_per_class": n,
                        "ref_quality": rq["name"], "class": int(c), "trial": trial,
                        "anisotropy_ratio": aniso,
                        "mean_cos": float(np.mean(cos)), "median_cos": float(np.median(cos)),
                        "frac_above_0.8": float(np.mean(cos > 0.8)),
                        "n_records": n, "ref_pool_size": len(ref_pool),
                    })
        print(f"{dataset}: {len([r for r in rows if r['dataset'] == dataset])} rows so far")

    out_csv = REPO_ROOT / "results" / "fig13_gram_inversion.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")

    aniso_csv = REPO_ROOT / "results" / "fig13_anisotropy.csv"
    with open(aniso_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(anisotropy_rows[0].keys()))
        w.writeheader()
        w.writerows(anisotropy_rows)

    config = {"seed": 0, "purpose": "FIG13 Gram inversion n-curve (H5)", "datasets": datasets}
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv, aniso_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
