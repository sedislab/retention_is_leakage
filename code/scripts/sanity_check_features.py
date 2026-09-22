#!/usr/bin/env python3
"""Feature-cache sanity check (build/00_BUILD_PLAN.md P1 task 5): linear probe on the cached
train split, top-1 on the cached test split. Appends one row to `results/tab01_feature_sanity.csv`
per (dataset, backbone), with a provenance-stamped run.

If CIFAR-100 with `vit_base_patch16_224.augreg_in21k` does not land in the high 80s / low 90s, the
extraction is wrong — stop and debug rather than continuing to P2. This script's job is to make that
judgment call legible, not to make it silently.

Usage: python scripts/sanity_check_features.py --dataset cifar100 --backbone vit_base_patch16_224.augreg_in21k
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import features, provenance  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--features-dir", default=str(REPO_ROOT / "features"))
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression

    train = features.load_cache(args.features_dir, args.dataset, args.backbone, "train")
    test = features.load_cache(args.features_dir, args.dataset, args.backbone, "test")

    clf = LogisticRegression(max_iter=2000, C=1.0)  # multinomial is the default for lbfgs since sklearn 1.5
    clf.fit(train["features"], train["labels"])
    top1 = float(clf.score(test["features"], test["labels"]))

    print(f"{args.dataset} | {args.backbone}: linear-probe top-1 = {top1:.4f} "
          f"(n_train={len(train['labels'])}, n_test={len(test['labels'])}, d={train['features'].shape[1]})")

    out_csv = REPO_ROOT / "results" / "tab01_feature_sanity.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["dataset", "backbone", "n_train", "n_test", "feature_dim", "top1_acc"])
        w.writerow([args.dataset, args.backbone, len(train["labels"]), len(test["labels"]),
                    train["features"].shape[1], f"{top1:.6f}"])

    config = {"seed": 0, "dataset": args.dataset, "backbone": args.backbone, "probe": "logistic_regression"}
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
