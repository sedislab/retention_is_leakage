#!/usr/bin/env python3
"""Add the requested empirical bound; preserve every existing M9 audit estimate."""

import csv
from pathlib import Path

import numpy as np
import run_lira
from fx9_io import merge, write
from p3fcl import metrics
from p3fcl import rng as rng_mod
from p3fcl.audit.empirical import epsilon_lower_bound
from run_m9_audit import CALIB_FRAC, CALIB_SEED, _eps_tag, _shadow_dir

ROOT = Path(__file__).resolve().parents[2]


def main():
    path = ROOT / "results/m9_audit_summary.csv"
    with path.open() as f:
        rows = list(csv.DictReader(f))
    paths = []
    for row in rows:
        eps = float(row["eps"])
        store = run_lira.load_shadow_store(_shadow_dir(eps), view="global")
        r = rng_mod.seeded(f"run_m9_audit::calib_split::cifar100::eps{_eps_tag(eps)}", CALIB_SEED)
        perm = r.permutation(len(store["shadow_ids"]))
        n = int(round(CALIB_FRAC * len(perm)))
        surfaces = run_lira.compute_log_lr_surfaces(store, perm[:n])
        surf = surfaces["trajectory"][perm[n:]]
        labs = store["in_out"][perm[n:]]
        scores, labels = [], []
        for j, t in enumerate(store["targets"]):
            col = surf[:, j, t["task"]]
            keep = np.isfinite(col)
            scores.extend(col[keep])
            labels.extend(labs[:, j][keep])
        report = metrics.membership_report(np.asarray(scores), np.asarray(labels))
        assert abs(report["auc"] - float(row["auc"])) < 1e-12
        assert abs(report["tpr_at_1pct_fpr"] - float(row["tpr1"])) < 1e-12
        delta = float(row["delta"])
        b1 = min(np.exp(eps) * 0.01 + delta, 1.0) if np.isfinite(eps) else 1.0
        b01 = min(np.exp(eps) * 0.001 + delta, 1.0) if np.isfinite(eps) else 1.0
        failed = any(
            report[key] > bound + 1e-6 and report[ci] > bound + 1e-9
            for key, ci, bound in [
                ("tpr_at_1pct_fpr", "tpr_at_1pct_fpr_ci_hi", b1),
                ("tpr_at_0.1pct_fpr", "tpr_at_0.1pct_fpr_ci_hi", b01),
            ]
        )
        fresh = dict(
            dataset="cifar100",
            unit="U1",
            eps=eps,
            gamma=1.0,
            delta=delta,
            n_shadows=len(store["shadow_ids"]),
            n_pos=int(np.sum(np.asarray(labels) == 1)),
            n_neg=int(np.sum(np.asarray(labels) == 0)),
            auc=report["auc"],
            tpr1=report["tpr_at_1pct_fpr"],
            tpr1_ci_hi=report["tpr_at_1pct_fpr_ci_hi"],
            bound_at_1pct_fpr=b1,
            tpr01=report["tpr_at_0.1pct_fpr"],
            tpr01_ci_hi=report["tpr_at_0.1pct_fpr_ci_hi"],
            **{"bound_at_0.1pct_fpr": b01},
            audit_passed=int(not failed),
        )
        for key, value in fresh.items():
            assert str(value) == row[key] or (
                isinstance(value, (int, float)) and abs(value - float(row[key])) < 1e-12
            ), (key, value, row[key])
        row = {**fresh, **epsilon_lower_bound(scores, labels, delta=delta)}
        output = ROOT / f"results/fx9_m9_audit_eps{_eps_tag(eps)}.csv"
        write(output, [row], dict(hypothesis="H7", purpose="preserved audit scores, unchanged estimates"))
        paths.append(output)
        print(
            f'FX9-7 AUDIT ACCEPT eps={eps} eps_lb={row["eps_lb"]:.12f} old_auc_error=0 old_tpr1_error=0 thresholds={row["thresholds"]}',
            flush=True,
        )
    merge(paths, path, ["dataset", "unit", "eps"], 3)


if __name__ == "__main__":
    main()
