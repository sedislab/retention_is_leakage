#!/usr/bin/env python3
"""FX9 accuracy half-lives and curves from the single halflife.accuracy_curve API."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from p3fcl import provenance
from p3fcl.halflife import bootstrap_accuracy_curve, half_life, has_signal, no_signal_result

REPO_ROOT = Path(__file__).resolve().parents[2]
K_SET = [0, 1, 2, 3]
E_MAX = 6
N_REPLICATES = 2000
BOOTSTRAP_SEED = 0


def load_rows(dataset, method, seeds):
    with (REPO_ROOT / 'results' / f'accuracy_matrix_{dataset}.csv').open() as f:
        rows = [r for r in csv.DictReader(f) if r['method'] == method and int(r['seed']) in seeds]
    if {int(r['seed']) for r in rows} != set(seeds):
        raise ValueError(f'{dataset}/{method}: missing accuracy seeds')
    return rows


def build_accuracy_outputs(dataset, method, family, view, seeds):
    point, boot = bootstrap_accuracy_curve(load_rows(dataset, method, seeds), K=K_SET, E=E_MAX,
                                          n_replicates=N_REPLICATES, seed=BOOTSTRAP_SEED)
    base_ci_lo = float(np.percentile(boot['excess'][:, 0], 2.5))
    ci_lo = ci_hi = None
    if has_signal(point.excess[0], base_ci_lo, 0, 0.02):
        hl = half_life(dict(enumerate(point.norm)), E=E_MAX)
        draws = [half_life(dict(enumerate(v)), E=E_MAX) for v in boot['norm'] if np.isfinite(v).all()]
        valid = [d['halflife'] for d in draws if d['status'] == 'ok']
        if len(valid) >= N_REPLICATES / 2:
            ci_lo, ci_hi = map(float, np.percentile(valid, [2.5, 97.5]))
    else:
        hl = no_signal_result()
    common = dict(dataset=dataset, method=method, family=family, view=view, quantity='acc')
    halflives = [{**common, 'ablation': 'n/a', **hl, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
                  'horizon_E': E_MAX, 'base_value': point.excess[0], 'base_ci_lo': base_ci_lo,
                  'floor': 'per-task a0(t)=1/classes_seen', 'halflife_expfit': '', 'r2': ''}]
    raw_ci = np.percentile(boot['raw'], [2.5, 97.5], axis=0)
    norm_ci = np.nanpercentile(boot['norm'], [2.5, 97.5], axis=0)
    curves = [{**common, 'elapsed': e, 'raw': point.raw[e], 'raw_ci_lo': raw_ci[0, e],
               'raw_ci_hi': raw_ci[1, e], 'norm': point.norm[e], 'norm_ci_lo': norm_ci[0, e],
               'norm_ci_hi': norm_ci[1, e], 'n_seeds': len(seeds)} for e in range(E_MAX + 1)]
    return halflives, curves


def build_accuracy_halflife(dataset, method, family, view, seeds):
    return build_accuracy_outputs(dataset, method, family, view, seeds)[0]


def main():
    dataset, method, family = sys.argv[1:4]
    view = sys.argv[4] if len(sys.argv) > 4 else 'full'
    seeds = list(map(int, sys.argv[5:])) if len(sys.argv) > 5 else list(range(5))
    manifest = provenance.run_manifest(dict(phase='FX9-1', hypothesis='H2', dataset=dataset,
                                           method=method, view=view, seeds=seeds), seed=0)
    halflives, curves = build_accuracy_outputs(dataset, method, family, view, seeds)
    paths = []
    for name, rows in [('fig02_halflife_acc', halflives), ('retention_acc', curves)]:
        path = REPO_ROOT / 'results' / f'{name}_{dataset}_{method}_{view}.csv'
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        paths.append(path)
    provenance.finalize(manifest, paths)
    print(f'{dataset}/{method}/{view}: raw_e0={curves[0]["raw"]:.12f} raw_e6={curves[-1]["raw"]:.12f} h={halflives[0]["halflife"]}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
