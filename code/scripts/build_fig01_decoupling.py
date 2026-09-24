#!/usr/bin/env python3
"""Compatibility CSV for consistency checks, copied only from retention_curves.csv."""
import csv
from pathlib import Path

from p3fcl import provenance

ROOT = Path(__file__).resolve().parents[2]


def main():
    with (ROOT/'results/retention_curves.csv').open() as f:
        source = list(csv.DictReader(f))
    grouped = {}
    for r in source:
        key = tuple(r[k] for k in ('dataset', 'method', 'family', 'view', 'elapsed'))
        grouped.setdefault(key, {})[r['quantity']] = r
    rows = []
    for key, vals in sorted(grouped.items()):
        r = dict(zip(('dataset', 'method', 'family', 'view', 'elapsed'), key))
        for quantity, prefix in [('acc', 'acc'), ('leak_tpr1', 'tpr1'), ('leak_auc', 'auc')]:
            r.update({f'{prefix}_mean': vals[quantity]['raw'], f'{prefix}_ci_lo': vals[quantity]['raw_ci_lo'],
                      f'{prefix}_ci_hi': vals[quantity]['raw_ci_hi'], f'{prefix}_norm': vals[quantity]['norm']})
        r['leak_norm'] = vals['leak_tpr1']['norm']
        rows.append(r)
    out = ROOT/'results/fig01_decoupling.csv'
    manifest = provenance.run_manifest(dict(phase='FX9-1', hypothesis='H2', source='retention_curves.csv'), seed=0)
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    provenance.finalize(manifest, [out])
    print(f'FIG01 rows={len(rows)}')


if __name__ == '__main__':
    main()
