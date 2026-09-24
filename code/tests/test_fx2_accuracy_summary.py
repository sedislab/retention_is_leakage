"""Regression checks for the FX9 shared accuracy curve and its bootstrap."""
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from p3fcl.halflife import accuracy_curve, bootstrap_accuracy_curve, half_life

ROOT = Path(__file__).resolve().parents[2]


def toy_rows():
    return [dict(dataset='toy', method='m0', seed=s, task_k=k, elapsed=e,
                 task_T=k+e, n_classes_seen=2*(k+e+1), acc=0.9-0.05*k-0.1*e)
            for s in range(3) for k in range(4) for e in range(7)]


def test_elapsed_lookup_and_chance_floor():
    rows = toy_rows()
    curve = accuracy_curve(rows)
    assert curve.raw[0] == pytest.approx(0.825)
    assert curve.raw[6] == pytest.approx(0.225)
    base = np.mean([0.9-0.05*k-1/(2*(k+1)) for k in range(4)])
    last = np.mean([0.3-0.05*k-1/(2*(k+7)) for k in range(4)])
    assert curve.norm[0] == 1.0
    assert curve.norm[6] == pytest.approx(last/base)


def test_missing_or_duplicate_cells_fail():
    with pytest.raises(ValueError, match='missing accuracy cell'):
        accuracy_curve(toy_rows()[1:])
    with pytest.raises(ValueError, match='duplicate accuracy cell'):
        accuracy_curve(toy_rows()+toy_rows()[:1])


def test_bootstrap_keeps_whole_trajectory_and_normalizes_each_draw():
    _, boot = bootstrap_accuracy_curve(toy_rows(), n_replicates=100)
    assert np.all(boot['norm'][:, 0] == 1)
    np.testing.assert_allclose(boot['raw'][:, 0]-boot['raw'][:, 6], 0.6)
    assert boot['raw'][:, 0].std() > 0


def test_real_m0_diagonal_and_existing_half_life():
    if not (ROOT/'results/accuracy_matrix_cifar100.csv').exists():
        pytest.skip('real matrices unavailable')
    with (ROOT/'results/fig02_halflife.csv').open() as f:
        old = list(csv.DictReader(f))
    for ds in ['cifar100', 'cub200', 'imagenet_r']:
        with (ROOT/f'results/accuracy_matrix_{ds}.csv').open() as f:
            matrix = list(csv.DictReader(f))
        for method in sorted({r['method'] for r in matrix}):
            rows = [r for r in matrix if r['method'] == method]
            curve = accuracy_curve(rows)
            diag = [float(r['acc']) for r in rows if int(r['task_k']) < 4 and int(r['elapsed']) == 0]
            assert curve.raw[0] == pytest.approx(np.mean(diag), abs=1e-12)
            assert curve.norm[0] == 1.0
            ref = [r for r in old if r['dataset'] == ds and r['method'] == method and r['quantity'] == 'acc']
            for r in ref:
                if r['status'] != 'no_signal':
                    assert half_life(dict(enumerate(curve.norm)))['halflife'] == pytest.approx(float(r['halflife']), abs=1e-9)
            if ds == 'cifar100' and method == 'm0_fedavg':
                assert curve.raw[0] == pytest.approx(0.93, abs=0.01)


def test_accuracy_summary_uses_shared_curve():
    spec = importlib.util.spec_from_file_location('acc_summary', ROOT/'code/scripts/build_fx2_accuracy_summary.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = mod.build_accuracy_halflife('cifar100', 'm0_fedavg', 'F1', 'full', list(range(5)))
    assert rows[0]['status'] == 'ok'
    assert 0 < rows[0]['halflife'] < 2
    assert rows[0]['ci_lo'] <= rows[0]['halflife'] <= rows[0]['ci_hi']
