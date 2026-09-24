import csv
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("build_fx9_joint"), importlib.import_module("check_fx9_consistency")


def test_joint_draw_uses_identical_seed_and_task_for_both_quantities(monkeypatch):
    joint, _ = scripts(monkeypatch)
    # Mock the statistic to isolate the sampling contract from ROC arithmetic.
    monkeypatch.setattr(joint.bootstrap_metrics, "tpr_at_fpr", lambda scores, labels, fpr: float(np.mean(scores)))
    sample = np.arange(2 * 4 * 7).reshape(2, 4, 7) / 100 + 0.1
    data = []
    for seed in range(2):
        k = np.repeat(np.arange(4), 3)
        data.append(
            dict(
                k=k,
                labels=np.zeros((2, 12)),
                scores={e: np.tile(sample[seed, k, e], (2, 1)) for e in range(7)},
            )
        )
    acc = SimpleNamespace(samples=sample, floors=np.zeros_like(sample))
    rng = np.random.default_rng(19)
    for _ in range(20):
        draws = joint.draw_indices(data, rng)
        raw, _, leak, _ = joint.statistic(data, acc, draws, [0, 6])
        assert raw[0] == pytest.approx(leak[0])
        assert raw[6] == pytest.approx(leak[6])


def test_paper_checker_rejects_wrong_cell_or_ambiguous_filter(tmp_path, monkeypatch):
    _, checker = scripts(monkeypatch)
    path = tmp_path / "source.csv"
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "value", "ci_lo"])
        w.writeheader()
        w.writerows([dict(dataset="a", value=".25", ci_lo=".1"), dict(dataset="b", value=".5", ci_lo=".4")])
    row = dict(
        key="test",
        source_csv="source.csv",
        row_filter=json.dumps({"dataset": "a"}),
        value=".25",
        value_column="value",
        ci_lo=".1",
        ci_lo_column="ci_lo",
        ci_hi="",
        ci_hi_column="",
    )
    checker.check_paper_rows([row], root=tmp_path)
    with pytest.raises(AssertionError):
        checker.check_paper_rows([{**row, "value": ".5"}], root=tmp_path)
    with pytest.raises(AssertionError):
        checker.check_paper_rows([{**row, "row_filter": "{}"}], root=tmp_path)


def test_constant_leakage_does_not_get_spurious_zero_width_halflife_ci(monkeypatch):
    scripts(monkeypatch)
    summary=importlib.import_module('build_fx2_summary')
    monkeypatch.setattr(summary,'N_REPLICATES',12)
    labels=np.tile(np.repeat([0,1],10)[:,None],(1,4))
    surf=np.repeat(labels[:,:,None],10,axis=2).astype(float)
    data=dict(k_of_target=np.arange(4),in_out=labels,surf_trajectory=surf,surf_last_round=surf)
    monkeypatch.setattr(summary,'_load_seed_npz',lambda *args:data)
    _,halves=summary.build_summary('toy','toy','F1','full',[0,1,2],ablations=('trajectory',))
    assert all(r['status']=='censored' and r['halflife']==6 for r in halves)
    assert all(r['ci_lo'] is None and r['ci_hi'] is None for r in halves)
