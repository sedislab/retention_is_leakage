from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_m9_sweep.py"
    spec = importlib.util.spec_from_file_location("run_m9_sweep", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m():
    return _load_module()


def test_load_hparam_table_keeps_only_selected_rows_for_the_dataset(m, tmp_path, monkeypatch):
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    rows = [
        {"dataset": "cifar100", "unit": "U1", "eps": "1.0", "p": "64", "lambda": "1.0", "clip_C": "0.5", "ref_val_final_avg_acc": "0.1", "selected": "1"},
        {"dataset": "cifar100", "unit": "U1", "eps": "1.0", "p": "32", "lambda": "10.0", "clip_C": "0.5", "ref_val_final_avg_acc": "0.05", "selected": "0"},
        {"dataset": "cub200", "unit": "U1", "eps": "1.0", "p": "128", "lambda": "1.0", "clip_C": "0.5", "ref_val_final_avg_acc": "0.2", "selected": "1"},
    ]
    with open(tmp_path / "results" / "m9_hparam_selection.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    table = m._load_hparam_table("cifar100")
    assert table == {("U1", 1.0): (64, 1.0, 0.5)}


def test_pca_cache_fits_each_p_only_once(m, monkeypatch):
    rng = np.random.default_rng(0)
    X_ref = rng.standard_normal((50, 20))
    calls = []
    real_fit_pca = m._fit_pca

    def counting_fit_pca(X, p_max, seed=0):
        calls.append(p_max)
        return real_fit_pca(X, p_max, seed)

    monkeypatch.setattr(m, "_fit_pca", counting_fit_pca)
    get = m._pca_cache("cifar100", X_ref)
    p8_a = get(8)
    p8_b = get(8)
    p4 = get(4)
    assert calls == [8, 4]  # second get(8) reused the cache, not refit
    np.testing.assert_array_equal(p8_a, p8_b)
    assert p8_a.shape == (20, 8)
    assert p4.shape == (20, 4)
