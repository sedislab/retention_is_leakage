from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_module(name: str, rel_path: str):
    path = Path(__file__).resolve().parents[1] / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def m():
    # Fresh module per test (not module-scoped): each test monkeypatches module globals
    # (REPO_ROOT, DATASET, EPS_LIST, ...) and those must not leak across tests.
    return _load_module("run_m9_audit", "scripts/run_m9_audit.py")


def _make_synthetic_features(n_classes, per_class, d, seed):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_classes, d)) * 5
    X, y = [], []
    for c in range(n_classes):
        X.append(centers[c] + rng.standard_normal((per_class, d)) * 0.4)
        y.append(np.full(per_class, c))
    return np.concatenate(X), np.concatenate(y)


def _write_hparam_table(path, dataset, rows):
    fields = ["dataset", "unit", "eps", "p", "lambda", "clip_C", "ref_val_final_avg_acc", "selected"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for eps, p, lam, clip_C in rows:
            w.writerow({
                "dataset": dataset, "unit": "U1", "eps": eps, "p": p, "lambda": lam,
                "clip_C": clip_C, "ref_val_final_avg_acc": 0.5, "selected": 1,
            })


def test_generate_and_score_end_to_end_on_synthetic_data(tmp_path, monkeypatch, m):
    """The real integration risk in `run_m9_audit.py` is wiring -- CSV parsing, PCA fitting,
    threading `method_config_override` into `shadow_runner.run_shadow_range`, and the online-LiRA
    scoring against the `global` view -- not the individual pieces, which already have their own
    unit tests (`test_shadow_runner.py`'s M9 tests, `run_lira.py`'s own tests). Runs the WHOLE
    `--shadows` then `--score` pipeline on tiny synthetic data end to end and checks it produces a
    sane, non-crashing CSV with the expected DP-bound-comparison columns."""
    from p3fcl import features

    d = 8
    X_ref, y_ref = _make_synthetic_features(n_classes=10, per_class=30, d=d, seed=0)
    X_train, y_train = _make_synthetic_features(n_classes=10, per_class=60, d=d, seed=1)
    features_dir = tmp_path / "features"
    features.save_cache(features_dir, "synthdset", "synthbb", "ref", X_ref, X_ref, y_ref, np.arange(len(y_ref)))
    features.save_cache(features_dir, "synthdset", "synthbb", "train", X_train, X_train, y_train, np.arange(len(y_train)))

    _write_hparam_table(
        tmp_path / "results" / "m9_hparam_selection.csv", "synthdset",
        rows=[(1.0, 4, 1.0, 5.0), (4.0, 4, 1.0, 5.0), (float("inf"), 4, 0.1, 5.0)],
    )

    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(m, "DATASET", "synthdset")
    monkeypatch.setattr(m, "BACKBONE", "synthbb")
    monkeypatch.setattr(m, "N_SHADOWS", 12)
    monkeypatch.setattr(m, "EPS_LIST", [1.0, 4.0, float("inf")])
    monkeypatch.setattr(m.shadow_runner, "FEATURES_DIR", features_dir)
    monkeypatch.setattr(m.shadow_runner, "BACKBONE", "synthbb")

    m.generate_shadows(workers=1)
    for eps in m.EPS_LIST:
        shadow_dir = m._shadow_dir(eps)
        assert shadow_dir.exists()
        npz_files = list(shadow_dir.glob("shadow_*.npz"))
        assert len(npz_files) == 12

    exit_code = m.score_audit()
    out_csv = tmp_path / "results" / "fig10_m9_audit.csv"
    assert out_csv.exists()
    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    epsilons_present = {row["eps"] for row in rows}
    assert epsilons_present == {"1.0", "4.0", "inf"}
    for row in rows:
        assert 0.0 <= float(row["tpr_empirical"]) <= 1.0
        assert 0.0 <= float(row["tpr_dp_bound"]) <= 1.0
    # exit_code is 0 (passed) or 2 (a real bound violation found) -- both are legitimate outcomes on
    # tiny random synthetic data with no real privacy story; only crashing (any other path) is a bug.
    assert exit_code in (0, 2)


def test_hparams_for_reads_the_selected_row_only(tmp_path, monkeypatch, m):
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(m, "DATASET", "synthdset")
    path = tmp_path / "results" / "m9_hparam_selection.csv"
    path.parent.mkdir(parents=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "unit", "eps", "p", "lambda", "clip_C", "selected"])
        w.writeheader()
        w.writerow({"dataset": "synthdset", "unit": "U1", "eps": 1.0, "p": 8, "lambda": 10.0, "clip_C": 3.0, "selected": 0})
        w.writerow({"dataset": "synthdset", "unit": "U1", "eps": 1.0, "p": 16, "lambda": 1.0, "clip_C": 4.0, "selected": 1})
    p, lam, clip_C = m._hparams_for(1.0)
    assert (p, lam, clip_C) == (16, 1.0, 4.0)


def test_hparams_for_raises_when_no_selected_row_matches(tmp_path, monkeypatch, m):
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(m, "DATASET", "synthdset")
    path = tmp_path / "results" / "m9_hparam_selection.csv"
    path.parent.mkdir(parents=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "unit", "eps", "p", "lambda", "clip_C", "selected"])
        w.writeheader()
    with pytest.raises(ValueError):
        m._hparams_for(1.0)


def test_eps_tag_formats_inf_and_integers():
    m = _load_module("run_m9_audit", "scripts/run_m9_audit.py")
    assert m._eps_tag(float("inf")) == "inf"
    assert m._eps_tag(1.0) == "1"
    assert m._eps_tag(4.0) == "4"
