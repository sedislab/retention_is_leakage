from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_fx2_decoupling_ratio.py"
    spec = importlib.util.spec_from_file_location("build_fx2_decoupling_ratio", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m():
    return _load_module()


def _write_halflife_csv(path, rows):
    fieldnames = ["dataset", "method", "family", "view", "quantity", "ablation", "halflife", "halflife_int", "status", "ci_lo", "ci_hi"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({**{k: "" for k in fieldnames}, **r})


def test_build_decoupling_ratio_point_case(m, tmp_path, monkeypatch):
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    _write_halflife_csv(tmp_path / "results" / "fig02_halflife.csv", [
        {"dataset": "cifar100", "method": "m1_glfc", "family": "F1", "view": "full",
         "quantity": "leak_tpr1", "ablation": "trajectory", "halflife": 1.5, "halflife_int": 2, "status": "ok"},
        {"dataset": "cifar100", "method": "m1_glfc", "family": "F1", "view": "full",
         "quantity": "acc", "ablation": "n/a", "halflife": 4.5, "halflife_int": 5, "status": "ok"},
    ])
    row = m.build_decoupling_ratio("cifar100", "m1_glfc", "F1", "full")
    assert row["h_leak"] == pytest.approx(1.5)
    assert row["h_acc"] == pytest.approx(4.5)
    assert row["ratio"] == pytest.approx(1.5 / 4.5)
    assert row["ratio_type"] == "point"
    assert row["ratio_ci_lo"] == "" and row["ratio_ci_hi"] == ""


def test_build_decoupling_ratio_by_construction_flag(m, tmp_path, monkeypatch):
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    _write_halflife_csv(tmp_path / "results" / "fig02_halflife.csv", [
        {"dataset": "cifar100", "method": "m4_proto", "family": "F2", "view": "full",
         "quantity": "leak_tpr1", "ablation": "trajectory", "halflife": 6.0, "halflife_int": "", "status": "censored"},
        {"dataset": "cifar100", "method": "m4_proto", "family": "F2", "view": "full",
         "quantity": "acc", "ablation": "n/a", "halflife": 6.0, "halflife_int": "", "status": "censored"},
    ])
    row = m.build_decoupling_ratio("cifar100", "m4_proto", "F2", "full", by_construction=True)
    assert row["ratio_type"] == "by_construction"
    assert row["ratio"] is None


def test_build_decoupling_ratio_raises_when_a_row_is_missing(m, tmp_path, monkeypatch):
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    _write_halflife_csv(tmp_path / "results" / "fig02_halflife.csv", [
        {"dataset": "cifar100", "method": "m1_glfc", "family": "F1", "view": "full",
         "quantity": "leak_tpr1", "ablation": "trajectory", "halflife": 1.5, "halflife_int": 2, "status": "ok"},
    ])
    with pytest.raises(ValueError):
        m.build_decoupling_ratio("cifar100", "m1_glfc", "F1", "full")
