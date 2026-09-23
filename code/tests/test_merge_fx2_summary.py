from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "merge_fx2_summary.py"
    spec = importlib.util.spec_from_file_location("merge_fx2_summary", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m():
    return _load_module()


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_merge_combines_shared_and_per_combo_files_with_dedup(m, tmp_path, monkeypatch):
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    (tmp_path / "results").mkdir()

    # Shared file: M0's accuracy row (written directly, not per-combo).
    _write_csv(tmp_path / "results" / "fig02_halflife.csv", [
        {"dataset": "cifar100", "method": "m0_fedavg", "family": "F1", "view": "full",
         "quantity": "acc", "ablation": "n/a", "halflife": "0.87"},
    ])
    # Two per-combo files (as fx2_leak_wave.pbs's array would produce).
    _write_csv(tmp_path / "results" / "fig02_halflife_cifar100_m1_glfc_full.csv", [
        {"dataset": "cifar100", "method": "m1_glfc", "family": "F1", "view": "full",
         "quantity": "leak_tpr1", "ablation": "trajectory", "halflife": "6.0"},
    ])
    _write_csv(tmp_path / "results" / "fig02_halflife_cub200_m4_proto_global.csv", [
        {"dataset": "cub200", "method": "m4_proto", "family": "F2", "view": "global",
         "quantity": "leak_tpr1", "ablation": "trajectory", "halflife": "2.5"},
    ])

    m.main()

    with open(tmp_path / "results" / "fig02_halflife.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    methods = {r["method"] for r in rows}
    assert methods == {"m0_fedavg", "m1_glfc", "m4_proto"}


def test_merge_per_combo_row_overrides_a_stale_shared_row_with_the_same_key(m, tmp_path, monkeypatch):
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    (tmp_path / "results").mkdir()

    key_cols_row = {
        "dataset": "cifar100", "method": "m1_glfc", "family": "F1", "view": "full",
        "quantity": "leak_tpr1", "ablation": "trajectory", "halflife": "STALE",
    }
    _write_csv(tmp_path / "results" / "fig02_halflife.csv", [key_cols_row])
    fresh_row = {**key_cols_row, "halflife": "FRESH"}
    _write_csv(tmp_path / "results" / "fig02_halflife_cifar100_m1_glfc_full.csv", [fresh_row])

    m.main()

    with open(tmp_path / "results" / "fig02_halflife.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["halflife"] == "FRESH"
