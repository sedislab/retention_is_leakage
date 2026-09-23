from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_fx2_accuracy_summary.py"
    spec = importlib.util.spec_from_file_location("build_fx2_accuracy_summary", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m():
    return _load_module()


def test_floor_at_task_matches_classes_seen_so_far(m):
    # cifar100: 100 classes / 10 tasks = 10 classes/task
    assert m._floor_at_task("cifar100", 0) == pytest.approx(1 / 10)
    assert m._floor_at_task("cifar100", 9) == pytest.approx(1 / 100)
    # cub200/imagenet_r: 200 classes / 10 tasks = 20 classes/task
    assert m._floor_at_task("cub200", 0) == pytest.approx(1 / 20)
    assert m._floor_at_task("cub200", 4) == pytest.approx(1 / 100)


def test_resample_k_reindexes_by_position_not_by_k(m):
    rng = np.random.default_rng(0)
    per_k = {0: {"a": 1}, 1: {"a": 2}, 2: {"a": 3}, 3: {"a": 4}}
    resampled = m._resample_k(per_k, rng)
    assert set(resampled.keys()) == {0, 1, 2, 3}  # positions, always 0..3
    assert all(v in per_k.values() for v in resampled.values())


def test_pooled_diff_at_e_averages_across_seeds_and_k(m):
    resampled_seeds = [
        {0: {0: 0.5, 1: 0.4}},  # seed A, one k
        {0: {0: 0.7, 1: 0.3}, 1: {0: 0.9, 1: 0.1}},  # seed B, two k's
    ]
    # e=0: mean(0.5, 0.7, 0.9) = 0.7
    assert m._pooled_diff_at_e(resampled_seeds, 0) == pytest.approx(0.7)
    assert m._pooled_diff_at_e(resampled_seeds, 1) == pytest.approx(np.mean([0.4, 0.3, 0.1]))


def test_pooled_diff_at_e_returns_none_when_nothing_present(m):
    assert m._pooled_diff_at_e([{0: {1: 0.5}}], 0) is None


def test_replicate_normalized_curve_divides_by_e0_and_rejects_nonpositive_base(m):
    resampled = [{0: {0: 0.5, 1: 0.25, 2: 0.1}}]
    curve = m._replicate_normalized_curve(resampled)
    assert curve[0] == pytest.approx(1.0)
    assert curve[1] == pytest.approx(0.5)
    assert curve[2] == pytest.approx(0.2)

    zero_base = [{0: {0: 0.0, 1: 0.1}}]
    assert m._replicate_normalized_curve(zero_base) is None
    negative_base = [{0: {0: -0.1, 1: 0.1}}]
    assert m._replicate_normalized_curve(negative_base) is None


def test_build_seed_diffs_drops_k_missing_any_elapsed(m, tmp_path, monkeypatch):
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(m, "N_TASKS", 5)  # small horizon so k=3 (E_MAX=6) can never have every e
    csv_path = tmp_path / "results"
    csv_path.mkdir()
    rows = []
    # k=0: full support for e=0..4 (only e up to N_TASKS-1-k=4 exist at all)
    for e in range(5):
        rows.append({"dataset": "cifar100", "method": "m0_fedavg", "seed": 0, "task_k": 0, "task_T": e, "elapsed": e, "acc": 0.5, "acc_train": 0.5})
    with open(csv_path / "accuracy_matrix_cifar100.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    seed_diffs = m._build_seed_diffs("cifar100", "m0_fedavg", [0])
    # E_MAX=6 requires e=0..6, but N_TASKS=5 means k=0 can reach at most e=4 -> k=0 dropped entirely.
    assert seed_diffs == [{}]


def test_build_accuracy_halflife_on_real_cifar100_m0_gives_a_fast_forgetting_halflife(m):
    """Real-data sanity check (not synthetic): M0 (plain FedAvg, no retention) is known to forget
    almost immediately -- its accuracy half-life should be small (well under 2 tasks), status 'ok'."""
    csv_path = Path(__file__).resolve().parents[2] / "results" / "accuracy_matrix_cifar100.csv"
    if not csv_path.exists():
        pytest.skip("results/accuracy_matrix_cifar100.csv not present in this environment")
    rows = m.build_accuracy_halflife("cifar100", "m0_fedavg", "F1", "full", [0, 1, 2, 3, 4])
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "ok"
    assert row["halflife"] < 2.0
    assert row["ci_lo"] is not None and row["ci_hi"] is not None
    assert row["ci_lo"] <= row["halflife"] <= row["ci_hi"]
