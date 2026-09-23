from __future__ import annotations

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


@pytest.fixture(scope="module")
def m():
    return _load_module("run_lira_pertask", "scripts/run_lira_pertask.py")


def test_pooled_scores_labels_pools_across_k_and_skips_out_of_range_rounds(m):
    # 3 shadows, 4 targets (2 at task k=0, 2 at task k=1), 5 rounds.
    n_shadows, n_targets, n_rounds = 3, 4, 5
    surf_eval = np.full((n_shadows, n_targets, n_rounds), np.nan)
    # target 0,1 born at k=0: released from round 0 onward
    surf_eval[:, 0, :] = [1.0, 1.1, 1.2, 1.3, 1.4]
    surf_eval[:, 1, :] = [2.0, 2.1, 2.2, 2.3, 2.4]
    # target 2,3 born at k=1: released from round 1 onward (round 0 stays NaN)
    surf_eval[:, 2, 1:] = [3.0, 3.1, 3.2, 3.3]
    surf_eval[:, 3, 1:] = [4.0, 4.1, 4.2, 4.3]
    in_out_eval = np.array([
        [True, True, False, False],
        [False, True, True, False],
        [True, False, False, True],
    ])
    idx_by_k = {0: [0, 1], 1: [2, 3]}

    # e=0: k=0 has round 0 valid, k=1 does NOT (round 1 is its birth round, e=0 means round k+0=1... wait
    # for k=1, T = k+e = 1, which IS valid (>=1, the birth round) -- both k's contribute at e=0.
    scores, labels, n_t = m._pooled_scores_labels(surf_eval, in_out_eval, idx_by_k, [0, 1], e=0, n_rounds=n_rounds)
    assert n_t == 4  # both k=0 (2 targets) and k=1 (2 targets) have a valid round at e=0
    assert len(scores) == 4 * n_shadows

    # e=4: k=0 -> T=4 (valid, last round); k=1 -> T=5 (out of range, n_rounds=5 means indices 0..4) -> only k=0 contributes
    scores4, labels4, n_t4 = m._pooled_scores_labels(surf_eval, in_out_eval, idx_by_k, [0, 1], e=4, n_rounds=n_rounds)
    assert n_t4 == 2  # only k=0's 2 targets
    assert len(scores4) == 2 * n_shadows


def test_pooled_scores_labels_returns_none_when_nothing_valid(m):
    surf_eval = np.full((2, 1, 3), np.nan)
    in_out_eval = np.array([[True], [False]])
    idx_by_k = {0: [0]}
    scores, labels, n_t = m._pooled_scores_labels(surf_eval, in_out_eval, idx_by_k, [5], e=0, n_rounds=3)
    assert scores is None and labels is None and n_t == 0


def test_report_row_has_every_plan_mandated_column(m):
    report = {
        "auc": 0.9, "tpr_at_1pct_fpr": 0.3, "tpr_at_1pct_fpr_ci_lo": 0.2, "tpr_at_1pct_fpr_ci_hi": 0.4,
        "tpr_at_0.1pct_fpr": 0.1, "tpr_at_0.1pct_fpr_ci_lo": 0.05, "tpr_at_0.1pct_fpr_ci_hi": 0.15,
        "n_pos": 10, "n_neg": 40,
    }
    row = m._report_row("cifar100", "m0_fedavg", "F1", "full", 0, 4096, 800, "pooled", 3, "trajectory", 20, report)
    required = {
        "dataset", "method", "family", "view", "seed", "n_shadows", "n_eval_shadows", "task_k",
        "elapsed", "auc", "tpr1", "tpr01", "n_targets", "n_pos", "n_neg",
    }
    assert required.issubset(row.keys())
    assert row["task_k"] == "pooled" and row["elapsed"] == 3
    assert row["cp_lo"] == 0.2 and row["cp_hi"] == 0.4


def test_rebuild_and_verify_targets_raises_on_mismatch(m, tmp_path, monkeypatch):
    """The core safety property FX2 §7a asks for: a shadow store whose stored target_ids don't match
    what `build_targets` recomputes from the declared stream/target config must raise, not silently
    mislabel every target's (task, client)."""
    from p3fcl import features

    rng = np.random.default_rng(0)
    n_classes, per_class, d = 4, 20, 6
    X, y, ids = [], [], []
    for c in range(n_classes):
        X.append(rng.standard_normal((per_class, d)))
        y.append(np.full(per_class, c))
    X, y = np.concatenate(X), np.concatenate(y)
    ids = np.arange(len(y))
    features.save_cache(tmp_path / "features", "synthdset", m.BACKBONE, "train", X, X, y, ids)
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)

    shadow_dir = tmp_path / "shadows"
    shadow_dir.mkdir()
    # A fabricated shadow file whose target_ids can never match any real build_targets() output for
    # this stream (out-of-range ids).
    np.savez(shadow_dir / "shadow_000000.npz", shadow_id=0, target_ids=np.array([999999, 999998]), in_out=np.array([True, False]))

    attack_cfg = {"stream": {"n_tasks": 2, "n_clients": 2, "beta": 1.0, "base_seed": 0}, "targets": {"per_shard": 2}}
    with pytest.raises(AssertionError):
        m._rebuild_and_verify_targets("synthdset", 0, shadow_dir, attack_cfg)


def test_rebuild_and_verify_targets_uses_the_stream_seed_not_the_configs_static_base_seed(m, tmp_path, monkeypatch):
    """Regression test: a real shadow store for stream seed S > 0 is generated with
    `--set stream.base_seed="${SEED}"` (`code/scripts/pbs/shadow_array.pbs`), i.e. BOTH the stream and
    the target-selection seed are S, not `attack_lira.yaml`'s static `stream.base_seed: 0`. Using the
    static config value here (as an earlier version of this function did) made the id-match assertion
    fail for every real seed > 0 store (caught on real cifar100 m0_fedavg seed1 data, job 183876)."""
    from p3fcl import features, streams

    rng = np.random.default_rng(0)
    n_classes, per_class, d = 4, 20, 6
    X, y = [], []
    for c in range(n_classes):
        X.append(rng.standard_normal((per_class, d)))
        y.append(np.full(per_class, c))
    X, y = np.concatenate(X), np.concatenate(y)
    ids = np.arange(len(y))
    features.save_cache(tmp_path / "features", "synthdset", m.BACKBONE, "train", X, X, y, ids)
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)

    stream_seed = 1
    attack_cfg = {"stream": {"n_tasks": 2, "n_clients": 2, "beta": 1.0, "base_seed": 0}, "targets": {"per_shard": 2}}
    real_stream = streams.build_stream(y, ids, n_tasks=2, n_clients=2, beta=1.0, seed=stream_seed)
    real_targets = m.build_targets(real_stream, targets_per_shard=2, base_seed=stream_seed)

    shadow_dir = tmp_path / "shadows"
    shadow_dir.mkdir()
    np.savez(
        shadow_dir / "shadow_000000.npz", shadow_id=0,
        target_ids=np.array([t["target_id"] for t in real_targets]),
        in_out=np.ones(len(real_targets), dtype=bool),
    )

    targets = m._rebuild_and_verify_targets("synthdset", stream_seed, shadow_dir, attack_cfg)
    assert [t["target_id"] for t in targets] == [t["target_id"] for t in real_targets]


def test_shadow_dir_prefers_legacy_layout_then_falls_back_to_v2(m, tmp_path, monkeypatch):
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    legacy = tmp_path / "shadows" / "cifar100" / "m0_fedavg"
    legacy.mkdir(parents=True)
    assert m._shadow_dir("cifar100", "m0_fedavg", 0) == legacy

    # seed>0 with no legacy dir present at all -> falls back to shadows_v2
    v2_expected = tmp_path / "shadows_v2" / "cifar100" / "m1_glfc" / "seed1"
    assert m._shadow_dir("cifar100", "m1_glfc", 1) == v2_expected


def test_shadow_dir_never_uses_the_stale_legacy_store_for_non_m0_methods(m, tmp_path, monkeypatch):
    """Regression test for a real bug: `shadows/<dataset>/<method>` (no seed suffix) is stale pre-fix
    data for every method except `m0_fedavg` (which reuses it on purpose). An earlier version of
    `_shadow_dir` fell back to it purely because it existed on disk -- and it DOES exist for m4_proto/
    m8_analytic/etc. on the real cluster, alongside their real `shadows_v2/.../seed0` data -- which
    would have silently scored the stale store instead of wave V2's fixed one."""
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    stale_legacy = tmp_path / "shadows" / "cifar100" / "m4_proto"
    stale_legacy.mkdir(parents=True)
    v2_dir = tmp_path / "shadows_v2" / "cifar100" / "m4_proto" / "seed0"
    v2_dir.mkdir(parents=True)

    assert m._shadow_dir("cifar100", "m4_proto", 0) == v2_dir
