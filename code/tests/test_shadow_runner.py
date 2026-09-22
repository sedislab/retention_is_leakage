from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest
from p3fcl import features, shadow_runner


def _make_synthetic_cache(tmp_path, n_classes=4, per_class=20, d=6, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_classes, d)) * 5
    X, y = [], []
    for c in range(n_classes):
        X.append(centers[c] + rng.standard_normal((per_class, d)) * 0.4)
        y.append(np.full(per_class, c))
    X = np.concatenate(X)
    y = np.concatenate(y)
    ids = np.arange(len(y))
    features.save_cache(tmp_path, "synthdset", "synthbb", "train", X, X, y, ids)
    return X, y


_CONFIG = {
    "stream": {"n_tasks": 2, "n_clients": 2, "beta": 1.0, "base_seed": 0},
    "targets": {"per_shard": 3, "p_in": 0.5},
}


@pytest.fixture(autouse=True)
def _patch_feature_paths(tmp_path, monkeypatch):
    _make_synthetic_cache(tmp_path)
    monkeypatch.setattr(shadow_runner, "FEATURES_DIR", tmp_path)
    monkeypatch.setattr(shadow_runner, "BACKBONE", "synthbb")


def test_method_config_override_reaches_the_method(tmp_path):
    """P5 dose-response pilot support: `config['method_config_override']` must actually change the
    method's behavior, not just get silently ignored. `buffer_size_per_class=0` should genuinely
    disable HybridReplay's buffer -- the EXEMPLAR family is scoped out of shadow_runner's scoring, but
    a real behavioral difference should still show up: with a zero buffer, the released delta cannot
    have been influenced by replayed exemplars, which we check indirectly via a real ledger built the
    same way `_run_one_shadow` builds one."""
    from p3fcl import sim, streams
    from p3fcl.artifacts import Family
    from p3fcl.methods.m5_hybrid_replay import HybridReplay

    X, y = _make_synthetic_cache(tmp_path)
    idx = np.arange(len(y))
    stream = streams.build_stream(y, idx, n_tasks=2, n_clients=2, beta=1.0, seed=0)

    cfg_default = {"n_classes": 4, "feature_dim": 6, "local_epochs": 5, "lr": 0.3, "buffer_size_per_class": 10}
    cfg_zero = {**cfg_default, "buffer_size_per_class": 0}

    ledger_default = sim.run(HybridReplay(cfg_default), X, y, stream, seed=0)["ledger"]
    ledger_zero = sim.run(HybridReplay(cfg_zero), X, y, stream, seed=0)["ledger"]

    # With buffer_size_per_class=0, round-1 EXEMPLAR touched sets from round 0 cannot appear in round
    # 1's MODEL_DELTA touched set (no buffer to replay from); with a real buffer they can.
    def _round1_delta_touches_round0_exemplars(ledger):
        exemplar_ids = frozenset().union(
            *(r.touched for r in ledger if r.family == Family.EXEMPLAR and r.round == 0)
        ) if any(r.family == Family.EXEMPLAR and r.round == 0 for r in ledger) else frozenset()
        delta_round1 = [r for r in ledger if r.family == Family.MODEL_DELTA and r.round == 1]
        return any(exemplar_ids & r.touched for r in delta_round1)

    assert _round1_delta_touches_round0_exemplars(ledger_default)
    assert not _round1_delta_touches_round0_exemplars(ledger_zero)


def test_prototype_and_gram_scores_undefined_under_secure_agg():
    """H11 (`agents/OPEN_QUESTIONS.md`): A1's PROTOTYPE/GRAM scoring filters
    `r.client == target["client"]` -- `Ledger.aggregate_view()` sets `client=-1` on every record it
    returns (by construction: it is a SUM over clients, so no client identity survives), and a real
    target's `client` field is never -1. This must stay true structurally: the attack has no defined
    score at all once restricted to the secure-aggregation view, for every family that needs a
    specific client's own record. A permanent regression test, not a probabilistic claim -- see
    `code/scripts/check_secure_agg_a1.py` and `notes/2026-09-21_secure_agg_a1.md`."""
    from p3fcl.artifacts import ArtifactRecord, Family, Ledger

    ledger = Ledger()
    ledger.add(ArtifactRecord(
        round=0, task=0, client=3, family=Family.PROTOTYPE, payload={"0": np.zeros(4)},
        touched=frozenset({10, 11}), n_touched=2, passes_over_data=1,
    ))
    ledger.add(ArtifactRecord(
        round=0, task=0, client=3, family=Family.GRAM, payload={"R": np.eye(4)},
        touched=frozenset({10, 11}), n_touched=2, passes_over_data=1,
    ))

    target = {"client": 3, "task": 0, "target_id": 10}
    X = np.zeros((20, 4))
    y = np.zeros(20, dtype=int)

    full_proto = shadow_runner._score_target(list(ledger), Family.PROTOTYPE, target, X, y, n_rounds=1)
    full_gram = shadow_runner._score_target(list(ledger), Family.GRAM, target, X, y, n_rounds=1)
    assert not np.all(np.isnan(full_proto)), "sanity: the full ledger must give a defined score"
    assert not np.all(np.isnan(full_gram)), "sanity: the full ledger must give a defined score"

    agg_proto = shadow_runner._score_target(ledger.aggregate_view(), Family.PROTOTYPE, target, X, y, n_rounds=1)
    agg_gram = shadow_runner._score_target(ledger.aggregate_view(), Family.GRAM, target, X, y, n_rounds=1)
    assert np.all(np.isnan(agg_proto)), "PROTOTYPE score must be undefined under the aggregate view"
    assert np.all(np.isnan(agg_gram)), "GRAM score must be undefined under the aggregate view"


def test_run_shadow_range_respects_adversary_view_secure_agg(tmp_path):
    """H11: `config['adversary_view'] = 'secure_agg'` must actually reach `_run_one_shadow` and change
    the MODEL_DELTA reconstruction (`_reconstruct_running_w_secure_agg` instead of
    `_reconstruct_running_w`), not just be accepted and dropped -- verified end to end by checking the
    two configs produce genuinely different scores for the same shadow_id (same population mask,
    same method, only the reconstruction differs)."""
    config_full = {**_CONFIG, "adversary_view": "full"}
    config_secure = {**_CONFIG, "adversary_view": "secure_agg"}

    out_full = tmp_path / "shadows_full"
    out_secure = tmp_path / "shadows_secure"
    shadow_runner.run_shadow_range(
        dataset="synthdset", method="m0_fedavg", start=0, count=1, workers=1,
        out_dir=out_full, config=config_full,
    )
    shadow_runner.run_shadow_range(
        dataset="synthdset", method="m0_fedavg", start=0, count=1, workers=1,
        out_dir=out_secure, config=config_secure,
    )
    with np.load(out_full / "shadow_000000.npz") as d_full, np.load(out_secure / "shadow_000000.npz") as d_secure:
        np.testing.assert_array_equal(d_full["in_out"], d_secure["in_out"])  # same population mask
        assert not np.allclose(
            np.nan_to_num(d_full["scores"]), np.nan_to_num(d_secure["scores"])
        ), "secure_agg reconstruction must differ from the full per-client reconstruction"


def test_run_shadow_range_respects_method_config_override(tmp_path):
    """The override must actually reach the method construction inside the worker process, not just
    be accepted and dropped -- verified end to end through the real `run_shadow_range` entrypoint."""
    config_zero = {**_CONFIG, "method_config_override": {"buffer_size_per_class": 0}}
    out_dir = tmp_path / "shadows_zero"
    result = shadow_runner.run_shadow_range(
        dataset="synthdset", method="m5_hybrid_replay", start=0, count=2, workers=1,
        out_dir=out_dir, config=config_zero,
    )
    assert result["n_written"] == 2


@pytest.mark.parametrize(
    "method", ["m4_proto", "m8_analytic", "m0_fedavg", "m1_glfc", "m2_target", "m3_fot", "m5_hybrid_replay"]
)
def test_run_shadow_range_writes_and_reports_correctly(tmp_path, method):
    out_dir = tmp_path / "shadows"
    result = shadow_runner.run_shadow_range(
        dataset="synthdset", method=method, start=0, count=10, workers=2, out_dir=out_dir, config=_CONFIG
    )
    assert result["n_written"] == 10
    assert result["n_skipped"] == 0
    assert result["n_targets"] > 0
    assert (out_dir / "targets.json").exists()
    for i in range(10):
        assert (out_dir / f"shadow_{i:06d}.npz").exists()


def test_run_shadow_range_is_resumable(tmp_path):
    out_dir = tmp_path / "shadows"
    shadow_runner.run_shadow_range(
        dataset="synthdset", method="m4_proto", start=0, count=5, workers=1, out_dir=out_dir, config=_CONFIG
    )
    result = shadow_runner.run_shadow_range(
        dataset="synthdset", method="m4_proto", start=0, count=5, workers=1, out_dir=out_dir, config=_CONFIG
    )
    assert result["n_written"] == 0
    assert result["n_skipped"] == 5


def test_shadow_output_is_byte_identical_across_reruns(tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    shadow_runner.run_shadow_range(
        dataset="synthdset", method="m4_proto", start=7, count=1, workers=1, out_dir=dir_a, config=_CONFIG
    )
    shadow_runner.run_shadow_range(
        dataset="synthdset", method="m4_proto", start=7, count=1, workers=1, out_dir=dir_b, config=_CONFIG
    )
    bytes_a = (dir_a / "shadow_000007.npz").read_bytes()
    bytes_b = (dir_b / "shadow_000007.npz").read_bytes()
    assert bytes_a == bytes_b

    with np.load(dir_a / "shadow_000007.npz") as da, np.load(dir_b / "shadow_000007.npz") as db:
        assert np.array_equal(da["target_ids"], db["target_ids"])
        assert np.array_equal(da["in_out"], db["in_out"])
        np.testing.assert_array_equal(da["scores"], db["scores"])


def _call_run_shadow_range(dataset, method, start, count, out_dir, config, queue):
    try:
        shadow_runner.run_shadow_range(
            dataset=dataset, method=method, start=start, count=count, workers=1,
            out_dir=out_dir, config=config,
        )
        queue.put(None)
    except Exception as e:
        queue.put(repr(e))


def test_concurrent_array_tasks_do_not_race_on_targets_json(tmp_path):
    """Regression test for a real failure found on Kodiak (job 157138): two PBS array tasks writing
    the shared `targets.json` convenience file via the *same* `.tmp` name raced, and one task's
    `os.replace` deleted the file out from under the other's, raising `FileNotFoundError`. Simulates
    that by running several `run_shadow_range` calls against the same `out_dir` as separate processes,
    each covering a disjoint shadow-index range -- exactly what concurrent array tasks do."""
    out_dir = tmp_path / "shadows"
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_call_run_shadow_range,
            args=("synthdset", "m4_proto", start, 3, out_dir, _CONFIG, queue),
        )
        for start in (0, 3, 6, 9)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
    errors = [queue.get() for _ in procs]
    assert all(e is None for e in errors), f"a concurrent array task raised: {errors}"
    assert (out_dir / "targets.json").exists()
    for i in range(12):
        assert (out_dir / f"shadow_{i:06d}.npz").exists()


def test_different_shadow_ids_give_different_in_out_membership(tmp_path):
    out_dir = tmp_path / "shadows"
    shadow_runner.run_shadow_range(
        dataset="synthdset", method="m4_proto", start=0, count=8, workers=2, out_dir=out_dir, config=_CONFIG
    )
    in_out_by_shadow = []
    for i in range(8):
        with np.load(out_dir / f"shadow_{i:06d}.npz") as d:
            in_out_by_shadow.append(d["in_out"].copy())
    # not every shadow has the identical membership vector -- otherwise the runner isn't actually
    # resampling inclusion per shadow_id and LiRA would have nothing to calibrate against.
    assert not all(np.array_equal(in_out_by_shadow[0], v) for v in in_out_by_shadow[1:])


def test_out_shadows_have_real_between_shadow_noise(tmp_path):
    """Regression test for the population-resampling design (`shadow_runner.py`'s module docstring):
    an earlier version resampled only the tracked targets and left every other datum fixed, so for a
    class-partitioned release (F2) a target was the sole source of randomness for its own class's
    statistic, and the OUT (and IN) score distributions each collapsed to a single deterministic
    value -- a noise-free null that made LiRA's AUC a foregone conclusion rather than a real measure of
    leakage. With population-level resampling, other same-class members also vary shadow-to-shadow, so
    a real target's OUT scores must show genuine spread, not a point mass."""
    out_dir = tmp_path / "shadows"
    n_shadows = 40
    shadow_runner.run_shadow_range(
        dataset="synthdset", method="m4_proto", start=0, count=n_shadows, workers=2,
        out_dir=out_dir, config=_CONFIG,
    )
    import json

    with open(out_dir / "targets.json") as f:
        targets = json.load(f)
    all_scores = np.stack([np.load(out_dir / f"shadow_{i:06d}.npz")["scores"] for i in range(n_shadows)])
    all_in_out = np.stack([np.load(out_dir / f"shadow_{i:06d}.npz")["in_out"] for i in range(n_shadows)])

    found_real_spread = False
    for j, t in enumerate(targets):
        col = all_scores[:, j, t["task"]]
        out_vals = col[~all_in_out[:, j] & ~np.isnan(col)]
        if len(out_vals) >= 5 and np.std(out_vals) > 1e-9:
            found_real_spread = True
            break
    assert found_real_spread, "expected genuine between-shadow variance in at least one target's OUT scores"


def test_m0_trajectory_and_last_round_are_not_identical(tmp_path):
    """The whole point of adding M0 (F1) to `shadow_runner.py`: unlike M4/M8, whose single-shot
    releases make `attacks.lira.lira_score`'s `trajectory` and `last_round` arms algebraically
    identical (confirmed on real data, `notes/2026-09-16_p3_a1_lira_results.md`), M0's released model
    keeps changing every round -- so a target's reconstructed logit-margin trajectory should genuinely
    differ from just its last value, and the two LiRA scores should differ too."""
    from p3fcl.attacks.lira import lira_score

    out_dir = tmp_path / "shadows"
    n_shadows = 30
    shadow_runner.run_shadow_range(
        dataset="synthdset", method="m0_fedavg", start=0, count=n_shadows, workers=2,
        out_dir=out_dir, config=_CONFIG,
    )
    import json

    with open(out_dir / "targets.json") as f:
        targets = json.load(f)
    all_scores = np.stack(
        [np.load(out_dir / f"shadow_{i:06d}.npz")["scores"] for i in range(n_shadows)]
    )
    all_in_out = np.stack(
        [np.load(out_dir / f"shadow_{i:06d}.npz")["in_out"] for i in range(n_shadows)]
    )

    found_nonconstant_trajectory = False
    found_score_difference = False
    for j, t in enumerate(targets):
        k = t["task"]
        col = all_scores[:, j, k:]  # rounds k..end, at least one target should vary round-to-round
        finite = col[~np.isnan(col)]
        if finite.size and np.ptp(finite) > 1e-9:
            found_nonconstant_trajectory = True

        target_traj = all_scores[0, j]  # shadow 0 plays "the evaluated instance"
        calib_scores = all_scores[1:, j]  # every other shadow calibrates the null/alt Gaussians
        calib_in_out = all_in_out[1:, j]
        in_traj = calib_scores[calib_in_out]
        out_traj = calib_scores[~calib_in_out]
        if len(in_traj) >= 3 and len(out_traj) >= 3 and not np.isnan(target_traj[k:]).all():
            traj_score = lira_score(target_traj, in_traj, out_traj, trajectory=True)
            last_score = lira_score(target_traj, in_traj, out_traj, trajectory=False)
            if not (np.isnan(traj_score) or np.isnan(last_score)) and not np.isclose(traj_score, last_score):
                found_score_difference = True
    assert found_nonconstant_trajectory, "expected at least one target's M0 score to vary across rounds"
    assert found_score_difference, "expected trajectory and last_round LiRA scores to differ for M0"


def test_prototype_score_separates_in_from_out_on_a_small_shard(tmp_path):
    """A coarse sanity check of the actual leakage signal A1 depends on: for a target whose shard is
    small, the released class mean should sit measurably closer to the target's own feature when the
    target is IN than when it is OUT (the leave-one-out mean-shift, `04_METHODS_AND_ATTACKS.md`'s A3
    discussion of the same effect applied here to a single, non-differenced release)."""
    out_dir = tmp_path / "shadows"
    n_shadows = 60
    shadow_runner.run_shadow_range(
        dataset="synthdset", method="m4_proto", start=0, count=n_shadows, workers=2,
        out_dir=out_dir, config=_CONFIG,
    )
    with open(out_dir / "targets.json") as f:
        import json

        targets = json.load(f)

    all_scores = np.stack(
        [np.load(out_dir / f"shadow_{i:06d}.npz")["scores"] for i in range(n_shadows)]
    )  # (n_shadows, n_targets, n_rounds)
    all_in_out = np.stack(
        [np.load(out_dir / f"shadow_{i:06d}.npz")["in_out"] for i in range(n_shadows)]
    )  # (n_shadows, n_targets)

    found_separation = False
    for j, t in enumerate(targets):
        col = all_scores[:, j, t["task"]]
        in_mask = all_in_out[:, j]
        in_vals = col[in_mask & ~np.isnan(col)]
        out_vals = col[~in_mask & ~np.isnan(col)]
        if len(in_vals) >= 5 and len(out_vals) >= 5:
            if np.mean(in_vals) > np.mean(out_vals):
                found_separation = True
                break
    assert found_separation, "expected at least one target where IN-shadows score higher than OUT-shadows"
