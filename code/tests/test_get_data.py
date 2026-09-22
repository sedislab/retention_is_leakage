from __future__ import annotations

import numpy as np
from p3fcl.get_data import _carve_canary, _stratified_split


def test_stratified_split_disjoint_and_covers_everything():
    labels = np.repeat(np.arange(5), 100)
    splits = _stratified_split(labels, seed=0, fracs={"train": 0.8, "test": 0.1, "ref": 0.1})
    all_ids = sorted(i for ids in splits.values() for i in ids)
    assert all_ids == list(range(len(labels)))
    # no overlap between any two buckets
    seen = set()
    for ids in splits.values():
        assert seen.isdisjoint(ids)
        seen.update(ids)


def test_stratified_split_is_stratified_per_class():
    labels = np.repeat(np.arange(5), 100)
    splits = _stratified_split(labels, seed=0, fracs={"train": 0.8, "ref": 0.2})
    train_labels = labels[splits["train"]]
    # each class should be ~80% of its 100 examples in train, i.e. roughly balanced representation
    for c in range(5):
        assert 70 <= int(np.sum(train_labels == c)) <= 90


def test_stratified_split_deterministic():
    labels = np.repeat(np.arange(3), 50)
    a = _stratified_split(labels, seed=42, fracs={"train": 0.5, "ref": 0.5})
    b = _stratified_split(labels, seed=42, fracs={"train": 0.5, "ref": 0.5})
    assert a == b


def test_carve_canary_is_subset_of_ref_and_never_exceeds_n():
    ref_ids = list(range(1000, 1100))
    canary = _carve_canary(ref_ids, seed=0, n=20)
    assert len(canary) == 20
    assert set(canary) <= set(ref_ids)


def test_carve_canary_empty_ref_gives_empty_canary():
    assert _carve_canary([], seed=0) == []


def test_carve_canary_n_larger_than_ref_caps_at_ref_size():
    ref_ids = [1, 2, 3]
    canary = _carve_canary(ref_ids, seed=0, n=500)
    assert sorted(canary) == ref_ids
