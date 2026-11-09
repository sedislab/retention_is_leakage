"""R4 subsampling (rebuttal/R4_PROTOCOL.md): stratified, seeded, nested in m; 'all' is the paper's input."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rb_r4_examples_per_class import subset_ids  # noqa: E402


def test_subset_ids_stratified_nested_seeded():
    y = np.repeat(np.arange(5), 40)
    a25, a10 = subset_ids(y, 25, 0), subset_ids(y, 10, 0)
    assert np.all(np.bincount(y[a25]) == 25) and np.all(np.bincount(y[a10]) == 10)
    assert set(a10) <= set(a25)
    assert np.array_equal(a25, np.sort(a25))
    assert np.array_equal(a25, subset_ids(y, 25, 0))
    assert not np.array_equal(a25, subset_ids(y, 25, 1))
    assert np.array_equal(subset_ids(y, "all", 0), np.arange(len(y)))
