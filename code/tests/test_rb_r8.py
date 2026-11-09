"""R8 helpers (rebuttal/R8_PROTOCOL.md)."""
import numpy as np

from p3fcl import streams
from p3fcl.rebuttal.r8 import on_edge, selection_federation


def test_selection_federation_matches_task_sizes_and_ref_composition():
    r = np.random.default_rng(0)
    y_ref = np.repeat(np.arange(8), [5, 5, 10, 10, 5, 5, 10, 10])
    X_ref = r.standard_normal((len(y_ref), 3))
    y_train = np.repeat(np.arange(8), 40)  # 320 train labels, balanced
    X_sel, y_sel, counts = selection_federation(X_ref, y_ref, y_train, "syn", n_tasks=4, seed=0)
    for cls in streams.class_incremental_tasks(y_ref, 4, 0):
        assert abs(int(np.isin(y_sel, cls).sum()) - int(np.isin(y_train, cls).sum())) <= len(cls)
        ref_frac = np.array([np.sum(y_ref == c) for c in cls], float)
        sel_frac = np.array([counts[int(c)] for c in cls], float)
        np.testing.assert_allclose(sel_frac / sel_frac.sum(), ref_frac / ref_frac.sum(), atol=0.02)
    # every selected row is a copy of a ref-train row of the same class
    for x, c in zip(X_sel[:20], y_sel[:20]):
        assert any(np.array_equal(x, X_ref[j]) for j in np.flatnonzero(y_ref == c))
    X2, y2, _ = selection_federation(X_ref, y_ref, y_train, "syn", n_tasks=4, seed=0)
    np.testing.assert_array_equal(X_sel, X2)
    np.testing.assert_array_equal(y_sel, y2)


def test_on_edge():
    grids = {"lr": [0.001, 0.01, 0.1], "mult": [0.1, 1.0, 4.0]}
    assert on_edge({"lr": 0.01, "mult": 1.0}, grids) == []
    assert on_edge({"lr": 0.001, "mult": 4.0}, grids) == ["lr_min", "mult_max"]
