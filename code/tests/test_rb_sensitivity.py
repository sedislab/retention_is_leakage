"""R7 summary logic (rebuttal/R7_PROTOCOL.md)."""
import numpy as np
import pytest

from p3fcl.rebuttal.sensitivity import ORACLE_LABEL, mean_ci, summarize_cell


def test_best_worst_rank_and_first_in_grid_order_ties():
    pts = [{"key": "a", "accs": [0.1, 0.1, 0.1]}, {"key": "b", "accs": [0.3, 0.3, 0.3]},
           {"key": "c", "accs": [0.3, 0.3, 0.3]}, {"key": "d", "accs": [0.0, 0.0, 0.0]},
           {"key": "e", "accs": [0.0, 0.0, 0.0]}]
    s = summarize_cell(pts, "a")
    assert s["best"]["key"] == "b" and s["n_tied_best"] == 2
    assert s["worst"]["key"] == "d" and s["n_tied_worst"] == 2
    assert s["selected_rank"] == 3 and s["n_points"] == 5
    assert s["oracle_gap"] == pytest.approx(0.2)
    assert summarize_cell(pts, "b")["selected_rank"] == 1


def test_mean_ci_matches_t_interval():
    m, lo, hi = mean_ci([0.1, 0.2, 0.3])
    assert m == pytest.approx(0.2)
    assert hi - m == pytest.approx(4.302652729911275 * 0.1 / np.sqrt(3))
    assert "not used for selection" in ORACLE_LABEL


def test_best_point_nan_last_and_first_tie():
    from p3fcl.rebuttal.sensitivity import best_point
    pts = [{"accs": [float("nan")] * 3}, {"accs": [0.2] * 3}, {"accs": [0.2] * 3}, {"accs": [0.1] * 3}]
    assert best_point(pts) == 1


def test_edges_and_extension_steps_at_most_twice():
    from p3fcl.rebuttal.sensitivity import edges_hit, extend_grid
    lr, mult = [0.01, 0.1, 1.0], [0.05, 0.25, 4.0]
    e = edges_hit(0.01, 4.0, lr, mult)
    assert e == ["lr_low", "mult_high"]
    assert edges_hit(0.1, 0.25, lr, mult) == []
    used = {}
    lr, mult, done = extend_grid(lr, mult, e, used)
    assert lr[0] == 0.005 and mult[-1] == 8.0 and done == e
    lr, mult, done = extend_grid(lr, mult, ["lr_low"], used)
    assert lr[0] == 0.002 and done == ["lr_low"]
    lr2, _, done = extend_grid(lr, mult, ["lr_low"], used)
    assert done == [] and lr2 == lr
