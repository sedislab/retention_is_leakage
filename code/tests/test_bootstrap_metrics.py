import numpy as np
import pytest
from p3fcl import bootstrap_metrics, metrics


def test_order_statistic_matches_reference_with_ties_and_budget_boundaries():
    rng = np.random.default_rng(283)
    for n in [2, 3, 10, 101, 999]:
        for _ in range(12):
            labels = rng.integers(0, 2, n)
            labels[0], labels[-1] = 0, 1
            n0 = np.count_nonzero(labels == 0)
            boundary = rng.integers(0, n0 + 1) / n0
            fprs = [
                0,
                0.001,
                0.01,
                1,
                boundary,
                np.nextafter(boundary, -np.inf),
                np.nextafter(boundary, np.inf),
            ]
            for scores in [rng.normal(size=n), rng.integers(-3, 4, n).astype(float), np.ones(n)]:
                for fpr in fprs:
                    assert bootstrap_metrics.tpr_at_fpr(scores, labels, fpr) == metrics.tpr_at_fpr(
                        scores, labels, fpr
                    )


@pytest.mark.parametrize("label", [0, 1])
def test_order_statistic_undefined_without_both_classes(label):
    assert np.isnan(bootstrap_metrics.tpr_at_fpr([1.0, 2.0], [label, label], 0.01))
