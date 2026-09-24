#!/usr/bin/env python3
"""PBS-only equivalence and timing on actual M0 evaluated shadow/target scores."""

import time

import numpy as np
from build_fx2_summary import _load_seed_npz, _pool_at_e, _resample_targets, _seed_arrays
from p3fcl import bootstrap_metrics, metrics

if __name__ == "__main__":
    data = [_seed_arrays(_load_seed_npz("cifar100", "m0_fedavg", s, "full"), "trajectory") for s in range(5)]
    rng = np.random.default_rng(325)
    reference_time = fast_time = 0.0
    cases = 0
    for _ in range(8):
        draw = [_resample_targets(data[i], rng) for i in rng.integers(0, 5, 5)]
        for elapsed in [0, 6]:
            scores, labels = _pool_at_e(draw, elapsed)
            for fpr in [0.01, 0.001]:
                start = time.monotonic()
                reference = metrics.tpr_at_fpr(scores, labels, fpr)
                reference_time += time.monotonic() - start
                start = time.monotonic()
                fast = bootstrap_metrics.tpr_at_fpr(scores, labels, fpr)
                fast_time += time.monotonic() - start
                assert reference == fast, (reference, fast, fpr)
                cases += 1
    print(
        f"FX9 BOOTSTRAP EQUIVALENCE cases={cases} max_abs_error=0 reference_seconds={reference_time:.6f} order_statistic_seconds={fast_time:.6f} speedup={reference_time/fast_time:.3f}",
        flush=True,
    )
