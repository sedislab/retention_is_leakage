#!/usr/bin/env python3
"""Maps a flat grid index to (method, dataset, n_tasks, seed) for the TAB05 utility-baseline sweep.
Kept as a separate tiny script so the PBS array wrapper can chunk indices in a shell loop without
duplicating the grid definition."""
import sys

METHODS = ["M0", "M1", "M2", "M3", "M4", "M5", "M8"]
DATASET_TASKS = [
    ("cifar100", 10), ("cifar100", 20),
    ("imagenet_r", 10), ("imagenet_r", 20),
    ("cub200", 10),
    ("camelyon17", 5),  # domain-incremental: n_tasks is ignored by streams.build_stream and will
                        # actually be 5 (one per hospital) regardless of this value; kept explicit
                        # for an honest "requested" record rather than passing a fake CIFAR-shaped number.
]
SEEDS = [0, 1, 2]

GRID = [(m, d, t, s) for m in METHODS for (d, t) in DATASET_TASKS for s in SEEDS]

if __name__ == "__main__":
    idx = int(sys.argv[1])
    m, d, t, s = GRID[idx]  # raises IndexError (nonzero exit) past the end -- callers rely on this
    print(f"{m} {d} {t} {s}")
