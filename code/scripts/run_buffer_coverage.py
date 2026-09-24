#!/usr/bin/env python3
"""Pre-freeze pass: `results/buffer_coverage.csv`. For M1 (GLFC) and M5 (Hybrid Replay), whose
per-client exemplar buffer is private local state (never released, only counted through `touched`
per CLAUDE.md non-negotiable #2), what fraction of training examples ever enter SOME client's
buffer, and how many samples actually land in each (client, class) buffer entry given the real
FX9 gate config (`exemplar_budget`/`buffer_size_per_class` = 10 for every dataset, verified against
`results/fx9_gate.csv`).

Not a new experiment: this re-runs exactly one real (non-shadow, non-shadow-sampled) federation per
(method, dataset, seed) -- the identical `sim.run(method, X, y, stream, seed=seed)` call already used
by `run_utility_baseline.py` to produce TAB05/the accuracy matrices -- and reads the method's own
`_buffer` state after the run completes. Because the class-incremental stream assigns each class to
exactly one task, a (client, class) buffer entry is written at most once over the whole run, so the
final `_buffer` state is exactly the union of every id ever buffered; nothing here is new randomness
or a new configuration, just a direct measurement of an already-fixed, already-relied-upon quantity.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import features, provenance, sim, streams  # noqa: E402
from p3fcl.experiment import DATASETS, attacked_method_config  # noqa: E402
from p3fcl.shadow_runner import METHOD_REGISTRY  # noqa: E402

BACKBONE = "vit_base_patch16_224.augreg_in21k"
FEATURES_DIR = REPO_ROOT / "features"
N_TASKS = 10
N_CLIENTS = 10
BETA = 0.5
SEEDS = [0, 1, 2]
METHODS = ["m1_glfc", "m5_hybrid_replay"]
BUDGET_KEY = {"m1_glfc": "exemplar_budget", "m5_hybrid_replay": "buffer_size_per_class"}


def main() -> int:
    rows = []
    for method_id in METHODS:
        cls = METHOD_REGISTRY[method_id][0]
        for dataset in DATASETS:
            cache = features.load_cache(FEATURES_DIR, dataset, BACKBONE, "train")
            X, y = cache["features"], cache["labels"]
            n_classes = int(y.max()) + 1
            cfg = attacked_method_config(method_id, n_classes, X.shape[1], dataset)
            for seed in SEEDS:
                idx = np.arange(len(y))
                stream = streams.build_stream(
                    y, idx, n_tasks=N_TASKS, n_clients=N_CLIENTS, beta=BETA, seed=seed,
                )
                method = cls(cfg)
                sim.run(method, X, y, stream, seed=seed)

                buffered_ids: set = set()
                entry_sizes = []
                for (_client, _cls_id), entries in method._buffer.items():
                    entry_sizes.append(len(entries))
                    buffered_ids.update(int(did) for _feat, did in entries)

                n_training_examples = len(y)
                row = dict(
                    method=method_id,
                    dataset=dataset,
                    seed=seed,
                    buffer_budget=cfg[BUDGET_KEY[method_id]],
                    n_training_examples=n_training_examples,
                    n_ever_buffered=len(buffered_ids),
                    coverage_fraction=len(buffered_ids) / n_training_examples,
                    n_client_class_buffers=len(entry_sizes),
                    mean_samples_per_class_per_client=float(np.mean(entry_sizes)) if entry_sizes else 0.0,
                )
                rows.append(row)
                print(
                    f"BUFFER COVERAGE {method_id}/{dataset}/seed{seed}: "
                    f"coverage={row['coverage_fraction']:.6f} "
                    f"({row['n_ever_buffered']}/{row['n_training_examples']}) "
                    f"mean_samples_per_class_per_client={row['mean_samples_per_class_per_client']:.4f} "
                    f"(budget={row['buffer_budget']}, n_client_class_buffers={row['n_client_class_buffers']})",
                    flush=True,
                )

    assert len(rows) == len(METHODS) * len(DATASETS) * len(SEEDS)
    out = REPO_ROOT / "results" / "buffer_coverage.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"BUFFER COVERAGE ACCEPT wrote {len(rows)} rows to {out}", flush=True)

    config = {
        "phase": "pre-freeze",
        "purpose": "M1/M5 private exemplar buffer coverage (CLAUDE.md non-negotiable #2)",
        "methods": METHODS,
        "datasets": DATASETS,
        "seeds": SEEDS,
    }
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out])
    return 0


if __name__ == "__main__":
    sys.exit(main())
