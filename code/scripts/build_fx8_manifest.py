#!/usr/bin/env python3
"""FX8 (`08_FIX_PLAN.md` §10, C2 dose-response rerun): builds `build/waves/fx8_dose_response.tsv`,
one row per PBS array subjob: `dataset, method, knob_name, knob_value, seed, start, count, out_dir,
method_config_override`. Mirrors `build_wave_v2_manifest.py`'s exact manifest shape and CSV-dialect
gotchas (see its own docstring: `lineterminator="\n"` so a trailing `\r` doesn't corrupt bash's
`IFS=$'\t' read`, `QUOTE_NONE` so the raw-JSON override field survives untouched).

Scope per `08_FIX_PLAN.md` §10: CIFAR-100 only, 512 shadows x 3 seeds per level. M5 (individual
retention) sweeps `buffer_size_per_class` (the plan's prose calls it "exemplar_budget", but that is
not the literal config key -- confirmed by reading `HybridReplay.__init__` and its
`MethodSpec.retention_knob_name`, see `notes/2026-09-23_fx8_dose_response_design_notes.md`); M2
(semantic retention) sweeps `n_synthetic_per_class` (this one IS the literal key). Six levels each:
{1, 2, 5, 10, 20, 50}. The M4 arm is dropped (`08_FIX_PLAN.md`: "Drop the M4 arm, because its knob
was dead code" -- H6's own finding that `prototype_momentum` never fires under a class-incremental
stream, one class per task).

Chunks of 64 over 512 shadows/(method,level,seed) give 8 subjobs each; 2 methods x 6 levels x 3 seeds
x 8 = 288 rows, matching `-J 0-287%48` in `code/scripts/pbs/fx8_dose_response.pbs`.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATASET = "cifar100"
KNOBS = {"m5_hybrid_replay": "buffer_size_per_class", "m2_target": "n_synthetic_per_class"}
LEVELS = [1, 2, 5, 10, 20, 50]
SEEDS = [0, 1, 2]
N_SHADOWS = 512
CHUNK = 64


def main() -> int:
    rows = []
    for method, knob_name in KNOBS.items():
        for level in LEVELS:
            override = json.dumps({knob_name: level}, sort_keys=True)
            for seed in SEEDS:
                out_dir = REPO_ROOT / "shadows_v2" / DATASET / method / f"dose_{knob_name}_{level}" / f"seed{seed}"
                for start in range(0, N_SHADOWS, CHUNK):
                    rows.append({
                        "dataset": DATASET, "method": method, "knob_name": knob_name,
                        "knob_value": level, "seed": seed, "start": start,
                        "count": min(CHUNK, N_SHADOWS - start),
                        "out_dir": str(out_dir), "method_config_override": override,
                    })

    out_path = REPO_ROOT / "build" / "waves" / "fx8_dose_response.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["dataset", "method", "knob_name", "knob_value", "seed", "start", "count",
                        "out_dir", "method_config_override"],
            delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE, quotechar="",
        )
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
