#!/usr/bin/env python3
"""FX4i (08_FIX_PLAN.md §4i): builds the Wave V2 manifest, `build/waves/v2_main.tsv`, one row per
PBS array subjob: `dataset, method, seed, start, count, out_dir, method_config_override`.

Chunks of 64 over 1024 shadows/(dataset,method,seed) give 16 subjobs each; 3 datasets x 6 methods x
3 seeds x 16 = 864 rows, matching the array geometry `-J 0-863%56` in
`code/scripts/pbs/shadow_wave.pbs`. M0 is excluded: per §4b/§4i, if M0 passes reconstruction test
(i) its existing `shadows/` store (seeds 0-2, shadow_id < 1024) is reused as-is.

If `results/fx4_gate.csv` exists and a (dataset, method) row has `config_id == "tuned"`, that row's
`used_cfg_json` becomes the shadow generation's `method_config_override` for every seed of that
(dataset, method) -- the wave must generate shadows under the SAME config the gate accepted, not
silently fall back to the TAB05 headline default. Absent `fx4_gate.csv` (gate not run yet) or a
missing `used_cfg_json` column (pre-fix gate CSV, before this column was added), no override is
applied and a warning is printed -- rerun the gate (or just that dataset/method) before trusting
those rows.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from p3fcl.paths import V2_SHADOW_ROOT

REPO_ROOT = Path(__file__).resolve().parents[2]

METHODS = ["m1_glfc", "m2_target", "m3_fot", "m5_hybrid_replay", "m4_proto", "m8_analytic"]
DATASETS = ["cifar100", "cub200", "imagenet_r"]
SEEDS = [0, 1, 2]
N_SHADOWS = 1024
CHUNK = 64


def _load_gate_overrides(gate_csv: Path) -> dict:
    """Returns `{(dataset, method): override_dict}` for every (dataset, method) the gate tuned."""
    if not gate_csv.exists():
        print(
            f"WARNING: {gate_csv} not found -- wave V2 will use TAB05 default configs for every "
            f"method. Run FX4g first if any method needed tuning.", file=sys.stderr,
        )
        return {}
    overrides = {}
    missing_col_seen = False
    with open(gate_csv) as f:
        for row in csv.DictReader(f):
            if row.get("config_id") != "tuned":
                continue
            key = (row["dataset"], row["method"])
            if not row.get("used_cfg_json"):
                missing_col_seen = True
                continue
            overrides[key] = json.loads(row["used_cfg_json"])
    if missing_col_seen:
        print(
            f"WARNING: {gate_csv} has 'tuned' rows without a 'used_cfg_json' column (pre-fix gate "
            f"run) -- those (dataset, method) pairs will use the TAB05 default instead of the tuned "
            f"config in this manifest. Rerun FX4g for them and rebuild this manifest.", file=sys.stderr,
        )
    return overrides


def main() -> int:
    gate_overrides = _load_gate_overrides(REPO_ROOT / "results" / "fx4_gate.csv")

    rows = []
    for dataset in DATASETS:
        for method in METHODS:
            override = gate_overrides.get((dataset, method), {})
            for seed in SEEDS:
                out_dir = REPO_ROOT / V2_SHADOW_ROOT / dataset / method / f"seed{seed}"
                for start in range(0, N_SHADOWS, CHUNK):
                    rows.append({
                        "dataset": dataset, "method": method, "seed": seed,
                        "start": start, "count": min(CHUNK, N_SHADOWS - start),
                        "out_dir": str(out_dir),
                        "method_config_override": json.dumps(override, sort_keys=True) if override else "",
                    })

    out_path = REPO_ROOT / "build" / "waves" / "v2_main.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        # `lineterminator="\n"`: this TSV is read by a PBS array job via `sed -n "${LINE}p"` + bash's
        # `IFS=$'\t' read`, which does not strip `\r` -- the csv module's default `\r\n` would land a
        # trailing `\r` on the last (often-empty) `method_config_override` field and make `[ -n
        # "$OVERRIDE" ]` true for every row. `quoting=csv.QUOTE_NONE`: the override field is a raw JSON
        # object, which contains `"` -- the csv module's default quoting would wrap the field in `"..."`
        # and double every internal `"`, and bash's `read` does not undo either, so the JSON the PBS
        # script hands to `python3 -c '...json.loads(sys.argv[1])'` would be corrupted. JSON never
        # contains a literal tab, so QUOTE_NONE is safe against this delimiter.
        w = csv.DictWriter(
            f,
            fieldnames=["dataset", "method", "seed", "start", "count", "out_dir", "method_config_override"],
            delimiter="\t",
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
            quotechar="",
        )
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path}")
    if gate_overrides:
        print(f"tuned overrides applied to: {sorted(gate_overrides.keys())}")
    else:
        print("no tuned overrides applied (every method uses the TAB05 default config)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
