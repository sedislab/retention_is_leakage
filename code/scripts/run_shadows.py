#!/usr/bin/env python3
"""Thin CLI wrapper matching `04_METHODS_AND_ATTACKS.md §3`'s documented interface:

    scripts/run_shadows.py --dataset D --method M --start i --count k --workers 36 --out DIR

All the real logic lives in `p3fcl.shadow_runner.run_shadow_range` (also reachable via
`python -m p3fcl.cli shadows`, which is what `pbs/shadow_array.pbs` actually invokes -- both paths
call the same function, so there is exactly one implementation of the shadow-federation runner).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import config as config_mod  # noqa: E402
from p3fcl import provenance  # noqa: E402
from p3fcl.shadow_runner import run_shadow_range  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(REPO_ROOT / "code" / "configs" / "attack_lira.yaml"))
    ap.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY.PATH=VALUE")
    args = ap.parse_args()

    cfg, _ = config_mod.load(args.config, args.overrides)
    cfg = dict(cfg)
    manifest = provenance.run_manifest(
        {"seed": cfg.get("seed", 0), "purpose": "shadow federation runner", "dataset": args.dataset,
         "method": args.method, "start": args.start, "count": args.count},
        seed=cfg.get("seed", 0),
    )
    result = run_shadow_range(
        dataset=args.dataset, method=args.method, start=args.start, count=args.count,
        workers=args.workers, out_dir=args.out, config=cfg,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "outputs"}, indent=2))
    provenance.finalize(manifest, result["outputs"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
