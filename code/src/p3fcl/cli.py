"""One entry point: `python -m p3fcl.cli <subcommand>`. Every subcommand takes `--config` plus
`--set key.path=value` overrides. Most subcommands are deliberate `NotImplementedError` stubs at P0 —
each says which phase implements it, so `cli.py --help` doubles as a phase map.
"""
from __future__ import annotations

import argparse
import json
import sys


def _add_common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--config", required=True)
    sp.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY.PATH=VALUE")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="p3fcl")
    sub = p.add_subparsers(dest="command", required=True)

    data_p = sub.add_parser("data", help="[P1] fetch/prepare/audit a dataset")
    data_p.add_argument("--dataset", required=True)
    group = data_p.add_mutually_exclusive_group(required=True)
    group.add_argument("--fetch", action="store_true")
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--audit", action="store_true")

    extract_p = sub.add_parser("extract", help="[P1] frozen-backbone feature extraction (GPU)")
    _add_common(extract_p)
    extract_p.add_argument("--dataset", required=True)
    extract_p.add_argument("--backbone", required=True)
    extract_p.add_argument("--splits", default="train,test,ref")
    extract_p.add_argument("--out", required=True)
    extract_p.add_argument("--batch-size", type=int, default=256)
    extract_p.add_argument("--amp", action="store_true")

    run_p = sub.add_parser("run", help="[P2+] run one method on one dataset stream")
    _add_common(run_p)

    shadows_p = sub.add_parser("shadows", help="[P3] shadow-federation runner (CPU array task)")
    _add_common(shadows_p)
    shadows_p.add_argument("--dataset", required=True)
    shadows_p.add_argument("--method", required=True)
    shadows_p.add_argument("--start", type=int, required=True)
    shadows_p.add_argument("--count", type=int, required=True)
    shadows_p.add_argument("--workers", type=int, default=1)
    shadows_p.add_argument("--out", required=True)

    attack_p = sub.add_parser("attack", help="[P3] run an attack against a ledger/shadow store")
    _add_common(attack_p)

    account_p = sub.add_parser("account", help="[P2/P6] run the DP accountant on a real ledger")
    _add_common(account_p)

    audit_p = sub.add_parser("audit", help="[P3] one-run canary audit")
    _add_common(audit_p)

    sweep_p = sub.add_parser("sweep", help="[P5] retention-strength dose-response sweep")
    _add_common(sweep_p)

    collect_p = sub.add_parser("collect", help="aggregate run outputs into a tidy results CSV")
    _add_common(collect_p)

    return p


def _cmd_data(args: argparse.Namespace) -> int:
    from . import get_data

    if args.fetch:
        result = get_data.fetch(args.dataset)
        print(json.dumps(result, indent=2, default=str))
    elif args.prepare:
        result = get_data.prepare(args.dataset)
        print(json.dumps(result, indent=2, default=str))
    else:
        get_data.audit(args.dataset)  # already prints its own report
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    from . import features, provenance

    config = {
        "seed": 0,
        "dataset": args.dataset,
        "backbone": args.backbone,
        "batch_size": args.batch_size,
        "amp": args.amp,
    }
    manifest = provenance.run_manifest(config, seed=0)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    outputs = []
    for split in splits:
        path = features.extract(
            dataset=args.dataset,
            backbone=args.backbone,
            split=split,
            batch_size=args.batch_size,
            device="cuda",
            amp=args.amp,
            out_dir=args.out,
        )
        print(f"extract: {args.dataset}|{args.backbone}|{split} -> {path}")
        outputs.append(path)
    provenance.finalize(manifest, outputs)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    raise NotImplementedError(
        "run: full CLI wiring lands with the method zoo in P2. P0 exercises M8/M4-proto directly "
        "via sim.run() in tests, not through this CLI path."
    )


def _cmd_shadows(args: argparse.Namespace) -> int:
    from . import config as config_mod
    from . import provenance
    from .shadow_runner import run_shadow_range

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


def _cmd_attack(args: argparse.Namespace) -> int:
    raise NotImplementedError("attack is Phase P3.")


def _cmd_account(args: argparse.Namespace) -> int:
    raise NotImplementedError(
        "account: CLI wiring against real ledgers lands in P2 (per-method) and is reported in P6 "
        "(FIG05/TAB02). dp.accountant.account() itself is already implemented and tested."
    )


def _cmd_audit(args: argparse.Namespace) -> int:
    raise NotImplementedError(
        "audit CLI wiring to M8/M9 lands in P3 task 7. audit.one_run.audit_from_guesses() is already "
        "implemented and tested."
    )


def _cmd_sweep(args: argparse.Namespace) -> int:
    raise NotImplementedError("sweep is Phase P5.")


def _cmd_collect(args: argparse.Namespace) -> int:
    raise NotImplementedError("collect: CSV aggregation lands alongside the first real per-run outputs.")


_COMMANDS = {
    "data": _cmd_data,
    "extract": _cmd_extract,
    "run": _cmd_run,
    "shadows": _cmd_shadows,
    "attack": _cmd_attack,
    "account": _cmd_account,
    "audit": _cmd_audit,
    "sweep": _cmd_sweep,
    "collect": _cmd_collect,
}


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.error(f"no handler registered for {args.command!r}")
    return handler(args) or 0


if __name__ == "__main__":
    sys.exit(main())
