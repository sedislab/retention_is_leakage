#!/usr/bin/env python3
"""PBS acceptance that the explicitly preserved score/sweep evidence is unchanged."""

import hashlib

from fx9_io import ROOT


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.digest()


def main():
    archive = ROOT / "archive/2026-09-23b_pre_fx9/results"
    paths = [
        archive / name
        for name in [
            "m9_sweep.csv",
            "fig18_natural_federation.csv",
            "a1_m0_budget_check.csv",
        ]
    ]
    for method in ["m0_fedavg", "m3_fot", "m8_analytic"]:
        for extension in ["csv", "npz"]:
            paths += list(archive.glob(f"a1_lira_pertask_*_{method}_seed*_*.{extension}"))
        paths += list(archive.glob(f"a1_lira_fixedk_summary_*_{method}_*.csv"))
        paths += [
            p
            for p in archive.glob(f"fig02_halflife_*_{method}_*.csv")
            if not p.name.startswith("fig02_halflife_acc_")
        ]
    assert len(paths) > 3
    for path in paths:
        assert digest(path) == digest(ROOT / "results" / path.name), path.name
    print(
        f"FX9 PRESERVED ACCEPT files={len(paths)} checksum_mismatches=0 M9_sweep_unchanged=1 FIG18_data_unchanged=1 M0_budget_unchanged=1",
        flush=True,
    )


if __name__ == "__main__":
    main()
