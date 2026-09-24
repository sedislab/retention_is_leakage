#!/usr/bin/env python3
"""FX9: assemble curves from current per-combo producers, never a merged input."""

import csv
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fx9_io import current, read, write
from p3fcl import provenance
from p3fcl.experiment import CHANGED_METHODS

ROOT = Path(__file__).resolve().parents[2]


def main():
    rows = []
    sources = sorted((ROOT / "results").glob("retention_acc_*.csv")) + sorted(
        (ROOT / "results").glob("retention_leak_*.csv")
    )
    for path in sources:
        with path.open() as f:
            rr = list(csv.DictReader(f))
        if any(r["method"] in CHANGED_METHODS for r in rr):
            current(path)
        rows.extend(rr)
    keys = [(r["dataset"], r["method"], r["view"], r["quantity"], r["elapsed"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate retention curve cells")
    expected = 3 * 11 * 3 * 7  # datasets × method/views × quantities × elapsed
    if len(rows) != expected:
        raise ValueError(f"incomplete curves: {len(rows)} != {expected}")
    horizon_sources = sorted((ROOT / "results").glob("fx9_horizon_*.csv"))
    assert len(horizon_sources) == 33
    horizon = {}
    for path in horizon_sources:
        current(path)
        rr = read(path)
        assert len(rr) == 1
        r = rr[0]
        horizon[(r["dataset"], r["method"], r["view"])] = r
    assert len(horizon) == 33
    for r in rows:
        if r["elapsed"] != "6" or r["quantity"] == "leak_auc":
            continue
        h = horizon[(r["dataset"], r["method"], r["view"])]
        prefix = "acc" if r["quantity"] == "acc" else "leak"
        assert abs(float(r["norm"]) - float(h[prefix + "_norm"])) < 1e-10
        for bound in ["lo", "hi"]:
            r["norm_ci_" + bound] = h[prefix + "_norm_ci_" + bound]
    out = ROOT / "results/retention_curves.csv"
    if out.exists():
        archive = ROOT / "archive/fx9_shared" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        archive.mkdir(parents=True)
        shutil.copy2(out, archive / out.name)
    manifest = provenance.run_manifest(
        dict(
            phase="FX9-1",
            hypothesis="H2",
            sources=[str(p) for p in sources + horizon_sources],
            horizon_CI="FX9 paired seed/task/target bootstrap",
        ),
        seed=0,
    )
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(
            sorted(
                rows, key=lambda r: (r["dataset"], r["method"], r["view"], r["quantity"], int(r["elapsed"]))
            )
        )
    provenance.finalize(manifest, [out])
    write(
        ROOT / "results/fig02_retention_at_horizon.csv",
        list(horizon.values()),
        dict(hypothesis="H2", sources=[str(p) for p in horizon_sources]),
    )
    print(f"retention_curves rows={len(rows)} source_files={len(sources)}")
    for key, h in sorted(horizon.items()):
        print(
            f"FX9-9 HORIZON ACCEPT {'/'.join(key)} "
            f"A6={h['acc_norm']} CI=[{h['acc_norm_ci_lo']},{h['acc_norm_ci_hi']}] "
            f"L6={h['leak_norm']} CI=[{h['leak_norm_ci_lo']},{h['leak_norm_ci_hi']}]",
            flush=True,
        )


if __name__ == "__main__":
    main()
