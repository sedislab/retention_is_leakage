#!/usr/bin/env python3
"""Real per-method accountant outputs only; no historical shared rows."""

from fx9_io import ROOT, current, read, write
from p3fcl.experiment import METHODS

if __name__ == "__main__":
    paths = [ROOT / f"results/fig05_accountant_{m}.csv" for m in METHODS] + [ROOT / "results/fig05_m9.csv"]
    rows = []
    for path in paths:
        current(path)
        rr = read(path)
        assert rr and {r["unit"] for r in rr} == {"U1", "U2", "U3", "U4"}
        rows.extend(rr)
    assert len({(r["method"], r["unit"], r["T"], r.get("eps0", "")) for r in rows}) == len(rows)
    write(ROOT / "results/fig05_eps_of_T.csv", rows, dict(hypothesis="H1", sources=[str(p) for p in paths]))
    print(f"FX9-7 FIG05 ACCEPT methods=8 M9_eps0=1,4 rows={len(rows)} pending_panels=0")
