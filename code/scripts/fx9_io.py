"""PBS-only result writers and exact, explicit per-combo merges for FX9."""

import csv
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from p3fcl import provenance
from p3fcl.experiment import FX9_START

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def current(path):
    meta = json.loads(Path(str(path) + ".meta.json").read_text())
    assert meta["pbs_jobid"] and meta["utc_start"] >= FX9_START, path
    return meta


def write(path, rows, config):
    assert os.environ.get("PBS_JOBID"), "production writes require PBS"
    path = Path(path)
    assert rows, path
    manifest = provenance.run_manifest({"phase": "FX9", **config}, seed=0)
    if path.exists():
        archive = ROOT / "archive/fx9_shared" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        archive.mkdir(parents=True)
        shutil.copy2(path, archive / path.name)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    provenance.finalize(manifest, [path])


def merge(paths, output, key, expected):
    rows = []
    for path in paths:
        current(path)
        rows.extend(read(path))
    assert len(rows) == expected, (output, len(rows), expected)
    assert len({tuple(str(r[k]) for k in key) for r in rows}) == len(rows), output
    write(output, rows, dict(sources=[str(p) for p in paths], reused_shared_rows=0))
    return rows
