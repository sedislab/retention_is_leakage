"""Run manifests, `RUN_LOG.jsonl`, `.meta.json` — CLAUDE.md non-negotiable #10: a run not in the log
did not happen.

**This project does not use git.** `source_hash()` — a SHA-256 over the sorted contents of
`code/src/**/*.py` and `code/configs/**` — is the identity of the code that produced a result
instead. It changes exactly when the code or configs change, commit or no commit, which is the
property `make verify` and any later reproduction attempt actually need.
"""
from __future__ import annotations

import functools
import getpass
import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from .config import config_hash

# code/src/p3fcl/provenance.py -> parents: [0]=p3fcl [1]=src [2]=code [3]=repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"
CODE_SRC_DIR = REPO_ROOT / "code" / "src"
CODE_CONFIGS_DIR = REPO_ROOT / "code" / "configs"

_source_hash_cache: str | None = None


def source_hash(force: bool = False) -> str:
    """SHA-256 over the sorted contents of `code/src/**/*.py` and `code/configs/**`. Cached per
    process (it is cheap but not free) — pass `force=True` to recompute."""
    global _source_hash_cache
    if _source_hash_cache is not None and not force:
        return _source_hash_cache

    files = sorted(set(CODE_SRC_DIR.rglob("*.py")) | set(CODE_CONFIGS_DIR.rglob("*")))
    files = [f for f in files if f.is_file()]
    h = hashlib.sha256()
    for f in files:
        h.update(str(f.relative_to(REPO_ROOT)).encode("utf-8"))
        h.update(f.read_bytes())
    _source_hash_cache = h.hexdigest()
    return _source_hash_cache


def run_manifest(config: dict, seed=None) -> dict:
    manifest = {
        "source_hash": source_hash(),
        "config_hash": config_hash(dict(config)),
        "config": dict(config),
        "seed": seed,
        "hostname": socket.gethostname(),
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "pbs_array_index": os.environ.get("PBS_ARRAY_INDEX"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "user": getpass.getuser(),
        "utc_start": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import torch  # noqa: F401  (torch is optional; only P1 GPU paths need it)

        manifest["torch"] = torch.__version__
    except ImportError:
        manifest["torch"] = None
    return manifest


def finalize(manifest: dict, outputs, results_dir=None) -> None:
    """Writes `<output>.meta.json` beside each output and appends one line to
    `results/RUN_LOG.jsonl`."""
    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = results_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = configs_dir / f"{manifest['config_hash']}.yaml"
    if not cfg_path.exists():
        cfg_path.write_text(yaml.safe_dump(manifest["config"], sort_keys=True))

    end = datetime.now(timezone.utc).isoformat()
    wall_seconds = None
    try:
        t0 = datetime.fromisoformat(manifest["utc_start"])
        t1 = datetime.fromisoformat(end)
        wall_seconds = (t1 - t0).total_seconds()
    except Exception:
        pass

    out_paths = [str(Path(p)) for p in outputs]
    for p in outputs:
        p = Path(p)
        meta = {k: v for k, v in manifest.items() if k != "config"}
        meta["output"] = str(p)
        meta["utc_end"] = end
        meta["wall_seconds"] = wall_seconds
        meta_path = p.with_suffix(p.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2, default=str))

    log_line = {
        "utc_start": manifest["utc_start"],
        "utc_end": end,
        "wall_seconds": wall_seconds,
        "source_hash": manifest["source_hash"],
        "config_hash": manifest["config_hash"],
        "seed": manifest["seed"],
        "hostname": manifest["hostname"],
        "pbs_jobid": manifest["pbs_jobid"],
        "exit_status": 0,
        "outputs": out_paths,
    }
    run_log = results_dir / "RUN_LOG.jsonl"
    with open(run_log, "a") as f:
        f.write(json.dumps(log_line, default=str) + "\n")


def logged_run(fn):
    """Decorator for CLI entry points. Wraps `fn(*args, config, seed, **kwargs)`, which must return
    a dict with an `"outputs"` key (list of `Path`). Builds the manifest before the call and
    finalizes (writes `.meta.json` + appends `RUN_LOG.jsonl`) after."""

    @functools.wraps(fn)
    def wrapper(*args, config: dict, seed=None, **kwargs):
        manifest = run_manifest(config, seed=seed)
        result = fn(*args, config=config, seed=seed, manifest=manifest, **kwargs)
        outputs = result.get("outputs", []) if isinstance(result, dict) else []
        finalize(manifest, outputs)
        return result

    return wrapper
