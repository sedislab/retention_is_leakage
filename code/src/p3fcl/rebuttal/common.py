"""Shared helpers for rebuttal scripts: the paper-code hash and the paired R1 setup."""
from __future__ import annotations

import csv
import hashlib
import json

import numpy as np
import yaml

from .. import features, provenance, streams
from ..paths import shadow_dir
from ..shadow_runner import BACKBONE, build_targets

REPO_ROOT = provenance.REPO_ROOT
REBUTTAL_SRC = provenance.CODE_SRC_DIR / "p3fcl" / "rebuttal"


def paper_source_hash() -> str:
    """`provenance.source_hash()` with `code/src/p3fcl/rebuttal/` left out. While the paper's code is
    unchanged it equals the hash in the paper's sidecars (31831aaa… at submission). So a rebuttal
    result carrying it was produced by the paper's code plus only new rebuttal files."""
    files = sorted(set(provenance.CODE_SRC_DIR.rglob("*.py")) | set(provenance.CODE_CONFIGS_DIR.rglob("*")))
    files = [f for f in files if f.is_file() and REBUTTAL_SRC not in f.parents]
    h = hashlib.sha256()
    for f in files:
        h.update(str(f.relative_to(REPO_ROOT)).encode("utf-8"))
        h.update(f.read_bytes())
    return h.hexdigest()


def manifest(config: dict, seed=0) -> dict:
    m = provenance.run_manifest(config, seed=seed)
    m["paper_source_hash"] = paper_source_hash()
    return m


def attack_config() -> dict:
    with open(REPO_ROOT / "code" / "configs" / "attack_lira.yaml") as f:
        return yaml.safe_load(f)


def wave_override(dataset: str, method: str, seed: int) -> dict:
    """The `method_config_override` the paper's shadow wave used for this store (V3 for M2; V2 rows
    for M3 carry none)."""
    wave = "v3_main.tsv" if method == "m2_target" else "v2_main.tsv"
    with open(REPO_ROOT / "build" / "waves" / wave, newline="") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t")
                if r["dataset"] == dataset and r["method"] == method and int(r["seed"]) == seed]
    overrides = {r["method_config_override"] or "" for r in rows}
    if len(overrides) != 1:
        raise RuntimeError(f"{wave}: {len(overrides)} distinct overrides for {dataset}/{method}/seed{seed}")
    o = overrides.pop()
    return json.loads(o) if o else {}


def paired_setup(dataset: str, method: str, seed: int) -> dict:
    """Train/ref features, the paper's base stream, target pool, task class sets and ref centres for
    one (dataset, method, seed) store. The target pool is checked against the paper's shadow store."""
    cfg = attack_config()
    sc = cfg["stream"]
    train = features.load_cache(REPO_ROOT / "features", dataset, BACKBONE, "train")
    ref = features.load_cache(REPO_ROOT / "features", dataset, BACKBONE, "ref")
    X, y = train["features"], train["labels"]
    base = streams.build_stream(y, np.arange(len(y)), n_tasks=sc["n_tasks"], n_clients=sc["n_clients"],
                                beta=sc["beta"], seed=seed)
    targets = build_targets(base, cfg["targets"]["per_shard"], seed)
    task_classes = streams.class_incremental_tasks(y, sc["n_tasks"], seed)
    from .release_audit import ref_task_centres
    centres = ref_task_centres(ref["features"], ref["labels"], task_classes)
    paper_dir = shadow_dir(dataset, method, seed, root=REPO_ROOT)
    with np.load(paper_dir / "shadow_000000.npz") as z:
        if [int(i) for i in z["target_ids"]] != [t["target_id"] for t in targets]:
            raise AssertionError(f"{paper_dir}: rebuilt target pool differs from the paper store")
    return {"X": X, "y": y, "base": base, "targets": targets, "task_classes": task_classes,
            "centres": centres, "p_in": float(cfg["targets"]["p_in"]), "paper_dir": paper_dir}
